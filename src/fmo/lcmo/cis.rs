use crate::fmo::{Monomer, SuperSystem};
use crate::initialization::Atom;
use ndarray::{Array2, ArrayView1, s, ArrayView2, Array1, ArrayView3};
use crate::excited_states::trans_charges_fast;
use crate::fmo::lcmo::helpers::*;

impl SuperSystem{
    pub fn build_cis_matrix(&self,lcmo_h:ArrayView2<f64>){
        // TODO: READ THIS FROM THE INPUT FILE
        // Number of active orbitals per monomer
        let n_occ_m: usize = 1;
        let n_virt_m: usize = 1;
        let n_le:usize = 2;
        let n_ct:usize = n_occ_m * n_virt_m;

        // Reference to the atoms.
        let atoms: &[Atom] = &self.atoms;

        // Get lcmo hamiltonian as input
        // build the lcmo hamiltonian
        // let lcmo_h:Array2<f64> = self.build_lcmo_hamiltonian();

        // The dimension of the cis matrix.
        // Number of fragments * number of local excitations +
        // number of pairs * (n_active_occ * n_active_virt)
        let dim_le:usize = self.n_mol * n_le;
        let dim_ct:usize = (self.pairs.len() + self.esd_pairs.len()) * n_ct;
        let dim_cis:usize = dim_le + dim_ct;
        let mut cis_matrix:Array2<f64> = Array2::zeros([dim_cis,dim_cis]);

        // the diagonal elements for the LE-LE block are set
        for (i, mol) in self.monomers.iter().enumerate() {
            // reference to the excited state energies of the monomers
            let excited_states:ArrayView1<f64> = mol.properties.excited_states().unwrap();
            // fill the diagonal of the matrix
            cis_matrix.slice_mut(s![i..(i+n_le),i..(i+n_le)]).assign(&excited_states.slice(s![0..n_le]));
        }
        // the diagonal elements for the CT-CT block are set
        for pair in self.pairs.iter(){
            // set pair indices
            let index_i:usize = pair.i;
            let index_j:usize = pair.j;
            // set n_atoms of fragments
            let n_atoms_i:usize = self.monomers[index_i].n_atoms;
            let n_atoms_j:usize = self.monomers[index_j].n_atoms;
            // reference to the mo coefficients of fragment I
            let c_mo_i:ArrayView2<f64> = self.monomers[index_i].properties.orbs().unwrap();
            // reference to the mo coefficients of fragment J
            let c_mo_j:ArrayView2<f64> = self.monomers[index_j].properties.orbs().unwrap();
            // reference to the overlap matrix of fragment I
            let s_i:ArrayView2<f64> = self.monomers[index_i].properties.s().unwrap();
            // reference to the overlap matrix of fragment J
            let s_j:ArrayView2<f64> = self.monomers[index_j].properties.s().unwrap();
            // occupied orbitals of fragment I.
            let occ_indices_i: &[usize] = &self.monomers[index_i].properties.occ_indices().unwrap()[0..n_occ_m];
            // virtual orbitals of fragment I.
            let virt_indices_i: &[usize] = &self.monomers[index_i].properties.virt_indices().unwrap()[0..n_virt_m];
            // occupied orbitals of fragment J.
            let occ_indices_j: &[usize] = &self.monomers[index_j].properties.occ_indices().unwrap()[0..n_occ_m];
            // virtual orbitals of fragment J.
            let virt_indices_j: &[usize] = &self.monomers[index_j].properties.virt_indices().unwrap()[0..n_virt_m];

            // get excited coefficients of the fragment I
            let exc_coeff_i:ArrayView3<f64> = self.monomers[index_i].properties.excited_coefficients().unwrap();
            // get excited coefficients of the fragment J
            let exc_coeff_j:ArrayView3<f64> = self.monomers[index_j].properties.excited_coefficients().unwrap();

            // calculate outer-diagonal LE-CT one-electron matrix elements
            // for the LE on fragment I and the CT from I->J
            let le_i_ct_ij:Array2<f64> = le_i_ct_one_electron_ij(
                occ_indices_i,
                virt_indices_j,
                exc_coeff_i,
                lcmo_h,
                n_le,
                n_ct,
            );
            let cis_index_le:usize = (index_i * n_le);
            let cis_index_ct:usize = dim_le + index_i * (self.n_mol-1) + index_j-1;
            cis_matrix.slice_mut(s![cis_index_le..cis_index_le+n_le,cis_index_ct..cis_index_ct+n_ct]).assign(&le_i_ct_ij);

            // calculate outer-diagonal LE-CT one-electron matrix elements
            // for the LE on fragment I and the CT from J->I
            let le_i_ct_ji:Array2<f64> = le_i_ct_one_electron_ji(
                occ_indices_j,
                virt_indices_i,
                exc_coeff_i,
                lcmo_h,
                n_le,
                n_ct,
            );
            let cis_index_ct:usize = dim_le + index_j * (self.n_mol-1) + index_i;
            cis_matrix.slice_mut(s![cis_index_le..cis_index_le+n_le,cis_index_ct..cis_index_ct+n_ct]).assign(&le_i_ct_ji);

            // calculate transition charges for monomer I
            let (q_ov_i,q_oo_i,q_vv_i):(Array2<f64>,Array2<f64>,Array2<f64>) = trans_charges_fast(
                n_atoms_i,
                &atoms[self.monomers[index_i].slice.atom_as_range()],
                c_mo_i,
                s_i,
                &occ_indices_i,
                &virt_indices_i,
            );

            // calculate transition charges for monomer J
            let (q_ov_j,q_oo_j,q_vv_j):(Array2<f64>,Array2<f64>,Array2<f64>) = trans_charges_fast(
                n_atoms_j,
                &atoms[self.monomers[index_j].slice.atom_as_range()],
                c_mo_j,
                s_j,
                &occ_indices_j,
                &virt_indices_j,
            );
            // get gamma_AB matrix of the pair
            let mut gamma_ab:ArrayView2<f64> = pair.properties.gamma().unwrap();
            let gamma_ab_off_diag:ArrayView2<f64> = gamma_ab.slice(s![..n_atoms_i,n_atoms_i..]);
            let gamma_ao_pair:ArrayView2<f64> = pair.properties.gamma_ao().unwrap();

            // electron-hole exchange integral (ia|ia)
            // get reference to full overlap matrix of the pair
            let s_pair:ArrayView2<f64> = pair.properties.s().unwrap();
            // calculate the exchange integral with a loop
            let exchange:Array1<f64> = inter_fragment_exchange_integral(
                c_mo_i,
                c_mo_j,
                s_pair,
                occ_indices_i,
                virt_indices_j,
                gamma_ao_pair,
            );
            // alternative way of calculating the exchange integral
            // using transition densities
            let qtrans_mu_nu:Array2<f64> = inter_fragment_trans_charge_ct(
                &atoms[self.monomers[index_i].slice.atom_as_range()],
                &atoms[self.monomers[index_j].slice.atom_as_range()],
                c_mo_i,
                c_mo_j,
                s_pair,
                occ_indices_i,
                virt_indices_j,
            );
            let exchange_alternative:Array1<f64> = qtrans_mu_nu.t().dot(&gamma_ab.dot(&qtrans_mu_nu)).into_shape([n_occ_m*n_virt_m]).unwrap();

            // coulomb-interaction integral (ii|aa) = sum_AB q_A^II gamma_AB q_B^aa
            // q_oo(I) * gamma_AB (pair) * q_vv(J)
            let coulomb:Array1<f64> = q_oo_i.t().dot(&gamma_ab_off_diag.dot(&q_vv_j)).into_shape(n_ct).unwrap();

            // add all terms of the diagonal CT block
            let mut ct_arr:Array1<f64> = Array1::zeros((n_ct));
            for i in 0..n_ct{
                ct_arr[i] = lcmo_h[[index_i,index_i]] - lcmo_h[[index_j,index_j]] + 2.0 * exchange[i] - coulomb[i];
            }
            // assign the array to the correct position in the cis matrix
            let cis_index_ab:usize = dim_le + index_i * (self.n_mol-1) + index_j-1;
            cis_matrix.slice_mut(s![cis_index_ab..(cis_index_ab+n_ct),cis_index_ab..(cis_index_ab+n_ct)]).assign(&ct_arr);


            // Calculate the off-diagonal elements of the LE - LE block
            // TODO: Try another version instead of a loop
            let le_le_matrix:Array2<f64> = le_le_two_electron_loop(
                c_mo_i,
                c_mo_j,
                exc_coeff_i,
                exc_coeff_j,
                virt_indices_i,
                virt_indices_j,
                n_le,
                gamma_ao_pair,
                s_pair,
            );
            // Assign the calculatged LE-LE block of I and J to the correct elements of the cis matrix
            cis_matrix.slice_mut(s![(index_i * n_le)..(index_i * n_le)+n_le,(index_j * n_le)..(index_j * n_le)+n_le]).assign(&le_le_matrix);

            // // Do the same procedure for the combination of the fragments j and i
            // // electron-hole exchange integral (ia|ia)
            // // calculate the exchange integral with a loop
            // let exchange:Array1<f64> = inter_fragment_exchange_integral(
            //     &atoms[self.monomers[index_j].slice.atom_as_range()],
            //     &atoms[self.monomers[index_i].slice.atom_as_range()],
            //     c_mo_j,
            //     c_mo_i,
            //     s_pair,
            //     occ_indices_j,
            //     virt_indices_i,
            //     gamma_ab,
            // );
            // // alternative way of calculating the exchange integral
            // // using transition densities
            // let qtrans_mu_nu:Array2<f64> = inter_fragment_trans_charges(
            //     &atoms[self.monomers[index_j].slice.atom_as_range()],
            //     &atoms[self.monomers[index_i].slice.atom_as_range()],
            //     c_mo_j,
            //     c_mo_i,
            //     s_pair,
            //     occ_indices_j,
            //     virt_indices_i,
            // );
            // let exchange_alternative:Array1<f64> = qtrans_mu_nu.t().dot(&gamma_ab.dot(&qtrans_mu_nu)).into_shape([n_occ_m*n_virt_m]);
            //
            // // coulomb-interaction integral (ii|aa) = sum_AB q_A^II gamma_AB q_B^aa
            // // q_oo(J) * gamma_AB (pair) * q_vv(I)
            // let coulomb:Array1<f64> = q_oo_j.t().dot(&gamma_ab_off_diag.t().dot(&q_vv_i)).into_shape(n_ct).unwrap();
            //
            // // add all terms of the diagonal CT block
            // let mut ct_arr:Array1<f64> = Array1::zeros((n_ct));
            // for i in 0..n_ct{
            //     ct_arr[i] = lcmo_h[[index_j,index_j]] - lcmo_h[[index_i,index_i]] + 2.0 * exchange[i] - coulomb[i];
            // }
            // // assign the array to the correct position in the cis matrix
            // let cis_index_ba:usize = dim_le + index_j * (self.n_mol-1) + index_i;
            // cis_matrix.slice_mut(s![cis_index_ba..(cis_index_ba+n_ct),cis_index_ba..(cis_index_ba+n_ct)]).assign(&ct_arr);
        }
    }
}