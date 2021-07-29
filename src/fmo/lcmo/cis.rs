use crate::fmo::{Monomer, SuperSystem};
use crate::initialization::Atom;
use ndarray::{Array2, ArrayView1, s, ArrayView2, Array1};
use crate::excited_states::trans_charges_fast;

impl SuperSystem{
    pub fn build_cis_matrix(&self){
        // TODO: READ THIS FROM THE INPUT FILE
        // Number of active orbitals per monomer
        let n_occ_m: usize = 1;
        let n_virt_m: usize = 1;
        let n_le:usize = 2;
        let n_ct:usize = n_occ_m * n_virt_m;

        // Reference to the atoms.
        let atoms: &[Atom] = &self.atoms;
        // build the lcmo hamiltonian
        let lcmo_h:Array2<f64> = self.build_lcmo_hamiltonian();

        // The dimension of the cis matrix.
        // Number of fragments * number of local excitations +
        // number of pairs * (n_active_occ * n_active_virt)
        let dim:usize = self.n_mol * n_le + self.pairs.len() * n_ct;
        let mut cis_matrix:Array2<f64> = Array2::zeros([dim,dim]);

        // the diagonal elements for the LE-LE block are set
        for (i, mol) in self.monomers.iter().enumerate() {
            // reference to the excited state energies of the monomers
            let excited_states:ArrayView1<f64> = mol.properties.excited_states().unwrap();
            // fill the diagonal of the matrix
            cis_matrix.slice_mut(s![i..(i+n_le),i..(i+n_le)]).assign(&excited_states.slice(s![0..n_le]));
        }
        // the diagonal elements for the CT-CT block are set
        for (i, pair) in self.pairs.iter().enumerate(){
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
            let occ_indices_i: [usize] = self.monomers[index_i].properties.occ_indices().unwrap()[0..n_occ_m];
            // virtual orbitals of fragment I.
            let virt_indices_i: [usize] = self.monomers[index_i].properties.virt_indices().unwrap()[0..n_virt_m];
            // occupied orbitals of fragment J.
            let occ_indices_j: [usize] = self.monomers[index_j].properties.occ_indices().unwrap()[0..n_occ_m];;
            // virtual orbitals of fragment J.
            let virt_indices_j: [usize] = self.monomers[index_j].properties.virt_indices().unwrap()[0..n_virt_m];;

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
            gamma_ab = gamma_ab.slice(s![..n_atoms_i,n_atoms_i..]);

            // eletron-hole exchange integral (ia|ia)
            // TODO: new transition charges between the monomer fragements are required!

            // coulomb-interaction integral (ii|aa) = sum_AB q_A^II gamma_AB q_B^aa
            // q_oo(I) * gamma_AB (pair) * q_vv(J)
            let coulomb:Array2<f64> = q_oo_i.t().dot(&gamma_ab.dot(&q_vv_j));


            // let mut ct_arr:Array1<f64> = Array1::zeros((n_ct));
            // for i in 0..n_ct{
            //     ct_arr[i] = lcmo_h[[index_i,index_i]] - lcmo_h[[index_j,index_j]];
            // }
            // cis_matrix.slice_mut(s![i..(i+n_ct),i..(i+n_ct)]).assign(&ct_arr);
        }
    }
}