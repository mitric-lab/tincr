use crate::excited_states::trans_charges_fast;
use crate::fmo::lcmo::helpers::*;
use crate::fmo::{Monomer, SuperSystem};
use crate::initialization::Atom;
use ndarray::prelude::*;
use std::ops::AddAssign;

impl SuperSystem {
    pub fn build_cis_matrix(&self, lcmo_h: ArrayView2<f64>) {
        // TODO: READ THIS FROM THE INPUT FILE
        // Number of active orbitals per monomer
        let n_occ_m: usize = 1;
        let n_virt_m: usize = 1;
        let n_le: usize = 2;
        let n_ct: usize = n_occ_m * n_virt_m;

        // Reference to the atoms.
        let atoms: &[Atom] = &self.atoms;

        // Get lcmo hamiltonian as input
        // build the lcmo hamiltonian
        // let lcmo_h:Array2<f64> = self.build_lcmo_hamiltonian();

        // The dimension of the cis matrix.
        // Number of fragments * number of local excitations +
        // number of pairs * (n_active_occ * n_active_virt)
        let dim_le: usize = self.n_mol * n_le;
        let dim_ct: usize = (self.pairs.len() + self.esd_pairs.len()) * n_ct;
        let dim_cis: usize = dim_le + dim_ct;
        let mut cis_matrix: Array2<f64> = Array2::zeros([dim_cis, dim_cis]);

        // the diagonal elements for the LE-LE block are set
        for (i, mol) in self.monomers.iter().enumerate() {
            // reference to the excited state energies of the monomers
            let excited_states: ArrayView1<f64> = mol.properties.excited_states().unwrap();
            // fill the diagonal of the matrix
            for le_index in 0..n_le {
                cis_matrix[[i + le_index, i + le_index]] = excited_states[le_index];
            }
        }
        // the diagonal elements for the CT-CT block are set
        for pair in self.pairs.iter() {
            // set pair indices
            let index_i: usize = pair.i;
            let index_j: usize = pair.j;
            // set n_atoms of fragments
            let n_atoms_i: usize = self.monomers[index_i].n_atoms;
            let n_atoms_j: usize = self.monomers[index_j].n_atoms;
            // reference to the mo coefficients of fragment I
            let c_mo_i: ArrayView2<f64> = self.monomers[index_i].properties.orbs().unwrap();
            // reference to the mo coefficients of fragment J
            let c_mo_j: ArrayView2<f64> = self.monomers[index_j].properties.orbs().unwrap();
            // reference to the overlap matrix of fragment I
            let s_i: ArrayView2<f64> = self.monomers[index_i].properties.s().unwrap();
            // reference to the overlap matrix of fragment J
            let s_j: ArrayView2<f64> = self.monomers[index_j].properties.s().unwrap();
            // active occupied orbitals of fragment I.
            let occ_indices_i: &[usize] =
                &self.monomers[index_i].properties.occ_indices().unwrap()[0..n_occ_m];
            // active virtual orbitals of fragment I.
            let virt_indices_i: &[usize] =
                &self.monomers[index_i].properties.virt_indices().unwrap()[0..n_virt_m];
            // active occupied orbitals of fragment J.
            let occ_indices_j: &[usize] =
                &self.monomers[index_j].properties.occ_indices().unwrap()[0..n_occ_m];
            // active virtual orbitals of fragment J.
            let virt_indices_j: &[usize] =
                &self.monomers[index_j].properties.virt_indices().unwrap()[0..n_virt_m];
            // occupied orbitals of fragment I.
            let full_occ_indices_i: &[usize] =
                &self.monomers[index_i].properties.occ_indices().unwrap();
            // virtual orbitals of fragment I.
            let full_virt_indices_i: &[usize] =
                &self.monomers[index_i].properties.virt_indices().unwrap();
            // occupied orbitals of fragment J.
            let full_occ_indices_j: &[usize] =
                &self.monomers[index_j].properties.occ_indices().unwrap();
            // virtual orbitals of fragment J.
            let full_virt_indices_j: &[usize] =
                &self.monomers[index_j].properties.virt_indices().unwrap();

            // full nocc, nvirt and norbs
            let full_nocc_i: usize = full_occ_indices_i.len();
            let full_nocc_j: usize = full_occ_indices_j.len();
            let full_nvirt_i: usize = full_virt_indices_i.len();
            let full_nvirt_j: usize = full_virt_indices_j.len();
            let norbs_i: usize = full_virt_indices_i[full_nvirt_i - 1] + 1;
            let norbs_j: usize = full_virt_indices_j[full_nvirt_j - 1] + 1;

            // get excited coefficients of the fragment I
            let exc_coeff_i: ArrayView3<f64> = self.monomers[index_i]
                .properties
                .excited_coefficients()
                .unwrap();
            // get excited coefficients of the fragment J
            let exc_coeff_j: ArrayView3<f64> = self.monomers[index_j]
                .properties
                .excited_coefficients()
                .unwrap();

            // get gamma matrix of the pair
            let mut gamma: ArrayView2<f64> = pair.properties.gamma().unwrap();
            let gamma_ab_off_diag: ArrayView2<f64> = gamma.slice(s![..n_atoms_i, n_atoms_i..]);

            // get reference to full overlap matrix of the pair
            let s_pair: ArrayView2<f64> = pair.properties.s().unwrap();
            let s_off_diag: ArrayView2<f64> = s_pair.slice(s![0..norbs_i, norbs_i..]);

            // calculate outer-diagonal LE-CT one-electron matrix elements
            // for the LE on fragment I and the CT from I->J
            let le_i_ct_ij: Array2<f64> = le_i_ct_one_electron_ij(
                occ_indices_i,
                virt_indices_j,
                exc_coeff_i,
                lcmo_h,
                n_le,
                n_ct,
            );
            let cis_index_le: usize = (index_i * n_le);
            // let cis_index_ct:usize = dim_le + index_i * (self.n_mol-1) + index_j-1; // index if n_occ and n_virt are 1
            let cis_index_ct: usize =
                dim_le + index_i * (self.n_mol - 1) * n_ct + (index_j - 1) * n_ct;
            cis_matrix
                .slice_mut(s![
                    cis_index_le..cis_index_le + n_le,
                    cis_index_ct..cis_index_ct + n_ct
                ])
                .add_assign(&le_i_ct_ij);

            // calculate outer-diagonal LE-CT one-electron matrix elements
            // for the LE on fragment I and the CT from J->I
            let le_i_ct_ji: Array2<f64> = le_i_ct_one_electron_ji(
                occ_indices_j,
                virt_indices_i,
                exc_coeff_i,
                lcmo_h,
                n_le,
                n_ct,
            );
            let cis_index_ct: usize = dim_le + index_j * (self.n_mol - 1) * n_ct + index_i * n_ct; // index if n_occ and n_virt are 1
            cis_matrix
                .slice_mut(s![
                    cis_index_le..cis_index_le + n_le,
                    cis_index_ct..cis_index_ct + n_ct
                ])
                .add_assign(&le_i_ct_ji);

            // calculate the CT transition charges for monomer I
            let (qov_ct_i, qoo_ct_i, qvv_ct_i): (Array2<f64>, Array2<f64>, Array2<f64>) =
                trans_charges_fast(
                    n_atoms_i,
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    c_mo_i,
                    s_i,
                    &occ_indices_i,
                    &virt_indices_i,
                );
            // calculate the CT transition charges for monomer J
            let (qov_ct_j, qoo_ct_j, qvv_ct_j): (Array2<f64>, Array2<f64>, Array2<f64>) =
                trans_charges_fast(
                    n_atoms_j,
                    &atoms[self.monomers[index_j].slice.atom_as_range()],
                    c_mo_j,
                    s_j,
                    &occ_indices_j,
                    &virt_indices_j,
                );

            // CT-CT diagonal block of the fragment combination I-J
            // electron-hole exchange integral (ia|ia)
            // calculate the exchange integral using transition densities
            let qov_ct_ij: Array2<f64> = inter_fragment_trans_charge_ct_ov(
                &atoms[self.monomers[index_i].slice.atom_as_range()],
                &atoms[self.monomers[index_j].slice.atom_as_range()],
                c_mo_i,
                c_mo_j,
                s_off_diag,
                occ_indices_i,
                virt_indices_j,
            );
            let exchange_ij: Array2<f64> = 2.0 * qov_ct_ij.t().dot(&gamma.dot(&qov_ct_ij));

            // coulomb-interaction integral (ii|aa) = sum_AB q_A^II gamma q_B^aa
            // q_oo(I) * gamma (pair) * q_vv(J)
            let coulomb_ij: Array2<f64> = qoo_ct_i.t().dot(&gamma_ab_off_diag.dot(&qvv_ct_j));

            // calculate the CT-CT interaction for the fragment combination J-I
            let qov_ct_ji: Array2<f64> = inter_fragment_trans_charge_ct_ov(
                &atoms[self.monomers[index_j].slice.atom_as_range()],
                &atoms[self.monomers[index_i].slice.atom_as_range()],
                c_mo_j,
                c_mo_i,
                s_off_diag.t(),
                occ_indices_j,
                virt_indices_i,
            );
            let exchange_ji: Array2<f64> = 2.0 * qov_ct_ji.t().dot(&gamma.dot(&qov_ct_ji));
            let coulomb_ji: Array2<f64> = qoo_ct_j.t().dot(&gamma_ab_off_diag.dot(&qvv_ct_i));

            // assign the arrays to the correct position in the cis matrix
            let cis_index_ab: usize =
                dim_le + index_i * (self.n_mol - 1) * n_ct + (index_j - 1) * n_ct;
            let cis_index_ba: usize = dim_le + index_j * (self.n_mol - 1) * n_ct + index_i * n_ct;
            // add fock-matrix term to the diagonal CT block
            for i in 0..n_ct {
                cis_matrix[[cis_index_ab + i, cis_index_ab + i]] +=
                    lcmo_h[[index_i, index_i]] - lcmo_h[[index_j, index_j]];
                cis_matrix[[cis_index_ba + i, cis_index_ba + i]] +=
                    lcmo_h[[index_j, index_j]] - lcmo_h[[index_i, index_i]];
            }
            cis_matrix
                .slice_mut(s![
                    cis_index_ab..(cis_index_ab + n_ct),
                    cis_index_ab..(cis_index_ab + n_ct)
                ])
                .add_assign(&(exchange_ij - coulomb_ij));
            cis_matrix
                .slice_mut(s![
                    cis_index_ba..(cis_index_ba + n_ct),
                    cis_index_ba..(cis_index_ba + n_ct)
                ])
                .add_assign(&(exchange_ji - coulomb_ji));

            // calculate the transition charges using the full space of orbitals for the
            // LE-LE calculation (Eq. 15)
            // interfragment transition charges
            let (qoo_lele_ij, qvv_lele_ij): (Array2<f64>, Array2<f64>) =
                inter_fragment_trans_charges_oovv(
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    &atoms[self.monomers[index_j].slice.atom_as_range()],
                    c_mo_i,
                    c_mo_j,
                    s_off_diag,
                    full_occ_indices_i,
                    full_occ_indices_j,
                    full_virt_indices_i,
                    full_virt_indices_j,
                );
            // transition charges of both fragments
            // calculate the LE transition charges for monomer I
            let (qov_le_i, qoo_le_i, qvv_le_i): (Array2<f64>, Array2<f64>, Array2<f64>) =
                trans_charges_fast(
                    n_atoms_i,
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    c_mo_i,
                    s_i,
                    full_occ_indices_i,
                    full_virt_indices_i,
                );
            // calculate the LE transition charges for monomer J
            let (qov_le_j, qoo_le_j, qvv_le_j): (Array2<f64>, Array2<f64>, Array2<f64>) =
                trans_charges_fast(
                    n_atoms_j,
                    &atoms[self.monomers[index_j].slice.atom_as_range()],
                    c_mo_j,
                    s_j,
                    full_occ_indices_j,
                    full_virt_indices_j,
                );
            let le_le_matrix: Array2<f64> = le_le_two_electron(
                n_le,
                exc_coeff_i,
                exc_coeff_j,
                gamma,
                gamma_ab_off_diag,
                qov_le_i.view(),
                qov_le_j.view(),
                qoo_lele_ij.view(),
                qvv_lele_ij.view(),
            );
            // // Calculate the off-diagonal elements of the LE - LE block
            // let le_le_matrix:Array2<f64> = le_le_two_electron_loop(
            //     c_mo_i,
            //     c_mo_j,
            //     exc_coeff_i,
            //     exc_coeff_j,
            //     virt_indices_i,
            //     virt_indices_j,
            //     n_le,
            //     gamma_ao_pair,
            //     s_pair,
            // );
            // Assign the calculatged LE-LE block of I and J to the correct elements of the cis matrix
            cis_matrix
                .slice_mut(s![
                    (index_i * n_le)..(index_i * n_le) + n_le,
                    (index_j * n_le)..(index_j * n_le) + n_le
                ])
                .assign(&le_le_matrix);

            // Calculate the LE-CT interaction
            // interfragment transition charges for LE-CT block
            // Integrals 2*(ia|jb) - (ij|ab) (Eq. 17/18)
            // intrafragment transition charges for fragment I are required
            let (qoo_lect_ii, qvv_lect_ii): (Array2<f64>, Array2<f64>) =
                inter_fragment_trans_charges_oovv(
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    c_mo_i,
                    c_mo_i,
                    s_i,
                    full_occ_indices_i,
                    occ_indices_i,
                    full_virt_indices_i,
                    virt_indices_i,
                );
            // interfragment transition charges
            let (qoo_lect_ij, qvv_lect_ij): (Array2<f64>, Array2<f64>) =
                inter_fragment_trans_charges_oovv(
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    &atoms[self.monomers[index_j].slice.atom_as_range()],
                    c_mo_i,
                    c_mo_j,
                    s_off_diag,
                    full_occ_indices_i,
                    occ_indices_j,
                    full_virt_indices_i,
                    virt_indices_j,
                );
            // LE-CT_ij block (Eq. 17)
            let le_ct_matrix_ij: Array2<f64> = le_ct_two_electron(
                n_le,
                n_ct,
                exc_coeff_i,
                gamma,
                gamma_ab_off_diag,
                qov_le_i.view(),
                qov_ct_ji.view(),
                qoo_lect_ij.view(),
                qvv_lect_ii.view(),
                n_occ_m,
                n_virt_m,
            );
            // LE-CT_ji block (Eq. 18)
            let le_ct_matrix_ji: Array2<f64> = le_ct_two_electron(
                n_le,
                n_ct,
                exc_coeff_i,
                gamma,
                gamma_ab_off_diag,
                qov_le_i.view(),
                qov_ct_ij.view(),
                qoo_lect_ii.view(),
                qvv_lect_ij.view(),
                n_occ_m,
                n_virt_m,
            );
            // assign matrices to the correct parts of the cis matrix
            cis_matrix
                .slice_mut(s![
                    (index_i * n_le)..(index_i * n_le) + n_le,
                    cis_index_ab..(cis_index_ab + n_ct)
                ])
                .assign(&le_ct_matrix_ji);
            cis_matrix
                .slice_mut(s![
                    (index_i * n_le)..(index_i * n_le) + n_le,
                    cis_index_ba..(cis_index_ba + n_ct)
                ])
                .assign(&le_ct_matrix_ij);

            // CT-CT outer-diagonal block (Eq. 20)
            // term 1 (ia|jb) on fragments (JI, IJ)
            let term_1: Array2<f64> = 2.0 * qov_ct_ji.t().dot(&gamma.dot(&qov_ct_ij));

            // term 2 (ij|ab) on fragments (JI,IJ)
            // calculate qoo and qvv transition charges
            let (qoo_ctct_ij, qvv_ctct_ij): (Array2<f64>, Array2<f64>) =
                inter_fragment_trans_charges_oovv(
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    &atoms[self.monomers[index_j].slice.atom_as_range()],
                    c_mo_i,
                    c_mo_j,
                    s_off_diag,
                    occ_indices_i,
                    occ_indices_j,
                    virt_indices_i,
                    virt_indices_j,
                );
            let qoo_ctct_ji: Array2<f64> = qoo_ctct_ij
                .into_shape([n_atoms_i + n_atoms_j, n_occ_m, n_occ_m])
                .unwrap()
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .into_shape([n_atoms_i + n_atoms_j, n_occ_m * n_occ_m])
                .unwrap()
                .to_owned();
            let term_2: Array2<f64> = qoo_ctct_ji.t().dot(&gamma.dot(&qvv_ctct_ij));

            cis_matrix
                .slice_mut(s![
                    cis_index_ab..(cis_index_ab + n_ct),
                    cis_index_ba..(cis_index_ba + n_ct)
                ])
                .add_assign(&(term_1 - term_2));
        }

        // ESD pair loop
        for pair in self.esd_pairs.iter() {
            // set pair indices
            let index_i: usize = pair.i;
            let index_j: usize = pair.j;
            // set n_atoms of fragments
            let n_atoms_i: usize = self.monomers[index_i].n_atoms;
            let n_atoms_j: usize = self.monomers[index_j].n_atoms;
            // reference to the mo coefficients of fragment I
            let c_mo_i: ArrayView2<f64> = self.monomers[index_i].properties.orbs().unwrap();
            // reference to the mo coefficients of fragment J
            let c_mo_j: ArrayView2<f64> = self.monomers[index_j].properties.orbs().unwrap();
            // reference to the overlap matrix of fragment I
            let s_i: ArrayView2<f64> = self.monomers[index_i].properties.s().unwrap();
            // reference to the overlap matrix of fragment J
            let s_j: ArrayView2<f64> = self.monomers[index_j].properties.s().unwrap();
            // active occupied orbitals of fragment I.
            let occ_indices_i: &[usize] =
                &self.monomers[index_i].properties.occ_indices().unwrap()[0..n_occ_m];
            // active virtual orbitals of fragment I.
            let virt_indices_i: &[usize] =
                &self.monomers[index_i].properties.virt_indices().unwrap()[0..n_virt_m];
            // active occupied orbitals of fragment J.
            let occ_indices_j: &[usize] =
                &self.monomers[index_j].properties.occ_indices().unwrap()[0..n_occ_m];
            // active virtual orbitals of fragment J.
            let virt_indices_j: &[usize] =
                &self.monomers[index_j].properties.virt_indices().unwrap()[0..n_virt_m];
            // occupied orbitals of fragment I.
            let full_occ_indices_i: &[usize] =
                &self.monomers[index_i].properties.occ_indices().unwrap();
            // virtual orbitals of fragment I.
            let full_virt_indices_i: &[usize] =
                &self.monomers[index_i].properties.virt_indices().unwrap();
            // occupied orbitals of fragment J.
            let full_occ_indices_j: &[usize] =
                &self.monomers[index_j].properties.occ_indices().unwrap();
            // virtual orbitals of fragment J.
            let full_virt_indices_j: &[usize] =
                &self.monomers[index_j].properties.virt_indices().unwrap();

            // get excited coefficients of the fragment I
            let exc_coeff_i: ArrayView3<f64> = self.monomers[index_i]
                .properties
                .excited_coefficients()
                .unwrap();
            // get excited coefficients of the fragment J
            let exc_coeff_j: ArrayView3<f64> = self.monomers[index_j]
                .properties
                .excited_coefficients()
                .unwrap();

            // get gamma matrix of the pair
            let mut gamma: ArrayView2<f64> = pair.properties.gamma().unwrap();
            let gamma_ab_off_diag: ArrayView2<f64> = gamma.slice(s![..n_atoms_i, n_atoms_i..]);

            // calculate the CT transition charges for monomer I
            let (qov_ct_i, qoo_ct_i, qvv_ct_i): (Array2<f64>, Array2<f64>, Array2<f64>) =
                trans_charges_fast(
                    n_atoms_i,
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    c_mo_i,
                    s_i,
                    &occ_indices_i,
                    &virt_indices_i,
                );
            // calculate the CT transition charges for monomer J
            let (qov_ct_j, qoo_ct_j, qvv_ct_j): (Array2<f64>, Array2<f64>, Array2<f64>) =
                trans_charges_fast(
                    n_atoms_j,
                    &atoms[self.monomers[index_j].slice.atom_as_range()],
                    c_mo_j,
                    s_j,
                    &occ_indices_j,
                    &virt_indices_j,
                );

            // CT-CT block of the cis matrix
            // the exchange term is neglected for the ESD pairs
            let coulomb_ij: Array2<f64> =
                -1.0 * qoo_ct_i.t().dot(&gamma_ab_off_diag.dot(&qvv_ct_j));
            let coulomb_ji: Array2<f64> =
                -1.0 * qoo_ct_j.t().dot(&gamma_ab_off_diag.dot(&qvv_ct_i));

            // assign the arrays to the correct position in the cis matrix
            let cis_index_ab: usize =
                dim_le + index_i * (self.n_mol - 1) * n_ct + (index_j - 1) * n_ct;
            let cis_index_ba: usize = dim_le + index_j * (self.n_mol - 1) * n_ct + index_i * n_ct;
            cis_matrix
                .slice_mut(s![
                    cis_index_ab..(cis_index_ab + n_ct),
                    cis_index_ab..(cis_index_ab + n_ct)
                ])
                .add_assign(&coulomb_ij);
            cis_matrix
                .slice_mut(s![
                    cis_index_ba..(cis_index_ba + n_ct),
                    cis_index_ba..(cis_index_ba + n_ct)
                ])
                .add_assign(&coulomb_ji);

            // LE-LE block of the cis matrix
            // calculate the LE transition charges for monomer I
            let (qov_le_i, qoo_le_i, qvv_le_i): (Array2<f64>, Array2<f64>, Array2<f64>) =
                trans_charges_fast(
                    n_atoms_i,
                    &atoms[self.monomers[index_i].slice.atom_as_range()],
                    c_mo_i,
                    s_i,
                    full_occ_indices_i,
                    full_virt_indices_i,
                );
            // calculate the LE transition charges for monomer J
            let (qov_le_j, qoo_le_j, qvv_le_j): (Array2<f64>, Array2<f64>, Array2<f64>) =
                trans_charges_fast(
                    n_atoms_j,
                    &atoms[self.monomers[index_j].slice.atom_as_range()],
                    c_mo_j,
                    s_j,
                    full_occ_indices_j,
                    full_virt_indices_j,
                );

            let le_le_matrix: Array2<f64> = le_le_two_electron_esd(
                n_le,
                exc_coeff_i,
                exc_coeff_j,
                gamma,
                gamma_ab_off_diag,
                qov_le_i.view(),
                qov_le_j.view(),
            );
            cis_matrix
                .slice_mut(s![
                    (index_i * n_le)..(index_i * n_le) + n_le,
                    (index_j * n_le)..(index_j * n_le) + n_le
                ])
                .assign(&le_le_matrix);
        }
    }
}
