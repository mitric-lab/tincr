use crate::fmo::{Monomer, Pair, SuperSystem, ExcitedStateMonomerGradient, ESDPair, ChargeTransferPair, PairType};
use crate::excited_states::{trans_charges};
use crate::gradients::helpers::{f_lr, f_v, h_minus, h_plus_no_lr, zvector_lc, zvector_no_lc, Hplus, HplusType, h_a_nolr, tda_zvector_no_lc, Hav, tda_zvector_lc};
use crate::utils::ToOwnedF;
use ndarray::prelude::*;
use ndarray_linalg::{into_col, into_row, IntoTriangular, Solve, UPLO};
use crate::fmo::helpers::get_pair_slice;
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_gradients_ao_wise, gamma_ao_wise, gamma_ao_wise_from_gamma_atomwise, gamma_atomwise_ab};
use crate::scc::h0_and_s::{h0_and_s_ab, h0_and_s_gradients};

impl SuperSystem{
    pub fn new_charge_transfer_pair_gradient(
        &mut self,
        ct_state:&ChargeTransferPair,
    ) -> Array1<f64> {
        // get pair type
        let pair_type: PairType = self.properties.type_of_pair(ct_state.m_h, ct_state.m_l);

        let ct_gradient:Array1<f64> = if pair_type == PairType::Pair {
            // get pair index
            let pair_index: usize = self.properties.index_of_pair(ct_state.m_h, ct_state.m_l);
            // get correct pair from pairs vector
            let pair_ij: &mut Pair = &mut self.pairs[pair_index];
            // get monomers
            let m_i: &Monomer = &self.monomers[pair_ij.i];
            let m_j: &Monomer = &self.monomers[pair_ij.j];

            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            pair_ij.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
            pair_ij.prepare_charge_transfer_gradient(&pair_atoms, m_i, m_j, ct_state);
            pair_ij.tda_gradient_lc(0)
            // reset gradient specific properties
            // pair_ij.properties.reset_gradient();
        } else {
            // Do something for ESD pairs
            // get pair index
            let pair_index: usize = self.properties.index_of_esd_pair(ct_state.m_h, ct_state.m_l);
            // get correct pair from pairs vector
            let pair_ij: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // get monomers
            let m_i: &Monomer = &self.monomers[pair_ij.i];
            let m_j: &Monomer = &self.monomers[pair_ij.j];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            // do a scc calculation of the ESD pair
            pair_ij.prepare_scc(&pair_atoms, m_i, m_j);
            pair_ij.run_scc(&pair_atoms, self.config.scf);

            pair_ij.prepare_lcmo_gradient(&pair_atoms);
            pair_ij.prepare_charg_transfer_gradient(&pair_atoms, m_i, m_j, ct_state);
            pair_ij.tda_gradient_nolc(0)
            // pair_ij.properties.reset_gradient();
        };

        return ct_gradient;
    }
}

impl Pair{
    pub fn prepare_charge_transfer_gradient(
        &mut self,
        atoms:&[Atom],
        m_i:&Monomer,
        m_j:&Monomer,
        ct_state:&ChargeTransferPair
    ){
        // check if occ and virt indices exist
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        if (self.properties.contains_key("occ_indices") == false) || (self.properties.contains_key("virt_indices") == true) {
            // calculate the number of electrons
            let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
            // get the indices of the occupied and virtual orbitals
            (0..self.n_orbs).for_each(|index| {
                if index < (n_elec / 2) {
                    occ_indices.push(index)
                } else {
                    virt_indices.push(index)
                }
            });

            self.properties.set_occ_indices(occ_indices.clone());
            self.properties.set_virt_indices(virt_indices.clone());
        }
        else{
            occ_indices = self.properties.occ_indices().unwrap().to_vec();
            virt_indices = self.properties.virt_indices().unwrap().to_vec();
        }
        let nocc:usize = occ_indices.len();
        let nvirt:usize = virt_indices.len();
        let nocc_i:usize = m_i.properties.occ_indices().unwrap().len();
        let nocc_j:usize = m_j.properties.occ_indices().unwrap().len();

        // set ct_energy
        let cis_energy:Array1<f64> = Array::from(vec!(ct_state.state_energy));
        // save in properties
        self.properties.set_ci_eigenvalues(cis_energy);

        // get the overlap matrices between the monomers and the dimer
        let s_i_ij:ArrayView2<f64> = self.properties.s_i_ij().unwrap();
        let s_j_ij:ArrayView2<f64> = self.properties.s_j_ij().unwrap();

        // if the hole is on fragment I
        if m_i.index == ct_state.m_h{
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_occ:ArrayView2<f64> = s_i_ij.slice(s![..nocc_i,..nocc]);
            let s_j_ij_virt:ArrayView2<f64> = s_j_ij.slice(s![nocc_j..,nocc..]);

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff:Array2<f64> = s_i_ij_occ.t().dot(&ct_state.eigenvectors.dot(&s_j_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff:Array3<f64> = Array3::zeros([1,nocc,nvirt]);
            cis_coeff.slice_mut(s![0,..,..]).assign(&transformed_ct_coeff);
            // save in properties
            self.properties.set_ci_coefficients(cis_coeff.into_shape([1,nocc*nvirt]).unwrap());
        }
        // if the hole is on fragment J
        else{
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_virt:ArrayView2<f64> = s_i_ij.slice(s![nocc_i..,nocc..]);
            let s_j_ij_occ:ArrayView2<f64> = s_j_ij.slice(s![..nocc_j,..nocc]);

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff:Array2<f64> = s_j_ij_occ.t().dot(&ct_state.eigenvectors.dot(&s_i_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff:Array3<f64> = Array3::zeros([1,nocc,nvirt]);
            cis_coeff.slice_mut(s![0,..,..]).assign(&transformed_ct_coeff);
            // save in properties
            self.properties.set_ci_coefficients(cis_coeff.into_shape([1,nocc*nvirt]).unwrap());
        }

        // calculate transition charges
        let tmp: (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
            self.n_atoms,
            atoms,
            self.properties.orbs().unwrap(),
            self.properties.s().unwrap(),
            &occ_indices,
            &virt_indices,
        );
        self.properties.set_q_ov(tmp.0);
        self.properties.set_q_oo(tmp.1);
        self.properties.set_q_vv(tmp.2);
    }
}

impl ESDPair{
    pub fn prepare_charg_transfer_gradient(
        &mut self,
        atoms:&[Atom],
        m_i:&Monomer,
        m_j:&Monomer,
        ct_state:&ChargeTransferPair
    ){
        // check if occ and virt indices exist
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        if (self.properties.contains_key("occ_indices") == false) || (self.properties.contains_key("virt_indices") == true) {
            // calculate the number of electrons
            let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
            // get the indices of the occupied and virtual orbitals
            (0..self.n_orbs).for_each(|index| {
                if index < (n_elec / 2) {
                    occ_indices.push(index)
                } else {
                    virt_indices.push(index)
                }
            });

            self.properties.set_occ_indices(occ_indices.clone());
            self.properties.set_virt_indices(virt_indices.clone());
        }
        else{
            occ_indices = self.properties.occ_indices().unwrap().to_vec();
            virt_indices = self.properties.virt_indices().unwrap().to_vec();
        }
        let nocc:usize = occ_indices.len();
        let nvirt:usize = virt_indices.len();
        let nocc_i:usize = m_i.properties.occ_indices().unwrap().len();
        let nocc_j:usize = m_j.properties.occ_indices().unwrap().len();

        // set ct_energy
        let cis_energy:Array1<f64> = Array::from(vec!(ct_state.state_energy));
        // save in properties
        self.properties.set_ci_eigenvalues(cis_energy);

        // get overlap matrix of the esd pair from the properties
        let s:ArrayView2<f64> = self.properties.s().unwrap();
        // Reference to the MO coefficients of monomer I.
        let orbs_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
        // Reference to the MO coefficients of monomer J.
        let orbs_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
        // Reference to the MO coefficients of the pair IJ.
        let orbs_ij: ArrayView2<f64> = self.properties.orbs().unwrap();
        // Overlap between orbitals of monomer I and dimer IJ.
        let s_i_ij: Array2<f64> = (orbs_i.t().dot(&s.slice(s![0..m_i.n_orbs, ..]))).dot(&orbs_ij);
        // Overlap between orbitals of monomer J and dimer IJ.
        let s_j_ij: Array2<f64> = (orbs_j.t().dot(&s.slice(s![m_i.n_orbs.., ..]))).dot(&orbs_ij);

        // if the hole is on fragment I
        if m_i.index == ct_state.m_h{
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_occ:ArrayView2<f64> = s_i_ij.slice(s![..nocc_i,..nocc]);
            let s_j_ij_virt:ArrayView2<f64> = s_j_ij.slice(s![nocc_j..,nocc..]);

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff:Array2<f64> = s_i_ij_occ.t().dot(&ct_state.eigenvectors.dot(&s_j_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff:Array3<f64> = Array3::zeros([1,nocc,nvirt]);
            cis_coeff.slice_mut(s![0,..,..]).assign(&transformed_ct_coeff);
            // save in properties
            self.properties.set_ci_coefficients(cis_coeff.into_shape([1,nocc*nvirt]).unwrap());
        }
        // if the hole is on fragment J
        else{
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_virt:ArrayView2<f64> = s_i_ij.slice(s![nocc_i..,nocc..]);
            let s_j_ij_occ:ArrayView2<f64> = s_j_ij.slice(s![..nocc_j,..nocc]);

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff:Array2<f64> = s_j_ij_occ.t().dot(&ct_state.eigenvectors.dot(&s_i_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff:Array3<f64> = Array3::zeros([1,nocc,nvirt]);
            cis_coeff.slice_mut(s![0,..,..]).assign(&transformed_ct_coeff);
            // save in properties
            self.properties.set_ci_coefficients(cis_coeff.into_shape([1,nocc*nvirt]).unwrap());
        }

        // calculate transition charges
        let tmp: (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
            self.n_atoms,
            atoms,
            self.properties.orbs().unwrap(),
            self.properties.s().unwrap(),
            &occ_indices,
            &virt_indices,
        );
        self.properties.set_q_ov(tmp.0);
        self.properties.set_q_oo(tmp.1);
        self.properties.set_q_vv(tmp.2);

        // prepare gamma and grad gamma AO matrix
        let g0_ao:Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
            self.properties.gamma().unwrap(),
            atoms,
            self.n_orbs
        );
        let (g1,g1_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
            &self.gammafunction,
            atoms,
            self.n_atoms,
            self.n_orbs,
        );
        self.properties.set_grad_gamma(g1);
        self.properties.set_gamma_ao(g0_ao);
        self.properties.set_grad_gamma_ao(g1_ao);

        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&atoms, self.n_orbs, &self.slako);
        self.properties.set_grad_s(grad_s);
        self.properties.set_grad_h0(grad_h0);
    }

}

// impl SuperSystem{
//     pub fn prepare_new_ct_gradient(&mut self,ct_state:&ChargeTransferPair){
//         let type_ij: PairType = self.properties.type_of_pair(ct_state.m_h, ct_state.m_l);
//
//         if type_ij == PairType::Pair{
//             let index:usize = self.properties.index_of_pair(ct_state.m_h,ct_state.m_l);
//             let pair:&mut Pair = &mut self.pairs[index];
//
//             // get pair atoms
//             let pair_atoms: Vec<Atom> = get_pair_slice(
//                 &self.atoms,
//                 self.monomers[pair.i].slice.atom_as_range(),
//                 self.monomers[pair.j].slice.atom_as_range(),
//             );
//             let m_i:&Monomer = &self.monomers[pair.i];
//             let m_j:&Monomer = &self.monomers[pair.j];
//             pair.prepare_new_ct_gradient(&pair_atoms,m_i,m_j);
//
//             let diff_p:Array2<f64> = pair.properties.p().unwrap() - pair.properties.p_ref().unwrap();
//             self.calculate_new_ct_gradient_nolc(
//                 ct_state,
//                 pair.properties.gamma().unwrap(),
//                 pair.properties.gamma_ao().unwrap(),
//                 pair.properties.grad_gamma_ao().unwrap(),
//                 pair.properties.grad_s().unwrap(),
//                 pair.properties.grad_h0().unwrap(),
//                 diff_p.view(),
//                 pair.properties.s().unwrap(),
//             );
//         }
//         if type_ij == PairType::ESD{
//             let index:usize = self.properties.index_of_esd_pair(ct_state.m_h,ct_state.m_l);
//             let pair:&mut ESDPair = &mut self.esd_pairs[index];
//
//             // get pair atoms
//             let pair_atoms: Vec<Atom> = get_pair_slice(
//                 &self.atoms,
//                 self.monomers[pair.i].slice.atom_as_range(),
//                 self.monomers[pair.j].slice.atom_as_range(),
//             );
//             let m_i:&Monomer = &self.monomers[pair.i];
//             let m_j:&Monomer = &self.monomers[pair.j];
//             pair_ij.prepare_scc(&pair_atoms, m_i, m_j);
//             pair_ij.run_scc(&pair_atoms, self.config.scf);
//             pair_ij.prepare_lcmo_gradient(&pair_atoms);
//
//             let diff_p:Array2<f64> = pair.properties.p().unwrap() - pair.properties.p_ref().unwrap();
//             self.calculate_new_ct_gradient_nolc(
//                 ct_state,
//                 pair.properties.gamma().unwrap(),
//                 pair.properties.gamma_ao().unwrap(),
//                 pair.properties.grad_gamma_ao().unwrap(),
//                 pair.properties.grad_s().unwrap(),
//                 pair.properties.grad_h0().unwrap(),
//                 diff_p.view(),
//                 pair.properties.s().unwrap(),
//             );
//         }
//     }
//
//     pub fn calculate_new_ct_gradient_nolc(
//         &self,
//         ct_state:&ChargeTransferPair,
//         g0:ArrayView2<f64>,
//         g0_ao:ArrayView2<f64>,
//         g1_ao:ArrayView3<f64>,
//         grad_s:ArrayView3<f64>,
//         grad_h:ArrayView3<f64>
//         diff_p:ArrayView2<f64>,
//         s:ArrayView2<f64>,
//     ){
//         let m_h:&Monomer = &self.monomers[ct_state.m_h];
//         let m_l:&Monomer = &self.monomers[ct_state.m_l];
//         let n_atoms:usize = ct_state.q_ov.dim().0;
//         let natoms_h:usize = m_h.n_atoms;
//         let natoms_l:usize = m_l.n_atoms;
//
//         // get occ and virt indices from properties
//         let occ_indices:&[usize] = m_h.properties.occ_indices().unwrap();
//         let virt_indices:&[usize] = m_l.properties.virt_indices().unwrap();
//
//         // set the occupied and virtual orbital energies
//         let orbe_h: ArrayView1<f64> = m_h.properties.orbe().unwrap();
//         let orbe_l: ArrayView1<f64> = m_l.properties.orbe().unwrap();
//         let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe_h[occ]).collect();
//         let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe_l[virt]).collect();
//
//         // transform the energies to a diagonal 2d matrix
//         let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
//         let ea: Array2<f64> = Array2::from_diag(&orbe_virt);
//
//         let n_occ: usize = orbe_occ.len();
//         let n_virt: usize = orbe_virt.len();
//
//         // excitation energy of the state
//         let omega_state:f64 = ct_state.state_energy;
//         // take state specific values from the excitation vectors
//         let x_state:ArrayView2<f64> = ct_state.eigenvectors.view();
//
//         // calculate the vectors u, v and t
//         // vectors U, V and T
//         let u_ab: Array2<f64> = 2.0 *x_state.t().dot(&x_state);
//         let u_ij: Array2<f64> = 2.0 *x_state.dot(&x_state.t());
//         let v_ab: Array2<f64> = 2.0 *ei.dot(&x_state).t().dot(&x_state);
//         let v_ij: Array2<f64> = 2.0 *x_state.dot(&ea).dot(&x_state.t());
//         let t_ab: Array2<f64> = x_state.t().dot(&x_state);
//         let t_ij: Array2<f64> = x_state.dot(&x_state.t());
//
//         // get the transition charges
//         let qtrans_ov: Array3<f64> = ct_state.q_ov.into_shape([n_atoms,n_occ,n_virt]).unwrap();
//         let qtrans_oo: ArrayView3<f64> = m_h.properties.q_oo().unwrap().into_shape((natoms_h,n_occ,n_occ)).unwrap();
//         let qtrans_vv: ArrayView3<f64> = m_l.properties.q_vv().unwrap().into_shape((natoms_l,n_virt,n_virt)).unwrap();
//
//         // compute hplus of tab and tij
//         let hplus_tab: Array2<f64> =
//             h_a_nolr(g0, qtrans_oo, qtrans_vv, t_ab.view());
//         let hplus_tij: Array2<f64> =
//             h_a_nolr(g0, qtrans_oo, qtrans_oo, t_ij.view());
//
//         // calculate q_ij
//         let g_ij: Array2<f64> = hplus_tab - hplus_tij;
//         let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;
//
//         // calculate q_ab
//         let q_ab: Array2<f64> = omega_state * u_ab + v_ab;
//
//         // calculate q_ia
//         let mut q_ia: Array2<f64> = x_state
//             .dot(&h_a_nolr(g0, qtrans_vv, qtrans_ov.view(), x_state).t());
//         q_ia = q_ia + h_a_nolr(g0, qtrans_ov.view(), qtrans_vv, t_ab.view());
//         q_ia = q_ia - h_a_nolr(g0, qtrans_ov.view(), qtrans_oo, t_ij.view());
//
//         // calculate q_ai
//         let q_ai: Array2<f64> = x_state.t().dot(&h_a_nolr(
//             g0,
//             qtrans_oo,
//             qtrans_ov.view(),
//             x_state,
//         ));
//
//         // calculate right hand side of the z-vector equation
//         let r_ia: Array2<f64> = &q_ai.t() - &q_ia;
//
//         // input for zvector routine
//         let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
//             .dot(&into_row(orbe_virt.clone()))
//             - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
//         let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape((n_occ * n_virt)).unwrap();
//         let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();
//
//         // calculate the z-vector
//         let z_ia: Array2<f64> = tda_zvector_no_lc(
//             omega_input.view(),
//             r_matrix.view(),
//             g0,
//             qtrans_ov.view(),
//             1,
//             Array1::zeros(n_atoms).view(),
//         );
//
//         // calculate w_ij
//         let mut w_ij: Array2<f64> =
//             q_ij + h_a_nolr(g0, qtrans_oo, qtrans_ov.view(), z_ia.view());
//         for i in 0..w_ij.dim().0 {
//             w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
//         }
//         // w_ia
//         let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia); //+ h_a_nolr(g0, qtrans_ov, qtrans_ov, z_ia.view());
//
//         // w_ai
//         let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();
//
//         // w_ab
//         let mut w_ab: Array2<f64> = q_ab;
//         for i in 0..w_ab.dim().0 {
//             w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
//         }
//
//         // build w matrix: combine w_ij, w_ia, w_ai and w_ab
//         let length: usize = n_occ + n_virt;
//         let mut w_matrix: Array2<f64> = Array::zeros((length, length));
//         for i in 0..w_ij.dim().0 {
//             w_matrix
//                 .slice_mut(s![i, ..w_ij.dim().1])
//                 .assign(&w_ij.slice(s![i, ..]));
//             w_matrix
//                 .slice_mut(s![i, w_ij.dim().1..])
//                 .assign(&w_ia.slice(s![i, ..]));
//         }
//         for i in 0..w_ai.dim().0 {
//             w_matrix
//                 .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
//                 .assign(&w_ai.slice(s![i, ..]));
//             w_matrix
//                 .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
//                 .assign(&w_ab.slice(s![i, ..]));
//         }
//
//         // calculate gradH: gradH0 + gradHexc
//         let f_dmd0: Array3<f64> = f_v(
//             diff_p.view(),
//             s,
//             grad_s,
//             g0_ao,
//             g1_ao,
//             n_atoms,
//             self.n_orbs,
//         );
//         let grad_h: Array3<f64> = &grad_h + &f_dmd0;
//
//         // set the occupied and virtuals orbital coefficients
//         let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
//         let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
//         let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
//         for (i, index) in occ_indices.iter().enumerate() {
//             orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
//         }
//         for (i, index) in virt_indices.iter().enumerate() {
//             orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
//         }
//
//         // transform t and z vectors to AO basis
//         let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
//         let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
//         let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));
//
//         // transform w matrix and excited state vectors to AO basis
//         let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
//         let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
//         let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));
//
//         // calculate contributions to the excited gradient
//         let f: Array3<f64> = f_v(
//             x_ao.view(),
//             s,
//             grad_s,
//             g0_ao,
//             g1_ao,
//             n_atoms,
//             self.n_orbs,
//         );
//
//         // assemble the excited gradient
//         let mut gradExc: Array1<f64> = Array::zeros(3 * n_atoms);
//         // gradH * (T + Z)
//         gradExc = gradExc
//             + grad_h
//             .into_shape([3 * n_atoms, self.n_orbs * self.n_orbs])
//             .unwrap()
//             .dot(
//                 &(t_vv - t_oo + z_ao)
//                     .into_shape(self.n_orbs * self.n_orbs)
//                     .unwrap(),
//             );
//         // - gradS * W
//         gradExc = gradExc
//             - grad_s
//             .into_shape([3 * n_atoms, self.n_orbs * self.n_orbs])
//             .unwrap()
//             .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
//         // 2.0 * sum (X+Y) F (X+Y)
//         gradExc = gradExc
//             + 2.0
//             * f.into_shape([3 * n_atoms, self.n_orbs * self.n_orbs])
//             .unwrap()
//             .dot(&x_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
//     }
//
//     pub fn calculate_new_ct_gradient_lc(&self, ct_state:&ChargeTransferPair){
//
//     }
// }