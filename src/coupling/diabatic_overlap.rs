use crate::fmo::basis::LocallyExcited;
use crate::fmo::lcmo::cis_gradient::{ReducedBasisState, ReducedCT, ReducedLE, ReducedParticle};
use crate::fmo::{Monomer, Pair, PairType, SuperSystem};
use crate::initialization::{Atom, System};
use ndarray::prelude::*;
use ndarray_linalg::Determinant;
use ndarray_npy::write_npy;
use ndarray_stats::QuantileExt;
use phf::Slice;

impl SuperSystem {
    pub fn build_occ_overlap(&self) -> (Array2<f64>, Array2<f64>, Vec<(usize, usize)>) {
        // get full overlap matrix
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        // get number of occupied orbitals of the SuperSystem
        let mut n_occs: usize = 0;
        let mut occ_vec: Vec<(usize, usize)> = Vec::new();

        let mut end: usize = 0;
        for monomer in self.monomers.iter() {
            let nocc: usize = monomer.properties.occ_indices().unwrap().len();
            n_occs += nocc;
            occ_vec.push((end, end + nocc));
            end += nocc;
        }

        // create empty matrix
        let mut s_transformed_occ: Array2<f64> = Array2::zeros((n_occs, n_occs));
        let mut s_transformed: Array2<f64> = Array2::zeros(s.dim());

        // transform the AO overlap matrices of the real pairs
        for pair in self.pairs.iter() {
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];

            let s_ij: ArrayView2<f64> = s.slice(s![m_i.slice.orb, m_j.slice.orb]);
            let orbs_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let orbs_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
            let nocc_i: usize = m_i.properties.occ_indices().unwrap().len();
            let nocc_j: usize = m_j.properties.occ_indices().unwrap().len();
            let start_i: usize = occ_vec[m_i.index].0;
            let end_i: usize = occ_vec[m_i.index].1;
            let start_j: usize = occ_vec[m_j.index].0;
            let end_j: usize = occ_vec[m_j.index].1;

            let transformed_s_ij: Array2<f64> = orbs_i.t().dot(&s_ij.dot(&orbs_j));
            s_transformed
                .slice_mut(s![m_i.slice.orb, m_j.slice.orb])
                .assign(&transformed_s_ij);
            s_transformed_occ
                .slice_mut(s![start_i..end_i, start_j..end_j])
                .assign(&transformed_s_ij.slice(s![..nocc_i, ..nocc_j]));
        }
        // array = array + array.T
        s_transformed = &s_transformed + &s_transformed.t();
        s_transformed_occ = &s_transformed_occ + &s_transformed_occ.t();

        // Add the diagonal
        let diag: Array2<f64> = Array::eye(s.dim().0);
        s_transformed = s_transformed + diag;
        let diag_occ: Array2<f64> = Array::eye(n_occs);
        s_transformed_occ = s_transformed_occ + diag_occ;

        return (s_transformed, s_transformed_occ, occ_vec);
    }

    pub fn diabatic_overlap(&self, lhs: &ReducedBasisState, rhs: &ReducedBasisState) -> f64 {
        match (lhs, rhs) {
            // Overlap between two LE states.
            (ReducedBasisState::LE(ref a), ReducedBasisState::LE(ref b)) => {
                println!("LE LE overlap");
                self.diabatic_overlap_le_le(a, b)
            }
            // Overlap between LE and CT state.
            (ReducedBasisState::LE(ref a), ReducedBasisState::CT(ref b)) => {
                println!("LE CT overlap");
                self.diabatic_overlap_le_ct(a, b)
            }
            // Overlap between CT and LE state.
            (ReducedBasisState::CT(ref a), ReducedBasisState::LE(ref b)) => {
                println!("CT LE overlap");
                self.diabatic_overlap_le_ct(b, a)
            }
            // Overlap between CT and CT
            (ReducedBasisState::CT(ref a), ReducedBasisState::CT(ref b)) => {
                println!("CT CT overlap");
                self.diabatic_overlap_ct_ct(a, b)
            }
        }
    }

    fn diabatic_overlap_le_le(&self, i: &ReducedLE, j: &ReducedLE) -> f64 {
        // check if the PAIR of the two monomers is an ESD pair
        let type_pair: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.monomer_index);

        if type_pair == PairType::ESD {
            // the overlap between the diabatic states is zero
            0.0
        }
        // check if LE states are on the same monomer
        else if i.monomer_index == j.monomer_index && i.state_index != j.state_index {
            // LE states on the same monomer are orthogonal
            // the overlap between the diabatic states is zero
            0.0
        } else if i.monomer_index == j.monomer_index && i.state_index == j.state_index {
            // LE states on the same monomer are orthogonal
            1.0
        } else {
            // get the AO overlap matrix between the different diabatic states
            // references to the monomers
            let m_i: &Monomer = &self.monomers[i.monomer_index];
            let m_j: &Monomer = &self.monomers[j.monomer_index];

            // slice the overlap matrix of the supersystem
            let s_total: ArrayView2<f64> = self.properties.s().unwrap();
            // let s_ao: ArrayView2<f64> = s_total.slice(s![m_i.slice.orb, m_j.slice.orb]);

            // // transform the AO overlap matrix to the MO basis
            let orbs_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let orbs_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
            // let s_mo: Array2<f64> = orbs_i.t().dot(&s_ao.dot(&orbs_j));

            let n_orb: usize = m_i.n_orbs + m_j.n_orbs;
            let mut s_mo: Array2<f64> = Array2::zeros([n_orb, n_orb]);
            s_mo.slice_mut(s![..m_i.n_orbs, ..m_i.n_orbs]).assign(
                &orbs_i
                    .t()
                    .dot(&s_total.slice(s![m_i.slice.orb, m_i.slice.orb]).dot(&orbs_i)),
            );
            s_mo.slice_mut(s![..m_i.n_orbs, m_i.n_orbs..]).assign(
                &orbs_i
                    .t()
                    .dot(&s_total.slice(s![m_i.slice.orb, m_j.slice.orb]).dot(&orbs_j)),
            );
            s_mo.slice_mut(s![m_i.n_orbs.., m_i.n_orbs..]).assign(
                &orbs_j
                    .t()
                    .dot(&s_total.slice(s![m_j.slice.orb, m_j.slice.orb]).dot(&orbs_j)),
            );
            s_mo.slice_mut(s![m_i.n_orbs.., ..m_i.n_orbs]).assign(
                &orbs_j
                    .t()
                    .dot(&s_total.slice(s![m_j.slice.orb, m_i.slice.orb]).dot(&orbs_i)),
            );

            // get the CIS coefficients of both LE states
            let cis_i: ArrayView2<f64> = m_i.properties.ci_coefficients().unwrap();
            let cis_j: ArrayView2<f64> = m_j.properties.ci_coefficients().unwrap();

            // reshape the CIS coefficients to 3d arrays
            let nstates: usize = self.config.lcmo.n_le;
            let nocc_i: usize = m_i.properties.n_occ().unwrap();
            let nvirt_i: usize = m_i.properties.n_virt().unwrap();
            let cis_i_3d: Array3<f64> = cis_i
                .into_shape([nocc_i, nvirt_i, nstates])
                .unwrap()
                .permuted_axes([2, 0, 1])
                .as_standard_layout()
                .to_owned();
            let cis_i_2d: ArrayView2<f64> = cis_i_3d.slice(s![i.state_index, .., ..]);

            let nocc_j: usize = m_j.properties.n_occ().unwrap();
            let nvirt_j: usize = m_j.properties.n_virt().unwrap();
            let cis_j_3d: Array3<f64> = cis_j
                .into_shape([nocc_j, nvirt_j, nstates])
                .unwrap()
                .permuted_axes([2, 0, 1])
                .as_standard_layout()
                .to_owned();
            let cis_j_2d: ArrayView2<f64> = cis_j_3d.slice(s![j.state_index, .., ..]);

            // get occupied part of MO overlap matrix
            let mut s_mo_occ: Array2<f64> = Array2::zeros([nocc_i + nocc_j, nocc_i + nocc_j]);
            s_mo_occ
                .slice_mut(s![..nocc_i, ..nocc_i])
                .assign(&s_mo.slice(s![..nocc_i, ..nocc_i]));
            s_mo_occ
                .slice_mut(s![..nocc_i, nocc_i..])
                .assign(&s_mo.slice(s![..nocc_i, m_i.n_orbs..m_i.n_orbs + nocc_j]));
            s_mo_occ
                .slice_mut(s![nocc_i.., ..nocc_i])
                .assign(&s_mo.slice(s![m_i.n_orbs..m_i.n_orbs + nocc_j, ..nocc_i]));
            s_mo_occ
                .slice_mut(s![nocc_i.., nocc_i..])
                .assign(&s_mo.slice(s![
                    m_i.n_orbs..m_i.n_orbs + nocc_j,
                    m_i.n_orbs..m_i.n_orbs + nocc_j
                ]));

            // call the CI_overlap routine
            let s_ci: f64 = diabtic_ci_overlap(s_mo.view(), s_mo_occ.view(), cis_i_2d, cis_j_2d);
            s_ci
        }
    }

    fn diabatic_overlap_le_ct(&self, i: &ReducedLE, j: &ReducedCT) -> f64 {
        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.hole.m_index);
        // The same for I and K
        let type_ik: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.electron.m_index);

        if type_ij == PairType::ESD && type_ik == PairType::ESD {
            // the overlap between the LE state and both monomers of the CT state is zero
            // thus, the coupling is zero
            0.0
        }
        // check if the LE state and the CT state share the monomer
        else if i.monomer_index == j.hole.m_index || i.monomer_index == j.electron.m_index {
            // references to the monomer and the pair
            let m_i: &Monomer = &self.monomers[i.monomer_index];
            // get the index of the pair
            let pair_index: usize = self
                .properties
                .index_of_pair(j.hole.m_index, j.electron.m_index);
            // get the pair from pairs vector
            let pair: &Pair = &self.pairs[pair_index];

            if pair.i == i.monomer_index {
                let m_j: &Monomer = &self.monomers[pair.j];
                // slice the overlap matrix of the supersystem
                let s_total: ArrayView2<f64> = self.properties.s().unwrap();

                let mut s_ao: Array2<f64> = Array2::zeros([pair.n_orbs, pair.n_orbs]);
                s_ao.slice_mut(s![..m_i.n_orbs, ..m_i.n_orbs])
                    .assign(&s_total.slice(s![m_i.slice.orb, m_i.slice.orb]));
                s_ao.slice_mut(s![..m_i.n_orbs, m_i.n_orbs..])
                    .assign(&s_total.slice(s![m_i.slice.orb, m_j.slice.orb]));
                s_ao.slice_mut(s![m_i.n_orbs.., ..m_i.n_orbs])
                    .assign(&s_total.slice(s![m_j.slice.orb, m_i.slice.orb]));
                s_ao.slice_mut(s![m_i.n_orbs.., m_i.n_orbs..])
                    .assign(&s_total.slice(s![m_j.slice.orb, m_j.slice.orb]));

                // transform the AO overlap matrix to the MO basis
                // let orbs_pair: ArrayView2<f64> = pair.properties.orbs().unwrap();
                // let s_mo:Array2<f64> = orbs_pair.t().dot(&s_ao.dot(&orbs_pair));

                0.0
            } else {
                let m_j: &Monomer = &self.monomers[pair.i];

                0.0
            }
        } else {
            // get the AO overlap matrix between the different diabatic states
            // references to the monomer and the pair
            let m_i: &Monomer = &self.monomers[i.monomer_index];
            // get the index of the pair
            let pair_index: usize = self
                .properties
                .index_of_pair(j.hole.m_index, j.electron.m_index);
            // get the pair from pairs vector
            let pair_jk: &Pair = &self.pairs[pair_index];
            let m_j: &Monomer = &self.monomers[pair_jk.i];
            let m_k: &Monomer = &self.monomers[pair_jk.j];

            // slice the overlap matrix of the supersystem
            let s_total: ArrayView2<f64> = self.properties.s().unwrap();
            let mut s_ao: Array2<f64> = Array2::zeros([m_i.n_orbs+pair_jk.n_orbs, pair_jk.n_orbs+pair_jk.n_orbs]);
            // get overlap I-I, J-J and K-K
            s_ao.slice_mut(s![..m_i.n_orbs, ..m_i.n_orbs])
                .assign(&s_total.slice(s![m_i.slice.orb, m_i.slice.orb]));
            s_ao.slice_mut(s![m_i.n_orbs..m_i.n_orbs+m_j.n_orbs, m_i.n_orbs..m_i.n_orbs+m_j.n_orbs])
                .assign(&s_total.slice(s![m_j.slice.orb, m_j.slice.orb]));
            s_ao.slice_mut(s![m_i.n_orbs+m_j.n_orbs.., m_i.n_orbs+m_j.n_orbs..])
                .assign(&s_total.slice(s![m_k.slice.orb, m_k.slice.orb]));
            // overlap between I-J and I-K
            let s_ij:ArrayView2<f64> = s_total.slice(s![m_i.slice.orb, m_j.slice.orb]);
            let s_ik:ArrayView2<f64> = s_total.slice(s![m_i.slice.orb, m_k.slice.orb]);
            s_ao.slice_mut(s![..m_i.n_orbs, m_i.n_orbs..m_i.n_orbs+m_j.n_orbs])
                .assign(&s_ij);
            s_ao.slice_mut(s![m_i.n_orbs..m_i.n_orbs+m_j.n_orbs,..m_i.n_orbs])
                .assign(&s_ij.t());
            s_ao.slice_mut(s![..m_i.n_orbs, m_i.n_orbs+m_j.n_orbs..])
                .assign(&s_ik);
            s_ao.slice_mut(s![m_i.n_orbs+m_j.n_orbs..,..m_i.n_orbs])
                .assign(&s_ik.t());
            // overlap between J-K
            let s_jk:ArrayView2<f64> = s_total.slice(s![m_j.slice.orb, m_k.slice.orb]);
            s_ao.slice_mut(s![m_i.n_orbs..m_i.n_orbs+m_j.n_orbs,m_i.n_orbs+m_j.n_orbs..])
                .assign(&s_jk);
            s_ao.slice_mut(s![m_i.n_orbs+m_j.n_orbs..,m_i.n_orbs..m_i.n_orbs+m_j.n_orbs])
                .assign(&s_jk.t());

            // transform the AO overlap matrix to the MO basis
            let mut s_mo: Array2<f64> = Array2::zeros([m_i.n_orbs+pair_jk.n_orbs, pair_jk.n_orbs+pair_jk.n_orbs]);
            let orbs_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let orbs_jk: ArrayView2<f64> = pair_jk.properties.orbs().unwrap();

            s_mo.slice_mut(s![..m_i.n_orbs, ..m_i.n_orbs]).assign(&orbs_i.t().dot(&s_ao.slice(s![..m_i.n_orbs, ..m_i.n_orbs]).dot(&orbs_i)));
            s_mo.slice_mut(s![m_i.n_orbs.., m_i.n_orbs..]).assign(&orbs_jk.t().dot(&s_ao.slice(s![m_i.n_orbs.., m_i.n_orbs..]).dot(&orbs_jk)));
            s_mo.slice_mut(s![..m_i.n_orbs, m_i.n_orbs..]).assign(&orbs_i.t().dot(&s_ao.slice(s![..m_i.n_orbs, m_i.n_orbs..]).dot(&orbs_jk)));
            s_mo.slice_mut(s![m_i.n_orbs.., ..m_i.n_orbs]).assign(&orbs_i.t().dot(&s_ao.slice(s![..m_i.n_orbs, m_i.n_orbs..]).dot(&orbs_jk)).t());

            // get the CIS coefficients of the LE
            let cis_i: ArrayView2<f64> = m_i.properties.ci_coefficients().unwrap();

            // get the CIS coefficients of the CT
            let nocc: usize = pair_jk.properties.occ_indices().unwrap().len();
            let nvirt: usize = pair_jk.properties.virt_indices().unwrap().len();
            // prepare the empty cis matrix
            let mut cis_jk: Array2<f64> = Array2::zeros([nocc, nvirt]);

            // get occupied and virtuals orbitals of the monomers of the CT state
            let nocc_j: usize = m_j.properties.occ_indices().unwrap().len();
            let nocc_k: usize = m_k.properties.occ_indices().unwrap().len();
            let nvirt_j: usize = m_j.properties.virt_indices().unwrap().len();
            let nvirt_k: usize = m_k.properties.virt_indices().unwrap().len();

            // get the overlap matrices between the monomers and the dimer
            let s_i_ij: ArrayView2<f64> = pair_jk.properties.s_i_ij().unwrap();
            let s_j_ij: ArrayView2<f64> = pair_jk.properties.s_j_ij().unwrap();

            if pair_jk.i == j.hole.m_index {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_occ: ArrayView2<f64> = s_i_ij.slice(s![..nocc_j, ..nocc]);
                let s_j_ij_virt: ArrayView2<f64> = s_j_ij.slice(s![nocc_k.., nocc..]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_j, nvirt_k]);
                ct_coefficients[[nocc_j - 1 - j.hole.ct_index, j.electron.ct_index]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_jk = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
            } else {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_virt: ArrayView2<f64> = s_i_ij.slice(s![nocc_j.., nocc..]);
                let s_j_ij_occ: ArrayView2<f64> = s_j_ij.slice(s![..nocc_k, ..nocc]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_k, nvirt_j]);
                ct_coefficients[[nocc_k - 1 - j.hole.ct_index, j.electron.ct_index]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_jk = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
            }

            // reshape the CIS coefficients to 3d arrays
            let nstates: usize = self.config.lcmo.n_le;
            let nocc_i: usize = m_i.properties.n_occ().unwrap();
            let nvirt_i: usize = m_i.properties.n_virt().unwrap();
            let cis_i_3d: Array3<f64> = cis_i
                .into_shape([nocc_i, nvirt_i, nstates])
                .unwrap()
                .permuted_axes([2, 0, 1])
                .as_standard_layout()
                .to_owned();
            let cis_i_2d: ArrayView2<f64> = cis_i_3d.slice(s![i.state_index, .., ..,]);

            // slice the MO overlap matrix
            let mut s_mo_occ:Array2<f64> = Array2::zeros([nocc_i+nocc,nocc_i+nocc]);
            s_mo_occ.slice_mut(s![..nocc_i,..nocc_i]).assign(&s_mo.slice(s![..nocc_i,..nocc_i]));
            s_mo_occ.slice_mut(s![nocc_i..,nocc_i..]).assign(&s_mo.slice(s![m_i.n_orbs..m_i.n_orbs+nocc,m_i.n_orbs..m_i.n_orbs+nocc]));
            s_mo_occ.slice_mut(s![..nocc_i,nocc_i..]).assign(&s_mo.slice(s![..nocc_i,m_i.n_orbs..m_i.n_orbs+nocc]));
            s_mo_occ.slice_mut(s![nocc_i..,..nocc_i]).assign(&s_mo.slice(s![m_i.n_orbs..m_i.n_orbs+nocc,..nocc_i]));

            // call the CI_overlap routine
            let s_ci: f64 =
                diabtic_ci_overlap(s_mo.view(), s_mo_occ.view(), cis_i_2d, cis_jk.view());
            s_ci
        }
    }

    fn diabatic_overlap_ct_ct(&self, state_i: &ReducedCT, state_j: &ReducedCT) -> f64 {
        let (i, j): (&ReducedParticle, &ReducedParticle) = (&state_i.hole, &state_i.electron);
        let (k, l): (&ReducedParticle, &ReducedParticle) = (&state_j.hole, &state_j.electron);

        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.m_index, j.m_index);
        // I and K
        let type_ik: PairType = self.properties.type_of_pair(i.m_index, k.m_index);
        // I and L
        let type_il: PairType = self.properties.type_of_pair(i.m_index, l.m_index);
        // J and K
        let type_jk: PairType = self.properties.type_of_pair(j.m_index, k.m_index);
        // J and L
        let type_jl: PairType = self.properties.type_of_pair(j.m_index, l.m_index);
        // K and L
        let type_kl: PairType = self.properties.type_of_pair(k.m_index, l.m_index);

        if type_ik == PairType::ESD
            && type_il == PairType::ESD
            && type_jk == PairType::ESD
            && type_jl == PairType::ESD
        {
            // if the distance between all monomers of the two CT states is greater than the ESD
            // limit, the coupling between the CT states is zero
            0.0
        } else if type_ij == PairType::Pair && type_kl == PairType::Pair {
            // get the indices of the pairs
            let pair_index_ij: usize = self
                .properties
                .index_of_pair(state_i.hole.m_index, state_i.electron.m_index);
            let pair_index_kl: usize = self
                .properties
                .index_of_pair(state_j.hole.m_index, state_j.electron.m_index);

            // get reference to the pairs
            let pair_ij: &Pair = &self.pairs[pair_index_ij];
            let pair_kl: &Pair = &self.pairs[pair_index_kl];

            // get the overlap matrix of the Supersystem
            let s_total: ArrayView2<f64> = self.properties.s().unwrap();

            // Check if both CT are localized on the same pair
            if pair_ij.i == pair_kl.i && pair_ij.j == pair_kl.j {
                // get the references to the monomers
                let m_a: &Monomer = &self.monomers[pair_ij.i];
                let m_b: &Monomer = &self.monomers[pair_ij.j];

                // prepare the AO overlap matrix
                let mut s_ao: Array2<f64> = Array2::zeros([pair_ij.n_orbs, pair_kl.n_orbs]);
                // fill the AO overlap matrix
                s_ao.slice_mut(s![..m_a.n_orbs, ..m_a.n_orbs])
                    .assign(&s_total.slice(s![m_a.slice.orb, m_a.slice.orb]));
                s_ao.slice_mut(s![m_a.n_orbs.., m_a.n_orbs..])
                    .assign(&s_total.slice(s![m_b.slice.orb, m_b.slice.orb]));
                s_ao.slice_mut(s![..m_a.n_orbs, m_a.n_orbs..])
                    .assign(&s_total.slice(s![m_a.slice.orb, m_b.slice.orb]));
                s_ao.slice_mut(s![m_a.n_orbs.., ..m_a.n_orbs])
                    .assign(&s_total.slice(s![m_b.slice.orb, m_a.slice.orb]));

                // get views of the MO coefficients for the pairs
                let orbs: ArrayView2<f64> = pair_ij.properties.orbs().unwrap();
                // transform the AO overlap matrix into the MO basis
                let s_mo: Array2<f64> = orbs.t().dot(&s_ao.dot(&orbs));

                // get the CI coefficients of the CT state I
                let nocc: usize = pair_ij.properties.occ_indices().unwrap().len();
                let nvirt: usize = pair_ij.properties.virt_indices().unwrap().len();
                // prepare the empty cis matrix
                let mut cis_1: Array2<f64> = Array2::zeros([nocc, nvirt]);
                let mut cis_2: Array2<f64> = Array2::zeros([nocc, nvirt]);

                // get occupied and virtuals orbitals of the monomers of the CT state
                let nocc_a: usize = m_a.properties.occ_indices().unwrap().len();
                let nocc_b: usize = m_b.properties.occ_indices().unwrap().len();
                let nvirt_a: usize = m_a.properties.virt_indices().unwrap().len();
                let nvirt_b: usize = m_b.properties.virt_indices().unwrap().len();

                // get the overlap matrices between the monomers and the dimer
                let s_i_ij: ArrayView2<f64> = pair_ij.properties.s_i_ij().unwrap();
                let s_j_ij: ArrayView2<f64> = pair_ij.properties.s_j_ij().unwrap();

                if pair_ij.i == state_i.hole.m_index {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_occ: ArrayView2<f64> = s_i_ij.slice(s![..nocc_a, ..nocc]);
                    let s_j_ij_virt: ArrayView2<f64> = s_j_ij.slice(s![nocc_b.., nocc..]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                    ct_coefficients[[
                        nocc_a - 1 - state_i.hole.ct_index,
                        state_i.electron.ct_index,
                    ]] = 1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis_1 = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
                } else {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_virt: ArrayView2<f64> = s_i_ij.slice(s![nocc_a.., nocc..]);
                    let s_j_ij_occ: ArrayView2<f64> = s_j_ij.slice(s![..nocc_b, ..nocc]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                    ct_coefficients[[
                        nocc_b - 1 - state_i.hole.ct_index,
                        state_i.electron.ct_index,
                    ]] = 1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis_1 = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
                }
                if pair_ij.i == state_j.hole.m_index {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_occ: ArrayView2<f64> = s_i_ij.slice(s![..nocc_a, ..nocc]);
                    let s_j_ij_virt: ArrayView2<f64> = s_j_ij.slice(s![nocc_b.., nocc..]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                    ct_coefficients[[
                        nocc_a - 1 - state_j.hole.ct_index,
                        state_j.electron.ct_index,
                    ]] = 1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis_2 = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
                } else {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_virt: ArrayView2<f64> = s_i_ij.slice(s![nocc_a.., nocc..]);
                    let s_j_ij_occ: ArrayView2<f64> = s_j_ij.slice(s![..nocc_b, ..nocc]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                    ct_coefficients[[
                        nocc_b - 1 - state_j.hole.ct_index,
                        state_j.electron.ct_index,
                    ]] = 1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis_2 = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
                }

                // get occupied MO overlap matrix
                let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

                // call the CI_overlap routine
                let s_ci: f64 = diabtic_ci_overlap_intra_monomer(
                    s_mo.view(),
                    s_mo_occ,
                    cis_1.view(),
                    cis_2.view(),
                );
                s_ci
            }
            // Check if the CT states share one monomer
            else if (pair_ij.i == pair_kl.i || pair_ij.i == pair_kl.j)
                || (pair_ij.j == pair_kl.i || pair_ij.j == pair_kl.j)
            {
                0.0
            } else {
                // get the references to the monomers
                let m_a: &Monomer = &self.monomers[pair_ij.i];
                let m_b: &Monomer = &self.monomers[pair_ij.j];
                let m_c: &Monomer = &self.monomers[pair_kl.i];
                let m_d: &Monomer = &self.monomers[pair_kl.j];

                // prepare the AO overlap matrix
                let mut s_ao: Array2<f64> = Array2::zeros([pair_ij.n_orbs, pair_kl.n_orbs]);
                // fill the AO overlap matrix
                s_ao.slice_mut(s![..m_a.n_orbs, ..m_c.n_orbs])
                    .assign(&s_total.slice(s![m_a.slice.orb, m_c.slice.orb]));
                s_ao.slice_mut(s![..m_a.n_orbs, m_c.n_orbs..])
                    .assign(&s_total.slice(s![m_a.slice.orb, m_d.slice.orb]));
                s_ao.slice_mut(s![m_a.n_orbs.., ..m_c.n_orbs])
                    .assign(&s_total.slice(s![m_b.slice.orb, m_c.slice.orb]));
                s_ao.slice_mut(s![m_a.n_orbs.., m_c.n_orbs..])
                    .assign(&s_total.slice(s![m_b.slice.orb, m_d.slice.orb]));

                // get views of the MO coefficients for the pairs
                let orbs_ij: ArrayView2<f64> = pair_ij.properties.orbs().unwrap();
                let orbs_kl: ArrayView2<f64> = pair_kl.properties.orbs().unwrap();
                // transform the AO overlap matrix into the MO basis
                let s_mo: Array2<f64> = orbs_ij.t().dot(&s_ao.dot(&orbs_kl));

                // get the CI coefficients of the CT state I
                let nocc_1: usize = pair_ij.properties.occ_indices().unwrap().len();
                let nvirt_1: usize = pair_ij.properties.virt_indices().unwrap().len();
                // prepare the empty cis matrix
                let mut cis_ij: Array2<f64> = Array2::zeros([nocc_1, nvirt_1]);

                // get occupied and virtuals orbitals of the monomers of the CT state
                let nocc_a: usize = m_a.properties.occ_indices().unwrap().len();
                let nocc_b: usize = m_b.properties.occ_indices().unwrap().len();
                let nvirt_a: usize = m_a.properties.virt_indices().unwrap().len();
                let nvirt_b: usize = m_b.properties.virt_indices().unwrap().len();

                // get the overlap matrices between the monomers and the dimer
                let s_i_ij: ArrayView2<f64> = pair_ij.properties.s_i_ij().unwrap();
                let s_j_ij: ArrayView2<f64> = pair_ij.properties.s_j_ij().unwrap();

                if pair_ij.i == state_i.hole.m_index {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_occ: ArrayView2<f64> = s_i_ij.slice(s![..nocc_a, ..nocc_1]);
                    let s_j_ij_virt: ArrayView2<f64> = s_j_ij.slice(s![nocc_b.., nocc_1..]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                    ct_coefficients[[
                        nocc_a - 1 - state_i.hole.ct_index,
                        state_i.electron.ct_index,
                    ]] = 1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis_ij = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
                } else {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_virt: ArrayView2<f64> = s_i_ij.slice(s![nocc_a.., nocc_1..]);
                    let s_j_ij_occ: ArrayView2<f64> = s_j_ij.slice(s![..nocc_b, ..nocc_1]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                    ct_coefficients[[
                        nocc_b - 1 - state_i.hole.ct_index,
                        state_i.electron.ct_index,
                    ]] = 1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis_ij = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
                }

                // get the CI coefficients of the CT state I
                let nocc_2: usize = pair_ij.properties.occ_indices().unwrap().len();
                let nvirt_2: usize = pair_ij.properties.virt_indices().unwrap().len();
                // prepare the empty cis matrix
                let mut cis_kl: Array2<f64> = Array2::zeros([nocc_2, nvirt_2]);

                // get occupied and virtuals orbitals of the monomers of the CT state
                let nocc_c: usize = m_c.properties.occ_indices().unwrap().len();
                let nocc_d: usize = m_d.properties.occ_indices().unwrap().len();
                let nvirt_c: usize = m_c.properties.virt_indices().unwrap().len();
                let nvirt_d: usize = m_d.properties.virt_indices().unwrap().len();

                // get the overlap matrices between the monomers and the dimer
                let s_i_ij: ArrayView2<f64> = pair_kl.properties.s_i_ij().unwrap();
                let s_j_ij: ArrayView2<f64> = pair_kl.properties.s_j_ij().unwrap();

                if pair_kl.i == state_j.hole.m_index {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_occ: ArrayView2<f64> = s_i_ij.slice(s![..nocc_c, ..nocc_2]);
                    let s_j_ij_virt: ArrayView2<f64> = s_j_ij.slice(s![nocc_d.., nocc_2..]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_c, nvirt_d]);
                    ct_coefficients[[
                        nocc_a - 1 - state_j.hole.ct_index,
                        state_j.electron.ct_index,
                    ]] = 1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis_kl = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
                } else {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_virt: ArrayView2<f64> = s_i_ij.slice(s![nocc_c.., nocc_2..]);
                    let s_j_ij_occ: ArrayView2<f64> = s_j_ij.slice(s![..nocc_d, ..nocc_2]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_d, nvirt_c]);
                    ct_coefficients[[
                        nocc_d - 1 - state_j.hole.ct_index,
                        state_j.electron.ct_index,
                    ]] = 1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis_kl = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
                }

                // call the CI_overlap routine
                // let s_ci: f64 = diabtic_ci_overlap(
                //     s_mo.view(),
                //     s_occ,
                //     cis_ij.view(),
                //     cis_kl.view(),
                // );
                // s_ci
                0.0
            }
        }
        // Placeholder
        else {
            0.0
        }
    }
}

fn diabtic_ci_overlap_intra_monomer(
    s_mo: ArrayView2<f64>,
    s_mo_occ: ArrayView2<f64>,
    cis_i: ArrayView2<f64>,
    cis_j: ArrayView2<f64>,
) -> f64 {
    // Compute the overlap between diabatic CI wavefunctions
    // Excitations i->a with coefficients |C_ia| < threshold will be neglected
    let threshold: f64 = 0.5;

    // get n_occ and n_virt from the shapes of the CIS coefficients
    let nocc: usize = cis_i.dim().0;
    let nvirt: usize = cis_i.dim().1;
    let norb: usize = nocc + nvirt;

    // change s_full and s_occ using s_mo
    let s_ij: Array2<f64> = s_mo_occ.to_owned();
    let det_ij: f64 = s_ij.det().unwrap();

    // scalar coupling array
    let mut s_ci: f64 = 0.0;

    // calculate the overlap between the excited states
    // iterate over the CI coefficients of the diabatic state J
    for i in 0..nocc {
        for (a_idx, a) in (nocc..norb).into_iter().enumerate() {
            // slice the CI coefficients of the diabatic state J at the indicies i and a
            let coeff_i = cis_i[[i, a_idx]];

            // if the value of the coefficient is smaller than the threshold,
            // exclude the excited state
            if coeff_i.abs() > threshold {
                let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                // occupied orbitals in the configuration state function |Psi_ia>
                // overlap <1,...,a,...|1,...,j,...>
                s_aj.slice_mut(s![i, ..]).assign(&s_mo.slice(s![a, ..nocc]));
                let det_aj: f64 = s_aj.det().unwrap();

                // iterate over the CI coefficients of the diabatic state I
                for j in 0..nocc {
                    for (b_idx, b) in (nocc..norb).into_iter().enumerate() {
                        // slice the CI coefficients of the diabtic state I at the indicies j and b
                        let coeff_j = cis_j[[j, b_idx]];

                        if coeff_j.abs() > threshold {
                            let mut s_ab: Array2<f64> = s_mo_occ.to_owned();
                            // select part of overlap matrix for orbitals
                            // in |Psi_ia> and |Psi_jb>
                            // <1,...,a,...|1,...,b,...>
                            s_ab.slice_mut(s![i, ..]).assign(&s_mo.slice(s![a, ..nocc]));
                            s_ab.slice_mut(s![.., j]).assign(&s_mo.slice(s![..nocc, b]));
                            s_ab[[i, j]] = s_mo[[a, b]];
                            let det_ab: f64 = s_ab.det().unwrap();

                            let mut s_ib: Array2<f64> = s_mo_occ.to_owned();
                            // <1,...,i,...|1,...,b,...>
                            s_ib.slice_mut(s![.., j]).assign(&s_mo.slice(s![..nocc, b]));
                            let det_ib: f64 = s_ib.det().unwrap();

                            let cc: f64 = coeff_j * coeff_i;
                            let det_sum: f64 = (det_ab * det_ij + det_aj * det_ib);
                            // see eqn. (9.39) in A. Humeniuk, PhD thesis (2018)
                            s_ci += cc * det_sum;
                        }
                    }
                }
            }
        }
    }
    return s_ci;
}

fn diabtic_ci_overlap(
    s_mo: ArrayView2<f64>,
    s_mo_occ: ArrayView2<f64>,
    cis_i: ArrayView2<f64>,
    cis_j: ArrayView2<f64>,
) -> f64 {
    // Compute the overlap between diabatic CI wavefunctions
    // Excitations i->a with coefficients |C_ia| < threshold will be neglected
    let threshold: f64 = 0.001;

    // get n_occ and n_virt from the shapes of the CIS coefficients
    let nocc_i: usize = cis_i.dim().0;
    let nvirt_i: usize = cis_i.dim().1;
    let norb_i: usize = nocc_i + nvirt_i;

    let nocc_j: usize = cis_j.dim().0;
    let nvirt_j: usize = cis_j.dim().1;
    let norb_j: usize = nocc_j + nvirt_j;

    // change s_full and s_occ using s_mo
    let s_ij: Array2<f64> = s_mo_occ.to_owned();
    let det_ij: f64 = s_ij.det().unwrap();

    // // get s_mo_occ
    // let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc_i, ..nocc_j]);

    // scalar coupling array
    let mut s_ci_lele: f64 = 0.0;
    let mut s_ci_gs_i_le_j: f64 = 0.0;
    let mut s_ci_le_i_gs_j: f64 = 0.0;
    let s_ci_gsgs: f64 = det_ij;

    // calculate the overlap between the excited states
    // iterate over the CI coefficients of the diabatic state J
    for i in 0..nocc_i {
        for (a_idx, a) in (nocc_i..norb_i).into_iter().enumerate() {
            // slice the CI coefficients of the diabatic state J at the indicies i and a
            let coeff_i = cis_i[[i, a_idx]];

            // if the value of the coefficient is smaller than the threshold,
            // exclude the excited state
            if coeff_i.abs() > threshold {
                let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                // occupied orbitals in the configuration state function |Psi_ia>
                // overlap <1,...,a,...|1,...,j,...>
                s_aj.slice_mut(s![i, ..nocc_i])
                    .assign(&s_mo.slice(s![a, ..nocc_i]));
                s_aj.slice_mut(s![i, nocc_i..])
                    .assign(&s_mo.slice(s![a, norb_i..norb_i + nocc_j]));
                let det_aj: f64 = s_aj.det().unwrap();

                // get the determinant between the LE of I and the ground state of J
                s_ci_le_i_gs_j += coeff_i * 2.0_f64.sqrt() * det_ij * det_aj;

                // iterate over the CI coefficients of the diabatic state I
                for (j_idx, j) in (nocc_i..nocc_i + nocc_j).into_iter().enumerate() {
                    for (b_idx, b) in (norb_i + nocc_j..norb_i + norb_j).into_iter().enumerate() {
                        // slice the CI coefficients of the diabtic state I at the indicies j and b
                        let coeff_j = cis_j[[j_idx, b_idx]];

                        if coeff_j.abs() > threshold {
                            let mut s_ab: Array2<f64> = s_mo_occ.to_owned();
                            // select part of overlap matrix for orbitals
                            // in |Psi_ia> and |Psi_jb>
                            // <1,...,a,...|1,...,b,...>
                            s_ab.slice_mut(s![i, ..nocc_i])
                                .assign(&s_mo.slice(s![a, ..nocc_i]));
                            s_ab.slice_mut(s![i, nocc_i..])
                                .assign(&s_mo.slice(s![a, norb_i..norb_i + nocc_j]));
                            s_ab.slice_mut(s![..nocc_i, j])
                                .assign(&s_mo.slice(s![..nocc_i, b]));
                            s_ab.slice_mut(s![nocc_i.., j])
                                .assign(&s_mo.slice(s![norb_i..norb_i + nocc_j, b]));
                            s_ab[[i, j]] = s_mo[[a, b]];
                            let det_ab: f64 = s_ab.det().unwrap();

                            let mut s_ib: Array2<f64> = s_mo_occ.to_owned();
                            // <1,...,i,...|1,...,b,...>
                            s_ib.slice_mut(s![..nocc_i, j])
                                .assign(&s_mo.slice(s![..nocc_i, b]));
                            s_ib.slice_mut(s![nocc_i.., j])
                                .assign(&s_mo.slice(s![norb_i..norb_i + nocc_j, b]));
                            let det_ib: f64 = s_ib.det().unwrap();

                            let cc: f64 = coeff_j * coeff_i;
                            let det_sum: f64 = (det_ab * det_ij + det_aj * det_ib);
                            // see eqn. (9.39) in A. Humeniuk, PhD thesis (2018)
                            s_ci_lele += cc * det_sum;
                        }
                    }
                }
            }
        }
    }

    for (j_idx, j) in (nocc_i..nocc_i + nocc_j).into_iter().enumerate() {
        for (b_idx, b) in (norb_i + nocc_j..norb_i + norb_j).into_iter().enumerate() {
            // slice the CI coefficients of the diabtic state I at the indicies j and b
            let coeff_j = cis_j[[j_idx, b_idx]];

            if coeff_j.abs() > threshold {
                let mut s_ib: Array2<f64> = s_mo_occ.to_owned();
                // <1,...,i,...|1,...,b,...>
                s_ib.slice_mut(s![..nocc_i, j])
                    .assign(&s_mo.slice(s![..nocc_i, b]));
                s_ib.slice_mut(s![nocc_i.., j])
                    .assign(&s_mo.slice(s![norb_i..norb_i + nocc_j, b]));
                let det_ib: f64 = s_ib.det().unwrap();

                // get the determinant between the LE of J and the ground state of I
                s_ci_gs_i_le_j += coeff_j * 2.0_f64.sqrt() * det_ij * det_ib;
            }
        }
    }

    // build the product between all overlap values
    let s_ci: f64 = s_ci_gs_i_le_j * s_ci_le_i_gs_j * s_ci_lele * s_ci_gsgs;
    println!(
        "ovlp gsgs {}, ovlp gs_le {}, ovlp le_gs {}, ovlp le_le {}",
        s_ci_gsgs, s_ci_gs_i_le_j, s_ci_le_i_gs_j, s_ci_lele
    );

    return s_ci;
}
