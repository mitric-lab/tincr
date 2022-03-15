use crate::initialization::{Atom,System};
use ndarray::prelude::*;
use ndarray_linalg::Determinant;
use ndarray_stats::QuantileExt;
use phf::Slice;
use crate::fmo::basis::LocallyExcited;
use crate::fmo::{SuperSystem, Monomer, PairType, Pair};
use crate::fmo::lcmo::cis_gradient::{ReducedBasisState, ReducedCT, ReducedLE, ReducedParticle};

impl SuperSystem{
    pub fn diabatic_overlap(&self,lhs: &ReducedBasisState, rhs: &ReducedBasisState,)->Array2<f64>{
        match (lhs, rhs) {
            // Overlap between two LE states.
            (ReducedBasisState::LE(ref a), ReducedBasisState::LE(ref b)) => {
                self.diabatic_overlap_le_le(a, b)
            }
            // Overlap between LE and CT state.
            (ReducedBasisState::LE(ref a), ReducedBasisState::CT(ref b)) => {
                self.diabatic_overlap_le_ct(a, b)
            }
            // Overlap between CT and LE state.
            (ReducedBasisState::CT(ref a), ReducedBasisState::LE(ref b)) => {
                self.diabatic_overlap_le_ct(b, a)
            }
            // Overlap between CT and CT
            (ReducedBasisState::CT(ref a), ReducedBasisState::CT(ref b)) => {
                self.diabatic_overlap_ct_ct(a,b)
            }
        }
    }

    pub fn diabatic_overlap_le_le(&self,i: &ReducedLE, j: &ReducedLE)->Array2<f64>{
        // check if the PAIR of the two monomers is an ESD pair
        let type_pair: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.monomer_index);

        if type_pair == PairType::ESD{
            // the overlap between the diabatic states is zero
            Array2::zeros((1,1))
        }
        // check if LE states are on the same monomer
        else if i.monomer_index == j.monomer_index{
            // LE states on the same monomer are orthogonal
            // the overlap between the diabatic states is zero
            Array2::zeros((1,1))
        }
        else{
            // get the AO overlap matrix between the different diabatic states
            // references to the monomers
            let m_i:&Monomer = &self.monomers[i.monomer_index];
            let m_j:&Monomer = &self.monomers[j.monomer_index];

            // slice the overlap matrix of the supersystem
            let s_total:ArrayView2<f64> = self.properties.s().unwrap();
            let s_ao:ArrayView2<f64> = s_total.slice(s![m_i.slice.orb,m_j.slice.orb]);

            // transform the AO overlap matrix to the MO basis
            let orbs_i:ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let orbs_j:ArrayView2<f64> = m_j.properties.orbs().unwrap();
            let s_mo:Array2<f64> = orbs_i.t().dot(&s_ao.dot(&orbs_j));

            // get the CIS coefficients of both LE states
            let cis_i:ArrayView2<f64> = m_i.properties.ci_coefficients().unwrap();
            let cis_j:ArrayView2<f64> = m_j.properties.ci_coefficients().unwrap();

            // call the CI_overlap routine
        }

    }

    pub fn diabatic_overlap_le_ct(&self,i: &ReducedLE, j: &ReducedCT)->Array2<f64>{
        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.hole.m_index);
        // The same for I and K
        let type_ik: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.electron.m_index);

        if type_ij == PairType::ESD && type_ik == PairType::ESD{
            // the overlap between the LE state and both monomers of the CT state is zero
            // thus, the coupling is zero
            Array2::zeros((1,1))
        }
        else{
            // get the AO overlap matrix between the different diabatic states
            // references to the monomer and the pair
            let m_i:&Monomer = &self.monomers[i.monomer_index];
            // get the index of the pair
            let pair_index: usize = self
                .properties
                .index_of_pair(j.hole.m_index, j.electron.m_index);
            // get the pair from pairs vector
            let pair_jk:&Pair = &self.pairs[pair_index];
            let m_j:&Monomer = &self.monomers[pair_jk.i];
            let m_k:&Monomer = &self.monomers[pair_jk.j];

            // slice the overlap matrix of the supersystem
            let s_total:ArrayView2<f64> = self.properties.s().unwrap();
            let mut s_ao:Array2<f64> = Array2::zeros([m_i.n_orbs,pair_jk.n_orbs]);
            s_ao.slice_mut(s![..,..m_j.n_orbs]).assign(&s_total.slice(s![m_i.slice.orb,m_j.slice.orb]));
            s_ao.slice_mut(s![..,m_j.n_orbs..]).assign(&s_total.slice(s![m_i.slice.orb,m_k.slice.orb]));

            // transform the AO overlap matrix to the MO basis
            let orbs_i:ArrayView2<f64> = m_i.properties.orbs().unwrap();
            let orbs_jk:ArrayView2<f64> = pair_jk.properties.orbs().unwrap();
            let s_mo:Array2<f64> = orbs_i.t().dot(&s_ao.dot(&orbs_jk));

            // get the CIS coefficients of the LE
            let cis_i:ArrayView2<f64> = m_i.properties.ci_coefficients().unwrap();

            // get the CIS coefficients of the CT
            let nocc:usize = pair_jk.properties.n_occ().unwrap();
            let nvirt:usize = pair_jk.properties.n_virt().unwrap();
            // prepare the empty cis matrix
            let mut cis_jk:Array2<f64> = Array2::zeros([nocc,nvirt]);

            // get occupied and virtuals orbitals of the monomers of the CT state
            let nocc_j:usize = m_j.properties.occ_indices().unwrap().len();
            let nocc_k:usize = m_k.properties.occ_indices().unwrap().len();
            let nvirt_j:usize = m_j.properties.virt_indices().unwrap().len();
            let nvirt_k:usize = m_k.properties.virt_indices().unwrap().len();

            // get the overlap matrices between the monomers and the dimer
            let s_i_ij:ArrayView2<f64> = pair_jk.properties.s_i_ij().unwrap();
            let s_j_ij:ArrayView2<f64> = pair_jk.properties.s_j_ij().unwrap();

            if pair_jk.i == j.hole.m_index{
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_occ:ArrayView2<f64> = s_i_ij.slice(s![..nocc_j,..nocc]);
                let s_j_ij_virt:ArrayView2<f64> = s_j_ij.slice(s![nocc_k..,nocc..]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients:Array2<f64> = Array2::zeros([nocc_j,nvirt_k]);
                ct_coefficients[[nocc_j-1-j.hole.ct_index,j.electron.ct_index]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_jk = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
            }
            else{
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_virt:ArrayView2<f64> = s_i_ij.slice(s![nocc_j..,nocc..]);
                let s_j_ij_occ:ArrayView2<f64> = s_j_ij.slice(s![..nocc_k,..nocc]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients:Array2<f64> = Array2::zeros([nocc_k,nvirt_j]);
                ct_coefficients[[nocc_k-1-j.hole.ct_index,j.electron.ct_index]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_jk = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
            }

            // call the CI_overlap routine
        }
    }

    pub fn diabatic_overlap_ct_ct(&self,state_i: &ReducedCT, state_j: &ReducedCT)->Array2<f64>{
        let (i, j):(&ReducedParticle,&ReducedParticle) = (&state_i.hole,&state_i.electron);
        let (k, l):(&ReducedParticle,&ReducedParticle) = (&state_j.hole,&state_j.electron);

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

        if type_ik == PairType::ESD && type_il == PairType::ESD && type_jk == PairType::ESD && type_jl == PairType::ESD{
            // if the distance between all monomers of the two CT states is greater than the ESD
            // limit, the coupling between the CT states is zero
            Array2::zeros((1,1))
        }
        else{
            // get the indices of the pairs
            let pair_index_ij: usize = self
                .properties
                .index_of_pair(state_i.hole.m_index, state_i.electron.m_index);
            let pair_index_kl: usize = self
                .properties
                .index_of_pair(state_j.hole.m_index, state_j.electron.m_index);

            // get reference to the pairs
            let pair_ij:&Pair = &self.pairs[pair_index_ij];
            let pair_kl:&Pair = &self.pairs[pair_index_kl];

            // get the references to the monomers
            let m_a:&Monomer = &self.monomers[pair_ij.i];
            let m_b:&Monomer = &self.monomers[pair_ij.j];
            let m_c:&Monomer = &self.monomers[pair_kl.i];
            let m_d:&Monomer = &self.monomers[pair_kl.j];

            // prepare the AO overlap matrix
            let mut s_ao:Array2<f64> = Array2::zeros([pair_ij.n_orbs,pair_kl.n_orbs]);
            // get the overlap matrix of the Supersystem
            let s_total:ArrayView2<f64> = self.properties.s().unwrap();

            // fill the AO overlap matrix
            s_ao.slice_mut(s![..m_a.n_orbs,..m_c.n_orbs]).assign(&s_total.slice(s![m_a.slice.orb,m_c.slice.orb]));
            s_ao.slice_mut(s![..m_a.n_orbs..,m_c.n_orbs..]).assign(&s_total.slice(s![m_a.slice.orb,m_d.slice.orb]));
            s_ao.slice_mut(s![m_a.n_orbs,..m_c.n_orbs]).assign(&s_total.slice(s![m_b.slice.orb,m_c.slice.orb]));
            s_ao.slice_mut(s![m_a.n_orbs,m_c.n_orbs..]).assign(&s_total.slice(s![m_b.slice.orb,m_d.slice.orb]));

            // get views of the MO coefficients for the pairs
            let orbs_ij:ArrayView2<f64> = pair_ij.properties.orbs().unwrap();
            let orbs_kl:ArrayView2<f64> = pair_kl.properties.orbs().unwrap();
            // transform the AO overlap matrix into the MO basis
            let s_mo:Array2<f64> = orbs_ij.t().dot(&s_ao.dot(&orbs_kl));

            // get the CI coefficients of the CT state I
            let nocc:usize = pair_ij.properties.n_occ().unwrap();
            let nvirt:usize = pair_ij.properties.n_virt().unwrap();
            // prepare the empty cis matrix
            let mut cis_ij:Array2<f64> = Array2::zeros([nocc,nvirt]);

            // get occupied and virtuals orbitals of the monomers of the CT state
            let nocc_a:usize = m_a.properties.occ_indices().unwrap().len();
            let nocc_b:usize = m_b.properties.occ_indices().unwrap().len();
            let nvirt_a:usize = m_a.properties.virt_indices().unwrap().len();
            let nvirt_b:usize = m_b.properties.virt_indices().unwrap().len();

            // get the overlap matrices between the monomers and the dimer
            let s_i_ij:ArrayView2<f64> = pair_ij.properties.s_i_ij().unwrap();
            let s_j_ij:ArrayView2<f64> = pair_ij.properties.s_j_ij().unwrap();

            if pair_ij.i == state_i.hole.m_index{
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_occ:ArrayView2<f64> = s_i_ij.slice(s![..nocc_a,..nocc]);
                let s_j_ij_virt:ArrayView2<f64> = s_j_ij.slice(s![nocc_b..,nocc..]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients:Array2<f64> = Array2::zeros([nocc_a,nvirt_b]);
                ct_coefficients[[nocc_a-1-state_i.hole.ct_index,state_i.electron.ct_index]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_ij = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
            }
            else{
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_virt:ArrayView2<f64> = s_i_ij.slice(s![nocc_a..,nocc..]);
                let s_j_ij_occ:ArrayView2<f64> = s_j_ij.slice(s![..nocc_b,..nocc]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients:Array2<f64> = Array2::zeros([nocc_b,nvirt_a]);
                ct_coefficients[[nocc_k-1-state_i.hole.ct_index,state_i.electron.ct_index]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_ij = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
            }

            // get the CI coefficients of the CT state I
            let nocc:usize = pair_ij.properties.n_occ().unwrap();
            let nvirt:usize = pair_ij.properties.n_virt().unwrap();
            // prepare the empty cis matrix
            let mut cis_kl:Array2<f64> = Array2::zeros([nocc,nvirt]);

            // get occupied and virtuals orbitals of the monomers of the CT state
            let nocc_c:usize = m_c.properties.occ_indices().unwrap().len();
            let nocc_d:usize = m_d.properties.occ_indices().unwrap().len();
            let nvirt_c:usize = m_c.properties.virt_indices().unwrap().len();
            let nvirt_d:usize = m_d.properties.virt_indices().unwrap().len();

            // get the overlap matrices between the monomers and the dimer
            let s_i_ij:ArrayView2<f64> = pair_kl.properties.s_i_ij().unwrap();
            let s_j_ij:ArrayView2<f64> = pair_kl.properties.s_j_ij().unwrap();

            if pair_kl.i == state_j.hole.m_index{
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_occ:ArrayView2<f64> = s_i_ij.slice(s![..nocc_c,..nocc]);
                let s_j_ij_virt:ArrayView2<f64> = s_j_ij.slice(s![nocc_d..,nocc..]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients:Array2<f64> = Array2::zeros([nocc_c,nvirt_d]);
                ct_coefficients[[nocc_a-1-state_j.hole.ct_index,state_j.electron.ct_index]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_kl = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
            }
            else{
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_virt:ArrayView2<f64> = s_i_ij.slice(s![nocc_c..,nocc..]);
                let s_j_ij_occ:ArrayView2<f64> = s_j_ij.slice(s![..nocc_d,..nocc]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients:Array2<f64> = Array2::zeros([nocc_d,nvirt_c]);
                ct_coefficients[[nocc_k-1-state_j.hole.ct_index,state_j.electron.ct_index]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_kl = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
            }

            // call the CI_overlap routine
        }
    }
}

pub fn diabtic_ci_overlap(s_mo:ArrayView2<f64>,cis_i:ArrayView3<f64>,cis_j:ArrayView3<f64>){
    // Compute the overlap between diabatic CI wavefunctions
    // Excitations i->a with coefficients |C_ia| < threshold will be neglected
    let threshold:f64 = 0.01;

    // get n_occ and n_virt from the shapes of the CIS coefficients


}