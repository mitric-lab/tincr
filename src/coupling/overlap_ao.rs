use crate::initialization::{Atom,System};
use ndarray::prelude::*;
use crate::param::slako_transformations::{slako_transformation, directional_cosines};
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
            let s_i_ij:ArrayView2<f64> = self.properties.s_i_ij().unwrap();
            let s_j_ij:ArrayView2<f64> = self.properties.s_j_ij().unwrap();

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
    }
}

pub fn overlap_between_timesteps(le_i: &LocallyExcited, le_j: &LocallyExcited)->Array2<f64>{
    /// compute overlap matrix elements between two sets of atoms using
    /// Slater-Koster rules
    ///
    let atoms_old_j = le_j.monomer.properties.old_atoms().unwrap();
    let mut s:Array2<f64> = Array2::zeros([le_i.monomer.n_orbs, le_j.monomer.n_orbs]);
    let mut mu:usize = 0;
    // iterate over the atoms of the system
    for (i, atom_i) in le_i.atoms.iter().enumerate(){
        // iterate over the orbitals on atom I
        for orbi in atom_i.valorbs.iter(){
            // iterate over the atoms of the old geometry
            let mut nu:usize = 0;
            for (j, atom_j) in atoms_old_j.iter().enumerate(){
                // iterate over the orbitals on atom J
                for orbj in atom_j.valorbs.iter(){
                    if atom_i <= atom_j {
                        let (r, x, y, z): (f64, f64, f64, f64) =
                            directional_cosines(&atom_i.xyz, &atom_j.xyz);
                        s[[mu, nu]] = slako_transformation(
                            r,
                            x,
                            y,
                            z,
                            &le_i.monomer.slako.get(atom_i.kind, atom_j.kind).s_spline,
                            orbi.l,
                            orbi.m,
                            orbj.l,
                            orbj.m,
                        );
                    }
                    else{
                        let (r, x, y, z): (f64, f64, f64, f64) =
                            directional_cosines(&atom_j.xyz, &atom_i.xyz);
                        s[[mu, nu]] = slako_transformation(
                            r,
                            x,
                            y,
                            z,
                            &le_i.monomer.slako.get(atom_j.kind, atom_i.kind).s_spline,
                            orbj.l,
                            orbj.m,
                            orbi.l,
                            orbi.m,
                        );
                    }
                    nu += 1;
                }
            }
            nu += 1;
        }
    }
    return s;
}

pub fn ci_overlap(le_i: &LocallyExcited, le_j: &LocallyExcited, n_states:usize)->Array2<f64>{
    /// Compute CI overlap between TD-DFT 'wavefunctions'
    /// Excitations i->a with coefficients |C_ia| < threshold will be neglected
    /// n_states: Includes the ground state
    let threshold:f64 = 0.01;

    // calculate the overlap between the new and old geometry
    let s_ao:Array2<f64> = overlap_between_timesteps(le_i, le_j);

    let old_orbs_j = le_j.monomer.properties.old_orbs().unwrap();

    let orbs_i:ArrayView2<f64> = le_i.monomer.properties.orbs().unwrap();
    // calculate the overlap between the molecular orbitals
    let s_mo:Array2<f64> = orbs_i.t().dot(&s_ao.dot(&old_orbs_j));

    // get occupied and virtual orbitals
    let occ_indices_i = le_i.monomer.properties.occ_indices().unwrap();
    let virt_indices_i = le_i.monomer.properties.virt_indices().unwrap();
    let n_occ_i:usize = occ_indices_i.len();
    let n_virt_i: usize = virt_indices_i.len();

    let occ_indices_j = le_j.monomer.properties.occ_indices().unwrap();
    let virt_indices_j = le_j.monomer.properties.virt_indices().unwrap();
    let n_occ_j:usize = occ_indices_j.len();
    let n_virt_j: usize = virt_indices_j.len();

    // slice s_mo to get the occupied part and calculate the determinant
    let s_ij:ArrayView2<f64> = s_mo.slice(s![..n_occ_i,..n_occ_j]);
    let det_ij = s_ij.det().unwrap();

    let n_roots: usize = 10;
    // scalar coupling array
    let mut s_ci:Array2<f64> = Array2::zeros((n_states,n_states));
    // get ci coefficients from properties
    // let n_roots:usize = self.config.excited.nstates;
    let ci_coeff:ArrayView2<f64> = le_i.monomer.properties.ci_coefficients().unwrap();
    let ci_coeff:ArrayView3<f64> = ci_coeff.t().into_shape([n_roots,n_occ_i,n_virt_i]).unwrap();
    let old_ci_coeff:ArrayView3<f64> = le_j.monomer.properties.old_ci_coeffs().unwrap().into_shape([n_roots,n_occ_j,n_virt_j]).unwrap();

    // overlap between ground states <Psi0|Psi0'>
    s_ci[[0,0]] = det_ij;

    // calculate the overlap between the excited states
    // iterate over the old CI coefficients
    for i in occ_indices_j.iter() {
        for a in virt_indices_j.iter() {
            // slice old CI coefficients at the indicies i and a
            let coeffs_i = old_ci_coeff.slice(s![..,*i,*a]);
            let max_coeff_i = coeffs_i.map(|val| val.abs()).max().unwrap().to_owned();

            // slice new CI coefficients at the indicies i and a
            let coeffs_new = ci_coeff.slice(s![..,*i,*a]);
            let max_coeff_new = coeffs_new.map(|val| val.abs()).max().unwrap().to_owned();

            // if the value of the coefficient is smaller than the threshold,
            // exclude the excited state
            if max_coeff_new > threshold{
                let mut s_ia: Array2<f64> = s_ij.to_owned();
                // overlap <Psi0|PsiJ'>
                s_ia.slice_mut(s![..,*i]).assign(&s_mo.slice(s![..n_occ_j,*a]));
                let det_ia:f64 = s_ia.det().unwrap();

                // overlaps between ground state <Psi0|PsiJ'> and excited states
                for state_j in (1..n_states){
                    let c0:f64 = coeffs_new[state_j-1];
                    s_ci[[0,state_j]] += c0 * 2.0_f64.sqrt() * (det_ia * det_ij);
                }

            }
            // if the value of the coefficient is smaller than the threshold,
            // exclude the excited state
            if max_coeff_i > threshold{
                let mut s_aj:Array2<f64> = s_ij.to_owned();
                // occupied orbitals in the configuration state function |Psi_ia>
                // oveerlap <1,...,a,...|1,...,j,...>
                s_aj.slice_mut(s![*i,..]).assign(&s_mo.slice(s![*a,..n_occ_i]));
                let det_aj:f64 = s_aj.det().unwrap();

                // overlaps between ground state <PsiI|Psi0'> and excited states
                for state_i in (1..n_states){
                    let c0:f64 = coeffs_i[state_i-1];
                    s_ci[[state_i,0]] += c0 * 2.0_f64.sqrt() * (det_aj * det_ij);
                }

                // iterate over the new CI coefficients
                for j in occ_indices_i.iter(){
                    for b in virt_indices_i.iter(){
                        // slice the new CI coefficients at the indicies j and b
                        let coeffs_j = ci_coeff.slice(s![..,*j,*b]);
                        let max_coeff_j = coeffs_j.map(|val| val.abs()).max().unwrap().to_owned();
                        // if the value of the coefficient is smaller than the threshold,
                        // exclude the excited state
                        if max_coeff_j > threshold{
                            let mut s_ab:Array2<f64> = s_ij.to_owned();
                            // select part of overlap matrix for orbitals
                            // in |Psi_ia> and |Psi_jb>
                            // <1,...,a,...|1,...,b,...>
                            s_ab.slice_mut(s![*i,..]).assign(&s_mo.slice(s![*a,..n_occ_i]));
                            s_ab.slice_mut(s![..,*j]).assign(&s_mo.slice(s![..n_occ_j,*b]));
                            s_ab[[*i,*j]] = s_mo[[*a,*b]];
                            let det_ab:f64 = s_ao.det().unwrap();

                            let mut s_ib:Array2<f64> = s_ij.to_owned();
                            // <1,...,i,...|1,...,b,...>
                            s_ib.slice_mut(s![..,*j]).assign(&s_mo.slice(s![..n_occ_j,*b]));
                            let det_ib:f64 = s_ib.det().unwrap();

                            // loop over excited states
                            for state_i in (1..n_states){
                                for state_j in (1..n_states){
                                    let cc:f64 = coeffs_i[state_i-1] * coeffs_j[state_j-1];
                                    // see eqn. (9.39) in A. Humeniuk, PhD thesis (2018)
                                    s_ci[[state_i,state_j]] += cc * (det_ab * det_ij + det_aj * det_ib);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return s_ci;
}


pub fn get_sign_of_array(arr:ArrayView1<f64>)->Array1<f64>{
    let mut sign:Array1<f64> = Array1::zeros(arr.len());
    arr.iter().enumerate().for_each(|(idx,val)|
        if val.is_sign_positive(){
            sign[idx] = 1.0;
        }else{
            sign[idx] = -1.0;
        }
    );
    return sign;
}