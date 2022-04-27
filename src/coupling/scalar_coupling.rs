use crate::coupling::helpers::get_sign_of_array;
use crate::defaults;
use crate::fmo::cis_gradient::{ReducedBasisState, ReducedCT, ReducedLE, ReducedParticle};
use crate::fmo::{ESDPair, Monomer, Pair, PairType, SuperSystem};
use crate::initialization::parameters::SlaterKoster;
use crate::initialization::{Atom, System};
use crate::param::slako_transformations::{directional_cosines, slako_transformation};
use ndarray::prelude::*;
use rayon::prelude::*;
use ndarray_linalg::Determinant;
use ndarray_npy::write_npy;

impl SuperSystem {
    pub fn nonadiabatic_scalar_coupling(
        &mut self,
        excitonic_coupling: ArrayView2<f64>,
        dt: f64,
    )->(Array2<f64>,Array2<f64>,Array2<f64>,Array1<f64>,Array1<f64>) {
        // get old supersystem from properties
        let old_supersystem = self.properties.old_supersystem();
        // get the old supersystem
        let old_system: &SuperSystem = if old_supersystem.is_some() {
            old_supersystem.unwrap()
        }
        // if the dynamic is at it's first step, calculate the coupling between the
        // starting geometry
        else {
            &self
        };
        // calculate the overlap of the wavefunctions
        let (sci_overlap,s_ao):(Array2<f64>,Array2<f64>) = self.scalar_coupling_ci_overlaps(old_system);
        let dim: usize = sci_overlap.dim().0;

        // align phases
        // The eigenvalue solver produces vectors with arbitrary global phases
        // (+1 or -1). The orbitals of the ground state can also change their signs.
        // Eigen states from neighbouring geometries should change continuously.
        let diag = sci_overlap.diag();
        // get signs of the diagonal
        let sign: Array1<f64> = get_sign_of_array(diag);

        // create 2D matrix from the sign array
        let p: Array2<f64> = Array::from_diag(&sign);
        // align the new CI coefficients with the old coefficients
        let p_exclude_gs: ArrayView2<f64> = p.slice(s![1.., 1..]);
        // align the excitonic coupling matrix using the p matrix
        let excitonic_coupling: Array2<f64> = p_exclude_gs.dot(&excitonic_coupling).dot(&p_exclude_gs);

        // align overlap matrix
        let mut s_ci = sci_overlap.dot(&p);

        // The relative signs for the overlap between the ground and excited states at different geometries
        // cannot be deduced from the diagonal elements of Sci. The phases are chosen such that the coupling
        // between S0 and S1-SN changes smoothly for most of the states.
        let last_coupling:Option<ArrayView2<f64>> = old_system.properties.last_scalar_coupling();
        if last_coupling.is_some() {
            let old_s_ci: ArrayView2<f64> = last_coupling.unwrap();
            let s: Array1<f64> =
                get_sign_of_array((&old_s_ci.slice(s![0, 1..]) / &s_ci.slice(s![0, 1..])).view());
            let w: Array1<f64> =
                (&old_s_ci.slice(s![0, 1..]) - &s_ci.slice(s![0, 1..])).map(|val| val.abs());
            let mean_sign: f64 = ((&w * &s).sum() / w.sum()).signum();
            for i in (1..dim) {
                s_ci[[0, i]] *= mean_sign;
                s_ci[[i, 0]] *= mean_sign;
            }
        }

        // set diagonal elements of coupl to zero
        let mut coupling: Array2<f64> = s_ci.clone();
        coupling = coupling - Array::from_diag(&s_ci.diag());
        // coupl[A,B] = <Psi_A(t)|Psi_B(t+dt)> - delta_AB
        //            ~ <Psi_A(t)|d/dR Psi_B(t)>*dR/dt dt
        // Because of the finite time-step it will not be completely antisymmetric,
        // so antisymmetrize it
        coupling = 0.5 * (&coupling - &coupling.t());

        // save the last coupling matrix
        let last_coupling: Array2<f64> = coupling.clone();
        self.properties.set_last_scalar_coupling(last_coupling);

        // coupl = <Psi_A|d/dR Psi_B>*dR/dt * dt
        coupling = coupling / dt;

        // align the CI coefficients
        self.scalar_coupling_align_coefficients(sign.view());

        // save the signs of the couplings
        self.properties.set_coupling_signs(sign.slice(s![1..]).to_owned());

        return (coupling,excitonic_coupling,s_ao,diag.to_owned(),sign)
    }

    pub fn scalar_coupling_align_coefficients(&mut self,signs:ArrayView1<f64>){
        let basis_states = self.properties.basis_states().unwrap();

        for (idx, state) in basis_states.iter().enumerate(){
            match state{
                ReducedBasisState::LE(ref state_a) => {
                    let mol:&mut Monomer = &mut self.monomers[state_a.monomer_index];
                    let mut ci_full:Array2<f64> = mol.properties.ci_coefficients().unwrap().to_owned();
                    let ci:ArrayView1<f64> = mol.properties.ci_coefficient(state_a.state_index).unwrap();
                    let ci_aligned:Array1<f64> = signs[idx+1] * &ci;
                    ci_full.slice_mut(s![..,state_a.state_index]).assign(&ci_aligned);
                    mol.properties.set_ci_coefficients(ci_full);
                },
                ReducedBasisState::CT(ref state_a) => {
                    // let (i, j): (&ReducedParticle, &ReducedParticle) = (&state_a.hole, &state_a.electron);
                    // let type_ij: PairType = self.properties.type_of_pair(i.m_index, j.m_index);
                    //
                    // if type_ij == PairType::Pair {
                    //     // get the indices of the pairs
                    //     let pair_index: usize = self
                    //         .properties
                    //         .index_of_pair(i.m_index, j.m_index);
                    //
                    //     // get reference to the pairs
                    //     let pair: &mut Pair = &mut self.pairs[pair_index];
                    //     // // check if pair is already aligned
                    //     // if !pair.properties.aligned_pair(){
                    //     //     // align the MO coefficients of the pair
                    //     //     let orbs:ArrayView2<f64> = pair.properties.orbs().unwrap();
                    //     //     let orbs_aligned:Array2<f64> = signs[idx+1] * &orbs;
                    //     //     pair.properties.set_orbs(orbs_aligned);
                    //     //     pair.properties.set_aligned_pair(true);
                    //     // }
                    // }
                    // else{
                    //     // get the indices of the pairs
                    //     let pair_index: usize = self
                    //         .properties
                    //         .index_of_esd_pair(i.m_index, j.m_index);
                    //
                    //     // get reference to the pairs
                    //     let pair: &mut ESDPair = &mut self.esd_pairs[pair_index];
                    //
                    //     // // check if pair is already aligned
                    //     // if !pair.properties.aligned_pair(){
                    //     //     // align the MO coefficients of the pair
                    //     //     let orbs:ArrayView2<f64> = pair.properties.orbs().unwrap();
                    //     //     let orbs_aligned:Array2<f64> = signs[idx+1] * &orbs;
                    //     //     pair.properties.set_orbs(orbs_aligned);
                    //     //     pair.properties.set_aligned_pair(true);
                    //     // }
                    // }
                },
            }
        }
    }

    pub fn scalar_coupling_ci_overlaps(&self, other: &Self) -> (Array2<f64>,Array2<f64>) {
        let basis_states = self.properties.basis_states().unwrap();
        let old_basis = other.properties.basis_states().unwrap();

        // get the slater koster parameters
        let slako = &self.monomers[0].slako;
        // calculate the overlap matrix between the timesteps
        let s: Array2<f64> = self.supersystem_overlap_between_timesteps(other, slako);
        // write_npy("complete_s.npy",&s);
        // empty coupling array
        let mut coupling: Array2<f64> =
            Array2::zeros([basis_states.len() + 1, basis_states.len() + 1]);

        // get the signs of the coupling
        let coupling_signs:Array1<f64> = if other.properties.coupling_signs().is_some(){
            other.properties.coupling_signs().unwrap().to_owned()
        }
        else{
            // println!("Coupling signs not available!");
            Array1::ones(old_basis.len())
        };

        for (idx_i, state_i) in basis_states.iter().enumerate() {
            // coupling between the ground state and the diabatic states
            // coupling[[idx_i + 1, 0]] =
            //     self.scalar_coupling_diabatic_gs(other, state_i, s.view(), true);

            for (idx_j, state_j) in old_basis.iter().enumerate() {
                // coupling between the diabatic states
                let sign:f64 = coupling_signs[idx_j];
                coupling[[idx_i + 1, idx_j + 1]] =
                    self.scalar_coupling_diabatic_states(other, state_i, state_j, s.view(),sign)
            }
        }
        // cooupling between the ground state and the diabatic state
        for (idx_j, state_j) in old_basis.iter().enumerate() {
            coupling[[0, idx_j + 1]] =
                self.scalar_coupling_diabatic_gs(other, state_j, s.view(), false);
        }

        // let coupling_vec:Vec<Array1<f64>> = basis_states.par_iter().map(|state_i|{
        //     let mut arr:Array1<f64> = Array1::zeros(basis_states.len());
        //     for (idx_j, state_j) in old_basis.iter().enumerate() {
        //         // coupling between the diabatic states
        //         let sign:f64 = coupling_signs[idx_j];
        //         arr[idx_j] = self.scalar_coupling_diabatic_states(other, state_i, state_j, s.view(),sign)
        //     }
        //     arr
        // }).collect();
        //
        // // slice the coupling matrix elements
        // for (idx,arr) in coupling_vec.iter().enumerate(){
        //     coupling.slice_mut(s![idx+1,1..]).assign(arr);
        // }
        // // parallel calculation
        // let diabatic_gs:Vec<f64> = basis_states.par_iter().map(|state|{
        //     // coupling between the ground state and the diabatic states
        //     self.scalar_coupling_diabatic_gs(other, state, s.view(), true)
        // }).collect();
        // // cooupling between the ground state and the diabatic state
        // let gs_diabatic:Vec<f64> = old_basis.par_iter().map(|state|{
        //     self.scalar_coupling_diabatic_gs(other, state, s.view(), false)
        // }).collect();
        // // slice coupling matrix
        // coupling.slice_mut(s![0,1..]).assign(&Array::from(gs_diabatic));
        // coupling.slice_mut(s![1..,0]).assign(&Array::from(diabatic_gs));

        (coupling,s)
    }

    pub fn scalar_coupling_diabatic_gs(
        &self,
        other: &Self,
        state: &ReducedBasisState,
        overlap: ArrayView2<f64>,
        gs_old: bool,
    ) -> f64 {
        match state {
            ReducedBasisState::LE(ref a) => self.scalar_coupling_le_gs(other, a, overlap, gs_old),
            ReducedBasisState::CT(ref a) => self.scalar_coupling_ct_gs(other, a, overlap, gs_old),
        }
    }

    pub fn scalar_coupling_le_gs(
        &self,
        other: &Self,
        state: &ReducedLE,
        overlap: ArrayView2<f64>,
        gs_old: bool,
    ) -> f64 {
        // get the monomers of the LE state of the new and old Supersystem
        let m_new: &Monomer = &self.monomers[state.monomer_index];
        let m_old: &Monomer = &other.monomers[state.monomer_index];

        // slice the overlap matrix to get the AO overlap of the LE
        let s_ao: ArrayView2<f64> = overlap.slice(s![m_new.slice.orb, m_old.slice.orb]);
        // get the MO coefficients of the old and the new geometry
        let orbs_old: ArrayView2<f64> = m_old.properties.orbs().unwrap();
        let orbs_new: ArrayView2<f64> = m_new.properties.orbs().unwrap();
        // transform the AO overlap to the MO basis
        let s_mo: Array2<f64> = orbs_new.t().dot(&s_ao.dot(&orbs_old));

        // get the CI matrix of the LE state
        let nocc: usize = m_new.properties.occ_indices().unwrap().len();
        let nvirt: usize = m_new.properties.virt_indices().unwrap().len();
        let mut ci: Array2<f64> = Array2::zeros([nocc, nvirt]);
        if gs_old {
            ci = m_new.properties.tdm(state.state_index).unwrap().to_owned();
        } else {
            ci = m_old.properties.tdm(state.state_index).unwrap().to_owned();
        }
        // get the occupied MO overlap matrix
        let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

        // calculate the ci_overlap
        self.ci_overlap_state_gs(s_mo.view(), s_mo_occ, ci.view(), gs_old)
    }

    pub fn ci_overlap_state_gs(
        &self,
        s_mo: ArrayView2<f64>,
        s_mo_occ: ArrayView2<f64>,
        ci: ArrayView2<f64>,
        gs_old: bool,
    ) -> f64 {
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        let threshold: f64 = 0.001;
        // get nocc and nvirt
        let nocc: usize = ci.dim().0;
        let nvirt: usize = ci.dim().1;

        // calc the ground state part of the ci overlap
        let det_ij: f64 = s_mo_occ.det().unwrap();
        let norb: usize = nocc + nvirt;

        // scalar coupling value
        let mut s_ci: f64 = 0.0;

        for i in 0..nocc {
            for (a_idx, a) in (nocc..norb).into_iter().enumerate() {
                // slice the CI coefficients of the diabatic state J at the indicies i and a
                let coeff = ci[[i, a_idx]];
                if coeff.abs() > threshold {
                    if gs_old {
                        let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                        // occupied orbitals in the configuration state function |Psi_ia>
                        // overlap <1,...,a,...|1,...,j,...>
                        s_aj.slice_mut(s![i, ..nocc])
                            .assign(&s_mo.slice(s![a, ..nocc]));
                        let det_aj: f64 = s_aj.det().unwrap();

                        // get the determinant between the LE of I and the ground state of J
                        s_ci += coeff * 2.0_f64.sqrt() * det_ij * det_aj;
                    } else {
                        let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                        // occupied orbitals in the configuration state function |Psi_ia>
                        // overlap <1,...,j,...|1,...,a,...>
                        s_aj.slice_mut(s![..nocc, i])
                            .assign(&s_mo.slice(s![..nocc, a]));
                        let det_aj: f64 = s_aj.det().unwrap();

                        // get the determinant between the LE of I and the ground state of J
                        s_ci += coeff * 2.0_f64.sqrt() * det_ij * det_aj;
                    }
                }
            }
        }
        s_ci
    }

    pub fn scalar_coupling_ct_gs(
        &self,
        other: &Self,
        state: &ReducedCT,
        overlap: ArrayView2<f64>,
        gs_old: bool,
    ) -> f64 {
        // get the monomers of the LE state of the new and old Supersystem
        let (i, j): (&ReducedParticle, &ReducedParticle) = (&state.hole, &state.electron);

        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.m_index, j.m_index);

        if type_ij == PairType::Pair {
            // get the indices of the pairs
            let pair_index: usize = self
                .properties
                .index_of_pair(state.hole.m_index, state.electron.m_index);

            // get reference to the pairs
            let pair_new: &Pair = &self.pairs[pair_index];
            let pair_old: &Pair = &other.pairs[pair_index];
            // get the references to the monomers
            let m_new_a: &Monomer = &self.monomers[pair_new.i];
            let m_new_b: &Monomer = &self.monomers[pair_new.j];
            let m_old_a: &Monomer = &other.monomers[pair_old.i];
            let m_old_b: &Monomer = &other.monomers[pair_old.i];

            // prepare the AO overlap matrix
            let mut s_ao: Array2<f64> = Array2::zeros([pair_new.n_orbs, pair_old.n_orbs]);
            // fill the AO overlap matrix
            s_ao.slice_mut(s![..m_new_a.n_orbs, ..m_old_a.n_orbs])
                .assign(&overlap.slice(s![m_new_a.slice.orb, m_old_a.slice.orb]));
            s_ao.slice_mut(s![m_new_a.n_orbs.., m_old_a.n_orbs..])
                .assign(&overlap.slice(s![m_new_b.slice.orb, m_old_b.slice.orb]));
            s_ao.slice_mut(s![..m_new_a.n_orbs, m_old_a.n_orbs..])
                .assign(&overlap.slice(s![m_new_a.slice.orb, m_old_b.slice.orb]));
            s_ao.slice_mut(s![m_new_a.n_orbs.., ..m_old_a.n_orbs])
                .assign(&overlap.slice(s![m_new_b.slice.orb, m_old_a.slice.orb]));

            // get views of the MO coefficients for the pairs
            let orbs_new: ArrayView2<f64> = pair_new.properties.orbs().unwrap();
            let orbs_old: ArrayView2<f64> = pair_old.properties.orbs().unwrap();
            // transform the AO overlap matrix into the MO basis
            let s_mo: Array2<f64> = orbs_new.t().dot(&s_ao.dot(&orbs_old));

            // get the number of occupied and virtual MOs of the pair
            let nocc: usize = pair_new.properties.occ_indices().unwrap().len();
            let nvirt: usize = pair_new.properties.virt_indices().unwrap().len();

            // get the occupied MO overlap matrix
            let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

            // get occupied and virtuals orbitals of the monomers of the CT state
            let nocc_a: usize = m_new_a.properties.occ_indices().unwrap().len();
            let nocc_b: usize = m_new_b.properties.occ_indices().unwrap().len();
            let nvirt_a: usize = m_new_a.properties.virt_indices().unwrap().len();
            let nvirt_b: usize = m_new_b.properties.virt_indices().unwrap().len();

            // prepare the empty cis matrix
            let mut cis: Array2<f64> = Array2::zeros([nocc, nvirt]);

            if gs_old {
                // get the overlap matrices between the monomers and the dimer
                let s_new_i_ij: ArrayView2<f64> = pair_new.properties.s_i_ij().unwrap();
                let s_new_j_ij: ArrayView2<f64> = pair_new.properties.s_j_ij().unwrap();

                if pair_new.i == state.hole.m_index {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_occ: ArrayView2<f64> = s_new_i_ij.slice(s![..nocc_a, ..nocc]);
                    let s_j_ij_virt: ArrayView2<f64> = s_new_j_ij.slice(s![nocc_b.., nocc..]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                    ct_coefficients[[nocc_a - 1 - state.hole.ct_index, state.electron.ct_index]] =
                        1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
                } else {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_virt: ArrayView2<f64> = s_new_i_ij.slice(s![nocc_a.., nocc..]);
                    let s_j_ij_occ: ArrayView2<f64> = s_new_j_ij.slice(s![..nocc_b, ..nocc]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                    ct_coefficients[[nocc_b - 1 - state.hole.ct_index, state.electron.ct_index]] =
                        1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
                }
                // calculate the ci_overlap
                self.ci_overlap_state_gs(s_mo.view(), s_mo_occ, cis.view(), true)
            } else {
                let s_old_i_ij: ArrayView2<f64> = pair_old.properties.s_i_ij().unwrap();
                let s_old_j_ij: ArrayView2<f64> = pair_old.properties.s_j_ij().unwrap();

                if pair_old.i == state.hole.m_index {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_occ: ArrayView2<f64> = s_old_i_ij.slice(s![..nocc_a, ..nocc]);
                    let s_j_ij_virt: ArrayView2<f64> = s_old_j_ij.slice(s![nocc_b.., nocc..]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                    ct_coefficients[[nocc_a - 1 - state.hole.ct_index, state.electron.ct_index]] =
                        1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
                } else {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_virt: ArrayView2<f64> = s_old_i_ij.slice(s![nocc_a.., nocc..]);
                    let s_j_ij_occ: ArrayView2<f64> = s_old_j_ij.slice(s![..nocc_b, ..nocc]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                    ct_coefficients[[nocc_b - 1 - state.hole.ct_index, state.electron.ct_index]] =
                        1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
                }
                // calculate the ci_overlap
                self.ci_overlap_state_gs(s_mo.view(), s_mo_occ, cis.view(), false)
            }
        } else {
            // get the indices of the ESD pairs
            let pair_index: usize = self
                .properties
                .index_of_esd_pair(state.hole.m_index, state.electron.m_index);

            // get reference to the pairs
            let pair_new: &ESDPair = &self.esd_pairs[pair_index];
            let pair_old: &ESDPair = &other.esd_pairs[pair_index];
            // get the references to the monomers
            let m_new_a: &Monomer = &self.monomers[pair_new.i];
            let m_new_b: &Monomer = &self.monomers[pair_new.j];
            let m_old_a: &Monomer = &other.monomers[pair_old.i];
            let m_old_b: &Monomer = &other.monomers[pair_old.i];

            // prepare the AO overlap matrix
            let mut s_ao: Array2<f64> = Array2::zeros([pair_new.n_orbs, pair_old.n_orbs]);
            // fill the AO overlap matrix
            s_ao.slice_mut(s![..m_new_a.n_orbs, ..m_old_a.n_orbs])
                .assign(&overlap.slice(s![m_new_a.slice.orb, m_old_a.slice.orb]));
            s_ao.slice_mut(s![m_new_a.n_orbs.., m_old_a.n_orbs..])
                .assign(&overlap.slice(s![m_new_b.slice.orb, m_old_b.slice.orb]));
            s_ao.slice_mut(s![..m_new_a.n_orbs, m_old_a.n_orbs..])
                .assign(&overlap.slice(s![m_new_a.slice.orb, m_old_b.slice.orb]));
            s_ao.slice_mut(s![m_new_a.n_orbs.., ..m_old_a.n_orbs])
                .assign(&overlap.slice(s![m_new_b.slice.orb, m_old_a.slice.orb]));

            // get views of the MO coefficients for the pairs
            let orbs_new: ArrayView2<f64> = pair_new.properties.orbs().unwrap();
            let orbs_old: ArrayView2<f64> = pair_old.properties.orbs().unwrap();
            // transform the AO overlap matrix into the MO basis
            let s_mo: Array2<f64> = orbs_new.t().dot(&s_ao.dot(&orbs_old));

            // get the number of occupied and virtual MOs of the pair
            let nocc: usize = pair_new.properties.occ_indices().unwrap().len();
            let nvirt: usize = pair_new.properties.virt_indices().unwrap().len();

            // get the occupied MO overlap matrix
            let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

            // get occupied and virtuals orbitals of the monomers of the CT state
            let nocc_a: usize = m_new_a.properties.occ_indices().unwrap().len();
            let nocc_b: usize = m_new_b.properties.occ_indices().unwrap().len();
            let nvirt_a: usize = m_new_a.properties.virt_indices().unwrap().len();
            let nvirt_b: usize = m_new_b.properties.virt_indices().unwrap().len();

            // prepare the empty cis matrix
            let mut cis: Array2<f64> = Array2::zeros([nocc, nvirt]);

            if gs_old {
                // get the overlap matrices between the monomers and the dimer
                let s_new_i_ij: ArrayView2<f64> = pair_new.properties.s_i_ij().unwrap();
                let s_new_j_ij: ArrayView2<f64> = pair_new.properties.s_j_ij().unwrap();

                if pair_new.i == state.hole.m_index {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_occ: ArrayView2<f64> = s_new_i_ij.slice(s![..nocc_a, ..nocc]);
                    let s_j_ij_virt: ArrayView2<f64> = s_new_j_ij.slice(s![nocc_b.., nocc..]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                    ct_coefficients[[nocc_a - 1 - state.hole.ct_index, state.electron.ct_index]] =
                        1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
                } else {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_virt: ArrayView2<f64> = s_new_i_ij.slice(s![nocc_a.., nocc..]);
                    let s_j_ij_occ: ArrayView2<f64> = s_new_j_ij.slice(s![..nocc_b, ..nocc]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                    ct_coefficients[[nocc_b - 1 - state.hole.ct_index, state.electron.ct_index]] =
                        1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
                }
                // calculate the ci_overlap
                self.ci_overlap_state_gs(s_mo.view(), s_mo_occ, cis.view(), true)
            } else {
                let s_old_i_ij: ArrayView2<f64> = pair_old.properties.s_i_ij().unwrap();
                let s_old_j_ij: ArrayView2<f64> = pair_old.properties.s_j_ij().unwrap();

                if pair_old.i == state.hole.m_index {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_occ: ArrayView2<f64> = s_old_i_ij.slice(s![..nocc_a, ..nocc]);
                    let s_j_ij_virt: ArrayView2<f64> = s_old_j_ij.slice(s![nocc_b.., nocc..]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                    ct_coefficients[[nocc_a - 1 - state.hole.ct_index, state.electron.ct_index]] =
                        1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
                } else {
                    // reduce overlap matrices to occupied and virtual blocks
                    let s_i_ij_virt: ArrayView2<f64> = s_old_i_ij.slice(s![nocc_a.., nocc..]);
                    let s_j_ij_occ: ArrayView2<f64> = s_old_j_ij.slice(s![..nocc_b, ..nocc]);

                    // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                    // transfer is set to the value 1.0. Everything else is set to null.
                    let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                    ct_coefficients[[nocc_b - 1 - state.hole.ct_index, state.electron.ct_index]] =
                        1.0;

                    // transform the CT matrix using the reduced overlap matrices between the monomers
                    // and the dimer
                    cis = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
                }
                // calculate the ci_overlap
                self.ci_overlap_state_gs(s_mo.view(), s_mo_occ, cis.view(), false)
            }
        }
    }

    pub fn scalar_coupling_diabatic_states(
        &self,
        other: &Self,
        lhs: &ReducedBasisState,
        rhs: &ReducedBasisState,
        overlap: ArrayView2<f64>,
        sign:f64,
    ) -> f64 {
        match (lhs, rhs) {
            // coupling between two LE states.
            (ReducedBasisState::LE(ref a), ReducedBasisState::LE(ref b)) => {
                if a.monomer_index == b.monomer_index {
                    self.scalar_coupling_le_le(other, a, b, overlap)
                } else {
                    0.0
                }
            }
            // coupling between LE and CT state.
            (ReducedBasisState::LE(ref a), ReducedBasisState::CT(ref b)) => 0.0,
            // coupling between CT and LE state.
            (ReducedBasisState::CT(ref a), ReducedBasisState::LE(ref b)) => 0.0,
            // coupling between CT and CT
            (ReducedBasisState::CT(ref a), ReducedBasisState::CT(ref b)) => {
                // if (a.hole.m_index == b.hole.m_index && a.electron.m_index == b.electron.m_index)
                //     || (a.hole.m_index == b.electron.m_index
                //         && a.electron.m_index == b.hole.m_index)
                // {
                if a.hole.m_index == b.hole.m_index && a.electron.m_index == b.electron.m_index
                {
                    // self.scalar_coupling_ct_ct_new(other, a, b, overlap,sign)
                    let sci = sign * self.scalar_coupling_ct_ct_new(other, a, b, overlap,sign);
                    // let sci = self.scalar_coupling_ct_ct_new(other, a, b, overlap,sign);
                    // let sci = self.scalar_coupling_ct_ct(other, a, b, overlap);
                    // println!("sci {}",sci);
                    sci
                } else {
                    0.0
                }
            }
        }
    }

    pub fn scalar_coupling_le_le(
        &self,
        other: &Self,
        state_new: &ReducedLE,
        state_old: &ReducedLE,
        overlap: ArrayView2<f64>,
    ) -> f64 {
        // get the monomers of the LE state of the new and old Supersystem
        let m_new: &Monomer = &self.monomers[state_new.monomer_index];
        let m_old: &Monomer = &other.monomers[state_old.monomer_index];

        // slice the overlap matrix to get the AO overlap of the LE
        let s_ao: ArrayView2<f64> = overlap.slice(s![m_new.slice.orb, m_old.slice.orb]);
        // get the MO coefficients of the old and the new geometry
        let orbs_old: ArrayView2<f64> = m_old.properties.orbs().unwrap();
        let orbs_new: ArrayView2<f64> = m_new.properties.orbs().unwrap();
        // transform the AO overlap to the MO basis
        let s_mo: Array2<f64> = orbs_new.t().dot(&s_ao.dot(&orbs_old));

        // get the number of occupied MOs
        let nocc: usize = m_new.properties.occ_indices().unwrap().len();
        // get the occupied MO overlap matrix
        let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

        // get the CI matrices of the LE states
        let ci_new: ArrayView2<f64> = m_new.properties.tdm(state_new.state_index).unwrap();
        let ci_old: ArrayView2<f64> = m_old.properties.tdm(state_old.state_index).unwrap();

        // calculate the ci_overlap
        self.ci_overlap_same_fragment(s_mo.view(), s_mo_occ, ci_new, ci_old)
    }

    pub fn ci_overlap_same_fragment(
        &self,
        s_mo: ArrayView2<f64>,
        s_mo_occ: ArrayView2<f64>,
        ci_new: ArrayView2<f64>,
        ci_old: ArrayView2<f64>,
    ) -> f64 {
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        let threshold: f64 = 0.001;
        // get nocc and nvirt
        let nocc: usize = ci_new.dim().0;
        let nvirt: usize = ci_new.dim().1;

        // calc the ground state part of the ci overlap
        let det_ij: f64 = s_mo_occ.det().unwrap();
        let norb: usize = nocc + nvirt;

        // scalar coupling value
        let mut s_ci: f64 = 0.0;

        for i in 0..nocc {
            for (a_idx, a) in (nocc..norb).into_iter().enumerate() {
                // slice the CI coefficients of the diabatic state J at the indicies i and a
                let coeff_i = ci_new[[i, a_idx]];

                if coeff_i.abs() > threshold {
                    let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                    // occupied orbitals in the configuration state function |Psi_ia>
                    // overlap <1,...,a,...|1,...,j,...>
                    s_aj.slice_mut(s![i, ..nocc])
                        .assign(&s_mo.slice(s![a, ..nocc]));
                    let det_aj: f64 = s_aj.det().unwrap();

                    for j in 0..nocc {
                        for (b_idx, b) in (nocc..norb).into_iter().enumerate() {
                            let coeff_j = ci_old[[j, b_idx]];

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

                                // calculate the ci overlap
                                s_ci += coeff_j * coeff_i * (det_ab * det_ij + det_aj * det_ib);
                            }
                        }
                    }
                }
            }
        }

        s_ci
    }

    pub fn scalar_coupling_ct_ct(
        &self,
        other: &Self,
        state_new: &ReducedCT,
        state_old: &ReducedCT,
        overlap: ArrayView2<f64>,
    ) -> f64 {
        // get the monomers of the LE state of the new and old Supersystem
        let (i, j): (&ReducedParticle, &ReducedParticle) = (&state_new.hole, &state_new.electron);
        let (k, l): (&ReducedParticle, &ReducedParticle) = (&state_old.hole, &state_old.electron);

        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.m_index, j.m_index);
        // K and L
        let type_kl: PairType = other.properties.type_of_pair(k.m_index, l.m_index);

        if type_ij == PairType::Pair && type_kl == PairType::Pair {
            // get the indices of the pairs
            let pair_index_new: usize = self
                .properties
                .index_of_pair(state_new.hole.m_index, state_new.electron.m_index);
            let pair_index_old: usize = other
                .properties
                .index_of_pair(state_old.hole.m_index, state_old.electron.m_index);

            // get reference to the pairs
            let pair_new: &Pair = &self.pairs[pair_index_new];
            let pair_old: &Pair = &other.pairs[pair_index_old];
            // get the references to the monomers
            let m_new_a: &Monomer = &self.monomers[pair_new.i];
            let m_new_b: &Monomer = &self.monomers[pair_new.j];
            let m_old_a: &Monomer = &other.monomers[pair_old.i];
            let m_old_b: &Monomer = &other.monomers[pair_old.i];

            // prepare the AO overlap matrix
            let mut s_ao: Array2<f64> = Array2::zeros([pair_new.n_orbs, pair_old.n_orbs]);
            // fill the AO overlap matrix
            s_ao.slice_mut(s![..m_new_a.n_orbs, ..m_old_a.n_orbs])
                .assign(&overlap.slice(s![m_new_a.slice.orb, m_old_a.slice.orb]));
            s_ao.slice_mut(s![m_new_a.n_orbs.., m_old_a.n_orbs..])
                .assign(&overlap.slice(s![m_new_b.slice.orb, m_old_b.slice.orb]));
            s_ao.slice_mut(s![..m_new_a.n_orbs, m_old_a.n_orbs..])
                .assign(&overlap.slice(s![m_new_a.slice.orb, m_old_b.slice.orb]));
            s_ao.slice_mut(s![m_new_a.n_orbs.., ..m_old_a.n_orbs])
                .assign(&overlap.slice(s![m_new_b.slice.orb, m_old_a.slice.orb]));

            // get views of the MO coefficients for the pairs
            let orbs_new: ArrayView2<f64> = pair_new.properties.orbs().unwrap();
            let orbs_old: ArrayView2<f64> = pair_old.properties.orbs().unwrap();
            // transform the AO overlap matrix into the MO basis
            let s_mo: Array2<f64> = orbs_new.t().dot(&s_ao.dot(&orbs_old));

            // get the number of occupied and virtual MOs of the pair
            let nocc: usize = pair_new.properties.occ_indices().unwrap().len();
            let nvirt: usize = pair_new.properties.virt_indices().unwrap().len();

            // get the occupied MO overlap matrix
            let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

            // get occupied and virtuals orbitals of the monomers of the CT state
            let nocc_a: usize = m_new_a.properties.occ_indices().unwrap().len();
            let nocc_b: usize = m_new_b.properties.occ_indices().unwrap().len();
            let nvirt_a: usize = m_new_a.properties.virt_indices().unwrap().len();
            let nvirt_b: usize = m_new_b.properties.virt_indices().unwrap().len();

            // prepare the empty cis matrix
            let mut cis_new: Array2<f64> = Array2::zeros([nocc, nvirt]);
            let mut cis_old: Array2<f64> = Array2::zeros([nocc, nvirt]);

            // get the overlap matrices between the monomers and the dimer
            let s_new_i_ij: ArrayView2<f64> = pair_new.properties.s_i_ij().unwrap();
            let s_new_j_ij: ArrayView2<f64> = pair_new.properties.s_j_ij().unwrap();
            let s_old_i_ij: ArrayView2<f64> = pair_old.properties.s_i_ij().unwrap();
            let s_old_j_ij: ArrayView2<f64> = pair_old.properties.s_j_ij().unwrap();

            if pair_new.i == state_new.hole.m_index {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_occ: ArrayView2<f64> = s_new_i_ij.slice(s![..nocc_a, ..nocc]);
                let s_j_ij_virt: ArrayView2<f64> = s_new_j_ij.slice(s![nocc_b.., nocc..]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                ct_coefficients[[
                    nocc_a - 1 - state_new.hole.ct_index,
                    state_new.electron.ct_index,
                ]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_new = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
            } else {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_virt: ArrayView2<f64> = s_new_i_ij.slice(s![nocc_a.., nocc..]);
                let s_j_ij_occ: ArrayView2<f64> = s_new_j_ij.slice(s![..nocc_b, ..nocc]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                ct_coefficients[[
                    nocc_b - 1 - state_new.hole.ct_index,
                    state_new.electron.ct_index,
                ]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_new = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
            }
            if pair_old.i == state_old.hole.m_index {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_occ: ArrayView2<f64> = s_old_i_ij.slice(s![..nocc_a, ..nocc]);
                let s_j_ij_virt: ArrayView2<f64> = s_old_j_ij.slice(s![nocc_b.., nocc..]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                ct_coefficients[[
                    nocc_a - 1 - state_old.hole.ct_index,
                    state_old.electron.ct_index,
                ]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_old = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
            } else {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_virt: ArrayView2<f64> = s_old_i_ij.slice(s![nocc_a.., nocc..]);
                let s_j_ij_occ: ArrayView2<f64> = s_old_j_ij.slice(s![..nocc_b, ..nocc]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                ct_coefficients[[
                    nocc_b - 1 - state_old.hole.ct_index,
                    state_old.electron.ct_index,
                ]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_old = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
            }
            // calculate the ci_overlap
            self.ci_overlap_same_fragment(s_mo.view(), s_mo_occ, cis_new.view(), cis_old.view())
        } else {
            // get the indices of the ESD pairs
            let pair_index_new: usize = self
                .properties
                .index_of_esd_pair(state_new.hole.m_index, state_new.electron.m_index);
            let pair_index_old: usize = other
                .properties
                .index_of_esd_pair(state_old.hole.m_index, state_old.electron.m_index);

            // get reference to the pairs
            let pair_new: &ESDPair = &self.esd_pairs[pair_index_new];
            let pair_old: &ESDPair = &other.esd_pairs[pair_index_old];
            // get the references to the monomers
            let m_new_a: &Monomer = &self.monomers[pair_new.i];
            let m_new_b: &Monomer = &self.monomers[pair_new.j];
            let m_old_a: &Monomer = &other.monomers[pair_old.i];
            let m_old_b: &Monomer = &other.monomers[pair_old.i];

            // prepare the AO overlap matrix
            let mut s_ao: Array2<f64> = Array2::zeros([pair_new.n_orbs, pair_old.n_orbs]);
            // fill the AO overlap matrix
            s_ao.slice_mut(s![..m_new_a.n_orbs, ..m_old_a.n_orbs])
                .assign(&overlap.slice(s![m_new_a.slice.orb, m_old_a.slice.orb]));
            s_ao.slice_mut(s![m_new_a.n_orbs.., m_old_a.n_orbs..])
                .assign(&overlap.slice(s![m_new_b.slice.orb, m_old_b.slice.orb]));
            s_ao.slice_mut(s![..m_new_a.n_orbs, m_old_a.n_orbs..])
                .assign(&overlap.slice(s![m_new_a.slice.orb, m_old_b.slice.orb]));
            s_ao.slice_mut(s![m_new_a.n_orbs.., ..m_old_a.n_orbs])
                .assign(&overlap.slice(s![m_new_b.slice.orb, m_old_a.slice.orb]));

            // get views of the MO coefficients for the pairs
            let orbs_new: ArrayView2<f64> = pair_new.properties.orbs().unwrap();
            let orbs_old: ArrayView2<f64> = pair_old.properties.orbs().unwrap();
            // transform the AO overlap matrix into the MO basis
            let s_mo: Array2<f64> = orbs_new.t().dot(&s_ao.dot(&orbs_old));

            // get the number of occupied and virtual MOs of the pair
            let nocc: usize = pair_new.properties.occ_indices().unwrap().len();
            let nvirt: usize = pair_new.properties.virt_indices().unwrap().len();

            // get the occupied MO overlap matrix
            let s_mo_occ: ArrayView2<f64> = s_mo.slice(s![..nocc, ..nocc]);

            // get occupied and virtuals orbitals of the monomers of the CT state
            let nocc_a: usize = m_new_a.properties.occ_indices().unwrap().len();
            let nocc_b: usize = m_new_b.properties.occ_indices().unwrap().len();
            let nvirt_a: usize = m_new_a.properties.virt_indices().unwrap().len();
            let nvirt_b: usize = m_new_b.properties.virt_indices().unwrap().len();

            // prepare the empty cis matrix
            let mut cis_new: Array2<f64> = Array2::zeros([nocc, nvirt]);
            let mut cis_old: Array2<f64> = Array2::zeros([nocc, nvirt]);

            // get the overlap matrices between the monomers and the dimer
            let s_new_i_ij: ArrayView2<f64> = pair_new.properties.s_i_ij().unwrap();
            let s_new_j_ij: ArrayView2<f64> = pair_new.properties.s_j_ij().unwrap();
            let s_old_i_ij: ArrayView2<f64> = pair_old.properties.s_i_ij().unwrap();
            let s_old_j_ij: ArrayView2<f64> = pair_old.properties.s_j_ij().unwrap();

            if pair_new.i == state_new.hole.m_index {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_occ: ArrayView2<f64> = s_new_i_ij.slice(s![..nocc_a, ..nocc]);
                let s_j_ij_virt: ArrayView2<f64> = s_new_j_ij.slice(s![nocc_b.., nocc..]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                ct_coefficients[[
                    nocc_a - 1 - state_new.hole.ct_index,
                    state_new.electron.ct_index,
                ]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_new = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
            } else {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_virt: ArrayView2<f64> = s_new_i_ij.slice(s![nocc_a.., nocc..]);
                let s_j_ij_occ: ArrayView2<f64> = s_new_j_ij.slice(s![..nocc_b, ..nocc]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                ct_coefficients[[
                    nocc_b - 1 - state_new.hole.ct_index,
                    state_new.electron.ct_index,
                ]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_new = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
            }
            if pair_old.i == state_old.hole.m_index {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_occ: ArrayView2<f64> = s_old_i_ij.slice(s![..nocc_a, ..nocc]);
                let s_j_ij_virt: ArrayView2<f64> = s_old_j_ij.slice(s![nocc_b.., nocc..]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_a, nvirt_b]);
                ct_coefficients[[
                    nocc_a - 1 - state_old.hole.ct_index,
                    state_old.electron.ct_index,
                ]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_old = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));
            } else {
                // reduce overlap matrices to occupied and virtual blocks
                let s_i_ij_virt: ArrayView2<f64> = s_old_i_ij.slice(s![nocc_a.., nocc..]);
                let s_j_ij_occ: ArrayView2<f64> = s_old_j_ij.slice(s![..nocc_b, ..nocc]);

                // Create matrix for the CT. Only the matrix element, which corresponds to the charge
                // transfer is set to the value 1.0. Everything else is set to null.
                let mut ct_coefficients: Array2<f64> = Array2::zeros([nocc_b, nvirt_a]);
                ct_coefficients[[
                    nocc_b - 1 - state_old.hole.ct_index,
                    state_old.electron.ct_index,
                ]] = 1.0;

                // transform the CT matrix using the reduced overlap matrices between the monomers
                // and the dimer
                cis_old = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));
            }
            // calculate the ci_overlap
            self.ci_overlap_same_fragment(s_mo.view(), s_mo_occ, cis_new.view(), cis_old.view())
        }
    }

    pub fn scalar_coupling_ct_ct_new(
        &self,
        other: &Self,
        state_new: &ReducedCT,
        state_old: &ReducedCT,
        overlap: ArrayView2<f64>,
        sign_old:f64,
    ) -> f64 {
        // get the monomers of the LE state of the new and old Supersystem
        let (i, j): (&ReducedParticle, &ReducedParticle) = (&state_new.electron, &state_new.hole);
        let (k, l): (&ReducedParticle, &ReducedParticle) = (&state_old.electron, &state_old.hole);

        // if i.m_index == k.m_index{

        // get the references to the monomers
        let m_new_hole: &Monomer = &self.monomers[j.m_index];
        let m_new_elec: &Monomer = &self.monomers[i.m_index];
        let m_old_hole: &Monomer = &other.monomers[l.m_index];
        let m_old_elec: &Monomer = &other.monomers[k.m_index];

        // println!("Monomer indices: new ij {} {}, old kl {} {}",i.m_index,j.m_index,k.m_index,l.m_index);

        // dimension of the MO overlap matrix
        let dim:usize = m_new_hole.n_orbs + m_new_elec.n_orbs;
        // get the MO coefficients of the monomers
        let orbs_j = m_new_hole.properties.orbs().unwrap();
        let orbs_i = m_new_elec.properties.orbs().unwrap();
        let orbs_l = m_old_hole.properties.orbs().unwrap();
        let orbs_k = m_old_elec.properties.orbs().unwrap();

        // write_npy("new_orbs.npy",&orbs_j);
        // write_npy("old_orbs.npy",&orbs_l);

        // prepare the MO overlap matrix
        let mut s_mo:Array2<f64> = Array2::zeros([dim,dim]);
        // fill the MO overlap matrix
        s_mo.slice_mut(s![..m_new_hole.n_orbs, ..m_old_hole.n_orbs])
            .assign(&orbs_j.t().dot(&overlap.slice(s![m_new_hole.slice.orb, m_new_hole.slice.orb]).dot(&orbs_l)));
        s_mo.slice_mut(s![m_new_hole.n_orbs.., m_old_hole.n_orbs..])
            .assign(&orbs_i.t().dot(&overlap.slice(s![m_new_elec.slice.orb, m_new_elec.slice.orb]).dot(&orbs_k)));
        s_mo.slice_mut(s![..m_new_hole.n_orbs, m_old_hole.n_orbs..])
            .assign(&orbs_j.t().dot(&overlap.slice(s![m_new_hole.slice.orb, m_new_elec.slice.orb]).dot(&orbs_k)));
        s_mo.slice_mut(s![m_new_hole.n_orbs.., ..m_old_hole.n_orbs])
            .assign(&orbs_i.t().dot(&overlap.slice(s![m_new_elec.slice.orb, m_new_hole.slice.orb]).dot(&orbs_l)));

        // occupied and virtual indices
        let nocc_j:usize = m_new_hole.properties.occ_indices().unwrap().len();
        let nocc_i:usize = m_new_elec.properties.occ_indices().unwrap().len();
        let nvirt_j:usize = m_new_hole.properties.virt_indices().unwrap().len();
        let nvirt_i:usize = m_new_elec.properties.virt_indices().unwrap().len();
        // number of orbitals
        let norb_i:usize = nocc_i+nvirt_i;
        let norb_j:usize = nocc_j+nvirt_j;
        let nocc:usize = nocc_i + nocc_j;
        let nvirt:usize = nvirt_i + nvirt_j;
        // slice the MO overlap matrix
        let mut s_mo_occ:Array2<f64> = Array2::zeros((nocc,nocc));
        s_mo_occ.slice_mut(s![..nocc_j,..nocc_j]).assign(&s_mo.slice(s![..nocc_j,..nocc_j]));
        s_mo_occ.slice_mut(s![nocc_j..,nocc_j..]).assign(&s_mo.slice(s![norb_j..norb_j+nocc_i,norb_j..norb_j+nocc_i]));
        s_mo_occ.slice_mut(s![..nocc_j,nocc_j..]).assign(&s_mo.slice(s![..nocc_j,norb_j..norb_j+nocc_i]));
        s_mo_occ.slice_mut(s![nocc_j..,..nocc_j]).assign(&s_mo.slice(s![norb_j..norb_j+nocc_i,..nocc_j]));

        // get the CI coefficients of the CT states
        let mut cis_new: Array2<f64> = Array2::zeros([nocc_j,nvirt_i]);
        cis_new[[
            nocc_j - 1 - state_new.hole.ct_index,
            state_new.electron.ct_index,
        ]] = 1.0;

        let mut cis_old: Array2<f64> = Array2::zeros([nocc_j,nvirt_i]);
        cis_old[[
            nocc_j - 1 - state_old.hole.ct_index,
            state_old.electron.ct_index,
        ]] = 1.0; // * sign_old;

        // write_npy("s_mo.npy",&s_mo);
        // write_npy("s_mo_occ.npy",&s_mo_occ);

        self.ci_overlap_ct_ct(
            s_mo.view(),
            s_mo_occ.view(),
            cis_new.view(),
            cis_old.view(),
            nocc_j,
            nvirt_j,
            nocc_i,
            nvirt_i
        )
    }

    pub fn ci_overlap_ct_ct(
        &self,
        s_mo: ArrayView2<f64>,
        s_mo_occ: ArrayView2<f64>,
        ci_new: ArrayView2<f64>,
        ci_old: ArrayView2<f64>,
        nocc_hole:usize,
        nvirt_hole:usize,
        nocc_elec:usize,
        nvirt_elec:usize,
    ) -> f64 {
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        let threshold: f64 = 0.001;

        // calc the ground state part of the ci overlap
        let det_ij: f64 = s_mo_occ.det().unwrap();

        // get indics for iterating over the overlap matrix
        let norb_hole: usize = nocc_hole + nvirt_hole;
        let norb_elec:usize = nocc_elec + nvirt_elec;
        let norb:usize = norb_hole + norb_elec;
        let nocc:usize = nocc_hole + nocc_elec;
        let start_virt_elec = nocc + nvirt_hole;

        // scalar coupling value
        let mut s_ci: f64 = 0.0;

        for i in 0..nocc_hole {
            for (a_idx, a) in (start_virt_elec..norb).into_iter().enumerate() {
                // slice the CI coefficients of the diabatic state J at the indicies i and a
                let coeff_i = ci_new[[i, a_idx]];

                if coeff_i.abs() > threshold {
                    let mut s_aj: Array2<f64> = s_mo_occ.to_owned();
                    // occupied orbitals in the configuration state function |Psi_ia>
                    // overlap <1,...,a,...|1,...,j,...>
                    s_aj.slice_mut(s![i, ..nocc_hole])
                        .assign(&s_mo.slice(s![a, ..nocc_hole]));
                    s_aj.slice_mut(s![i, nocc_hole..nocc])
                        .assign(&s_mo.slice(s![a, norb_hole..norb_hole+nocc_elec]));
                    let det_aj: f64 = s_aj.det().unwrap();

                    for j in 0..nocc_hole {
                        for (b_idx, b) in (start_virt_elec..norb).into_iter().enumerate() {
                            let coeff_j = ci_old[[j, b_idx]];

                            if coeff_j.abs() > threshold {
                                let mut s_ab: Array2<f64> = s_mo_occ.to_owned();
                                // select part of overlap matrix for orbitals
                                // in |Psi_ia> and |Psi_jb>
                                // <1,...,a,...|1,...,b,...>
                                s_ab.slice_mut(s![i, ..nocc_hole])
                                    .assign(&s_mo.slice(s![a, ..nocc_hole]));
                                s_ab.slice_mut(s![i, nocc_hole..nocc])
                                    .assign(&s_mo.slice(s![a, norb_hole..norb_hole+nocc_elec]));

                                s_ab.slice_mut(s![..nocc_hole, j])
                                    .assign(&s_mo.slice(s![..nocc_hole, b]));
                                s_ab.slice_mut(s![nocc_hole..nocc, j])
                                    .assign(&s_mo.slice(s![norb_hole..norb_hole+nocc_elec, b]));
                                s_ab[[i, j]] = s_mo[[a, b]];
                                let det_ab: f64 = s_ab.det().unwrap();

                                let mut s_ib: Array2<f64> = s_mo_occ.to_owned();
                                // <1,...,i,...|1,...,b,...>
                                s_ib.slice_mut(s![..nocc_hole, j])
                                    .assign(&s_mo.slice(s![..nocc_hole, b]));
                                s_ib.slice_mut(s![nocc_hole..nocc, j])
                                    .assign(&s_mo.slice(s![norb_hole..norb_hole+nocc_elec, b]));
                                let det_ib: f64 = s_ib.det().unwrap();

                                // calculate the ci overlap
                                s_ci += coeff_j * coeff_i * (det_ab * det_ij + det_aj * det_ib);
                            }
                        }
                    }
                }
            }
        }

        s_ci
    }

    pub fn supersystem_overlap_between_timesteps(
        &self,
        other: &Self,
        skt: &SlaterKoster,
    ) -> Array2<f64> {
        // get the atoms of the old Supersystem
        let old_atoms = &other.atoms;
        // get the atoms of the Supersystem at the new geometry
        let atoms = &self.atoms;
        // get the number of the orbitals of the Supersystem
        let n_orbs: usize = self.properties.n_occ().unwrap() + self.properties.n_virt().unwrap();
        // empty matrix for the overlap
        let mut s: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);

        let mut mu: usize = 0;
        // iterate over the atoms of the system
        for (idx_i, atom_i) in atoms.iter().enumerate() {
            // iterate over the orbitals on atom I
            for orbi in atom_i.valorbs.iter() {
                // iterate over the atoms of the old geometry
                let mut nu: usize = 0;
                for (j, atom_j) in old_atoms.iter().enumerate() {
                    // iterate over the orbitals on atom J
                    for orbj in atom_j.valorbs.iter() {
                        if (atom_i - atom_j).norm() < defaults::PROXIMITY_CUTOFF {
                            if atom_i <= atom_j {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atom_i.xyz, &atom_j.xyz);
                                s[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt.get(atom_i.kind, atom_j.kind).s_spline,
                                    orbi.l,
                                    orbi.m,
                                    orbj.l,
                                    orbj.m,
                                );
                            } else {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atom_j.xyz, &atom_i.xyz);
                                s[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt.get(atom_j.kind, atom_i.kind).s_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                );
                            }
                        }
                        nu += 1;
                    }
                }
                mu += 1;
            }
        }
        s
    }
}

impl System {
    pub fn get_scalar_coupling(
        &self,
        dt: f64,
        old_atoms: &[Atom],
        old_orbs: ArrayView2<f64>,
        old_ci_coeff: ArrayView2<f64>,
        old_s_ci: ArrayView2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        // TODO: save and load arrays from the previous iteration.
        let n_states: usize = self.config.excited.nstates + 1;
        // scalar coupling matrix
        let s_ci: Array2<f64> = self.ci_overlap(old_atoms, old_orbs, old_ci_coeff, n_states);
        // align phases
        // The eigenvalue solver produces vectors with arbitrary global phases
        // (+1 or -1). The orbitals of the ground state can also change their signs.
        // Eigen states from neighbouring geometries should change continuously.
        let diag = s_ci.diag();
        // get signs of the diagonal
        let sign: Array1<f64> = get_sign_of_array(diag);

        let p: Array2<f64> = Array::from_diag(&sign);
        // align the new CI coefficients with the old coefficients
        let p_exclude_gs: ArrayView2<f64> = p.slice(s![1.., 1..]);
        //TODO:save aligned coeff as old ci coefficients
        let aligned_coeff: Array2<f64> = self
            .properties
            .ci_coefficients()
            .unwrap()
            .dot(&p_exclude_gs);

        // align overlap matrix
        let mut s_ci = s_ci.dot(&p);

        // The relative signs for the overlap between the ground and excited states at different geometries
        // cannot be deduced from the diagonal elements of Sci. The phases are chosen such that the coupling
        // between S0 and S1-SN changes smoothly for most of the states.
        let s: Array1<f64> =
            get_sign_of_array((&old_s_ci.slice(s![0, 1..]) / &s_ci.slice(s![0, 1..])).view());
        let w: Array1<f64> =
            (&old_s_ci.slice(s![0, 1..]) - &s_ci.slice(s![0, 1..])).map(|val| val.abs());
        let mean_sign: f64 = ((&w * &s).sum() / w.sum()).signum();
        for i in (1..n_states) {
            s_ci[[0, i]] *= mean_sign;
            s_ci[[i, 0]] *= mean_sign;
        }

        // coupl[A,B] = <Psi_A(t)|Psi_B(t+dt)> - delta_AB
        //            ~ <Psi_A(t)|d/dR Psi_B(t)>*dR/dt dt
        // The scalar coupling matrix should be more or less anti-symmetric
        // provided the time-step is small enough
        // set diagonal elements of coupl to zero
        let mut coupling: Array2<f64> = s_ci.clone();
        coupling = coupling - Array::from_diag(&s_ci.diag());

        // Because of the finite time-step it will not be completely antisymmetric,
        coupling = 0.5 * (&coupling - &coupling.t());

        // coupl = <Psi_A|d/dR Psi_B>*dR/dt * dt
        coupling = coupling / dt;

        return (coupling, s_ci);
    }
}
