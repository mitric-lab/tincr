use crate::fmo::{SuperSystem, ExcitonStates, BasisState, Monomer, ExcitedStateMonomerGradient, PairType};
use crate::initialization::{Atom, MO};
use ndarray::prelude::*;
use crate::utils::array_helper::argsort_abs;
use crate::gradients::helpers::gradient_v_rep;
use std::ops::AddAssign;

impl SuperSystem {
    pub fn fmo_cis_gradient(&mut self,state:usize)->Array1<f64>{
        // get the coefficient of the state
        let coeff:ArrayView1<f64> = self.properties.ci_coefficient(state).unwrap();

        // Sort the indices by coefficients of the state
        let sorted_indices: Vec<usize> = argsort_abs(coeff.view());

        // create diabatic basis
        let basis: Vec<BasisState> = self.create_diab_basis();

        // Get the contribution of the basis states to the state
        let threshold = 0.001;
        let mut contributions:Vec<ReducedBasisState> = Vec::new();
        // Reverse the Iterator to write the largest amplitude first.
        for i in sorted_indices.into_iter().rev() {
            // Amplitude of the current transition.
            let c: f64 = coeff[i].abs();

            // Only write transition with an amplitude higher than a certain threshold.
            if c > threshold {
                let state = basis.get(i).unwrap();

                // calculate the energy of the basis state
                let energy_state:f64 = self.exciton_coupling(state,state);

                // get the state/MO indices of the LE/CT state and the
                // indices of the monomers
                let (state_ind,m_index):(Vec<usize>,Vec<usize>) = match state{
                    BasisState::LE(ref a) =>{
                        // get index and the Atom vector of the monomer
                        let le_state:usize = a.n;
                        let monomer_ind:usize = a.monomer.index;

                        (vec![le_state],vec![monomer_ind])
                    }
                    BasisState::CT(ref a) =>{
                        // get indices
                        let index_i:usize = a.hole.idx;
                        let index_j:usize = a.electron.idx;

                        // get Atom vector and nocc of the monomer I
                        let mol_i:&Monomer = &self.monomers[index_i];
                        let nocc_i:usize = mol_i.properties.occ_indices().unwrap().len();
                        drop(mol_i);

                        // get Atom vector and nocc of the monomer J
                        let mol_j:&Monomer = &self.monomers[index_j];
                        let nocc_j:usize = mol_j.properties.occ_indices().unwrap().len();
                        drop(mol_j);

                        // get ct indices of the MOs
                        let mo_i:usize = (a.hole.mo.idx as i32 - (nocc_i-1) as i32).abs() as usize;
                        let mo_j:usize = a.electron.mo.idx - nocc_j;

                        (vec![mo_i,mo_j],vec![index_i,index_j])
                    }
                };
                let reduced_state:ReducedBasisState =
                    ReducedBasisState::new(energy_state,m_index,state_ind,c.powi(2));
                contributions.push(reduced_state);
            }
        }

        // iterate over the basis states and calculate their gradient
        let mut gradients:Vec<Array1<f64>> = Vec::new();
        let n_atoms:usize = self.atoms.len();
        let mut gradient:Array1<f64> = Array1::zeros(3*n_atoms);

        for state in contributions.iter(){
            // let (gradient):(Array1<f64>) = match state.type_of_state{
            match state.type_of_state{
                BasisStateType::LE =>{
                    // get index and the Atom vector of the monomer
                    let le_state:usize = state.state_indices[0];
                    let monomer_ind:usize = state.monomer_indices[0];
                    let mol:&mut Monomer = &mut self.monomers[monomer_ind];
                    let monomer_atoms:&[Atom] = &self.atoms[mol.slice.atom_as_range()];

                    // calculate the gradient
                    mol.prepare_excited_gradient(monomer_atoms);
                    let grad = mol.tda_gradient_lc(le_state) * state.coefficient;

                    gradient.slice_mut(s![mol.slice.grad]).add_assign(&grad);
                    // grad
                }
                BasisStateType::CT =>{
                    let energy_state:f64 = state.energy;
                    // get indices
                    let index_i:usize = state.monomer_indices[0];
                    let index_j:usize = state.monomer_indices[1];

                    // get Atom vector and nocc of the monomer I
                    let mol_i:&Monomer = &self.monomers[index_i];
                    let n_atoms_i:usize = mol_i.n_atoms;
                    let atoms_slice_i = mol_i.slice.grad;
                    drop(mol_i);

                    // get Atom vector and nocc of the monomer J
                    let mol_j:&Monomer = &self.monomers[index_j];
                    let n_atoms_j:usize = mol_j.n_atoms;
                    let atoms_slice_j = mol_j.slice.grad;
                    drop(mol_j);

                    // get ct indices of the MOs
                    let mo_i:usize = state.state_indices[0];
                    let mo_j:usize = state.state_indices[1];

                    if index_i < index_j{
                        let grad = self.ct_gradient_new(
                            index_i,
                            index_j,
                            mo_i,
                            mo_j,
                            energy_state,
                            true,
                        ) *state.coefficient;
                        let grad_i:Array1<f64> = grad.slice(s![0..3*n_atoms_i]).to_owned();
                        let grad_j:Array1<f64> = grad.slice(s![3*n_atoms_i..]).to_owned();
                        gradient.slice_mut(s![atoms_slice_i]).add_assign(&grad_i);
                        gradient.slice_mut(s![atoms_slice_j]).add_assign(&grad_j);
                    }
                    else{
                        let grad = self.ct_gradient_new(
                            index_j,
                            index_i,
                            mo_j,
                            mo_i,
                            energy_state,
                            false,
                        ) *state.coefficient;
                        let grad_j:Array1<f64> = grad.slice(s![0..3*n_atoms_j]).to_owned();
                        let grad_i:Array1<f64> = grad.slice(s![3*n_atoms_j..]).to_owned();
                        gradient.slice_mut(s![atoms_slice_i]).add_assign(&grad_i);
                        gradient.slice_mut(s![atoms_slice_j]).add_assign(&grad_j);
                    }

                    // grad
                }
            }
            // gradients.push(gradient);
        }
        // Reorder the calculated gradient using the monomer indices of the specific states
        return gradient;
    }
}

struct ReducedBasisState{
    pub energy:f64,
    pub monomer_indices:Vec<usize>,
    pub state_indices:Vec<usize>,
    pub type_of_state:BasisStateType,
    pub coefficient:f64,
}

impl ReducedBasisState{
    pub fn new(
        energy:f64,
        monomer_indices:Vec<usize>,
        state_indices:Vec<usize>,
        coefficient:f64,
    )->ReducedBasisState{
        if monomer_indices.len() > 1{
            ReducedBasisState{
                energy:energy,
                monomer_indices:monomer_indices,
                state_indices:state_indices,
                type_of_state:BasisStateType::CT,
                coefficient:coefficient,
            }
        }
        else{
            ReducedBasisState{
                energy:energy,
                monomer_indices:monomer_indices,
                state_indices:state_indices,
                type_of_state:BasisStateType::LE,
                coefficient:coefficient,
            }
        }
    }
}

pub enum BasisStateType{
    LE,
    CT
}