use crate::fmo::{SuperSystem, ExcitonStates, BasisState, Monomer, ExcitedStateMonomerGradient};
use crate::initialization::{Atom, MO};
use ndarray::prelude::*;
use crate::utils::array_helper::argsort_abs;

impl SuperSystem {
    pub fn fmo_cis_gradient(&mut self,state:usize){
        // get the coefficient of the state
        let coeff:ArrayView1<f64> = self.properties.ci_coefficient(state).unwrap();

        // Sort the indices by coefficients of the state
        let sorted_indices: Vec<usize> = argsort_abs(coeff.view());

        // create diabatic basis
        let basis: Vec<BasisState> = self.create_diab_basis();

        // Get the contribution of the basis states to the state
        let threshold = 0.1;
        let mut contributions:Vec<&BasisState> = Vec::new();
        // Reverse the Iterator to write the largest amplitude first.
        for i in sorted_indices.into_iter().rev() {
            // Amplitude of the current transition.
            let c: f64 = coeff[i].abs();

            // Only write transition with an amplitude higher than a certain threshold.
            if c > threshold {
                 contributions.push(basis.get(i).unwrap());
            }
        }

        // iterate over the basis states and calculate their gradient
        for state in contributions.iter(){
            // calculate the energy of the basis state
            let energy_state:f64 = self.exciton_coupling(state,state);

            let (gradient,indices):(Array1<f64>,Vec<usize>) = match state{
                BasisState::LE(ref a) =>{
                    // get index and the Atom vector of the monomer
                    let le_state:usize = a.n;
                    let monomer_ind:usize = a.monomer.index;
                    let mol:&mut Monomer = &mut self.monomers[monomer_ind];
                    let monomer_atoms:&[Atom] = a.atoms;

                    // calculate the gradient
                    mol.prepare_excited_gradient(monomer_atoms);
                    let grad = mol.tda_gradient_lc(le_state);

                    (grad,vec![monomer_ind])
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
                    let mo_i:usize = (a.hole.mo.idx as i32 - nocc_i as i32).abs() as usize;
                    let mo_j:usize = a.electron.mo.idx - nocc_j;

                    let grad = self.ct_gradient_new(
                        index_i,
                        index_j,
                        mo_i,
                        mo_j,
                        energy_state,
                        true
                    );

                    (grad,vec![index_i,index_j])
                }
            };
        }
    }
}