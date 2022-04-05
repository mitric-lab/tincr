use crate::fmo::cis_gradient::ReducedBasisState;
use crate::fmo::{Monomer, SuperSystem};
use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use ndarray::prelude::*;

impl System {
    pub fn calculate_energies_and_gradient(&mut self, state: usize) -> (Array1<f64>, Array1<f64>) {
        let mut gradient: Array1<f64> = Array::zeros(3 * self.n_atoms);
        let mut energies: Array1<f64> = Array1::zeros(self.config.excited.nstates + 1);

        if state == 0 {
            // ground state energy
            self.prepare_scc();
            let gs_energy: f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            // // calculate excited states
            // self.prepare_tda();
            // self.run_tda(self.config.excited.nstates, 100, 1e-4);
            // let ci_energies:ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            // energies.slice_mut(s![1..]).assign(&(gs_energy +&ci_energies));

            gradient = self.ground_state_gradient(true);
        } else {
            // ground state energy
            let excited_state: usize = state - 1;
            self.prepare_scc();
            let gs_energy: f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            // calculate excited states
            self.prepare_tda();
            self.run_tda(self.config.excited.nstates, 200, 1e-4);
            let ci_energies: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            energies
                .slice_mut(s![1..])
                .assign(&(gs_energy + &ci_energies));

            gradient = self.ground_state_gradient(true);

            self.prepare_excited_grad();

            if self.config.lc.long_range_correction {
                gradient = gradient + self.tda_gradient_lc(excited_state);
            } else {
                gradient = gradient + self.tda_gradient_nolc(excited_state);
            }
        }
        self.properties.reset();

        return (energies, gradient);
    }
}

impl SuperSystem {
    pub fn calculate_energies_and_gradient(&mut self, state: usize) -> (Array1<f64>, Array1<f64>) {
        let n_atoms: usize = self.atoms.len();
        let mut gradient: Array1<f64> = Array::zeros(3 * n_atoms);
        let n_states: usize = self.config.excited.nstates + 1;
        let mut energies: Array1<f64> = Array1::zeros(n_states);

        if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            let gs_energy = self.run_scc().unwrap();
            energies[0] = gs_energy;
            gradient = self.ground_state_gradient();

            // // calculate excited states
            // self.create_exciton_hamiltonian();
            // let ex_energies:ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            // energies.slice_mut(s![1..]).assign(&(gs_energy +&ex_energies));
        } else {
            // ground state energy and gradient
            self.prepare_scc();
            let gs_energy = self.run_scc().unwrap();
            energies[0] = gs_energy;
            gradient = self.ground_state_gradient();

            // calculate excited states
            self.create_exciton_hamiltonian();
            let ex_energies: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            energies
                .slice_mut(s![1..])
                .assign(&(gs_energy + &ex_energies.slice(s![..n_states - 1])));

            let excited_state: usize = state - 1;
            gradient = gradient + self.fmo_cis_gradient(excited_state);
        }

        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.properties.reset();

        return (energies, gradient);
    }

    pub fn calculate_ehrenfest_gradient(
        &mut self,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
    ) -> (
        f64,
        Array2<f64>,
        Array1<f64>,
        Vec<Array2<f64>>,
        Vec<Array1<f64>>,
        Vec<Array2<f64>>,
        Vec<Array2<f64>>,
        Vec<Array2<f64>>,
    ) {
        // reset old data
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut() {
            esd_pair.properties.reset();
        }
        self.properties.reset();
        // ground state energy and gradient
        self.prepare_scc();
        let gs_energy = self.run_scc().unwrap();
        let gs_gradient = self.ground_state_gradient();

        // mutable gradient
        let mut gradient: Array1<f64> = gs_gradient;

        // calculate excited states
        let (diabatic_hamiltonian): (Array2<f64>) =
            self.create_diabatic_hamiltonian();
        // get the basis states
        let states = self.properties.basis_states().unwrap();

        // get cis matrices
        let mut cis_vec: Vec<Array2<f64>> = Vec::new();
        let mut qtrans_vec: Vec<Array1<f64>> = Vec::new();
        let mut mo_vec: Vec<Array2<f64>> = Vec::new();
        let mut h_vec:Vec<Array2<f64>> = Vec::new();
        let mut x_vec:Vec<Array2<f64>> = Vec::new();

        // iterate over states
        for (idx, state) in states.iter().enumerate() {
            match state {
                ReducedBasisState::LE(ref a) => {
                    let tdm: ArrayView2<f64> = self.monomers[a.monomer_index]
                        .properties
                        .tdm(a.state_index)
                        .unwrap();
                    let ci_coeff: ArrayView1<f64> = self.monomers[a.monomer_index]
                        .properties
                        .ci_coefficient(a.state_index)
                        .unwrap();
                    let q_ov: ArrayView2<f64> =
                        self.monomers[a.monomer_index].properties.q_ov().unwrap();
                    cis_vec.push(tdm.to_owned());
                    qtrans_vec.push(q_ov.dot(&ci_coeff));
                    mo_vec.push(
                        self.monomers[a.monomer_index]
                            .properties
                            .orbs()
                            .unwrap()
                            .to_owned(),
                    );
                    h_vec.push(self.monomers[a.monomer_index]
                                   .properties
                                   .h_coul_transformed()
                                   .unwrap()
                                   .to_owned());
                    x_vec.push(self.monomers[a.monomer_index]
                        .properties
                        .x()
                        .unwrap()
                        .to_owned());
                }
                ReducedBasisState::CT(ref a) => {}
            }
        }

        // // iterate over states
        // for (idx,state) in states.iter().enumerate(){
        //     if state_coefficients[idx] > thresh{
        //         match state{
        //             ReducedBasisState::LE(ref a) => {
        //                 let grad:Array1<f64> = self.exciton_le_gradient_without_davidson(a.monomer_index,a.state_index);
        //                 let mol:&Monomer = &self.monomers[a.monomer_index];
        //                 gradient.slice_mut(s![mol.slice.grad]).assign(&grad);
        //             },
        //             ReducedBasisState::CT(ref a) => {
        //                 let mut hole_i:bool = true;
        //                 let mut monomer_i:usize = 0;
        //                 let mut monomer_j:usize = 1;
        //                 let mut ct_i:usize = 0;
        //                 let mut ct_j:usize = 0;
        //
        //                 if a.hole.m_index < a.electron.m_index{
        //                     monomer_i = a.hole.m_index;
        //                     ct_i = a.hole.ct_index;
        //                     monomer_j = a.electron.m_index;
        //                     ct_j = a.electron.ct_index;
        //                 }
        //                 else{
        //                     hole_i = false;
        //                     monomer_i = a.electron.m_index;
        //                     ct_i = a.electron.ct_index;
        //                     monomer_j = a.hole.m_index;
        //                     ct_j = a.hole.ct_index;
        //                 }
        //
        //                 let mut grad:Array1<f64> =
        //                     self.ct_gradient_new(
        //                         monomer_i,
        //                         monomer_j,
        //                         ct_i,
        //                         ct_j,
        //                         a.energy,hole_i
        //                     );
        //
        //                 grad = grad +
        //                     self.calculate_cphf_correction(
        //                         monomer_i,
        //                         monomer_j,
        //                         ct_i,
        //                         ct_j,
        //                         hole_i
        //                     );
        //
        //                 let mol_i = &self.monomers[monomer_i];
        //                 let mol_j = &self.monomers[monomer_j];
        //                 gradient.slice_mut(s![mol_i.slice.grad]).assign(&grad.slice(s![..mol_i.n_atoms*3]));
        //                 gradient.slice_mut(s![mol_j.slice.grad]).assign(&grad.slice(s![mol_i.n_atoms*3..]));
        //             },
        //         }
        //     }
        // }

        return (
            gs_energy,
            diabatic_hamiltonian,
            gradient,
            cis_vec,
            qtrans_vec,
            mo_vec,
            h_vec,
            x_vec
        );
    }
}
