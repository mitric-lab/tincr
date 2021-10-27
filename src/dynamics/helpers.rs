use crate::initialization::System;
use ndarray::prelude::*;
use crate::scc::scc_routine::RestrictedSCC;
use crate::fmo::SuperSystem;

impl System{
    pub fn calculate_energies_and_gradient(&mut self, state:usize)->(Array1<f64>,Array1<f64>){
        let mut gradient: Array1<f64> = Array::zeros(3 * self.n_atoms);
        let mut energies:Array1<f64> = Array1::zeros(self.config.excited.nstates+1);

        if state == 0{
            // ground state energy
            self.prepare_scc();
            let gs_energy:f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            // // calculate excited states
            // self.prepare_tda();
            // self.run_tda(self.config.excited.nstates, 100, 1e-4);
            // let ci_energies:ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            // energies.slice_mut(s![1..]).assign(&(gs_energy +&ci_energies));

            gradient = self.ground_state_gradient(true);
        }
        else{
            // ground state energy
            let excited_state:usize = state -1;
            self.prepare_scc();
            let gs_energy:f64 = self.run_scc().unwrap();
            energies[0] = gs_energy;

            // calculate excited states
            self.prepare_tda();
            self.run_tda(self.config.excited.nstates, 200, 1e-4);
            let ci_energies:ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            energies.slice_mut(s![1..]).assign(&(gs_energy +&ci_energies));

            gradient = self.ground_state_gradient(true);

            self.prepare_excited_grad();

            if self.config.lc.long_range_correction{
                gradient = gradient + self.tda_gradient_lc(excited_state);
            }
            else{
                gradient = gradient + self.tda_gradient_nolc(excited_state);
            }
        }
        self.properties.reset();

        return (energies,gradient);
    }
}

impl SuperSystem{
    pub fn calculate_energies_and_gradient(&mut self, state:usize)->(Array1<f64>,Array1<f64>){
        let n_atoms:usize = self.atoms.len();
        let mut gradient: Array1<f64> = Array::zeros(3 * n_atoms);
        let n_states:usize = self.config.excited.nstates+1;
        let mut energies:Array1<f64> = Array1::zeros(n_states);

        if state == 0{
            // ground state energy and gradient
            self.prepare_scc();
            let gs_energy = self.run_scc().unwrap();
            energies[0] = gs_energy;
            gradient = self.ground_state_gradient();

            // // calculate excited states
            // self.create_exciton_hamiltonian();
            // let ex_energies:ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            // energies.slice_mut(s![1..]).assign(&(gs_energy +&ex_energies));
        }
        else{
            // ground state energy and gradient
            self.prepare_scc();
            let gs_energy = self.run_scc().unwrap();
            energies[0] = gs_energy;
            gradient = self.ground_state_gradient();

            // calculate excited states
            self.create_exciton_hamiltonian();
            let ex_energies:ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
            energies.slice_mut(s![1..]).assign(&(gs_energy +&ex_energies.slice(s![..n_states-1])));

            let excited_state:usize = state -1;
            gradient = gradient + self.fmo_cis_gradient(excited_state);
        }

        for monomer in self.monomers.iter_mut(){
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.properties.reset();

        return (energies,gradient);
    }
}