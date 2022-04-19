use crate::fmo::SuperSystem;
use crate::initialization::System;
use ndarray::prelude::*;
use rusty_fish::interface::QuantumChemistryInterface;
use std::panic::resume_unwind;
use std::time::Instant;
use crate::RestrictedSCC;

impl QuantumChemistryInterface for System {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        state: usize,
    ) -> (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>) {
        //Return enegies, forces, non-adiabtic coupling and the transition dipole
        // update the coordinats of the system
        self.update_xyz(coordinates.to_owned().into_shape(3 * self.n_atoms).unwrap());
        // calculate the energy and the gradient of the state
        let (energies, gradient): (Array1<f64>, Array1<f64>) =
            self.calculate_energies_and_gradient(state);
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();

        // placeholder arrays for NAD-coupling and transition dipoles
        let tmp_1: Array3<f64> = Array3::zeros((1, 1, 1));
        let tmp_2: Array3<f64> = Array3::zeros((1, 1, 1));

        return (energies, gradient, tmp_1, tmp_2);
    }

    fn compute_ehrenfest(
        &mut self,
        coordinates: ArrayView2<f64>,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
        dt:f64,
    ) -> (
        f64,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Vec<Array1<f64>>,
        Vec<Array2<f64>>,
        Vec<Array2<f64>>,
        Vec<Array2<f64>>,
        Array1<f64>,
        Array1<f64>,
    ) {
        todo!()
    }
}

impl QuantumChemistryInterface for SuperSystem {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        state: usize,
    ) -> (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>) {
        // Return energies, forces, non-adiabtic coupling and the transition dipole#
        let n_atoms: usize = self.atoms.len();
        // update the coordinats of the system
        self.update_xyz(coordinates.into_shape(3 * n_atoms).unwrap().to_owned());
        // calculate the energy and the gradient of the state
        let (energies, gradient): (Array1<f64>, Array1<f64>) =
            self.calculate_energies_and_gradient(state);
        let gradient: Array2<f64> = gradient.into_shape([n_atoms, 3]).unwrap();

        // placeholder arrays for NAD-coupling and transition dipoles
        let tmp_1: Array3<f64> = Array3::zeros((1, 1, 1));
        let tmp_2: Array3<f64> = Array3::zeros((1, 1, 1));

        return (energies, gradient, tmp_1, tmp_2);
    }

    fn compute_ehrenfest(
        &mut self,
        coordinates: ArrayView2<f64>,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
        dt:f64,
    ) -> (
        f64,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Vec<Array1<f64>>,
        Vec<Array2<f64>>,
        Vec<Array2<f64>>,
        Vec<Array2<f64>>,
        Array1<f64>,
        Array1<f64>,
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

        // Return energies, forces and the nonadiabtic coupling
        let n_atoms: usize = self.atoms.len();
        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * n_atoms).unwrap().to_owned());

        // calculate the ground state energy
        self.prepare_scc();
        let gs_energy = self.run_scc().unwrap();

        // calculate diabatic hamiltonian
        let (diabatic_hamiltonian): (Array2<f64>) =
            self.create_diabatic_hamiltonian();

        // calculate the nonadiabatic coupling
        let (coupling,diabatic_hamiltonian,s,diag,signs):(Array2<f64>,Array2<f64>,Array2<f64>,Array1<f64>,Array1<f64>)
            = self.nonadiabatic_scalar_coupling(diabatic_hamiltonian.view(),dt);
        let mut s_vec:Vec<Array2<f64>> = Vec::new();
        s_vec.push(s);

        // calculate diabatic coupling and the gradient
        let (gradient, cis_vec, qtrans_vec, mo_vec,h_vec,x_vec): (
            Array1<f64>,
            Vec<Array2<f64>>,
            Vec<Array1<f64>>,
            Vec<Array2<f64>>,
            Vec<Array2<f64>>,
            Vec<Array2<f64>>,
        ) = self.calculate_ehrenfest_gradient(state_coefficients, thresh);

        // reshape the gradient
        let gradient: Array2<f64> = gradient.into_shape([n_atoms, 3]).unwrap();

        // reset the old supersystem
        self.properties.reset_supersystem();

        // create diabatic hamiltonian with dimension +1
        let dim:usize = diabatic_hamiltonian.dim().0+1;
        let mut new_diabatic:Array2<f64> = Array2::zeros([dim,dim]);
        new_diabatic.slice_mut(s![1..,1..]).assign(&diabatic_hamiltonian);

        return (
            gs_energy,
            gradient,
            new_diabatic,
            coupling,
            cis_vec,
            qtrans_vec,
            mo_vec,
            s_vec,
            x_vec,
            diag,
            signs,
        );
    }
}
