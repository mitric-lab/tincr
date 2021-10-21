use rusty_fish::interface::QuantumChemistryInterface;
use crate::initialization::System;
use ndarray::prelude::*;
use crate::fmo::SuperSystem;

impl QuantumChemistryInterface for System{
    fn compute_data(
        &mut self,
        coordinates:ArrayView2<f64>,
        state:usize
    )->(
        Array1<f64>,
        Array2<f64>,
        Array3<f64>,
        Array3<f64>)
    {
        //Return enegies, forces, non-adiabtic coupling and the transition dipole
        // update the coordinats of the system
        self.update_xyz(coordinates.to_owned().into_shape(3*self.n_atoms).unwrap());
        // calculate the energy and the gradient of the state
        let (energies,gradient):(Array1<f64>,Array1<f64>) = self.calculate_energies_and_gradient(state);
        let gradient:Array2<f64> = gradient.into_shape([self.n_atoms,3]).unwrap();

        // placeholder arrays for NAD-coupling and transition dipoles
        let tmp_1:Array3<f64> = Array3::zeros((1,1,1));
        let tmp_2:Array3<f64> = Array3::zeros((1,1,1));

        return (energies,gradient,tmp_1,tmp_2);
    }
}

impl QuantumChemistryInterface for SuperSystem{
    fn compute_data(
        &mut self,
        coordinates:ArrayView2<f64>,
        state:usize
    )->(
        Array1<f64>,
        Array2<f64>,
        Array3<f64>,
        Array3<f64>)
    {
        //Return enegies, forces, non-adiabtic coupling and the transition dipole#
        let n_atoms:usize = self.atoms.len();
        // update the coordinats of the system
        self.update_xyz(coordinates.into_shape(3*n_atoms).unwrap().to_owned());
        // calculate the energy and the gradient of the state
        let (energies,gradient):(Array1<f64>,Array1<f64>) = self.calculate_energies_and_gradient(state);
        let gradient:Array2<f64> = gradient.into_shape([n_atoms,3]).unwrap();

        // placeholder arrays for NAD-coupling and transition dipoles
        let tmp_1:Array3<f64> = Array3::zeros((1,1,1));
        let tmp_2:Array3<f64> = Array3::zeros((1,1,1));

        return (energies,gradient,tmp_1,tmp_2);
    }
}