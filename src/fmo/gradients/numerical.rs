use crate::fmo::SuperSystem;
use ndarray::prelude::*;


//! These function do not provide numerical gradients. The intention for this file is to create
//! wrapper functions of the kind: `Fn(Array1<f64>) -> f64`, that take the coordinates of a molecule
//! and return a part of the FMO energy (e.g. monomer energy, pair energy, embedding energy...). This
//! should then allow the use of these functions for the generation of numerical gradients (using the
//! Ridder's method as implemented in [ridders_method](crate::gradients::numerical::ridders_method)).
//! In this way the analytic gradients can be tested.

impl SuperSystem{
    pub fn monomer_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {



    }



}
