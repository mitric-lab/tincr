use ndarray::prelude::*;
use crate::data::{Storage, EnergyData};

impl<'a> Storage<'a> {
    /// Setter function for total_energy.
    pub fn set_total_energy(&mut self, total_energy: f64) {
        self.energies.total_energy = Some(total_energy);
    }

    /// Setter function for last_energy.
    pub fn set_last_energy(&mut self, last_energy: f64) {
        self.energies.last_energy = Some(last_energy);
    }
}

