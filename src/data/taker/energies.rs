use ndarray::prelude::*;
use crate::data::{Storage, EnergyData};

impl<'a> Storage<'a> {
    pub fn take_total_energy(&mut self) -> f64 {
        self.energies.total_energy.take().expect("EnergyData:total_energy: The total energy was not set")
    }

    pub fn take_last_energy(&mut self) -> f64 {
        self.energies.last_energy.take().expect("EnergyData:last_energy: The last energy was not set")
    }
}
