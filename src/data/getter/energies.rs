use ndarray::prelude::*;
use crate::data::{Storage, EnergyData};

impl<'a> Storage<'a> {
    pub fn total_energy(&self) -> f64 {
        match &self.energies.total_energy {
            Some(value) => *value,
            None => panic!("EnergyData::total_energy; The total energy is not set!"),
        }
    }

    pub fn last_energy(&self) -> f64 {
        match &self.energies.last_energy {
            Some(value) => *value,
            None => panic!("EnergyData::last_energy; The last energy is not set!"),
        }
    }
}


