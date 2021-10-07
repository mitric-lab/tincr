use ndarray::prelude::*;
use crate::data::{Storage, EnergyData};


impl EnergyData {
    pub fn new() -> Self {
        Self {
            total_energy: None,
            last_energy: None,
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}
