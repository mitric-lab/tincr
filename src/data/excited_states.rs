use ndarray::prelude::*;
use crate::excited_states::ProductCache;
use crate::data::{Storage, ExcitedStateData};

impl ExcitedStateData {
    /// Constructor that sets all fields to None.
    pub fn new() -> Self {
        Self {
            omega: None,
            cis_eigenvalues: None,
            x_plus_y: None,
            x_minus_y: None,
            tr_dipoles: None,
            osc_strengths: None,
            cache: None,
            z_vector: None,
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

impl<'a> Storage<'a> {
    /// Check if the orbital energy differences are set.
    pub fn omega_is_set(&self) -> bool {
        self.excited_states.omega.is_some()
    }
}