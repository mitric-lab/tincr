use ndarray::prelude::*;
use crate::excited_states::ProductCache;
use crate::data::{Storage, ExcitedStateData};


impl<'a> Storage<'a> {
    /// Setter function for the omega.
    pub fn set_omega(&mut self, omega: Array1<f64>) {
        self.excited_states.omega = Some(omega);
    }

    /// Setter function for the cis_eigenvalues.
    pub fn set_cis_eigenvalues(&mut self, cis_eigenvalues: Array1<f64>) {
        self.excited_states.cis_eigenvalues = Some(cis_eigenvalues);
    }

    /// Setter function for the cis_coefficients.
    pub fn set_cis_coefficients(&mut self, cis_coefficients: Array2<f64>) {
        self.excited_states.x_plus_y= Some(cis_coefficients);
    }

    /// Setter function for X+Y coefficients.
    pub fn set_x_plus_y(&mut self, x_plus_y: Array2<f64>) {
        self.excited_states.x_plus_y= Some(x_plus_y);
    }

    /// Setter function for X-Y coefficients.
    pub fn set_x_minus_y(&mut self, x_minus_y: Array2<f64>) {
        self.excited_states.x_minus_y= Some(x_minus_y);
    }

    /// Setter function for the tr_dipoles.
    pub fn set_tr_dipoles(&mut self, tr_dipoles: Array2<f64>) {
        self.excited_states.tr_dipoles = Some(tr_dipoles);
    }

    /// Setter function for the osc_strengths.
    pub fn set_osc_strengths(&mut self, osc_strengths: Array1<f64>) {
        self.excited_states.osc_strengths = Some(osc_strengths);
    }

    /// Setter function for the cache.
    pub fn set_cache(&mut self, cache: ProductCache) {
        self.excited_states.cache = Some(cache);
    }

    /// Setter function for the z_vector.
    pub fn set_z_vector(&mut self, z_vector: Array1<f64>) {
        self.excited_states.z_vector = Some(z_vector);
    }
}