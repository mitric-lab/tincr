use ndarray::prelude::*;
use crate::excited_states::ProductCache;
use crate::data::{Storage, ExcitedStateData};


impl<'a> Storage<'a> {

    /// Take the energy differences between occupieds and virtuals and return them.
    pub fn take_omega(&mut self) -> Array1<f64> {
        self.excited_states.omega.take().expect("ExcitedStateData:omega; The energy differences were not set.")
    }

    /// Take the eigenvalues of the excited states and return them.
    pub fn take_cis_eigenvalues(&mut self) -> Array1<f64> {
        self.excited_states.cis_eigenvalues.take().expect("ExcitedStateData:cis_eigenvalues; The eigenvalues of the excited states were not set.")
    }

    /// Take the eigenvectors of the excited states and return them.
    pub fn take_x_plus_y(&mut self) -> Array2<f64> {
        self.excited_states.x_plus_y.take().expect("ExcitedStateData:x_plus_y; X+Y of the excited states were not set.")
    }

    /// Take the eigenvectors of the excited states and return them.
    pub fn take_x_minus_y(&mut self) -> Array2<f64> {
        self.excited_states.x_minus_y.take().expect("ExcitedStateData:x_minus_y; The eigenvectors of the excited states were not set.")
    }

    /// Take the eigenvectors of the excited states and return them.
    pub fn take_cis_coefficients(&mut self) -> Array2<f64> {
        self.excited_states.x_plus_y.take().expect("ExcitedStateData:cis_coefficients; The eigenvectors of the excited states were not set.")
    }

    /// Take the transition dipole moments between the ground state and the excited states and return them.
    pub fn take_tr_dipoles(&mut self) -> Array2<f64> {
        self.excited_states.tr_dipoles.take().expect("ExcitedStateData:tr_dipoles; The transition dipole moments between ground and excited states were not set.")
    }

    /// Take the oscillator strengths between the ground state and the excited states and return them.
    pub fn take_osc_strengths(&mut self) -> Array1<f64> {
        self.excited_states.osc_strengths.take().expect("ExcitedStateData:osc_strengths; The oscillator strengths between ground and excited states were not set.")
    }

    /// Take the product cache and return it.
    pub fn take_cache(&mut self) -> ProductCache {
        self.excited_states.cache.take().expect("ExcitedStateData:cache; The product cache was not set.")
    }

    /// Take the z-vector and return it.
    pub fn take_z_vector(&mut self) -> Array1<f64> {
        self.excited_states.z_vector.take().expect("ExcitedStateData:z_vector; The z-vector was not set.")
    }
}

