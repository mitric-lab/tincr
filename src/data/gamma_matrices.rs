use ndarray::prelude::*;
use crate::data::{Storage, GammaData};

impl<'a> GammaData<'a> {
    pub fn new() -> Self {
        GammaData {
            gammafunction: None,
            gamma_function_lc: None,
            gamma: None,
            gamma_lr: None,
            gamma_ao: None,
            gamma_lr_ao: None,
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

impl<'a> Storage<'a> {
    /// Check if the gamma matrix is set.
    pub fn gamma_is_set(&self) -> bool {
        self.gammas.gamma.is_some()
    }
}