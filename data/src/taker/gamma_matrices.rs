use ndarray::prelude::*;
use crate::data::{Storage, GammaData};

impl<'a> Storage<'a> {
    pub fn take_gamma(&mut self) -> Array2<f64> {
        self.gammas.gamma.take().expect("ExcitedStateData:gamma; Gamma matrix was not set.")
    }

    pub fn take_gamma_lr(&mut self) -> Array2<f64> {
        self.gammas.gamma_lr.take().expect("ExcitedStateData:gamma_lr; Gamma matrix (with linear response) was not set.")
    }

    pub fn take_gamma_ao(&mut self) -> Array2<f64> {
        self.gammas.gamma_ao.take().expect("ExcitedStateData:gamma_ao; Gamma matrix (in AO basis) was not set.")
    }

    pub fn take_gamma_lr_ao(&mut self) -> Array2<f64> {
        self.gammas.gamma_lr_ao.take().expect("ExcitedStateData:gamma_lr_ao; Gamma matrix (with linear response) was not set.")
    }
}
