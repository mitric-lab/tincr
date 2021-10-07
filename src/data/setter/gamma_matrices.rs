use ndarray::prelude::*;
use crate::data::{Storage, GammaData};


impl<'a> Storage<'a> {
    /// Set the Gamma matrix in atom basis.
    pub fn set_gamma(&mut self, gamma: Array2<f64>) {
        self.gammas.gamma = Some(gamma);
    }

    /// Set the Gamma matrix in AO basis.
    pub fn set_gamma_ao(&mut self, gamma_ao: Array2<f64>) {
        self.gammas.gamma_ao = Some(gamma_ao);
    }

    /// Set the screened Gamma matrix in atom basis.
    pub fn set_gamma_lr(&mut self, gamma_lr: Array2<f64>) {
        self.gammas.gamma_lr = Some(gamma_lr);
    }

    /// Set the screened Gamma matrix in AO basis.
    pub fn set_gamma_lr_ao(&mut self, gamma_lr_ao: Array2<f64>) {
        self.gammas.gamma_lr_ao = Some(gamma_lr_ao);
    }
}