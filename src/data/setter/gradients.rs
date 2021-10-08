use crate::data::{GradData, Storage};
use ndarray::prelude::*;


impl<'a> Storage<'a> {
    /// Set the gradient of the overlap matrix.
    pub fn set_grad_s(&mut self, s: Array3<f64>) {
        self.gradients.s = Some(s);
    }

    /// Set the gradient of the one-electron integrals in the Fock matrix.
    pub fn set_grad_h0(&mut self, h0: Array3<f64>) {
        self.gradients.h0 = Some(h0);
    }

    /// Set the gradient of the partial charges.
    pub fn set_grad_dq(&mut self, dq: Array2<f64>) {
        self.gradients.dq = Some(dq);
    }

    /// Set the diagonal elements of the gradient of the partial charges.
    pub fn set_grad_dq_diag(&mut self, dq_diag: Array1<f64>) {
        self.gradients.dq_diag = Some(dq_diag);
    }

    /// Set the gradient of the unscreened gamma matrix.
    pub fn set_grad_gamma(&mut self, gamma: Array3<f64>) {
        self.gradients.gamma = Some(gamma);
    }

    /// Set the gradient of the screened gamma matrix.
    pub fn set_grad_gamma_lr(&mut self, gamma_lr: Array3<f64>) {
        self.gradients.gamma_lr = Some(gamma_lr);
    }

    /// Set the gradient of the unscreened gamma matrix in AO basis.
    pub fn set_grad_gamma_ao(&mut self, gamma_ao: Array3<f64>) {
        self.gradients.gamma_ao = Some(gamma_ao);
    }

    /// Set the gradient of the screened gamma matrix in AO basis.
    pub fn set_grad_gamma_lr_ao(&mut self, gamma_lr_ao: Array3<f64>) {
        self.gradients.gamma_lr_ao = Some(gamma_lr_ao);
    }
}