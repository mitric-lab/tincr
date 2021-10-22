use crate::data::{GradData, Storage};
use ndarray::prelude::*;

impl<'a> Storage<'a> {
    pub fn take_grad_s(&mut self) -> Array3<f64> {
        self.gradients.s.take().expect("take_grad_s: GradData.s has not been set")
    }

    pub fn take_grad_h0(&mut self) -> Array3<f64> {
        self.gradients.h0.take().expect("take_grad_h0: GradData.h0 has not been set")
    }

    pub fn take_grad_dq(&mut self) -> Array2<f64> {
        self.gradients.dq.take().expect("take_grad_dq: GradData.dq has not been set")
    }

    pub fn take_grad_dq_diag(&mut self) -> Array1<f64> {
        self.gradients.dq_diag.take().expect("take_grad_dq_diag: GradData.dq_diag has not been set")
    }

    pub fn take_grad_gamma(&mut self) -> Array3<f64> {
        self.gradients.gamma.take().expect("take_grad_gamma: GradData.gamma has not been set")
    }

    pub fn take_grad_gamma_lr(&mut self) -> Array3<f64> {
        self.gradients.gamma_lr.take().expect("take_grad_gamma_lr: GradData.gamma_lr has not been set")
    }

    pub fn take_grad_gamma_ao(&mut self) -> Array3<f64> {
        self.gradients.gamma_ao.take().expect("take_grad_gamma_ao: GradData.gamma_ao has not been set")
    }

    pub fn take_grad_gamma_lr_ao(&mut self) -> Array3<f64> {
        self.gradients.gamma_lr_ao.take().expect("take_grad_gamma_lr_ao: GradData.gamma_lr_ao has not been set")
    }
}