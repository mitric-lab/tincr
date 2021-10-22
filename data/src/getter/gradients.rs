use crate::data::{GradData, Storage};
use ndarray::prelude::*;



impl<'a> Storage<'a> {
    pub fn grad_s(&self) -> ArrayView3<f64> {
        match &self.gradients.s {
            Some(ref value) => value.view(),
            None => panic!("GradData::s; The overlap matrix is not set!"),
        }
    }

    pub fn grad_h0(&self) -> ArrayView3<f64> {
        match &self.gradients.h0 {
            Some(ref value) => value.view(),
            None => panic!("GradData::h0; The one-electron integrals are not set!"),
        }
    }

    pub fn grad_dq(&self) -> ArrayView2<f64> {
        match &self.gradients.dq {
            Some(ref value) => value.view(),
            None => panic!("GradData::dq; The partial charges are not set!"),
        }
    }

    pub fn grad_dq_diag(&self) -> ArrayView1<f64> {
        match &self.gradients.dq_diag {
            Some(ref value) => value.view(),
            None => panic!("GradData::dq_diag; The diagonal elements of the partial charges are not set!"),
        }
    }

    pub fn grad_gamma(&self) -> ArrayView3<f64> {
        match &self.gradients.gamma {
            Some(ref value) => value.view(),
            None => panic!("GradData::gamma; The unscreened gamma matrix is not set!"),
        }
    }

    pub fn grad_gamma_lr(&self) -> ArrayView3<f64> {
        match &self.gradients.gamma_lr {
            Some(ref value) => value.view(),
            None => panic!("GradData::gamma_lr; The screened gamma matrix is not set!"),
        }
    }

    pub fn grad_gamma_ao(&self) -> ArrayView3<f64> {
        match &self.gradients.gamma_ao {
            Some(ref value) => value.view(),
            None => panic!("GradData::gamma_ao; The unscreened gamma matrix in AO basis is not set!"),
        }
    }

    pub fn grad_gamma_lr_ao(&self) -> ArrayView3<f64> {
        match &self.gradients.gamma_lr_ao {
            Some(ref value) => value.view(),
            None => panic!("GradData::gamma_lr_ao; The screened gamma matrix in AO basis is not set!"),
        }
    }
}



