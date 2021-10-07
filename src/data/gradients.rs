use crate::data::{GradData, Storage};
use ndarray::prelude::*;

impl GradData {
    pub fn new() -> Self {
        Self {
            s: None,
            h0: None,
            dq: None,
            dq_diag: None,
            gamma: None,
            gamma_lr: None,
            gamma_ao: None,
            gamma_lr_ao: None
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

