use ndarray::prelude::*;
use crate::data::{Storage, GammaData};


impl<'a> Storage<'a> {
    /// Check if the gamma matrix is set.
    pub fn gamma_is_set(&self) -> bool {
        self.gammas.gamma.is_some()
    }
}