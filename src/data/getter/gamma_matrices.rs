use ndarray::prelude::*;
use crate::data::{Storage, GammaData};



impl<'a> Storage<'a> {
    pub fn gamma(&self) -> ArrayView2<f64> {
        match &self.gammas.gamma {
            Some(value) => value.view(),
            None => panic!("GammaData::gamma; The gamma matrix is not set!"),
        }
    }

    pub fn gamma_lr(&self) -> ArrayView2<f64> {
        match &self.gammas.gamma_lr {
            Some(value) => value.view(),
            None => panic!("GammaData::gamma_lr; The gamma matrix is not set!"),
        }
    }

    pub fn gamma_ao(&self) -> ArrayView2<f64> {
        match &self.gammas.gamma_ao {
            Some(value) => value.view(),
            None => panic!("GammaData::gamma_ao; The gamma matrix is not set!"),
        }
    }

    pub fn gamma_lr_ao(&self) -> ArrayView2<f64> {
        match &self.gammas.gamma_lr_ao {
            Some(value) => value.view(),
            None => panic!("GammaData::gamma_lr_ao; The gamma matrix is not set!"),
        }
    }
}


