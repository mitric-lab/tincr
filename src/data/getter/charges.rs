use ndarray::prelude::*;
use crate::data::{Storage, ChargeData};


impl<'a> Storage<'a> {
    pub fn dq(&self) -> ArrayView1<f64> {
        match &self.charges.dq {
            Some(value) => value.view(),
            None => panic!("ChargeData::dq; The partial charges are not set!"),
        }
    }

    pub fn delta_dq(&self) -> ArrayView1<f64> {
        match &self.charges.delta_dq {
            Some(value) => value.view(),
            None => panic!("ChargeData::delta_dq; The charge differences are not set!"),
        }
    }

    pub fn q_ov(&self) -> ArrayView2<f64> {
        match &self.charges.q_ov {
            Some(value) => value.view(),
            None => panic!("ChargeData::q_ov; The transition charges are not set!"),
        }
    }

    pub fn q_oo(&self) -> ArrayView2<f64> {
        match &self.charges.q_oo {
            Some(value) => value.view(),
            None => panic!("ChargeData::q_oo; The transition charges are not set!"),
        }
    }

    pub fn q_vv(&self) -> ArrayView2<f64> {
        match &self.charges.q_vv {
            Some(value) => value.view(),
            None => panic!("ChargeData::q_vv; The transition charges are not set!"),
        }
    }

    pub fn q_vo(&self) -> ArrayView2<f64> {
        match &self.charges.q_vo {
            Some(value) => value.view(),
            None => panic!("ChargeData::q_vo; The transition charges are not set!"),
        }
    }

    pub fn q_trans(&self) -> ArrayView2<f64> {
        match &self.charges.q_trans {
            Some(value) => value.view(),
            None => panic!("ChargeData::q_trans; The transition charges are not set!"),
        }
    }

    pub fn esp_q(&self) -> ArrayView1<f64> {
        match &self.charges.esp_q {
            Some(value) => value.view(),
            None => panic!("ChargeData::esp_q; The partial charges are not set!"),
        }
    }

    pub fn dq_alpha(&self) -> ArrayView1<f64> {
        match &self.charges.dq_alpha {
            Some(value) => value.view(),
            None => panic!("ChargeData::dq_alpha; The partial charges are not set!"),
        }
    }

    pub fn dq_beta(&self) -> ArrayView1<f64> {
        match &self.charges.dq_beta {
            Some(value) => value.view(),
            None => panic!("ChargeData::dq_beta; The partial charges are not set!"),
        }
    }
}


