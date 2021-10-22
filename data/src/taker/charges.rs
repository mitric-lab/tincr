use ndarray::prelude::*;
use crate::data::{Storage, ChargeData};

impl<'a> Storage<'a> {
    /// Take the partial charges and return them.
    pub fn take_dq(&mut self) -> Array1<f64> {
        self.charges.dq.take().expect("ChargeData:dq; Partial charges were not set.")
    }

    /// Take the charge differences and return them.
    pub fn take_delta_dq(&mut self) -> Array1<f64> {
        self.charges.delta_dq
            .take()
            .expect("ChargeData:delta_dq; Charge differences were not set.")
    }

    /// Take the partial charges in AO basisand return them.
    pub fn take_q_ao(&mut self) -> Array1<f64> {
        self.charges.q_ao.take().expect("ChargeData:q_ao; Partial charges in AO basis were not set.")
    }


    /// Take the transition charges between occupied and virtual orbitals and return them.
    pub fn take_q_ov(&mut self) -> Array2<f64> {
        self.charges.q_ov
            .take()
            .expect("ChargeData:q_ov; Transition charges between occupied and virtual orbitals were not set.")
    }

    /// Take the transition charges between occupied and occupied orbitals and return them.
    pub fn take_q_oo(&mut self) -> Array2<f64> {
        self.charges.q_oo
            .take()
            .expect("ChargeData:q_oo; Transition charges between occupied and occupied orbitals were not set.")
    }

    /// Take the transition charges between virtual and virtual orbitals and return them.
    pub fn take_q_vv(&mut self) -> Array2<f64> {
        self.charges.q_vv
            .take()
            .expect("ChargeData:q_vv; Transition charges between virtual and virtual orbitals were not set.")
    }

    /// Take the transition charges between virtual and occupied orbitals and return them.
    pub fn take_q_vo(&mut self) -> Array2<f64> {
        self.charges.q_vo
            .take()
            .expect("ChargeData:q_vo; Transition charges between virtual and occupied orbitals were not set.")
    }

    /// Take the transition charges for excited states and return them.
    pub fn take_q_trans(&mut self) -> Array2<f64> {
        self.charges.q_trans
            .take()
            .expect("ChargeData:q_trans; Transition charges for excited states were not set.")
    }

    /// Take the partial charges from the electrostatic potential of other monomers and return them.
    pub fn take_esp_q(&mut self) -> Array1<f64> {
        self.charges.esp_q
            .take()
            .expect("ChargeData:esp_q; Partial charges from the electrostatic potential of other monomers were not set.")
    }

    /// Take the partial charges corresponding to alpha electrons and return them.
    pub fn take_dq_alpha(&mut self) -> Array1<f64> {
        self.charges.dq_alpha
            .take()
            .expect("ChargeData:dq_alpha; Partial charges corresponding to alpha electrons were not set.")
    }

    /// Take the partial charges corresponding to beta electrons and return them.
    pub fn take_dq_beta(&mut self) -> Array1<f64> {
        self.charges.dq_beta
            .take()
            .expect("ChargeData:dq_beta; Partial charges corresponding to beta electrons were not set.")
    }
}
