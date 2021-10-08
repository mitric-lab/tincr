use ndarray::prelude::*;
use crate::data::{Storage, ChargeData};

impl<'a> Storage<'a> {
    /// Set partial charges.
    pub fn set_dq(&mut self, dq: Array1<f64>) {
        self.charges.dq = Some(dq);
    }

    pub fn set_if_unset_dq(&mut self, dq: Array1<f64>) {
        self.charges.dq.get_or_insert(dq);
    }

    /// Set charge differences between charges of a molecular dimer and the corresponding monomers.
    pub fn set_delta_dq(&mut self, delta_dq: Array1<f64>) {
        self.charges.delta_dq = Some(delta_dq);
    }

    /// Set the partial in the basis of the AOs.
    pub fn set_q_ao(&mut self, q_ao: Array1<f64>) {
        self.charges.q_ao = Some(q_ao);
    }

    /// Set the partial charges in the basis of the AOs if they were not set yet.
    pub fn set_if_unset_q_ao(&mut self, q_ao: Array1<f64>) {
        self.charges.q_ao.get_or_insert(q_ao);
    }

    /// Set transition charges between occupied and virtual orbitals.
    pub fn set_q_ov(&mut self, q_ov: Array2<f64>) {
        self.charges.q_ov = Some(q_ov);
    }

    /// Set transition charges between occupied and occupied orbitals.
    pub fn set_q_oo(&mut self, q_oo: Array2<f64>) {
        self.charges.q_oo = Some(q_oo);
    }

    /// Set transition charges between virtual and virtual orbitals.
    pub fn set_q_vv(&mut self, q_vv: Array2<f64>) {
        self.charges.q_vv = Some(q_vv);
    }

    /// Set transition charges between virtual and occupied orbitals.
    pub fn set_q_vo(&mut self, q_vo: Array2<f64>) {
        self.charges.q_vo = Some(q_vo);
    }

    /// Set transition charges for excited states.
    pub fn set_q_trans(&mut self, q_trans: Array2<f64>) {
        self.charges.q_trans = Some(q_trans);
    }

    /// Set partial charges from the electrostatic potential of other monomers.
    pub fn set_esp_q(&mut self, esp_q: Array1<f64>) {
        self.charges.esp_q = Some(esp_q);
    }

    /// Set partial charges corresponding to alpha electrons.
    pub fn set_dq_alpha(&mut self, dq_alpha: Array1<f64>) {
        self.charges.dq_alpha = Some(dq_alpha);
    }

    /// Set partial charges corresponding to alpha electrons if they were not set before.
    pub fn set_if_unset_dq_alpha(&mut self, dq_alpha: Array1<f64>) {
        self.charges.dq_alpha.get_or_insert(dq_alpha);
    }

    /// Set partial charges corresponding to beta electrons.
    pub fn set_dq_beta(&mut self, dq_beta: Array1<f64>) {
        self.charges.dq_beta = Some(dq_beta);
    }

    /// Set partial charges corresponding to beta electrons if they were not set before.
    pub fn set_if_unset_dq_beta(&mut self, dq_beta: Array1<f64>) {
        self.charges.dq_beta.get_or_insert(dq_beta);
    }

}