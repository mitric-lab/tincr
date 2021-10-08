use ndarray::prelude::*;
use crate::data::{Storage, ChargeData};


impl ChargeData {
    /// Constructor.
    pub fn new() -> Self {
        Self {
            dq: None,
            delta_dq: None,
            q_ao: None,
            q_ov: None,
            q_oo: None,
            q_vv: None,
            q_vo: None,
            q_trans: None,
            esp_q: None,
            dq_alpha: None,
            dq_beta: None,
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

impl<'a> Storage<'a> {
    /// Check if the o-v charges are set.
    pub fn q_ov_is_set(&self) -> bool {
        self.charges.q_ov.is_some()
    }
}