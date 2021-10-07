use ndarray::prelude::*;
use crate::data::{OrbitalData, Storage};

impl OrbitalData {
    /// Constructor.
    pub fn new() -> Self {
        Self {
            orbs: None,
            orbe: None,
            occupation: None,
            occ_indices: None,
            virt_indices: None,
            p: None,
            p_ref: None,
            delta_p: None,
            p_alpha: None,
            p_beta: None,
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Returns the number of occupied orbitals.
    pub fn n_occ(&self) -> usize {
        match &self.occ_indices {
            Some(occs) => occs.len(),
            None => panic!("OrbitalData:n_occ; Indices of occupied orbitals are not set."),
        }
    }

    /// Returns the number of virtual orbitals.
    pub fn n_virt(&self) -> usize {
        match &self.virt_indices {
            Some(virts) => virts.len(),
            None => panic!("OrbitalData:n_virt; Indices of virtual orbitals are not set"),
        }
    }

    /// Returns the index of the HOMO.
    pub fn homo(&self) -> usize {
        self.n_occ() - 1
    }

    /// Returns the index of he LUMO.
    pub fn lumo(&self) -> usize {
        self.n_occ()
    }

}
