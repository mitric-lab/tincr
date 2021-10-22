use ndarray::prelude::*;
use crate::data::{Storage, SpatialOrbitals};

/// Type that characterizes if a MO is occupied with 1 or 2 electrons.
#[derive(Clone, Copy)]
pub enum OrbType {
    Unrestricted = 1,
    Restricted = 2,
}


impl SpatialOrbitals {
    /// Constructor.
    pub fn new(n_orbs: usize, n_elec: usize, occ: OrbType) -> Self {
        // Number of occupied orbitals.
        let n_occ: usize = n_elec / occ as usize;

        // Create a list with the indices of the occupied and virtual orbitals.
        let occ_indices: Vec<usize> = (0..n_occ).collect();
        let virt_indices: Vec<usize> = (n_occ..n_orbs()).collect();

        // And create a list with occupation pattern for the orbitals.
        let occupation: Vec<f64> = (0..n_occ).map(|_o| occ as f64)
            .chain((n_occ..n_orbs()).map(|_v| 0.0)).collect();


        Self {
            orbs: None,
            orbe: None,
            n_elec,
            occupation: Some(occupation),
            occ_indices: Some(occ_indices),
            virt_indices: Some(virt_indices),
            active_occ: None,
            active_virt: None,
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
