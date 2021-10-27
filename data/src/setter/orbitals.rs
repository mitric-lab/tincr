use ndarray::prelude::*;
use crate::data::{SpatialOrbitals, Storage};


impl<'a> Storage<'a> {
    /// Set MO coefficients.
    pub fn set_orbs(&mut self, orbs: Array2<f64>) {
        self.orbitals.orbs = Some(orbs);
    }

    /// Set MO energies.
    pub fn set_orbe(&mut self, orbe: Array1<f64>) {
        self.orbitals.orbe = Some(orbe);
    }

    /// Set MO occupation numbers.
    pub fn set_occupation(&mut self, occupation: Vec<f64>) {
        self.orbitals.occupation = Some(occupation);
    }

    /// Set indices of occupied MOs.
    pub fn set_occ_indices(&mut self, occ_indices: Vec<usize>) {
        self.orbitals.occ_indices = Some(occ_indices);
    }

    /// Set indices of virtual MOs.
    pub fn set_virt_indices(&mut self, virt_indices: Vec<usize>) {
        self.orbitals.virt_indices = Some(virt_indices);
    }

    /// Set density matrix in AO basis.
    pub fn set_p(&mut self, p: Array2<f64>) {
        self.orbitals.p = Some(p);
    }

    /// Set density matrix in AO basis if it was not set before.
    pub fn set_if_unset_p(&mut self, p: Array2<f64>) {
        self.orbitals.p.get_or_insert(p);
    }

    /// Set reference density matrix in AO basis.
    pub fn set_p_ref(&mut self, p_ref: Array2<f64>) {
        self.orbitals.p_ref = Some(p_ref);
    }

    /// Set reference density matrix in AO basis if it was not set before.
    pub fn set_if_unset_p_ref(&mut self, p_ref: Array2<f64>) {
        self.orbitals.p_ref.get_or_insert(p_ref);
    }


    /// Set difference between the density matrix and the reference density matrix.
    pub fn set_delta_p(&mut self, delta_p: Array2<f64>) {
        self.orbitals.delta_p = Some(delta_p);
    }

    /// Set density matrix of alpha electrons in AO basis.
    pub fn set_p_alpha(&mut self, p_alpha: Array2<f64>) {
        self.orbitals.p_alpha = Some(p_alpha);
    }

    /// Set density matrix of alpha electrons in AO basis if it was not set before.
    pub fn set_if_unset_p_alpha(&mut self, p_alpha: Array2<f64>) {
        self.orbitals.p_alpha.get_or_insert(p_alpha);
    }

    /// Set density matrix of beta electrons in AO basis.
    pub fn set_p_beta(&mut self, p_beta: Array2<f64>) {
        self.orbitals.p_beta = Some(p_beta);
    }

    /// Set density matrix of beta electrons in AO basis if it was not set before.
    pub fn set_if_unset_p_beta(&mut self, p_beta: Array2<f64>) {
        self.orbitals.p_beta.get_or_insert(p_beta);
    }
}

