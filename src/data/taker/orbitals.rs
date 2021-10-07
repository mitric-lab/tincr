use ndarray::prelude::*;
use crate::data::{OrbitalData, Storage};


impl<'a> Storage<'a> {
    /// Take the MO coefficients and return them.
    pub fn take_orbs(&mut self) -> Array2<f64> {
        self.orbitals.orbs.take().expect("OrbitalData:orbs; The MO coefficients were not set.")
    }

    /// Take the MO energies and return them.
    pub fn take_orbe(&mut self) -> Array1<f64> {
        self.orbitals.orbe.take().expect("OrbitalData:orbe; The MO energies were not set.")
    }

    /// Take the MO occupation numbers and return them.
    pub fn take_occupation(&mut self) -> Vec<f64> {
        self.orbitals.occupation.take().expect("OrbitalData:occupation; The MO occupation numbers were not set.")
    }

    /// Take the indices of occupied MOs and return them.
    pub fn take_occ_indices(&mut self) -> Vec<usize> {
        self.orbitals.occ_indices.take().expect("OrbitalData:occ_indices; The indices of occupied MOs were not set.")
    }

    /// Take the indices of virtual MOs and return them.
    pub fn take_virt_indices(&mut self) -> Vec<usize> {
        self.orbitals.virt_indices.take().expect("OrbitalData:virt_indices; The indices of virtual MOs were not set.")
    }

    /// Take the density matrix in AO basis and return it.
    pub fn take_p(&mut self) -> Array2<f64> {
        self.orbitals.p.take().expect("OrbitalData:p; The density matrix in AO basis was not set.")
    }

    /// Take the reference density matrix in AO basis and return it.
    pub fn take_p_ref(&mut self) -> Array2<f64> {
        self.orbitals.p_ref.take().expect("OrbitalData:p_ref; The reference density matrix in AO basis was not set.")
    }

    /// Take the difference between the density matrix and the reference density matrix and return it.
    pub fn take_delta_p(&mut self) -> Array2<f64> {
        self.orbitals.delta_p.take().expect("OrbitalData:delta_p; The difference between the density matrix and the reference density matrix was not set.")
    }

    /// Take the density matrix of alpha electrons in AO basis and return it.
    pub fn take_p_alpha(&mut self) -> Array2<f64> {
        self.orbitals.p_alpha.take().expect("OrbitalData:p_alpha; The density matrix of alpha electrons in AO basis was not set.")
    }

    /// Take the density matrix of beta electrons in AO basis and return it.
    pub fn take_p_beta(&mut self) -> Array2<f64> {
        self.orbitals.p_beta.take().expect("OrbitalData:p_beta; The density matrix of beta electrons in AO basis was not set.")
    }
}
