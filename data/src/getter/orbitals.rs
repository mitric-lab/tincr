use ndarray::prelude::*;
use crate::data::{SpatialOrbitals, Storage};


impl<'a> Storage<'a> {

    /// Returns the index of the HOMO.
    pub fn homo(&self) -> usize {
        self.orbitals.homo()
    }

    /// Returns the index of the LUMO.
    pub fn lumo(&self) -> usize {
        self.orbitals.lumo()
    }

    /// Returns the number of occupied orbitals.
    pub fn n_occ(&self) -> usize {
        self.orbitals.n_occ()
    }

    /// Returns the number of virtual orbitals.
    pub fn n_virt(&self) -> usize {
        self.orbitals.n_virt()
    }

    /// Returns the number of orbitals.
    pub fn n_orbs(&self) -> usize {
        self.orbitals.occupation.as_ref().len()
    }

    pub fn orbs_slice(&self, start: usize, end: Option<usize>) -> ArrayView2<f64> {
        let slice = if matches!(end, None) {s![.., start..]} else {s![.., start..end.unwrap()]};
        match &self.orbitals.orbs {
            Some(value) => value.slice(slice),
            None => panic!("OrbitalData::orbs; The MO coefficients are not set!"),
        }
    }

    pub fn mo_coeff(&self, mo_idx: usize) -> ArrayView1<f64> {
        match &self.orbitals.orbs {
            Some(value) => value.column(mo_idx),
            None => panic!("OrbitalData::orbs; The MO coefficients are not set!"),
        }
    }

    pub fn orbs(&self) -> ArrayView2<f64> {
        match &self.orbitals.orbs {
            Some(value) => value.view(),
            None => panic!("OrbitalData::orbs; The MO coefficients are not set!"),
        }
    }

    pub fn orbe(&self) -> ArrayView1<f64> {
        match &self.orbitals.orbe {
            Some(value) => value.view(),
            None => panic!("OrbitalData::orbe; The MO energies are not set!"),
        }
    }

    pub fn occupation(&self) -> &[f64] {
        match &self.orbitals.occupation {
            Some(value) => value,
            None => panic!("OrbitalData::occupation; The MO occupation numbers are not set!"),
        }
    }

    pub fn occ_indices(&self) -> &[usize] {
        match &self.orbitals.occ_indices {
            Some(value) => value,
            None => panic!("OrbitalData::occ_indices; The MO occupation numbers are not set!"),
        }
    }

    pub fn virt_indices(&self) -> &[usize] {
        match &self.orbitals.virt_indices {
            Some(value) => value,
            None => panic!("OrbitalData::virt_indices; The MO occupation numbers are not set!"),
        }
    }

    pub fn p(&self) -> ArrayView2<f64> {
        match &self.orbitals.p {
            Some(value) => value.view(),
            None => panic!("OrbitalData::p; The density matrix is not set!"),
        }
    }

    pub fn p_ref(&self) -> ArrayView2<f64> {
        match &self.orbitals.p_ref {
            Some(value) => value.view(),
            None => panic!("OrbitalData::p_ref; The reference density matrix is not set!"),
        }
    }

    pub fn delta_p(&self) -> ArrayView2<f64> {
        match &self.orbitals.delta_p {
            Some(value) => value.view(),
            None => panic!("OrbitalData::delta_p; The density matrix difference is not set!"),
        }
    }

    pub fn p_alpha(&self) -> ArrayView2<f64> {
        match &self.orbitals.p_alpha {
            Some(value) => value.view(),
            None => panic!("OrbitalData::p_alpha; The alpha density matrix is not set!"),
        }
    }

    pub fn p_beta(&self) -> ArrayView2<f64> {
        match &self.orbitals.p_beta {
            Some(value) => value.view(),
            None => panic!("OrbitalData::p_beta; The beta density matrix is not set!"),
        }
    }
}
