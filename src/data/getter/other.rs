use ndarray::prelude::*;
use crate::data::{OtherData, Storage};
use crate::scc::mixer::BroydenMixer;
use crate::fmo::{BasisState, PairType};
use ndarray::Slice;


impl<'a> Storage<'a> {
    pub fn h0(&self) -> ArrayView2<f64> {
        match &self.other.h0 {
            Some(value) => value.view(),
            None => panic!("OtherData::h0; The one-electron integrals in AO basis are not set!"),
        }
    }

    pub fn s(&self) -> ArrayView2<f64> {
        match &self.other.s {
            Some(value) => value.view(),
            None => panic!("OtherData::s; The overlap matrix in AO basis is not set!"),
        }
    }

    pub fn s_slice(&self, rows: Slice, cols: Slice) -> ArrayView2<f64> {
        match &self.other.s {
            Some(value) => value.slice(s![rows, cols]),
            None => panic!("OtherData::s; The overlap matrix in AO basis is not set!"),
        }
    }

    pub fn x(&self) -> ArrayView2<f64> {
        match &self.other.x {
            Some(value) => value.view(),
            None => panic!("OtherData::x; The S^{-1/2} matrix is not set!"),
        }
    }

    pub fn broyden_mixer(&mut self) -> &mut BroydenMixer {
        match &mut self.other.broyden_mixer {
            Some(ref mut value) => value,
            None => panic!("OtherData::broyden_mixer; The Broyden mixer is not set!"),
        }
    }

    pub fn atomic_numbers(&self) -> &Vec<u8> {
        match &self.other.atomic_numbers {
            Some(value) => value,
            None => panic!("OtherData::atomic_numbers; The atomic numbers are not set!"),
        }
    }

    pub fn v(&self) -> ArrayView2<f64> {
        match &self.other.v {
            Some(value) => value.view(),
            None => panic!("OtherData::v; The electrostatic potential is not set!"),
        }
    }

    pub fn fock(&self) -> ArrayView2<f64> {
        match &self.other.fock {
            Some(value) => value.view(),
            None => panic!("OtherData::fock; The Fock matrix is not set!"),
        }
    }

    /// Get the type of a pair formed by two monomers.
    pub fn type_of_pair(&self, i: usize, j: usize) -> PairType {
        if i == j {
            PairType::None
        } else {
            match &self.other.pair_types {
                Some(v) => v.get(&(i, j)).unwrap_or_else(|| v.get(&(j, i)).unwrap()).to_owned(),
                None => panic!("OtherData:pair_types; The types of the pair were not set"),
            }
        }
    }

    /// Get the index of a pair formed by two monomers.
    pub fn index_of_pair(&self, i: usize, j: usize) -> usize {
        if i == j {
            panic!("I = J, A pair cannot consist of the same two monomers!")
        } else {
            match &self.other.pair_indices {
                Some(v) => v.get(&(i, j)).unwrap_or_else(|| v.get(&(j, i)).unwrap()).to_owned(),
                None => panic!("OtherData:pair_indices; The indices of the pair were not set"),
            }
        }
    }

    /// Get the LCMO-FMO Fock matrix.
    pub fn lcmo_fock(&self) -> ArrayView2<f64> {
        match &self.other.lcmo_fock {
            Some(value) => value.view(),
            None => panic!("OtherData:lcmo_fock; The LCMO-Fock matrix was not set"),
        }
    }

    /// Get the derivative of the lr-corrected two electron integrals.
    pub fn flr_dmd0(&self) -> ArrayView3<f64> {
        match &self.other.flr_dmd0 {
            Some(value) => value.view(),
            None => panic!("OtherData:flr_dmd0; Flr DMD0 was not set"),
        }
    }

}

