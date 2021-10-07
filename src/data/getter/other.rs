use ndarray::prelude::*;
use crate::data::{OtherData, Storage};
use crate::scc::mixer::BroydenMixer;
use crate::fmo::BasisState;


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

    pub fn basis_states(&self) -> &'a Vec<BasisState> {
        match &self.other.diabatic_basis_states {
            Some(value) => value,
            None => panic!("OtherData::fock; The Fock matrix is not set!"),
        }
    }
}

