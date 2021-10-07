use ndarray::prelude::*;
use crate::data::{OtherData, Storage};
use crate::scc::mixer::BroydenMixer;

impl<'a> Storage<'a> {
    /// Set the Fock matrix.
    pub fn set_fock(&mut self, fock: Array2<f64>) {
        self.other.fock = Some(fock);
    }

    /// Set the overlap matrix.
    pub fn set_s(&mut self, s: Array2<f64>) {
        self.other.s = Some(s);
    }

    /// Set the inverse of the overlap matrix.
    pub fn set_x(&mut self, x: Array2<f64>) {
        self.other.x = Some(x);
    }

    /// Set the atomic numbers of the atoms of the system.
    pub fn set_atomic_numbers(&mut self, atomic_numbers: Vec<u8>) {
        self.other.atomic_numbers = Some(atomic_numbers);
    }

    /// Set the Broyden mixer.
    pub fn set_broyden_mixer(&mut self, broyden_mixer: BroydenMixer) {
        self.other.broyden_mixer = Some(broyden_mixer);
    }

    /// Set the electrostatic potential.
    pub fn set_v(&mut self, v: Array2<f64>) {
        self.other.v = Some(v);
    }

    /// Set the one-electron integral matrix in AO basis.
    pub fn set_h0(&mut self, h0: Array2<f64>) {
        self.other.h0 = Some(h0);
    }
}