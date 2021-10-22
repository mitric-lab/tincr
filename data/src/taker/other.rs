use ndarray::prelude::*;
use crate::data::{OtherData, Storage};
use crate::scc::mixer::BroydenMixer;

impl<'a> Storage<'a> {
    /// Takes the Fock matrix from the struct.
    pub fn take_fock(&mut self) -> Array2<f64> {
        self.other.fock.take().expect("Error: No fock matrix was set.")
    }

    /// Takes the overlap matrix from the struct.
    pub fn take_s(&mut self) -> Array2<f64> {
        self.other.s.take().expect("Error: No overlap matrix was set.")
    }

    /// Takes the S^{-1/2} matrix from the struct.
    pub fn take_x(&mut self) -> Array2<f64> {
        self.other.x.take().expect("Error: No S^{-1/2} matrix was set.")
    }

    /// Takes the atomic numbers from the struct.
    pub fn take_atomic_numbers(&mut self) -> Vec<u8> {
        self.other.atomic_numbers.take().expect("Error: No atomic numbers were set.")
    }

    /// Takes the Broyden mixer from the struct.
    pub fn take_broyden_mixer(&mut self) -> BroydenMixer {
        self.other.broyden_mixer.take().expect("Error: No Broyden mixer was set.")
    }

    /// Takes the electrostatic potential from the struct.
    pub fn take_v(&mut self) -> Array2<f64> {
        self.other.v.take().expect("Error: No electrostatic potential was set.")
    }

    /// Takes the Fock matrix from the struct.
    pub fn take_h0(&mut self) -> Array2<f64> {
        self.other.h0.take().expect("Error: No fock matrix was set.")
    }
}