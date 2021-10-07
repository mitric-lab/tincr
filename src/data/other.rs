use ndarray::prelude::*;
use crate::data::{OtherData, Storage};
use crate::scc::mixer::BroydenMixer;


impl<'a> OtherData<'a> {
    pub fn new() -> Self {
        Self {
            h0: None,
            s: None,
            x: None,
            broyden_mixer: None,
            atomic_numbers: None,
            v: None,
            fock: None,
            diabatic_basis_states: None
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}
