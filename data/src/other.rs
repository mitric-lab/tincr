use ndarray::prelude::*;
use crate::data::{OtherData, Storage};
use crate::scc::mixer::BroydenMixer;
use hashbrown::HashMap;


impl OtherData {
    pub fn new() -> Self {
        Self {
            x: None,
            broyden_mixer: None,
            atomic_numbers: None,
            v: None,
            fock: None,
            pair_types: None,
            pair_indices: None,
            lcmo_fock: None,
            flr_dmd0: None
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

impl<'a> Storage<'a> {
    /// Check if the overlap matrix is set.
    pub fn s_is_set(&self) -> bool {
        self.other.s.is_some()
    }
}
