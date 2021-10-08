use ndarray::prelude::*;
use crate::data::{OtherData, Storage};
use crate::scc::mixer::BroydenMixer;
use hashbrown::HashMap;
use crate::fmo::PairType;
use crate::fmo::basis::BasisState;

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

    /// Set the indices of the monomers to the type of pair they form.
    pub fn set_pair_types(&mut self, map: HashMap<(usize, usize), PairType>) {
        self.other.pair_types = Some(map);
    }

    /// Set the HashMap that maps the monomers to the index of the pair they form.
    pub fn set_pair_indices(&mut self, map: HashMap<(usize, usize), usize>) {
        self.other.pair_indices = Some(map);
    }

    /// Set the LCMO-FMO Fock matrix.
    pub fn set_lcmo_fock(&mut self, fock: Array2<f64>) {
        self.other.lcmo_fock = Some(fock);
    }

    /// Set the derivative of the lr-corrected two electron integrals.
    pub fn set_flr_dmd0(&mut self, value: Array3<f64>) {
        self.other.flr_dmd0 = Some(value);
    }

    /// Set the diabatic basis states.
    pub fn set_diabatic_basis_states(&mut self, basis: Vec<BasisState<'a>>) {
        self.other.diabatic_basis_states = Some(basis);
    }
}