use ndarray::prelude::*;
use crate::io::{Configuration, frame_to_coordinates};
use core::::{AtomSlice, Geometry};
use crate::scc::gamma_approximation::GammaFunction;
use chemfiles::Frame;
use ndarray::{Slice, SliceInfo};
use hashbrown::HashMap;
use std::ops::Range;
use crate::fmo::helpers::MolecularSlice;
use crate::data::Storage;
use crate::param::reppot::RepulsivePotential;
use crate::param::slako::SlaterKoster;


/// Type that holds a molecular system that contains all data for the quantum chemical routines.
///
/// This type is only used for FMO calculations. This type is a similar to the [System] type that
/// is used in non-FMO calculations
pub struct Monomer<'a> {
    pub atoms: AtomSlice<'a>,
    /// Index of the monomer in the [SuperSystem]
    pub index: usize,
    /// Different Slices that correspond to this monomer
    pub slice: MolecularSlice,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub data: Storage<'a>,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. d/dR for each unique pair of atoms as a spline.
    pub vrep: &'a RepulsivePotential,
    /// Slater-Koster parameters for the H0 and overlap matrix elements. For each unique atom pair
    /// the matrix elements and their derivatives can be splined.
    pub slako: &'a SlaterKoster,
}


impl<'a> Monomer<'a> {
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    pub fn n_orbs(&self) -> usize {
        self.data.n_orbs()()
    }
}

impl<'a> PartialEq for Monomer<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<'a> Eq for Monomer<'a> {}
