use ndarray::prelude::*;
use crate::io::{Configuration, frame_to_coordinates};
use crate::initialization::{Atom, Geometry};
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::properties::Properties;
use crate::scc::gamma_approximation::GammaFunction;
use chemfiles::Frame;
use ndarray::{Slice, SliceInfo};
use hashbrown::HashMap;
use std::ops::Range;
use crate::fmo::helpers::MolecularSlice;


/// Type that holds a molecular system that contains all data for the quantum chemical routines.
///
/// This type is only used for FMO calculations. This type is a similar to the [System] type that
/// is used in non-FMO calculations
pub struct Monomer {
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Index of the monomer in the [SuperSystem]
    pub index: usize,
    /// Different Slices that correspond to this monomer
    pub slice: MolecularSlice,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. d/dR for each unique pair of atoms as a spline.
    pub vrep: RepulsivePotential,
    /// Slater-Koster parameters for the H0 and overlap matrix elements. For each unique atom pair
    /// the matrix elements and their derivatives can be splined.
    pub slako: SlaterKoster,
    /// Type of Gamma function. This can be either `Gaussian` or `Slater` type.
    pub gammafunction: GammaFunction,
    /// Gamma function for the long-range correction. Only used if long-range correction is requested
    pub gammafunction_lc: Option<GammaFunction>,
}


impl Monomer {
    pub fn set_mo_indices(&mut self, n_elec: usize) {
        // get the indices of the occupied and virtual orbitals
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        (0..self.n_orbs).for_each(|index| if index < (n_elec/2) {occ_indices.push(index)} else {virt_indices.push(index)});
        self.properties.set_occ_indices(occ_indices);
        self.properties.set_virt_indices(virt_indices);
    }
}

impl PartialEq for Monomer {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for Monomer {

}
