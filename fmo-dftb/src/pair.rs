use crate::constants::{VDW_SUM, VDW_RADII};
use crate::fmo::Monomer;
use core::::{Atom, AtomSlice, Geometry};
use crate::io::Configuration;
use crate::scc::gamma_approximation::GammaFunction;
use ndarray::prelude::*;
use std::collections::HashMap;
use crate::utils::Timer;
use log::info;
use ndarray::stack;
use std::ops::Add;
use std::fmt;
use crate::data::{Storage, Parametrization, SpatialOrbitals, OrbType};
use crate::param::reppot::RepulsivePotential;
use crate::param::slako::SlaterKoster;
use soa_derive::soa_zip;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PairType {
    Pair,
    ESD,
    None,
}

impl fmt::Display for PairType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PairType::Pair => write!(f, "Pair"),
            PairType::ESD => write!(f, "ESD"),
            PairType::None => write!(f, "None"),
        }
    }
}


/// Check if the monomers are close to each other or not.
pub fn get_pair_type(mi_atoms: AtomSlice, mj_atoms: AtomSlice, vdw_scaling: f64) -> PairType {
    // Check if the shortest distance between two monomers is within the sum of the van-der-Waals
    // radii of the closest atom pair multiplied by a scaling factor. This threshold in terms of
    // DFTB was taken from https://pubs.acs.org/doi/pdf/10.1021/ct500489d (see page 4805).
    // the threshold is generally used in FMO theory and was originally presented in
    // Chem. Phys. Lett. 2002, 351, 475−480
    // For every atom we do a conversion from the u8 type usize. But it was checked and it
    // it does not seem to have a large effect on the performance.
    let mut kind: PairType = PairType::ESD;
    'pair_loop: for (xyz_i, num_i) in soa_zip!(mi_atoms, [xyz, number]) {
        for (xyz_j, num_j) in soa_zip!(mj_atoms, [xyz, number]) {
            if (xyz_i - xyz_j).norm() < vdw_scaling * VDW_SUM[*num_i as usize][num_j as usize] {
                kind = PairType::Pair;
                break 'pair_loop;
            }
        }
    }
    return kind;
}


/// Type that holds a fragment pair that uses the ESD approximation. For this kind of pair no SCC
/// calculation is required and therefore only a minimal amount of information is stored in this type.
pub struct ESDPair<'a> {
    /// Index of the first monomer
    pub i: usize,
    /// Index of the second monomer
    pub j: usize,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Type that holds calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub data: Storage<'a>,
}

impl<'a> ESDPair<'a> {
    pub fn new(m1: &Monomer, m2: &Monomer, params: Parametrization<'a>) -> Self {
        let orbitals = SpatialOrbitals::new(m1.n_orbs()() + m2.n_orbs()(), m1.n_el, OrbType::Restricted);
        let data = Storage::new_with_orbitals(params, orbitals);
        Self {
            i: m1.index,
            j: m2.index,
            n_atoms: m1.atoms.len() + m2.atoms.len(),
            n_orbs: m1.n_orbs() + m2.n_orbs(),
            data,
        }
    }
}


/// Type that holds a fragment pair that contains all data for the quantum chemical routines.
/// For this type of pair full scc are implemented. This type is only used for FMO calculations
/// and is a similar to the [Monomer] type that but holds less properties.
pub struct Pair<'a> {
    pub atoms_i: AtomSlice<'a>,
    pub atoms_j: AtomSlice<'a>,
    /// Index of the first monomer contained in the pair
    pub i: usize,
    /// Index of the second monomer contained in the pair
    pub j: usize,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub data: Storage<'a>,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. dR for each unique pair of atoms as a spline.
    pub vrep: &'a RepulsivePotential,
    /// Slater-Koster parameters for the H0 and overlap matrix elements. For each unique atom pair
    /// the matrix elements and their derivatives can be splined.
    pub slako: &'a SlaterKoster,
}

impl<'a> Pair<'a> {
    pub fn new<'b>(m1: &'b Monomer<'a>, m2: &'b Monomer<'a>, atoms: (AtomSlice<'a>, AtomSlice<'a>), params: Parametrization<'a>) -> Self {
        let orbitals = SpatialOrbitals::new(m1.n_orbs()() + m2.n_orbs()(), m1.n_el, OrbType::Restricted);
        let data = Storage::new_with_orbitals(params, orbitals);
        Self {
            atoms_i: atoms.0,
            atoms_j: atoms.1,
            i: m1.index,
            j: m2.index,
            data,
            vrep: m1.vrep,
            slako: m1.slako,
        }
    }
}

impl<'a> PartialEq for Pair<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.j == other.j
    }
}

impl<'a> Add for &Monomer<'a> {
    type Output = Pair<'a>;

    fn add(self, rhs: Self) -> Self::Output {

    }
}





