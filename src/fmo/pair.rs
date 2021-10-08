use crate::constants::{VDW_SUM, VDW_RADII};
use crate::fmo::Monomer;
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::initialization::{Atom, Geometry};
use crate::properties::Properties;
use crate::io::Configuration;
use crate::scc::gamma_approximation::GammaFunction;
use ndarray::prelude::*;
use std::collections::HashMap;
use crate::utils::Timer;
use log::info;
use ndarray::stack;
use std::ops::Add;
use std::fmt;

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
pub fn get_pair_type(mi_atoms: &[Atom], mj_atoms: &[Atom], vdw_scaling: f64) -> PairType {
    // Check if the shortest distance between two monomers is within the sum of the van-der-Waals
    // radii of the closest atom pair multiplied by a scaling factor. This threshold in terms of
    // DFTB was taken from https://pubs.acs.org/doi/pdf/10.1021/ct500489d (see page 4805).
    // the threshold is generally used in FMO theory and was originally presented in
    // Chem. Phys. Lett. 2002, 351, 475−480
    // For every atom we do a conversion from the u8 type usize. But it was checked and it
    // it does not seem to have a large effect on the performance.
    let mut kind: PairType = PairType::ESD;
    'pair_loop: for atomi in mi_atoms.iter() {
        for atomj in mj_atoms.iter() {
            if (atomi - atomj).norm() < vdw_scaling * VDW_SUM[atomi.number as usize][atomj.number as usize] {
                kind = PairType::Pair;
                break 'pair_loop;
            }
        }
    }
    return kind;
}


/// Type that holds a fragment pair that use the ESD approximation. For this kind of pair no SCC
/// calculation is required and therefore only a minimal amount of information is stored in this type.
#[derive(Debug)]
pub struct ESDPair {
    /// Index of the first monomer
    pub i: usize,
    /// Index of the second monomer
    pub j: usize,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Type that holds calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
}

impl ESDPair {
    pub fn new(i: usize, j:usize, monomer1: &Monomer, monomer2: &Monomer) -> Self {
        Self {
            i: i,
            j: j,
            n_atoms: monomer1.n_atoms + monomer2.n_atoms,
            n_orbs: monomer1.n_orbs + monomer2.n_orbs,
            properties: Properties::new(),
        }
    }
}


/// Type that holds a fragment pair that contains all data for the quantum chemical routines.
/// For this type of pair full scc are implemented. This type is only used for FMO calculations
/// and is a similar to the [Monomer] type that but holds less properties.
#[derive(Debug)]
pub struct Pair {
    /// Index of the first monomer contained in the pair
    pub i: usize,
    /// Index of the second monomer contained in the pair
    pub j: usize,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Number of valence electrons
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. dR for each unique pair of atoms as a spline.
    pub vrep: RepulsivePotential,
    /// Slater-Koster parameters for the H0 and overlap matrix elements. For each unique atom pair
    /// the matrix elements and their derivatives can be splined.
    pub slako: SlaterKoster,
    /// Type of Gamma function. This can be either `Gaussian` or `Slater` type.
    pub gammafunction: GammaFunction,
    /// Gamma function for the long-range correction. Only used if long-range correction is requested
    pub gammafunction_lc: Option<GammaFunction>,
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.j == other.j
    }
}

impl<'a> Add for &Monomer {
    type Output = Pair;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output{
            i: self.index,
            j: rhs.index,
            n_atoms: self.n_atoms + rhs.n_atoms,
            n_orbs: self.n_orbs + rhs.n_orbs,
            properties: Properties::new(),
            vrep: self.vrep.clone(),
            slako: self.slako.clone(),
            gammafunction: self.gammafunction.clone(),
            gammafunction_lc: self.gammafunction_lc.clone()
        }
    }
}





