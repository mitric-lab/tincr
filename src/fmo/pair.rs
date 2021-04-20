use crate::constants::{VDW_SUM, VDW_RADII};
use crate::fmo::Monomer;
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::initialization::{Atom, Geometry, Properties};
use crate::io::Configuration;
use crate::scc::gamma_approximation::GammaFunction;
use ndarray::prelude::*;
use std::collections::HashMap;
use crate::utils::Timer;
use log::info;
use ndarray::stack;
use std::ops::Add;

pub enum PairType {
    Pair,
    ESD,
    None,
}

/// Check if the monomers are close to each other or not.
pub fn get_pair_type(monomer1: &Monomer, monomer2: &Monomer, vdw_scaling: f64) -> PairType {
    // Check if the shortest distance between two monomers is within the sum of the van-der-Waals
    // radii of the closest atom pair multiplied by a scaling factor. This threshold in terms of
    // DFTB was taken from https://pubs.acs.org/doi/pdf/10.1021/ct500489d (see page 4805).
    // the threshold is generally used in FMO theory and was originally presented in
    // Chem. Phys. Lett. 2002, 351, 475−480
    // For every atom we do a conversion from the u8 type usize. But it was checked and it
    // it does not seem to have a large effect on the performance.
    let mut kind: PairType = PairType::ESD;
    'pair_loop: for atomi in monomer1.atoms.iter() {
        for atomj in monomer2.atoms.iter() {
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
/// For this type of pair full scc are implemented. This type is only used for FMO calculations.
/// This type is a similar to the [Monomer] type that but holds some further properties that are
/// specific to fragment pairs
pub struct Pair {
    /// Index of the first monomer contained in the pair
    pub i: usize,
    /// Index of the second monomer contained in the pair
    pub j: usize,
    /// Type that holds all the input settings from the user.
    pub config: Configuration,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// Number of valence electrons
    pub n_elec: usize,
    /// Number of unpaired electrons (singlet -> 0, doublet -> 1, triplet -> 2)
    pub n_unpaired: usize,
    /// Vector with the data of the individual atoms that are stored as [Atom] type. This is not an
    /// efficient solution because there are only a few [Atom] types and they are copied for each
    /// atom in the molecule. However, to the best of the authors knowledge the self referential
    /// structs are not trivial to implement in Rust. There are ways to do this by using crates like
    /// `owning_ref`, `rental`, or `ouroboros` or by using the [Pin](std::marker::Pin) type. The other
    /// crates does not fit exactly to our purpose and the [Pin](std::marker::Pin) requires a lot of
    /// extra effort. The computational cost for the clones was measured to be on the order of
    /// ~ 0.003 seconds (on a mac mini with M1) and to consume about 9 MB of memory for 20.000 [Atoms]
    pub atoms: Vec<Atom>,
    /// Type that stores the  coordinates and matrices that depend on the position of the atoms
    pub geometry: Geometry,
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



impl Add for &Monomer {
    type Output = Pair;

    fn add(self, rhs: Self) -> Self::Output {
        let coordinates: Array2<f64> = stack![Axis(0), self.geometry.coordinates.view(), rhs.geometry.coordinates.view()];
        let geom: Geometry = Geometry::from(coordinates);
        let mut atoms: Vec<Atom> = self.atoms.clone();
        atoms.append(&mut rhs.atoms.clone());
        Self::Output{
            i: self.index,
            j: rhs.index,
            config: self.config.clone(),
            n_atoms: self.n_atoms + rhs.n_atoms,
            n_orbs: self.n_orbs + rhs.n_orbs,
            n_elec: self.n_elec + rhs.n_elec,
            n_unpaired: self.n_unpaired + rhs.n_unpaired,
            atoms,
            geometry: geom,
            properties: Properties::new(),
            vrep: self.vrep.clone(),
            slako: self.slako.clone(),
            gammafunction: self.gammafunction.clone(),
            gammafunction_lc: self.gammafunction_lc.clone()
        }
    }
}





