use crate::initialization::atom::Atom;
use crate::initialization::geometry::*;
use crate::initialization::parameters::*;
use crate::initialization::properties2::Properties;
use crate::io::{frame_to_coordinates, Configuration};
use crate::param::Element;
use crate::{constants, defaults};
use approx::AbsDiffEq;
use chemfiles::Frame;
use itertools::Itertools;
use log::{debug, error, info, trace, warn};
use ndarray::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;
use crate::scc::gamma_approximation::GammaFunction;
use crate::scc::gamma_approximation;
use owning_ref::VecRef;

/// Type that holds a molecular system that contains all data for the quantum chemical routines.
/// This type is only used for non-FMO calculations. In the case of FMO based calculation
/// the `FMO`, `Molecule` and `Pair` types are used instead.
pub struct System<'a> {
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
    /// Vector with references to the individual atoms that are stored as `Atom` type
    pub atoms: Vec<&'a Atom>,
    /// Vector that stores the data of the unique atoms in an `Atom` type
    pub unique_atoms: Vec<Atom>,
    /// Type that stores the  coordinates and matrices that depend on the position of the atoms
    pub geometry: Geometry,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub properties: Properties,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. d/dR for each unique pair of atoms as a spline.
    pub vrep: RepulsivePotential<'a>,
    /// Slater-Koster parameters for the H0 and overlap matrix elements. For each unique atom pair
    /// the matrix elements and their derivatives can be splined.
    pub slako: SlaterKoster<'a>,
    /// Type of Gamma function. This can be either `Gaussian` or `Slater` type.
    pub gammafunction: GammaFunction,
    /// Gamma function for the long-range correction. Only used if long-range correction is requested
    pub gammafunction_lc: Option<GammaFunction>,
}

impl<'a> From<(Vec<u8>, Array2<f64>, Configuration)> for System<'a> {
    /// Creates a new [System] from a [Vec](alloc::vec) of atomic numbers, the coordinates as an [Array2](ndarray::Array2) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(molecule: (Vec<u8>, Array2<f64>, Configuration)) -> Self {
        let mut unique_numbers: Vec<u8> = molecule.0.clone();
        unique_numbers.sort_unstable(); // fast sort of atomic numbers
        unique_numbers.dedup(); // delete duplicates
        let unique_numbers: Vec<u8> = Vec::from(unique_numbers); // allocate the unique numbers
        // create the unique Atoms
        let unique_atoms: Vec<Atom> = unique_numbers
            .iter()
            .map(|number| Atom::from(*number))
            .collect();
        let mut num_to_atom: HashMap<&u8, &Atom> = HashMap::with_capacity(unique_numbers.len());
        // insert the atomic numbers and the reference to atoms in the HashMap
        unique_numbers
            .iter()
            .zip(unique_atoms.iter())
            .map(|(num, atom)| num_to_atom.insert(num, atom));
        // get all references to the Atom's from the HashMap
        let atoms: Vec<&Atom> = molecule.0.iter().map(|num| num_to_atom[num]).collect();
        // calculate the number of electrons
        let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
        // get the number of unpaired electrons from the input option
        let n_unpaired: usize = match molecule.2.mol.multiplicity {
            1u8 => 0,
            2u8 => 1,
            3u8 => 2,
            _=> panic!("The specified multiplicity is not implemented")
        };
        // calculate the number of atomic orbitals for the whole system as the sum of the atomic
        // orbitals per atom
        let n_orbs: usize = atoms.iter().fold(0, |n, atom| n + atom.n_orbs);
        // Create the Geometry from the coordinates. At this point the coordinates have to be
        // transformed already in atomic units
        let geom: Geometry = Geometry::from(molecule.1);
        // Create a new and empty Properties type
        let properties: Properties = Properties::new();
        let mut slako: SlaterKoster = SlaterKoster::new();
        let mut vrep: RepulsivePotential = RepulsivePotential::new();
        // add all unique element pairs
        let element_iter = unique_numbers.iter().map(|num| Element::from(*num));
        for (kind1, kind2) in element_iter.cartesian_product(element_iter.clone()) {
            slako.add(&kind1, &kind2);
            vrep.add(&kind1, &kind2);
        }
        // initialize the gamma function
        // TODO: Check which Gamma function is specified in the input
        let sigma: HashMap<u8, f64> = gamma_approximation::gaussian_decay(&unique_atoms);
        let mut c: HashMap<(u8, u8), f64> = HashMap::new();
        let mut gf = gamma_approximation::GammaFunction::Gaussian { sigma, c, r_lr: 0.0 };
        gf.initialize();
        // initialize the gamma function for long-range correction if it is requested
        let mut gf_lc: Option<GammaFunction> = if molecule.2.lc.long_range_correction {
            let sigma: HashMap<u8, f64> = gamma_approximation::gaussian_decay(&unique_atoms);
            let mut c: HashMap<(u8, u8), f64> = HashMap::new();
            let mut tmp = gamma_approximation::GammaFunction::Gaussian { sigma, c, r_lr: molecule.2.lc.long_range_radius };
            tmp.initialize();
            Some(tmp)
        } else {
            None
        };
        Self{
            config: molecule.2,
            n_atoms: molecule.0.len(),
            n_orbs: n_orbs,
            n_elec: n_elec,
            n_unpaired: n_unpaired,
            atoms: atoms,
            unique_atoms: unique_atoms,
            geometry: geom,
            properties: properties,
            vrep: vrep,
            slako: slako,
            gammafunction: gf,
            gammafunction_lc: gf_lc
        }
    }
}

impl From<(Frame, Configuration)> for System<'_> {
    /// Creates a new [System] from a [Frame](chemfiles::Frame) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(frame: (Frame, Configuration)) -> Self {
        let (numbers, coords) = frame_to_coordinates(frame.0);
        Self::from((numbers, coords, frame.1))
    }
}
