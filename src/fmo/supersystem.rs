use crate::fmo::fragmentation::get_fragments;
use crate::fmo::{Monomer, Pair};
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::initialization::{
    get_unique_atoms, initialize_gamma_function, Atom, Geometry, Properties,
};
use crate::io::{frame_to_coordinates, Configuration, read_file_to_frame};
use crate::param::elements::Element;
use crate::scc::gamma_approximation;
use crate::scc::gamma_approximation::{GammaFunction, gamma_atomwise};
use chemfiles::Frame;
use ndarray::prelude::*;
use std::collections::HashMap;
use itertools::Itertools;
use crate::utils::Timer;
use log::info;

pub struct SuperSystem {
    /// Type that holds all the input settings from the user.
    pub config: Configuration,
    /// Vector with the data and the positions of the individual
    /// atoms that are stored as [Atom](crate::initialization::Atom)
    pub atoms: Vec<Atom>,
    /// Number of fragments in the whole system, this corresponds to self.molecules.len()
    pub n_mol: usize,
    /// List of individuals fragments which are stored as a [Monomer](crate::fmo::Monomer)
    pub monomers: Vec<Monomer>,
    /// HashMap with that holds the pairs of two fragments. Each pair is stored as [Pair](crate::fmo::Pair)
    /// that holds basic information about the pair.
    pub pairs: HashMap<(usize, usize), Pair>,
    /// Type that can hold calculated properties e.g. gamma matrix for the whole FMO system
    pub properties: Properties,
    /// Type that stores the coordinates of the whole FMO system
    pub geometry: Geometry,
    /// Type of Gamma function. This can be either `Gaussian` or `Slater` type.
    pub gammafunction: GammaFunction,
    /// Gamma function for the long-range correction. Only used if long-range correction is requested
    pub gammafunction_lc: Option<GammaFunction>,
}

impl From<(Frame, Configuration)> for SuperSystem {
    /// Creates a new [SuperSystem] from a [Vec](alloc::vec) of atomic numbers, the coordinates as an [Array2](ndarray::Array2) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(input: (Frame, Configuration)) -> Self {
        // get the atomic numbers and positions from the input data. The [Frame] needs to be cloned,
        // as it will be also used for the fragmentation into monomers
        let (atomic_numbers, coordinates) = frame_to_coordinates(input.0.clone());
        // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
        let (unique_atoms, num_to_atom): (Vec<Atom>, HashMap<u8, Atom>) =
            get_unique_atoms(&atomic_numbers);
        // get the number of unpaired electrons from the input option
        let n_unpaired: usize = match input.1.mol.multiplicity {
            1u8 => 0,
            2u8 => 1,
            3u8 => 2,
            _ => panic!("The specified multiplicity is not implemented"),
        };
        // Create the Geometry from the coordinates. At this point the coordinates have to be
        // transformed already in atomic units
        let geom: Geometry = Geometry::from(coordinates);
        // Create a new and empty Properties type
        let mut properties: Properties = Properties::new();
        let mut slako: SlaterKoster = SlaterKoster::new();
        let mut vrep: RepulsivePotential = RepulsivePotential::new();
        // add all unique element pairs
        let element_iter = unique_atoms.iter().map(|atom| Element::from(atom.number));
        for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
            slako.add(kind1, kind2);
            vrep.add(kind1, kind2);
        }

        let gf: GammaFunction = initialize_gamma_function(&unique_atoms, 0.0);
        // initialize the gamma function for long-range correction if it is requested
        let gf_lc: Option<GammaFunction> = if input.1.lc.long_range_correction {
            Some(initialize_gamma_function(
                &unique_atoms,
                input.1.lc.long_range_radius,
            ))
        } else {
            None
        };

        let molecules: Vec<Frame> = get_fragments(input.0);
        let mut monomers: Vec<Monomer> = Vec::with_capacity(molecules.len());
        // PARALLEL: this loop should be parallelized
        for mol_frame in molecules.into_iter() {
            monomers.push(Monomer::new(
                input.1.clone(),
                mol_frame,
                num_to_atom.clone(),
                slako.clone(),
                vrep.clone(),
                gf.clone(),
                gf_lc.clone(),
            ));
        }
        // get all the [Atom]s for the SuperSystem. They are just copied from the individual monomers.
        // This ensures that they are also grouped for each monomer and are in the same order
        let mut atoms: Vec<Atom> = Vec::with_capacity(atomic_numbers.len());
        monomers.iter().for_each(|monomer| {
            monomer
                .atoms
                .iter()
                .for_each(|atom| atoms.push(atom.clone()))
        });
        // calculate the number of atomic orbitals for the whole system as the sum of the monomer
        // number of orbitals
        let n_orbs: usize = monomers.iter().fold(0, |n, monomer| n + monomer.n_orbs);
        // calculate the number of electrons
        let n_elec: usize = monomers.iter().fold(0, |n, monomer| n + monomer.n_elec);

        // Compute the gamma function between all atoms if it is requested in the user input
        // TODO: Insert a new key for this option
        if true {
            properties.set_gamma(gamma_atomwise(&gf, &atoms, atoms.len()));
        }

        let mut pairs: Vec<Pair> = Vec::with_capacity(monomers.len() * (monomers.len() - 1) / 2);
        // the construction of the [Pair]s requires that the [Atom]s in the atoms are ordered after
        // each monomer
        // TODO: Read the vdw scaling parameter from the input file instead of setting hard to 2.0
        // PARALLEL: this loop should be parallelized
        for (i, monomer_i) in monomers.iter().enumerate() {
            for (j, monomer_j) in monomers[(i+1)..].iter().enumerate() {
                pairs.push(Pair::new((i, j), (monomer_i, monomer_j), 2.0));
            }
        }

        Self {
            config: input.1,
            atoms: atoms,
            n_mol: monomers.len(),
            monomers: monomers,
            geometry: geom,
            properties: properties,
            gammafunction: gf,
            gammafunction_lc: gf_lc,
            pairs: Default::default(),
        }
    }
}


impl From<(&str, Configuration)> for SuperSystem {
    /// Creates a new [SuperSystem] from a &str and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(filename_and_config: (&str, Configuration)) -> Self {
        let frame: Frame = read_file_to_frame(filename_and_config.0);
        Self::from((frame, filename_and_config.1))
    }
}