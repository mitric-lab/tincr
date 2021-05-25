use crate::fmo::fragmentation::{build_graph, fragmentation, Graph};
use crate::fmo::helpers::MolecularSlice;
use crate::fmo::{get_pair_type, ESDPair, Monomer, Pair, PairType};
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::initialization::{
    get_unique_atoms, initialize_gamma_function, Atom, Geometry, Properties,
};
use crate::io::{frame_to_atoms, frame_to_coordinates, read_file_to_frame, Configuration};
use crate::param::elements::Element;
use crate::scc::gamma_approximation;
use crate::scc::gamma_approximation::{gamma_atomwise, GammaFunction};
use crate::utils::Timer;
use chemfiles::Frame;
use hashbrown::HashMap;
use itertools::{sorted, Itertools};
use log::info;
use ndarray::prelude::*;
use std::hash::Hash;

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
    /// [Vec] that holds the pairs of two fragments if they are close to each other. Each pair is
    /// stored as [Pair](crate::fmo::Pair) that holds all information necessary for scc calculations
    pub pairs: Vec<Pair>,
    /// [Vec] that holds pairs for which the energy is only calculated within the ESD approximation.
    /// Only a small and minimal amount of information is stored in this [ESDPair] type.
    pub esd_pairs: Vec<ESDPair>,
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
        // Measure the time for the building of the struct
        let timer: Timer = Timer::start();

        // Get all [Atom]s from the Frame and also a HashMap with the unique atoms
        let (atoms, unique_atoms): (Vec<Atom>, Vec<Atom>) = frame_to_atoms(input.0);

        // Get the number of unpaired electrons from the input option
        let n_unpaired: usize = match input.1.mol.multiplicity {
            1u8 => 0,
            2u8 => 1,
            3u8 => 2,
            _ => panic!("The specified multiplicity is not implemented"),
        };

        // Create a new Properties type, which is empty
        let mut properties: Properties = Properties::new();

        // and initialize the SlaterKoster and RepulsivePotential Tables
        let mut slako: SlaterKoster = SlaterKoster::new();
        let mut vrep: RepulsivePotential = RepulsivePotential::new();

        // Find all unique pairs of atom and fill in the SK and V-Rep tables
        let element_iter = unique_atoms.iter().map(|atom| Element::from(atom.number));
        for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
            slako.add(kind1, kind2);
            vrep.add(kind1, kind2);
        }

        // Initialize the unscreened Gamma function -> r_lr == 0.00
        let gf: GammaFunction = initialize_gamma_function(&unique_atoms, 0.0);

        // Initialize the screened gamma function only if LRC is requested
        let gf_lc: Option<GammaFunction> = if input.1.lc.long_range_correction {
            Some(initialize_gamma_function(
                &unique_atoms,
                input.1.lc.long_range_radius,
            ))
        } else {
            None
        };

        // Get all [Atom]s of the SuperSystem in a sorted order that corresponds to the order of
        // the monomers
        let mut sorted_atoms: Vec<Atom> = Vec::with_capacity(atoms.len());

        // Build a connectivity graph to distinguish the individual monomers from each other
        let graph: Graph = build_graph(atoms.len(), &atoms);

        // Here does the fragmentation happens
        let monomer_indices: Vec<Vec<usize>> = fragmentation(&graph);

        // Vec that stores all [Monomer]s
        let mut monomers: Vec<Monomer> = Vec::with_capacity(monomer_indices.len());

        // The [Monomer]s are initialized
        // PARALLEL: this loop should be parallelized
        let mut at_counter: usize = 0;
        let mut orb_counter: usize = 0;
        for (idx, indices) in monomer_indices.into_iter().enumerate() {

            // Clone the atoms that belong to this monomer, they will be stored in the sorted list
            let mut monomer_atoms: Vec<Atom> =
                indices.into_iter().map(|i| atoms[i].clone()).collect();

            // Count the number of orbitals
            let m_n_orbs: usize = monomer_atoms.iter().fold(0, |n, atom| n + atom.n_orbs);

            // Create the slices for the atoms, grads and orbitals
            let m_slice: MolecularSlice =
                MolecularSlice::new(at_counter, monomer_atoms.len(), orb_counter, m_n_orbs);

            // Create the Monomer object
            let current_monomer = Monomer {
                n_atoms: monomer_atoms.len(),
                n_orbs: m_n_orbs,
                index: idx,
                slice: m_slice,
                properties: Properties::new(),
                vrep: vrep.clone(),
                slako: slako.clone(),
                gammafunction: gf.clone(),
                gammafunction_lc: gf_lc.clone(),
            };

            // Increment the counter
            at_counter += current_monomer.n_atoms;
            orb_counter += current_monomer.n_orbs;

            // Save the Monomer
            monomers.push(current_monomer);

            // Save the Atoms from the current Monomer
            sorted_atoms.append(&mut monomer_atoms);
        }
        // Rename the sorted atoms
        let atoms: Vec<Atom> = sorted_atoms;

        // Calculate the number of atomic orbitals for the whole system as the sum of the monomer
        // number of orbitals
        let n_orbs: usize = orb_counter;

        // Compute the Gamma function between all atoms if it is requested in the user input
        // TODO: Insert a input option for this choice
        if true {
            properties.set_gamma(gamma_atomwise(&gf, &atoms, atoms.len()));
        }

        // Initialize the close pairs and the ones that are treated within the ES-dimer approx
        let mut pairs: Vec<Pair> = Vec::new();
        let mut esd_pairs: Vec<ESDPair> = Vec::new();
        // The construction of the [Pair]s requires that the [Atom]s in the atoms are ordered after
        // each monomer
        // TODO: Read the vdw scaling parameter from the input file instead of setting hard to 2.0
        // PARALLEL: this loop should be parallelized
        for (i, m_i) in monomers.iter().enumerate() {
            for (j, m_j) in monomers[(i + 1)..].iter().enumerate() {
                match get_pair_type(&atoms[m_i.slice.atom], &atoms[m_j.slice.atom], 1.8) {
                    PairType::Pair => pairs.push(m_i + m_j),
                    PairType::ESD => {
                        esd_pairs.push(ESDPair::new(i, (i + j + 1), m_i, m_j))
                    }
                    _ => {}
                }
            }
        }
        info!("{}", timer);

        Self {
            config: input.1,
            atoms: atoms,
            n_mol: monomers.len(),
            monomers: monomers,
            geometry: geom,
            properties: properties,
            gammafunction: gf,
            gammafunction_lc: gf_lc,
            pairs,
            esd_pairs,
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
