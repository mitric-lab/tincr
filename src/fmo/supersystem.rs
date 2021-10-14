use crate::fmo::fragmentation::{build_graph, fragmentation, Graph};
use crate::fmo::helpers::{MolecularSlice, MolIndices, MolIncrements};
use crate::fmo::{get_pair_type, ESDPair, Monomer, Pair, PairType};
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster, RepulsivePotentialTable, SlaterKosterTable, SkfHandler};
use crate::properties::Properties;
use crate::initialization::{get_unique_atoms, initialize_gamma_function, Atom, Geometry, get_unique_atoms_mio};
use crate::io::{frame_to_atoms, frame_to_coordinates, read_file_to_frame, Configuration};
use crate::param::elements::Element;
use crate::scc::gamma_approximation;
use crate::scc::gamma_approximation::{gamma_atomwise, GammaFunction, gamma_atomwise_par};
use crate::utils::Timer;
use chemfiles::Frame;
use hashbrown::{HashMap, HashSet};
use itertools::{sorted, Itertools};
use log::info;
use ndarray::prelude::*;
use std::hash::Hash;
use std::result::IntoIter;
use std::vec;
use ndarray::Slice;
use crate::scc::h0_and_s::h0_and_s;

#[derive(Debug)]
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

        // create mutable Vectors
        let mut unique_atoms: Vec<Atom> = Vec::new();
        let mut atoms: Vec<Atom> = Vec::new();
        let mut skf_handlers: Vec<SkfHandler> = Vec::new();

        if input.1.slater_koster.use_mio == true {
            // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
            // if use_mio is true, create a vector of homonuclear SkfHandlers and a vector
            // of heteronuclear SkfHandlers

            let mut num_to_atom: HashMap<u8, Atom> = HashMap::new();
            let (numbers, coords) = frame_to_coordinates(input.0);

            let tmp: (Vec<Atom>, HashMap<u8, Atom>, Vec<SkfHandler>) =
                get_unique_atoms_mio(&numbers, &input.1);
            unique_atoms = tmp.0;
            num_to_atom = tmp.1;
            skf_handlers = tmp.2;

            // get all the Atom's from the HashMap
            numbers
                .iter()
                .for_each(|num| atoms.push((*num_to_atom.get(num).unwrap()).clone()));
            // set the positions for each atom
            coords
                .outer_iter()
                .enumerate()
                .for_each(|(idx, position)| {
                    atoms[idx].position_from_slice(position.as_slice().unwrap())
                });
        } else {
            // get the unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s
            let tmp: (Vec<Atom>, Vec<Atom>) = frame_to_atoms(input.0);
            atoms = tmp.0;
            unique_atoms = tmp.1;
        }

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

        if input.1.slater_koster.use_mio == true {
            for handler in skf_handlers.iter() {
                let repot_table: RepulsivePotentialTable =
                    RepulsivePotentialTable::from(handler);
                let slako_table_ab: SlaterKosterTable =
                    SlaterKosterTable::from((handler, None, "ab"));
                let slako_handler_ba: SkfHandler = SkfHandler::new(
                    handler.element_b,
                    handler.element_a,
                    input.1.slater_koster.mio_directory.clone(),
                );
                let slako_table: SlaterKosterTable =
                    SlaterKosterTable::from((&slako_handler_ba, Some(slako_table_ab), "ba"));

                // insert the tables into the hashmaps
                slako
                    .map
                    .insert((handler.element_a, handler.element_b), slako_table);
                vrep.map
                    .insert((handler.element_a, handler.element_b), repot_table);
            }
        } else {
            let element_iter = unique_atoms.iter().map(|atom| Element::from(atom.number));
            for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
                slako.add(kind1, kind2);
                vrep.add(kind1, kind2);
            }
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
        let mut mol_indices: MolIndices = MolIndices::new();

        for (idx, indices) in monomer_indices.into_iter().enumerate() {
            // Clone the atoms that belong to this monomer, they will be stored in the sorted list
            let mut monomer_atoms: Vec<Atom> =
                indices.into_iter().map(|i| atoms[i].clone()).collect();

            // Count the number of orbitals
            let m_n_orbs: usize = monomer_atoms.iter().fold(0, |n, atom| n + atom.n_orbs);

            // Count the number of electrons.
            let n_elec: usize = monomer_atoms.iter().map(|atom| atom.n_elec).sum();

            // Number of occupied orbitals.
            let n_occ: usize = (n_elec / 2);

            // Number of virtual orbitals.
            let n_virt: usize = m_n_orbs - n_occ;

            let mut props: Properties = Properties::new();
            props.set_n_occ(n_occ);
            props.set_n_virt(n_virt);

            let increments: MolIncrements = MolIncrements {
                atom: monomer_atoms.len(),
                orbs: m_n_orbs,
                occs: n_occ,
                virts: n_virt,
            };

            // Create the slices for the atoms, grads and orbitals
            let m_slice: MolecularSlice =
                MolecularSlice::new(mol_indices.clone(), increments);

            // Create the Monomer object
            let mut current_monomer = Monomer {
                n_atoms: monomer_atoms.len(),
                n_orbs: m_n_orbs,
                index: idx,
                slice: m_slice,
                properties: props,
                vrep: vrep.clone(),
                slako: slako.clone(),
                gammafunction: gf.clone(),
                gammafunction_lc: gf_lc.clone(),
            };
            // Compute the number of electrons for the monomer and set the indices of the
            // occupied and virtual orbitals.
            current_monomer.set_mo_indices(n_elec);

            // Increment the indices..
            mol_indices.add(increments);

            // Save the current Monomer.
            monomers.push(current_monomer);

            // Save the Atoms from the current Monomer
            sorted_atoms.append(&mut monomer_atoms);
        }
        // Rename the sorted atoms
        let atoms: Vec<Atom> = sorted_atoms;

        // Calculate the number of atomic orbitals for the whole system as the sum of the monomer
        // number of orbitals
        let n_orbs: usize = mol_indices.orbs;
        // Set the number of occupied and virtual orbitals.
        properties.set_n_occ(mol_indices.occs);
        properties.set_n_virt(mol_indices.virts);

        // Compute the Gamma function between all atoms if it is requested in the user input
        // TODO: Insert a input option for this choice
        if true {
            properties.set_gamma(gamma_atomwise(&gf, &atoms, atoms.len()));
            properties.set_gamma_lr(gamma_atomwise(&initialize_gamma_function(
                &unique_atoms,
                input.1.lc.long_range_radius,
            ), &atoms, atoms.len()));
            // properties.set_gamma(gamma_atomwise_par(&gf, &atoms));
            // properties.set_gamma_lr(gamma_atomwise_par(&initialize_gamma_function(
            //     &unique_atoms,
            //     input.1.lc.long_range_radius,
            // ), &atoms));
        }

        // Initialize the close pairs and the ones that are treated within the ES-dimer approx
        let mut pairs: Vec<Pair> = Vec::new();
        let mut esd_pairs: Vec<ESDPair> = Vec::new();

        // Create a HashMap that maps the Monomers to the type of Pair. To identify if a pair of
        // monomers are considered a real pair or should be treated with the ESD approx.
        let mut pair_iter:usize = 0;
        let mut esd_iter:usize = 0;
        let mut pair_indices: HashMap<(usize, usize),usize> = HashMap::new();
        let mut esd_pair_indices:HashMap<(usize, usize),usize> = HashMap::new();
        let mut pair_types: HashMap<(usize, usize), PairType> = HashMap::new();

        // The construction of the [Pair]s requires that the [Atom]s in the atoms are ordered after
        // each monomer
        // TODO: Read the vdw scaling parameter from the input file instead of setting hard to 2.0
        // PARALLEL: this loop should be parallelized
        for (i, m_i) in monomers.iter().enumerate() {
            for (j, m_j) in monomers[(i + 1)..].iter().enumerate() {
                match get_pair_type(
                    &atoms[m_i.slice.atom_as_range()],
                    &atoms[m_j.slice.atom_as_range()],
                    2.0,
                ) {
                    PairType::Pair => {
                        pairs.push(m_i + m_j);
                        pair_types.insert((m_i.index, m_j.index), PairType::Pair);
                        pair_indices.insert((m_i.index, m_j.index),pair_iter);
                        pair_iter += 1;
                    },
                    PairType::ESD => {
                        esd_pairs.push(ESDPair::new(i, (i + j + 1), m_i, m_j));
                        pair_types.insert((m_i.index, m_j.index), PairType::ESD);
                        esd_pair_indices.insert((m_i.index, m_j.index),esd_iter);
                        esd_iter += 1;
                    },
                    _ => {}
                }
            }
        }
        properties.set_pair_types(pair_types);
        properties.set_pair_indices(pair_indices);
        properties.set_esd_pair_indices(esd_pair_indices);

        info!("{}", timer);

        let (s, h0) = h0_and_s(n_orbs, &atoms, &slako);
        properties.set_s(s);

        Self {
            config: input.1,
            atoms: atoms,
            n_mol: monomers.len(),
            monomers: monomers,
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

impl SuperSystem {
    pub fn update_xyz(&mut self, coordinates: Array1<f64>) {
        let coordinates: Array2<f64> = coordinates.into_shape([self.atoms.len(), 3]).unwrap();
        // TODO: The IntoIterator trait was released for ndarray 0.15. The dependencies should be
        // updated, so that this can be used. At the moment of writing ndarray-linalg is not yet
        // compatible with ndarray 0.15x
        // PARALLEL
        for (atom, xyz) in self.atoms
            .iter_mut()
            .zip(coordinates.outer_iter()) {
            atom.position_from_ndarray(xyz.to_owned());
        }
        //.for_each(|(atom, xyz)| atom.position_from_ndarray(xyz.to_owned()))
    }

    pub fn get_xyz(&self) -> Array1<f64> {
        let xyz_list: Vec<Vec<f64>> = self
            .atoms
            .iter()
            .map(|atom| atom.xyz.iter().cloned().collect())
            .collect();
        Array1::from_shape_vec((3 * self.atoms.len()), itertools::concat(xyz_list)).unwrap()
    }

    pub fn gamma_a(&self, a: usize, lrc: LRC) -> ArrayView2<f64> {
        self.gamma_a_b(a, a, lrc)
    }

    pub fn gamma_a_b(&self, a: usize, b: usize, lrc: LRC) -> ArrayView2<f64> {
        let atoms_a: Slice = self.monomers[a].slice.atom.clone();
        let atoms_b: Slice = self.monomers[b].slice.atom.clone();
        match lrc {
            LRC::ON => self.properties.gamma_lr_slice(atoms_a, atoms_b).unwrap(),
            LRC::OFF => self.properties.gamma_slice(atoms_a, atoms_b).unwrap(),
        }
    }

    pub fn gamma_ab_c(&self, a: usize, b: usize, c: usize, lrc: LRC) -> Array2<f64> {
        let n_atoms_a: usize = self.monomers[a].n_atoms;
        let mut gamma: Array2<f64> = Array2::zeros([n_atoms_a + self.monomers[b].n_atoms, self.monomers[c].n_atoms]);
        gamma.slice_mut(s![0..n_atoms_a, ..]).assign(&self.gamma_a_b(a, c, lrc));
        gamma.slice_mut(s![n_atoms_a.., ..]).assign(&self.gamma_a_b(b, c, lrc));
        gamma
    }

    pub fn gamma_ab_cd(&self, a: usize, b: usize, c: usize, d:usize, lrc: LRC) -> Array2<f64> {
        let n_atoms_a: usize = self.monomers[a].n_atoms;
        let n_atoms_c: usize = self.monomers[c].n_atoms;
        let mut gamma: Array2<f64> = Array2::zeros([n_atoms_a + self.monomers[b].n_atoms, n_atoms_c + self.monomers[d].n_atoms]);
        gamma.slice_mut(s![0..n_atoms_a, ..n_atoms_c]).assign(&self.gamma_a_b(a, c, lrc));
        gamma.slice_mut(s![n_atoms_a.., ..n_atoms_c]).assign(&self.gamma_a_b(b, c, lrc));
        gamma.slice_mut(s![0..n_atoms_a, n_atoms_c..]).assign(&self.gamma_a_b(a, d, lrc));
        gamma.slice_mut(s![n_atoms_a.., n_atoms_c..]).assign(&self.gamma_a_b(b, d, lrc));
        gamma
    }

    pub fn update_s(&mut self){
        let n_orbs:usize = self.properties.n_occ().unwrap() + self.properties.n_virt().unwrap();
        let slako = &self.monomers[0].slako;
        let (s, h0) = h0_and_s(n_orbs, &self.atoms, slako);
        self.properties.set_s(s);
    }
}

#[derive(Copy, Clone)]
pub enum LRC {
    ON,
    OFF,
}