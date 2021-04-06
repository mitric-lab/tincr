use crate::h0_and_s::h0_and_s;
use crate::initialization::atom::Atom;
use crate::initialization::geometry::*;
use crate::initialization::parameters::*;
use crate::initialization::parametrization::{get_gamma_matrix, Parametrization};
use crate::initialization::properties::ElectronicData;
use crate::initialization::properties2::Properties;
use crate::io::{frame_to_coordinates, Configuration};
use crate::param::Element;
use crate::{constants, defaults};
use approx::AbsDiffEq;
use chemfiles::Frame;
use itertools::Itertools;
use log::{debug, error, info, trace, warn};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::Norm;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::hash::Hash;

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
}

impl From<(Vec<u8>, Array2<f64>, Configuration)> for System {
    /// Creates a new [System] from a [Vec](alloc::vec) of atomic numbers, the coordinates as an [Array2](ndarray::Array2) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(molecule: (Vec<u8>, Array2<f64>, Configuration)) -> Self {
        let mut unique_numbers: &[u8] = &molecule.0;
        unique_numbers.sort_unstable(); // fast sort of atomic numbers
        unique_numbers.dedup(); // delete duplicates
        let unique_numbers: Vec<u8> = Vec::from(unique_numbers); // allocate the unique numbers
        // create the unique Atoms
        let unique_atoms: Vec<Atoms> = unique_numbers
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
        for (kind1, kind2) in element_iter.cartesian_product(element_iter) {
            slako.add(&kind1, &kind2);
            vrep.add(&kind1, &kind2);
        }
        Self{
            config: molecule.2,
            n_atoms: molecule.0.len(),
            n_orbs: n_orbs,
            atoms: atoms,
            unique_atoms: unique_atoms,
            geometry: geom,
            properties: properties,
            vrep: vrep,
            slako: slako,
        }
    }
}

impl From<(Frame, Configuration)> for System {
    /// Creates a new [System] from a [Frame](chemfiles::Frame) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(frame: (Frame, Configuration)) -> Self {
        let (numbers, coords) = frame_to_coordinates(frame.0);
        Self::from((numbers, coords, frame.1))
    }
}

impl System {

    pub fn set_atomic_numbers(&mut self, numbers: Option<&[u8]>) {
        self.atomic_numbers = match numbers {
            Some(x) => Some(Vec::from(x)),
            None => None,
        };
    }

    pub fn set_positions(&mut self, positions: Option<Array2<f64>>) {
        self.positions = match positions {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_repr(&mut self, repr: Option<String>) {
        self.repr = match repr {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_proximity_matrix(&mut self, pmatrix: Option<Array2<bool>>) {
        self.proximity_matrix = match pmatrix {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_distance_matrix(&mut self, dist_matrix: Option<Array2<f64>>) {
        self.distance_matrix = match dist_matrix {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_directions_matrix(&mut self, dir_matrix: Option<Array3<f64>>) {
        self.directions_matrix = match dir_matrix {
            Some(x) => Some(x),
            None => None,
        };
    }

    // pub fn set_adjacency_matrix(&mut self, adj_matrix: Option<Array2<bool>>) {
    //     self.adjacency_matrix = match adj_matrix {
    //         Some(x) => Some(x),
    //         None => None,
    //     };
    // }

    pub fn reset(&mut self) {
        self.positions = None;
        self.distance_matrix = None;
        self.directions_matrix = None;
        self.proximity_matrix = None;
        // self.adjacency_matrix = None;
        if self.electronic_structure.is_some() {
            self.electronic_structure.reset()
        }
    }

    pub fn from_geometry(smiles: String, numbers: &[u8], coordinates: Array2<f64>) -> Self {
        let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
            build_geometric_matrices(&numbers, coordinates.view(), None);
        let repr_string: String = smiles;
        Molecule {
            atomic_numbers: Some(Vec::from(numbers)),
            positions: Some(coordinates),
            repr: Some(repr_string),
            proximity_matrix: Some(prox_matrix),
            distance_matrix: Some(dist_matrix),
            directions_matrix: Some(dir_matrix),
            // adjacency_matrix: Some(adj_matrix),
            electronic_structure: ElectronicData::new(),
        }
    }

    pub fn from_frame(frame: Frame) -> Molecule {
        let (smiles, numbers, coords) = frame_to_coordinates(frame);
        Molecule::from_geometry(smiles, &numbers, coords)
    }

    pub fn update_geometry(&mut self, atomic_numbers: &[u8], coordinates: Array2<f64>) {
        let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
            build_geometric_matrices(atomic_numbers, coordinates.view(), None);
        self.set_distance_matrix(Some(dist_matrix));
        self.set_directions_matrix(Some(dir_matrix));
        self.set_proximity_matrix(Some(prox_matrix));
        // self.set_adjacency_matrix(Some(adj_matrix));
        self.set_positions(Some(coordinates));
    }

    pub fn dimer_from_monomers(m1: &Molecule, m2: &Molecule) -> Molecule {
        let mut numbers: Vec<u8> = Vec::with_capacity(
            m1.atomic_numbers.as_ref().unwrap().len() + m2.atomic_numbers.as_ref().unwrap().len(),
        );
        numbers.extend_from_slice(&m1.atomic_numbers.as_ref().unwrap());
        numbers.extend_from_slice(&m2.atomic_numbers.as_ref().unwrap());
        let mut repr: String = format!(
            "{}{}",
            m1.repr.as_ref().unwrap().clone(),
            m2.repr.as_ref().unwrap().clone()
        );
        let positions: Array2<f64> = stack![
            Axis(0),
            m1.positions.as_ref().unwrap().view(),
            m2.positions.as_ref().unwrap().view()
        ];
        let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
            build_geometric_matrices_from_monomers(positions.view(), &m1, &m2, None);
        let es: ElectronicData =
            ElectronicData::dimer_from_monomers(&m1.electronic_structure, &m2.electronic_structure);
        Molecule {
            atomic_numbers: Some(numbers),
            positions: Some(positions),
            repr: Some(repr),
            proximity_matrix: Some(prox_matrix),
            distance_matrix: Some(dist_matrix),
            directions_matrix: Some(dir_matrix),
            // adjacency_matrix: Some(adj_matrix),
            electronic_structure: es,
        }
    }

    pub fn make_scc_ready(
        &mut self,
        n_orbs: usize,
        valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
        skt: &HashMap<(u8, u8), SlaterKosterTable>,
        orbital_energies: &HashMap<u8, HashMap<(i8, i8), f64>>,
        hubbard_u: &HashMap<u8, f64>,
        r_lr: Option<f64>,
    ) {
        // H0, S, gamma, gamma_LRC is needed for SCC routine
        // get H0 and overlap matrix
        let (h0, s): (Array2<f64>, Array2<f64>) = h0_and_s(
            self.atomic_numbers.as_ref().unwrap(),
            self.positions.as_ref().unwrap().view(),
            n_orbs,
            valorbs,
            self.proximity_matrix.as_ref().unwrap().view(),
            skt,
            orbital_energies,
        );
        self.electronic_structure.set_h0(Some(h0));
        self.electronic_structure.set_s(Some(s));
        // get gamma
        let (gm, gm_ao): (Array2<f64>, Array2<f64>) = get_gamma_matrix(
            &self.atomic_numbers.as_ref().unwrap(),
            self.atomic_numbers.as_ref().unwrap().len(),
            n_orbs,
            self.distance_matrix.as_ref().unwrap().view(),
            &hubbard_u,
            &valorbs,
            Some(0.0),
        );
        self.electronic_structure.set_gamma_atom_wise(Some(gm));
        self.electronic_structure.set_gamma_ao_wise(Some(gm_ao));
        if r_lr.is_some() {
            // get gamma lrc
            let (gm_lrc, gm_lrc_ao): (Array2<f64>, Array2<f64>) = get_gamma_matrix(
                &self.atomic_numbers.as_ref().unwrap(),
                self.atomic_numbers.as_ref().unwrap().len(),
                n_orbs,
                self.distance_matrix.as_ref().unwrap().view(),
                &hubbard_u,
                &valorbs,
                r_lr,
            );
            self.electronic_structure.set_gamma_atom_wise(Some(gm_lrc));
            self.electronic_structure.set_gamma_ao_wise(Some(gm_lrc_ao));
        }
    }

    pub fn make_dimer_scc_ready(
        &mut self,
        n_orbs: usize,
        valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
        skt: &HashMap<(u8, u8), SlaterKosterTable>,
        orbital_energies: &HashMap<u8, HashMap<(i8, i8), f64>>,
        hubbard_u: &HashMap<u8, f64>,
        r_lr: Option<f64>,
    ) {
    }
}
