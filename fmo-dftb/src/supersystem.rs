use crate::fmo::fragmentation::{build_graph, fragmentation, Graph};
use crate::fmo::helpers::{MolecularSlice, MolIndices, MolIncrements};
use crate::fmo::{get_pair_type, ESDPair, Monomer, Pair, PairType};
use core::::{get_unique_atoms, initialize_gamma_function, AtomSlice, Geometry, Atom};
use crate::io::{frame_to_atoms, frame_to_coordinates, read_file_to_frame, Configuration};
use crate::param::elements::Element;
use crate::scc::gamma_approximation;
use crate::scc::gamma_approximation::{gamma_atomwise, GammaFunction};
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
use crate::data::Storage;
use crate::param::skf_handler::SkfHandler;
use crate::param::slako::{SlaterKoster, SlaterKosterTable};
use crate::param::reppot::{RepulsivePotential, RepulsivePotentialTable};

pub struct SuperSystem<'a> {
    /// Type that holds all the input settings from the user.
    pub config: &'a Configuration,
    /// Vector with the data and the positions of the individual
    /// atoms that are stored as [Atom](crate::initialization::Atom)
    pub atoms: AtomSlice<'a>,
    /// List of individuals fragments which are stored as a [Monomer](crate::fmo::Monomer)
    pub monomers: Vec<Monomer<'a>>,
    /// [Vec] that holds the pairs of two fragments if they are close to each other. Each pair is
    /// stored as [Pair](crate::fmo::Pair) that holds all information necessary for scc calculations
    pub pairs: Vec<Pair<'a>>,
    /// [Vec] that holds pairs for which the energy is only calculated within the ESD approximation.
    /// Only a small and minimal amount of information is stored in this [ESDPair] type.
    pub esd_pairs: Vec<ESDPair<'a>>,
    /// Type that can hold calculated properties e.g. gamma matrix for the whole FMO system
    pub data: Storage<'a>,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. d/dR for each unique pair of atoms as a spline.
    pub vrep: &'a RepulsivePotential,
    /// Slater-Koster parameters for the core Hamiltonian and Overlap matrix elements. For each
    /// unique atom pair the matrix elements and their derivatives can be computed from a spline.
    pub slako: &'a SlaterKoster,
}



impl<'a> SuperSystem<'a> {
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
            LRC::ON => self.data.gamma_lr_slice(atoms_a, atoms_b),
            LRC::OFF => self.data.gamma_slice(atoms_a, atoms_b),
        }
    }

    pub fn gamma_ab_c(&self, a: usize, b: usize, c: usize, lrc: LRC) -> Array2<f64> {
        let n_atoms_a: usize = self.monomers[a].atoms.len();
        let mut gamma: Array2<f64> = Array2::zeros([n_atoms_a + self.monomers[b].atoms.len(), self.monomers[c].atoms.len()]);
        gamma.slice_mut(s![0..n_atoms_a, ..]).assign(&self.gamma_a_b(a, c, lrc));
        gamma.slice_mut(s![n_atoms_a.., ..]).assign(&self.gamma_a_b(b, c, lrc));
        gamma
    }

    pub fn gamma_ab_cd(&self, a: usize, b: usize, c: usize, d:usize, lrc: LRC) -> Array2<f64> {
        let n_atoms_a: usize = self.monomers[a].atoms.len();
        let n_atoms_c: usize = self.monomers[c].atoms.len();
        let mut gamma: Array2<f64> = Array2::zeros([n_atoms_a + self.monomers[b].atoms.len(), n_atoms_c + self.monomers[d].atoms.len()]);
        gamma.slice_mut(s![0..n_atoms_a, ..n_atoms_c]).assign(&self.gamma_a_b(a, c, lrc));
        gamma.slice_mut(s![n_atoms_a.., ..n_atoms_c]).assign(&self.gamma_a_b(b, c, lrc));
        gamma.slice_mut(s![0..n_atoms_a, n_atoms_c..]).assign(&self.gamma_a_b(a, d, lrc));
        gamma.slice_mut(s![n_atoms_a.., n_atoms_c..]).assign(&self.gamma_a_b(b, d, lrc));
        gamma
    }
}
