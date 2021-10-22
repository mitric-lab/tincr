use core::::atom::*;
use core::::geometry::*;
use core::::{
    get_unique_atoms, initialize_gamma_function, initialize_unrestricted_elec,
};
use tincr::io::{frame_to_coordinates, read_file_to_frame, Configuration};
use tincr::param::elements::Element;
use tincr::scc::gamma_approximation;
use tincr::scc::gamma_approximation::GammaFunction;
use chemfiles::Frame;
use hashbrown::HashMap;
use itertools::Itertools;
use ndarray::prelude::*;
use std::borrow::BorrowMut;
use tincr::fmo::Fragment;
use tincr::src::Storage;
use tincr::param::reppot::{RepulsivePotential, RepulsivePotentialTable};
use tincr::param::slako::{SlaterKoster, SlaterKosterTable};
use tincr::param::skf_handler::SkfHandler;


/// Type that holds a molecular system that contains all data for the quantum chemical routines.
/// This type is only used for non-FMO calculations. In the case of FMO based calculation
/// the `FMO`, `Molecule` and `Pair` types are used instead.
pub struct System<'a> {
    /// Type that holds all the input settings from the user.
    pub config: &'a Configuration,
    /// SoA of the individual atoms that are of the [Atom] type.
    pub atoms: AtomSlice<'a>,
    /// Type that holds the calculated properties e.g. gamma matrix, overlap matrix and so on.
    pub data: Storage<'a>,
    /// Repulsive potential type. Type that contains the repulsion energy and its derivative
    /// w.r.t. d/dR for each unique pair of atoms as a spline.
    pub vrep: &'a RepulsivePotential,
    /// Slater-Koster parameters for the core Hamiltonian and Overlap matrix elements. For each
    /// unique atom pair the matrix elements and their derivatives can be computed from a spline.
    pub slako: &'a SlaterKoster,
}


impl<'a> System<'a>{
    // pub fn update_xyz(&mut self, coordinates: Array1<f64>) {
    //     let coordinates: Array2<f64> = coordinates.into_shape([self.atoms.len(), 3]).unwrap();
    //     // TODO: The IntoIterator trait was released for ndarray 0.15. The dependencies should be
    //     // updated, so that this can be used. At the moment of writing ndarray-linalg is not yet
    //     // compatible with ndarray 0.15x
    //     // PARALLEL
    //     for (atom, xyz) in self.atoms
    //         .iter_mut()
    //         .zip(coordinates.outer_iter()){
    //         atom.position_from_ndarray(xyz.to_owned());
    //     }
    //     //.for_each(|(atom, xyz)| atom.position_from_ndarray(xyz.to_owned()))
    // }

    pub fn n_orbs(&self) -> usize {
        self.data.n_orbs()
    }

    pub fn get_xyz(&self) -> Array1<f64> {
        let xyz_list: Vec<Vec<f64>> = self
            .atoms
            .xyz
            .iter()
            .map(|xyz| xyz.iter().cloned().collect())
            .collect();
        Array1::from_shape_vec((3 * self.atoms.len()), itertools::concat(xyz_list)).unwrap()
    }
}
