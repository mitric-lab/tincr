use ndarray::prelude::*;
use crate::io::{Configuration, frame_to_coordinates};
use crate::initialization::{Atom, Geometry, Properties};
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::scc::gamma_approximation::GammaFunction;
use chemfiles::Frame;
use ndarray::{Slice, SliceInfo};
use hashbrown::HashMap;
use std::ops::Range;

/// Type that holds a molecular system that contains all data for the quantum chemical routines.
/// This type is only used for FMO calculations. This type is a similar to the [System] type that
/// is used in non-FMO calculations
pub struct Monomer {
    /// Type that holds all the input settings from the user.
    pub config: Configuration,
    /// Number of atoms
    pub n_atoms: usize,
    /// Index of the monomer in the [SuperSystem]
    pub index: usize,
    /// [Slice](ndarray::prelude::Slice) for the atoms corresponding to this monomer
    pub atom_slice: Range<usize>,
    /// Gradient slice, this is the atom slice multiplied by the factor 3
    pub grad_slice: Slice,
    /// Number of atomic orbitals
    pub n_orbs: usize,
    /// [Slice](ndarray::prelude::Slice) for the orbitals corresponding to this monomer
    pub orb_slice: Slice,
    /// Number of valence electrons
    pub n_elec: usize,
    /// Number of unpaired electrons (singlet -> 0, doublet -> 1, triplet -> 2)
    pub n_unpaired: usize,
    /// Indices of occupied orbitals starting from zero
    pub occ_indices: Vec<usize>,
    /// Indices of virtual orbitals
    pub virt_indices: Vec<usize>,
    /// List index of first active occupied orbital
    pub first_active_occ: usize,
    /// List index of last of active virtual orbital
    pub last_active_virt: usize,
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
    /// w.r.t. d/dR for each unique pair of atoms as a spline.
    pub vrep: RepulsivePotential,
    /// Slater-Koster parameters for the H0 and overlap matrix elements. For each unique atom pair
    /// the matrix elements and their derivatives can be splined.
    pub slako: SlaterKoster,
    /// Type of Gamma function. This can be either `Gaussian` or `Slater` type.
    pub gammafunction: GammaFunction,
    /// Gamma function for the long-range correction. Only used if long-range correction is requested
    pub gammafunction_lc: Option<GammaFunction>,
}


impl Monomer {
    /// Creates a new [Monomer] from a [Vec](alloc::vec) of atomic numbers, the coordinates as an [Array2](ndarray::Array2) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    pub fn new(config: Configuration, atoms: Vec<Atom>, index: usize, at_index: usize, orb_index: usize, slako: SlaterKoster, vrep: RepulsivePotential, gf: GammaFunction, gf_lc: Option<GammaFunction>) -> Self {
        // get the atomic numbers and positions from the input data
        let n_atoms: usize = atoms.len();
        let atom_slice: Range<usize> = at_index..(at_index+n_atoms);//Slice::from(at_index..(at_index+n_atoms));
        let grad_slice: Slice = Slice::from((at_index*3)..(at_index+n_atoms)*3);
        // calculate the number of electrons
        let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
        // get the number of unpaired electrons from the input option
        let n_unpaired: usize = 0;
        // calculate the number of atomic orbitals for the whole system as the sum of the atomic
        // orbitals per atom
        let n_orbs: usize = atoms.iter().fold(0, |n, atom| n + atom.n_orbs);
        let orb_slice: Slice = Slice::from(orb_index..(orb_index+n_orbs));
        // get the indices of the occupied and virtual orbitals
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        (0..n_orbs).for_each(|index| if index < (n_elec/2) {occ_indices.push(index)} else {virt_indices.push(index)});
        // TODO: Get the number of active occupied and virtual orbitals for FMO
        let first_active_occ = occ_indices.len() - 0;
        let active_virt = 0;
        // Create the Geometry from the coordinates. At this point the coordinates have to be
        // transformed already in atomic units
        let geom: Geometry = Geometry::new();
        // Create a new and empty Properties type
        let properties: Properties = Properties::new();

        Self{
            config,
            n_atoms,
            index,
            atom_slice,
            grad_slice,
            n_orbs,
            orb_slice,
            n_elec,
            n_unpaired,
            occ_indices,
            virt_indices: virt_indices,
            first_active_occ: first_active_occ,
            last_active_virt: active_virt,
            atoms,
            geometry: geom,
            properties,
            vrep,
            slako,
            gammafunction: gf,
            gammafunction_lc: gf_lc
        }
    }
}



