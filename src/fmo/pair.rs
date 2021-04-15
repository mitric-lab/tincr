use crate::constants::VDW_SUM;
use crate::fmo::Monomer;
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::initialization::{Atom, Geometry, Properties};
use crate::io::Configuration;
use crate::scc::gamma_approximation::GammaFunction;
use ndarray::prelude::*;
use std::collections::HashMap;
use crate::utils::Timer;
use log::info;

pub enum PairApproximation {
    Pair,
    ESD,
    None,
}

pub struct Pair {
    /// Approximation type that is used for this dimer
    pub kind: PairApproximation,
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

impl Pair {
    /// Create a new pair and check if the monomers are close to each other or not.
    pub fn new(indices: (usize, usize), monomers: (&Monomer, &Monomer), vdw_scaling: f64) -> Self {
        // Check if the shortest distance between two monomers is within the sum of the van-der-Waals
        // radii of the closest atom pair multiplied by a scaling factor. This threshold in terms of
        // DFTB was taken from https://pubs.acs.org/doi/pdf/10.1021/ct500489d (see page 4805).
        // the threshold is generally used in FMO theory and was originally presented in
        // Chem. Phys. Lett. 2002, 351, 475−480
        // For every atom we do a conversion from the u8 type usize. But it was checked and it
        // it does not seem to have a large effect on the performance.
        let is_in_proximity = monomers.0.atoms.iter().find(|atomi| {
            monomers
                .1
                .atoms
                .iter()
                .find(|atomj| {
                    (*atomi - *atomj).norm() < (vdw_scaling * VDW_SUM[atomi.number as usize][atomj.number as usize])
                })
                .is_some()
        });
        let kind: PairApproximation = if is_in_proximity.is_some() {
            PairApproximation::Pair
        } else {
            PairApproximation::ESD
        };

        Self {
            kind: kind,
            i: indices.0,
            j: indices.1,
            n_atoms: monomers.0.n_atoms + monomers.1.n_atoms,
            n_orbs: monomers.0.n_orbs + monomers.1.n_orbs,
            properties: Properties::new(),
        }
    }
}
