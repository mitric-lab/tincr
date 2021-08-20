mod system;
mod monomer;
mod utils;
mod moments;

use ndarray::prelude::*;
use crate::excited_states::{ProductCache, orbe_differences, initial_subspace};
use crate::initialization::Atom;
use crate::excited_states::solvers::davidson::Davidson;
use crate::fmo::{Fragment, Monomer};
use crate::excited_states::tda::utils::print_states;
use moments::{mulliken_dipoles, oscillator_strength};


pub trait TDA {
    fn run_tda(&mut self, atoms: &[Atom], n_roots: usize, max_iter:usize, tolerance: f64);
}

impl TDA for Monomer {
    fn run_tda(&mut self, atoms: &[Atom], n_roots: usize, max_iter: usize, tolerance: f64)  {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let davidson: Davidson = Davidson::new(self, guess, n_roots, tolerance, max_iter).unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&davidson.eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(davidson.eigenvalues.view(), tr_dipoles.view());

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(davidson.eigenvalues);
        self.properties.set_ci_coefficients(davidson.eigenvectors);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        print_states(&self, n_roots);
    }
}

