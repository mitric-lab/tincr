mod moments;
mod monomer;
mod new_mod;
mod system;
mod utils;

use crate::excited_states::solvers::davidson::Davidson;
use crate::excited_states::tda::new_mod::TdaStates;
use crate::excited_states::{initial_subspace, orbe_differences, ExcitedState, ProductCache};
use crate::fmo::{Fragment, Monomer};
use crate::initialization::{Atom, System};
use moments::{mulliken_dipoles, oscillator_strength};
use ndarray::prelude::*;
use ndarray_npy::write_npy;

impl Monomer {
    pub fn run_tda(&mut self, atoms: &[Atom], n_roots: usize, max_iter: usize, tolerance: f64) {
        // Set an empty product cache.
        self.data.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: ArrayView1<f64> = self.data.omega();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let davidson: Davidson = Davidson::new(self, guess, n_roots, tolerance, max_iter).unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.data.q_ov();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&davidson.eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(davidson.eigenvalues.view(), tr_dipoles.view());

        let n_occ: usize = self.data.n_occ();
        let n_virt: usize = self.data.n_virt();
        let tdm: Array3<f64> = davidson
            .eigenvectors
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: TdaStates = TdaStates {
            total_energy: self.data.last_energy(),
            energies: davidson.eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.data.orbs().to_owned(),
        };

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.data.set_cis_eigenvalues(davidson.eigenvalues);
        self.data.set_cis_coefficients(davidson.eigenvectors);
        self.data.set_q_trans(q_trans);
        self.data.set_tr_dipoles(tr_dipoles);
        self.data.set_osc_strengths(f);

        println!("{}", states);
        // states.ntos_to_molden(&atoms, 0, "/Users/hochej/Downloads/s1.molden");
    }
}

impl System {
    pub fn run_tda(&mut self, n_roots: usize, max_iter: usize, tolerance: f64) {
        // Set an empty product cache.
        self.data.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: ArrayView1<f64> = self.data.omega();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let davidson: Davidson = Davidson::new(self, guess, n_roots, tolerance, max_iter).unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.data.q_ov();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&davidson.eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &self.atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(davidson.eigenvalues.view(), tr_dipoles.view());

        let n_occ: usize = self.data.n_occ();
        let n_virt: usize = self.data.n_virt();
        let tdm: Array3<f64> = davidson
            .eigenvectors
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: TdaStates = TdaStates {
            total_energy: self.data.last_energy(),
            energies: davidson.eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.data.orbs().to_owned(),
        };


        write_npy("/Users/hochej/Downloads/full_energies.npy", &davidson.eigenvalues.view());

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.data.set_cis_eigenvalues(davidson.eigenvalues);
        self.data.set_cis_coefficients(davidson.eigenvectors);
        self.data.set_q_trans(q_trans);
        self.data.set_tr_dipoles(tr_dipoles);
        self.data.set_osc_strengths(f);

        println!("{}", states);

        //print_states(&self, n_roots);
    }
}
