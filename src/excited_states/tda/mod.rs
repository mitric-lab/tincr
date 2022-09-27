mod moments;
mod monomer;
mod new_mod;
mod system;
mod utils;
mod ct_pair;

use crate::excited_states::solvers::davidson::Davidson;
use crate::excited_states::tda::new_mod::TdaStates;
use crate::excited_states::{initial_subspace, orbe_differences, ExcitedState, ProductCache};
use crate::fmo::{Fragment, Monomer};
use crate::initialization::{Atom, System};
use moments::{mulliken_dipoles, oscillator_strength};
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, UPLO};
use ndarray_npy::write_npy;

impl Monomer {
    pub fn run_tda(&mut self, atoms: &[Atom], n_roots: usize, max_iter: usize, tolerance: f64) {
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

        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        let n_virt: usize = self.properties.virt_indices().unwrap().len();
        let tdm: Array3<f64> = davidson
            .eigenvectors
            .clone()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: TdaStates = TdaStates {
            total_energy: self.properties.last_energy().unwrap(),
            energies: davidson.eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.properties.orbs().unwrap().to_owned(),
        };

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(davidson.eigenvalues);
        self.properties.set_ci_coefficients(davidson.eigenvectors);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        println!("{}", states);
        // states.ntos_to_molden(&atoms, 0, "/Users/hochej/Downloads/s1.molden");
    }
}

impl System {
    pub fn run_tda(&mut self, n_roots: usize, max_iter: usize, tolerance: f64) {
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
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &self.atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(davidson.eigenvalues.view(), tr_dipoles.view());

        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        let n_virt: usize = self.properties.virt_indices().unwrap().len();
        let tdm: Array3<f64> = davidson
            .eigenvectors
            .clone()
            .as_standard_layout().to_owned()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: TdaStates = TdaStates {
            total_energy: self.properties.last_energy().unwrap(),
            energies: davidson.eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.properties.orbs().unwrap().to_owned(),
        };


        write_npy("full_energies.npy", &davidson.eigenvalues.view());
        write_npy("oscillator_strengths.npy",&f);

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(davidson.eigenvalues);
        self.properties.set_ci_coefficients(davidson.eigenvectors);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        println!("{}", states);

        //print_states(&self, n_roots);
    }

    pub fn tda_full_matrix(&mut self){
        let h: Array2<f64> = self.fock_and_coulomb() - self.exchange();
        let (eigenvalues, eigenvectors) = h.eigh(UPLO::Upper).unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&eigenvectors);

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &self.atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(eigenvalues.view(), tr_dipoles.view());

        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        let n_virt: usize = self.properties.virt_indices().unwrap().len();
        let tdm: Array3<f64> = eigenvectors
            .clone()
            .as_standard_layout().to_owned()
            .into_shape([n_occ, n_virt, f.len()])
            .unwrap();

        let states: TdaStates = TdaStates {
            total_energy: self.properties.last_energy().unwrap(),
            energies: eigenvalues.clone(),
            tdm: tdm,
            f: f.clone(),
            tr_dip: tr_dipoles.clone(),
            orbs: self.properties.orbs().unwrap().to_owned(),
        };


        write_npy("full_energies.npy", &eigenvalues.view());
        write_npy("oscillator_strengths.npy",&f);

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(eigenvalues);
        self.properties.set_ci_coefficients(eigenvectors);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        println!("{}", states);
    }

    fn exchange(&self) -> Array2<f64> {
        // Number of occupied orbitals.
        let n_occ: usize = self.occ_indices.len();
        // Number of virtual orbitals.
        let n_virt: usize = self.virt_indices.len();
        // Reference to the o-o transition charges.
        let qoo: ArrayView2<f64> = self.properties.q_oo().unwrap();
        // Reference to the v-v transition charges.
        let qvv: ArrayView2<f64> = self.properties.q_vv().unwrap();
        // Reference to the screened Gamma matrix.
        let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
        // The exchange part to the CIS Hamiltonian is computed.
        let result = qoo
            .t()
            .dot(&gamma_lr.dot(&qvv))
            .into_shape((n_occ, n_occ, n_virt, n_virt))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape([n_occ * n_virt, n_occ * n_virt])
            .unwrap()
            .to_owned();
        result
    }

    // The one-electron and Coulomb contribution to the CIS Hamiltonian is computed.
    fn fock_and_coulomb(&self) -> Array2<f64> {
        // Reference to the o-v transition charges.
        let qov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // Reference to the unscreened Gamma matrix.
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        // Reference to the energy differences of the orbital energies.
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The sum of one-electron part and Coulomb part is computed and retzurned.
        Array2::from_diag(&omega) + 2.0 * qov.t().dot(&gamma.dot(&qov))
    }
}
