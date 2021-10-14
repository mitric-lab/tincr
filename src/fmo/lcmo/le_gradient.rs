use crate::fmo::{Monomer, SuperSystem, ExcitonStates, BasisState, LocallyExcited, ExcitedStateMonomerGradient};
use crate::initialization::{Atom, MO};
use ndarray::prelude::*;
use nalgebra::{max, Vector3};
use crate::excited_states::tda::*;
use ndarray_linalg::{Eigh, UPLO};
use std::fmt::{Display, Formatter};
use crate::excited_states::ExcitedState;
use crate::io::settings::LcmoConfig;
use ndarray_npy::write_npy;
use crate::utils::Timer;

impl SuperSystem{
    pub fn exciton_le_energy(
        &mut self,
        monomer_index:usize,
        state:usize,
    )->f64{
        let lcmo_config: LcmoConfig = self.config.lcmo.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // get the monomer
        let mol = &mut self.monomers[monomer_index];
        // Compute the excited states for the monomer.
        mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
        mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le,  max_iter, tolerance);

        // switch to immutable borrow for the monomer
        let mol= &self.monomers[monomer_index];

        // Calculate transition charges
        let homo: usize = mol.properties.homo().unwrap();
        let q_ov: ArrayView2<f64> = mol.properties.q_ov().unwrap();

        // Create the LE state
        let tdm: ArrayView1<f64> = mol.properties.ci_coefficient(state).unwrap();
        let le_state:BasisState = BasisState::LE(LocallyExcited {
            monomer: mol,
            n: state,
            atoms: &atoms[mol.slice.atom_as_range()],
            q_trans: q_ov.dot(&tdm),
            occs: mol.properties.orbs_slice(0, Some(homo+1)).unwrap(),
            virts: mol.properties.orbs_slice(homo + 1, None).unwrap(),
            tdm: tdm,
            tr_dipole: mol.properties.tr_dipole(state).unwrap(),
        });

        let val:f64 = self.exciton_coupling(&le_state,&le_state);
        return val;
    }

    pub fn singular_le_gradient(
        &mut self,
        monomer_index:usize,
        state:usize,
    )->Array1<f64>{
        let lcmo_config: LcmoConfig = self.config.lcmo.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];

        // parameters for the TDA routine
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // get the monomer
        let mol = &mut self.monomers[monomer_index];
        // Compute the excited states for the monomer.
        mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
        mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le,  max_iter, tolerance);

        // calculate the gradient
        mol.prepare_excited_gradient(&atoms[mol.slice.atom_as_range()]);
        let grad = mol.tda_gradient_lc(state);

        return grad;
    }
}