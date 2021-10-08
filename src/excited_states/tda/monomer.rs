use crate::excited_states::solvers::DavidsonEngine;
use crate::excited_states::{orbe_differences, trans_charges, ProductCache};
use crate::fmo::Monomer;
use crate::initialization::Atom;
use ndarray::prelude::*;

impl Monomer {
    pub fn prepare_tda(&mut self, atoms: &[Atom]) {
        let occ_indices: &[usize] = self.data.occ_indices();
        let virt_indices: &[usize] = self.data.virt_indices();

        // The index of the HOMO (zero based).
        let homo: usize = self.data.homo();

        // The index of the LUMO (zero based).
        let lumo: usize = self.data.lumo();

        if !self.data.q_ov_is_set() {
            let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
                self.n_atoms,
                atoms,
                self.data.orbs(),
                self.data.s(),
                &occ_indices,
                &virt_indices,
            );
            // and stored.
            self.data.set_q_oo(qoo);
            self.data.set_q_ov(qov);
            self.data.set_q_vv(qvv);
        }

        if !self.data.omega_is_set() {
            // Reference to the orbital energies.
            // Check if the orbital energy differences were already computed.
            let orbe: ArrayView1<f64> = self.data.orbe();

            // Energies of the occupied orbitals.
            let orbe_occ: ArrayView1<f64> = orbe.slice(s![0..homo + 1]);

            // Energies of the virtual orbitals.
            let orbe_virt: ArrayView1<f64> = orbe.slice(s![lumo..]);

            // Energy differences between virtual and occupied orbitals.
            let omega: Array1<f64> = orbe_differences(orbe_occ, orbe_virt);

            // Energy differences are stored in the molecule.
            self.data.set_omega(omega);
        }
    }
}
