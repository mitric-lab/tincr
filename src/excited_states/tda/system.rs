use crate::excited_states::davidson::Davidson;
use crate::excited_states::solvers::DavidsonEngine;
use crate::excited_states::tda::*;
use crate::excited_states::{initial_subspace, orbe_differences, trans_charges, ProductCache};
use crate::initialization::{Atom, System};
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, UPLO};

impl System {
    pub fn prepare_tda(&mut self) {
        if self.properties.q_ov().is_none() {
            let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
                self.n_atoms,
                &self.atoms,
                self.properties.orbs().unwrap(),
                self.properties.s().unwrap(),
                &self.occ_indices,
                &self.virt_indices,
            );
            // And stored in the properties HashMap.
            self.properties.set_q_oo(qoo);
            self.properties.set_q_ov(qov);
            self.properties.set_q_vv(qvv);
        }

        if self.properties.omega().is_none() {
            // Reference to the orbital energies.
            // Check if the orbital energy differences were already computed.
            let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();

            // The index of the HOMO (zero based).
            let homo: usize = self.occ_indices[self.occ_indices.len() - 1];

            // The index of the LUMO (zero based).
            let lumo: usize = self.virt_indices[0];

            // Energies of the occupied orbitals.
            let orbe_occ: ArrayView1<f64> = orbe.slice(s![0..homo + 1]);

            // Energies of the virtual orbitals.
            let orbe_virt: ArrayView1<f64> = orbe.slice(s![lumo..]);

            // Energy differences between virtual and occupied orbitals.
            let omega: Array1<f64> = orbe_differences(orbe_occ, orbe_virt);

            // Energy differences are stored in the molecule.
            self.properties.set_omega(omega);

            self.properties.set_homo(homo);
            self.properties.set_lumo(lumo);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::Properties;
    use crate::initialization::System;
    use crate::scc::scc_routine::RestrictedSCC;
    use crate::utils::*;
    use approx::AbsDiffEq;
    use std::borrow::BorrowMut;

    pub const EPSILON: f64 = 1e-15;

    // The Exchange contribution to the CIS Hamiltonian is computed.
    fn exchange(molecule: &System) -> Array2<f64> {
        // Number of occupied orbitals.
        let n_occ: usize = molecule.occ_indices.len();
        // Number of virtual orbitals.
        let n_virt: usize = molecule.virt_indices.len();
        // Reference to the o-o transition charges.
        let qoo: ArrayView2<f64> = molecule.properties.q_oo().unwrap();
        // Reference to the v-v transition charges.
        let qvv: ArrayView2<f64> = molecule.properties.q_vv().unwrap();
        // Reference to the screened Gamma matrix.
        let gamma_lr: ArrayView2<f64> = molecule.properties.gamma_lr().unwrap();
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
    fn fock_and_coulomb(molecule: &System) -> Array2<f64> {
        // Reference to the o-v transition charges.
        let qov: ArrayView2<f64> = molecule.properties.q_ov().unwrap();
        // Reference to the unscreened Gamma matrix.
        let gamma: ArrayView2<f64> = molecule.properties.gamma().unwrap();
        // Reference to the energy differences of the orbital energies.
        let omega: ArrayView1<f64> = molecule.properties.omega().unwrap();
        // The sum of one-electron part and Coulomb part is computed and retzurned.
        Array2::from_diag(&omega) + 2.0 * qov.t().dot(&gamma.dot(&qov))
    }

    fn test_tda_without_lc(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let mut molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        molecule.gammafunction_lc = None;
        molecule.prepare_scc();
        molecule.run_scc();
        println!("ORBES {}", molecule.properties.orbe().unwrap());
        let (u, v) = molecule.run_tda();
        let h: Array2<f64> = fock_and_coulomb(&molecule);
        let (u_ref, v_ref) = h.eigh(UPLO::Upper).unwrap();
        assert!(
            u.abs_diff_eq(&u_ref.slice(s![0..4]), 1e-10),
            "Molecule: {}, Eigenvalues (ref): {}  Eigenvalues: {}",
            name,
            u_ref.slice(s![0..6]),
            u
        );
    }

    fn test_tda_with_lc(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let mut molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        molecule.prepare_scc();
        molecule.run_scc();
        let (u, v) = molecule.run_tda();
        let h: Array2<f64> = fock_and_coulomb(&molecule) - exchange(&molecule);
        let (u_ref, v_ref) = h.eigh(UPLO::Upper).unwrap();
        assert!(
            u.abs_diff_eq(&u_ref.slice(s![0..4]), 1e-10),
            "Molecule: {}, Eigenvalues (ref): {}  Eigenvalues: {}",
            name,
            u_ref.slice(s![0..8]),
            u
        );
    }

    #[test]
    fn tda_without_lc() {
        let _ = env_logger::builder().is_test(true).try_init();
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_tda_without_lc(get_molecule(molecule, "no_lc_gs"));
        }
    }

    #[test]
    fn tda_with_lc() {
        let _ = env_logger::builder().is_test(true).try_init();
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_tda_with_lc(get_molecule(molecule, "no_lc_gs"));
        }
    }
}
