use crate::excited_states::solvers::davidson::DavidsonEngine;
use crate::excited_states::{trans_charges, ProductCache, orbe_differences, initial_subspace};
use crate::initialization::{System, Atom};
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, UPLO};
use crate::excited_states::davidson::Davidson;

pub trait TDA {
    fn run_tda(&mut self) -> (Array1<f64>, Array2<f64>);
}

impl TDA for System {
    fn run_tda(&mut self) -> (Array1<f64>, Array2<f64>) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());
        // Reference to the atoms of the molecule.
        let atoms: &[Atom] = &self.atoms;
        // Reference to the MO coefficients of the molecule.
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        // Reference to the overlap matrix of the molecule.
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        // Check if transition charges are already computed.
        if self.properties.q_ov().is_none() {
            // If not, they are computed now.
            let (qov, qoo, qvv): (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
                self.n_atoms,
                atoms,
                orbs,
                s,
                &self.occ_indices,
                &self.virt_indices,
            );
            // And stored in the properties HashMap.
            self.properties.set_q_oo(qoo);
            self.properties.set_q_ov(qov);
            self.properties.set_q_vv(qvv);
        }
        // Check the same for the orbital energy differences.
        if self.properties.omega().is_none() {
            let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
            let homo: usize = self.occ_indices[self.occ_indices.len()-1];
            let lumo: usize = self.virt_indices[0];
            let orbe_occ: ArrayView1<f64> = orbe.slice(s![0..homo+1]);
            let orbe_virt: ArrayView1<f64> = orbe.slice(s![lumo..]);
            let omega: Array1<f64> = orbe_differences(orbe_occ, orbe_virt);
            self.properties.set_omega(omega);
        }
        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        let n_roots: usize = 4;
        let tolerance: f64 = 1e-4;
        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);
        // Iterative Davidson diagonalization of the CIS Hamiltonian in a matrix free way.
        let davidson: Davidson = Davidson::new(self, guess, n_roots, tolerance, 50).unwrap();
        // The eigenvalues and eigenvectors are returned.
        (davidson.eigenvalues, davidson.eigenvectors)
    }
}

impl DavidsonEngine for System {
    /// The products of the TDA/CIS-Hamiltonian with the subspace vectors is computed.
    fn compute_products<'a>(&mut self, x: ArrayView2<'a, f64>) -> Array2<f64> {
        // Mutable reference to the product cache.
        let mut cache: ProductCache = self.properties.take_cache().unwrap();
        // Transition charges between occupied-virtual orbitals, of shape: [n_atoms, n_occ * n_virt]
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // The gamma matrix of the shape: [n_atoms, n_atoms]
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        // The energy differences between virtual and occupied orbitals, shape: [n_occ * n_virt]
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The number of products that need to be computed in the current iteration.
        let n_prod: usize = x.ncols();
        // The number of products that are already computed.
        let n_old: usize = cache.count("TDA");
        // Only the new vectors are computed.
        let compute_vectors: ArrayView2<f64> = if n_prod <= n_old {
            // If the subspace vectors space was collapsed, the cache needs to be cleared.
            cache.reset();
            // All vectors have to be computed.
            x
        } else {
            // Otherwise only the new products have to be computed.
            x.slice_move(s![.., n_old..])
        };
        // The number of vectors that needs to be computed in this iteration.
        let n_comp: usize = compute_vectors.ncols();

        // The product of the Fock matrix elements with the subspace vectors is computed.
        let fock: Array2<f64> =
            &omega.broadcast((n_comp, omega.len())).unwrap().t() * &compute_vectors;

        // The product of the Coulomb matrix elements with the supspace vectors is computed.
        let mut two_el: Array2<f64> = 2.0 * q_ov.t().dot(&gamma.dot(&q_ov.dot(&compute_vectors)));

        // If long-range correction is requested the exchange part needs to be computed.
        //if self.gammafunction_lc.is_some() {
        if false {
            // Reference to the transition charges between occupied-occupied orbitals.
            let q_oo: ArrayView2<f64> = self.properties.q_oo().unwrap();
            // Number of occupied orbitals.
            let n_occ: usize = (q_oo.dim().1 as f64).sqrt() as usize;
            // Reference to the transition charges between virtual-virtual orbitals.
            let q_vv: ArrayView2<f64> = self.properties.q_vv().unwrap();
            // Number of virtual orbitals.
            let n_virt: usize = (q_vv.dim().1 as f64).sqrt() as usize;
            // Reference to the screened Gamma matrix.
            let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
            // The contraction with the subpspace vectors is more complex than in the case
            // of the Coulomb part.
            // Contraction of the Gamma matrix with the o-o transition charges.
            let gamma_oo: Array2<f64> = gamma_lr
                .dot(&q_oo)
                .into_shape([self.n_atoms * n_occ, n_occ])
                .unwrap();
            // Initialization of the product of the exchange part with the subspace part.
            let mut k_x: Array2<f64> = Array::zeros(two_el.raw_dim());
            // Iteration over the subspace vectors.
            for (i, mut k) in k_x.axis_iter_mut(Axis(1)).enumerate() {
                // The current vector reshaped into the form of n_occ, n_virt
                let xi = compute_vectors
                    .slice(s![.., i])
                    .into_shape((n_occ, n_virt))
                    .unwrap();
                // The v-v transition have to be reshaped as well.
                let q_vv_r = q_vv.into_shape((self.n_atoms * n_virt, n_virt)).unwrap();
                // Contraction of the v-v transition charges with the subspace vector and the
                // and the product of the Gamma matrix wit the o-o transition charges.
                k.assign(
                    &gamma_oo.t().dot(
                        &xi.dot(&q_vv_r.t())
                            .into_shape((n_occ, self.n_atoms, n_virt))
                            .unwrap()
                            .permuted_axes([1, 0, 2])
                            .as_standard_layout()
                            .into_shape((self.n_atoms * n_occ, n_virt))
                            .unwrap(),
                    ),
                );
            }
            // The product of the Exchange part with the subspace vector is added to the Coulomb part.
            two_el = &two_el - &k_x;
        }

        // The new products are saved in the cache.
        let ax: Array2<f64> = cache.add("TDA", fock + two_el).to_owned();
        self.properties.set_cache(cache);
        // The product of the CIS-Hamiltonian with the subspace vectors is returned.
        ax
    }

    /// The preconditioner and a shift are applied to the residual vectors.
    /// The energy difference of the virtual and occupied orbitals is used as a preconditioner.
    fn precondition(&self, r_k: ArrayView1<f64>, w_k: f64) -> Array1<f64> {
        // The denominator is build from the orbital energy differences and the shift value.
        let mut denom: Array1<f64> = w_k - &self.properties.omega().unwrap();
        // Values smaller than 0.0001 are replaced by 1.0.
        denom.mapv_inplace(|x| if x < 0.0001 { 1.0 } else { x });
        &r_k / &denom
    }

    fn get_size(&self) -> usize {
        self.occ_indices.len() * self.virt_indices.len()
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

    pub const EPSILON: f64 = 1e-15;

    fn exact_tda(molecule: &System) -> (Array1<f64>, Array2<f64>) {
        let qov: ArrayView2<f64> = molecule.properties.q_ov().unwrap();
        let gamma: ArrayView2<f64> = molecule.properties.gamma().unwrap();
        let omega: ArrayView1<f64> = molecule.properties.omega().unwrap();
        let h_cis: Array2<f64> = Array2::from_diag(&omega) + 2.0 * qov.t().dot(&gamma.dot(&qov));
        h_cis.eigh(UPLO::Upper).unwrap()
    }

    fn test_tda(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let mut molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let atomic_numbers: Vec<u8> = molecule.atoms.iter().map(|atom| atom.number).collect();
        molecule.prepare_scc();
        molecule.run_scc();
        let (u, v) = molecule.run_tda();
        let (u_ref, v_ref) = exact_tda(&molecule);

        assert!(
            u.abs_diff_eq(&u_ref.slice(s![0..4]), 1e-16),
            "Molecule: {}, Eigenvalues (ref): {}  Eigenvalues: {}",
            name,
            u_ref.slice(s![0..4]),
            u
        );
    }

    #[test]
    fn tda_energies() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_tda(get_molecule(molecule, "no_lc_gs"));
        }
    }
}
