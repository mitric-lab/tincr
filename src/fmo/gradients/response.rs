use crate::fmo::*;
use crate::initialization::*;
use crate::scc::gamma_approximation::gamma_gradients_atomwise_2d;
use crate::scc::*;
use crate::utils::Timer;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::RawData;
use ndarray_linalg::SolveH;
use ndarray_stats::DeviationExt;
use std::iter::FromIterator;
use std::ops::{AddAssign, SubAssign};

impl SuperSystem {
    /// Compute the Lagrangian of the response contribution to the FMO gradient.
    ///
    /// The lagrangian is calculated following Eq. 5 and 7 of Ref. [1]. In general a double loop
    /// through the monomers and the real pairs is necessary. This would lead to a O(N_M x N_P) scaling,
    /// where N_M/N_P is the number of monomers/pairs, respectively. Instead, first the product of
    /// difference charge differences and the gamma matrix is calculated for all pairs. Afterwards,
    /// the lagrangian is computed for each monomer from this pre calculated product and the term
    /// in which the monomer is in one of the pairs is subtracted. This will reduce the scaling to
    /// O(N_M' + N_P).
    ///
    /// Requires (precomputed quantities):
    ///  - SuperSystem: dq, gamma
    ///  - Monomer's: q_vo,
    ///  - Pair's: ddq
    ///
    /// [1]: J. Phys. Chem. Lett. 2015, 6, 5034--5039, DOI: 10.1021/acs.jpclett.5b02490
    fn response_lagrangian(&mut self) -> Vec<Array1<f64>> {
        // Reference to the gamma matrix.
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();

        // Reference to the charge differences.
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();

        // HashMap that maps the monomers to the pair in which it is included.
        let mut m_to_p: HashMap<usize, Vec<&Pair>> = HashMap::new();

        // Product of gamma and the difference of charge differences
        let mut ddq_gamma: Array1<f64> = Array1::zeros([self.atoms.len()]);

        // Loop over all pairs to compute the product of ddq times gamma
        for pair in self.pairs.iter() {
            // Reference to the monomer I and J
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];

            // The pair for monomer I is saved
            match m_to_p.get_mut(&pair.i) {
                Some(ref mut pair_list) => pair_list.push(pair),
                None => {
                    m_to_p.insert(pair.i, vec![pair]);
                }
            }

            // The pair for monomer J is saved
            match m_to_p.get_mut(&pair.j) {
                Some(pair_list) => pair_list.push(pair),
                None => {
                    m_to_p.insert(pair.j, vec![pair]);
                }
            }

            // Reference to the difference of charge differences of the pair.
            let ddq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

            // Vector matrix product for the ddq of I
            ddq_gamma += &ddq
                .slice(s![..m_i.n_atoms])
                .dot(&gamma.slice(s![m_i.slice.atom, 0..]));

            // Vector matrix product for the ddq of J
            ddq_gamma += &ddq
                .slice(s![m_i.n_atoms..])
                .dot(&gamma.slice(s![m_j.slice.atom, 0..]));
        }

        // PARALLEL
        // Loop over all monomers to compute the product with the transition charges
        let lagrangians: Vec<Array1<f64>> = self
            .monomers
            .iter()
            .enumerate()
            .map(|(idx, mol)| {
                // New array for the corrected ddq.
                let mut ddq_gamma_mol: Array1<f64> = ddq_gamma.slice(s![mol.slice.atom]).to_owned();

                match m_to_p.get_mut(&idx) {
                    // If this monomer is part of pairs, then it is necessary to subtract their interaction
                    Some(pair_list) => {
                        // Possible duplicates are removed.
                        pair_list.dedup();
                        // Iteration over the pairs that belongs to the current monomer
                        for pair in pair_list.iter() {
                            // Reference to the monomer I and J
                            let m_i: &Monomer = &self.monomers[pair.i];
                            let m_j: &Monomer = &self.monomers[pair.j];

                            // Reference to the difference of charge differences of the pair.
                            let ddq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();

                            // Vector matrix product for the ddq of I
                            ddq_gamma_mol -= &ddq
                                .slice(s![..m_i.n_atoms])
                                .dot(&gamma.slice(s![m_i.slice.atom, mol.slice.atom]));

                            // Vector matrix product for the ddq of J
                            ddq_gamma_mol -= &ddq
                                .slice(s![m_i.n_atoms..])
                                .dot(&gamma.slice(s![m_j.slice.atom, mol.slice.atom]));
                        }
                    }
                    // If there is no list, nothing is done.
                    None => {}
                }

                // Reference to the transition charges.
                let q_vo: ArrayView2<f64> = mol.properties.q_vo().unwrap();

                // Vector matrix product of the pre computed term and the transition charges.
                4.0 * ddq_gamma_mol.dot(&q_vo)
            })
            .collect();

        // Returns the Lagrangian's for each monomer.
        lagrangians
    }

    pub fn self_consistent_z_vector(&mut self, epsilon: f64) {
        // Compute the initial Lagrangian's for each monomer.
        let mut lagrangians: Vec<Array1<f64>> = self.response_lagrangian();

        // Reference to the gamma matrix.
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();

        // The Z-vectors are stored as Vec of Array1's.
        let mut z_vectors: Vec<Array1<f64>> = Vec::with_capacity(self.n_mol);

        // Energy differences between occupied and virtual orbitals for each monomer.
        let mut omegas: Vec<Array1<f64>> = Vec::with_capacity(self.n_mol);

        // PARALLEL
        // 1. Compute the energy differences between the orbitals
        // 2. Construction of initial Z-vectors.
        for (i, m_i) in self.monomers.iter().enumerate() {
            // Reference to the indices of occupied orbitals
            let occ_indices: &[usize] = m_i.properties.occ_indices().unwrap();
            // Reference to the indices of the virtual orbitals
            let virt_indices: &[usize] = m_i.properties.virt_indices().unwrap();
            // Reference to the orbital energies.
            let orbe: ArrayView1<f64> = m_i.properties.orbe().unwrap();

            let mut omega: Array2<f64> = Array2::zeros([occ_indices.len(), virt_indices.len()]);

            for (ii, occ) in occ_indices.into_iter().enumerate() {
                for (aa, virt) in virt_indices.into_iter().enumerate() {
                    omega[[ii, aa]] = orbe[*occ] - orbe[*virt];
                }
            }
            omegas.push(
                omega
                    .into_shape([occ_indices.len() * virt_indices.len()])
                    .unwrap(),
            );

            // Reference to the transition charges between occupied and virtual orbitals.
            let q_vo_i: ArrayView2<f64> = m_i.properties.q_vo().unwrap();
            // The A matrix of monomer I and I.
            let a_ii: Array2<f64> = Array2::from_diag(&omegas[i])
                - 4.0
                    * q_vo_i
                        .t()
                        .dot(&gamma.slice(s![m_i.slice.atom, m_i.slice.atom]).dot(&q_vo_i));
            // Solve the linear system to get the Z-vector and save it.
            z_vectors.push(a_ii.solveh(&lagrangians[i]).unwrap());
        }

        let mut converged: Vec<bool> = vec![false; self.n_mol];

        // Z-vector loop
        while {
            println!("Z-vector Iteration");
            // PARALLEL
            for (i, m_i) in self.monomers.iter().enumerate() {
                // Reference to the transition charges between occupied and virtual orbitals.
                let q_vo_i: ArrayView2<f64> = m_i.properties.q_vo().unwrap();

                let mut x_i: Array1<f64> = lagrangians[i].clone();

                for (j, m_j) in self.monomers.iter().enumerate() {
                    // If monomer I equals J, skip the iteration.
                    if i == j {
                        continue;
                    }
                    // Reference to the transition charges of monomer J.
                    let q_vo_j: ArrayView2<f64> = m_j.properties.q_vo().unwrap();
                    // The A matrix between monomer J and I.
                    let a_ji: Array2<f64> = -4.0
                        * q_vo_j
                            .t()
                            .dot(&gamma.slice(s![m_j.slice.atom, m_i.slice.atom]).dot(&q_vo_i));
                    // Update the Lagrangian of monomer I.
                    x_i -= &a_ji.t().dot(&z_vectors[j]);
                }

                // The A matrix of monomer I.
                let a_ii: Array2<f64> = Array2::from_diag(&omegas[i])
                    - 4.0
                        * q_vo_i
                            .t()
                            .dot(&gamma.slice(s![m_i.slice.atom, m_i.slice.atom]).dot(&q_vo_i));

                // Compute the new Z-vector.
                let z_i: Array1<f64> = a_ii.solveh(&x_i).unwrap();

                println!("RMSD {}", z_i.root_mean_sq_err(&z_vectors[i]).unwrap());
                // Check if the Z-vector is converged.
                converged[i] = z_i.root_mean_sq_err(&z_vectors[i]).unwrap() < epsilon;

                // The new Z-vector is saved.
                z_vectors[i] = z_i;
            }
            converged.contains(&false)
        } {}

        // The Z-vectors are transformed into the AO-basis. This allows that the B-matrix can be
        // computed directly in the AO-basis and it is easier to use the H' and S' matrices in the
        // routine of the monomer gradients. PARALLEL
        self.monomers
            .iter_mut()
            .zip(z_vectors.into_iter())
            .for_each(|(mol, z_i)| {
                // Number of occupied orbitals in Monomer I.
                let n_occ: usize = mol.properties.occ_indices().unwrap().len();
                // Number of virtual orbitals in Monomer I.
                let n_virt: usize = mol.properties.virt_indices().unwrap().len();
                // MO coefficient matrix: rows: AO indices, columns: MO indices.
                let orbs: ArrayView2<f64> = mol.properties.orbs().unwrap();
                // Transform the Z-Vector in the n_virt, n_occ shape.
                let z: Array2<f64> = (z_i).into_shape([n_virt, n_occ]).unwrap();
                // Basis transformation from MO basis to AO basis.
                let z_vector: Array1<f64> = orbs
                    .slice(s![0.., n_occ..])
                    .dot(&(z.dot(&orbs.slice(s![0.., ..n_occ]).t())))
                    .into_shape([mol.n_orbs.pow(2)])
                    .unwrap();
                // Save the transformed Z-vectors for each monomer.
                mol.properties.set_z_vector(z_vector);
            });
        println!("Z-vector converged");
    }

    /// The third part of the B-matrix times the Z-vector is calculated.
    pub fn response_embedding_gradient(&mut self) {
        // Reference to the gamma matrix.
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();


    }
}
