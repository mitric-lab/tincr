use crate::fmo::*;
use core::::*;
use crate::scc::gamma_approximation::gamma_gradients_atomwise_2d;
use crate::scc::*;
use crate::utils::Timer;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::RawData;
use ndarray_linalg::Solve;
use ndarray_stats::DeviationExt;
use std::iter::FromIterator;
use std::ops::{AddAssign, SubAssign};

impl<'a> SuperSystem<'a> {
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
        let gamma: ArrayView2<f64> = self.data.gamma();

        // Reference to the charge differences.
        let dq: ArrayView1<f64> = self.data.dq();

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
            let ddq: ArrayView1<f64> = pair.data.delta_dq();

            // Vector matrix product for the ddq of I
            ddq_gamma += &ddq
                .slice(s![..m_i.atoms.len()])
                .dot(&gamma.slice(s![m_i.slice.atom, 0..]));

            // Vector matrix product for the ddq of J
            ddq_gamma += &ddq
                .slice(s![m_i.atoms.len()..])
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
                            let ddq: ArrayView1<f64> = pair.data.delta_dq();

                            // Vector matrix product for the ddq of I
                            ddq_gamma_mol -= &ddq
                                .slice(s![..m_i.atoms.len()])
                                .dot(&gamma.slice(s![m_i.slice.atom, mol.slice.atom]));

                            // Vector matrix product for the ddq of J
                            ddq_gamma_mol -= &ddq
                                .slice(s![m_i.atoms.len()..])
                                .dot(&gamma.slice(s![m_j.slice.atom, mol.slice.atom]));
                        }
                    }
                    // If there is no list, nothing is done.
                    None => {}
                }

                // Reference to the transition charges.
                let q_vo: ArrayView2<f64> = mol.data.q_vo();

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
        let gamma: ArrayView2<f64> = self.data.gamma();

        // The Z-vectors are stored as Vec of Array1's.
        let mut z_vectors: Vec<Array1<f64>> = Vec::with_capacity(self.n_mol);

        // Energy differences between occupied and virtual orbitals for each monomer.
        let mut omegas: Vec<Array1<f64>> = Vec::with_capacity(self.n_mol);

        // PARALLEL
        // 1. Compute the energy differences between the orbitals
        // 2. Construction of initial Z-vectors.
        for (i, m_i) in self.monomers.iter().enumerate() {
            // Reference to the indices of occupied orbitals
            let occ_indices: &[usize] = m_i.data.occ_indices();
            // Reference to the indices of the virtual orbitals
            let virt_indices: &[usize] = m_i.data.virt_indices();
            // Reference to the orbital energies.
            let orbe: ArrayView1<f64> = m_i.data.orbe();

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
            let q_vo_i: ArrayView2<f64> = m_i.data.q_vo();
            // The A matrix of monomer I and I.
            let a_ii: Array2<f64> = Array2::from_diag(&omegas[i])
                - 4.0
                    * q_vo_i
                        .t()
                        .dot(&gamma.slice(s![m_i.slice.atom, m_i.slice.atom]).dot(&q_vo_i));
            // Solve the linear system to get the Z-vector and save it.
            z_vectors.push(a_ii.solve(&lagrangians[i]).unwrap());
        }

        let mut converged: Vec<bool> = vec![false; self.n_mol];

        // Z-vector loop
        while {
            println!("Z-vector Iteration");
            // PARALLEL
            for (i, m_i) in self.monomers.iter().enumerate() {
                // Reference to the transition charges between occupied and virtual orbitals.
                let q_vo_i: ArrayView2<f64> = m_i.data.q_vo();

                let mut x_i: Array1<f64> = lagrangians[i].clone();

                for (j, m_j) in self.monomers.iter().enumerate() {
                    // If monomer I equals J, skip the iteration.
                    if i == j {
                        continue;
                    }
                    // Reference to the transition charges of monomer J.
                    let q_vo_j: ArrayView2<f64> = m_j.data.q_vo();
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
                let z_i: Array1<f64> = a_ii.solve(&x_i).unwrap();

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
                let n_occ: usize = mol.data.n_occ();
                // Number of virtual orbitals in Monomer I.
                let n_virt: usize = mol.data.n_virt();
                // MO coefficient matrix: rows: AO indices, columns: MO indices.
                let orbs: ArrayView2<f64> = mol.data.orbs();
                // Transform the Z-Vector in the n_virt, n_occ shape.
                let z: Array2<f64> = (z_i).into_shape([n_virt, n_occ]).unwrap();
                // Basis transformation from MO basis to AO basis.
                let mut z_vector: Array2<f64> = orbs
                    .slice(s![0.., n_occ..])
                    .dot(&(z.dot(&orbs.slice(s![0.., ..n_occ]).t())));
                // Symmetrize the Z-vector.
                z_vector = 0.5 * (&z_vector + &z_vector.t());
                // Flatten the Array into a one dimensional vector.
                let z_vector: Array1<f64> = z_vector
                    .into_shape([mol.n_orbs().pow(2)])
                    .unwrap();
                // Save the transformed Z-vectors for each monomer.
                mol.data.set_z_vector(z_vector);
            });
        println!("Z-vector converged");
    }

    /// The third part of the B-matrix times the Z-vector is calculated.
    pub fn response_embedding_gradient(&mut self) -> Array1<f64> {
        // Reference to the gamma matrix.
        let gamma: ArrayView2<f64> = self.data.gamma();

        // Initialize the gradient
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);

        // Reference to the atoms
        let atoms: &[Atom] = &self.atoms;

        // PARALLEL
        for m_i in self.monomers.iter() {
            // Initialize the product of Gamma matrix with the derivative of the charges.
            let mut grad_esp: Array2<f64> = Array2::zeros([3 * self.atoms.len(), m_i.atoms.len()]);
            // The column slice of the Gamma matrix that corresponds to the Monomer I.
            let gamma_i: ArrayView2<f64> = gamma.slice(s![0.., m_i.slice.atom]);
            // Reference to the overlap matrix of Monomer I, reshaped into a vector.
            let s_i: ArrayView1<f64> = m_i.data.s().into_shape([m_i.n_orbs().pow(2)]).unwrap();

            // Reference to the Z-vector of Monomer I.
            let z_i: ArrayView1<f64> = m_i.data.z_vector();

            // Loop over all Monomers to sum up the matrix product.
            for m_j in self.monomers.iter() {
                // Reference to the charge derivatives of Monomer J
                let grad_dq_j: ArrayView2<f64> = m_j.data.grad_dq();
                // Compute the product of the charge derivative and the Gamma matrix.
                grad_esp.slice_mut(s![m_j.slice.grad, 0..]).add_assign(&grad_dq_j.dot(&gamma_i.slice(s![m_j.slice.atom, 0..])));
            }
            // The second axis is transformed from the shape of natoms_I into n_orbs_I * norbs_I, by
            // computing the dyadic tensor of the Vector.
            let grad_esp_ao: Array2<f64> = grad_atomvec_to_aomat(grad_esp, m_i.n_orbs(), &atoms[m_i.slice.atom_as_range()], 3 * self.atoms.len());

            // The product of the overlap matrix and the Z-vector is computed and contracted with
            // the gradient contribution
            gradient += &grad_esp_ao.dot(&(&z_i * &s_i));
        }
        // Returns the gradient.
        gradient
    }
}


fn grad_atomvec_to_aomat(esp_atomwise: Array2<f64>, n_orbs: usize, atoms: &[Atom], f: usize) -> Array2<f64> {
    let mut esp_ao_row: Array2<f64> = Array2::zeros([f, n_orbs]);
    let mut mu: usize = 0;
    for (atom, grad_esp_at) in atoms.iter().zip(esp_atomwise.axis_iter(Axis(1))) {
        for _ in 0..atom.n_orbs() {
            esp_ao_row.slice_mut(s![0.., mu]).add_assign(&grad_esp_at);
            mu = mu + 1;
        }
    }
    let esp_ao_column: Array3<f64> = esp_ao_row.clone().insert_axis(Axis(2));
    let esp_ao: Array3<f64> = &esp_ao_column.t().broadcast((n_orbs, n_orbs, f)).unwrap() + &esp_ao_row.t();
    return esp_ao.t().as_standard_layout().into_shape([f, n_orbs * n_orbs]).unwrap().to_owned();
}
