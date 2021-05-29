use crate::fmo::*;
use crate::initialization::*;
use crate::scc::gamma_approximation::gamma_gradients_atomwise_2d;
use crate::scc::*;
use crate::utils::Timer;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::RawData;
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
    /// in which the monomer is in one of the pairs is substracted. This will reduce the scaling to
    /// O(N_M' + N_P).
    ///
    /// [1]: J. Phys. Chem. Lett. 2015, 6, 5034--5039, DOI: 10.1021/acs.jpclett.5b02490
    pub fn response_lagrangian(&mut self) {
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
                None => {m_to_p.insert(pair.i, vec![pair]);},
            }

            // The pair for monomer J is saved
            match m_to_p.get_mut(&pair.j) {
                Some(pair_list) => pair_list.push(pair),
                None => {m_to_p.insert(pair.j, vec![pair]);}
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

        // Loop over all monomers to compute the product with the transition charges
        for (idx, mol) in self.monomers.iter().enumerate() {
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
            //let q_vo: ArrayView2<f64> = self.properties.q_vo().unwrap();

            // Vector matrix product of the pre computed term and the transition charges.
            //let l: Array1<f64> = ddq_gamma_mol.dot(&q_vo);

            // Save the Lagrangian for this monomer
            //mol.properties.set_l(l);
        }
    }
}
