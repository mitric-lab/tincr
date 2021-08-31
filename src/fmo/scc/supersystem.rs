use crate::fmo::helpers::get_pair_slice;
use crate::fmo::scc::helpers::*;
use crate::fmo::scc::logging;
use crate::fmo::{Monomer, SuperSystem, Pair};
use crate::initialization::Atom;
use crate::scc::get_repulsive_energy;
use ndarray::prelude::*;
use std::ops::SubAssign;
use rayon::prelude::*;

impl SuperSystem {
    pub fn monomer_scc(&mut self, max_iter: usize) -> (f64, Array1<f64>) {
        // Vector that holds the information if the scc calculation of the individual monomer
        // is converged or not
        let mut converged: Vec<bool> = vec![false; self.n_mol];
        // charge differences of all atoms. these are needed to compute the electrostatic potential
        // that acts on the monomers.
        let mut dq: Array1<f64> = Array1::zeros([self.atoms.len()]);
        let atoms = &self.atoms;
        let scf_config = &self.config.scf;

        // charge consistent loop for the monomers
        'scf_loop: for iter in 0..max_iter {
            // the matrix vector product of the gamma matrix for all atoms and the charge differences
            // yields the electrostatic potential for all atoms. this is then converted into ao basis
            // and given to each monomer scc step
            let esp_at: Array1<f64> = self.properties.gamma().unwrap().dot(&dq);
            // for (i, mol) in self.monomers.iter_mut().enumerate() {
            //     let v_esp: Array2<f64> = atomvec_to_aomat(
            //         esp_at.slice(s![mol.slice.atom]),
            //         mol.n_orbs,
            //         &self.atoms[mol.slice.atom_as_range()],
            //     );
            //     if !converged[i] {
            //         converged[i] = mol.scc_step(
            //             &self.atoms[mol.slice.atom_as_range()],
            //             v_esp,
            //             self.config.scf,
            //         );
            //     }
            //     // save the dq's from the monomer calculation
            //     dq.slice_mut(s![mol.slice.atom])
            //         .assign(&mol.properties.dq().unwrap());
            // }

            // Parallelization
            let loop_output:Vec<bool> = self.monomers.par_iter_mut().map(|mol|{
                let v_esp: Array2<f64> = atomvec_to_aomat(
                    esp_at.slice(s![mol.slice.atom]),
                    mol.n_orbs,
                    &atoms[mol.slice.atom_as_range()],
                );
                mol.scc_step(
                    &atoms[mol.slice.atom_as_range()],
                    v_esp,
                    *scf_config,
                )
            }).collect();
            for mol in self.monomers.iter() {
                // save the dq's from the monomer calculation
                dq.slice_mut(s![mol.slice.atom])
                    .assign(&mol.properties.dq().unwrap());
            }
            converged = loop_output;

            let n_converged: usize = converged.iter().filter(|&n| *n == true).count();
            logging::fmo_monomer_iteration(iter, n_converged, self.n_mol);
            // the loop ends if all monomers are converged
            if n_converged == self.n_mol {
                break 'scf_loop;
            }
        }

        let mut monomer_energies: f64 = 0.0;
        for mol in self.monomers.iter_mut() {
            let scf_energy: f64 = mol.properties.last_energy().unwrap();
            let e_rep: f64 = get_repulsive_energy(
                &self.atoms[mol.slice.atom_as_range()],
                mol.n_atoms,
                &mol.vrep,
            );
            mol.properties.set_last_energy(scf_energy + e_rep);
            monomer_energies += scf_energy + e_rep;
        }
        return (monomer_energies, dq);
    }

    pub fn pair_scc(&mut self, dq: ArrayView1<f64>) -> f64 {
        // this is the electrostatic potential that acts on the pairs
        // PARALLEL: The dot product could be parallelized and then it is not necessary to convert
        // the ArrayView into an owned ArrayBase
        //let esp_at: Array1<f64> = self.properties.gamma().unwrap().dot(&dq);
        for mol in self.monomers.iter_mut() {
            let mut esp_slice: Array1<f64> = self
                .properties
                .gamma()
                .unwrap()
                .slice(s![mol.slice.atom, 0..])
                .dot(&dq);
            esp_slice.sub_assign(
                &self
                    .properties
                    .gamma()
                    .unwrap()
                    .slice(s![mol.slice.atom, mol.slice.atom])
                    .dot(&mol.properties.dq().unwrap()),
            );
            // mol.properties
            //     .set_esp_q(esp_at.slice(s![mol.slice.atom]).to_owned());
            mol.properties.set_esp_q(esp_slice);
        }
        // the final scc energy of all pairs. the energies of the corresponding monomers will be
        // subtracted from this energy
        let mut pair_energies: f64 = 0.0;

        let atoms: &[Atom] = &self.atoms[..];
        // SCC iteration for each pair that is treated exact
        // for pair in self.pairs.iter_mut() {
        //     // Get references to the corresponding monomers
        //     let m_i: &Monomer = &self.monomers[pair.i];
        //     let m_j: &Monomer = &self.monomers[pair.j];
        //
        //     // The atoms are in general a non-contiguous range of the atoms
        //     let pair_atoms: Vec<Atom> =
        //         get_pair_slice(&atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());
        //     pair.prepare_scc(&pair_atoms[..], m_i, m_j);
        //
        //     // do the SCC iterations
        //     pair.run_scc(&*pair_atoms, self.config.scf);
        //
        //     // and compute the SCC energy
        //     pair_energies += pair.properties.last_energy().unwrap()
        //         - m_i.properties.last_energy().unwrap()
        //         - m_j.properties.last_energy().unwrap();
        //
        //     // Difference between density matrix of the pair and the density matrix of the
        //     // corresponding monomers
        //     let p: ArrayView2<f64> = pair.properties.p().unwrap();
        //     let mut delta_p: Array2<f64> = p.to_owned();
        //     delta_p
        //         .slice_mut(s![0..m_i.n_orbs, 0..m_i.n_orbs])
        //         .sub_assign(&m_i.properties.p().unwrap());
        //     delta_p
        //         .slice_mut(s![m_i.n_orbs.., m_i.n_orbs..])
        //         .sub_assign(&m_j.properties.p().unwrap());
        //     pair.properties.set_delta_p(delta_p);
        // }
        // return pair_energies;

        // Parallelization
        let monomers:&Vec<Monomer> = &self.monomers;
        let scf_config = &self.config.scf;
        let pair_energy:Vec<f64> = self.pairs.par_iter_mut().map(|pair| {
            // Get references to the corresponding monomers
            let m_i: &Monomer = &monomers[pair.i];
            let m_j: &Monomer = &monomers[pair.j];

            // The atoms are in general a non-contiguous range of the atoms
            let pair_atoms: Vec<Atom> =
                get_pair_slice(&atoms, m_i.slice.atom_as_range(), m_j.slice.atom_as_range());
            pair.prepare_scc(&pair_atoms[..], m_i, m_j);

            // do the SCC iterations
            pair.run_scc(&*pair_atoms, *scf_config);

            // and compute the SCC energy
            let pair_energ:f64 = pair.properties.last_energy().unwrap()
                - m_i.properties.last_energy().unwrap()
                - m_j.properties.last_energy().unwrap();

            // Difference between density matrix of the pair and the density matrix of the
            // corresponding monomers
            let p: ArrayView2<f64> = pair.properties.p().unwrap();
            let mut delta_p: Array2<f64> = p.to_owned();
            delta_p
                .slice_mut(s![0..m_i.n_orbs, 0..m_i.n_orbs])
                .sub_assign(&m_i.properties.p().unwrap());
            delta_p
                .slice_mut(s![m_i.n_orbs.., m_i.n_orbs..])
                .sub_assign(&m_j.properties.p().unwrap());
            pair.properties.set_delta_p(delta_p);

            pair_energ
        }).collect();
        let pair_energy:Array1<f64> = Array::from(pair_energy);

        return pair_energy.sum();
    }

    pub fn embedding_energy(&self) -> f64 {
        // Reference to the Gamma matrix of the full system.
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        // The embedding energy is initialized to zero.
        let mut embedding: f64 = 0.0;
        // for pair in self.pairs.iter() {
        //     // Reference to Monomer I.
        //     let m_i: &Monomer = &self.monomers[pair.i];
        //     // Reference to Monomer J.
        //     let m_j: &Monomer = &self.monomers[pair.j];
        //     // Reference to the charge differences of Monomer I.
        //     let dq_i: ArrayView1<f64> = m_i.properties.dq().unwrap();
        //     // Reference to the charge differences of Monomer J.
        //     let dq_j: ArrayView1<f64> = m_j.properties.dq().unwrap();
        //     // Electrostatic potential that acts on I without the self interaction with I.
        //     let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
        //     // ESP that acts on J without self-interaction.
        //     let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
        //     // Difference between the charge differences of the pair and the corresp. monomers
        //     let ddq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();
        //     // The interaction with the other Monomer in the pair is subtracted.
        //     let esp_q_i: Array1<f64> =
        //         &esp_q_i - &gamma.slice(s![m_i.slice.atom, m_j.slice.atom]).dot(&dq_j);
        //     let esp_q_j: Array1<f64> =
        //         &esp_q_j - &gamma.slice(s![m_j.slice.atom, m_i.slice.atom]).dot(&dq_i);
        //     // The embedding energy for Monomer I in the pair is computed.
        //     embedding += esp_q_i.dot(&ddq.slice(s![..m_i.n_atoms]));
        //     // The embedding energy for Monomer J in the pair is computed.
        //     embedding += esp_q_j.dot(&ddq.slice(s![m_i.n_atoms..]));
        // }
        //
        // return embedding;

        // Parallelization
        let embedding_energies:Vec<f64> = self.pairs.par_iter().map(|pair| {
            // Reference to Monomer I.
            let m_i: &Monomer = &self.monomers[pair.i];
            // Reference to Monomer J.
            let m_j: &Monomer = &self.monomers[pair.j];
            // Reference to the charge differences of Monomer I.
            let dq_i: ArrayView1<f64> = m_i.properties.dq().unwrap();
            // Reference to the charge differences of Monomer J.
            let dq_j: ArrayView1<f64> = m_j.properties.dq().unwrap();
            // Electrostatic potential that acts on I without the self interaction with I.
            let esp_q_i: ArrayView1<f64> = m_i.properties.esp_q().unwrap();
            // ESP that acts on J without self-interaction.
            let esp_q_j: ArrayView1<f64> = m_j.properties.esp_q().unwrap();
            // Difference between the charge differences of the pair and the corresp. monomers
            let ddq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();
            // The interaction with the other Monomer in the pair is subtracted.
            let esp_q_i: Array1<f64> =
                &esp_q_i - &gamma.slice(s![m_i.slice.atom, m_j.slice.atom]).dot(&dq_j);
            let esp_q_j: Array1<f64> =
                &esp_q_j - &gamma.slice(s![m_j.slice.atom, m_i.slice.atom]).dot(&dq_i);
            // The embedding energy for Monomer I in the pair is computed.
            let mut embedd:f64 = esp_q_i.dot(&ddq.slice(s![..m_i.n_atoms]));
            // The embedding energy for Monomer J in the pair is computed.
            embedd += esp_q_j.dot(&ddq.slice(s![m_i.n_atoms..]));
            embedd
        }).collect();
        let embedding_energies:Array1<f64> = Array::from(embedding_energies);

        return embedding_energies.sum();
    }

    pub fn esd_pair_energy(&mut self) -> f64 {
        let mut esd_energy: f64 = 0.0;
        // for esd_pair in self.esd_pairs.iter() {
        //     let m_i: &Monomer = &self.monomers[esd_pair.i];
        //     let m_j: &Monomer = &self.monomers[esd_pair.j];
        //     esd_energy += m_i
        //         .properties
        //         .dq()
        //         .unwrap()
        //         .dot(
        //             &self
        //                 .properties
        //                 .gamma()
        //                 .unwrap()
        //                 .slice(s![m_i.slice.atom, m_j.slice.atom]),
        //         )
        //         .dot(&m_j.properties.dq().unwrap());
        // }
        // return esd_energy;

        // Parallelization
        let esd_energies:Vec<f64> = self.esd_pairs.par_iter().map(|esd_pair| {
            let m_i: &Monomer = &self.monomers[esd_pair.i];
            let m_j: &Monomer = &self.monomers[esd_pair.j];
            let esd_energy:f64 = m_i
                .properties
                .dq()
                .unwrap()
                .dot(
                    &self
                        .properties
                        .gamma()
                        .unwrap()
                        .slice(s![m_i.slice.atom, m_j.slice.atom]),
                )
                .dot(&m_j.properties.dq().unwrap());
            esd_energy
        }).collect();
        let esd_energies:Array1<f64> = Array::from(esd_energies);

        return esd_energies.sum();
    }
}
