use crate::fmo::{SuperSystem, Monomer};
use crate::fmo::scc::helpers::*;
use crate::fmo::scc::logging;
use ndarray::prelude::*;
use crate::scc::get_repulsive_energy;
use std::ops::SubAssign;
use crate::initialization::Atom;


impl SuperSystem {
    pub fn monomer_scc(&mut self, max_iter: usize) -> (f64, Array1<f64>) {
        // Vector that holds the information if the scc calculation of the individual monomer
        // is converged or not
        let mut converged: Vec<bool> = vec![false; self.n_mol];
        // charge differences of all atoms. these are needed to compute the electrostatic potential
        // that acts on the monomers.
        let mut dq: Array1<f64> = Array1::zeros([self.atoms.len()]);
        // charge consistent loop for the monomers
        'scf_loop: for iter in 0..max_iter {
            // the matrix vector product of the gamma matrix for all atoms and the charge differences
            // yields the electrostatic potential for all atoms. this is then converted into ao basis
            // and given to each monomer scc step
            let esp_at: Array1<f64> = self.properties.gamma().unwrap().dot(&dq);
            for (i, mol) in self.monomers.iter_mut().enumerate() {
                let v_esp: Array2<f64> =
                    atomvec_to_aomat(esp_at.slice(s![mol.slice.atom]), mol.n_orbs, &mol.atoms);
                if !converged[i] {
                    println!("esp {}", &esp_at);
                    converged[i] = mol.scc_step(&self.atoms[mol.slice.atom_as_range()], v_esp);
                }
                // save the dq's from the monomer calculation
                dq.slice_mut(s![mol.slice.atom])
                    .assign(&mol.properties.dq().unwrap());
            }
            let n_converged: usize = converged.iter().filter(|&n| *n == true).count();
            logging::fmo_monomer_iteration(iter, n_converged, self.n_mol);
            // the loop ends if all monomers are converged
            if n_converged == self.n_mol {
                break 'scf_loop;
            }
        }
        let mut monomer_energies: f64 = 0.0;
        for mol in self.monomers.iter_mut() {
            let scf_energy: f64 =  mol.properties.last_energy().unwrap();
            let e_rep: f64 = get_repulsive_energy(&mol.atoms, mol.n_atoms, &mol.vrep);
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

        // SCC iteration for each pair that is treated exact
        for pair in self.pairs.iter_mut() {

            // Get references to the corresponding monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];

            // The atoms are in general a non-contiguous range of the atoms
            let pair_atoms: Vec<Atom> = self.atoms[m_i.slice.atom_as_range()]
                .clone()
                .append(self.atoms[m_j.slice.atom_as_range()].clone());

            pair.prepare_scc(&*pair_atoms, m_i, m_j);

            // do the SCC iterations
            pair.run_scc(&*pair_atoms);

            // and compute the SCC energy
            pair_energies += pair.properties.last_energy().unwrap()
                - m_i.properties.last_energy().unwrap()
                - m_j.properties.last_energy().unwrap();
        }
        return pair_energies;
    }

    pub fn embedding_energy(&self) -> f64 {
        let mut embedding: f64 = 0.0;
        for pair in self.pairs.iter() {
            // and compute the SCC energy
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];
            embedding += m_i
                .properties
                .esp_q()
                .unwrap()
                .dot(&pair.properties.delta_dq().unwrap().slice(s![..m_i.n_atoms]));
            embedding += m_j
                .properties
                .esp_q()
                .unwrap()
                .dot(&pair.properties.delta_dq().unwrap().slice(s![m_i.n_atoms..]));
            embedding -= self
                .properties
                .gamma()
                .unwrap()
                .slice(s![m_i.slice.atom, m_j.slice.atom])
                .dot(&m_j.properties.dq().unwrap())
                .dot(&pair.properties.delta_dq().unwrap().slice(s![..m_i.n_atoms]));
            embedding -= self
                .properties
                .gamma()
                .unwrap()
                .slice(s![m_j.slice.atom, m_i.slice.atom])
                .dot(&m_i.properties.dq().unwrap())
                .dot(&pair.properties.delta_dq().unwrap().slice(s![m_i.n_atoms..]));
        }
        return embedding;
    }

    pub fn esd_pair_energy(&mut self) -> f64 {
        let mut esd_energy: f64 = 0.0;
        for esd_pair in self.esd_pairs.iter() {
            let m_i: &Monomer = &self.monomers[esd_pair.i];
            let m_j: &Monomer = &self.monomers[esd_pair.j];
            esd_energy += m_i
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
        }
        return esd_energy;
    }

}