use crate::fmo::{Monomer, Pair, SuperSystem};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_ao_wise, gamma_atomwise, gamma_atomwise_ab};
use crate::scc::h0_and_s::{h0_and_s, h0_and_s_ab};
use crate::scc::mixer::{BroydenMixer, Mixer};
use crate::scc::mulliken::mulliken;
use crate::scc::scc_routine::{RestrictedSCC, SCCError};
use crate::scc::{
    construct_h1, density_matrix, density_matrix_ref, get_electronic_energy, get_repulsive_energy,
    lc_exact_exchange,
};
use approx::AbsDiffEq;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::prelude::*;
use ndarray::stack;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use ndarray_stats::QuantileExt;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use std::ops::SubAssign;
use crate::utils::Timer;
use log::info;

impl RestrictedSCC for SuperSystem {
    ///  To run the SCC calculation of the FMO [SuperSystem] the following properties need to be set:
    /// For each [Monomer]
    /// - H0
    /// - S: overlap matrix in AO basis
    /// - Gamma matrix (and long-range corrected Gamma matrix if we use LRC)
    /// - If there are no charge differences, `dq`, from a previous calculation
    ///  they are initialized to zeros
    /// - the density matrix and reference density matrix
    fn prepare_scc(&mut self) {
        // prepare all individual monomers
        self.monomers
            .par_iter_mut()
            .for_each(|molecule: &mut Monomer| {
                molecule.prepare_scc();
            });
        println!("FINISHED");
    }

    fn run_scc(&mut self) -> Result<f64, SCCError> {
        let timer: Timer = Timer::start();
        // SCC settings from the user input
        let temperature: f64 = self.config.scf.electronic_temperature;
        let max_iter: usize = self.config.scf.scf_max_cycles;
        print_fmo_scc_init(max_iter);
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
                    converged[i] = mol.scc_step(v_esp);
                }
                // save the dq's from the monomer calculation
                dq.slice_mut(s![mol.slice.atom])
                    .assign(&mol.properties.dq().unwrap());
            }
            let n_converged: usize = converged.iter().filter(|&n| *n == true).count();
            print_fmo_monomer_iteration(iter, n_converged, self.n_mol);
            // the loop ends if all monomers are converged
            if n_converged == self.n_mol {
                break 'scf_loop;
            }
        }
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
        // SCC iteration for each pair that is treated exact
        for pair in self.pairs.iter_mut() {
            pair.prepare_scc(&self.monomers[pair.i], &self.monomers[pair.j]);
            pair.run_scc();
        }
        // Assembling of the energy following Eq. 11 in
        // https://pubs.acs.org/doi/pdf/10.1021/ct500489d
        // E = sum_I^N E_I^ + sum_(I>J)^N ( E_(IJ) - E_I - E_J ) + sum_(I>J)^(N) DeltaE_(IJ)^V
        let mut monomer_energies: f64 = 0.0;
        for mol in self.monomers.iter_mut() {
            let scf_energy: f64 =  mol.properties.last_energy().unwrap();
            let e_rep: f64 = get_repulsive_energy(&mol.atoms, mol.n_atoms, &mol.vrep);
            mol.properties.set_last_energy(scf_energy + e_rep);
            monomer_energies += scf_energy + e_rep;
        }
        let mut pair_energies: f64 = 0.0;
        let mut embedding: f64 = 0.0;
        for pair in self.pairs.iter() {
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];
            pair_energies += pair.properties.last_energy().unwrap()
                - m_i.properties.last_energy().unwrap()
                - m_j.properties.last_energy().unwrap();
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
            println!("ddq {}", &pair.properties.delta_dq().unwrap());
        }
        let mut esd_pair_energies: f64 = 0.0;
        for esd_pair in self.esd_pairs.iter() {
            let m_i: &Monomer = &self.monomers[esd_pair.i];
            let m_j: &Monomer = &self.monomers[esd_pair.j];
            esd_pair_energies += m_i
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
        let total_energy: f64 = monomer_energies + pair_energies + embedding + esd_pair_energies;
        self.properties.set_dq(dq);
        print_fmo_scc_end(timer, monomer_energies, pair_energies, embedding, esd_pair_energies);
        Ok(total_energy)
    }
}




