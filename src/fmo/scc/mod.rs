mod pair;
mod monomer;
pub(crate) mod helpers;
mod logging;
mod supersystem;


use crate::fmo::{Monomer, Pair, SuperSystem};
use crate::fmo::scc::supersystem::*;
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
use crate::fmo::scc::helpers::{atomvec_to_aomat, get_dispersion_energy};

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
        let n_orbs: usize = self.monomers.iter().map(|mol| mol.n_orbs).sum();
        let (s, h0) = h0_and_s(n_orbs, &self.atoms, &self.monomers[0].slako);
        self.properties.set_s(s);
        self.properties.set_gamma(gamma_atomwise(&self.gammafunction, &self.atoms, self.atoms.len()));
        if self.gammafunction_lc.is_some(){
            self.properties.set_gamma_lr(gamma_atomwise(&self.gammafunction_lc.clone().unwrap(), &self.atoms, self.atoms.len()));
        }

        let atoms: &[Atom] = &self.atoms;
        // self.properties.set_gamma(gamma_atomwise(&self.gammafunction, &atoms, self.atoms.len()));
        self.monomers
            .par_iter_mut()
            .for_each(|mol: &mut Monomer| {
                mol.prepare_scc(&atoms[mol.slice.atom_as_range()]);
            });
    }

    fn run_scc(&mut self) -> Result<f64, SCCError> {
        let timer: Timer = Timer::start();

        // SCC settings from the user input
        let temperature: f64 = self.config.scf.electronic_temperature;
        let max_iter: usize = self.config.scf.scf_max_cycles;
        logging::fmo_scc_init(max_iter);

        let mut e_disp: f64 = 0.0;
        if self.config.dispersion.use_dispersion{
            e_disp = get_dispersion_energy(&self.atoms, &self.config.dispersion);
        }

        // Assembling of the energy following Eq. 11 in
        // https://pubs.acs.org/doi/pdf/10.1021/ct500489d
        // E = sum_I^N E_I^ + sum_(I>J)^N ( E_(IJ) - E_I - E_J ) + sum_(I>J)^(N) DeltaE_(IJ)^V

        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        // println!("Time monomers {}",timer);

        // Do the SCC-calculation for each pair individually
        let mut pair_energies: f64 = self.pair_scc(dq.view());

        // println!("time pairs {}",timer);

        // Compute the embedding energy from all pairs
        let mut embedding: f64 = self.embedding_energy();

        // println!("time embedding {}",timer);

        // Compute the energy from pairs that are far apart. The electrostatic dimer approximation
        // is used in this case.
        let mut esd_pair_energies: f64 = self.esd_pair_energy();

        // println!("time esd {}",timer);

        // Sum up all the individual energies
        let total_energy: f64 = monomer_energies + pair_energies + embedding + esd_pair_energies +e_disp;

        // Save the charge differences of all monomers in the SuperSystem
        self.properties.set_dq(dq);

        // Print information of the SCC-routine
        logging::fmo_scc_end(timer, monomer_energies, pair_energies, embedding, esd_pair_energies, e_disp);

        self.properties.set_last_energy(total_energy);
        // Return the energy
        Ok(total_energy)
    }
}
