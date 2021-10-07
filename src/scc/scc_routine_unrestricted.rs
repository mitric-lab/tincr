use crate::initialization::system::System;
use crate::scc::mixer::BroydenMixer;
use crate::scc::gamma_approximation::{gamma_atomwise, gamma_ao_wise};
use crate::scc::h0_and_s::h0_and_s;
use crate::scc::helpers::density_matrix_ref;
use crate::scc::level_shifting::LevelShifter;
use crate::scc::mixer::*;
use crate::scc::mulliken::mulliken;
use crate::scc::logging::*;
use crate::scc::{fermi_occupation, get_repulsive_energy, construct_h1,construct_h_magnetization, density_matrix, enable_level_shifting, get_electronic_energy, get_electronic_energy_unrestricted, lc_exact_exchange, get_frontier_orbitals};
use crate::utils::Timer;
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use std::fmt;

#[derive(Debug, Clone)]
pub struct SCCError {
    pub message: String,
    iteration: usize,
    energy_diff: f64,
    charge_diff_alpha: f64,
    charge_diff_beta: f64,
}

impl SCCError{
    pub fn new(iter: usize, energy_diff: f64, charge_diff_alpha:f64,charge_diff_beta:f64) -> Self {
        let message: String = format! {"SCC-Routine failed in Iteration: {}. The charge\
                                        difference at the last iteration was {} and the energy\
                                        difference was {}",
                                       iter,
                                       charge_diff_alpha,
                                       charge_diff_beta};
        Self {
            message,
            iteration: iter,
            energy_diff: energy_diff,
            charge_diff_alpha: charge_diff_alpha,
            charge_diff_beta:charge_diff_beta,
        }
    }
}

impl fmt::Display for SCCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write! {f, "{}", self.message.as_str()}
    }
}

impl std::error::Error for SCCError {
    fn description(&self) -> &str {
        self.message.as_str()
    }
}

/// Trait that optimizes the Kohn-Sham orbitals iteratively by employing the
/// spin-unrestricted (spin-polarized) self-consistent charge scheme to find the ground state energy.
/// Two sets of charges/charge differences are used
pub trait UnrestrictedSCC {
    fn prepare_unrestricted_scc(&mut self);
    fn run_unrestricted_scc(&mut self) -> Result<f64, SCCError>;
}

impl<'a> UnrestrictedSCC for System{
    ///  To run the SCC calculation the following properties in the molecule need to be set:
    /// - H0
    /// - S: overlap matrix in AO basis
    /// - Gamma matrix (and long-range corrected Gamma matrix if we use LRC)
    /// - If there are no charge differences, `dq`, from a previous calculation
    ///  they are initialized to zeros
    /// - the density matrix and reference density matrix
    fn prepare_unrestricted_scc(&mut self) {
        // get H0 and S
        let (s, h0): (Array2<f64>, Array2<f64>) =
            h0_and_s(self.n_orbs, &self.atoms,  &self.slako);
        // and save it in the molecule properties
        self.data.set_h0(h0);
        self.data.set_s(s);
        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = self.atoms.iter().map(|atom| atom.number).collect();
        self.data.set_atomic_numbers(atomic_numbers);
        // get the gamma matrix
        let gamma: Array2<f64> = gamma_atomwise(
            &self.gammafunction,
            &self.atoms,
            self.n_atoms,
        );
        // and save it as a `Property`
        self.data.set_gamma(gamma);

        // if the system contains a long-range corrected Gammafunction the gamma matrix will be computed
        if self.gammafunction_lc.is_some() {
            let (gamma_lr, gamma_lr_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.data.set_gamma_lr(gamma_lr);
            self.data.set_gamma_lr_ao(gamma_lr_ao);
        }

        // if this is the first SCC calculation the charge differences will be initialized to zeros
        if !self.properties.contains_key("dq_alpha") {
            self.data.set_dq_alpha(Array1::zeros(self.n_atoms));
        }
        if !self.properties.contains_key("dq_beta") {
            self.data.set_dq_beta(Array1::zeros(self.n_atoms));
        }

        // this is also only needed in the first SCC calculation
        if !self.properties.contains_key("ref_density_matrix") {
            self.data.set_p_ref(density_matrix_ref(self.n_orbs, &self.atoms));
        }

        // in the first SCC calculation the density matrix is set to the reference density matrix
        if !self.properties.contains_key("P_alpha") {
            self.data.set_p_alpha(self.data.p_ref().to_owned());
        }
        if !self.properties.contains_key("P_beta") {
            self.data.set_p_beta(self.data.p_ref().to_owned());
        }
    }

    // SCC Routine for a single molecule and for spin-unpolarized systems
    fn run_unrestricted_scc(&mut self) -> Result<f64, SCCError> {
        let timer: Timer = Timer::start();

        // SCC settings from the user input
        let temperature: f64 = self.config.scf.electronic_temperature;
        let max_iter: usize = self.config.scf.scf_max_cycles;
        let scf_charge_conv: f64 = self.config.scf.scf_charge_conv;
        let scf_energy_conv: f64 = self.config.scf.scf_energy_conv;

        // the properties that are changed during the SCC routine are taken
        // and will be inserted at the end of the SCC routine
        let mut p_alpha: Array2<f64> = self.data.take_p_alpha();
        let mut p_beta: Array2<f64> = self.data.take_p_beta();
        let mut dq_alpha: Array1<f64> = self.data.take_dq_alpha();
        let mut dq_beta: Array1<f64> = self.data.take_dq_beta();
        let mut q_alpha: Array1<f64>;
        let mut q_beta: Array1<f64>;

        // molecular properties, we take all properties that are needed from the Properties type
        let s: ArrayView2<f64> = self.data.s();
        let h0: ArrayView2<f64> = self.data.h0();
        let gamma: ArrayView2<f64> = self.data.gamma();
        let p0: Array2<f64> = 0.5 * &self.data.p_ref();

        // the orbital energies and coefficients can be safely reset, since the
        // Hamiltonian does not depends on the charge differences and not on the orbital coefficients
        let mut orbs_alpha: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);
        let mut orbs_beta: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);
        let mut orbe_alpha: Array1<f64> = Array1::zeros([self.n_orbs]);
        let mut orbe_beta: Array1<f64> = Array1::zeros([self.n_orbs]);
        // orbital occupation numbers
        let mut f_alpha: Vec<f64> = vec![0.0; self.n_orbs];
        let mut f_beta: Vec<f64> = vec![0.0; self.n_orbs];

        // variables that are updated during the iterations
        let mut last_energy: f64 = 0.0;
        let mut total_energy: Result<f64, SCCError> = Ok(0.0);
        let mut delta_dq_max_alpha: f64 = 0.0;
        let mut delta_dq_max_beta: f64 = 0.0;
        let mut scf_energy: f64 = 0.0;
        let mut converged: bool = false;
        // add nuclear energy to the total scf energy
        let rep_energy: f64 = get_repulsive_energy(&self.atoms, self.n_atoms, &self.vrep);

        // initialize the charge mixer
        let mut broyden_mixer_alpha: BroydenMixer = BroydenMixer::new(self.n_atoms);
        let mut broyden_mixer_beta: BroydenMixer = BroydenMixer::new(self.n_atoms);

        if log_enabled!(Level::Info) {
            print_scc_init(max_iter, temperature, rep_energy);
        }
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();

        // create array of spin couplings
        let spin_couplings:Array1<f64>= self.atoms.iter().map(|atom| {atom.spin_coupling}).collect();

        'scf_loop: for i in 0..max_iter {
            let h_coul: Array2<f64> = construct_h1(self.n_orbs, &self.atoms, gamma.view(), (&dq_alpha+&dq_beta).view()) * s.view();
            let h_exchange:Array2<f64> = construct_h_magnetization(self.n_orbs, &self.atoms,(&dq_alpha-&dq_beta).view(),spin_couplings.view()) * s.view();
            let mut h_alpha: Array2<f64> = h_coul.clone() + h_exchange.view() + h0.view();
            let mut h_beta: Array2<f64> = h_coul.clone() - h_exchange.view() + h0.view();

            // H' = X^t.H.X
            h_alpha = x.t().dot(&h_alpha).dot(&x);
            h_beta = x.t().dot(&h_beta).dot(&x);
            let tmp_alpha: (Array1<f64>, Array2<f64>) = h_alpha.eigh(UPLO::Upper).unwrap();
            let tmp_beta: (Array1<f64>, Array2<f64>) = h_beta.eigh(UPLO::Upper).unwrap();
            orbe_alpha = tmp_alpha.0;
            orbe_beta = tmp_beta.0;
            // C = X.C'
            orbs_alpha = x.dot(&tmp_alpha.1);
            orbs_beta = x.dot(&tmp_beta.1);

            // compute the fermi orbital occupation
            let tmp_alpha: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
                orbe_alpha.view(),
                0,
                self.alpha_elec,
                temperature,
            );
            f_alpha = tmp_alpha.1;
            let tmp_beta: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
                orbe_beta.view(),
                0,
                self.beta_elec,
                temperature,
            );
            f_beta = tmp_beta.1;

            // calculate the density matrix
            p_alpha = density_matrix(orbs_alpha.view(), &f_alpha[..]);
            p_beta = density_matrix(orbs_beta.view(), &f_beta[..]);

            // update partial charges using Mulliken analysis
            let new_dq_alpha: Array1<f64> = mulliken(
                p_alpha.view(),
                s.view(),
                &self.atoms,
            );
            let new_dq_beta:  Array1<f64> = mulliken(
                p_beta.view(),
                s.view(),
                &self.atoms,
            );

            // charge difference to previous iteration
            let delta_dq_alpha: Array1<f64> = &new_dq_alpha - &dq_alpha;
            let delta_dq_beta: Array1<f64> = &new_dq_beta - &dq_beta;

            delta_dq_max_alpha = *delta_dq_alpha.map(|x| x.abs()).max().unwrap();
            delta_dq_max_beta = *delta_dq_beta.map(|x| x.abs()).max().unwrap();

            if log_enabled!(Level::Trace) {
                print_orbital_information(orbe_alpha.view(), &f_alpha);
                print_orbital_information(orbe_beta.view(), &f_beta);
            }

            // check if charge difference to the previous iteration is lower than 1e-5
            if (delta_dq_max_alpha < scf_charge_conv) && (delta_dq_max_beta < scf_charge_conv)
                && (last_energy - scf_energy).abs() < scf_energy_conv
            {
                converged = true;
            }

            // Broyden mixing of partial charges # changed new_dq to dq
            dq_alpha = broyden_mixer_alpha.next(dq_alpha, delta_dq_alpha);
            dq_beta = broyden_mixer_beta.next(dq_beta, delta_dq_beta);

            // if log_enabled!(Level::Debug) {
            //     print_charges(q_alpha.view(), dq_alpha.view());
            //     print_charges(q_beta.view(), dq_beta.view());
            // }

            // compute electronic energy
            scf_energy = get_electronic_energy_unrestricted(
                p_alpha.view(),
                p_beta.view(),
                h0.view(),
                dq_alpha.view(),
                dq_beta.view(),
                gamma.view(),
                spin_couplings.view(),
            );

            if log_enabled!(Level::Info) {
                print_energies_at_iteration_unrestricted(i, scf_energy, rep_energy, last_energy, delta_dq_max_alpha, delta_dq_max_beta)
            }

            if converged {
                total_energy = Ok(scf_energy + rep_energy);
                break 'scf_loop;
            }
            total_energy = Err(SCCError::new(i,last_energy - scf_energy,delta_dq_max_alpha,delta_dq_max_beta));
            // save the scf energy from the current iteration
            last_energy = scf_energy;
        }
        if log_enabled!(Level::Info) {
            print_unrestricted_scc_end(timer, self.config.jobtype.as_str(), scf_energy, rep_energy, orbe_alpha.view(), &f_alpha,orbe_beta.view(),&f_beta);
        }
        return total_energy;
    }
}
