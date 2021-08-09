use crate::initialization::system::System;
use crate::scc::mixer::BroydenMixer;
use crate::scc::gamma_approximation::{gamma_atomwise, gamma_ao_wise};
use crate::scc::h0_and_s::h0_and_s;
use crate::scc::helpers::density_matrix_ref;
use crate::scc::level_shifting::LevelShifter;
use crate::scc::mixer::*;
use crate::scc::mulliken::mulliken;
use crate::scc::logging::*;
use crate::scc::{fermi_occupation, get_repulsive_energy, construct_h1, density_matrix, enable_level_shifting, get_electronic_energy, lc_exact_exchange, get_frontier_orbitals};
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
    charge_diff: f64,
}

impl SCCError{
    pub fn new(iter: usize, energy_diff: f64, charge_diff:f64) -> Self {
        let message: String = format! {"SCC-Routine failed in Iteration: {}. The charge\
                                        difference at the last iteration was {} and the energy\
                                        difference was {}",
                                       iter,
                                       charge_diff,
                                       charge_diff};
        Self {
            message,
            iteration: iter,
            energy_diff: energy_diff,
            charge_diff: charge_diff,
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
/// spin-restricted (spin-unpolarized) self-consistent charge scheme to find the ground state energy.
/// Only one set of charges/charge differences is used
pub trait RestrictedSCC {
    fn prepare_scc(&mut self);
    fn run_scc(&mut self) -> Result<f64, SCCError>;
}

impl<'a> RestrictedSCC for System {
    ///  To run the SCC calculation the following properties in the molecule need to be set:
    /// - H0
    /// - S: overlap matrix in AO basis
    /// - Gamma matrix (and long-range corrected Gamma matrix if we use LRC)
    /// - If there are no charge differences, `dq`, from a previous calculation
    ///  they are initialized to zeros
    /// - the density matrix and reference density matrix
    fn prepare_scc(&mut self) {
        // get H0 and S
        let (s, h0): (Array2<f64>, Array2<f64>) =
            h0_and_s(self.n_orbs, &self.atoms,  &self.slako);
        // and save it in the molecule properties
        self.properties.set_h0(h0);
        self.properties.set_s(s);
        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = self.atoms.iter().map(|atom| atom.number).collect();
        self.properties.set_atomic_numbers(atomic_numbers);
        // get the gamma matrix
        let gamma: Array2<f64> = gamma_atomwise(
            &self.gammafunction,
            &self.atoms,
            self.n_atoms,
        );
        // and save it as a `Property`
        self.properties.set_gamma(gamma);

        // if the system contains a long-range corrected Gammafunction the gamma matrix will be computed
        if self.gammafunction_lc.is_some() {
            let (gamma_lr, gamma_lr_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.properties.set_gamma_lr(gamma_lr);
            self.properties.set_gamma_lr_ao(gamma_lr_ao);
        }

        // if this is the first SCC calculation the charge differences will be initialized to zeros
        if !self.properties.contains_key("dq") {
            self.properties.set_dq(Array1::zeros(self.n_atoms));
        }

        // this is also only needed in the first SCC calculation
        if !self.properties.contains_key("ref_density_matrix") {
            self.properties.set_p_ref(density_matrix_ref(self.n_orbs, &self.atoms));
        }

        // in the first SCC calculation the density matrix is set to the reference density matrix
        if !self.properties.contains_key("P") {
            self.properties.set_p(self.properties.p_ref().unwrap().to_owned());
        }
    }

    // SCC Routine for a single molecule and for spin-unpolarized systems
    fn run_scc(&mut self) -> Result<f64, SCCError> {
        let timer: Timer = Timer::start();

        // SCC settings from the user input
        let temperature: f64 = self.config.scf.electronic_temperature;
        let max_iter: usize = self.config.scf.scf_max_cycles;
        let scf_charge_conv: f64 = self.config.scf.scf_charge_conv;
        let scf_energy_conv: f64 = self.config.scf.scf_energy_conv;

        // the properties that are changed during the SCC routine are taken
        // and will be inserted at the end of the SCC routine
        let mut p: Array2<f64> = self.properties.take_p().unwrap();
        let mut dq: Array1<f64> = self.properties.take_dq().unwrap();
        let mut q: Array1<f64>;

        // molecular properties, we take all properties that are needed from the Properties type
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();

        // the orbital energies and coefficients can be safely reset, since the
        // Hamiltonian does not depends on the charge differences and not on the orbital coefficients
        let mut orbs: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);
        let mut orbe: Array1<f64> = Array1::zeros([self.n_orbs]);
        // orbital occupation numbers
        let mut f: Vec<f64> = vec![0.0; self.n_orbs];

        // variables that are updated during the iterations
        let mut last_energy: f64 = 0.0;
        let mut total_energy: Result<f64, SCCError> = Ok(0.0);
        let mut delta_dq_max: f64 = 0.0;
        let mut scf_energy: f64 = 0.0;
        let mut converged: bool = false;
        // add nuclear energy to the total scf energy
        let rep_energy: f64 = get_repulsive_energy(&self.atoms, self.n_atoms, &self.vrep);

        // initialize the charge mixer
        let mut broyden_mixer: BroydenMixer = BroydenMixer::new(self.n_atoms);
        // initialize the orbital level shifter
        let mut level_shifter: LevelShifter = LevelShifter::default();

        if log_enabled!(Level::Info) {
            print_scc_init(max_iter, temperature, rep_energy);
        }
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();

        'scf_loop: for i in 0..max_iter {
            let h_coul: Array2<f64> = construct_h1(self.n_orbs, &self.atoms, gamma.view(), dq.view()) * s.view();
            let mut h: Array2<f64> = h_coul + h0.view();

            if self.gammafunction_lc.is_some() {
                let h_x: Array2<f64> =
                    lc_exact_exchange(s.view(), self.properties.gamma_lr_ao().unwrap(), p0.view(), p.view());
                h = h + h_x;
            }

            if level_shifter.is_on {
                if level_shifter.is_empty() {
                    level_shifter = LevelShifter::new(
                        self.n_orbs,
                        get_frontier_orbitals(self.n_elec).1,
                    );
                } else {
                    if delta_dq_max < (1.0e5 * scf_charge_conv) {
                        level_shifter.reduce_weight();
                    }
                    if delta_dq_max < (1.0e3 * scf_charge_conv) {
                        level_shifter.turn_off();
                    }
                }
                let shift: Array2<f64> = level_shifter.shift(orbs.view());
                h = h + shift;
            }

            // H' = X^t.H.X
            h = x.t().dot(&h).dot(&x);
            let tmp: (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Upper).unwrap();
            orbe = tmp.0;
            // C = X.C'
            orbs = x.dot(&tmp.1);

            // compute the fermi orbital occupation
            let tmp: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
                orbe.view(),
                self.n_elec,
                self.n_unpaired as f64,
                temperature,
            );
            f = tmp.1;

            if !level_shifter.is_on {
                level_shifter.is_on = enable_level_shifting(orbe.view(), self.n_elec);
            }

            // calculate the density matrix
            p = density_matrix(orbs.view(), &f[..]);

            // update partial charges using Mulliken analysis
            let (new_q, new_dq): (Array1<f64>, Array1<f64>) = mulliken(
                p.view(),
                p0.view(),
                s.view(),
                &self.atoms,
                self.n_atoms,
            );

            // charge difference to previous iteration
            let delta_dq: Array1<f64> = &new_dq - &dq;

            delta_dq_max = *delta_dq.map(|x| x.abs()).max().unwrap();

            if log_enabled!(Level::Trace) {
                print_orbital_information(orbe.view(), &f);
            }

            // check if charge difference to the previous iteration is lower than 1e-5
            if (delta_dq_max < scf_charge_conv)
                && (last_energy - scf_energy).abs() < scf_energy_conv
            {
                converged = true;
            }

            // Broyden mixing of partial charges # changed new_dq to dq
            dq = broyden_mixer.next(dq, delta_dq);
            q = new_q;

            if log_enabled!(Level::Debug) {
                print_charges(q.view(), dq.view());
            }

            // compute electronic energy
            scf_energy = get_electronic_energy(
                p.view(),
                p0.view(),
                s.view(),
                h0.view(),
                dq.view(),
                gamma.view(),
                self.properties.gamma_lr_ao(),
            );

            if log_enabled!(Level::Info) {
                print_energies_at_iteration(i, scf_energy, rep_energy, last_energy, delta_dq_max, level_shifter.weight)
            }

            if converged {
                total_energy = Ok(scf_energy + rep_energy);
                break 'scf_loop;
            }
            total_energy = Err(SCCError::new(i,last_energy - scf_energy,delta_dq_max));
            // save the scf energy from the current iteration
            last_energy = scf_energy;
        }
        if log_enabled!(Level::Info) {
            print_scc_end(timer, self.config.jobtype.as_str(), scf_energy, rep_energy, orbe.view(), &f);
        }
        self.properties.set_orbs(orbs);
        self.properties.set_orbe(orbe);
        self.properties.set_p(p);
        self.properties.set_dq(dq);
        return total_energy;
    }
}