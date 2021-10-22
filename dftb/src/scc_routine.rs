use core::::system::System;
use core::::mixer::BroydenMixer;
use core::::gamma_approximation::{gamma_atomwise, gamma_ao_wise};
use core::::helpers::density_matrix_ref;
use core::::level_shifting::LevelShifter;
use core::::mixer::*;
use core::::mulliken::mulliken;
use core::::logging::*;
use core::::{fermi_occupation, get_repulsive_energy, construct_h1, density_matrix, enable_level_shifting, get_electronic_energy, lc_exact_exchange, get_frontier_orbitals};
use tincr_core::utils::Timer;
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_stats::{QuantileExt, DeviationExt};
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

impl<'a> RestrictedSCC for System<'a> {
    ///  To run the SCC calculation the following properties in the molecule need to be set:
    /// - H0
    /// - S: overlap matrix in AO basis
    /// - Gamma matrix (and long-range corrected Gamma matrix if we use LRC)
    /// - If there are no charge differences, `dq`, from a previous calculation
    ///  they are initialized to zeros
    /// - the density matrix and reference density matrix
    fn prepare_scc(&mut self) {
        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = self.atoms.number.iter().cloned().collect();
        self.data.set_atomic_numbers(atomic_numbers);

        // if this is the first SCC calculation the charge differences will be initialized to zeros
        self.data.set_if_unset_dq(Array1::zeros(self.atoms.len()));

        // this is also only needed in the first SCC calculation
        self.data.set_if_unset_p_ref(density_matrix_ref(self.n_orbs(), &self.atoms));

        // in the first SCC calculation the density matrix is set to the reference density matrix
        self.data.set_if_unset_p(self.data.p_ref().to_owned());

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
        let mut p: Array2<f64> = self.data.take_p();
        let mut dq: Array1<f64> = self.data.take_dq();
        let mut q: Array1<f64>;
        let mut q_ao: Array1<f64> = Array1::zeros([self.n_orbs()]);

        // molecular properties, we take all properties that are needed from the Properties type
        let s: ArrayView2<f64> = self.data.s();
        let h0: ArrayView2<f64> = self.data.h0();
        let gamma: ArrayView2<f64> = self.data.gamma();
        let p0: ArrayView2<f64> = self.data.p_ref();
        let mut dp: Array2<f64> = &p - &p0;

        // the orbital energies and coefficients can be safely reset, since the
        // Hamiltonian does not depends on the charge differences and not on the orbital coefficients
        let mut orbs: Array2<f64> = Array2::zeros([self.n_orbs(), self.n_orbs()]);
        let mut orbe: Array1<f64> = Array1::zeros([self.n_orbs()]);
        // orbital occupation numbers
        let mut f: Vec<f64> = vec![0.0; self.n_orbs()];

        // variables that are updated during the iterations
        let mut last_energy: f64 = 0.0;
        let mut total_energy: Result<f64, SCCError> = Ok(0.0);
        let mut scf_energy: f64 = 0.0;
        let mut diff_dq_max: f64 = 0.0;
        let mut converged: bool = false;
        // add nuclear energy to the total scf energy
        let rep_energy: f64 = get_repulsive_energy(&self.atoms, self.atoms.len(), &self.vrep);

        // initialize the charge mixer
        let mut broyden_mixer: BroydenMixer = BroydenMixer::new(self.n_orbs());
        // initialize the orbital level shifter
        let mut level_shifter: LevelShifter = LevelShifter::default();

        if log_enabled!(Level::Info) {
            print_scc_init(max_iter, temperature, rep_energy);
        }
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();

        'scf_loop: for i in 0..max_iter {
            let h_coul: Array2<f64> = construct_h1(self.n_orbs(), &self.atoms, gamma.view(), dq.view()) * s.view();
            let mut h: Array2<f64> = h_coul + h0.view();

            if self.gammafunction_lc.is_some() {
                let h_x: Array2<f64> =
                    lc_exact_exchange(s.view(), self.data.gamma_lr_ao(), dp.view());
                h = h + h_x;
            }
            let mut h_save:Array2<f64> = h.clone();

            if level_shifter.is_on {
                if level_shifter.is_empty() {
                    level_shifter = LevelShifter::new(
                        self.n_orbs(),
                        get_frontier_orbitals(self.n_elec).1,
                    );
                } else {
                    if diff_dq_max < (1.0e5 * scf_charge_conv) {
                        level_shifter.reduce_weight();
                    }
                    if diff_dq_max < (1.0e3 * scf_charge_conv) {
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

            // New Mulliken charges for each atomic orbital.
            let q_ao_n: Array1<f64> = s.dot(&p).diag().to_owned();

            // Charge difference to previous iteration
            let delta_dq: Array1<f64> = &q_ao_n - &q_ao;

            diff_dq_max = q_ao.root_mean_sq_err(&q_ao_n).unwrap();

            if log_enabled!(Level::Trace) {
                print_orbital_information(orbe.view(), &f);
            }

            // check if charge difference to the previous iteration is lower than 1e-5
            if (diff_dq_max < scf_charge_conv)
                && (last_energy - scf_energy).abs() < scf_energy_conv
            {
                converged = true;
            }

            // Broyden mixing of Mulliken charges per orbital.
            q_ao = broyden_mixer.next(q_ao, delta_dq);

            // The density matrix is updated in accordance with the Mulliken charges.
            p = p * &(&q_ao / &q_ao_n);
            dp = &p - &p0;
            dq = mulliken(dp.view(), s.view(), &self.atoms);

            if log_enabled!(Level::Debug) {
                q = mulliken(p.view(), s.view(), &self.atoms);
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
                Some(self.data.gamma_lr_ao()),
            );

            if log_enabled!(Level::Info) {
                print_energies_at_iteration(i, scf_energy, rep_energy, last_energy, diff_dq_max, level_shifter.weight)
            }

            if converged {
                total_energy = Ok(scf_energy + rep_energy);
                self.data.set_fock(h_save);
                break 'scf_loop;
            }
            total_energy = Err(SCCError::new(i,last_energy - scf_energy,diff_dq_max));
            // save the scf energy from the current iteration
            last_energy = scf_energy;
        }
        if log_enabled!(Level::Info) {
            print_scc_end(timer, self.config.jobtype.as_str(), scf_energy, rep_energy, orbe.view(), &f);
        }

        self.data.set_orbs(orbs);
        self.data.set_orbe(orbe);
        self.data.set_occupation(f);
        self.data.set_p(p);
        self.data.set_dq(dq);
        return total_energy;
    }
}