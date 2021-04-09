use crate::initialization::molecule::System;
use crate::initialization::properties::ElectronicData;
use crate::io::SccConfig;
use crate::scc::mixer::BroydenMixer;
use crate::scc::gamma_approximation::{gamma_atomwise, gamma_ao_wise};
use crate::scc::h0_and_s::h0_and_s;
use crate::scc::helpers::density_matrix_ref;
use crate::scc::level_shifting::LevelShifter;
use crate::scc::mixer::*;
use crate::scc::mulliken::mulliken;
use crate::scc::{fermi_occupation, get_repulsive_energy, construct_h1, density_matrix, enable_level_shifting, get_electronic_energy, lc_exact_exchange, get_frontier_orbitals};
use crate::utils::Timer;
use approx::AbsDiffEq;
use itertools::Itertools;
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use peroxide::fuga::gamma;
use std::cmp::max;
use std::fmt;
use std::iter::FromIterator;
use std::ops::Deref;
use std::time::Instant;


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
            h0_and_s(self.n_orbs, &self.atoms, &self.geometry, &self.slako);
        // and save it in the molecule properties
        self.properties.set_h0(h0);
        self.properties.set_s(s);
        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = self.atoms.iter().map(|atom| atom.number).collect();
        self.properties.set_atomic_numbers(atomic_numbers);
        // get the gamma matrix
        let gamma: Array2<f64> = gamma_atomwise(
            &self.gammafunction,
            self.properties.atomic_numbers().unwrap(),
            self.n_atoms,
            self.geometry.distances.as_ref().unwrap().view(),
        );
        // and save it as a `Property`
        self.properties.set_gamma(gamma);

        // if the system contains a long-range corrected Gammafunction the gamma matrix will be computed
        if self.gammafunction_lc.is_some() {
            let (gamma_lr, gamma_lr_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                self.properties.atomic_numbers().unwrap(),
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
                self.geometry.distances.as_ref().unwrap().view(),
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
        let mut q: Array1<f64> = Array1::from_shape_vec((self.n_atoms), self.atoms.iter().map(|atom| atom.n_elec as f64).collect()).unwrap();

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
        let rep_energy: f64 = get_repulsive_energy(&self.atoms, self.geometry.coordinates.view(), self.n_atoms, &self.vrep);

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
                    lc_exact_exchange(s.view(), self.properties.gamma_lr_ao().unwrap(), p0.view(), p.view(), h.dim().0);
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
                self.n_unpaired,
                temperature,
            );
            let mu: f64 = tmp.0;
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
        return total_energy;
    }
}

fn print_scc_init(max_iter: usize, temperature: f64, rep_energy: f64) {
    info!("{:^80}", "");
    info!("{: ^80}", "SCC-Routine");
    info!("{:-^80}", "");
    //info!("{: <25} {}", "convergence criterium:", scf_conv);
    info!("{: <25} {}", "max. iterations:", max_iter);
    info!("{: <25} {} K", "electronic temperature:", temperature);
    info!("{: <25} {:.14} Hartree", "repulsive energy:", rep_energy);
    info!("{:^80}", "");
    info!(
        "{: <45} ",
        "SCC Iterations: all quantities are in atomic units"
    );
    info!("{:-^75} ", "");
    info!(
        "{: <5} {: >18} {: >18} {: >18} {: >12}",
        "Iter.", "SCC Energy", "Energy diff.", "dq diff.", "Lvl. shift"
    );
    info!("{:-^75} ", "");
}

fn print_charges(q: ArrayView1<f64>, dq: ArrayView1<f64>) {
    debug!("");
    debug!("{: <35} ", "atomic charges and partial charges");
    debug!("{:-^35}", "");
    for (idx, (qi, dqi)) in q.iter().zip(dq.iter()).enumerate() {
        debug!("Atom {: >4} q: {:>18.14} dq: {:>18.14}", idx + 1, qi, dqi);
    }
    debug!("{:-^55}", "");
}

fn print_energies_at_iteration(iter: usize, scf_energy: f64, rep_energy: f64, energy_old: f64, dq_diff_max:f64, ls_weight: f64) {
    if iter == 0 {
        info!(
            "{: >5} {:>18.10e} {:>18.13} {:>18.10e} {:>12.4}",
            iter + 1,
            scf_energy + rep_energy,
            0.0,
            dq_diff_max,
            ls_weight
        );
    } else {
        info!(
            "{: >5} {:>18.10e} {:>18.10e} {:>18.10e} {:>12.4}",
            iter + 1,
            scf_energy + rep_energy,
            energy_old - scf_energy,
            dq_diff_max,
            ls_weight
        );
    }
}

fn print_scc_end(timer: Timer, jobtype: &str, scf_energy: f64, rep_energy: f64, orbe: ArrayView1<f64>, f: &[f64]) {
    info!("{:-^75} ", "");
    info!("{: ^75}", "SCC converged");
    info!("{:^80} ", "");
    info!("final energy: {:18.14} Hartree", scf_energy + rep_energy);
    info!("{:-<80} ", "");
    info!("{}", timer);
    if jobtype == "sp" {
        print_orbital_information(orbe.view(), &f);
    }
}


fn print_orbital_information(orbe: ArrayView1<f64>, f: &[f64]) -> () {
    info!("{:^80} ", "");
    info!(
        "{:^8} {:^6} {:>18.14} | {:^8} {:^6} {:>18.14}",
        "Orb.", "Occ.", "Energy/Hartree", "Orb.", "Occ.", "Energy/Hartree"
    );
    info!("{:-^71} ", "");
    let n_orbs: usize = orbe.len();
    for i in (0..n_orbs).step_by(2) {
        if i + 1 < n_orbs {
            info!(
                "MO:{:>5} {:>6} {:>18.14} | MO:{:>5} {:>6} {:>18.14}",
                i + 1,
                f[i],
                orbe[i],
                i + 2,
                f[i + 1],
                orbe[i + 1]
            );
        } else {
            info!("MO:{:>5} {:>6} {:>18.14} |", i + 1, f[i], orbe[i]);
        }
    }
    info!("{:-^71} ", "");
}

#[test]
fn h1_construction() {
    let mol: Molecule = get_water_molecule();

    let (gm, gm_a0): (Array2<f64>, Array2<f64>) = get_gamma_matrix(
        &mol.atomic_numbers,
        mol.n_atoms,
        mol.calculator.n_orbs,
        mol.distance_matrix.view(),
        &mol.calculator.hubbard_u,
        &mol.calculator.valorbs,
        Some(0.0),
    );
    let dq: Array1<f64> = array![0.4900936727759634, -0.2450466365939161, -0.2450470361820512];
    let h1: Array2<f64> = construct_h1(&mol, gm.view(), dq.view());
    let h1_ref: Array2<f64> = array![
        [
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0138472664342115,
            0.0138473229910027
        ],
        [
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0138472664342115,
            0.0138473229910027
        ],
        [
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0138472664342115,
            0.0138473229910027
        ],
        [
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0138472664342115,
            0.0138473229910027
        ],
        [
            0.0138472664342115,
            0.0138472664342115,
            0.0138472664342115,
            0.0138472664342115,
            -0.0019095797643945,
            -0.0019095232076034
        ],
        [
            0.0138473229910027,
            0.0138473229910027,
            0.0138473229910027,
            0.0138473229910027,
            -0.0019095232076034,
            -0.0019094666508122
        ]
    ];
    assert!(h1.abs_diff_eq(&h1_ref, 1e-06));
}

#[test]
fn density_matrix_test() {
    let orbs: Array2<f64> = array![
        [
            8.7633819729731810e-01,
            -7.3282344651120093e-07,
            -2.5626947507277165e-01,
            5.9002324562689818e-16,
            4.4638746598694098e-05,
            6.5169204671842440e-01
        ],
        [
            1.5609816246174135e-02,
            -1.9781345423283686e-01,
            -3.5949495734350723e-01,
            -8.4834397825097219e-01,
            2.8325036729124353e-01,
            -2.9051012618890415e-01
        ],
        [
            2.5012007142380756e-02,
            -3.1696154856040176e-01,
            -5.7602794926745904e-01,
            5.2944545948125210e-01,
            4.5385929584577422e-01,
            -4.6549179289356957e-01
        ],
        [
            2.0847639428373754e-02,
            5.2838141513310655e-01,
            -4.8012912372328725e-01,
            6.2706942441176916e-16,
            -7.5667689316910480e-01,
            -3.8791258902430070e-01
        ],
        [
            1.6641902124514810e-01,
            -3.7146607333368298e-01,
            2.5136104623642469e-01,
            -4.0589133922668506e-16,
            -7.2004448997899451e-01,
            -7.6949321948982752e-01
        ],
        [
            1.6641959153799185e-01,
            3.7146559134907692e-01,
            2.5135994443678955e-01,
            -1.0279311111336347e-15,
            7.1993588930877628e-01,
            -7.6959837655952745e-01
        ]
    ];
    let f: Vec<f64> = vec![2., 2., 2., 2., 0., 0.];
    let p: Array2<f64> = density_matrix(orbs.view(), &f[..]);
    let p_ref: Array2<f64> = array![
        [
            1.6672853597938484,
            0.2116144144027609,
            0.3390751794256264,
            0.282623268095985,
            0.162846907840508,
            0.1628473832190555
        ],
        [
            0.2116144144027609,
            1.776595917657724,
            -0.357966065395007,
            0.1368169475860144,
            -0.0285685423132669,
            -0.3224914900258041
        ],
        [
            0.3390751794256264,
            -0.357966065395007,
            1.426421833339374,
            0.2192252885141318,
            -0.0457761047995666,
            -0.5167363487612707
        ],
        [
            0.282623268095985,
            0.1368169475860144,
            0.2192252885141318,
            1.020291038750183,
            -0.6269841692414225,
            0.1581194812138258
        ],
        [
            0.162846907840508,
            -0.0285685423132669,
            -0.0457761047995666,
            -0.6269841692414225,
            0.4577294196704165,
            -0.0942187608833704
        ],
        [
            0.1628473832190555,
            -0.3224914900258041,
            -0.5167363487612707,
            0.1581194812138258,
            -0.0942187608833704,
            0.4577279753425148
        ]
    ];
    assert!(p.abs_diff_eq(&p_ref, 1e-15));
}

#[test]
fn reference_density_matrix() {
    let mol: Molecule = get_water_molecule();
    let p0: Array2<f64> = density_matrix_ref(&mol);
    let p0_ref: Array2<f64> = array![
        [2., 0., 0., 0., 0., 0.],
        [0., 1.3333333333333333, 0., 0., 0., 0.],
        [0., 0., 1.3333333333333333, 0., 0., 0.],
        [0., 0., 0., 1.3333333333333333, 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]
    ];
    assert!(p0.abs_diff_eq(&p0_ref, 1e-16));
}

#[test]
fn self_consistent_charge_routine() {
    let mut mol: Molecule = get_water_molecule();
    let energy = run_scc(&mut mol);
    //println!("ENERGY: {}", energy);
    //TODO: CREATE AN APPROPIATE TEST FOR THE SCC ROUTINE
}

#[test]
fn self_consistent_charge_routine_near_coin() {
    let atomic_numbers: Vec<u8> = vec![1, 6, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1];
    let mut positions: Array2<f64> = array![
        [1.14035341, -0.13021522, 2.08719024],
        [0.50220664, 0.05063317, 1.22118011],
        [-0.88326674, 0.10942181, 1.29559480],
        [-1.44213805, 0.04044044, 2.22946088],
        [-1.48146499, 0.27160316, 0.02973104],
        [-2.56237600, 0.24057787, -0.12419723],
        [-0.55934487, 0.38982195, -1.09018600],
        [-0.82622551, 1.09380623, -1.89412324],
        [0.63247888, -0.46827911, -1.31954527],
        [1.20025191, -0.17363757, -2.22072997],
        [1.09969583, 0.23621820, -0.12214916],
        [2.06782038, 0.75551464, -0.16994068],
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        charge,
        multiplicity,
        Some(0.0),
        None,
        config,
        None,
    );
    let energy = run_scc(&mut mol);
    //println!("ENERGY: {}", energy);
    //TODO: CREATE AN APPROPIATE TEST FOR THE SCC ROUTINE
}

#[test]
fn test_scc_routine_benzene() {
    let mut mol: Molecule = get_benzene_molecule();
    let energy = run_scc(&mut mol);
}
