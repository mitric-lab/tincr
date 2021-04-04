use tincr::constants::*;
use tincr::defaults;
use tincr::h0_and_s::h0_and_s;
use tincr::io::*;
use approx::AbsDiffEq;
use itertools::Itertools;
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use std::cmp::max;
use std::iter::FromIterator;
use std::ops::Deref;
use std::time::Instant;
use crate::initialization::Molecule;
use crate::scc::broyden::BroydenMixer;
use crate::h0_and_s::h0_and_s;
use crate::scc::{fermi_occupation, get_repulsive_energy};
use crate::scc::mulliken::mulliken;
use crate::scc::level_shifting::LevelShifter;
use crate::scc::helpers::density_matrix_ref;
use crate::scc::mixer::*;
use crate::io::SccConfig;
use crate::utils::Timer;
use crate::initialization::properties::ElectronicData;

/// Trait that optimizes the Kohn-Sham orbitals iteratively by employing the
/// self-consistent charge scheme to find the ground state energy
pub trait SCCRoutine {
    fn new(n_atoms: usize) -> Self;
    fn run(&mut self) -> Result<f64, E, >;
}

/// Spin-restricted (spin-unpolarized) SCC Routine.
/// Only one set of partial charges/charge differences is used.
pub struct RestrictedSCC<'a, T: Mixer> {
    config: &'a SccConfig,
    data: &'a ElectronicData,
    mixer: T,

}

impl SCCRoutine for RestrictedSCC<T> {
    fn new(n_atoms: usize) -> Self {
        todo!()
    }

    /// Do the SCC Iteration until the charge differences and energy converges
    fn run(&mut self) -> Result<f64, _> {
        let timer: Timer = Timer::start();



        info!({}, timer);
    }
}

// This routine is very messy und should be rewritten in a clean form
impl Molecule {

    pub fn run_scc(&mut self) -> (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) {
        let scc_timer: Instant = Instant::now();

        let temperature: f64 = molecule.config.scf.electronic_temperature;
        let max_iter: usize = molecule.config.scf.scf_max_cycles;
        let scf_charge_conv: f64 = molecule.config.scf.scf_charge_conv;
        let scf_energy_conv: f64 = molecule.config.scf.scf_energy_conv;

        let mut level_shift_flag: bool = false;
        let mut level_shifter: LevelShifter = LevelShifter::empty();
        // construct reference density matrix
        let p0: Array2<f64> = density_matrix_ref(&molecule.atomic_numbers.unwrap(), );
        let mut p: Array2<f64> = p0.clone();
        // charge guess
        let mut dq: Array1<f64> = molecule.final_charges.clone();
        let mut q: Array1<f64> = Array::from_iter(molecule.calculator.q0.iter().cloned());
        let mut energy_old: f64 = 0.0;
        let mut scf_energy: f64 = 0.0;
        let mut charge_diff: f64 = 0.0;
        let mut orbs: Array2<f64> =
            Array2::zeros([molecule.calculator.n_orbs, molecule.calculator.n_orbs]);
        let mut orbe: Array1<f64> = Array1::zeros([molecule.calculator.n_orbs]);
        let mut f: Vec<f64> = Array1::zeros([molecule.calculator.n_orbs]).to_vec();
        let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(
            &molecule.atomic_numbers,
            molecule.positions.view(),
            molecule.calculator.n_orbs,
            &molecule.calculator.valorbs,
            molecule.proximity_matrix.view(),
            &molecule.calculator.skt,
            &molecule.calculator.orbital_energies,
        );

        let mut broyden_mixer: BroydenMixer = BroydenMixer::new(molecule.n_atoms);

        let mut converged: bool = false;

        //  compute A = S^(-1/2)
        // 1. diagonalize S
        //let (w, v): (Array1<f64>, Array2<f64>) = s.eigh(UPLO::Upper).unwrap();
        // 2. compute inverse square root of the eigenvalues
        //let w12: Array2<f64> = Array2::from_diag(&w.map(|x| x.pow(-0.5)));
        // 3. and transform back
        //let a: Array2<f64> = v.dot(&w12.dot(&v.t()));

        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();

        // add nuclear energy to the total scf energy
        let rep_energy: f64 = get_repulsive_energy(&molecule);

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

        'scf_loop: for i in 0..max_iter {
            let h1: Array2<f64> = construct_h1(&molecule, molecule.g0.view(), dq.view());
            let h_coul: Array2<f64> = h1 * s.view();
            let mut h: Array2<f64> = h_coul + h0.view();

            //let mut prev_h_X:Array2<f64>
            if molecule.calculator.r_lr.is_none() || molecule.calculator.r_lr.unwrap() > 0.0 {
                let h_x: Array2<f64> = lc_exact_exchange(&s, &molecule.g0_lr_ao, &p0, &p, h.dim().0);
                h = h + h_x;
            }

            if level_shift_flag {
                if level_shifter.is_empty() {
                    level_shifter = LevelShifter::new(
                        molecule.calculator.n_orbs,
                        get_frontier_orbitals(molecule.calculator.n_elec).1,
                    );
                } else {
                    if charge_diff < (1.0e5 * scf_charge_conv) {
                        level_shifter.reduce_weight();
                    }
                    if charge_diff < (1.0e3 * scf_charge_conv) {
                        level_shift_flag == false;
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
            orbs = x.dot(&tmp.1);

            // construct density matrix
            let tmp: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
                orbe.view(),
                molecule.calculator.q0.iter().sum::<f64>() as usize - molecule.charge as usize,
                molecule.calculator.nr_unpaired_electrons,
                temperature,
            );
            let mu: f64 = tmp.0;
            f = tmp.1;

            if level_shift_flag == false {
                level_shift_flag = enable_level_shifting(orbe.view(), molecule.calculator.n_elec);
            }

            // calculate the density matrix
            p = density_matrix(orbs.view(), &f[..]);
            //println!("P {}", p);
            //println!("F {:?}", f);
            // update partial charges using Mulliken analysis
            let (new_q, new_dq): (Array1<f64>, Array1<f64>) = mulliken(
                p.view(),
                p0.view(),
                s.view(),
                &molecule.calculator.orbs_per_atom,
                molecule.n_atoms,
            );
            // charge difference to previous iteration
            let dq_diff: Array1<f64> = &new_dq - &dq;

            charge_diff = dq_diff.map(|x| x.abs()).max().unwrap().to_owned();

            if log_enabled!(Level::Trace) {
                print_orbital_information(orbe.view(), &f);
            }
            // check if charge difference to the previous iteration is lower then 1e-5
            if (&charge_diff < &scf_charge_conv) && &(energy_old - scf_energy).abs() < &scf_energy_conv
            {
                converged = true;
            }
            // Broyden mixing of partial charges # changed new_dq to dq
            dq = broyden_mixer.next(dq, dq_diff);
            q = new_q;
            debug!("");
            debug!("{: <35} ", "atomic charges and partial charges");
            debug!("{:-^35}", "");
            if log_enabled!(Level::Debug) {
                for (idx, (qi, dqi)) in q.iter().zip(dq.iter()).enumerate() {
                    debug!("Atom {: >4} q: {:>18.14} dq: {:>18.14}", idx + 1, qi, dqi);
                }
            }
            debug!("{:-^55}", "");
            // compute electronic energy
            scf_energy = get_electronic_energy(
                &molecule,
                p.view(),
                &p0,
                &s,
                h0.view(),
                dq.view(),
                (&molecule.g0).deref().view(),
                &molecule.g0_lr_ao,
            );
            if i == 0 {
                info!(
                    "{: >5} {:>18.10e} {:>18.13} {:>18.10e} {:>12.4}",
                    i + 1,
                    scf_energy + rep_energy,
                    0.0,
                    charge_diff,
                    level_shifter.weight
                );
            } else {
                info!(
                    "{: >5} {:>18.10e} {:>18.10e} {:>18.10e} {:>12.4}",
                    i + 1,
                    scf_energy + rep_energy,
                    energy_old - scf_energy,
                    charge_diff,
                    level_shifter.weight
                );
            }
            energy_old = scf_energy;

            assert_ne!(i + 1, max_iter, "SCF not converged");

            if converged {
                break 'scf_loop;
                molecule.set_final_charges(dq);
            }
        }
        info!("{:-^75} ", "");
        info!("{: ^75}", "SCC converged");
        info!("{:^80} ", "");
        info!("final energy: {:18.14} Hartree", scf_energy + rep_energy);
        info!("{:-<80} ", "");
        info!(
            "{:>68} {:>8.2} s",
            "elapsed time:",
            scc_timer.elapsed().as_secs_f32()
        );
        drop(scc_timer);
        if molecule.config.jobtype == "sp" {
            print_orbital_information(orbe.view(), &f);
        }
        return (scf_energy + rep_energy, orbs, orbe, s, f);
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