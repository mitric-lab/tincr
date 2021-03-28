use crate::broyden::*;
use tincr::calculator::*;
use tincr::constants::*;
use tincr::defaults;
use crate::diis::*;
use tincr::fermi_occupation;
use tincr::h0_and_s::h0_and_s;
use tincr::molecule::*;
use tincr::mulliken::*;
use approx::AbsDiffEq;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use std::cmp::max;
use std::iter::FromIterator;

// Routine for unrestricted SCC calculations
pub fn run_unrestricted_scc(
    molecule: &Molecule,
    max_iter: Option<usize>,
    scf_conv: Option<f64>,
    temperature: Option<f64>,
) -> f64 {
    let max_iter: usize = max_iter.unwrap_or(defaults::MAX_ITER);
    let scf_conv: f64 = 0.0;
    let temperature: f64 = temperature.unwrap_or(defaults::TEMPERATURE);

    // construct reference density matrix
    let p0: Array2<f64> = density_matrix_ref(&molecule);

    // initialize density matrix for alpha and beta spin orbitals
    let mut p_alpha: Array2<f64> = Array2::zeros(p0.raw_dim());
    let mut p_beta: Array2<f64> = Array2::zeros(p0.raw_dim());

    // charge guess for alpha and beta spin orbitals
    let mut dq_alpha: Array1<f64> = Array1::zeros([molecule.n_atoms]);
    let mut dq_beta: Array1<f64> = Array1::zeros([molecule.n_atoms]);
    let mut q_alpha: Array1<f64> = Array::from_iter(molecule.calculator.q0.iter().cloned());
    let mut q_beta: Array1<f64> = Array::from_iter(molecule.calculator.q0.iter().cloned());

    let mut energy_old: f64 = 0.0;
    let mut scf_energy: f64 = 0.0;
    let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(
        &molecule.atomic_numbers,
        molecule.positions.view(),
        molecule.calculator.n_orbs,
        &molecule.calculator.valorbs,
        molecule.proximity_matrix.view(),
        &molecule.calculator.skt,
        &molecule.calculator.orbital_energies,
    );
    let (gm, gm_a0): (Array2<f64>, Array2<f64>) = get_gamma_matrix(
        &molecule.atomic_numbers,
        molecule.n_atoms,
        molecule.calculator.n_orbs,
        molecule.distance_matrix.view(),
        &molecule.calculator.hubbard_u,
        &molecule.calculator.valorbs,
        Some(0.0),
    );

    let mut broyden_mixer_alpha: BroydenMixer = BroydenMixer::new(molecule.n_atoms);
    let mut broyden_mixer_beta: BroydenMixer = BroydenMixer::new(molecule.n_atoms);

    let mut converged: bool = false;

    //  compute A = S^(-1/2)
    // 1. diagonalize S
    let (w, v): (Array1<f64>, Array2<f64>) = s.eigh(UPLO::Upper).unwrap();
    // 2. compute inverse square root of the eigenvalues
    let w12: Array2<f64> = Array2::from_diag(&w.map(|x| x.pow(-0.5)));
    // 3. and transform back
    let a: Array2<f64> = v.dot(&w12.dot(&v.t()));

    // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
    // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
    let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();

    // add nuclear energy to the total scf energy
    let rep_energy: f64 = get_repulsive_energy(&molecule);
    'scf_loop: for i in 0..max_iter {
        // coulomb part
        let h1: Array2<f64> = construct_h1(&molecule, gm.view(), dq_alpha.view(), dq_beta.view());
        // exchange part
        let h2: Array2<f64> = construct_h2(&molecule, dq_alpha.view(), dq_beta.view());
        let h_coul: Array2<f64> = h1 * s.view();
        let h_exchange: Array2<f64> = h2 * s.view();
        let mut h_alpha: Array2<f64> = h_coul.clone() + h_exchange.view() + h0.view();
        let mut h_beta: Array2<f64> = h_coul - h_exchange.view() + h0.view();

        // H' = X^t.H.X
        let hp_alpha: Array2<f64> = x.t().dot(&h_alpha).dot(&x);
        let hp_beta: Array2<f64> = x.t().dot(&h_beta).dot(&x);

        let (orbe_alpha, cp_alpha): (Array1<f64>, Array2<f64>) =
            hp_alpha.eigh(UPLO::Upper).unwrap();
        let (orbe_beta, cp_beta): (Array1<f64>, Array2<f64>) = hp_beta.eigh(UPLO::Upper).unwrap();

        // C = X.C'
        let orbs_alpha: Array2<f64> = x.dot(&cp_alpha);
        let orbs_beta: Array2<f64> = x.dot(&cp_beta);

        // get chemical potential and occupation pattern for alpha electrons
        let (_, f_alpha): (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
            orbe_alpha.view(),
            molecule.calculator.q0.iter().sum::<f64>() as usize
                - molecule.charge as usize
                - molecule.calculator.nr_unpaired_electrons,
            molecule.calculator.nr_unpaired_electrons,
            temperature,
        );

        // get chemical potential and occupation pattern for beta electrons
        let (_, f_beta): (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
            orbe_beta.view(),
            molecule.calculator.q0.iter().sum::<f64>() as usize
                - molecule.charge as usize
                - molecule.calculator.nr_unpaired_electrons,
            molecule.calculator.nr_unpaired_electrons,
            temperature,
        );

        // calculate the density matrix
        p_alpha = density_matrix(orbs_alpha.view(), &f_alpha[..]);
        p_beta = density_matrix(orbs_beta.view(), &f_beta[..]);

        // update alpha partial charges using Mulliken analysis
        let (new_q_alpha, new_dq_alpha): (Array1<f64>, Array1<f64>) = mulliken(
            p_alpha.view(),
            p0.view(),
            s.view(),
            &molecule.calculator.orbs_per_atom,
            molecule.n_atoms,
        );

        // update beta partial charges using Mulliken analysis
        let (new_q_beta, new_dq_beta): (Array1<f64>, Array1<f64>) = mulliken(
            p_alpha.view(),
            p0.view(),
            s.view(),
            &molecule.calculator.orbs_per_atom,
            molecule.n_atoms,
        );

        // charge difference to previous iteration
        let dq_diff_alpha: Array1<f64> = &new_dq_alpha - &dq_alpha;
        let dq_diff_beta: Array1<f64> = &new_dq_beta - &dq_beta;

        // check if both charge differences to the previus iteration is lower then 1e-5
        if (dq_diff_alpha.map(|x| x.abs()).max().unwrap() < &scf_conv
            && dq_diff_beta.map(|x| x.abs()).max().unwrap() < &scf_conv)
        {
            converged = true;
        }
        // Broyden mixing of partial charges
        dq_alpha = broyden_mixer_alpha.next(new_dq_alpha, dq_diff_alpha);
        dq_beta = broyden_mixer_beta.next(new_dq_beta, dq_diff_beta);
        q_alpha = new_q_alpha;
        q_beta = new_q_beta;

        // compute electronic energy from alpha and beta electrons
        let scf_energy: f64 = get_electronic_energy(
            p_beta.view(),
            h0.view(),
            dq_alpha.view(),
            dq_beta.view(),
            molecule.calculator.spin_couplings.view(),
            gm.view(),
        );

        println!(
            "Iteration {} => SCF-Energy = {:.8} hartree",
            i,
            scf_energy + rep_energy
        );
        assert_ne!(i + 1, max_iter, "SCF not converged");

        if converged {
            break 'scf_loop;
        }
    }
    println!("SCF Converged!");
    return scf_energy + rep_energy;
}

/// Compute energy due to core electrons and nuclear repulsion
fn get_repulsive_energy(molecule: &Molecule) -> f64 {
    let mut e_nuc: f64 = 0.0;
    for (i, (z_i, posi)) in molecule.atomic_numbers[1..molecule.n_atoms]
        .iter()
        .zip(
            molecule
                .positions
                .slice(s![1..molecule.n_atoms, ..])
                .outer_iter(),
        )
        .enumerate()
    {
        for (z_j, posj) in molecule.atomic_numbers[0..i + 1]
            .iter()
            .zip(molecule.positions.slice(s![0..i + 1, ..]).outer_iter())
        {
            let z_1: u8;
            let z_2: u8;
            if z_i > z_j {
                z_1 = *z_j;
                z_2 = *z_i;
            } else {
                z_1 = *z_i;
                z_2 = *z_j;
            }
            let r: f64 = (&posi - &posj).norm();
            // nucleus-nucleus and core-electron repulsion
            e_nuc += &molecule.calculator.v_rep[&(z_1, z_2)].spline_eval(r);
        }
    }
    return e_nuc;
}

/// the repulsive potential, the dispersion correction and only depend on the nuclear
/// geometry and do not change during the SCF cycle
fn get_nuclear_energy() {}

/// Compute electronic energies
/// TODO: CHECK EXCHANGE PART!
fn get_electronic_energy(
    p: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq_alpha: ArrayView1<f64>,
    dq_beta: ArrayView1<f64>,
    spin_couplings: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
) -> f64 {
    let dq: Array1<f64> = &dq_alpha + &dq_beta;
    //println!("P {}", p);
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));
    // electronic energy as sum of band structure energy and Coulomb energy
    //println!("E BS {} E COUL {} dQ {}", e_band_structure, e_coulomb, dq);
    let m_squared: Array1<f64> = (&dq_alpha - &dq_beta).iter().map(|x| x * x).collect();
    let e_exchange: f64 = 0.5 * m_squared.dot(&spin_couplings);
    // exchange part
    let e_elec: f64 = e_band_structure + e_coulomb + e_exchange;
    // long-range Hartree-Fock exchange
    // if ....Iteration {} =>
    //println!("               E_bs = {:.7}  E_coulomb = {:.7}", e_band_structure, e_coulomb);
    return e_elec;
}

/// Construct the density matrix
/// P_mn = sum_a f_a C_ma* C_na
fn density_matrix(orbs: ArrayView2<f64>, f: &[f64]) -> Array2<f64> {
    let occ_indx: Vec<usize> = f.iter().positions(|&x| x > 0.0).collect();
    let occ_orbs: Array2<f64> = orbs.select(Axis(1), &occ_indx);
    let f_occ: Vec<f64> = f.iter().filter(|&&x| x > 0.0).cloned().collect();
    // THIS IS NOT AN EFFICIENT WAY TO BUILD THE LEFT HAND SIDE
    let mut f_occ_mat: Vec<f64> = Vec::new();
    for i in 0..occ_orbs.nrows() {
        for val in f_occ.iter() {
            f_occ_mat.push(*val);
        }
    }
    let f_occ_mat: Array2<f64> = Array2::from_shape_vec(occ_orbs.raw_dim(), f_occ_mat).unwrap();
    let p: Array2<f64> = (f_occ_mat * &occ_orbs).dot(&occ_orbs.t());
    return p;
}

/// Construct reference density matrix
/// all atoms should be neutral
/// takes n_orbs, atomic_numbers, valorbs, valorbs_occupation
fn density_matrix_ref(mol: &Molecule) -> Array2<f64> {
    let mut p0: Array2<f64> = Array2::zeros((mol.calculator.n_orbs, mol.calculator.n_orbs));
    // iterate over orbitals on center i
    let mut idx: usize = 0;
    for zi in mol.atomic_numbers.iter() {
        // how many electrons are put into the nl-shell
        for (iv, _) in mol.calculator.valorbs[zi].iter().enumerate() {
            p0[[idx, idx]] = mol.calculator.valorbs_occupation[zi][iv] as f64;
            idx += 1;
        }
    }
    return p0;
}

fn construct_h1(
    mol: &Molecule,
    gamma: ArrayView2<f64>,
    dq_alpha: ArrayView1<f64>,
    dq_beta: ArrayView1<f64>,
) -> Array2<f64> {
    let dq: Array1<f64> = &dq_alpha + &dq_beta;
    let e_stat_pot: Array1<f64> = gamma.dot(&dq);
    let mut h1: Array2<f64> = Array2::zeros([mol.calculator.n_orbs, mol.calculator.n_orbs]);

    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, z_i) in mol.atomic_numbers.iter().enumerate() {
        for _ in &mol.calculator.valorbs[z_i] {
            nu = 0;
            for (j, z_j) in mol.atomic_numbers.iter().enumerate() {
                for _ in &mol.calculator.valorbs[z_j] {
                    h1[[mu, nu]] = 0.5 * (e_stat_pot[i] + e_stat_pot[j]);
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h1;
}

fn construct_h2(
    mol: &Molecule,
    dq_alpha: ArrayView1<f64>,
    dq_beta: ArrayView1<f64>,
) -> Array2<f64> {
    let mut h2: Array2<f64> = Array2::zeros([mol.calculator.n_orbs, mol.calculator.n_orbs]);
    let m: Array1<f64> = &dq_alpha - &dq_beta;
    let tmp: Array1<f64> = &mol.calculator.spin_couplings * &m;
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, z_i) in mol.atomic_numbers.iter().enumerate() {
        for _ in &mol.calculator.valorbs[z_i] {
            nu = 0;
            for (j, z_j) in mol.atomic_numbers.iter().enumerate() {
                for _ in &mol.calculator.valorbs[z_j] {
                    h2[[mu, nu]] = 0.5 * (tmp[i] + tmp[j]);
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h2;
}
