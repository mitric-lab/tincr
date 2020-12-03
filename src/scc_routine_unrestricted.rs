use crate::broyden::*;
use crate::calculator::*;
use crate::constants::*;
use crate::defaults;
use crate::diis::*;
use crate::fermi_occupation;
use crate::h0_and_s::h0_and_s;
use crate::molecule::*;
use crate::mulliken::*;
use approx::AbsDiffEq;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use std::cmp::max;
use std::iter::FromIterator;

// This routine is very messy und should be rewritten in a clean form
pub fn run_scc(
    molecule: &Molecule,
    max_iter: Option<usize>,
    scf_conv: Option<f64>,
    temperature: Option<f64>,
) -> f64 {
    let max_iter: usize = max_iter.unwrap_or(defaults::MAX_ITER);
    let scf_conv: f64 = scf_conv.unwrap_or(defaults::SCF_CONV);
    let temperature: f64 = temperature.unwrap_or(defaults::TEMPERATURE);

    // construct reference density matrix
    let p0: Array2<f64> = density_matrix_ref(&molecule);
    let mut p: Array2<f64> = Array2::zeros(p0.raw_dim());
    // charge guess
    let mut dq: Array1<f64> = Array1::zeros([molecule.n_atoms]);
    let mut q: Array1<f64> = Array::from_iter(molecule.calculator.q0.iter().cloned());
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

    let mut broyden_mixer: BroydenMixer = BroydenMixer::new(molecule.n_atoms);

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
        let h1: Array2<f64> = construct_h1(&molecule, gm.view(), dq.view());
        let h_coul: Array2<f64> = h1 * s.view();
        let mut h: Array2<f64> = h_coul + h0.view();

        // H' = X^t.H.X
        let hp: Array2<f64> = x.t().dot(&h).dot(&x);

        let (orbe, cp): (Array1<f64>, Array2<f64>) = hp.eigh(UPLO::Upper).unwrap();
        // C = X.C'
        let orbs: Array2<f64> = x.dot(&cp);

        // construct density matrix
        let tmp: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
            orbe.view(),
            molecule.calculator.q0.iter().sum::<f64>() as usize - molecule.charge as usize,
            molecule.calculator.nr_unpaired_electrons,
            temperature,
        );
        let mu: f64 = tmp.0;
        let f: Vec<f64> = tmp.1;

        // calculate the density matrix
        p = density_matrix(orbs.view(), &f[..]);

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

        // check if charge difference to the previus iteration is lower then 1e-5
        if (dq_diff.map(|x| x.abs()).max().unwrap() < &scf_conv) {
            converged = true;
        }
        // Broyden mixing of partial charges
        dq = broyden_mixer.next(new_dq, dq_diff);
        q = new_q;

        // compute electronic energy
        scf_energy = get_electronic_energy(p.view(), h0.view(), dq.view(), gm.view());

        energy_old = scf_energy;
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
fn get_electronic_energy(
    p: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
) -> f64 {
    //println!("P {}", p);
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));
    // electronic energy as sum of band structure energy and Coulomb energy
    //println!("E BS {} E COUL {} dQ {}", e_band_structure, e_coulomb, dq);
    let e_elec: f64 = e_band_structure + e_coulomb;
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

fn construct_h1(mol: &Molecule, gamma: ArrayView2<f64>, dq: ArrayView1<f64>) -> Array2<f64> {
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


