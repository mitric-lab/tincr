use crate::constants::*;
use crate::defaults;
use crate::fermi_occupation;
use crate::h0_and_s::h0_and_s_ab;
use crate::molecule::*;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use std::cmp::max;

// INCOMPLETE
pub fn run_scc(
    molecule: &Molecule,
    max_iter: Option<usize>,
    scf_conv: Option<f64>,
    temperature: Option<f64>,
) -> f64 {
    let max_iter: usize = max_iter.unwrap_or(defaults::MAX_ITER);
    let scf_conv: f64 = scf_conv.unwrap_or(defaults::SCF_CONV);
    let temperature: f64 = temperature.unwrap_or(defaults::TEMPERATURE);

    // charge guess
    let dq: Array1<f64> = Array1::zeros([molecule.n_atoms]);
    let ddip: Array2<f64> = Array2::zeros([molecule.n_atoms, 3]);
    let converged: bool = false;
    let shift_flag: bool = false;
    let mixing_flag: bool = false;
    let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s_ab(&molecule, &molecule);
    let (gm, gm_a0): (Array2<f64>, Array2<f64>) = get_gamma_matrix(&molecule, Some(0.0));

    let mut diis_error: Vec<Array1<f64>> = Vec::new();
    let mut fock_list: Vec<Array2<f64>> = Vec::new();

    for i in 0..max_iter {
        let h1: Array2<f64> = construct_h1(&molecule, gm.view(), dq.view());
        let h_coul: Array2<f64> = h1 * s.view();
        let h: Array2<f64> = h_coul + h0.view();
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // H' = X^t.H.X
        let hp: Array2<f64> = x.t().dot(&h).dot(&x);
        let (orbe, cp): (Array1<f64>, Array2<f64>) = hp.eigh(UPLO::Upper).unwrap();
        // C = X.C'
        let orbs: Array2<f64> = cp.dot(&x);
        // construct density matrix
        let tmp: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
            orbe.view(),
            molecule.q0.iter().sum() - molecule.charge as usize,
            molecule.nr_unpaired_electrons,
            temperature,
        );
        let mu: f64 = tmp.0;
        let f: Vec<f64> = tmp.1;
        // calculate the density matrix
        let p: Array2<f64> = density_matrix(orbs.view(), &f[..]);
        // use DIIS to speed up convergence
        // limit size of DIIS vector
        let mut diis_count: usize = fock_list.len();
        if diis_count > defaults::DIIS_LIMIT {
            // remove oldest vector
            fock_list.remove(0);
            diis_error.remove(0);
            diis_count -= 1;
        }

    }
    return 1.0;
}

/// Construct the density matrix
/// P_mn = sum_a f_a C_ma* C_na
fn density_matrix(orbs: ArrayView2<f64>, f: &[f64]) -> Array2<f64> {
    let occ_indx: Vec<usize> = f.iter().positions(|&x| x > 0.0).collect();
    let occ_orbs: Array2<f64> = orbs.a.slice_axis(Axis(1), Slice::from(occ_indx));
    let f_occ: Vec<f64> = f.iter().filter(|&x| x > &0.0).collect();
    let lhs: Vec<f64> = f_occ
        .iter()
        .zip(&occ_indx)
        .map(|(&i1, &i2)| i1 * i2)
        .collect();
    let p: Array2<f64> = Array2::from_shape_vec((lhs.len(), 1), lhs).dot(&occ_orbs.t());
    return p;
}

/// Construct reference density matrix
/// all atoms should be neutral
fn density_matrix_ref(mol: &Molecule) -> Array2<f64> {
    let mut p0: Array2<f64> = Array2::zeros((mol.n_orbs, mol.n_orbs));
    // iterate over orbitals on center i
    let mut idx: usize = 0;
    for (_i, (zi, _posi)) in mol.iter_atomlist() {
        // how many electrons are put into the nl-shell
        for (iv, (_ni, _li, _mi)) in mol.valorbs[*zi].iter().enumerate() {
            p0[[idx, idx]] = mol.valorbs_occupation[*zi][*iv];
            idx += 1;
        }
    }
    return p0;
}

// Mulliken Charges
fn mulliken(
    p: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    s: ArrayView2<f64>,
    orbs_per_atom: Vec<i64>,
    n_atom: usize,
) -> (Array1<f64>, Array1<f64>) {
    let dp = &p - &p0;

    let mut q: Array1<f64> = Array1::<f64>::zeros(n_atom);
    let mut dq: Array1<f64> = Array1::<f64>::zeros(n_atom);

    // iterate over atoms A
    let mut mu = 0;
    // WARNING: this loop cannot be parallelized easily because mu is incremented
    // inside the loop
    for A in 0..n_atom {
        // iterate over orbitals on atom A
        for muA in 0..orbs_per_atom[A] {
            let mut nu = 0;
            // iterate over atoms B
            for B in 0..n_atom {
                // iterate over orbitals on atom B
                for nuB in 0..orbs_per_atom[B] {
                    q[A] = q[A] + (&p[[mu, nu]] * &s[[mu, nu]]);
                    dq[A] = dq[A] + (&dp[[mu, nu]] * &s[[mu, nu]]);
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    (q, dq)
}

fn construct_h1(mol: &Molecule, gamma: ArrayView2<f64>, dq: ArrayView1<f64>) -> Array2<f64> {
    let e_stat_pot: Array1<f64> = gamma.dot(&dq);
    let mut h1: Array2<f64> = Array2::zeros([mol.n_orbs, mol.n_orbs]);

    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, (z_i, pos_i)) in mol.iter_atomlist().enumerate() {
        for (n_i, l_i, m_i) in &mol.valorbs[z_i] {
            nu = 0;
            for (j, (z_j, pos_j)) in mol.iter_atomlist().enumerate() {
                for (n_j, l_j, m_j) in &mol.valorbs[z_j] {
                    h1[[mu, nu]] = 0.5 * (e_stat_pot[i] + e_stat_pot[j]);
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h1;
}

#[test]
fn h1_construction() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers, positions, charge, multiplicity);
    let (gm, _gm_a0): (Array2<f64>, Array2<f64>) = get_gamma_matrix(&mol, Some(0.0));
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
    assert!(h1.all_close(&h1_ref, 1e-06));
}

#[test]
fn test_mat() {
    let mut a: Array2<f64> = array![[0.75592895, 1.13389342], [0.37796447, 1.88982237]];
    a = a.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
    let b: Array2<f64> = a.dot(&a);
    let (c, d) = b.eigh(UPLO::Upper).unwrap();
    println!("b : {}", c);
    assert_eq!(1, 2);
}
