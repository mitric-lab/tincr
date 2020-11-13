use crate::constants::*;
use crate::defaults;
use crate::h0_and_s::h0_and_s_ab;
use crate::molecule::*;
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
    let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s_ab(&mol, &mol);
    let (gm, gm_a0): (Array2<f64>, Array2<f64>) = get_gamma_matrix(&mol, Some(0.0));
    for i in 0..max_iter {
        let h1: Array2<f64> = construct_h1(&mol, gm.view(), dq.view());
        let h_coul: Array2<f64> = h1.view() * s.view();
        let h: Array2<f64> = h_coul.view() + h0.view();
        //let x = a.solveh_into(b).unwrap();
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // H' = X^t.H.X
        let hp: Array2<f64> = x.conjugate().t().dot(&h).dot(&x);
        let (orbe, cp): (Array1<f64>, Array2<f64>) = hp.eigh().unwrap();
        // C = X.C'
        let orbs: Array2<f64> = cp.dot(&x);
        // construct density matrix
    }
    return 1.0;
}

fn density_matrix(orbs: Array2<f64>, f: Vec<f64>) -> Array2<f64> {
    //let occ_indx:
    //let occ_orbs: Vec<bool> = ;
    //let P: Array2<f64> = (f * occ_orbs).dot(&occ_orbs.t());

    return P;
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
