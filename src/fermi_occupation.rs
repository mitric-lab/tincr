use crate::constants;
use crate::zbrent::zbrent;
use ndarray::{array, Array, Array1, ArrayView1, ArrayView2};
use std::cmp::{max, min};

/// Find the occupation of single-particle state a at finite temperature T
/// according to the Fermi distribution:
///     $f_a = f(en_a) = 2 /(exp(en_a - mu)/(kB*T) + 1)$
/// The chemical potential is determined from the condition that
/// sum_a f_a = Nelec
///
/// Parameters:
/// ===========
/// orbe: orbital energies
/// Nelec_paired: number of paired electrons, these electron will be placed in the same orbital
/// Nelec_unpaired: number of unpaired electrons, these electrons will sit in singly occupied
///                 orbitals (only works at T=0)
/// T: temperature in Kelvin
///
/// Returns:
/// ========
/// mu: chemical potential
/// f: list of occupations f[a] for orbitals (in the same order as the energies in orbe)
pub fn fermi_occupation(
    orbe: ArrayView1<f64>,
    n_elec_paired: usize,
    n_elec_unpaired: usize,
    t: f64,
) -> (f64, Vec<f64>) {
    let mut fermi_occ: Vec<f64> = Vec::new();
    let mut mu: f64 = 0.0;
    if t == 0.0 {
        let result: (f64, Vec<f64>) = fermi_occupation_t0(orbe, n_elec_paired, n_elec_unpaired);
        mu = result.0;
        fermi_occ = result.1;
    } else {
        let n_elec: usize = n_elec_paired + n_elec_unpaired;
        let sort_indx: Vec<usize> = argsort(&orbe.to_vec());
        // highest doubly occupied orbital
        let h_idx: usize = max((n_elec / 2) - 1, 0);
        let homo: f64 = orbe[sort_indx[h_idx]];
        // LUMO + 1
        let lp1_idx: usize = max((n_elec / 2) + 1, sort_indx.len() - 1);
        let lumop1: f64 = orbe[sort_indx[lp1_idx]];
        // search for fermi energy in the interval [HOMO, LUMO+1]
        let func = |x: f64| -> f64 { fa_minus_nelec(x, orbe.view(), fermi, t, n_elec) };
        mu = zbrent(func, homo, lumop1, 1.0e-08, 100);
        let dn: f64 = func(mu);
        assert!(dn.abs() <= 1.0e-08);
        for en in orbe.iter() {
            fermi_occ.push(fermi(*en, mu, t));
        }
    }
    return (mu, fermi_occ);
}

/// Find the occupation of single-particle states at T=0
fn fermi_occupation_t0(
    orbe: ArrayView1<f64>,
    n_elec_paired: usize,
    n_elec_unpaired: usize,
) -> (f64, Vec<f64>) {
    let mut n_elec_paired: f64 = n_elec_paired as f64;
    let mut n_elec_unpaired: f64 = n_elec_unpaired as f64;
    let sort_indx: Vec<usize> = argsort(orbe.as_slice().unwrap());
    let mut fermi_occ: Vec<f64> = vec![0.0; orbe.len()];
    for a in sort_indx.iter() {
        if n_elec_paired > 0.0 {
            fermi_occ[*a] = 2.0_f64.min(n_elec_paired);
            n_elec_paired = n_elec_paired - 2.0;
        } else {
            if n_elec_unpaired > 0.0 {
                fermi_occ[*a] = 1.0_f64.min(n_elec_unpaired);
                n_elec_unpaired = n_elec_unpaired - 1.0;
            }
        }
    }
    return (0.0, fermi_occ);
}

// original code from from https://qiita.com/osanshouo/items/71b0272cd5e156cbf5f2
fn argsort(v: &[f64]) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    idx
}

fn fermi(en: f64, mu: f64, t: f64) -> f64 {
    2.0 / (((en - mu) / (constants::K_BOLTZMANN * t)).exp() + 1.0)
}

fn fa_minus_nelec(
    mu: f64,
    orbe: ArrayView1<f64>,
    fermi_function: fn(f64, f64, f64) -> f64,
    t: f64,
    n_elec: usize,
) -> f64 {
    // find the root of this function to enforce sum_a f_a = Nelec
    let mut sum_fa: f64 = 0.0;
    for en_a in orbe.iter() {
        sum_fa = sum_fa + fermi_function(*en_a, mu, t)
    }
    return sum_fa - (n_elec as f64);
}

#[test]
fn fermi_occ_at_t0() {
    let orbe: Array1<f64> = array![
        -0.8274698649039047,
        -0.4866977381657900,
        -0.4293504325361446,
        -0.3805317817759825,
        0.4597732008355508,
        0.5075648461370381
    ];
    let temperature: f64 = 0.0;
    let n_elec: usize = 8;
    let n_elec_unpaired: usize = 0;
    let mu_ref: f64 = 0.0;
    let occ_ref: Vec<f64> = vec![2.0, 2.0, 2.0, 2.0, 0.0, 0.0];
    let result: (f64, Vec<f64>) =
        fermi_occupation(orbe.view(), n_elec, n_elec_unpaired, temperature);
    let mu: f64 = result.0;
    let occ: Vec<f64> = result.1;
    assert!((mu - mu_ref).abs() < 1e-8);
    assert_eq!(occ, occ_ref);
}

#[test]
fn fermi_occ_at_t100k() {
    let orbe: Array1<f64> = array![
        -0.8274698649039047,
        -0.4866977381657900,
        -0.4293504325361446,
        -0.3805317817759825,
        0.4597732008355508,
        0.5075648461370381
    ];
    let temperature: f64 = 100.0;
    let n_elec: usize = 8;
    let n_elec_unpaired: usize = 0;
    let mu_ref: f64 = -0.3692029124379807;
    let occ_ref: Vec<f64> = vec![2.0, 2.0, 2.0, 1.9999999999999996, 0.0, 0.0];
    let result: (f64, Vec<f64>) =
        fermi_occupation(orbe.view(), n_elec, n_elec_unpaired, temperature);
    let mu: f64 = result.0;
    let occ: Vec<f64> = result.1;
    // TODO: Check the differences to DFTBaby
    //assert!((mu-mu_ref).abs() < 1e-4);
    //assert_eq!(occ, occ_ref);
    assert!((occ.iter().sum::<f64>() - n_elec as f64).abs() < 1e-08);
}

#[test]
fn fermi_occ_at_t100000k() {
    let orbe: Array1<f64> = array![
        -0.8274698649039047,
        -0.4866977381657900,
        -0.4293504325361446,
        -0.3805317817759825,
        0.4597732008355508,
        0.5075648461370381
    ];
    let temperature: f64 = 100000.0;
    let n_elec: usize = 8;
    let n_elec_unpaired: usize = 0;
    let mu_ref: f64 = 0.1259066212142123;
    let occ_ref: Vec<f64> = vec![
        1.906094704507551,
        1.747482667930276,
        1.7047529931205556,
        1.6638147878851874,
        0.5168129976856992,
        0.4611095080472262,
    ];
    let result: (f64, Vec<f64>) =
        fermi_occupation(orbe.view(), n_elec, n_elec_unpaired, temperature);
    let mu: f64 = result.0;
    let occ: Vec<f64> = result.1;
    assert!((mu - mu_ref).abs() < 1e-4);
    //assert_eq!(occ, occ_ref);
    assert!((occ.iter().sum::<f64>() - n_elec as f64).abs() < 1e-08);
}
