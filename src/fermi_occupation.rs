use std::cmp::{max, min};
use ndarray::{Array, Array1};

fn fermi(en: f64, mu: f64, T: f64) -> f64 {
    return 2.0 / (((en - mu) / (kBoltzmann * T)).exp() + 1.0);
}

fn func(
    mu: f64,
    orbe: Array1<f64>,
    fermi_function: fn(f64, f64, f64) -> f64,
    t: f64,
    n_elec: usize,
) -> f64 {
    // find the root of this function to enforce sum_a f_a = Nelec
    let mut sum_fa: f64 = 0.0;
    for en_a in orbe.iter() {
        sum_fa = sum_fa + fermi_function(en_a, mu, t)
    }
    return sum_fa - (n_elec as f64);
}

// stolen from https://qiita.com/osanshouo/items/71b0272cd5e156cbf5f2
fn argsort<T: Ord>(v: &[T]) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].cmp(&v[j]));
    idx
}




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
fn fermi_occupation(
    orbe: Array1<f64>,
    n_elec_paired: usize,
    n_elec_unpaired: usize,
    t: f64,
) -> Vec<f64> {


    let n_elec: usize = n_elec_paired + n_elec_unpaired;
    let sort_indx: Vec<usize> = argsort(&orbe.to_vec());
    let h_idx: usize = max((n_elec / 2) - 1, 0);
    let homo: f64 = orbe[sort_indx[h_idx]];
    let lp1_idx: usize = max((n_elec / 2) + 1, sort_indx.len() - 1);
    let lumop1: f64 = orbe[sort_indx[lp1_idx]];
    // highest doubly occupied orbital
    //let homo
    // LUMO + 1
    // look for fermi energy in the interval [HOMO, LUMO+1]
    // find root of (sum_a fa - Nelec) by bisection
    // fa = func(a);
    // fb = func(b);
    // sign change within interval
    assert!(fa * fb <= 0.0);
    let dx = b - a;
    let dy = max(fa, fb);
    let tolerance: f64 = 1.0e-8;
    while (dx * *2 + dy * *2 > tolerance) {
        let c: f64 = (a + b) / 2.0;
        //let fa = func(a);
        //let fc = func(c);
        if fa * fc <= 0.0 {
            let b = c;
            let dx = c - a;
            let dy = fc;
        } else {
            let a = c;
            let dx = b - c;
            let dy = fc;
        }
    }
}