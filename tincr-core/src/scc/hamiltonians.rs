use itertools::Itertools;
use log::{debug};
use ndarray::prelude::*;
use crate::{Atom, defaults, AtomSlice};
use crate::scc::energies::get_homo_lumo_gap;


// find indices of HOMO and LUMO orbitals (starting from 0)
pub fn get_frontier_orbitals(n_elec: usize) -> (usize, usize) {
    let homo: usize = (n_elec / 2) - 1;
    let lumo: usize = homo + 1;
    return (homo, lumo);
}

// find indices of HOMO and LUMO orbitals (starting from 0)
pub fn get_frontier_orbitals_from_occ(f: &[f64]) -> (usize, usize) {
    let n_occ: usize = f
        .iter()
        .enumerate()
        .filter_map(|(idx, val)| if *val > 0.5 { Some(idx) } else { None })
        .collect::<Vec<usize>>()
        .len();
    let homo: usize = n_occ - 1;
    let lumo: usize = homo + 1;
    return (homo, lumo);
}

/// the repulsive potential, the dispersion correction and only depend on the nuclear
/// geometry and do not change during the SCF cycle
fn get_nuclear_energy() {}

/// Construct part of the Hamiltonian corresponding to long range
/// Hartree-Fock exchange
/// H^x_mn = -1/2 sum_ab (P_ab-P0_ab) (ma|bn)_lr
/// The Coulomb potential in the electron integral is replaced by
/// 1/r ----> erf(r/R_lr)/r
pub fn lc_exact_exchange(
    s: ArrayView2<f64>,
    g0_lr_ao: ArrayView2<f64>,
    dp: ArrayView2<f64>,
) -> Array2<f64> {
    let mut hx: Array2<f64> = (&g0_lr_ao * &s.dot(&dp)).dot(&s);
    hx = hx + &g0_lr_ao * &(s.dot(&dp)).dot(&s);
    hx = hx + (s.dot(&(&dp * &g0_lr_ao))).dot(&s);
    hx = hx + s.dot(&(&g0_lr_ao * &dp.dot(&s)));
    hx = hx * -0.125;
    return hx;
}



/// Construct the density matrix
/// P_mn = sum_a f_a C_ma* C_na
pub fn density_matrix(orbs: ArrayView2<f64>, f: &[f64]) -> Array2<f64> {
    let occ_indx: Vec<usize> = f.iter().positions(|&x| x > 0.0).collect();
    let occ_orbs: Array2<f64> = orbs.select(Axis(1), &occ_indx);
    let f_occ: Vec<f64> = f.iter().filter(|&&x| x > 0.0).cloned().collect();
    // THIS IS NOT AN EFFICIENT WAY TO BUILD THE LEFT HAND SIDE
    let mut f_occ_mat: Vec<f64> = Vec::new();
    for _ in 0..occ_orbs.nrows() {
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
pub fn density_matrix_ref(n_orbs: usize, atoms: AtomSlice) -> Array2<f64> {
    let mut p0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    // iterate over orbitals on center i
    let mut idx: usize = 0;
    for valorbs_occ in atoms.valorbs_occupation.iter() {
        // how many electrons are put into the nl-shell
        for occ in valorbs_occ.iter() {
            p0[[idx, idx]] = *occ;
            idx += 1;
        }
    }
    return p0;
}

pub fn construct_h1(
    n_orbs: usize,
    atoms: AtomSlice,
    gamma: ArrayView2<f64>,
    dq: ArrayView1<f64>,
) -> Array2<f64> {
    let e_stat_pot: Array1<f64> = gamma.dot(&dq);
    let mut h1: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
    let mut mu: usize = 0;
    let mut nu: usize;
    for (n_orbs_i, esp_i) in atoms.n_orbs.iter().zip(e_stat_pot.iter()){
        for _ in 0..*n_orbs_i {
            nu = 0;
            for (n_orbs_j, esp_j) in atoms.n_orbs.iter().zip(e_stat_pot.iter()) {
                for _ in 0..*n_orbs_j {
                    h1[[mu, nu]] = 0.5 * (esp_i + esp_j);
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h1;
}

pub fn construct_h_magnetization(
    n_orbs: usize,
    atoms: &[Atom],
    dq: ArrayView1<f64>,
    spin_couplings:ArrayView1<f64>
) -> Array2<f64> {
    let pot: Array1<f64> = &dq * &spin_couplings;
    let mut h: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..(atomi.n_orbs) {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..(atomj.n_orbs) {
                    h[[mu, nu]] = 0.5 * (pot[i] + pot[j]);
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h;
}

pub fn enable_level_shifting(orbe: ArrayView1<f64>, n_elec: usize) -> bool {
    let hl_idxs: (usize, usize) = get_frontier_orbitals(n_elec);
    let gap: f64 = get_homo_lumo_gap(orbe.view(), hl_idxs);
    debug!("HOMO - LUMO gap:          {:>18.14}", gap);
    gap < defaults::HOMO_LUMO_TOL
}