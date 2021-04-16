use crate::broyden::*;
use crate::calculator::*;
use crate::constants::*;
use crate::defaults;
use crate::diis::*;
use crate::fermi_occupation;
use crate::h0_and_s::h0_and_s;
use crate::io::*;
use crate::molecule::*;
use crate::mulliken::*;
use crate::test::{get_benzene_molecule, get_water_molecule};
use approx::AbsDiffEq;
use itertools::Itertools;
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use rayon::prelude::*;
use std::cmp::max;
use std::iter::FromIterator;
use std::ops::{AddAssign, Deref};
use std::time::Instant;

pub fn generate_initial_fmo_monomer_guess(fragments: &mut Vec<Molecule>) -> (Vec<ncc_matrices>) {
    // create initial dq guess by doing one ncc run
    let x_vec: Vec<ncc_matrices> = fragments
        .iter_mut()
        .map(|frag| {
            let (energy, s, x_option, h, h0_coul, om_monomer): (
                f64,
                Array2<f64>,
                Option<Array2<f64>>,
                Option<Array2<f64>>,
                Option<Array2<f64>>,
                Option<Array1<f64>>,
            ) = fmo_ncc(frag, None, None, None, None, None, None, None, false);
            let matrices: ncc_matrices = ncc_matrices::new(s, h, x_option);
            matrices
        })
        .collect();
    return x_vec;
}

pub struct ncc_matrices {
    pub s: Array2<f64>,
    pub h0: Option<Array2<f64>>,
    pub x: Option<Array2<f64>>,
}
impl ncc_matrices {
    pub(crate) fn new(
        s: Array2<f64>,
        h0: Option<Array2<f64>>,
        x: Option<Array2<f64>>,
    ) -> (ncc_matrices) {
        let result = ncc_matrices { s: s, h0: h0, x: x };
        return result;
    }
}

// This routine is very messy und should be rewritten in a clean form
pub fn fmo_ncc(
    molecule: &mut Molecule,
    x_mat: Option<Array2<f64>>,
    s_mat: Option<Array2<f64>>,
    h_mat: Option<Array2<f64>>,
    dq_k: Option<Vec<Array1<f64>>>,
    g0_total: Option<ArrayView2<f64>>,
    index: Option<usize>,
    frag_atoms_index: Option<Vec<usize>>,
    calc_gradients: bool,
) -> (
    f64,
    Array2<f64>,
    Option<Array2<f64>>,
    Option<Array2<f64>>,
    Option<Array2<f64>>,
    Option<Array1<f64>>,
) {
    let temperature: f64 = molecule.config.scf.electronic_temperature;
    // construct reference density matrix
    let p0: Array2<f64> = density_matrix_ref(&molecule);
    let mut p: Array2<f64> = molecule.final_p_matrix.clone();
    let mut dq: Array1<f64> = molecule.final_charges.clone();
    let mut q: Array1<f64> = Array::from_iter(molecule.calculator.q0.iter().cloned());
    let mut charge_diff: f64 = 0.0;
    let mut x: Array2<f64> =
        Array2::zeros((molecule.calculator.n_orbs, molecule.calculator.n_orbs));
    let mut s: Array2<f64> =
        Array2::zeros((molecule.calculator.n_orbs, molecule.calculator.n_orbs));
    let mut h0: Array2<f64> =
        Array2::zeros((molecule.calculator.n_orbs, molecule.calculator.n_orbs));
    let mut h_opt: Option<Array2<f64>> = None;
    let mut x_option: Option<Array2<f64>> = None;
    let mut dq_monomers: Vec<Array1<f64>> = Vec::new();
    let mut h0_coul_opt: Option<Array2<f64>> = None;
    if dq_k.is_some() {
        dq_monomers = dq_k.unwrap();
    }

    if x_mat.is_none() {
        let (s_mat, h0_mat): (Array2<f64>, Array2<f64>) = h0_and_s(
            &molecule.atomic_numbers,
            molecule.positions.view(),
            molecule.calculator.n_orbs,
            &molecule.calculator.valorbs,
            molecule.proximity_matrix.view(),
            &molecule.calculator.skt,
            &molecule.calculator.orbital_energies,
        );
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x_temp: Array2<f64> = s_mat.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        x = x_temp;
        x_option = Some(x.clone());
        h0 = h0_mat;
        h_opt = Some(h0.clone());
        s = s_mat;
    } else {
        x = x_mat.unwrap();
        h0 = h_mat.unwrap();
        s = s_mat.unwrap();
    }
    let mut broyden_mixer: BroydenMixer = BroydenMixer::new(molecule.n_atoms);

    // add nuclear energy to the total scf energy
    let rep_energy: f64 = get_repulsive_energy(&molecule);

    let h1: Array2<f64> = construct_h1(&molecule, molecule.g0.view(), dq.view());
    let h_coul: Array2<f64> = h1 * s.view();
    let mut h: Array2<f64> = &h_coul + &h0;

    // return h0 + h_coul if calc_gradients is true
    if calc_gradients {
        h0_coul_opt = Some(h_coul + h0.view());
    }

    // calculate ESP term V
    let mut v_mat: Array2<f64> = Array::zeros(s.raw_dim());
    let mut om_monomer: Option<Array1<f64>> = None;
    if dq_monomers.len() > 0 {
        if g0_total.is_some() {
            let ind: usize = index.unwrap();
            let atoms_ind: Vec<usize> = frag_atoms_index.unwrap();
            let g0: ArrayView2<f64> = g0_total.unwrap();
            let mut esp: Array1<f64> = Array1::zeros(molecule.n_atoms);

            for (ind_k, dq_frag) in dq_monomers.iter().enumerate() {
                if ind_k != ind {
                    let g0_slice: ArrayView2<f64> = g0.slice(s![
                        atoms_ind[ind]..atoms_ind[ind] + molecule.n_atoms,
                        atoms_ind[ind_k]..atoms_ind[ind_k] + dq_frag.len()
                    ]);
                    // let esp: Array1<f64> = g0_slice.dot(dq_frag);
                    // let mut esp_ao: Array1<f64> = Array1::zeros(molecule.calculator.n_orbs);
                    //
                    // let mut mu: usize = 0;
                    // for (i, z_i) in molecule.atomic_numbers.iter().enumerate() {
                    //     for _ in &molecule.calculator.valorbs[z_i] {
                    //         esp_ao[mu] = esp[i];
                    //         mu = mu + 1;
                    //     }
                    // }
                    // let esp_arr: Array2<f64> = esp_ao.clone().insert_axis(Axis(1));
                    // v_temp.add_assign(
                    //     &(&esp_arr.broadcast((esp_ao.len(), esp_ao.len())).unwrap() + &esp_ao),
                    // );
                    esp = esp + g0_slice.dot(dq_frag);
                }
            }
            om_monomer = Some(esp.clone());
            let mut esp_ao: Array1<f64> = Array1::zeros(molecule.calculator.n_orbs);

            let mut mu: usize = 0;
            for (i, z_i) in molecule.atomic_numbers.iter().enumerate() {
                for _ in &molecule.calculator.valorbs[z_i] {
                    esp_ao[mu] = esp[i];
                    mu = mu + 1;
                }
            }
            let esp_arr: Array2<f64> = esp_ao.clone().insert_axis(Axis(1));
            let v_temp: Array2<f64> =
                &esp_arr.broadcast((esp_ao.len(), esp_ao.len())).unwrap() + &esp_ao;

            v_mat = 0.5 * v_temp * s.view();
        } else {
            // TODO: calculate g0 on the fly
        }
        h = h + v_mat;
    }

    //let mut prev_h_X:Array2<f64>
    if molecule.calculator.r_lr.is_none() || molecule.calculator.r_lr.unwrap() > 0.0 {
        let h_x: Array2<f64> = lc_exact_exchange(&s, &molecule.g0_lr_ao, &p0, &p, h.dim().0);
        h = h + h_x;
    }
    // H' = X^t.H.X
    h = x.t().dot(&h).dot(&x);
    let tmp: (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Upper).unwrap();
    let orbe: Array1<f64> = tmp.0;
    let orbs: Array2<f64> = x.dot(&tmp.1);

    // construct density matrix
    let tmp: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
        orbe.view(),
        molecule.calculator.q0.iter().sum::<f64>() as usize - molecule.charge as usize,
        molecule.calculator.nr_unpaired_electrons,
        temperature,
    );
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
    charge_diff = dq_diff.map(|x| x.abs()).max().unwrap().to_owned();

    // Broyden mixing of partial charges # changed new_dq to dq
    dq = broyden_mixer.next(dq, dq_diff);
    q = new_q;

    // compute electronic energy
    let scf_energy: f64 = get_electronic_energy(
        &molecule,
        p.view(),
        &p0,
        &s,
        h0.view(),
        dq.view(),
        (&molecule.g0).deref().view(),
        &molecule.g0_lr_ao,
    );
    molecule.set_final_charges(dq);
    molecule.set_final_p_mat(p);

    return (
        scf_energy + rep_energy,
        s,
        x_option,
        h_opt,
        h0_coul_opt,
        om_monomer,
    );
}

pub fn fmo_pair_ncc(
    molecule: &mut Molecule,
    atoms_a: usize,
    atoms_b: usize,
    index_a: usize,
    index_b: usize,
    calc_gradients: bool,
    om_matrices: &Vec<Array1<f64>>,
) -> (f64, Array2<f64>, Option<Array2<f64>>) {
    let temperature: f64 = molecule.config.scf.electronic_temperature;
    // construct reference density matrix
    let p0: Array2<f64> = density_matrix_ref(&molecule);
    let mut p: Array2<f64> = molecule.final_p_matrix.clone();
    let mut dq: Array1<f64> = molecule.final_charges.clone();
    let mut q: Array1<f64> = Array::from_iter(molecule.calculator.q0.iter().cloned());
    let mut charge_diff: f64 = 0.0;
    let mut h0_coul_opt: Option<Array2<f64>> = None;

    let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(
        &molecule.atomic_numbers,
        molecule.positions.view(),
        molecule.calculator.n_orbs,
        &molecule.calculator.valorbs,
        molecule.proximity_matrix.view(),
        &molecule.calculator.skt,
        &molecule.calculator.orbital_energies,
    );
    // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
    // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
    let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
    let mut broyden_mixer: BroydenMixer = BroydenMixer::new(molecule.n_atoms);

    // add nuclear energy to the total scf energy
    let rep_energy: f64 = get_repulsive_energy(&molecule);

    let h1: Array2<f64> = construct_h1(&molecule, molecule.g0.view(), dq.view());
    let h_coul: Array2<f64> = h1 * s.view();
    let mut h: Array2<f64> = &h_coul + &h0;

    // return h0 + h_coul if calc_gradients is true
    if calc_gradients {
        h0_coul_opt = Some(h_coul + h0.view());
    }

    // calculate ESP term V
    let g0_ab: ArrayView2<f64> = molecule.g0.slice(s![0..atoms_a, atoms_a..]);
    let esp_a: Array1<f64> = -g0_ab.dot(&molecule.final_charges.slice(s![atoms_a..])) + om_matrices[index_a].view();
    let esp_b: Array1<f64> = -g0_ab.t().dot(&molecule.final_charges.slice(s![0..atoms_a])) + om_matrices[index_b].view();

    let esp: Array1<f64> = stack(Axis(0), &[esp_a.view(), esp_b.view()]).unwrap();
    let mut esp_ao: Array1<f64> = Array1::zeros(molecule.calculator.n_orbs);

    let mut mu: usize = 0;
    for (i, z_i) in molecule.atomic_numbers.iter().enumerate() {
        for _ in &molecule.calculator.valorbs[z_i] {
            esp_ao[mu] = esp[i];
            mu = mu + 1;
        }
    }
    let esp_arr: Array2<f64> = esp_ao.clone().insert_axis(Axis(1));
    let v_temp: Array2<f64> = &esp_arr.broadcast((esp_ao.len(), esp_ao.len())).unwrap() + &esp_ao;
    let v_mat: Array2<f64> = 0.5 * v_temp * s.view();
    h = h + v_mat;

    //let mut prev_h_X:Array2<f64>
    if molecule.calculator.r_lr.is_none() || molecule.calculator.r_lr.unwrap() > 0.0 {
        let h_x: Array2<f64> = lc_exact_exchange(&s, &molecule.g0_lr_ao, &p0, &p, h.dim().0);
        h = h + h_x;
    }
    // H' = X^t.H.X
    h = x.t().dot(&h).dot(&x);
    let tmp: (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Upper).unwrap();
    let orbe: Array1<f64> = tmp.0;
    let orbs: Array2<f64> = x.dot(&tmp.1);

    // construct density matrix
    let tmp: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
        orbe.view(),
        molecule.calculator.q0.iter().sum::<f64>() as usize - molecule.charge as usize,
        molecule.calculator.nr_unpaired_electrons,
        temperature,
    );
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
    charge_diff = dq_diff.map(|x| x.abs()).max().unwrap().to_owned();

    // Broyden mixing of partial charges # changed new_dq to dq
    dq = broyden_mixer.next(dq, dq_diff);
    q = new_q;

    // compute electronic energy
    let scf_energy: f64 = get_electronic_energy(
        &molecule,
        p.view(),
        &p0,
        &s,
        h0.view(),
        dq.view(),
        (&molecule.g0).deref().view(),
        &molecule.g0_lr_ao,
    );
    molecule.set_final_charges(dq);
    molecule.set_final_p_mat(p);

    return (scf_energy + rep_energy, s, h0_coul_opt);
}

// find indeces of HOMO and LUMO orbitals (starting from 0)
fn get_frontier_orbitals(n_elec: usize) -> (usize, usize) {
    let homo: usize = (n_elec / 2) - 1;
    let lumo: usize = homo + 1;
    return (homo, lumo);
}

// find indeces of HOMO and LUMO orbitals (starting from 0)
fn get_frontier_orbitals_from_occ(f: &[f64]) -> (usize, usize) {
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

// compute HOMO-LUMO gap in Hartree
fn get_homo_lumo_gap(orbe: ArrayView1<f64>, homo_lumo_idx: (usize, usize)) -> f64 {
    orbe[homo_lumo_idx.1] - orbe[homo_lumo_idx.0]
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
    mol: &Molecule,
    p: ArrayView2<f64>,
    p0: &Array2<f64>,
    s: &Array2<f64>,
    h0: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
    g0_lr_ao: &Array2<f64>,
) -> f64 {
    //println!("P {}", p);
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));
    // electronic energy as sum of band structure energy and Coulomb energy
    //println!("E BS {} E COUL {} dQ {}", e_band_structure, e_coulomb, dq);
    let mut e_elec: f64 = e_band_structure + e_coulomb;
    // add lc exchange to electronic energy
    if mol.calculator.r_lr.is_none() || mol.calculator.r_lr.unwrap() > 0.0 {
        let e_hf_x: f64 = lc_exchange_energy(s, g0_lr_ao, p0, &p.to_owned());
        e_elec += e_hf_x;
    }
    // long-range Hartree-Fock exchange
    // if ....Iteration {} =>
    //println!("               E_bs = {:.7}  E_coulomb = {:.7}", e_band_structure, e_coulomb);
    return e_elec;
}

fn lc_exact_exchange(
    s: &Array2<f64>,
    g0_lr_ao: &Array2<f64>,
    p0: &Array2<f64>,
    p: &Array2<f64>,
    dim: usize,
) -> (Array2<f64>) {
    // construct part of the Hamiltonian matrix corresponding to long range
    // Hartree-Fock exchange
    // H^x_mn = -1/2 sum_ab (P_ab-P0_ab) (ma|bn)_lr
    // The Coulomb potential in the electron integral is replaced by
    // 1/r ----> erf(r/R_lr)/r
    let mut hx: Array2<f64> = Array::zeros((dim, dim));
    let dp: Array2<f64> = p - p0;

    hx = hx + (g0_lr_ao * &s.dot(&dp)).dot(s);
    hx = hx + g0_lr_ao * &(s.dot(&dp)).dot(s);
    hx = hx + (s.dot(&(&dp * g0_lr_ao))).dot(s);
    hx = hx + s.dot(&(g0_lr_ao * &dp.dot(s)));
    hx = hx * (-0.125);
    return hx;
}

fn lc_exchange_energy(
    s: &Array2<f64>,
    g0_lr_ao: &Array2<f64>,
    p0: &Array2<f64>,
    p: &Array2<f64>,
) -> f64 {
    let dp: Array2<f64> = p - p0;
    let mut e_hf_x: f64 = 0.0;

    e_hf_x += ((s.dot(&dp.dot(s))) * &dp * g0_lr_ao).sum();
    e_hf_x += (s.dot(&dp) * dp.dot(s) * g0_lr_ao).sum();
    e_hf_x *= (-0.125);
    return e_hf_x;
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
pub fn density_matrix_ref(mol: &Molecule) -> Array2<f64> {
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
