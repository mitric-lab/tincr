use crate::constants::ATOM_NAMES;
use crate::defaults;
use crate::gamma_approximation;
use crate::molecule::Molecule;
use crate::parameters::*;
use itertools::Itertools;
use log::{debug, error, info, trace, warn};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use std::collections::HashMap;
use std::f64::consts::{E, PI};
use std::hash::Hash;
use std::ops::{Deref, Neg};

pub enum Calculator {
    DFTB(DFTBCalculator),
}

#[derive(Clone)]
pub struct DFTBCalculator {
    pub valorbs: HashMap<u8, Vec<(i8, i8, i8)>>,
    pub hubbard_u: HashMap<u8, f64>,
    pub spin_couplings: Array1<f64>,
    pub valorbs_occupation: HashMap<u8, Vec<f64>>,
    pub orbital_energies: HashMap<u8, HashMap<(i8, i8), f64>>,
    pub skt: HashMap<(u8, u8), SlaterKosterTable>,
    pub v_rep: HashMap<(u8, u8), RepulsivePotentialTable>,
    pub q0: Vec<f64>,
    pub nr_unpaired_electrons: usize,
    pub orbs_per_atom: Vec<usize>,
    pub n_orbs: usize,
    pub active_orbitals: Option<(usize, usize)>,
    pub r_lr: Option<f64>,
    pub active_occ: Option<Vec<usize>>,
    pub active_virt: Option<Vec<usize>>,
    pub full_occ: Option<Vec<usize>>,
    pub full_virt: Option<Vec<usize>>,
    pub n_elec: usize,
}

impl DFTBCalculator {
    pub fn new(
        atomic_numbers: &[u8],
        atomtypes: &HashMap<u8, String>,
        active_orbitals: Option<(usize, usize)>,
        distance_matrix: &Array2<f64>,
        r_lr: Option<f64>,
    ) -> DFTBCalculator {
        let mut unique_numbers: Vec<u8> = Vec::from(atomic_numbers);
        unique_numbers.sort_unstable(); // fast sort of atomic numbers
        unique_numbers.dedup(); // delete duplicates
        let (valorbs, valorbs_occupation, ne_val, orbital_energies, hubbard_u, spin_coupling_map): (
            HashMap<u8, Vec<(i8, i8, i8)>>,
            HashMap<u8, Vec<f64>>,
            HashMap<u8, i8>,
            HashMap<u8, HashMap<(i8, i8), f64>>,
            HashMap<u8, f64>,
            HashMap<u8, f64>,
        ) = get_electronic_configuration(&atomtypes);
        // NOTE: The spin-coupling constants are taken from
        // https://dftb.org/fileadmin/DFTB/public/slako/mio/mio-1-1.spinw.txt
        // However, the coupling constants are only tabulated there on an angular momentum level
        // and we use only one spin-coupling constant per element type. Therefore, the average was
        // used in the confined_pseudo_atom parameter files
        let tmp: Vec<f64> = atomic_numbers
            .iter()
            .map(|x| spin_coupling_map[x])
            .collect();
        let spin_couplings: Array1<f64> = Array1::from(tmp);
        let q0: Vec<f64> = atomic_numbers.iter().map(|zi| ne_val[zi] as f64).collect();
        let orbs_per_atom: Vec<usize> = atomic_numbers.iter().map(|zi| valorbs[zi].len()).collect();
        let (skt, vrep): (
            HashMap<(u8, u8), SlaterKosterTable>,
            HashMap<(u8, u8), RepulsivePotentialTable>,
        ) = get_parameters(unique_numbers);
        let mut n_orbs: usize = 0;
        for zi in atomic_numbers {
            n_orbs = n_orbs + &valorbs[zi].len();
        }

        let number_electrons_per_atom: Vec<usize> =
            atomic_numbers.iter().map(|x| ne_val[x] as usize).collect();
        let number_electrons: usize = Array1::from(number_electrons_per_atom).sum();
        info!("{: <25} {}", "number of orbitals:", n_orbs);
        info!("{: <25} {}", "number of electrons", number_electrons);

        DFTBCalculator {
            valorbs: valorbs,
            hubbard_u: hubbard_u,
            spin_couplings: spin_couplings,
            valorbs_occupation: valorbs_occupation,
            orbital_energies: orbital_energies,
            skt: skt,
            v_rep: vrep,
            q0: q0,
            nr_unpaired_electrons: 0,
            orbs_per_atom: orbs_per_atom,
            n_orbs: n_orbs,
            active_orbitals: active_orbitals,
            r_lr: r_lr,
            active_occ: None,
            active_virt: None,
            full_occ: None,
            full_virt: None,
            n_elec: number_electrons,
        }
    }

    pub fn set_active_orbitals(&mut self, f: Vec<f64>) {
        let tmp: (usize, usize) = self.active_orbitals.unwrap_or(defaults::ACTIVE_ORBITALS);
        let mut nr_active_occ: usize = tmp.0;
        let mut nr_active_virt: usize = tmp.1;
        let f: Array1<f64> = Array::from_vec(f);
        let occ_indices: Array1<usize> = f
            .indexed_iter()
            .filter_map(|(index, &item)| if item > 0.1 { Some(index) } else { None })
            .collect();
        let virt_indices: Array1<usize> = f
            .indexed_iter()
            .filter_map(|(index, &item)| if item <= 0.1 { Some(index) } else { None })
            .collect();

        if nr_active_occ >= occ_indices.len() {
            nr_active_occ = occ_indices.len();
        }
        if nr_active_virt >= virt_indices.len() {
            nr_active_virt = virt_indices.len();
        }

        let active_occ_indices: Vec<usize> = (occ_indices
            .slice(s![(occ_indices.len() - nr_active_occ)..])
            .to_owned())
        .to_vec();
        let active_virt_indices: Vec<usize> =
            (virt_indices.slice(s![..nr_active_virt]).to_owned()).to_vec();

        let full_occ_indices: Vec<usize> = occ_indices.to_vec();
        let full_virt_indices: Vec<usize> = virt_indices.to_vec();

        self.active_occ = Some(active_occ_indices);
        self.active_virt = Some(active_virt_indices);
        self.full_occ = Some(full_occ_indices);
        self.full_virt = Some(full_virt_indices);
    }
}

pub fn import_pseudo_atom(zi: &u8) -> (PseudoAtom, PseudoAtom) {
    let symbol: &str = ATOM_NAMES[*zi as usize];
    let free_atom: PseudoAtom = get_free_pseudo_atom(symbol);
    let confined_atom: PseudoAtom = get_confined_pseudo_atom(symbol);
    return (confined_atom, free_atom);
}

// pub fn set_active_orbitals(
//     f: Vec<f64>,
//     active_orbitals: Option<(usize, usize)>,
// ) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
//     let tmp: (usize, usize) = active_orbitals.unwrap_or(defaults::ACTIVE_ORBITALS);
//     let mut nr_active_occ: usize = tmp.0;
//     let mut nr_active_virt: usize = tmp.1;
//     let f: Array1<f64> = Array::from_vec(f);
//     let occ_indices: Array1<usize> = f
//         .indexed_iter()
//         .filter_map(|(index, &item)| if item > 0.1 { Some(index) } else { None })
//         .collect();
//     let virt_indices: Array1<usize> = f
//         .indexed_iter()
//         .filter_map(|(index, &item)| if item <= 0.1 { Some(index) } else { None })
//         .collect();
//
//     if nr_active_occ >= occ_indices.len() {
//         nr_active_occ = occ_indices.len();
//     }
//     if nr_active_virt >= virt_indices.len() {
//         nr_active_virt = virt_indices.len();
//     }
//
//     let active_occ_indices: Vec<usize> = (occ_indices
//         .slice(s![(occ_indices.len() - nr_active_occ)..])
//         .to_owned())
//     .to_vec();
//     let active_virt_indices: Vec<usize> =
//         (virt_indices.slice(s![..nr_active_virt]).to_owned()).to_vec();
//
//     let full_occ_indices: Vec<usize> = occ_indices.to_vec();
//     let full_virt_indices: Vec<usize> = virt_indices.to_vec();
//
//     return (
//         active_occ_indices,
//         active_virt_indices,
//         full_occ_indices,
//         full_virt_indices,
//     );
// }

pub fn construct_gaussian_overlap(molecule: &Molecule) -> (Array2<f64>) {
    let n_at = molecule.n_atoms;
    let mut gauss_omega: Array2<f64> = Array::zeros((n_at, n_at));
    let mut sigma: Array1<f64> = Array::zeros(n_at);
    for (index, i) in molecule.atomic_numbers.iter().enumerate() {
        sigma[index] = 1.329 / (8.0 * 2.0_f64.log(E)) * 1.0 / (molecule.calculator.hubbard_u[i]);
    }
    for (i, pos_i) in molecule.positions.outer_iter().enumerate() {
        for (j, pos_j) in molecule.positions.outer_iter().enumerate() {
            let r_ij: Array1<f64> = pos_i.to_owned() - pos_j.to_owned();
            let sig_ab: f64 = sigma[i].powi(2) + sigma[j].powi(2);
            let exp_arg: f64 = -0.5 * 1.0 / sig_ab * r_ij.dot(&r_ij);
            gauss_omega[[i, j]] = 1.0 / (2.0_f64 * PI * sig_ab).powf(3.0 / 2.0) * exp_arg.exp();
        }
    }
    return gauss_omega;
}

pub fn lambda2_calc_oia(
    molecule: &Molecule,
    active_occ: &Vec<usize>,
    active_virt: &Vec<usize>,
    qtrans_oo: &Array3<f64>,
    qtrans_vv: &Array3<f64>,
) -> (Array2<f64>) {
    let gauss_omega: Array2<f64> = construct_gaussian_overlap(molecule);
    let n_at: usize = molecule.n_atoms;
    let dim_o: usize = active_occ.len();
    let dim_v: usize = active_virt.len();
    let mut o_ia: Array2<f64> = Array::zeros((dim_o, dim_v));
    for (i, index_occ) in active_occ.iter().enumerate() {
        let o_ii: f64 = qtrans_oo
            .slice(s![.., i, i])
            .dot(&gauss_omega.dot(&qtrans_oo.slice(s![.., i, i])));
        for (j, index_virt) in active_virt.iter().enumerate() {
            let o_jj: f64 = qtrans_vv
                .slice(s![.., j, j])
                .dot(&gauss_omega.dot(&qtrans_vv.slice(s![.., j, j])));
            o_ia[[i, j]] = qtrans_oo
                .slice(s![.., i, i])
                .dot(&gauss_omega.dot(&qtrans_vv.slice(s![.., j, j])));
            o_ia[[i, j]] = o_ia[[i, j]] / (o_ii * o_jj).sqrt();
        }
    }
    return o_ia;
}

pub fn get_gamma_matrix(
    atomic_numbers: &[u8],
    n_atoms: usize,
    n_orbs: usize,
    distances: ArrayView2<f64>,
    hubbard_u: &HashMap<u8, f64>,
    valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
    r_lr: Option<f64>,
) -> (Array2<f64>, Array2<f64>) {
    // initialize gamma matrix
    let sigma: HashMap<u8, f64> = gamma_approximation::gaussian_decay(hubbard_u);
    let mut c: HashMap<(u8, u8), f64> = HashMap::new();
    let r_lr: f64 = r_lr.unwrap_or(defaults::LONG_RANGE_RADIUS);
    let mut gf = gamma_approximation::GammaFunction::Gaussian { sigma, c, r_lr };
    gf.initialize();
    let (gm, gm_ao): (Array2<f64>, Array2<f64>) =
        gamma_approximation::gamma_ao_wise(gf, atomic_numbers, n_atoms, n_orbs, distances, valorbs);

    return (gm, gm_ao);
}

pub fn get_only_gamma_matrix_atomwise(
    atomic_numbers: &[u8],
    n_atoms: usize,
    distances: ArrayView2<f64>,
    hubbard_u: &HashMap<u8, f64>,
    r_lr: Option<f64>,
) -> (Array2<f64>) {
    // initialize gamma matrix
    let sigma: HashMap<u8, f64> = gamma_approximation::gaussian_decay(hubbard_u);
    let mut c: HashMap<(u8, u8), f64> = HashMap::new();
    let r_lr: f64 = r_lr.unwrap_or(defaults::LONG_RANGE_RADIUS);
    let mut gf = gamma_approximation::GammaFunction::Gaussian { sigma, c, r_lr };
    gf.initialize();
    let g0: Array2<f64> =
        gamma_approximation::gamma_atomwise(gf, atomic_numbers, n_atoms, distances);

    return g0;
}

pub fn get_gamma_gradient_matrix_atom_wise_outer_diagonal(
    atomic_numbers_dimer: &[u8],
    atomic_numbers_frag:&[u8],
    n_atoms_dimer: usize,
    n_atoms_frag:usize,
    distances: ArrayView2<f64>,
    directions: ArrayView3<f64>,
    hubbard_u: &HashMap<u8, f64>,
    r_lr: Option<f64>,
) -> Array3<f64> {
    // initialize gamma matrix
    let sigma: HashMap<u8, f64> = gamma_approximation::gaussian_decay(hubbard_u);
    let mut c: HashMap<(u8, u8), f64> = HashMap::new();
    let r_lr: f64 = r_lr.unwrap_or(defaults::LONG_RANGE_RADIUS);
    let mut gf = gamma_approximation::GammaFunction::Gaussian { sigma, c, r_lr };
    gf.initialize();
    let g1:Array3<f64> = gamma_approximation::gamma_gradients_atomwise_outer_diagonal(gf, atomic_numbers_dimer, atomic_numbers_frag,n_atoms_dimer,n_atoms_frag, distances,directions);

    return g1;
}

pub fn get_gamma_gradient_matrix(
    atomic_numbers: &[u8],
    n_atoms: usize,
    n_orbs: usize,
    distances: ArrayView2<f64>,
    directions: ArrayView3<f64>,
    hubbard_u: &HashMap<u8, f64>,
    valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
    r_lr: Option<f64>,
) -> (Array3<f64>, Array3<f64>) {
    // initialize gamma matrix
    let sigma: HashMap<u8, f64> = gamma_approximation::gaussian_decay(hubbard_u);
    let mut c: HashMap<(u8, u8), f64> = HashMap::new();
    let r_lr: f64 = r_lr.unwrap_or(defaults::LONG_RANGE_RADIUS);
    let mut gf = gamma_approximation::GammaFunction::Gaussian { sigma, c, r_lr };
    gf.initialize();
    let (g1, g1_ao): (Array3<f64>, Array3<f64>) = gamma_approximation::gamma_gradients_ao_wise(
        gf,
        atomic_numbers,
        n_atoms,
        n_orbs,
        distances,
        directions,
        valorbs,
    );
    return (g1, g1_ao);
}

fn get_parameters(
    numbers: Vec<u8>,
) -> (
    HashMap<(u8, u8), SlaterKosterTable>,
    HashMap<(u8, u8), RepulsivePotentialTable>,
) {
    // find unique atom pairs and initialize Slater-Koster tables
    let atompairs = numbers.clone().into_iter().cartesian_product(numbers);
    let mut skt: HashMap<(u8, u8), SlaterKosterTable> = HashMap::new();
    let mut v_rep: HashMap<(u8, u8), RepulsivePotentialTable> = HashMap::new();
    'pair_loop: for pair in atompairs {
        let zi: u8 = pair.0;
        let zj: u8 = pair.1;
        // the cartesian product creates all combinations, but we only need one
        if zi > zj {
            continue 'pair_loop;
        }
        // load precalculated slako table
        let mut slako_module: SlaterKosterTable =
            get_slako_table(ATOM_NAMES[zi as usize], ATOM_NAMES[zj as usize]);
        slako_module.s_spline = slako_module.spline_overlap();
        slako_module.h_spline = slako_module.spline_hamiltonian();
        // load repulsive potential table
        let reppot_module: RepulsivePotentialTable =
            get_reppot_table(ATOM_NAMES[zi as usize], ATOM_NAMES[zj as usize]);
        skt.insert((zi, zj), slako_module);
        v_rep.insert((zi, zj), reppot_module);
    }
    return (skt, v_rep);
}

fn get_electronic_configuration(
    atomtypes: &HashMap<u8, String>,
) -> (
    HashMap<u8, Vec<(i8, i8, i8)>>,
    HashMap<u8, Vec<f64>>,
    HashMap<u8, i8>,
    HashMap<u8, HashMap<(i8, i8), f64>>,
    HashMap<u8, f64>,
    HashMap<u8, f64>,
) {
    // find quantum numbers of valence orbitals
    let mut valorbs: HashMap<u8, Vec<(i8, i8, i8)>> = HashMap::new();
    let mut valorbs_occupation: HashMap<u8, Vec<f64>> = HashMap::new();
    let mut ne_val: HashMap<u8, i8> = HashMap::new();
    let mut orbital_energies: HashMap<u8, HashMap<(i8, i8), f64>> = HashMap::new();
    let mut hubbard_u: HashMap<u8, f64> = HashMap::new();
    let mut spin_couplings: HashMap<u8, f64> = HashMap::new();
    for (zi, symbol) in atomtypes.iter() {
        let (atom, free_atom): (PseudoAtom, PseudoAtom) = import_pseudo_atom(zi);
        let mut occ: Vec<f64> = Vec::new();
        let mut vo_vec: Vec<(i8, i8, i8)> = Vec::new();
        let mut val_e: i8 = 0;
        hubbard_u.insert(*zi, atom.hubbard_u);
        spin_couplings.insert(*zi, atom.spin_coupling_constant);
        for i in atom.valence_orbitals {
            let n: i8 = atom.nshell[i as usize];
            let l: i8 = atom.angular_momenta[i as usize];
            for m in l.neg()..l + 1 {
                vo_vec.push((n - 1, l, m));
                occ.push(atom.orbital_occupation[i as usize] as f64 / (2 * l + 1) as f64);
            }
            val_e += atom.orbital_occupation[i as usize];
        }
        valorbs.insert(*zi, vo_vec);
        valorbs_occupation.insert(*zi, occ);
        ne_val.insert(*zi, val_e);
        let mut energies_zi: HashMap<(i8, i8), f64> = HashMap::new();
        for (n, (l, en)) in free_atom
            .nshell
            .iter()
            .zip(free_atom.angular_momenta.iter().zip(free_atom.energies))
        {
            energies_zi.insert((*n - 1, *l), en);
        }
        orbital_energies.insert(*zi, energies_zi);
    }
    return (
        valorbs,
        valorbs_occupation,
        ne_val,
        orbital_energies,
        hubbard_u,
        spin_couplings,
    );
}
