use crate::constants::{
    ATOM_NAMES,
};
use crate::defaults;
use crate::gamma_approximation;
use crate::parameters::*;
use combinations::Combinations;
use ndarray::prelude::*;
use ndarray::*;
use peroxide::special::function::gamma;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Neg;
use ndarray_linalg::*;


pub struct Molecule {
    atomic_numbers: Vec<u8>,
    positions: Array2<f64>,
    charge: i8,
    multiplicity: u8,
    pub n_atoms: usize,
    pub n_orbs: usize,
    pub valorbs: HashMap<u8, Vec<(i8, i8, i8)>>,
    pub hubbard_u: HashMap<u8, f64>,
    valorbs_occupation: HashMap<u8, Vec<i8>>,
    atomtypes: HashMap<u8, String>,
    pub orbital_energies: HashMap<u8, HashMap<(i8, i8), f64>>,
    pub skt: HashMap<(u8, u8), SlaterKosterTable>,
    v_rep: HashMap<(u8, u8), RepulsivePotentialTable>,
    pub proximity_matrix: Array2<bool>,
}

impl Molecule {
    pub(crate) fn new(
        atomic_numbers: Vec<u8>,
        positions: Array2<f64>,
        charge: Option<i8>,
        multiplicity: Option<u8>,
    ) -> Molecule {
        let (atomtypes, unique_numbers): (HashMap<u8, String>, Vec<u8>) =
            get_atomtypes(atomic_numbers.clone());
        let (valorbs, valorbs_occupation, ne_val, orbital_energies, hubbard_u): (
            HashMap<u8, Vec<(i8, i8, i8)>>,
            HashMap<u8, Vec<i8>>,
            HashMap<u8, i8>,
            HashMap<u8, HashMap<(i8, i8), f64>>,
            HashMap<u8, f64>,
        ) = get_electronic_configuration(&atomtypes);

        let (skt, vrep): (
            HashMap<(u8, u8), SlaterKosterTable>,
            HashMap<(u8, u8), RepulsivePotentialTable>,
        ) = get_parameters(unique_numbers);

        let charge: i8 = charge.unwrap_or(defaults::CHARGE);
        let multiplicity: u8 = multiplicity.unwrap_or(defaults::MULTIPLICITY);

        let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
            distance_matrix(positions.view(), None);

        let n_atoms: usize = positions.cols();

        let mut n_orbs: usize = 0;
        for zi in &atomic_numbers {
            n_orbs = n_orbs + &valorbs[zi].len();
        }
        let mol = Molecule {
            atomic_numbers: atomic_numbers,
            positions: positions,
            charge: charge,
            multiplicity: multiplicity,
            n_atoms: n_atoms,
            n_orbs: n_orbs,
            valorbs: valorbs,
            hubbard_u: hubbard_u,
            valorbs_occupation: valorbs_occupation,
            atomtypes: atomtypes,
            orbital_energies: orbital_energies,
            skt: skt,
            v_rep: vrep,
            proximity_matrix: prox_matrix,
        };

        return mol;
    }

    pub fn iter_atomlist(
        &self,
    ) -> std::iter::Zip<
        std::slice::Iter<'_, u8>,
        ndarray::iter::AxisIter<'_, f64, ndarray::Dim<[usize; 1]>>,
    > {
        self.atomic_numbers.iter().zip(self.positions.outer_iter())
    }
}

fn import_pseudo_atom(zi: &u8) -> (PseudoAtom, PseudoAtom) {
    let symbol: &str = ATOM_NAMES[*zi as usize];
    let free_atom: PseudoAtom = get_free_pseudo_atom(symbol);
    let confined_atom: PseudoAtom = get_confined_pseudo_atom(symbol);
    return (confined_atom, free_atom);
}

fn get_gamma_matrix(
    mol: &Molecule,
    r_lr: Option<f64>,
    distances: ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    // initialize gamma matrix
    let sigma: HashMap<u8, f64> = gamma_approximation::gaussian_decay(&mol.hubbard_u);
    let mut c: HashMap<(u8, u8), f64> = HashMap::new();
    let r_lr: f64 = r_lr.unwrap_or(defaults::LONG_RANGE_RADIUS);
    let mut gf = gamma_approximation::GammaFunction::Gaussian { sigma, c, r_lr };
    gf.initialize();
    let (gm, gm_ao): (Array2<f64>, Array2<f64>) = gamma_approximation::gamma_ao_wise(gf, mol, distances);
    return (gm, gm_ao);
}

fn get_parameters(
    numbers: Vec<u8>,
) -> (
    HashMap<(u8, u8), SlaterKosterTable>,
    HashMap<(u8, u8), RepulsivePotentialTable>,
) {
    // find unique atom pairs and initialize Slater-Koster tables
    let atompairs: Vec<Vec<u8>>;
    if numbers.len() > 2 {
        // this only works if we have more than two types of elements
        atompairs = Combinations::new(numbers, 2).collect();
    } else {
        // otherwise we can directly use numbers
        atompairs = vec![numbers];
    }
    let mut skt: HashMap<(u8, u8), SlaterKosterTable> = HashMap::new();
    let mut v_rep: HashMap<(u8, u8), RepulsivePotentialTable> = HashMap::new();
    for pair in atompairs {
        let zi: u8 = pair[0];
        let zj: u8 = pair[1];
        assert!(zi <= zj);
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

fn get_atomtypes(atomic_numbers: Vec<u8>) -> (HashMap<u8, String>, Vec<u8>) {
    // find unique atom types
    let mut unique_numbers: Vec<u8> = atomic_numbers;
    unique_numbers.sort_unstable(); // fast sort of atomic numbers
    unique_numbers.dedup(); // delete duplicates
    let mut atomtypes: HashMap<u8, String> = HashMap::new();
    for zi in &unique_numbers {
        atomtypes.insert(*zi, String::from(ATOM_NAMES[*zi as usize]));
    }
    return (atomtypes, unique_numbers);
}

fn get_electronic_configuration(
    atomtypes: &HashMap<u8, String>,
) -> (
    HashMap<u8, Vec<(i8, i8, i8)>>,
    HashMap<u8, Vec<i8>>,
    HashMap<u8, i8>,
    HashMap<u8, HashMap<(i8, i8), f64>>,
    HashMap<u8, f64>,
) {
    // find quantum numbers of valence orbitals
    let mut valorbs: HashMap<u8, Vec<(i8, i8, i8)>> = HashMap::new();
    let mut valorbs_occupation: HashMap<u8, Vec<i8>> = HashMap::new();
    let mut ne_val: HashMap<u8, i8> = HashMap::new();
    let mut orbital_energies: HashMap<u8, HashMap<(i8, i8), f64>> = HashMap::new();
    let mut hubbard_u: HashMap<u8, f64> = HashMap::new();
    for (zi, symbol) in atomtypes.iter() {
        let (atom, free_atom): (PseudoAtom, PseudoAtom) = import_pseudo_atom(zi);
        let mut occ: Vec<i8> = Vec::new();
        let mut vo_vec: Vec<(i8, i8, i8)> = Vec::new();
        let val_e: i8 = 0;
        hubbard_u.insert(*zi, atom.hubbard_u);
        for i in atom.valence_orbitals {
            let n: i8 = atom.nshell[i as usize];
            let l: i8 = atom.angular_momenta[i as usize];
            for m in l.neg()..l + 1 {
                vo_vec.push((n - 1, l, m));
                occ.push(atom.orbital_occupation[i as usize] / (2 * l + 1));
            }
            let val_e: i8 = val_e + atom.orbital_occupation[i as usize];
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
    );
}

fn distance_matrix(
    coordinates: ArrayView2<f64>,
    cutoff: Option<f64>,
) -> (Array2<f64>, Array3<f64>, Array2<bool>) {
    let cutoff: f64 = cutoff.unwrap_or(defaults::PROXIMITY_CUTOFF);
    let n_atoms: usize = coordinates.cols();
    let mut dist_matrix: Array2<f64> = Array::zeros((n_atoms, n_atoms));
    let mut directions_matrix: Array3<f64> = Array::zeros((n_atoms, n_atoms, 3));
    let mut prox_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    for (i, pos_i) in coordinates.outer_iter().enumerate() {
        for (j, pos_j) in coordinates.slice(s![i.., ..]).outer_iter().enumerate() {
            let r: Array1<f64> = &pos_i - &pos_j;
            let r_ij = r.norm();
            dist_matrix[[i, j]] = r_ij;
            //directions_matrix[[i, j]] = &r/&r_ij;
            if r_ij <= cutoff {
                prox_matrix[[i, j]] = true;
                prox_matrix[[j, i]] = true;
            }
        }
    }
    let dist_matrix = &dist_matrix + &dist_matrix.t();
    //let directions_matrix = directions_matrix - directions_matrix.t();
    return (dist_matrix, directions_matrix, prox_matrix);
}
