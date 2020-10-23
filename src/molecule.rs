use crate::constants::ATOM_NAMES;
use crate::gamma_approximation;
use crate::parameters::*;
use combinations::Combinations;
use ndarray::prelude::*;
use peroxide::special::function::gamma;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Neg;

pub(crate) struct Molecule {
    atomic_numbers: Vec<u8>,
    positions: Array2<f64>,
    charge: i8,
    multiplicity: u8,
    pub(crate) n_atoms: usize,
    pub(crate) n_orbs: usize,
    pub(crate) valorbs: HashMap<u8, Vec<(u8, u8, u8)>>,
    hubbard_u: HashMap<u8, f64>,
    valorbs_occupation: HashMap<u8, Vec<u8>>,
    atomtypes: HashMap<u8, String>,
    orbital_energies: HashMap<u8, HashMap<(u8, u8), f64>>,
    skt: HashMap<(u8, u8), SlaterKosterTable>,
    v_rep: HashMap<(u8, u8), RepulsivePotentialTable>,
}

impl Molecule {
    fn initialize(&self) {

        // initialize gamma matrix
        //let sigma: HashMap<u8, f64> = gamma_approximation::gaussian_decay(&hubbard_u);
        //let mut c: HashMap<(u8, u8), f64> = HashMap::new();
        //let r_lr: f64 = 3.03;
        //let mut gf = gamma_approximation::GammaFunction::Gaussian{sigma, c, r_lr};
        //gf.initialize();
        //let gm: Array2<f64> = gamma_approximation::gamma_ao_wise(gf, );
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


fn get_parameters(numbers: Vec<u8>) -> () {
    // find unique atom pairs and initialize Slater-Koster tables
    let atompairs: Vec<Vec<u8>> = Combinations::new(numbers, 2).collect();
    let mut skt: HashMap<(u8, u8), SlaterKosterTable> = HashMap::new();
    let mut v_rep: HashMap<(u8, u8), RepulsivePotentialTable> = HashMap::new();
    for pair in atompairs {
        let zi: u8 = pair[0];
        let zj: u8 = pair[1];
        assert!(zi <= zj);
        // load precalculated slako table
        let slako_module: SlaterKosterTable =
            get_slako_table(ATOM_NAMES[zi as usize], ATOM_NAMES[zj as usize]);
        // load repulsive potential table
        let reppot_module: RepulsivePotentialTable =
            get_reppot_table(ATOM_NAMES[zi as usize], ATOM_NAMES[zj as usize]);
        skt.insert((zi, zj), slako_module);
        v_rep.insert((zi, zj), reppot_module);
    }
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
    HashMap<u8, (i8, i8, i8)>,
    HashMap<u8, Vec<i8>>,
    HashMap<u8, i8>,
    HashMap<u8, HashMap<(i8, i8), f64>>,
    HashMap<u8, f64>,
) {
    // find quantum numbers of valence orbitals
    let mut valorbs: HashMap<u8, (i8, i8, i8)> = HashMap::new();
    let mut valorbs_occupation: HashMap<u8, Vec<i8>> = HashMap::new();
    let mut ne_val: HashMap<u8, i8> = HashMap::new();
    let mut orbital_energies: HashMap<u8, HashMap<(i8, i8), f64>> = HashMap::new();
    let mut hubbard_u: HashMap<u8, f64> = HashMap::new();
    for (zi, symbol) in atomtypes.iter() {
        let (atom, free_atom): (PseudoAtom, PseudoAtom) = import_pseudo_atom(zi);
        let mut occ: Vec<i8> = Vec::new();
        let val_e: i8 = 0;
        hubbard_u.insert(*zi, atom.hubbard_u[*zi]);
        for i in atom.valence_orbitals {
            let n: i8 = atom.nshell[i as usize];
            let l: i8 = atom.angular_momenta[i as usize];
            for m in l.neg()..l + 1 {
                valorbs.insert(*zi, (n - 1, l, m));
                occ.push(atom.orbital_occupation[i as usize] / (2 * l + 1));
            }
            let val_e: i8 = val_e + atom.orbital_occupation[i as usize];
        }
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
