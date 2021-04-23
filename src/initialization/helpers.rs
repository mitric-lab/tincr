use hashbrown::HashMap;
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{GammaFunction, gaussian_decay};

/// Finds the unique elements in a large list of elements/atoms that are specified by their atomic
/// numbers. For each of these unique elements a [Atom] is created and stored in a Vec<Atom>.
/// Furthermore, a HashMap<u8, Atom> is created that links an atomic number to an [Atom] so that
/// it can be cloned for every atom in the molecule.
pub fn get_unique_atoms(atomic_numbers: &[u8]) -> (Vec<Atom>, HashMap<u8, Atom>) {
    let mut unique_numbers: Vec<u8> = atomic_numbers.to_owned();
    unique_numbers.sort_unstable(); // fast sort of atomic numbers
    unique_numbers.dedup(); // delete duplicates
    // create the unique Atoms
    let unique_atoms: Vec<Atom> = unique_numbers
        .iter()
        .map(|number| Atom::from(*number))
        .collect();
    let mut num_to_atom: HashMap<u8, Atom> = HashMap::with_capacity(unique_numbers.len());
    // insert the atomic numbers and the reference to atoms in the HashMap
    for (num, atom) in unique_numbers.into_iter().zip(unique_atoms.clone().into_iter()) {
        num_to_atom.insert(num, atom);
    }
    return (unique_atoms, num_to_atom);
}

pub fn initialize_gamma_function(unique_atoms: &[Atom], r_lr: f64) -> GammaFunction {
    // initialize the gamma function
    let sigma: HashMap<u8, f64> = gaussian_decay(&unique_atoms);
    let c: HashMap<(u8, u8), f64> = HashMap::new();
    let mut gf = GammaFunction::Gaussian { sigma, c, r_lr: r_lr};
    gf.initialize();
    gf
}