use crate::initialization::Atom;
use crate::io::Configuration;
use crate::param::Element;
use crate::scc::gamma_approximation::{gaussian_decay, GammaFunction};
use hashbrown::HashMap;
use itertools::Itertools;
use crate::param::skf_handler::SkfHandler;
use ndarray::prelude::*;
use crate::param::slako::{ParamFiles, SlaterKoster};
use crate::param::reppot::RepulsivePotential;


/// The initial atoms are created from the atomic numbers and the coordinates. The unique atoms
/// and all atoms are returned.
pub fn create_atoms(at_num: &[u8], coords: ArrayView2<f64>) -> (Vec<Atom>, Vec<Atom>) {
    // Unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s are
    // constructed
    let (u_atoms, n_to_at): (Vec<Atom>, HashMap<u8, Atom>) = get_unique_atoms(&at_num);

    // Create the list of all atoms.
    let mut atoms: Vec<Atom> = Vec::with_capacity(at_num.len());
    at_num.iter().for_each(|num| atoms.push((*n_to_at.get(num).unwrap()).clone()));

    // Positions are set for each atom.
    coords.outer_iter().zip(atoms.iter_mut())
        .for_each(|(pos, atom)| {
            atom.position_from_slice(position.as_slice().unwrap())
        });

    // (unique atoms, all atoms)
    (u_atoms, atoms)
}


/// Creates the tables with the Slater-Koster splines and the splines of the repulsive potentials.
/// At the moment it is either possible to read in own data files in the RON format or to read in
/// SKF files as used by the DFTB+ program.
pub fn get_parametrization(u_atoms: &[Atom], param_files: ParamFiles) -> (SlaterKoster, RepulsivePotential) {
    let mut slako: SlaterKoster = SlaterKoster::new();
    let mut vrep: RepulsivePotential = RepulsivePotential::new();
    let element_iter = u_atoms.iter().map(|atom| atom.kind);

    match param_files {
        ParamFiles::SKF(path) => {
            let handlers: Vec<SkfHandler> = create_skf_handler(&u_atoms, &path);
            for handler in handlers.iter() {
                // Repulsive potentials are created.
                let repot_table = RepulsivePotentialTable::from(handler);
                // Slater-Koster tables are created for atom pair A-B.
                let slako_table_ab = SlaterKosterTable::from((handler, None, "ab"));
                // and B-A.
                let sh_ba = SkfHandler::new(handler.el_b,handler.el_a,&path);
                let slako_table = SlaterKosterTable::from((&sh_ba, Some(slako_table_ab), "ba"));
                // The parameter tables are inserted into the HashMaps.
                slako.map.insert((handler.el_a, handler.el_b), slako_table);
                vrep.map.insert((handler.el_a, handler.el_b), repot_table);
            }
        }
        ParamFiles::OWN => {
            for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
                slako.add(kind1, kind2);
                vrep.add(kind1, kind2);
            }
        }
    }
    (slako, vrep)
}


/// The SKF handlers for all unique element pair is created.
pub fn create_skf_handler(u_atoms: &[Atom], path_skf: &str) -> Vec<SkfHandler> {
    // SKF handlers for homo-nuclear and hetero-nuclear combinations are created.
    let mut homo_skf: Vec<SkfHandler> = Vec::new();
    let mut hetero_skf: Vec<SkfHandler> = Vec::new();

    let element_iter = u_atoms.iter().map(|atom| atom.kind);
    for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
        if kind1.number() > kind2.number() {
            continue;
        }
        if kind1 == kind2 {
            homo_skf.push(SkfHandler::new(kind1, kind2, &path_skf));
        } else {
            hetero_skf.push(SkfHandler::new(kind1, kind2, &path_skf));
        }
    }
    // combine homo- and heteronuclear skf_handlers. TODO: Is this necessary??
    let mut skf_handlers: Vec<SkfHandler> = Vec::new();
    skf_handlers.append(&mut homonuc_skf);
    skf_handlers.append(&mut heteronuc_skf);

   skf_handlers
}


/// Finds the unique elements in a large list of elements/atoms that are specified by their atomic
/// numbers. For each of these unique elements a [Atom] is created and stored in a Vec<Atom>.
/// Furthermore, a HashMap<u8, Atom> is created that links an atomic number to an [Atom] so that
/// it can be cloned for every atom in the molecule.
pub fn get_unique_atoms(atomic_numbers: &[u8]) -> (Vec<Atom>, HashMap<u8, Atom>) {
    let mut unique_numbers: Vec<u8> = atomic_numbers.to_owned();
    // Sort of atomic numbers
    unique_numbers.sort_unstable();
    // Delete duplicates
    unique_numbers.dedup();
    // Create the unique Atoms
    let unique_atoms: Vec<Atom> = unique_numbers
        .iter()
        .map(|number| Atom::from(*number))
        .collect();
    let mut num_to_atom: HashMap<u8, Atom> = HashMap::with_capacity(unique_numbers.len());
    // insert the atomic numbers and the reference to atoms in the HashMap
    for (num, atom) in unique_numbers
        .into_iter()
        .zip(unique_atoms.clone().into_iter())
    {
        num_to_atom.insert(num, atom);
    }
    return (unique_atoms, num_to_atom);
}



pub fn initialize_gamma_function(unique_atoms: &[Atom], r_lr: f64) -> GammaFunction {
    // initialize the gamma function
    let sigma: HashMap<u8, f64> = gaussian_decay(&unique_atoms);
    let c: HashMap<(u8, u8), f64> = HashMap::new();
    let mut gf = GammaFunction::Gaussian {
        sigma,
        c,
        r_lr: r_lr,
    };
    gf.initialize();
    gf
}

pub fn initialize_unrestricted_elec(charge: i8, n_elec: usize, multiplicity: u8) -> (f64, f64) {
    let mut alpha_electrons: f64 = 0.0;
    let mut beta_electrons: f64 = 0.0;

    if multiplicity == 1 && charge == 0 {
        alpha_electrons = (n_elec / 2) as f64;
        beta_electrons = (n_elec / 2) as f64;
    } else if multiplicity == 3 && charge == 0 {
        alpha_electrons = (n_elec / 2) as f64 + 0.5;
        beta_electrons = (n_elec / 2) as f64 - 0.5;
    } else if multiplicity == 2 {
        if charge == 1 {
            alpha_electrons = (n_elec / 2) as f64;
            beta_electrons = (n_elec / 2) as f64 - 1.0;
        } else if charge == -1 {
            alpha_electrons = (n_elec / 2) as f64 + 1.0;
            beta_electrons = (n_elec / 2) as f64;
        }
    }
    return (alpha_electrons, beta_electrons);
}
