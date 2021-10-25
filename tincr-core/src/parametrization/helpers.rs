use core::types::{Atom, AtomVec, AtomSlice};
use hashbrown::HashMap;
use itertools::Itertools;
use ndarray::prelude::*;
use chemfiles::Frame;



/// Creates the tables with the Slater-Koster splines and the splines of the repulsive potentials.
/// At the moment it is either possible to read in own data files in the RON format or to read in
/// SKF files as used by the DFTB+ program.
pub fn get_parametrization(u_atoms: AtomSlice, param_files: ParamFiles) -> (SlaterKoster, RepulsivePotential) {
    let mut slako: SlaterKoster = SlaterKoster::new();
    let mut vrep: RepulsivePotential = RepulsivePotential::new();
    let element_iter = u_atoms.iter().map(|atom| atom.kind);

    match param_files {
        ParamFiles::SKF(path) => {
            let handlers: Vec<SkfHandler> = create_skf_handler(u_atoms, &path);
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
pub fn create_skf_handler(u_atoms: AtomSlice, path_skf: &str) -> Vec<SkfHandler> {
    // SKF handlers for homo-nuclear and hetero-nuclear combinations are created.
    let mut skf_handler: Vec<SkfHandler> = Vec::new();

    let element_iter = u_atoms.kind.iter();
    for (kind1, kind2) in element_iter.clone().cartesian_product(element_iter) {
        if kind1.number() > kind2.number() {
            continue;
        }
        if kind1 == kind2 {
            skf_handler.push(SkfHandler::new(*kind1, *kind2, &path_skf));
        } else {
            skf_handler.push(SkfHandler::new(*kind1, *kind2, &path_skf));
        }
    }
   skf_handler
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
