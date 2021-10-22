use core::types::{Atom, AtomVec, AtomSlice};
use tincr::io::{Configuration, frame_to_coordinates};
use tincr::param::elements::Element;
use tincr::scc::gamma_approximation::{gaussian_decay, GammaFunction, gamma_atomwise};
use hashbrown::HashMap;
use itertools::Itertools;
use tincr::param::skf_handler::SkfHandler;
use ndarray::prelude::*;
use tincr::param::slako::{ParamFiles, SlaterKoster, SlaterKosterTable};
use tincr::param::reppot::{RepulsivePotential, RepulsivePotentialTable};
use tincr::src::{Parametrization};
use chemfiles::Frame;


pub struct Parameters {
    pub slako: SlaterKoster,
    pub vrep: RepulsivePotential,
    pub gf: GammaFunction,
    pub gf_lrc: GammaFunction,
}

impl Parameters {

    /// Initialization.
    /// 1. Parameter files are read.
    /// 2. Gamma matrices are computed for the initial geometry.
    /// 3. H0 and S matrices are computed for the initial geometry.
    pub fn new(config: &Configuration, u_atoms: AtomSlice) -> Self {
        // Check if SKF files should be used for the parametrization or own files.
        let param_files: ParamFiles = if config.slater_koster.use_skf_files {
            ParamFiles::SKF(&config.slater_koster.path_to_skf)
        } else {
            ParamFiles::OWN
        };

        // The Slater-Koster parameters and Repulsive potentials are constructed from the param file.
        let (slako, vrep) = get_parametrization(u_atoms, param_files);

        // The Gamma matrices are computed. The first is the unscreened Gamma matrix that is used
        // for the Coulomb interaction, while the second one is the screened Gamma matrix which is
        // used for the Exchange part (but only if the long-range correction is requested).
        let gf = initialize_gamma_function(u_atoms, 0.0);
        let gf_lrc = initialize_gamma_function(u_atoms, config.lc.long_range_radius);

        // The parametrization is combined into an own type and the matrices resulting from it too.
        Self {
            slako,
            vrep,
            gf,
            gf_lrc
        }
    }

    pub fn compute_matrices<'a>(&self, config: &Configuration, atoms: AtomSlice<'a>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Option<Array2<f64>>) {
        let gamma: Array2<f64> = gamma_atomwise(&self.gf, atoms);
        let gamma_lrc: Option<Array2<f64>> = if config.lc.long_range_correction {
            Some(gamma_atomwise(&self.gf_lrc, atoms))
        } else {
            None
        };

        let n_orbs: usize = atoms.n_orbs().iter().sum();

        // The overlap matrix, S, and the one-electron integral Fock matrix is computed from the
        // precalculated splines that are tabulated in the parameter files.
        let (s, h0): (Array2<f64>, Array2<f64>) = self.slako.h0_and_s(n_orbs, atoms);

        (s, h0, gamma, gamma_lrc)
    }
}




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





pub fn initialize_gamma_function(unique_atoms: AtomSlice, r_lr: f64) -> GammaFunction {
    // initialize the gamma function
    let sigma: HashMap<u8, f64> = gaussian_decay(unique_atoms);
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
