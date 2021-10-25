pub mod gamma_approximation;
pub mod pseudo_atom;
pub mod reppot;
mod skf_handler;
pub mod slako;
pub mod slako_transformations;

mod reppot_spline;
mod sk_matrix_elements;
mod slako_spline;

// pub struct Parameters {
//     pub slako: SlaterKoster,
//     pub vrep: RepulsivePotential,
//     pub gf: GammaFunction,
//     pub gf_lrc: GammaFunction,
// }
//
// impl Parameters {
//
//     /// Initialization.
//     /// 1. Parameter files are read.
//     /// 2. Gamma matrices are computed for the initial geometry.
//     /// 3. H0 and S matrices are computed for the initial geometry.
//     pub fn new(config: &Configuration, u_atoms: AtomSlice) -> Self {
//         // Check if SKF files should be used for the parametrization or own files.
//         let param_files: ParamFiles = if config.slater_koster.use_skf_files {
//             ParamFiles::SKF(&config.slater_koster.path_to_skf)
//         } else {
//             ParamFiles::OWN
//         };
//
//         // The Slater-Koster parameters and Repulsive potentials are constructed from the param file.
//         let (slako, vrep) = get_parametrization(u_atoms, param_files);
//
//         // The Gamma matrices are computed. The first is the unscreened Gamma matrix that is used
//         // for the Coulomb interaction, while the second one is the screened Gamma matrix which is
//         // used for the Exchange part (but only if the long-range correction is requested).
//         let gf = initialize_gamma_function(u_atoms, 0.0);
//         let gf_lrc = initialize_gamma_function(u_atoms, config.lc.long_range_radius);
//
//         // The parametrization is combined into an own type and the matrices resulting from it too.
//         Self {
//             slako,
//             vrep,
//             gf,
//             gf_lrc
//         }
//     }
//
//     pub fn compute_matrices<'a>(&self, config: &Configuration, atoms: AtomSlice<'a>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Option<Array2<f64>>) {
//         let gamma: Array2<f64> = gamma_atomwise(&self.gf, atoms);
//         let gamma_lrc: Option<Array2<f64>> = if config.lc.long_range_correction {
//             Some(gamma_atomwise(&self.gf_lrc, atoms))
//         } else {
//             None
//         };
//
//         let n_orbs: usize = atoms.n_orbs().iter().sum();
//
//         // The overlap matrix, S, and the one-electron integral Fock matrix is computed from the
//         // precalculated splines that are tabulated in the parameter files.
//         let (s, h0): (Array2<f64>, Array2<f64>) = self.slako.h0_and_s(n_orbs, atoms);
//
//         (s, h0, gamma, gamma_lrc)
//     }
// }
