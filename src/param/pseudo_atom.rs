use crate::param::skf_handler::{SkfHandler, process_slako_line};
use ndarray::prelude::*;
use hashbrown::HashMap;
use crate::utils::get_path_prefix;
use std::path::Path;
use std::fs;
use ron::from_str;
use serde::{Deserialize, Serialize};

fn get_nan_vec() -> Vec<f64> {
    vec![f64::NAN]
}

fn get_nan_value() -> f64 {
    f64::NAN
}

fn init_none() -> Option<(Vec<f64>, Vec<f64>, usize)> {
    None
}

fn get_inf_value() -> f64 {
    f64::INFINITY
}

fn init_hashmap() -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
    HashMap::new()
}

/// A type that contains the atom-wise parameters for the DFTB calculation. The same `PseudoAtom`
/// type is used for the free and the confined atoms. The data will be serialized from the Ron files.
#[derive(Serialize, Deserialize)]
pub struct PseudoAtom {
    z: u8,
    pub hubbard_u: f64,
    n_elec: u8,
    #[serde(default = "get_inf_value")]
    r0: f64,
    r: Vec<f64>,
    radial_density: Vec<f64>,
    pub occupation: Vec<(u8, u8, u8)>,
    effective_potential: Vec<f64>,
    orbital_names: Vec<String>,
    pub energies: Vec<f64>,
    radial_wavefunctions: Vec<Vec<f64>>,
    pub angular_momenta: Vec<i8>,
    pub valence_orbitals: Vec<u8>,
    pub nshell: Vec<i8>,
    pub orbital_occupation: Vec<i8>,
    #[serde(default = "get_nan_value")]
    pub spin_coupling_constant: f64,
    #[serde(default = "get_nan_value")]
    energy_1s: f64,
    #[serde(default = "get_nan_value")]
    energy_2s: f64,
    #[serde(default = "get_nan_value")]
    energy_3s: f64,
    #[serde(default = "get_nan_value")]
    energy_4s: f64,
    #[serde(default = "get_nan_value")]
    energy_2p: f64,
    #[serde(default = "get_nan_value")]
    energy_3p: f64,
    #[serde(default = "get_nan_value")]
    energy_4p: f64,
    #[serde(default = "get_nan_value")]
    energy_3d: f64,
    #[serde(default = "get_nan_vec")]
    orbital_1s: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_2s: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_3s: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_4s: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_2p: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_3p: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_4p: Vec<f64>,
    #[serde(default = "get_nan_vec")]
    orbital_3d: Vec<f64>,
}

impl PseudoAtom {
    pub fn free_atom(element: &str) -> PseudoAtom {
        let path_prefix: String = get_path_prefix();
        let filename: String = format!(
            "{}/src/param/slaterkoster/free_pseudo_atom/{}.ron",
            path_prefix,
            element.to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");
        from_str(&data).expect("RON file was not well-formatted")
    }

    pub fn confined_atom(element: &str) -> PseudoAtom {
        let path_prefix: String = get_path_prefix();
        let filename: String = format!(
            "{}/src/param/slaterkoster/confined_pseudo_atom/{}.ron",
            path_prefix,
            element.to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");
        from_str(&data).expect("RON file was not well-formatted")
    }
}

pub struct PseudoAtomMio {
    z: u8,
    pub hubbard_u: f64,
    n_elec: u8,
    pub energies: Vec<f64>,
    pub angular_momenta: Vec<i8>,
    pub valence_orbitals: Vec<u8>,
    pub nshell: Vec<i8>,
    pub orbital_occupation: Vec<i8>,
}

impl From<&SkfHandler> for PseudoAtomMio {
    fn from(skf_handler: &SkfHandler) -> Self {
        // Extract the individual lines of the SKF file.
        let lines: Vec<&str> = skf_handler.data_string.split("\n").collect();

        // Read Ed Ep Es SPE Ud Up Us fd fp fs from the second line of the SKF file.
        // Ed Ep Es: one-site energies
        // Ud Up Us: Hubbard Us of the different angular momenta
        // fd fp fs: occupation numbers of the orbitals
        let second_line: Vec<f64> = process_slako_line(lines[1]);
        let energies: Array1<f64> = array![second_line[2], second_line[1], second_line[0]];
        let occupations_numbers: Array1<i8> = array![
            second_line[9] as i8,
            second_line[8] as i8,
            second_line[7] as i8
        ];
        let hubbard_u: Array1<f64> = array![second_line[6], second_line[5], second_line[4]];

        let electron_count: u8 = skf_handler.el_a.number();
        let mut valence_orbitals: Vec<u8> = Vec::new();
        let mut nshell: Vec<i8> = Vec::new();
        // Set nshell depending on the electron count of the atom
        if electron_count < 3 {
            nshell.push(1);
        } else if electron_count < 11 {
            nshell.push(2);
            nshell.push(2);
        } else if electron_count < 19 {
            nshell.push(3);
            nshell.push(3);
        }

        // Fill the angular momenta
        let mut angular_momenta: Vec<i8> = Vec::new();
        for (it, occ) in occupations_numbers.iter().enumerate() {
            if occ > &0 {
                valence_orbitals.push(it as u8);
                if it == 0 {
                    angular_momenta.push(0);
                } else if it == 1 {
                    angular_momenta.push(1);
                } else if it == 2 {
                    angular_momenta.push(2);
                }
            }
        }

        // Create a PseudoAtom
        let pseudo_atom: PseudoAtomMio = PseudoAtomMio {
            z: skf_handler.el_a.number(),
            hubbard_u: hubbard_u[0],
            energies: energies.to_vec(),
            angular_momenta: angular_momenta,
            valence_orbitals: valence_orbitals,
            nshell: nshell,
            orbital_occupation: occupations_numbers.to_vec(),
            n_elec: skf_handler.el_a.number(),
        };
        pseudo_atom
    }
}
