use serde::{Deserialize, Serialize};
use serde_json::from_str;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn get_nan_vec() -> Vec<f64> {
    vec![f64::NAN]
}

fn get_nan_value() -> f64 {
    f64::NAN
}

fn get_inf_value() -> f64 {
    f64::INFINITY
}

#[derive(Serialize, Deserialize)]
struct PseudoAtom {
    z: u8,
    n_elec: u8,
    #[serde(default = "get_inf_value")]
    r0: f64,
    r: Vec<f64>,
    radial_density: Vec<f64>,
    occupation: Vec<(u8, u8, u8)>,
    effective_potential: Vec<f64>,
    orbital_names: Vec<String>,
    energies: Vec<f64>,
    radial_wavefunctions: Vec<Vec<f64>>,
    angular_momenta: Vec<u8>,
    valence_orbitals: Vec<u8>,
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

#[derive(Serialize, Deserialize)]
struct SlaterKosterTable {
    dipole: HashMap<(u8, u8, u8), Vec<f64>>,
    h: HashMap<(u8, u8, u8), Vec<f64>>,
    s: HashMap<(u8, u8, u8), Vec<f64>>,
    z1: u8,
    z2: u8,
    d: Vec<f64>,
    index_to_symbol: HashMap<u8, String>,
}

#[derive(Serialize, Deserialize)]
struct RepulsivePotentialTable {
    vrep: Vec<f64>,
    z1: u8,
    z2: u8,
    d: Vec<f64>,
}

#[test]
fn test_free_pseudo_atom() {
    let path: &Path = Path::new("./src/param/slaterkoster/free_pseudo_atom/h.json");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = serde_json::from_str(&data).expect("JSON was not well-formatted");
    assert_eq! {pseudo_atom.z, 1};
}

#[test]
fn test_confined_pseudo_atom() {
    let path: &Path = Path::new("./src/param/slaterkoster/confined_pseudo_atom/h.json");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = serde_json::from_str(&data).expect("JSON was not well-formatted");
    assert_eq! {pseudo_atom.z, 1};
}

#[test]
fn test_slako_tables() {
    let path: &Path = Path::new("./src/param/slaterkoster/slako_tables/h_h.json");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table : SlaterKosterTable = serde_json::from_str(&data).expect("JSON was not well-formatted");
    assert_eq! {slako_table.z, 1};
}