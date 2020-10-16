use peroxide::numerical::spline::CubicSpline;
use ron::de::from_str;
use serde::{Deserialize, Serialize};
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

fn init_hashmap() -> HashMap<u8, CubicSpline> {
    HashMap::new()
}

///
#[derive(Serialize, Deserialize)]
pub struct PseudoAtom {
    z: u8,
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
    pub angular_momenta: Vec<u8>,
    pub valence_orbitals: Vec<u8>,
    pub nshell: Vec<u8>,
    pub orbital_occupation: Vec<u8>;
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
pub struct SlaterKosterTable {
    dipole: HashMap<(u8, u8, u8), Vec<f64>>,
    h: HashMap<(u8, u8, u8), Vec<f64>>,
    s: HashMap<(u8, u8, u8), Vec<f64>>,
    z1: u8,
    z2: u8,
    d: Vec<f64>,
    index_to_symbol: HashMap<u8, String>,
    #[serde(default = "init_hashmap")]
    s_spline: HashMap<u8, CubicSpline>,
    #[serde(default = "init_hashmap")]
    h_spline: HashMap<u8, CubicSpline>,
}

impl SlaterKosterTable {
    fn spline_overlap(&self) -> bool {
        let mut splines: HashMap<u8, CubicSpline> = HashMap::new();
        for ((l1, l2, i), value) in self.s {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(i, CubicSpline::from_nodes(x, y));
        }
        return true;
    }
    fn spline_hamiltonian(&self) -> bool {
        let mut splines: HashMap<u8, CubicSpline> = HashMap::new();
        for ((l1, l2, i), value) in self.h {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(i, CubicSpline::from_nodes(x, y));
        }
        return true;
    }
}

#[derive(Serialize, Deserialize)]
pub struct RepulsivePotentialTable {
    /// RepulsivePotentialTable should be a struct with the following members
    ///
    /// z1,z2: atomic numbers of atom pair
    /// d: Vec, distance between atoms
    /// vrep: repulsive potential on the grid d
    ///
    /// smooth_decay controls whether vrep and its derivatives are set abruptly to
    /// 0 after the cutoff radius or whether a smoothing function is added.
    /// WARNING: the smooting function can change the nuclear repulsion energy between
    /// atoms that are far apart. Therefore you should check visually the tails of
    /// the repulsive potential and check that the additional energy is not negligible.
    vrep: Vec<f64>,
    z1: u8,
    z2: u8,
    d: Vec<f64>,
}

pub fn get_free_pseudo_atom(element: &str) -> PseudoAtom {
    let filename: String = format!("./src/param/slaterkoster/free_pseudo_atom/{}.ron", element);
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    return pseudo_atom;
}

pub fn get_confined_pseudo_atom(element: &str) -> PseudoAtom {
    let filename: String = format!(
        "./src/param/slaterkoster/confined_pseudo_atom/{}.ron",
        element
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    return pseudo_atom;
}

pub fn get_slako_table(element1: &str, element2: &str) -> SlaterKosterTable {
    let filename: String = format!(
        "./src/param/slaterkoster/slako-tables/{}_{}.ron",
        element1, element2
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    return slako_table;
}

pub fn get_reppot_table(element1: &str, element2: &str) -> RepulsivePotentialTable {
    let filename: String = format!(
        "./src/param/repulsive_potential/reppot-tables/{}_{}.ron",
        element1, element2
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let reppot_table: RepulsivePotentialTable =
        from_str(&data).expect("RON file was not well-formatted");
    return reppot_table;
}

#[test]
fn test_free_pseudo_atom() {
    let path: &Path = Path::new("./src/param/slaterkoster/free_pseudo_atom/h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {pseudo_atom.z, 1};
}

#[test]
fn test_confined_pseudo_atom() {
    let path: &Path = Path::new("./src/param/slaterkoster/confined_pseudo_atom/h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {pseudo_atom.z, 1};
}

#[test]
fn test_slako_tables() {
    let path: &Path = Path::new("./src/param/slaterkoster/slako_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {slako_table.z1, 1};
    assert_eq! {slako_table.z2, 1};
}

#[test]
fn test_repulsive_potential_tables() {
    let path: &Path = Path::new("./src/param/repulsive_potential/reppot_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let reppot_table: RepulsivePotentialTable =
        from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {reppot_table.z1, 1};
    assert_eq! {reppot_table.z2, 1};
}
