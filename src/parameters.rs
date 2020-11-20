use peroxide::numerical::spline::CubicSpline;
use ron::de::from_str;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use rusty_fitpack;
use rusty_fitpack::splev_uniform;
use ndarray::{Array, Array1, Array2};
use ndarray::array;

fn get_nan_vec() -> Vec<f64> {
    vec![f64::NAN]
}

fn get_nan_value() -> f64 {
    f64::NAN
}

fn init_none() -> Option<(Vec<f64>, Vec<f64>, usize)> {None}

fn get_inf_value() -> f64 {
    f64::INFINITY
}

fn init_hashmap() -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
    HashMap::new()
}

///
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
    pub s_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
    #[serde(default = "init_hashmap")]
    pub h_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)>
}

impl SlaterKosterTable {
    pub fn spline_overlap(&self) -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)>  {
        let mut splines: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
        for ((l1, l2, i), value) in &self.s {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(*i, rusty_fitpack::splrep(x, y, None, None, None, None, None, None, None, None, None, None));
        }
        return splines;
    }
    pub fn spline_hamiltonian(&self) -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
        let mut splines: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
        for ((l1, l2, i), value) in &self.h {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(*i, rusty_fitpack::splrep(x, y, None, None, None, None, None, None, None, None, None, None));
        }
        return splines;
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
    /// WARNING: the smoothing function can change the nuclear repulsion energy between
    /// atoms that are far apart. Therefore you should check visually the tails of
    /// the repulsive potential and check that the additional energy is not negligible.
    vrep: Vec<f64>,
    z1: u8,
    z2: u8,
    d: Vec<f64>,
    #[serde(default = "init_none")]
    spline_rep: Option<(Vec<f64>, Vec<f64>, usize)>
}

impl RepulsivePotentialTable {
    pub fn spline_rep(&mut self)  {
        let spline: (Vec<f64>, Vec<f64>, usize) = rusty_fitpack::splrep(self.d.clone(), self.vrep.clone(), None,
                                                                            None, None, None, None, None,
                                                                            None, None, None, None);
        self.spline_rep = Some(spline);
    }
    pub fn spline_eval(&self, x: f64) -> f64 {
        match &self.spline_rep {
            Some((t, c, k)) => rusty_fitpack::splev_uniform(t, c, *k, x),
            None => panic!("No spline represantation available"),
        }
    }
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
        "./src/param/slaterkoster/slako_tables/{}_{}.ron",
        element1, element2
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    return slako_table;
}

pub fn get_reppot_table(element1: &str, element2: &str) -> RepulsivePotentialTable {
    let filename: String = format!(
        "./src/param/repulsive_potential/reppot_tables/{}_{}.ron",
        element1, element2
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let mut reppot_table: RepulsivePotentialTable =
        from_str(&data).expect("RON file was not well-formatted");
    reppot_table.spline_rep();
    return reppot_table;
}

#[test]
fn test_load_free_pseudo_atom() {
    let path: &Path = Path::new("./src/param/slaterkoster/free_pseudo_atom/h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {pseudo_atom.z, 1};
}

#[test]
fn test_load_confined_pseudo_atom() {
    let path: &Path = Path::new("./src/param/slaterkoster/confined_pseudo_atom/h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {pseudo_atom.z, 1};
}

#[test]
fn test_load_slako_tables() {
    let path: &Path = Path::new("./src/param/slaterkoster/slako_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {slako_table.z1, 1};
    assert_eq! {slako_table.z2, 1};
}

#[test]
fn test_load_repulsive_potential_tables() {
    let path: &Path = Path::new("./src/param/repulsive_potential/reppot_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let reppot_table: RepulsivePotentialTable =
        from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {reppot_table.z1, 1};
    assert_eq! {reppot_table.z2, 1};
}


/// import numpy as np
// from DFTB2 import DFTB2
// import XYZ
// import GammaApproximation
// atomlist = XYZ.read_xyz("h2o.xyz")[0]
// dftb = DFTB2(atomlist)
// dftb.setGeometry(atomlist, charge=0)
//
// dftb.getEnergy()
//
// for key, value in dftb.SKT.items():
//     print(key, value)
// d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
// np.set_printoptions(16)
// print(dftb.SKT[(1,1)].S_spl(0, d))
#[test]
fn test_spline_overlap_integrals() {
    let path: &Path = Path::new("./src/param/slaterkoster/slako_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    let spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = slako_table.spline_overlap();
    let mut y_values: Vec<f64> =  Vec::new();
    let x_values: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4];
    for x in x_values {
        y_values.push(splev_uniform(&spline[&0].0, &spline[&0].1, spline[&0].2, x));
    }
    let y_values: Array1<f64> = Array::from_shape_vec((14), y_values).unwrap();
    let y_values_ref: Array1<f64> = array![
        0.9953396476468342,
        0.9812384772492724,
        0.9583521361528490,
        0.9274743570042232,
        0.8895949507497998,
        0.8458257726181956,
        0.7973774727029854,
        0.7454487849069387,
        0.6912337281934855,
        0.6358463027455588,
        0.5803233164667398,
        0.5255800129748242,
        0.4724037942538298,
        0.4214524357395346];
    assert!(y_values.all_close(&y_values_ref, 1e-08));
}

#[test]
fn test_spline_h0() {
    let path: &Path = Path::new("./src/param/slaterkoster/slako_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    let spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)>= slako_table.spline_hamiltonian();
    let mut y_values: Vec<f64> =  Vec::new();
    let x_values: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4];
    for x in x_values {
        y_values.push(splev_uniform(&spline[&0].0, &spline[&0].1, spline[&0].2, x));
    }
    let y_values: Array1<f64> = Array::from_shape_vec((14), y_values).unwrap();
    let y_values_ref: Array1<f64> = array![
         -0.7020075123394368,
         -0.6827454396001111,
         -0.6559933747633552,
         -0.6249508412681278,
         -0.5919398382976603,
         -0.5585510127756821,
         -0.5258067834653534,
         -0.4942857974427148,
         -0.4642429835387089,
         -0.4357098171167648,
         -0.4085694762270275,
         -0.3826380560498423,
         -0.3577041260543390,
         -0.3335812815557643];
    assert!(y_values.all_close(&y_values_ref, 1e-08));
}