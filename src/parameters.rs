use crate::defaults;
use approx::AbsDiffEq;
use ndarray::array;
use ndarray::s;
use ndarray::{Array, Array1, Array2};
use ron::de::from_str;
use rusty_fitpack;
use rusty_fitpack::splev_uniform;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use crate::constants::{TAUSYMBOLS_AB, TAUSYMBOLS_BA, SYMBOL_2_TAU, ATOM_NAMES, ELEMENT_TO_Z};

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

#[derive(Serialize, Deserialize, Clone)]
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
    pub h_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
    #[serde(default = "get_nan_value")]
    pub dmax: f64,
}

impl SlaterKosterTable {
    pub fn new(
        dipole: HashMap<(u8, u8, u8), Vec<f64>>,
        h: HashMap<(u8, u8, u8), Vec<f64>>,
        s: HashMap<(u8, u8, u8), Vec<f64>>,
        z1: u8,
        z2: u8,
        d: Vec<f64>,
    )->SlaterKosterTable{
        let dmax:f64 = d[d.len()-1];
        let s_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = init_hashmap();
        let h_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = init_hashmap();
        let mut index_to_symbol:HashMap<u8, String> = HashMap::new();
        index_to_symbol.insert(0,String::from("ss_sigma"));
        index_to_symbol.insert(2 ,String::from("ss_sigma"));
        index_to_symbol.insert(3 ,String::from("sp_sigma"));
        index_to_symbol.insert(4 ,String::from("sd_sigma"));
        index_to_symbol.insert(5 ,String::from("ps_sigma"));
        index_to_symbol.insert(6 ,String::from("pp_pi"));
        index_to_symbol.insert(7 ,String::from("pp_sigma"));
        index_to_symbol.insert(8 ,String::from("pd_pi"));
        index_to_symbol.insert(9 ,String::from("pd_sigma"));
        index_to_symbol.insert(10,String::from("ds_sigma"));
        index_to_symbol.insert(11,String::from("dp_pi"));
        index_to_symbol.insert(12,String::from("dp_sigma"));
        index_to_symbol.insert(13,String::from("dd_delta"));
        index_to_symbol.insert(14,String::from("dd_pi"));

        SlaterKosterTable{
            dipole:dipole,
            h:h,
            s:s,
            z1:z1,
            z2:z2,
            d:d,
            index_to_symbol,
            h_spline:h_spline,
            s_spline:s_spline,
            dmax:dmax,
        }
    }
    pub fn spline_overlap(&self) -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
        let mut splines: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
        for ((l1, l2, i), value) in &self.s {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(
                *i,
                rusty_fitpack::splrep(
                    x, y, None, None, None, None, None, None, None, None, None, None,
                ),
            );
        }
        return splines;
    }
    pub fn spline_hamiltonian(&self) -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
        let mut splines: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
        for ((l1, l2, i), value) in &self.h {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(
                *i,
                rusty_fitpack::splrep(
                    x, y, None, None, None, None, None, None, None, None, None, None,
                ),
            );
        }
        return splines;
    }
}

#[derive(Serialize, Deserialize, Clone)]
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
    spline_rep: Option<(Vec<f64>, Vec<f64>, usize)>,
    #[serde(default = "get_nan_value")]
    dmax: f64
}

impl RepulsivePotentialTable {
    pub fn spline_rep(&mut self) {
        let spline: (Vec<f64>, Vec<f64>, usize) = rusty_fitpack::splrep(
            self.d.clone(),
            self.vrep.clone(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        self.spline_rep = Some(spline);
    }
    pub fn spline_eval(&self, x: f64) -> f64 {
        match &self.spline_rep {
            Some((t, c, k)) => if x <= self.dmax {rusty_fitpack::splev_uniform(t, c, *k, x)} else {0.0},
            None => panic!("No spline represantation available"),
        }
    }
    pub fn spline_deriv(&self, x: f64) -> f64 {
        match &self.spline_rep {
            Some((t, c, k)) => if x <= self.dmax{rusty_fitpack::splder_uniform(t, c, *k, x, 1)} else {0.0},
            None => panic!("No spline represantation available"),
        }
    }
}

fn get_path_prefix() -> String {
    let key: &str = defaults::SOURCE_DIR_VARIABLE;
    match env::var(key) {
        Ok(val) => val,
        Err(e) => panic!("The environment variable {} was not set", key),
    }
}

pub fn get_free_pseudo_atom(element: &str) -> PseudoAtom {
    let path_prefix: String = get_path_prefix();
    let filename: String = format!(
        "{}/src/param/slaterkoster/free_pseudo_atom/{}.ron",
        path_prefix, element
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    return pseudo_atom;
}

pub fn get_confined_pseudo_atom(element: &str) -> PseudoAtom {
    let path_prefix: String = get_path_prefix();
    let filename: String = format!(
        "{}/src/param/slaterkoster/confined_pseudo_atom/{}.ron",
        path_prefix, element
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    return pseudo_atom;
}

pub fn get_slako_table(element1: &str, element2: &str) -> SlaterKosterTable {
    let path_prefix: String = get_path_prefix();
    let filename: String = format!(
        "{}/src/param/slaterkoster/slako_tables/{}_{}.ron",
        path_prefix, element1, element2
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let mut slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    slako_table.dmax = slako_table.d[slako_table.d.len()-1];
    return slako_table;
}

pub fn get_slako_table_mio(element1: &str, element2: &str) -> SlaterKosterTable {
    let path_prefix: String = String::from("/home/einseler/software/mio-0-1");
    let element_1:String = some_kind_of_uppercase_first_letter(element1);
    let element_2:String = some_kind_of_uppercase_first_letter(element2);
    let filename: String = format!(
        "{}/{}-{}.skf",
        path_prefix, element_1, element_2
    );
    println!("filename {}",filename);
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let mut slako_table_final: SlaterKosterTable = read_mio_slako_data(&data,element1,element2,"ab",None);

    if element1 != element2{
        let element_1:String = some_kind_of_uppercase_first_letter(element2);
        let element_2:String = some_kind_of_uppercase_first_letter(element1);
        let filename: String = format!(
            "{}/{}-{}.skf",
            path_prefix, element_1, element_2
        );
        println!("filename {}",filename);
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");
        slako_table_final = read_mio_slako_data(&data,element1,element2,"ba",Some(slako_table_final.clone()));
    }
    return slako_table_final;
}

pub fn read_mio_slako_data(data:&String,element1: &str, element2: &str,order:&str,slako:Option<SlaterKosterTable>)->SlaterKosterTable{
    let mut strings: Vec<&str> = data.split("\n").collect();
    // first line
    let first_line:Vec<f64> = process_slako_line(strings[0]);
    let grid_dist:f64 = first_line[0];
    let npoints:usize = first_line[1] as usize;
    // println!("grid dist {} and npoints {}",grid_dist,npoints);

    // create grid
    let d_arr:Array1<f64> = Array1::linspace(0.0,grid_dist*((npoints-1) as f64),npoints);

    // remove first line
    strings.remove(0);
    if element1 == element2{
        // remove second line
        strings.remove(0);
    }
    // remove second/third line
    strings.remove(0);

    let next_line:Vec<f64> = process_slako_line(strings[0]);
    let length:usize = next_line.len()/2;
    assert!(length == 10);

    let mut tausymbols:Vec<&str> = Vec::new();
    if order == "ab"{
        tausymbols = TAUSYMBOLS_AB.iter().cloned().collect();
    }
    else if order == "ba"{
        tausymbols = TAUSYMBOLS_BA.iter().cloned().collect();
    }
    let tausymbols:Array1<&str> = Array::from(tausymbols);
    let length_tau:usize = tausymbols.len();

    let mut h: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();
    let mut s: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();
    let mut dipole: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();
    if slako.is_some(){
        let slako_table:SlaterKosterTable = slako.unwrap();
        h = slako_table.h.clone();
        s = slako_table.s.clone();
        dipole = slako_table.dipole.clone();
    }

    let mut vec_h_arrays:Vec<Array1<f64>> = Vec::new();
    let mut vec_s_arrays:Vec<Array1<f64>> = Vec::new();
    for it in (0..10){
        vec_s_arrays.push(Array1::zeros(npoints));
        vec_h_arrays.push(Array1::zeros(npoints));
    }
    let temp_vec:Vec<f64> = Array1::zeros(npoints).to_vec();

    for it in (0..npoints){
        let next_line:Vec<f64> = process_slako_line(strings[0]);
        for (pos, tausym) in tausymbols.slice(s![-10..]).iter().enumerate(){
            let symbol:(u8,i32,u8,i32) = SYMBOL_2_TAU[*tausym];
            let l1:u8 = symbol.0;
            let l2:u8 = symbol.2;

            let mut orbital_parity:f64 = 0.0;
            if order == "ba"{
                orbital_parity = -1.0_f64.powi((l1+l2) as i32);
            }
            else{
                orbital_parity = 1.0;
            }
            // let index:u8 = get_tau_2_index(symbol);
            // h[&(l1,l2,index)][it] = orbital_parity * next_line[pos];
            // s[&(l1,l2,index)][it] = orbital_parity * next_line[length_tau+pos];
            vec_h_arrays[pos][it] = orbital_parity * next_line[pos];
            vec_s_arrays[pos][it] = orbital_parity * next_line[length_tau+pos];
        }
    }
    for (pos,tausymbol) in tausymbols.slice(s![-10..]).iter().enumerate(){
        let symbol:(u8,i32,u8,i32) = SYMBOL_2_TAU[*tausymbol];
        // println!("Symbol {:?}",symbol);
        let index:u8 = get_tau_2_index(symbol);
        h.insert((symbol.0,symbol.2,index),vec_h_arrays[pos].to_vec());
        s.insert((symbol.0,symbol.2,index),vec_s_arrays[pos].to_vec());
        dipole.insert((symbol.0,symbol.2,index),temp_vec.clone());
    }

    let z1:u8 = ELEMENT_TO_Z[element1];
    let z2:u8 = ELEMENT_TO_Z[element2];

    let slako:SlaterKosterTable = SlaterKosterTable::new(dipole,h,s,z1,z2,d_arr.to_vec());
    return slako;
}

pub fn process_slako_line(line:&str)->Vec<f64>{
    // convert a line into a list of column values respecting the
    // strange format conventions used in DFTB+ Slater-Koster files.
    // In Slater-Koster files used by DFTB+ zero columns
    // are not written: e.g. 4*0.0 has to be replaced
    // by four columns with zeros 0.0 0.0 0.0 0.0.

    let line:String = line.replace(","," ");
    let new_line:Vec<&str> = line.split(" ").collect();
    // println!("new line {:?}",new_line);
    let mut float_vec:Vec<f64> = Vec::new();
    for string in new_line{
        if string.contains("*"){
            let temp:Vec<&str> = string.split("*").collect();
            let count:usize = temp[0].trim().parse::<usize>().unwrap();
            let value:f64 = temp[1].trim().parse::<f64>().unwrap();
            for it in (0..count){
                float_vec.push(value);
            }
        }
        else{
            if string.len() > 0 && string.contains("\t")==false{
                // println!("string {:?}",string);
                let value:f64 = string.trim().parse::<f64>().unwrap();
                float_vec.push(value);
            }
        }
    }
    return float_vec;
}

fn get_tau_2_index(tuple:(u8,i32,u8,i32))->u8{
    let v1:u8 = tuple.0;
    let v2:i32 = tuple.1;
    let v3:u8 = tuple.2;
    let v4:i32 = tuple.3;
    let value:u8 = match (v1, v2, v3, v4) {
        (0,0,0,0) => 0,
        (0,0,1,0) => 2,
        (0,0,2,0) => 3,
        (1,0,0,0) => 4,
        (1,-1,1,-1) => 5,
        (1,0,1,0) => 6,
        (1,1,1,1) => 5,
        (1,-1,2,-1) => 7,
        (1,0,2,0) => 8,
        (1,1,2,1) => 7,
        (2,0,0,0) => 9,
        (2,-1,1,-1) => 10,
        (2,0,1,0) => 11,
        (2,1,1,1) => 10,
        (2,-2,2,-2) => 12,
        (2,-1,2,-1) => 13,
        (2,0,2,0) => 14,
        (2,1,2,1) => 13,
        (2,2,2,2) => 12,
        _ => panic!("false combination for tau_2_index!"),
    };
    return value;
}

pub fn get_reppot_table(element1: &str, element2: &str) -> RepulsivePotentialTable {
    let path_prefix: String = get_path_prefix();
    let filename: String = format!(
        "{}/src/param/repulsive_potential/reppot_tables/{}_{}.ron",
        path_prefix, element1, element2
    );
    let path: &Path = Path::new(&filename);
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let mut reppot_table: RepulsivePotentialTable =
        from_str(&data).expect("RON file was not well-formatted");
    reppot_table.spline_rep();
    reppot_table.dmax = reppot_table.d[reppot_table.d.len() -1];
    return reppot_table;
}

fn some_kind_of_uppercase_first_letter(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
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
    let mut y_values: Vec<f64> = Vec::new();
    let x_values: Vec<f64> = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    ];
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
        0.4214524357395346
    ];
    assert!(y_values.abs_diff_eq(&y_values_ref, 1e-16));
}

#[test]
fn test_spline_h0() {
    let path: &Path = Path::new("./src/param/slaterkoster/slako_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    let spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = slako_table.spline_hamiltonian();
    let mut y_values: Vec<f64> = Vec::new();
    let x_values: Vec<f64> = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    ];
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
        -0.3335812815557643
    ];
    assert!(y_values.abs_diff_eq(&y_values_ref, 1e-16));
}
