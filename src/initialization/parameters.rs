use crate::param::Element;
use crate::constants;
use crate::defaults;
use ndarray::prelude::*;
use ron::de::from_str;
use rusty_fitpack;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use crate::utils::get_path_prefix;

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

/// Type that holds the mapping between element pairs and their [SlaterKosterTable].
/// This is basically a struct that allows to get the [SlaterKosterTable] without a strict
/// order of the [Element] tuple.
#[derive(Clone)]
pub struct SlaterKoster{
    map: HashMap<(Element, Element), SlaterKosterTable>,
}

impl SlaterKoster {
    /// Create a new [SlaterKoster] type, that maps a tuple of [Element] s to a [SlaterKosterTable].
    pub fn new() -> Self {
        SlaterKoster {
            map: HashMap::new(),
        }
    }

    /// Add a new [SlaterKosterTable] from a tuple of two [Element]s. THe
    pub fn add(&mut self, kind1: Element, kind2: Element) {
        self.map
            .insert((kind1, kind2), SlaterKosterTable::new(kind1, kind2));
    }

    pub fn get(&self, kind1: Element, kind2: Element) -> &SlaterKosterTable {
        self.map
            .get(&(kind1, kind2))
            .unwrap_or(self.map.get(&(kind2, kind1)).unwrap())
    }
}

/// Type that holds the pairwise atomic parameters for the Slater-Koster matrix elements
#[derive(Serialize, Deserialize, Clone)]
pub struct SlaterKosterTable {
    dipole: HashMap<(u8, u8, u8), Vec<f64>>,
    h: HashMap<(u8, u8, u8), Vec<f64>>,
    s: HashMap<(u8, u8, u8), Vec<f64>>,
    /// Atomic number of the first element of the atom pair
    z1: u8,
    /// Atomic number of the second element of the atom pair
    z2: u8,
    /// Grid with the atom-atom distances in bohr for which the H0 and overlap matrix elements
    /// are tabulated
    d: Vec<f64>,
    /// Maximal atom-atom distance of the grid. This is obtained by taking `d.max()`
    /// `dmax` is only checked in
    /// [get_h0_and_s_mu_nu](crate::param::slako_transformations::get_h0_and_s_mu_nu)
    #[serde(default = "get_nan_value")]
    pub dmax: f64,
    index_to_symbol: HashMap<u8, String>,
    #[serde(default = "init_hashmap")]
    /// Spline representation for the overlap matrix elements
    pub s_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
    /// Spline representation for the H0 matrix elements
    #[serde(default = "init_hashmap")]
    pub h_spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
}

impl SlaterKosterTable {
    /// Creates a new [SlaterKosterTable] from two elements and splines the H0 and overlap
    /// matrix elements
    pub fn new(kind1: Element, kind2: Element) -> Self {
        let path_prefix: String = get_path_prefix();
        let (kind1, kind2) = if kind1 > kind2 {
            (kind2, kind1)
        } else{
            (kind1, kind2)
        };
        let filename: String = format!(
            "{}/src/param/slaterkoster/slako_tables/{}_{}.ron",
            path_prefix,
            kind1.symbol().to_lowercase(),
            kind2.symbol().to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect(&*format! {"Unable to read file {}", &filename});
        let mut slako_table: SlaterKosterTable =
            from_str(&data).expect("RON file was not well-formatted");
        slako_table.dmax = slako_table.d[slako_table.d.len()-1];
        slako_table.s_spline = slako_table.spline_overlap();
        slako_table.h_spline = slako_table.spline_hamiltonian();
        slako_table
    }

    pub(crate) fn spline_overlap(&self) -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
        let mut splines: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
        for ((_l1, _l2, i), value) in &self.s {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(
                *i,
                rusty_fitpack::splrep(
                    x, y, None, None, None, None, None, None, None, None, None, None,
                ),
            );
        }
        splines
    }

    pub(crate) fn spline_hamiltonian(&self) -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
        let mut splines: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();
        for ((_l1, _l2, i), value) in &self.h {
            let x: Vec<f64> = self.d.clone();
            let y: Vec<f64> = value.clone();
            splines.insert(
                *i,
                rusty_fitpack::splrep(
                    x, y, None, None, None, None, None, None, None, None, None, None,
                ),
            );
        }
        splines
    }
}

/// Type that holds the mapping between element pairs and their [RepulsivePotentialTable].
/// This is basically a struct that allows to get the [RepulsivePotentialTable] without a s
/// order of the [Element] tuple.
#[derive(Clone)]
pub struct RepulsivePotential {
    map: HashMap<(Element, Element), RepulsivePotentialTable>,
}

impl RepulsivePotential {
    /// Create a new RepulsivePotential, to map the [Element] pairs to a [RepulsivePotentialTable]
    pub fn new() -> Self {
        RepulsivePotential {
            map: HashMap::new(),
        }
    }

    /// Add a [RepulsivePotentialTable] from a pair of two [Element]s
    pub fn add(&mut self, kind1: Element, kind2: Element) {
        self.map.insert(
            (kind1, kind2),
            RepulsivePotentialTable::new(kind1, kind2),
        );
    }

    /// Return the [RepulsivePotentialTable] for the tuple of two [Element]s. The order of
    /// the tuple does not play a role.
    pub fn get(&self, kind1: Element, kind2: Element) -> &RepulsivePotentialTable {
        self.map
            .get(&(kind1, kind2))
            .unwrap_or(self.map.get(&(kind2, kind1)).unwrap())
    }
}

/// Type that contains the repulsive potential between a pair of atoms and their derivative as
/// splines.
#[derive(Serialize, Deserialize, Clone)]
pub struct RepulsivePotentialTable {
    /// Repulsive energy in Hartree on the grid `d`
    vrep: Vec<f64>,
    /// Atomic number of first element of the pair
    z1: u8,
    /// Atomic number of the second element of the pair
    z2: u8,
    /// Grid for which the repulsive energies are tabulated in bohr.
    d: Vec<f64>,
    /// Spline representation as a tuple of ticks, coefficients and the degree
    #[serde(default = "init_none")]
    spline_rep: Option<(Vec<f64>, Vec<f64>, usize)>,
    /// Maximal atom-atom distance for which the repulsive energy is tabulated in the parameter file.
    /// The value is set from d.max()
    #[serde(default = "get_nan_value")]
    dmax: f64,
}

impl RepulsivePotentialTable {
    /// Create a new [RepulsivePotentialTable] from two [Elements]. The parameter file will be read
    /// and the repulsive energy will be splined and the spline representation will be stored.
    pub fn new(kind1: Element, kind2: Element) -> Self {
        let path_prefix: String = get_path_prefix();
        let (kind1, kind2) = if kind1 > kind2 {
            (kind2, kind1)
        } else{
            (kind1, kind2)
        };
        let filename: String = format!(
            "{}/src/param/repulsive_potential/reppot_tables/{}_{}.ron",
            path_prefix,
            kind1.symbol().to_lowercase(),
            kind2.symbol().to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");
        let mut reppot_table: RepulsivePotentialTable =
            from_str(&data).expect("RON file was not well-formatted");
        reppot_table.spline_rep();
        reppot_table.dmax = reppot_table.d[reppot_table.d.len()-1];
        return reppot_table;
    }

    /// Create the spline representation by calling the [splrep](rusty_fitpack::splrep) Routine.
    fn spline_rep(&mut self) {
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

    /// Evaluate the spline at the atom-atom distance `x`. The units of x are in bohr.
    /// If `x` > `dmax` then 0 is returned, where `dmax` is the maximal atom-atom distance on the grid
    /// for which the repulsive energy is reported
    pub fn spline_eval(&self, x: f64) -> f64 {
        match &self.spline_rep {
            Some((t, c, k)) => {
                if x <= self.dmax {
                    rusty_fitpack::splev_uniform(t, c, *k, x)
                } else {
                    0.0
                }
            }
            None => panic!("No spline representation available"),
        }
    }
    /// Evaluate the first derivate of the energy w.r.t. to the atomic displacements. The units
    /// of the distance `x` are also in bohr.  If `x` > `dmax` then 0 is returned,
    /// where `dmax` is the maximal atom-atom distance on the grid
    /// for which the repulsive energy is reported
    pub fn spline_deriv(&self, x: f64) -> f64 {
        match &self.spline_rep {
            Some((t, c, k)) => {
                if x <= self.dmax {
                    rusty_fitpack::splder_uniform(t, c, *k, x, 1)
                } else {
                    0.0
                }
            }
            None => panic!("No spline representation available"),
        }
    }
}

impl From<SkfHandler> for RepulsivePotentialTable{
    fn from(skf_handler:SkfHandler)->Self{
        let lines:Vec<&str> = skf_handler.data_string.split("\n").collect();

        let mut count: usize = 0;
        // search beginning of repulsive potential in the skf file
        for (it, line) in lines.iter().enumerate() {
            if line.contains("Spline") {
                count = it;
                break;
            }
        }

        // get number of points and the cutoff from the second line
        let second_line:Vec<f64> = process_slako_line(lines[count+1]);
        let n_int:usize = second_line[0] as usize;
        let cutoff:f64 = second_line[1];

        // Line 3: V(r < r0) = exp(-a1*r+a2) + a3   is r too small to be covered by the spline
        let third_line: Vec<f64> = process_slako_line(lines[count + 2]);
        let a_1: f64 = third_line[0];
        let a_2: f64 = third_line[1];
        let a_3: f64 = third_line[2];

        // read spline values from the skf file
        let mut rs: Array1<f64> = Array1::zeros(n_int);
        let mut cs: Array2<f64> = Array2::zeros((4, n_int));
        // start from the 4th line after "Spline"
        count = count + 3;
        let mut end: f64 = 0.0;
        let mut iteration_count: usize = 0;
        for it in (count..(n_int + count)) {
            let next_line: Vec<f64> = process_slako_line(lines[it]);
            rs[iteration_count] = next_line[0];
            let array: Array1<f64> = array![next_line[2], next_line[3], next_line[4], next_line[5]];
            cs.slice_mut(s![.., iteration_count]).assign(&array);
            end = next_line[1];

            iteration_count += 1;
        }
        assert!(end == cutoff);

        // Now we evaluate the spline on a equidistant grid
        let npoints: usize = 100;
        let d_arr: Array1<f64> = Array1::linspace(0.0, cutoff, npoints);
        let mut v_rep: Array1<f64> = Array1::zeros(npoints);

        let mut spline_counter: usize = 0;
        for (i, di) in d_arr.iter().enumerate() {
            if di < &rs[0] {
                v_rep[i] = (-&a_1 * di + a_2).exp() + a_3;
            } else {
                // find interval such that r[j] <= di < r[j+1]
                while di >= &rs[spline_counter + 1] && spline_counter < (n_int - 2) {
                    spline_counter += 1;
                }
                if spline_counter < (n_int - 2) {
                    assert!(rs[spline_counter] <= *di);
                    assert!(di < &rs[spline_counter + 1]);
                    let c_arr: ArrayView1<f64> = cs.slice(s![.., spline_counter]);
                    v_rep[i] = c_arr[0]
                        + c_arr[1] * (di - rs[spline_counter])
                        + c_arr[2] * (di - rs[spline_counter]).powi(2)
                        + c_arr[3] * (di - rs[spline_counter]).powi(3);
                } else {
                    v_rep[i] = 0.0;
                }
            }
        }

        let dmax:f64 = d_arr[d_arr.len()-1];

        Self{
            dmax:dmax,
            z1:skf_handler.z1,
            z2:skf_handler.z2,
            vrep:v_rep.to_vec(),
            d:d_arr.to_vec(),
            spline_rep:None,
        }
    }
}

pub struct SkfHandler{
    z1:u8,
    z2:u8,
    data_string:String,
}

impl SkfHandler{
    pub fn new(
        z1:u8,
        z2:u8,
    )->SkfHandler{
        let path_prefix: String = String::from(defaults::MIO_DIR);
        let element_1:&str = constants::ATOM_NAMES_UPPER[z1 as usize];
        let element_2:&str = constants::ATOM_NAMES_UPPER[z2 as usize];
        let filename: String = format!("{}/{}-{}.skf", path_prefix, element_1, element_2);
        let path: &Path = Path::new(&filename);
        let data: String = fs::read_to_string(path).expect("Unable to read file");

        SkfHandler{
            z1:z1,
            z2:z2,
            data_string:data,
        }
    }
}

pub fn process_slako_line(line: &str) -> Vec<f64> {
    // convert a line into a list of column values respecting the
    // strange format conventions used in DFTB+ Slater-Koster files.
    // In Slater-Koster files used by DFTB+ zero columns
    // are not written: e.g. 4*0.0 has to be replaced
    // by four columns with zeros 0.0 0.0 0.0 0.0.

    let line: String = line.replace(",", " ");
    let new_line: Vec<&str> = line.split(" ").collect();
    // println!("new line {:?}",new_line);
    let mut float_vec: Vec<f64> = Vec::new();
    for string in new_line {
        if string.contains("*") {
            let temp: Vec<&str> = string.split("*").collect();
            let count: usize = temp[0].trim().parse::<usize>().unwrap();
            let value: f64 = temp[1].trim().parse::<f64>().unwrap();
            for it in (0..count) {
                float_vec.push(value);
            }
        } else {
            if string.len() > 0 && string.contains("\t") == false {
                // println!("string {:?}",string);
                let value: f64 = string.trim().parse::<f64>().unwrap();
                float_vec.push(value);
            }
        }
    }
    return float_vec;
}

