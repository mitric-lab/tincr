use crate::parametrization::skf_handler::*;
use crate::utils::get_path_prefix;
use crate::{constants, Element, SorH, Spline};
use hashbrown::HashMap;
use ndarray::prelude::*;
use ron::from_str;
use serde::{Deserialize, Serialize};
use std::collections::HashMap as StdHashMap;
use std::fs;
use std::path::Path;

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

fn init_hashmap() -> HashMap<u8, Spline> {
    HashMap::new()
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
    index_to_symbol: StdHashMap<u8, String>,
    #[serde(default = "init_hashmap")]
    /// Spline representation for the overlap matrix elements
    pub s_spline: HashMap<u8, Spline>,
    /// Spline representation for the H0 matrix elements
    #[serde(default = "init_hashmap")]
    pub h_spline: HashMap<u8, Spline>,
}

impl SlaterKosterTable {
    /// Creates a new [SlaterKosterTable] from two elements and splines the H0 and overlap
    /// matrix elements
    pub fn new(kind1: Element, kind2: Element) -> Self {
        let path_prefix: String = get_path_prefix();
        let (kind1, kind2) = if kind1 > kind2 {
            (kind2, kind1)
        } else {
            (kind1, kind2)
        };
        let filename: String = format!(
            "{}/src/param/slaterkoster/slako_tables/{}_{}.ron",
            path_prefix,
            kind1.symbol().to_lowercase(),
            kind2.symbol().to_lowercase()
        );
        let path: &Path = Path::new(&filename);
        let data: String =
            fs::read_to_string(path).expect(&*format! {"Unable to read file {}", &filename});
        let mut slako_table: SlaterKosterTable =
            from_str(&data).expect("RON file was not well-formatted");
        slako_table.dmax = slako_table.d[slako_table.d.len() - 1];
        slako_table.s_spline = slako_table.spline_overlap();
        slako_table.h_spline = slako_table.spline_hamiltonian();
        slako_table
    }

    fn spline_overlap(&self) -> HashMap<u8, Spline> {
        self.s
            .iter()
            .map(|((_, _, i), y)| (*i, Spline::new(&self.d, y)))
            .collect::<HashMap<u8, Spline>>()
    }

    fn spline_hamiltonian(&self) -> HashMap<u8, Spline> {
        self.h
            .iter()
            .map(|((_, _, i), y)| (*i, Spline::new(&self.d, y)))
            .collect::<HashMap<u8, Spline>>()
    }

    pub fn get_splines(&self, s_or_h: SorH) -> &HashMap<u8, Spline> {
        match s_or_h {
            SorH::S => &self.s_spline,
            SorH::H0 => &self.h_spline,
        }
    }
}

impl From<(&SkfHandler, Option<SlaterKosterTable>, &str)> for SlaterKosterTable {
    fn from(skf: (&SkfHandler, Option<SlaterKosterTable>, &str)) -> Self {
        // Extract the individual lines of the SKF file.
        let mut lines: Vec<&str> = skf.0.data_string.split('\n').collect();

        // Read the first line of the skf file
        // It contains the r0 parameter/the grid distance and the number of grid points
        let first_line: Vec<f64> = process_slako_line(lines[0]);
        let grid_dist: f64 = first_line[0];
        let npoints: usize = first_line[1] as usize;

        // Remove the first line
        lines.remove(0);
        if skf.0.el_a.number() == skf.0.el_b.number() {
            // remove second line
            lines.remove(0);
        }
        // Remove second/third line
        lines.remove(0);

        // Create the grid
        let d_arr: Array1<f64> =
            Array1::linspace(0.02, grid_dist * ((npoints - 1) as f64), npoints);

        let next_line: Vec<f64> = process_slako_line(lines[0]);
        let length: usize = next_line.len() / 2;
        assert!(length == 10);

        // Create vector of tausymbols, which correspond to the orbital combinations
        let tausymbols: Array1<&str> = match skf.2 {
            "ab" => (constants::TAUSYMBOLS_AB.iter().cloned().collect()),
            "ba" => (constants::TAUSYMBOLS_BA.iter().cloned().collect()),
            _ => panic!("Wrong order specified! Only 'ab' or 'ba' is allowed!"),
        };
        let length_tau: usize = tausymbols.len();

        // Define hashmaps for the H0 matrix elements, the overlap matrix elements and the dipoles.
        let mut h: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();
        let mut s: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();
        let mut dipole: HashMap<(u8, u8, u8), Vec<f64>> = HashMap::new();

        // Fill hashmaps for H0 and S with the values of the Slater-Koster-Table
        // for the order 'ab' to combine them with 'ba'
        if skf.1.is_some() {
            let slako_table: SlaterKosterTable = skf.1.unwrap();
            h = slako_table.h.clone();
            s = slako_table.s.clone();
            dipole = slako_table.dipole;
        }

        // Create a vector of arrays for the spline values of H0 and S
        // for each of the corresponding tausymbols
        let mut vec_h_arrays: Vec<Array1<f64>> = Vec::new();
        let mut vec_s_arrays: Vec<Array1<f64>> = Vec::new();
        for _ in 0..10 {
            vec_s_arrays.push(Array1::zeros(npoints));
            vec_h_arrays.push(Array1::zeros(npoints));
        }
        let temp_vec: Vec<f64> = Array1::zeros(npoints).to_vec();

        // Fill all arrays with spline values.
        for it in 0..npoints {
            let next_line: Vec<f64> = process_slako_line(lines[it]);
            for (pos, tausym) in tausymbols.slice(s![-10..]).iter().enumerate() {
                let symbol: (u8, i32, u8, i32) = constants::SYMBOL_2_TAU[*tausym];
                let l1: u8 = symbol.0;
                let l2: u8 = symbol.2;

                let orbital_parity = if skf.2 == "ba" {
                    -(1.0_f64.powi((l1 + l2) as i32))
                } else {
                    1.0
                };
                vec_h_arrays[pos][it] = orbital_parity * next_line[pos];
                vec_s_arrays[pos][it] = orbital_parity * next_line[length_tau + pos];
            }
        }

        // Fill the HashMaps with the spline values.
        for (pos, tausymbol) in tausymbols.slice(s![-10..]).iter().enumerate() {
            let symbol: (u8, i32, u8, i32) = constants::SYMBOL_2_TAU[*tausymbol];
            let index: u8 = get_tau_2_index(symbol);
            if !h.contains_key(&(symbol.0, symbol.2, index)) {
                h.insert((symbol.0, symbol.2, index), vec_h_arrays[pos].to_vec());
            }
            if !s.contains_key(&(symbol.0, symbol.2, index)) {
                s.insert((symbol.0, symbol.2, index), vec_s_arrays[pos].to_vec());
            }
            dipole.insert((symbol.0, symbol.2, index), temp_vec.clone());
        }

        // Create the Slater-Koster-Table
        let dmax: f64 = d_arr[d_arr.len() - 1];
        let mut slako: SlaterKosterTable = SlaterKosterTable {
            dipole,
            s,
            h,
            d: d_arr.to_vec(),
            dmax,
            z1: skf.0.el_a.number(),
            z2: skf.0.el_b.number(),
            h_spline: init_hashmap(),
            s_spline: init_hashmap(),
            index_to_symbol: get_index_to_symbol(),
        };
        if skf.2 == "ba" || (skf.0.el_a == skf.0.el_b) {
            slako.s_spline = slako.spline_overlap();
            slako.h_spline = slako.spline_hamiltonian();
        }
        slako
    }
}
