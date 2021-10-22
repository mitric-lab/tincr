use crate::param::Element;
use crate::utils::get_path_prefix;
use std::fs;
use std::path::Path;
use crate::param::skf_handler::{SkfHandler, process_slako_line};
use hashbrown::HashMap;
use ndarray::prelude::*;
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

/// Type that holds the mapping between element pairs and their [RepulsivePotentialTable].
/// This is basically a struct that allows to get the [RepulsivePotentialTable] without a s
/// order of the [Element] tuple.
#[derive(Clone)]
pub struct RepulsivePotential {
    pub map: HashMap<(Element, Element), RepulsivePotentialTable>,
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
        self.map.insert((kind1, kind2), RepulsivePotentialTable::new(kind1, kind2));
    }

    /// Return the [RepulsivePotentialTable] for the tuple of two [Element]s. The order of
    /// the tuple does not play a role.
    pub fn get(&self, kind1: Element, kind2: Element) -> &RepulsivePotentialTable {
        self.map
            .get(&(kind1, kind2))
            .unwrap_or_else(||self.map.get(&(kind2, kind1)).unwrap())
    }

    /// Compute energy due to core electrons and nuclear repulsion.
    pub fn get_repulsive_energy(&self, atoms: Atomslice) -> f64 {
        let mut e_nuc: f64 = 0.0;
        for (i, atomi) in atoms[1..atoms.len()].iter().enumerate(){
            for atomj in atoms[0..i + 1].iter() {
                let r: f64 = (atomi - atomj).norm();
                // nucleus-nucleus and core-electron repulsion
                e_nuc += self.get(atomi.kind, atomj.kind).spline_eval(r);
            }
        }
        return e_nuc;
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
        } else {
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
        reppot_table.dmax = reppot_table.d[reppot_table.d.len() - 1];
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

impl From<&SkfHandler> for RepulsivePotentialTable {
    fn from(skf_handler: &SkfHandler) -> Self {
        // Extract the individual lines of the SKF file.
        let lines: Vec<&str> = skf_handler.data_string.split("\n").collect();

        let mut count: usize = 0;
        // Search beginning of repulsive potential in the SKF file
        for (it, line) in lines.iter().enumerate() {
            if line.contains("Spline") {
                count = it;
                break;
            }
        }

        // Number of points and the cutoff are read from the second line.
        let second_line: Vec<f64> = process_slako_line(lines[count + 1]);
        let n_int: usize = second_line[0] as usize;
        let cutoff: f64 = second_line[1];

        // Line 3: V(r < r0) = exp(-a1*r+a2) + a3   is r too small to be covered by the spline
        let third_line: Vec<f64> = process_slako_line(lines[count + 2]);
        let a_1: f64 = third_line[0];
        let a_2: f64 = third_line[1];
        let a_3: f64 = third_line[2];

        // Read spline values from the SKF file.
        let mut rs: Array1<f64> = Array1::zeros(n_int);
        let mut cs: Array2<f64> = Array2::zeros((4, n_int));
        // Start from the 4th line after the word "Spline"
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
        assert!((end - cutoff).abs() < f64::EPSILON);

        // The spline is evaluated on an equidistant grid.
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

        let dmax: f64 = d_arr[d_arr.len() - 1];

        let mut rep_table: RepulsivePotentialTable = RepulsivePotentialTable {
            dmax: dmax,
            z1: skf_handler.el_a.number(),
            z2: skf_handler.el_b.number(),
            vrep: v_rep.to_vec(),
            d: d_arr.to_vec(),
            spline_rep: None,
        };
        rep_table.spline_rep();
        rep_table
    }
}


