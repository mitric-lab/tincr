use crate::constants;
use crate::param::Element;
use crate::utils::get_path_prefix;
use std::path::Path;
use ron::from_str;
use std::fs;
use serde::{Deserialize, Serialize};
use crate::param::skf_handler::{process_slako_line, SkfHandler, get_tau_2_index, get_index_to_symbol};
use ndarray::prelude::*;
use std::collections::HashMap;
use core::::{Atom, AtomSlice};
use crate::defaults::PROXIMITY_CUTOFF;
use crate::param::slako_transformations::{directional_cosines, slako_transformation, slako_transformation_gradients};

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


pub enum ParamFiles<'a> {
    SKF(&'a str),
    OWN,
}

/// Type that holds the mapping between element pairs and their [SlaterKosterTable].
/// This is basically a struct that allows to get the [SlaterKosterTable] without a strict
/// order of the [Element] tuple.
#[derive(Clone)]
pub struct SlaterKoster {
    pub map: HashMap<(Element, Element), SlaterKosterTable>,
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

    pub fn add_from_handler(&mut self, kind1: Element, kind2: Element,handler:SkfHandler,optional_table:Option<SlaterKosterTable>,order:&str){
        self.map
            .insert((kind1, kind2), SlaterKosterTable::from((&handler,optional_table,order)));
    }

    pub fn get(&self, kind1: Element, kind2: Element) -> &SlaterKosterTable {
        self.map
            .get(&(kind1, kind2))
            .unwrap_or_else(||self.map.get(&(kind2, kind1)).unwrap())
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


impl From<(&SkfHandler, Option<SlaterKosterTable>, &str)> for SlaterKosterTable {
    fn from(skf: (&SkfHandler, Option<SlaterKosterTable>, &str)) -> Self {
        // Extract the individual lines of the SKF file.
        let mut lines: Vec<&str> = skf.0.data_string.split("\n").collect();

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
        let tausymbols: Array1<&str> = match (skf.2) {
            ("ab") => (constants::TAUSYMBOLS_AB.iter().cloned().collect()),
            ("ba") => (constants::TAUSYMBOLS_BA.iter().cloned().collect()),
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
            dipole = slako_table.dipole.clone();
        }

        // Create a vector of arrays for the spline values of H0 and S
        // for each of the corresponding tausymbols
        let mut vec_h_arrays: Vec<Array1<f64>> = Vec::new();
        let mut vec_s_arrays: Vec<Array1<f64>> = Vec::new();
        for it in (0..10) {
            vec_s_arrays.push(Array1::zeros(npoints));
            vec_h_arrays.push(Array1::zeros(npoints));
        }
        let temp_vec: Vec<f64> = Array1::zeros(npoints).to_vec();

        // Fill all arrays with spline values.
        for it in (0..npoints) {
            let next_line: Vec<f64> = process_slako_line(lines[it]);
            for (pos, tausym) in tausymbols.slice(s![-10..]).iter().enumerate() {
                let symbol: (u8, i32, u8, i32) = constants::SYMBOL_2_TAU[*tausym];
                let l1: u8 = symbol.0;
                let l2: u8 = symbol.2;

                let mut orbital_parity: f64 = 0.0;
                if skf.2 == "ba" {
                    orbital_parity = -1.0_f64.powi((l1 + l2) as i32);
                } else {
                    orbital_parity = 1.0;
                }
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
            dipole: dipole,
            s: s,
            h: h,
            d: d_arr.to_vec(),
            dmax: dmax,
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


impl SlaterKoster {
    /// Computes the H0 and S outer diagonal block for two sets of atoms
    pub fn h0_and_s_ab(
        &self,
        n_orbs_a: usize,
        n_orbs_b: usize,
        atoms_a: &[Atom],
        atoms_b: &[Atom],
    ) -> (Array2<f64>, Array2<f64>) {
        let mut h0: Array2<f64> = Array2::zeros((n_orbs_a, n_orbs_b));
        let mut s: Array2<f64> = Array2::zeros((n_orbs_a, n_orbs_b));
        // iterate over atoms
        let mut mu: usize = 0;
        for (i, atomi) in atoms_a.iter().enumerate() {
            // iterate over orbitals on center i
            for orbi in atomi.valorbs.iter() {
                // iterate over atoms
                let mut nu: usize = 0;
                for (j, atomj) in atoms_b.iter().enumerate() {
                    // iterate over orbitals on center j
                    for orbj in atomj.valorbs.iter() {
                        //if geometry.proximities.as_ref().unwrap()[[i, j]] {
                        if (atomi-atomj).norm() < PROXIMITY_CUTOFF {
                            if atomi<=atomj {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atomi.xyz, &atomj.xyz);
                                s[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &self.get(atomi.kind, atomj.kind).s_spline,
                                    orbi.l,
                                    orbi.m,
                                    orbj.l,
                                    orbj.m,
                                );
                                h0[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &self.get(atomi.kind, atomj.kind).h_spline,
                                    orbi.l,
                                    orbi.m,
                                    orbj.l,
                                    orbj.m,
                                );
                            } else {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atomj.xyz, &atomi.xyz);
                                s[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &self.get(atomj.kind, atomi.kind).s_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                );
                                h0[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &self.get(atomj.kind, atomi.kind).h_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                );
                            }
                        }
                        nu = nu + 1;
                    }
                }
                mu = mu + 1;
            }
        }
        return (s, h0);
    }

    /// Computes the H0 and S matrix elements for a single molecule.
    pub fn h0_and_s(&self, n_orbs: usize, atoms: AtomSlice) -> (Array2<f64>, Array2<f64>) {
        let mut h0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
        let mut s: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
        // iterate over atoms
        let mut mu: usize = 0;
        for (i, xyz_i) in atoms.iter().enumerate() {
            // iterate over orbitals on center i
            for orbi in atomi.valorbs.iter() {
                // iterate over atoms
                let mut nu: usize = 0;
                for (j, atomj) in atoms.iter().enumerate() {
                    // iterate over orbitals on center j
                    for orbj in atomj.valorbs.iter() {
                        //if geometry.proximities.as_ref().unwrap()[[i, j]] {
                        if (atomi-atomj).norm() < PROXIMITY_CUTOFF {
                            if mu < nu {
                                if atomi <= atomj {
                                    if i != j {
                                        let (r, x, y, z): (f64, f64, f64, f64) =
                                            directional_cosines(&atomi.xyz, &atomj.xyz);
                                        s[[mu, nu]] = slako_transformation(
                                            r,
                                            x,
                                            y,
                                            z,
                                            &self.get(atomi.kind, atomj.kind).s_spline,
                                            orbi.l,
                                            orbi.m,
                                            orbj.l,
                                            orbj.m,
                                        );
                                        h0[[mu, nu]] = slako_transformation(
                                            r,
                                            x,
                                            y,
                                            z,
                                            &self.get(atomi.kind, atomj.kind).h_spline,
                                            orbi.l,
                                            orbi.m,
                                            orbj.l,
                                            orbj.m,
                                        );
                                    }
                                } else {
                                    let (r, x, y, z): (f64, f64, f64, f64) =
                                        directional_cosines(&atomj.xyz, &atomi.xyz);
                                    s[[mu, nu]] = slako_transformation(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &self.get(atomj.kind, atomi.kind).s_spline,
                                        orbj.l,
                                        orbj.m,
                                        orbi.l,
                                        orbi.m,
                                    );
                                    h0[[mu, nu]] = slako_transformation(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &self.get(atomj.kind, atomi.kind).h_spline,
                                        orbj.l,
                                        orbj.m,
                                        orbi.l,
                                        orbi.m,
                                    );
                                }

                            } else if mu == nu {
                                assert_eq!(atomi.number, atomj.number);
                                h0[[mu, nu]] = orbi.energy;
                                s[[mu, nu]] = 1.0;
                            } else {
                                s[[mu, nu]] = s[[nu, mu]];
                                h0[[mu, nu]] = h0[[nu, mu]];
                            }
                        }
                        nu = nu + 1;
                    }
                }
                mu = mu + 1;
            }
        }
        return (s, h0);
    }

    // gradients of overlap matrix S and 0-order hamiltonian matrix H0
    // using Slater-Koster Rules
    //
    // Parameters:
    // ===========
    // atomlist: list of tuples (Zi,[xi,yi,zi]) of atom types and positions
    // valorbs: list of valence orbitals with quantum numbers (ni,li,mi)
    // SKT: Slater Koster table
    // Mproximity: M[i,j] == 1, if the atoms i and j are close enough
    // so that the gradients for matrix elements
    // between orbitals on i and j should be computed
    pub fn h0_and_s_gradients(&self, atoms: &[Atom], n_orbs: usize) -> (Array3<f64>, Array3<f64>) {
        let n_atoms: usize = atoms.len();
        let mut grad_h0: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
        let mut grad_s: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
        // iterate over atoms
        let mut mu: usize = 0;
        for (i, atomi) in atoms.iter().enumerate() {
            // iterate over orbitals on center i
            for orbi in atomi.valorbs.iter() {
                // iterate over atoms
                let mut nu: usize = 0;
                for (j, atomj) in atoms.iter().enumerate() {
                    // iterate over orbitals on center j
                    for orbj in atomj.valorbs.iter() {
                        if (atomi-atomj).norm() < PROXIMITY_CUTOFF && mu != nu {
                            let mut s_deriv: Array1<f64> = Array1::zeros([3]);
                            let mut h0_deriv: Array1<f64> = Array1::zeros([3]);
                            if atomi <= atomj {
                                if i != j {
                                    // the hardcoded Slater-Koster rules compute the gradient
                                    // with respect to r = posj - posi
                                    // but we want the gradient with respect to posi, so an additional
                                    // minus sign is introduced
                                    let (r, x, y, z): (f64, f64, f64, f64) =
                                        directional_cosines(&atomi.xyz, &atomj.xyz);
                                    s_deriv = -1.0 * slako_transformation_gradients(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &self.get(atomi.kind, atomj.kind).s_spline,
                                        orbi.l,
                                        orbi.m,
                                        orbj.l,
                                        orbj.m,
                                    );
                                    h0_deriv = -1.0 * slako_transformation_gradients(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &self.get(atomi.kind, atomj.kind).h_spline,
                                        orbi.l,
                                        orbi.m,
                                        orbj.l,
                                        orbj.m,
                                    );
                                }
                            } else {
                                // swap atoms if Zj > Zi, since posi and posj are swapped, the gradient
                                // with respect to r = posi - posj equals the gradient with respect to
                                // posi, so no additional minus sign is needed.
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(&atomj.xyz, &atomi.xyz);
                                s_deriv = slako_transformation_gradients(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &self.get(atomi.kind, atomj.kind).s_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                );
                                h0_deriv = slako_transformation_gradients(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &self.get(atomi.kind, atomj.kind).h_spline,
                                    orbj.l,
                                    orbj.m,
                                    orbi.l,
                                    orbi.m,
                                );
                            }

                            grad_s
                                .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                                .assign(&s_deriv);
                            grad_h0
                                .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                                .assign(&h0_deriv);
                            // S and H0 are hermitian/symmetric
                            grad_s
                                .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                                .assign(&s_deriv);
                            grad_h0
                                .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                                .assign(&h0_deriv);
                        }
                        nu = nu + 1;
                    }
                }
                mu = mu + 1;
            }
        }
        return (grad_s, grad_h0);
    }

}
