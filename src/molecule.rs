use crate::calculator::*;
use crate::constants::ATOM_NAMES;
use crate::defaults;
use crate::gamma_approximation;
use crate::graph::build_connectivity_matrix;
use crate::graph::*;
use crate::parameters::*;
use approx::AbsDiffEq;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use petgraph::algo::*;
use petgraph::data::*;
use petgraph::dot::{Config, Dot};
use petgraph::graph::*;
use petgraph::stable_graph::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Deref, Neg};
use log::{debug, error, info, trace, warn};
use crate::defaults::LONG_RANGE_RADIUS;
use crate::io::GeneralConfig;


#[derive(Clone)]
pub struct Molecule {
    pub(crate) atomic_numbers: Vec<u8>,
    pub(crate) positions: Array2<f64>,
    pub charge: i8,
    pub multiplicity: u8,
    pub n_atoms: usize,
    atomtypes: HashMap<u8, String>,
    pub proximity_matrix: Array2<bool>,
    pub distance_matrix: Array2<f64>,
    pub directions_matrix: Array3<f64>,
    pub calculator: DFTBCalculator,
    pub connectivity_matrix: Array2<bool>,
    pub full_graph: StableUnGraph<u8, f64>,
    pub full_graph_indices: Vec<NodeIndex>,
    pub sub_graphs: Vec<StableUnGraph<u8, f64>>,
    pub config: GeneralConfig,
}

impl Molecule {
    pub(crate) fn new(
        atomic_numbers: Vec<u8>,
        positions: Array2<f64>,
        charge: Option<i8>,
        multiplicity: Option<u8>,
        r_lr: Option<f64>,
        active_orbitals: Option<(usize, usize)>,
        config: GeneralConfig,
    ) -> Molecule {
        let (atomtypes, unique_numbers): (HashMap<u8, String>, Vec<u8>) =
            get_atomtypes(atomic_numbers.clone());
        let charge: i8 = charge.unwrap_or(defaults::CHARGE);
        let multiplicity: u8 = multiplicity.unwrap_or(defaults::MULTIPLICITY);
        let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
            distance_matrix(positions.view(), None);

        let n_atoms: usize = positions.nrows();

        let calculator: DFTBCalculator = DFTBCalculator::new(
            &atomic_numbers,
            &atomtypes,
            active_orbitals,
            &dist_matrix,
            r_lr,
        );
        //(&atomic_numbers, &atomtypes, model);

        let connectivity_matrix: Array2<bool> =
            build_connectivity_matrix(n_atoms, &dist_matrix, &atomic_numbers);

        let (graph, graph_indexes, subgraphs): (
            StableUnGraph<u8, f64>,
            Vec<NodeIndex>,
            Vec<StableUnGraph<u8, f64>>,
        ) = build_graph(&atomic_numbers, &connectivity_matrix, &dist_matrix);

        info!("{: <25} {}", "charge:", charge);
        info!("{: <25} {}", "multiplicity:", multiplicity);
        info!("{: <25} {:.8} bohr", "long-range radius:", r_lr.unwrap_or(LONG_RANGE_RADIUS));
        info!("{:-^80}", "");
        info!("{:^80}", "");

        let mol = Molecule {
            atomic_numbers: atomic_numbers,
            positions: positions,
            charge: charge,
            multiplicity: multiplicity,
            n_atoms: n_atoms,
            atomtypes: atomtypes,
            proximity_matrix: prox_matrix,
            distance_matrix: dist_matrix,
            directions_matrix: dir_matrix,
            calculator: calculator,
            connectivity_matrix: connectivity_matrix,
            full_graph: graph,
            full_graph_indices: graph_indexes,
            sub_graphs: subgraphs,
            config: config,
        };

        return mol;
    }

    pub fn iter_atomlist(
        &self,
    ) -> std::iter::Zip<
        std::slice::Iter<'_, u8>,
        ndarray::iter::AxisIter<'_, f64, ndarray::Dim<[usize; 1]>>,
    > {
        self.atomic_numbers.iter().zip(self.positions.outer_iter())
    }

    pub fn update_geometry(&mut self, coordinates: Array2<f64>) {
        self.positions = coordinates;
        let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
            distance_matrix(self.positions.view(), None);

        self.distance_matrix = dist_matrix.clone();
        self.directions_matrix = dir_matrix;
        self.proximity_matrix = prox_matrix;
        self.calculator
            .update_gamma_matrices(dist_matrix, &self.atomic_numbers);
    }
}

fn get_atomtypes(atomic_numbers: Vec<u8>) -> (HashMap<u8, String>, Vec<u8>) {
    // find unique atom types
    let mut unique_numbers: Vec<u8> = atomic_numbers;
    unique_numbers.sort_unstable(); // fast sort of atomic numbers
    unique_numbers.dedup(); // delete duplicates
    let mut atomtypes: HashMap<u8, String> = HashMap::new();
    for zi in &unique_numbers {
        atomtypes.insert(*zi, String::from(ATOM_NAMES[*zi as usize]));
    }
    return (atomtypes, unique_numbers);
}

pub fn distance_matrix(
    coordinates: ArrayView2<f64>,
    cutoff: Option<f64>,
) -> (Array2<f64>, Array3<f64>, Array2<bool>) {
    let cutoff: f64 = cutoff.unwrap_or(defaults::PROXIMITY_CUTOFF);
    let n_atoms: usize = coordinates.nrows();
    let mut dist_matrix: Array2<f64> = Array::zeros((n_atoms, n_atoms));
    let mut directions_matrix: Array3<f64> = Array::zeros((n_atoms, n_atoms, 3));
    let mut prox_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    for (i, pos_i) in coordinates.outer_iter().enumerate() {
        for (j0, pos_j) in coordinates.slice(s![i.., ..]).outer_iter().enumerate() {
            let j: usize = j0 + i;
            let r: Array1<f64> = &pos_i - &pos_j;
            let r_ij = r.norm();
            if i != j {
                dist_matrix[[i, j]] = r_ij;
                dist_matrix[[j, i]] = r_ij;
                let e_ij: Array1<f64> = r / r_ij;
                directions_matrix.slice_mut(s![i, j, ..]).assign(&e_ij);
                directions_matrix.slice_mut(s![j, i, ..]).assign(&-e_ij);
            }
            if r_ij <= cutoff {
                prox_matrix[[i, j]] = true;
                prox_matrix[[j, i]] = true;
            }
        }
    }
    //let directions_matrix = directions_matrix - directions_matrix.t();
    return (dist_matrix, directions_matrix, prox_matrix);
}

/// Test of distance matrix and proximity matrix of a water molecule. The xyz geometry of the
/// water molecule is
/// ```no_run
/// 3
//
// O          0.34215        1.17577        0.00000
// H          1.31215        1.17577        0.00000
// H          0.01882        1.65996        0.77583
///```
///
///
#[test]
fn test_distance_matrix() {
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
        distance_matrix(positions.view(), None);

    let dist_matrix_ref: Array2<f64> = array![
        [0.0000000000000000, 1.8330342089215557, 1.8330287870558954],
        [1.8330342089215557, 0.0000000000000000, 2.9933251510242216],
        [1.8330287870558954, 2.9933251510242216, 0.0000000000000000]
    ];

    let direction: Array3<f64> = array![
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-1.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.3333308828545918, -0.4991664249199420, -0.7998271080477469]
        ],
        [
            [1.0000000000000000, -0.0000000000000000, -0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.8164964343992185, -0.3056755882657617, -0.4897918000045972]
        ],
        [
            [-0.3333308828545918, 0.4991664249199420, 0.7998271080477469],
            [-0.8164964343992185, 0.3056755882657617, 0.4897918000045972],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ]
    ];
    assert!(dir_matrix.abs_diff_eq(&direction, 1.0e-14));
    assert!(dist_matrix.abs_diff_eq(&dist_matrix_ref, 1e-05));

    let prox_matrix_ref: Array2<bool> =
        array![[true, true, true], [true, true, true], [true, true, true]];
    assert_eq!(prox_matrix, prox_matrix_ref);
}
