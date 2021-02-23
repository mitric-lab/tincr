use crate::constants;
use crate::defaults;
use crate::gradients;
use crate::molecule::distance_matrix;
use crate::Molecule;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use petgraph::graph::*;
use std::collections::HashMap;
use petgraph::dot::Dot;

pub fn build_connectivity_matrix(mol: &Molecule) -> (Array2<bool>) {
    let mut connectivtiy_matrix: Array2<bool> = Array::default((mol.n_atoms, mol.n_atoms));
    for i in (0..mol.n_atoms).combinations(2) {
        let r_cov = (constants::COVALENCE_RADII[&mol.atomic_numbers[i[0]]]
            + constants::COVALENCE_RADII[&mol.atomic_numbers[i[1]]])
            / 0.52917720859;
        if mol.distance_matrix[[i[0], i[1]]] < (1.3 * r_cov) {
            connectivtiy_matrix[[i[0], i[1]]] = true;
            connectivtiy_matrix[[i[1], i[0]]] = true;
        }
    }
    return connectivtiy_matrix;
}

pub fn build_graph(mol: &Molecule,connectivity_matrix:&Array2<bool>)->(Graph<HashMap<u8,Array1<f64>>, f64>,Vec<NodeIndex>) {
    let mut graph: Graph<HashMap<u8,Array1<f64>>, f64> = Graph::<HashMap<u8,Array1<f64>>, f64>::new();
    for (i,z_i) in mol.atomic_numbers.iter().enumerate(){
        let mut hash:HashMap<u8,Array1<f64>> = HashMap::new();
        hash.insert(*z_i,mol.positions.slice(s![i,..]).to_owned());
        graph.add_node(hash);
    }
    let indexes:Vec<NodeIndex> = graph.node_indices().collect::<Vec<_>>();
    for (i,index) in connectivity_matrix.outer_iter().enumerate(){
        for (j,ind) in index.iter().enumerate(){
            let j0 = j+i;
            if connectivity_matrix[[i,j0]]{
                graph.add_edge(indexes[i],indexes[j],mol.distance_matrix[[i,j]]);
            }
        }
    }
    return (graph,indexes);
}

#[test]
fn connectivity_routine() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule =
        Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    let connectivity_matrix: Array2<bool> = build_connectivity_matrix(&mol);

    let (graph,indexes):(Graph<HashMap<u8,Array1<f64>>, f64>,Vec<NodeIndex>) = build_graph(&mol,&connectivity_matrix);

    println!("connectivity_matrix {}", connectivity_matrix);
    assert!(1 == 2);
}
