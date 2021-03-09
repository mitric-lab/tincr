use crate::constants;
use crate::defaults;
use crate::gradients;
use crate::molecule::distance_matrix;
use crate::Molecule;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::{indices, Data};
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use petgraph::algo::*;
use petgraph::data::*;
use petgraph::dot::{Config, Dot};
use petgraph::graph::*;
use petgraph::stable_graph::*;
use std::collections::HashMap;

pub fn build_connectivity_matrix(
    n_atoms: usize,
    distance_matrix: &Array2<f64>,
    atomic_numbers: &Vec<u8>,
) -> (Array2<bool>) {
    let mut connectivtiy_matrix: Array2<bool> = Array::default((n_atoms, n_atoms));
    for i in (0..n_atoms).combinations(2) {
        let r_cov = (constants::COVALENCE_RADII[&atomic_numbers[i[0]]]
            + constants::COVALENCE_RADII[&atomic_numbers[i[1]]])
            / 0.52917720859;
        if distance_matrix[[i[0], i[1]]] < (1.3 * r_cov) {
            connectivtiy_matrix[[i[0], i[1]]] = true;
            connectivtiy_matrix[[i[1], i[0]]] = true;
        }
    }
    return connectivtiy_matrix;
}

pub fn build_graph(
    atomic_numbers: &Vec<u8>,
    connectivity_matrix: &Array2<bool>,
    distance_matrix: &Array2<f64>,
) -> (
    StableUnGraph<u8, f64>,
    Vec<NodeIndex>,
    Vec<StableUnGraph<u8, f64>>,
) {
    let mut graph: StableUnGraph<u8, f64> = StableUnGraph::<u8, f64>::default();
    for (i, z_i) in atomic_numbers.iter().enumerate() {
        graph.add_node(*z_i);
    }
    let indexes: Vec<NodeIndex> = graph.node_indices().collect::<Vec<_>>();
    for (i, index) in connectivity_matrix.outer_iter().enumerate() {
        for (j, ind) in index.iter().enumerate() {
            if j < index.len() - i {
                let j0: usize = j + i;
                if connectivity_matrix[[i, j0]] {
                    graph.add_edge(indexes[i], indexes[j0], distance_matrix[[i, j0]]);
                }
            }
        }
    }
    let mut subgraph_vector: Vec<StableUnGraph<u8, f64>> = Vec::new();
    //subgraph test
    let mut result_indixes: Vec<usize> = Vec::new();
    while result_indixes.len() < indexes.len() {
        let mut search_index: NodeIndex = node_index(0);
        for i in indexes.iter() {
            if result_indixes.binary_search(&i.index()).is_err() {
                search_index = *i;
                break;
            }
        }
        let dijkstra = dijkstra(&graph, search_index, None, |_| 1);
        let mut indices_for_graph: Vec<NodeIndex> = Vec::new();
        for (i, index) in dijkstra.iter() {
            result_indixes.push(i.index());
            indices_for_graph.push(*i);
        }
        indices_for_graph.sort();
        let mut sub_graph: StableUnGraph<u8, f64> = graph.clone();
        for i in indexes.iter() {
            if indices_for_graph.binary_search(&i).is_err() {
                sub_graph.remove_node(*i);
            }
        }
        subgraph_vector.push(sub_graph);
        result_indixes.sort();
    }
    return (graph, indexes, subgraph_vector);
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

    println!("connectivity_matrix {}", mol.connectivity_matrix);
    let (graph, indexes, subgraphs): (
        StableUnGraph<u8, f64>,
        Vec<NodeIndex>,
        Vec<StableUnGraph<u8, f64>>,
    ) = build_graph(
        &mol.atomic_numbers,
        &mol.connectivity_matrix,
        &mol.distance_matrix,
    );
    println!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));

    assert!(1 == 2);
}

#[test]
fn connectivity_dimer_routine() {
    // ethene dimer
    let atomic_numbers: Vec<u8> = vec![6, 6, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1];
    let mut positions: Array2<f64> = array![
        [-0.7575800000, 0.0000000000, -0.0000000000],
        [0.7575800000, 0.0000000000, 0.0000000000],
        [-1.2809200000, 0.9785000000, -0.0000000000],
        [-1.2809200000, -0.9785000000, 0.0000000000],
        [1.2809200000, -0.9785000000, -0.0000000000],
        [1.2809200000, 0.9785000000, 0.0000000000],
        [-0.7575800000, 0.0000000000, 4.0000000000],
        [0.7575800000, 0.0000000000, 4.0000000000],
        [-1.2809200000, 0.9785000000, 4.0000000000],
        [-1.2809200000, -0.9785000000, 4.0000000000],
        [1.2809200000, -0.9785000000, 4.0000000000],
        [1.2809200000, 0.9785000000, 4.0000000000]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule =
        Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    println!("connectivity_matrix {}", mol.connectivity_matrix);
    let (graph, indexes, subgraphs): (
        StableUnGraph<u8, f64>,
        Vec<NodeIndex>,
        Vec<StableUnGraph<u8, f64>>,
    ) = build_graph(
        &mol.atomic_numbers,
        &mol.connectivity_matrix,
        &mol.distance_matrix,
    );
    println!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));

    println!("len subgraphs {}", subgraphs.len());
    println!("subgraph 1");
    println!(
        "{:?}",
        Dot::with_config(&subgraphs[0], &[Config::EdgeNoLabel])
    );
    println!("subgraph 2");
    println!(
        "{:?}",
        Dot::with_config(&subgraphs[1], &[Config::EdgeNoLabel])
    );

    assert!(1 == 2);
}

#[test]
fn connectivity_benzene_dimer_routine() {
    // ethene dimer
    let atomic_numbers: Vec<u8> = vec![
        1, 1, 6, 6, 1, 6, 6, 1, 6, 6, 1, 1, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 1, 1,
    ];
    let mut positions: Array2<f64> = array![
        [2.4235700000, 2.9118100000, 0.7233500000],
        [-0.0586300000, 2.9447100000, 0.8501500000],
        [1.8795700000, 2.0141100000, 1.0340500000],
        [0.4903700000, 2.0326100000, 1.1049500000],
        [3.6703700000, 0.8405100000, 1.3020500000],
        [2.5772700000, 0.8550100000, 1.3578500000],
        [-0.2010300000, 0.8918100000, 1.4996500000],
        [-1.2942300000, 0.9062100000, 1.5553500000],
        [1.8857700000, -0.2857900000, 1.7525500000],
        [0.4965700000, -0.2673900000, 1.8234500000],
        [2.4347700000, -1.1979900000, 2.0073500000],
        [-0.0477300000, -1.1649900000, 2.1339500000],
        [-1.9508900000, 0.1533600000, -1.2834600000],
        [-0.8576900000, 0.1389600000, -1.3392600000],
        [-0.7152900000, 2.1918600000, -1.9887600000],
        [-0.1662900000, 1.2797600000, -1.7338600000],
        [-0.7043900000, -1.9178400000, -0.7048600000],
        [-0.1600900000, -1.0202400000, -1.0153600000],
        [1.2229100000, 1.2612600000, -1.8048600000],
        [1.2291100000, -1.0386400000, -1.0862600000],
        [1.7669100000, 2.1589600000, -2.1154600000],
        [1.9206100000, 0.1021600000, -1.4809600000],
        [1.7781100000, -1.9508400000, -0.8314600000],
        [3.0137100000, 0.0876600000, -1.5368600000]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule =
        Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    println!("connectivity_matrix {}", mol.connectivity_matrix);
    let (graph, indexes, subgraphs): (
        StableUnGraph<u8, f64>,
        Vec<NodeIndex>,
        Vec<StableUnGraph<u8, f64>>,
    ) = build_graph(
        &mol.atomic_numbers,
        &mol.connectivity_matrix,
        &mol.distance_matrix,
    );
    println!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));

    println!("len subgraphs {}", subgraphs.len());
    println!("subgraph 1");
    println!(
        "{:?}",
        Dot::with_config(&subgraphs[0], &[Config::EdgeNoLabel])
    );
    println!("subgraph 2");
    println!(
        "{:?}",
        Dot::with_config(&subgraphs[1], &[Config::EdgeNoLabel])
    );

    let tree: Graph<u8, f64> = Graph::from_elements(min_spanning_tree(&graph));
    println!("tree");
    println!("{:?}", Dot::with_config(&tree, &[Config::EdgeNoLabel]));

    assert!(1 == 2);
}
