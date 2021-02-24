use crate::constants;
use crate::defaults;
use crate::gradients;
use crate::molecule::distance_matrix;
use crate::Molecule;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::{Data, indices};
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use petgraph::graph::*;
use petgraph::stable_graph::*;
use std::collections::HashMap;
use petgraph::dot::{Dot, Config};
use petgraph::data::*;
use petgraph::algo::*;

pub fn build_connectivity_matrix(n_atoms:usize,distance_matrix:&Array2<f64>,atomic_numbers:&Vec<u8>) -> (Array2<bool>) {
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

pub fn build_graph(mol: &Molecule)->(StableUnGraph<u8, f64>,Vec<NodeIndex>,Vec<StableUnGraph<u8, f64>>) {
    let mut graph: StableUnGraph<u8, f64> = StableUnGraph::<u8, f64>::default();
    for (i,z_i) in mol.atomic_numbers.iter().enumerate(){
        //let mut hash:HashMap<u8,Array1<f64>> = HashMap::new();
        //hash.insert(*z_i,mol.positions.slice(s![i,..]).to_owned());
        graph.add_node(*z_i);
    }
    let indexes:Vec<NodeIndex> = graph.node_indices().collect::<Vec<_>>();
    for (i,index) in mol.connectivity_matrix.outer_iter().enumerate(){
        for (j,ind) in index.iter().enumerate(){
            if mol.connectivity_matrix[[i,j]]{
                graph.add_edge(indexes[i],indexes[j],mol.distance_matrix[[i,j]]);
            }
        }
    }
    let mut subgraph_vector:Vec<StableUnGraph<u8, f64>> = Vec::new();
        //subgraph test
    let mut result_indixes: Vec<usize> = Vec::new();
    while result_indixes.len() < indexes.len(){
        let mut search_index:NodeIndex = node_index(0);
        for i in indexes.iter(){
            println!("index {}",i.index());
            if result_indixes.binary_search(&i.index()).is_err(){
                search_index = *i;
                break
            }
        }
        println!("search index {}", search_index.index());
        let dijkstra = dijkstra(&graph,search_index,None,|_| 1);
        println!("hash dijkstra {:?}",dijkstra);
        let mut indices_for_graph: Vec<NodeIndex> = Vec::new();
        for (i, index) in dijkstra.iter(){
            result_indixes.push(i.index());
            indices_for_graph.push(*i);
        }
        indices_for_graph.sort();
        let mut sub_graph: StableUnGraph<u8, f64> = graph.clone();
        for i in indexes.iter(){
            if indices_for_graph.binary_search(&i).is_err(){
                sub_graph.remove_node(*i);
            }
        }
        subgraph_vector.push(sub_graph);
        result_indixes.sort();
        println!("results indeces {:?}",result_indixes);
    }

    return (graph,indexes,subgraph_vector);
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
    let (graph,indexes,tree,subgraphs):(StableUnGraph<u8, f64>,Vec<NodeIndex>,StableUnGraph<u8, f64>,Vec<StableUnGraph<u8, f64>>) = build_graph(&mol);
    println!("{:?}",Dot::with_config(&graph,&[Config::EdgeNoLabel]));
    println!("tree");
    println!("{:?}",Dot::with_config(&tree,&[Config::EdgeNoLabel]));

    assert!(1 == 2);
}

#[test]
fn connectivity_dimer_routine(){
    let atomic_numbers: Vec<u8> = vec![6,6,1,1,1,1,6,6,1,1,1,1];
    let mut positions: Array2<f64> = array![
        [-0.7575800000,   0.0000000000,    -0.0000000000],
        [0.7575800000,    0.0000000000,    0.0000000000],
        [-1.2809200000,   0.9785000000,    -0.0000000000],
        [-1.2809200000,   -0.9785000000,   0.0000000000],
        [1.2809200000,    -0.9785000000,   -0.0000000000],
        [1.2809200000,    0.9785000000,    0.0000000000],
        [-0.7575800000,   0.0000000000,    4.0000000000],
        [0.7575800000,    0.0000000000,    4.0000000000],
        [-1.2809200000,   0.9785000000,    4.0000000000],
        [-1.2809200000,   -0.9785000000,   4.0000000000],
        [1.2809200000,    -0.9785000000,   4.0000000000],
        [1.2809200000,    0.9785000000,    4.0000000000]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule =
        Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    println!("connectivity_matrix {}", mol.connectivity_matrix);
    let (graph,indexes,tree,subgraphs):(StableUnGraph<u8, f64>,Vec<NodeIndex>,StableUnGraph<u8, f64>,Vec<StableUnGraph<u8, f64>>) = build_graph(&mol);
    println!("{:?}",Dot::with_config(&graph,&[Config::EdgeNoLabel]));
    println!("tree");
    println!("{:?}",Dot::with_config(&tree,&[Config::EdgeNoLabel]));

    println!("len subgraphs {}",subgraphs.len());
    println!("subgraph 1");
    println!("{:?}",Dot::with_config(&subgraphs[0],&[Config::EdgeNoLabel]));
    println!("subgraph 2");
    println!("{:?}",Dot::with_config(&subgraphs[1],&[Config::EdgeNoLabel]));


    assert!(1 == 2);

}
