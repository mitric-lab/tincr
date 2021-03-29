use crate::initialization::parameters::*;
use crate::initialization::molecule::Molecule;
use crate::{constants, defaults};
use approx::AbsDiffEq;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::Norm;
use std::collections::HashMap;
use std::hash::Hash;
use crate::constants::BOND_THRESHOLD;

pub fn build_geometric_matrices(
    atomic_numbers: &[u8],
    coordinates: ArrayView2<f64>,
    cutoff: Option<f64>,
) -> (Array2<f64>, Array3<f64>, Array2<bool>) {
    let cutoff: f64 = cutoff.unwrap_or(defaults::PROXIMITY_CUTOFF);
    let n_atoms: usize = coordinates.nrows();
    let mut dist_matrix: Array2<f64> = Array::zeros((n_atoms, n_atoms));
    let mut directions_matrix: Array3<f64> = Array::zeros((n_atoms, n_atoms, 3));
    let mut prox_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    // let mut adj_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    for (i, (zi, posi)) in atomic_numbers.iter().zip(coordinates.outer_iter()).enumerate() {
        for (j0, (zj, posj)) in atomic_numbers.iter().zip(coordinates.slice(s![i.., ..]).outer_iter()).enumerate() {
            let j: usize = j0 + i;
            let r: Array1<f64> = &posi - &posj;
            let r_ij: f64 = r.norm();
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
                // if r_ij <= BOND_THRESHOLD[&(*zi, *zj)] {
                //     adj_matrix[[i, j]] = true;
                //     adj_matrix[[j, i]] = true;
                // }
            }
        }
    }
    return (dist_matrix, directions_matrix, prox_matrix);
}

// We only build the upper right and lower left block of the geometric matrices. The diagonal blocks
// are taken from the monomers.
pub fn build_geometric_matrices_from_monomers(
    coordinates: ArrayView2<f64>,
    m1: &Molecule,
    m2: &Molecule,
    cutoff: Option<f64>,
) -> (Array2<f64>, Array3<f64>, Array2<bool>) {
    let cutoff: f64 = cutoff.unwrap_or(defaults::PROXIMITY_CUTOFF);
    let n_at_1: usize = m1.atomic_numbers.unwrap().len();
    let n_at_2: usize = m2.atomic_numbers.unwrap().len();
    let n_atoms: usize = n_at_1 + n_at_2;
    let mut dist_matrix: Array2<f64> = Array::zeros((n_atoms, n_atoms));
    let mut directions_matrix: Array3<f64> = Array::zeros((n_atoms, n_atoms, 3));
    let mut prox_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    // let mut adj_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    // fill the upper left block with the matrices from the first monomer
    dist_matrix
        .slice_mut(s![0..n_at_1, 0..n_at_1])
        .assign(&m1.distance_matrix.unwrap());
    directions_matrix
        .slice_mut(s![0..n_at_1, 0..n_at_1, ..])
        .assign(&m1.directions_matrix.unwrap());
    prox_matrix
        .slice_mut(s![0..n_at_1, 0..n_at_1])
        .assign(&m1.proximity_matrix.unwrap());
    // adj_matrix
    //     .slice_mut(s![0..n_at_1, 0..n_at_1])
    //     .assign(&m1.adjacency_matrix.unwrap());
    // fill the lower right block with the matrices from the second monomer
    dist_matrix
        .slice_mut(s![n_at_1.., n_at_1..])
        .assign(&m2.distance_matrix.unwrap());
    directions_matrix
        .slice_mut(s![n_at_1.., n_at_1.., ..])
        .assign(&m2.directions_matrix.unwrap());
    prox_matrix
        .slice_mut(s![n_at_1.., n_at_1..])
        .assign(&m2.proximity_matrix.unwrap());
    // adj_matrix
    //     .slice_mut(s![n_at_1.., n_at_1..])
    //     .assign(&m2.adjacency_matrix.unwrap());

    for (i, (zi, posi)) in m1.atomic_numbers.unwrap().iter().zip(coordinates.slice(s![..n_at_1, ..]).outer_iter()).enumerate() {
        for (j0, (zj, posj)) in m2.atomic_numbers.unwrap().iter().zip(coordinates.slice(s![n_at_1.., ..]).outer_iter()).enumerate() {
            let j: usize = j0 + n_at_1;
            let r: Array1<f64> = &posi - &posj;
            let r_ij: f64 = r.norm();
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
                // if r_ij <= BOND_THRESHOLD[&(*zi, *zj)] {
                //     adj_matrix[[i, j]] = true;
                //     adj_matrix[[j, i]] = true;
                // }
            }
        }
    }
    return (dist_matrix, directions_matrix, prox_matrix);
}
