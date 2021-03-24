use crate::parameters::*;
use crate::{constants, defaults};
use approx::AbsDiffEq;
use log::{debug, error, info, trace, warn};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::Norm;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone)]
pub struct Molecule {
    pub atomic_numbers: Option<Vec<u8>>,
    pub positions: Option<Array2<f64>>,
    pub repr: Option<String>,
    pub proximity_matrix: Option<Array2<bool>>,
    pub distance_matrix: Option<Array2<f64>>,
    pub directions_matrix: Option<Array3<f64>>,
    pub adjacency_matrix: Option<Array2<bool>>,
    electronic_structure: ElectronicStructure,
}

impl Molecule {
    pub fn new() -> Molecule {
        let mol = Molecule {
            atomic_numbers: None,
            positions: None,
            repr: None,
            proximity_matrix: None,
            distance_matrix: None,
            directions_matrix: None,
            adjacency_matrix: None,
            electronic_structure: ElectronicStructure::new(),
        };

        return mol;
    }

    // pub fn atomic_numbers(&self) -> Option<&[u8]> {
    //     match self.atomic_numbers {
    //         Some(ref x) => Some(x),
    //         None => None,
    //     }
    // }
    //
    // pub fn positions(&self) -> Option<ArrayView2<f64>> {
    //     match self.positions {
    //         Some(ref x) => Some(x.view()),
    //         None => None,
    //     }
    // }
    //
    // pub fn repr(&self) -> Option<&str> {
    //     match self.repr {
    //         Some(ref x) => Some(x),
    //         None => None,
    //     }
    // }
    //
    // pub fn proximity_matrix(&self) -> Option<ArrayView2<f64>> {
    //     match self.proximity_matrix {
    //         Some(ref x) => Some(x.view()),
    //         None => None,
    //     }
    // }
    //
    // pub fn distance_matrix(&self) -> Option<ArrayView2<f64>> {
    //     match self.distance_matrix {
    //         Some(ref x) => Some(x.view()),
    //         None => None,
    //     }
    // }
    //
    // pub fn directions_matrix(&self) -> Option<ArrayView3<f64>> {
    //     match self.directions_matrix {
    //         Some(ref x) => Some(x.view()),
    //         None => None,
    //     }
    // }
    //
    // pub fn adjacency_matrix(&self) -> Option<ArrayView2<f64>> {
    //     match self.adjacency_matrix {
    //         Some(ref x) => Some(x.view()),
    //         None => None,
    //     }
    // }

    pub fn set_atomic_numbers(&mut self, numbers: Option<&[u8]>) {
        self.atomic_numbers = match numbers {
            Some(x) => Some(Vec::from(x)),
            None => None,
        };
    }

    pub fn set_positions(&mut self, positions: Option<Array2<f64>>) {
        self.positions = match positions {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_repr(&mut self, repr: Option<String>) {
        self.repr = match repr {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_proximity_matrix(&mut self, pmatrix: Option<Array2<bool>>) {
        self.proximity_matrix = match pmatrix {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_distance_matrix(&mut self, dist_matrix: Option<Array2<f64>>) {
        self.distance_matrix = match dist_matrix {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_directions_matrix(&mut self, dir_matrix: Option<Array3<f64>>) {
        self.directions_matrix = match dir_matrix {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_adjacency_matrix(&mut self, adj_matrix: Option<Array2<bool>>) {
        self.adjacency_matrix = match adj_matrix {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn reset(&mut self) {
        self.positions = None;
        self.distance_matrix = None;
        self.directions_matrix = None;
        self.proximity_matrix = None;
        self.adjacency_matrix = None;
    }

    pub fn from_geometry(numbers: &[u8], coordinates: Array2<f64>) -> Molecule {
        let (dist_matrix, dir_matrix, prox_matrix, adj_matrix): (
            Array2<f64>,
            Array3<f64>,
            Array2<bool>,
            Array2<bool>,
        ) = build_geometric_matrices(coordinates.view(), None);
        let repr_string: String = create_smiles_repr(&numbers, coordinates.view());
        let mol = Molecule {
            atomic_numbers: Some(Vec::from(numbers)),
            positions: Some(coordinates),
            repr: Some(repr_string),
            proximity_matrix: Some(prox_matrix),
            distance_matrix: Some(dist_matrix),
            directions_matrix: Some(dir_matrix),
            adjacency_matrix: Some(adj_matrix),
            electronic_structure: ElectronicStructure::new(),
        };
    }

    pub fn update_geometry(&mut self, coordinates: Array2<f64>) {
        let (dist_matrix, dir_matrix, prox_matrix, adj_matrix): (
            Array2<f64>,
            Array3<f64>,
            Array2<bool>,
            Array2<bool>,
        ) = build_geometric_matrices(coordinates.view(), None);
        self.set_distance_matrix(Some(dist_matrix));
        self.set_directions_matrix(Some(dir_matrix));
        self.set_proximity_matrix(Some(prox_matrix));
        self.set_adjacency_matrix(Some(adj_matrix));
        self.set_positions(Some(coordinates));
    }
}

struct ElectronicStructure {
    pub h0: Option<Array2<f64>>,
    pub s: Option<Array2<f64>>,
    pub dq: Option<Array1<f64>>,
    pub p: Option<Array2<f64>>,
    pub gamma_atom_wise: Option<Array2<f64>>,
    pub gamma_ao_wise: Option<Array2<f64>>,
    pub gamma_lrc_atom_wise: Option<Array2<f64>>,
    pub gamma_lrc_ao_wise: Option<Array2<f64>>,
    pub gamma_atom_wise_grad: Option<Array3<f64>>,
    pub gamma_ao_wise_grad: Option<Array3<f64>>,
    pub gamma_lrc_atom_wise_grad: Option<Array3<f64>>,
    pub gamma_lrc_ao_wise_grad: Option<Array3<f64>>,
}

impl ElectronicStructure {

    pub fn new() -> ElectronicStructure {
        ElectronicStructure{
            h0: None,
            s: None,
            dq: None,
            p: None,
            gamma_atom_wise: None,
            gamma_ao_wise: None,
            gamma_lrc_atom_wise: None,
            gamma_lrc_ao_wise: None,
            gamma_atom_wise_grad: None,
            gamma_ao_wise_grad: None,
            gamma_lrc_atom_wise_grad: None,
            gamma_lrc_ao_wise_grad: None,
        }
    }

    pub fn set_h0(&mut self, h0: Option<Array2<f64>>) {
        self.h0 = match h0 {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_s(&mut self, overlap: Option<Array2<f64>>) {
        self.s = match overlap {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_dq(&mut self, dq: Option<Array1<f64>>) {
        self.dq = match dq {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_p(&mut self, p: Option<Array2<f64>>) {
        self.p = match p {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_gamma_atom_wise(&mut self, gamma_atom_wise: Option<Array2<f64>>) {
        self.gamma_atom_wise = match gamma_atom_wise {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_gamma_ao_wise(&mut self, gamma_ao_wise: Option<Array2<f64>>) {
        self.gamma_ao_wise = match gamma_ao_wise {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_gamma_lrc_atom_wise(&mut self, gamma_lrc_atom_wise: Option<Array2<f64>>) {
        self.gamma_lrc_atom_wise = match gamma_lrc_atom_wise {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_gamma_lrc_ao_wise(&mut self, gamma_lrc_atom_wise: Option<Array2<f64>>) {
        self.gamma_lrc_atom_wise = match gamma_lrc_atom_wise {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_gamma_atom_wise_grad(&mut self, gamma_atom_wise_grad: Option<Array3<f64>>) {
        self.gamma_atom_wise_grad = match gamma_atom_wise_grad{
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_gamma_ao_wise_grad(&mut self, set_gamma_ao_wise_grad: Option<Array3<f64>>) {
        self.set_gamma_ao_wise_grad = match set_gamma_ao_wise_grad {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_gamma_lrc_atom_wise_grad(&mut self, gamma_lrc_atom_wise_grad: Option<Array3<f64>>) {
        self.gamma_lrc_atom_wise_grad = match gamma_lrc_atom_wise_grad {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn set_gamma_lrc_ao_wise_grad(&mut self, gamma_lrc_ao_wise_grad: Option<Array3<f64>>) {
        self.gamma_lrc_ao_wise_grad = match gamma_lrc_ao_wise_grad {
            Some(x) => Some(x),
            None => None,
        };
    }

    pub fn reset(&mut self) {
        self.h0 = None;
        self.s = None;
        self.dq = None;
        self.p = None;
        self.gamma_atom_wise = None;
        self.gamma_ao_wise = None;
        self.gamma_lrc_atom_wise = None;
        self.gamma_lrc_ao_wise = None;
        self.gamma_atom_wise_grad = None;
        self.gamma_ao_wise_grad = None;
        self.gamma_lrc_atom_wise_grad = None;
        self.gamma_lrc_ao_wise_grad = None;
    }
}


fn create_smiles_repr(numbers: &[u8], coordinates: ArrayView2<f64>) -> String {
    return String::from("");
}

fn build_geometric_matrices(
    coordinates: ArrayView2<f64>,
    cutoff: Option<f64>,
) -> (Array2<f64>, Array3<f64>, Array2<bool>, Array2<bool>) {
    let cutoff: f64 = cutoff.unwrap_or(defaults::PROXIMITY_CUTOFF);
    let n_atoms: usize = coordinates.nrows();
    let mut dist_matrix: Array2<f64> = Array::zeros((n_atoms, n_atoms));
    let mut directions_matrix: Array3<f64> = Array::zeros((n_atoms, n_atoms, 3));
    let mut prox_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    let mut adj_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    for (i, pos_i) in coordinates.outer_iter().enumerate() {
        for (j0, pos_j) in coordinates.slice(s![i.., ..]).outer_iter().enumerate() {
            let j: usize = j0 + i;
            let r: Array1<f64> = &pos_i - &pos_j;
            let r_ij: f64 = r.norm();
            let r_cov: f64 = (constants::COVALENCE_RADII[&atomic_numbers[i[0]]]
                + constants::COVALENCE_RADII[&atomic_numbers[i[1]]]);
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
                if r_ij <= (1.3 * r_cov) {
                    adj_matrix[[i, j]] = true;
                    adj_matrix[[j, i]] = true;
                }
            }
        }
    }
    return (dist_matrix, directions_matrix, prox_matrix, adj_matrix);
}
