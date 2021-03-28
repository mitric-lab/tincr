use crate::calculator::get_gamma_matrix;
use crate::h0_and_s::h0_and_s;
use crate::initialization::parameters::*;
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
        Molecule {
            atomic_numbers: None,
            positions: None,
            repr: None,
            proximity_matrix: None,
            distance_matrix: None,
            directions_matrix: None,
            adjacency_matrix: None,
            electronic_structure: ElectronicStructure::new(),
        }
    }

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
        if self.electronic_structure.is_some() {
            self.electronic_structure.reset()
        }
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

    pub fn dimer_from_monomers(m1: &Molecule, m2: &Molecule) -> Molecule {
        let numbers: Vec<u8> = [&m1.atomic_numbers, &m2.atomic_numbers].concat();
        let repr: String = [&m1.repr, &m2.repr].concat();
        let positions: Array2<f64> =
            concatenate![Axis(0), m1.positions.view(), m2.positions.view()];
        let (dist_matrix, dir_matrix, prox_matrix, adj_matrix): (
            Array2<f64>,
            Array3<f64>,
            Array2<bool>,
            Array2<bool>,
        ) = build_geometric_matrices_from_monomers(positions.view(), &m1, &m2, None);
        let es: ElectronicStructure = ElectronicStructure::dimer_from_monomers(
            &m1.electronic_structure,
            &m2.electronic_structure,
        );
        Molecule {
            atomic_numbers: Some(numbers),
            positions: Some(positions),
            repr: Some(repr),
            proximity_matrix: Some(prox_matrix),
            distance_matrix: Some(dist_matrix),
            directions_matrix: Some(dir_matrix),
            adjacency_matrix: Some(adj_matrix),
            electronic_structure: es,
        }
    }

    pub fn make_scc_ready(
        &mut self,
        n_orbs: usize,
        valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
        skt: &HashMap<(u8, u8), SlaterKosterTable>,
        orbital_energies: &HashMap<u8, HashMap<(i8, i8), f64>>,
        hubbard_u: &HashMap<u8, f64>,
        r_lr: Option<f64>,
    ) {
        // H0, S, gamma, gamma_LRC is needed for SCC routine
        // get H0 and overlap matrix
        let (h0, s): (Array2<f64>, Array2<f64>) = h0_and_s(
            &self.atomic_numbers.unwrap(),
            self.positions.unwrap().view(),
            n_orbs,
            valorbs,
            self.proximity_matrix.unwrap().view(),
            skt,
            orbital_energies,
        );
        self.electronic_structure.set_h0(Some(h0));
        self.electronic_structure.set_s(Some(s));
        // get gamma
        let (gm, gm_ao): (Array2<f64>, Array2<f64>) = get_gamma_matrix(
            &self.atomic_numbers.unwrap(),
            self.atomic_numbers.unwrap().len(),
            n_orbs,
            self.distance_matrix.unwrap().view(),
            &hubbard_u,
            &valorbs,
            Some(0.0),
        );
        self.electronic_structure.set_gamma_atom_wise(Some(gm));
        self.electronic_structure.set_gamma_ao_wise(Some(gm_ao));
        if r_lr.is_some() {
            // get gamma lrc
            let (gm_lrc, gm_lrc_ao): (Array2<f64>, Array2<f64>) = get_gamma_matrix(
                &self.atomic_numbers.unwrap(),
                self.atomic_numbers.unwrap().len(),
                n_orbs,
                self.distance_matrix.unwrap().view(),
                &hubbard_u,
                &valorbs,
                r_lr,
            );
            self.electronic_structure.set_gamma_atom_wise(Some(gm_lrc));
            self.electronic_structure.set_gamma_ao_wise(Some(gm_lrc_ao));
        }
    }

    pub fn make_dimer_scc_ready(
        &mut self,
        n_orbs: usize,
        valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
        skt: &HashMap<(u8, u8), SlaterKosterTable>,
        orbital_energies: &HashMap<u8, HashMap<(i8, i8), f64>>,
        hubbard_u: &HashMap<u8, f64>,
        r_lr: Option<f64>) {

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
        ElectronicStructure {
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
        self.gamma_atom_wise_grad = match gamma_atom_wise_grad {
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

    pub fn set_dq_to_zeros(&mut self, n_atoms: usize) {
        self.set_dq(Some(Array::zeros([n_atoms])));
    }

    pub fn set_h0_from_monomers(&mut self, h0_1: ArrayView2<f64>, h0_2: ArrayView2<f64>) {
        let n_orb_1: usize = h0_1.ncols();
        let n_orb_2: usize = h0_2.ncols();
        let n_orbs: usize = n_orb_1 + n_orb_2;
        let mut h0: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
        h0.slice_mut(s![..n_orb_1, ..n_orb_1]).assign(&h0_1);
        h0.slice_mut(s![n_orb_1.., n_orb_1..]).assign(&h0_2);
        self.set_h0(Some(h0));
    }

    pub fn set_overlap_from_monomers(&mut self, s1: ArrayView2<f64>, s2: ArrayView2<f64>) {
        let n_orb_1: usize = s1.ncols();
        let n_orb_2: usize = s2.ncols();
        let n_orbs: usize = n_orb_1 + n_orb_2;
        let mut overlap: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
        overlap.slice_mut(s![..n_orb_1, ..n_orb_1]).assign(&s1);
        overlap.slice_mut(s![n_orb_1.., n_orb_1..]).assign(&s2);
        self.set_s(Some(overlap));
    }

    pub fn set_dq_from_monomers(&mut self, dq1: &ArrayView1<f64>, dq2: &ArrayView1<f64>) {
        self.set_dq(Some(concatenate![Axis(0), &dq1, &dq2]));
    }

    pub fn set_density_matrix_from_monomers(&mut self, p1: ArrayView2<f64>, p2: ArrayView2<f64>) {
        let n_orb_1: usize = p1.ncols();
        let n_orb_2: usize = p2.ncols();
        let n_orbs: usize = n_orb_1 + n_orb_2;
        let mut density_matrix: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
        density_matrix
            .slice_mut(s![..n_orb_1, ..n_orb_1])
            .assign(&p1);
        density_matrix
            .slice_mut(s![n_orb_1.., n_orb_1..])
            .assign(&p2);
        self.set_p(Some(density_matrix));
    }

    pub fn set_gamma_atom_wise_from_monomers(&mut self, g1: ArrayView2<f64>, g2: ArrayView2<f64>) {
        let n_at_1: usize = g1.ncols();
        let n_at_2: usize = g2.ncols();
        let n_atoms: usize = n_at_1 + n_at_2;
        let mut gamma_atom_wise: Array2<f64> = Array2::zeros([n_atoms, n_atoms]);
        gamma_atom_wise
            .slice_mut(s![..n_at_1, ..n_at_1])
            .assign(&g1);
        gamma_atom_wise
            .slice_mut(s![n_at_1.., n_at_1..])
            .assign(&g2);
        self.set_gamma_atom_wise(Some(gamma_atom_wise));
    }

    pub fn set_gamma_ao_wise_from_monomers(&mut self, g1: ArrayView2<f64>, g2: ArrayView2<f64>) {
        let n_orb_1: usize = g1.ncols();
        let n_orb_2: usize = g2.ncols();
        let n_orbs: usize = n_orb_1 + n_orb_2;
        let mut gamma_ao_wise: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
        gamma_ao_wise
            .slice_mut(s![..n_orb_1, ..n_orb_1])
            .assign(&g1);
        gamma_ao_wise
            .slice_mut(s![n_orb_1.., n_orb_1..])
            .assign(&g2);
        self.set_gamma_ao_wise(Some(gamma_ao_wise));
    }

    pub fn set_gamma_lrc_atom_wise_from_monomers(
        &mut self,
        g1: ArrayView2<f64>,
        g2: ArrayView2<f64>,
    ) {
        let n_at_1: usize = g1.ncols();
        let n_at_2: usize = g2.ncols();
        let n_atoms: usize = n_at_1 + n_at_2;
        let mut gamma_lrc_atom_wise: Array2<f64> = Array2::zeros([n_atoms, n_atoms]);
        gamma_lrc_atom_wise
            .slice_mut(s![..n_at_1, ..n_at_1])
            .assign(&g1);
        gamma_lrc_atom_wise
            .slice_mut(s![n_at_1.., n_at_1..])
            .assign(&g2);
        self.set_gamma_lrc_atom_wise(Some(gamma_lrc_atom_wise));
    }

    pub fn set_gamma_lrc_ao_wise_from_monomers(
        &mut self,
        g1: ArrayView2<f64>,
        g2: ArrayView2<f64>,
    ) {
        let n_orb_1: usize = g1.ncols();
        let n_orb_2: usize = g2.ncols();
        let n_orbs: usize = n_orb_1 + n_orb_2;
        let mut gamma_lrc_ao_wise: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
        gamma_lrc_ao_wise
            .slice_mut(s![..n_orb_1, ..n_orb_1])
            .assign(&g1);
        gamma_lrc_ao_wise
            .slice_mut(s![n_orb_1.., n_orb_1..])
            .assign(&g2);
        self.set_gamma_lrc_ao_wise(Some(gamma_lrc_ao_wise));
    }

    pub fn set_gamma_atom_wise_grad_from_monomers(
        &mut self,
        g1_grad: ArrayView3<f64>,
        g2_grad: ArrayView3<f64>,
    ) {
        let (f1, n_at_1, _) = (usize, usize, usize) = g1_grad.dim();
        let (f2, n_at_2, _) = (usize, usize, usize) = g2_grad.dim();
        let n_atoms: usize = n_at_1 + n_at_2;
        let mut gamma_atom_wise_grad: Array3<f64> = Array3::zeros([f1 + f2, n_atoms, n_atoms]);
        gamma_atom_wise_grad
            .slice_mut(s![..f1, ..n_at_1, ..n_at_1])
            .assign(&g1_grad);
        gamma_atom_wise_grad
            .slice_mut(s![f1.., n_at_1.., n_at_1..])
            .assign(&g2_grad);
        self.set_gamma_atom_wise_grad(Some(gamma_atom_wise_grad));
    }

    pub fn set_gamma_ao_wise_grad_from_monomers(
        &mut self,
        g1_grad: ArrayView3<f64>,
        g2_grad: ArrayView3<f64>,
    ) {
        let (f1, n_orb_1, _) = (usize, usize, usize) = g1_grad.dim();
        let (f2, n_orb_2, _) = (usize, usize, usize) = g2_grad.dim();
        let n_orbs: usize = n_orb_1 + n_orb_2;
        let mut gamma_ao_wise_grad: Array3<f64> = Array3::zeros([f1 + f2, n_orbs, n_orbs]);
        gamma_ao_wise_grad
            .slice_mut(s![..f1, ..n_orb_1, ..n_orb_1])
            .assign(&g1_grad);
        gamma_ao_wise_grad
            .slice_mut(s![f1.., n_orb_1.., n_orb_1..])
            .assign(&g2_grad);
        self.set_gamma_ao_wise_grad(Some(gamma_ao_wise_grad));
    }

    pub fn set_gamma_lrc_atom_wise_grad_from_monomers(
        &mut self,
        g1_grad: ArrayView3<f64>,
        g2_grad: ArrayView3<f64>,
    ) {
        let (f1, n_at_1, _) = (usize, usize, usize) = g1_grad.dim();
        let (f2, n_at_2, _) = (usize, usize, usize) = g2_grad.dim();
        let n_atoms: usize = n_at_1 + n_at_2;
        let mut gamma_lrc_atom_wise_grad: Array3<f64> = Array3::zeros([f1 + f2, n_atoms, n_atoms]);
        gamma_lrc_atom_wise_grad
            .slice_mut(s![..f1, ..n_at_1, ..n_at_1])
            .assign(&g1_grad);
        gamma_lrc_atom_wise_grad
            .slice_mut(s![f1.., n_at_1.., n_at_1..])
            .assign(&g2_grad);
        self.set_gamma_lrc_atom_wise_grad(Some(gamma_lrc_atom_wise_grad));
    }

    pub fn set_gamma_lrc_ao_wise_grad_from_monomers(
        &mut self,
        g1_grad: ArrayView3<f64>,
        g2_grad: ArrayView3<f64>,
    ) {
        let (f1, n_orb_1, _) = (usize, usize, usize) = g1_grad.dim();
        let (f2, n_orb_2, _) = (usize, usize, usize) = g2_grad.dim();
        let n_orbs: usize = n_orb_1 + n_orb_2;
        let mut gamma_lrc_ao_wise_grad: Array3<f64> = Array3::zeros([f1 + f2, n_orbs, n_orbs]);
        gamma_lrc_ao_wise_grad
            .slice_mut(s![..f1, ..n_orb_1, ..n_orb_1])
            .assign(&g1_grad);
        gamma_lrc_ao_wise_grad
            .slice_mut(s![f1.., n_orb_1.., n_orb_1..])
            .assign(&g2_grad);
        self.set_gamma_lrc_ao_wise_grad(Some(gamma_lrc_ao_wise_grad));
    }

    pub fn dimer_from_monomers(
        e1: &ElectronicStructure,
        e2: &ElectronicStructure,
    ) -> ElectronicStructure {
        let mut es: ElectronicStructure = ElectronicStructure::new();
        if e1.is_some() && e2.is_some() {
            match (&e1.h0, &e2.h0) {
                (Some(x), Some(y)) => es.set_h0_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.s, &e2.s) {
                (Some(x), Some(y)) => es.set_overlap_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.dq, &e2.dq) {
                (Some(x), Some(y)) => es.set_dq_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.p, &e2.p) {
                (Some(x), Some(y)) => es.set_density_matrix_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.gamma_atom_wise, &e2.gamma_atom_wise) {
                (Some(x), Some(y)) => es.set_gamma_atom_wise_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.gamma_ao_wise, &e2.gamma_ao_wise) {
                (Some(x), Some(y)) => es.set_gamma_ao_wise_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.gamma_lrc_atom_wise, &e2.gamma_lrc_atom_wise) {
                (Some(x), Some(y)) => es.set_gamma_lrc_atom_wise_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.gamma_lrc_ao_wise, &e2.gamma_lrc_ao_wise) {
                (Some(x), Some(y)) => es.set_gamma_lrc_ao_wise_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.gamma_atom_wise_grad, &e2.gamma_atom_wise_grad) {
                (Some(x), Some(y)) => es.set_gamma_atom_wise_grad_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.gamma_ao_wise_grad, &e2.gamma_ao_wise_grad) {
                (Some(x), Some(y)) => es.set_gamma_ao_wise_grad_from_monomers(x.view(), y.view()),
                _ => None,
            };
            match (&e1.gamma_lrc_atom_wise_grad, &e2.gamma_lrc_atom_wise_grad) {
                (Some(x), Some(y)) => {
                    es.set_gamma_lrc_atom_wise_grad_from_monomers(x.view(), y.view())
                }
                _ => None,
            };
            match (&e1.gamma_lrc_ao_wise_grad, &e2.gamma_lrc_ao_wise_grad) {
                (Some(x), Some(y)) => {
                    es.set_gamma_lrc_ao_wise_grad_from_monomers(x.view(), y.view())
                }
                _ => None,
            };
        }
        return el;
    }

    pub fn reset(&mut self) {
        self.h0 = None;
        self.s = None;
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

    pub fn is_some(&self) -> bool {
        if self.h0.is_some() {
            true
        } else if self.s.is_some() {
            true
        } else if self.dq.is_some() {
            true
        } else if self.p.is_some() {
            true
        } else if self.gamma_atom_wise.is_some() {
            true
        } else if self.gamma_ao_wise.is_some() {
            true
        } else if self.gamma_lrc_atom_wise.is_some() {
            true
        } else if self.gamma_lrc_ao_wise.is_some() {
            true
        } else if self.gamma_atom_wise_grad.is_some() {
            true
        } else if self.gamma_ao_wise_grad.is_some() {
            true
        } else if self.gamma_lrc_atom_wise_grad.is_some() {
            true
        } else if self.gamma_lrc_ao_wise_grad.is_some() {
            true
        } else {
            false
        }
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

// We only build the upper right and lower left block of the geometric matrices. The diagonal blocks
// are taken from the monomers.
fn build_geometric_matrices_from_monomers(
    coordinates: ArrayView2<f64>,
    m1: &Molecule,
    m2: &Molecule,
    cutoff: Option<f64>,
) -> (Array2<f64>, Array3<f64>, Array2<bool>, Array2<bool>) {
    let cutoff: f64 = cutoff.unwrap_or(defaults::PROXIMITY_CUTOFF);
    let n_at_1: usize = m1.atomic_numbers.len();
    let n_at_2: usize = m2.atomic_numbers.len();
    let n_atoms: usize = n_at_1 + n_at_2;
    let mut dist_matrix: Array2<f64> = Array::zeros((n_atoms, n_atoms));
    let mut directions_matrix: Array3<f64> = Array::zeros((n_atoms, n_atoms, 3));
    let mut prox_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    let mut adj_matrix: Array2<bool> = Array::from_elem((n_atoms, n_atoms), false);
    // fill the upper left block with the matrices from the first monomer
    dist_matrix
        .slice_mut(s![0..n_at_1, 0..n_at_1])
        .assign(m1.distance_matrix.view());
    directions_matrix
        .slice_mut(s![0..n_at_1, 0..n_at_1, ..])
        .assign(m1.directions_matrix.view());
    prox_matrix
        .slice_mut(s![0..n_at_1, 0..n_at_1])
        .assign(m1.proximity_matrix.view());
    adj_matrix
        .slice_mut(s![0..n_at_1, 0..n_at_1])
        .assign(m1.distance_matrix.view());
    // fill the lower right block with the matrices from the second monomer
    dist_matrix
        .slice_mut(s![n_at_1.., n_at_1..])
        .assign(m2.distance_matrix.view());
    directions_matrix
        .slice_mut(s![n_at_1.., n_at_1.., ..])
        .assign(m2.directions_matrix.view());
    prox_matrix
        .slice_mut(s![n_at_1.., n_at_1..])
        .assign(m2.proximity_matrix.view());
    adj_matrix
        .slice_mut(s![n_at_1.., n_at_1..])
        .assign(m2.distance_matrix.view());

    for (i, pos_i) in coordinates.slice(s![..n_at_1, ..]).outer_iter().enumerate() {
        for (j0, pos_j) in coordinates.slice(s![n_at_1.., ..]).outer_iter().enumerate() {
            let j: usize = j0 + n_at_1;
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
