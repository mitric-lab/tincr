#![allow(clippy::if_same_then_else)]
use crate::scc::h0_and_s::h0_and_s;
use crate::initialization::parameters::*;
use crate::{constants, defaults};
use approx::AbsDiffEq;
use log::{debug, error, info, trace, warn};
use ndarray::prelude::*;
#[macro_use]
use ndarray::stack;
use ndarray_linalg::Norm;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone)]
pub struct ElectronicData {
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

impl ElectronicData {
    pub fn new() -> Self {
        ElectronicData {
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

    pub fn set_gamma_ao_wise_grad(&mut self, gamma_ao_wise_grad: Option<Array3<f64>>) {
        self.gamma_ao_wise_grad = gamma_ao_wise_grad;
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

    pub fn set_dq_from_monomers(&mut self, dq1: ArrayView1<f64>, dq2: ArrayView1<f64>) {
        self.set_dq(Some(stack![Axis(0), dq1, dq2]));
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
        let (f1, n_at_1, _): (usize, usize, usize) = g1_grad.dim();
        let (f2, n_at_2, _): (usize, usize, usize) = g2_grad.dim();
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
        let (f1, n_orb_1, _): (usize, usize, usize) = g1_grad.dim();
        let (f2, n_orb_2, _): (usize, usize, usize) = g2_grad.dim();
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
        let (f1, n_at_1, _): (usize, usize, usize) = g1_grad.dim();
        let (f2, n_at_2, _): (usize, usize, usize) = g2_grad.dim();
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
        let (f1, n_orb_1, _): (usize, usize, usize) = g1_grad.dim();
        let (f2, n_orb_2, _): (usize, usize, usize) = g2_grad.dim();
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
        e1: &ElectronicData,
        e2: &ElectronicData,
    ) -> Self {
        let mut es: ElectronicData = ElectronicData::new();
        if e1.is_some() && e2.is_some() {
            match (&e1.h0, &e2.h0) {
                (Some(x), Some(y)) => es.set_h0_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.s, &e2.s) {
                (Some(x), Some(y)) => es.set_overlap_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.dq, &e2.dq) {
                (Some(x), Some(y)) => es.set_dq_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.p, &e2.p) {
                (Some(x), Some(y)) => es.set_density_matrix_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.gamma_atom_wise, &e2.gamma_atom_wise) {
                (Some(x), Some(y)) => es.set_gamma_atom_wise_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.gamma_ao_wise, &e2.gamma_ao_wise) {
                (Some(x), Some(y)) => es.set_gamma_ao_wise_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.gamma_lrc_atom_wise, &e2.gamma_lrc_atom_wise) {
                (Some(x), Some(y)) => es.set_gamma_lrc_atom_wise_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.gamma_lrc_ao_wise, &e2.gamma_lrc_ao_wise) {
                (Some(x), Some(y)) => es.set_gamma_lrc_ao_wise_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.gamma_atom_wise_grad, &e2.gamma_atom_wise_grad) {
                (Some(x), Some(y)) => es.set_gamma_atom_wise_grad_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.gamma_ao_wise_grad, &e2.gamma_ao_wise_grad) {
                (Some(x), Some(y)) => es.set_gamma_ao_wise_grad_from_monomers(x.view(), y.view()),
                _ => (),
            };
            match (&e1.gamma_lrc_atom_wise_grad, &e2.gamma_lrc_atom_wise_grad) {
                (Some(x), Some(y)) => {
                    es.set_gamma_lrc_atom_wise_grad_from_monomers(x.view(), y.view())
                }
                _ => (),
            };
            match (&e1.gamma_lrc_ao_wise_grad, &e2.gamma_lrc_ao_wise_grad) {
                (Some(x), Some(y)) => {
                    es.set_gamma_lrc_ao_wise_grad_from_monomers(x.view(), y.view())
                }
                _ => (),
            };
        }
        return es;
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
