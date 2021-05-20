


use crate::fmo::{atomvec_to_aomat, Monomer, Pair, SuperSystem};
use crate::initialization::parameters::RepulsivePotential;
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{
    gamma_ao_wise, gamma_atomwise, gamma_atomwise_ab, gamma_gradients_atomwise,
};
use crate::scc::h0_and_s::{h0_and_s, h0_and_s_ab, h0_and_s_gradients};
use crate::scc::mixer::{BroydenMixer, Mixer};
use crate::scc::mulliken::mulliken;
use crate::scc::scc_routine::{RestrictedSCC, SCCError};
use crate::scc::{
    construct_h1, density_matrix, density_matrix_ref, get_electronic_energy, get_repulsive_energy,
    lc_exact_exchange,
};
use crate::utils::Timer;
use approx::AbsDiffEq;
use log::info;
use nalgebra::Vector3;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::prelude::*;
use ndarray::stack;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use ndarray_stats::QuantileExt;
use nshare::ToNdarray1;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use std::iter::FromIterator;
use std::ops::{AddAssign, SubAssign};

mod monomer;
mod pair;

pub use monomer::*;
pub use pair::*;

pub trait GroundStateGradient {
    fn get_grad_dq(&self, grad_s: ArrayView3<f64>, p: ArrayView2<f64>) -> Array2<f64>;
    fn scc_gradient(&mut self) -> Array1<f64>;

}

//  Compute the gradient of the repulsive potential
//  Parameters:
//  ===========
//  atomlist: list of tuples (Zi, [xi,yi,zi]) for each atom
//  distances: matrix with distances between atoms, distance[i,j]
//    is the distance between atoms i and j
//  directions: directions[i,j,:] is the unit vector pointing from
//    atom j to atom i
//  VREP: dictionary, VREP[(Zi,Zj)] has to be an instance of RepulsivePotential
//    for the atom pair Zi-Zj
fn gradient_v_rep(atoms: &[Atom], v_rep: &RepulsivePotential) -> Array1<f64> {
    let n_atoms: usize = atoms.len();
    let mut grad: Array1<f64> = Array1::zeros([3 * n_atoms]);
    for (i, atomi) in atoms.iter().enumerate() {
        let mut grad_i: Array1<f64> = Array::zeros([3]);
        for (j, atomj) in atoms.iter().enumerate() {
            if i != j {
                let mut r: Vector3<f64> = atomi - atomj;
                let r_ij: f64 = r.norm();
                r /= r_ij;
                let v_ij_deriv: f64 = v_rep.get(atomi.kind, atomj.kind).spline_deriv(r_ij);
                r *= v_ij_deriv;
                // let v: ArrayView1<f64> = unsafe {
                //     ArrayView1::from_shape_ptr(
                //         (r.shape().0, ).strides((r.strides().0, )),
                //         r.as_ptr(),
                //     )
                // };
                let v = Array1::from_iter(r.iter());
                grad_i = &grad_i + &v;
            }
        }
        grad.slice_mut(s![i * 3..i * 3 + 3]).assign(&grad_i);
    }
    return grad;
}