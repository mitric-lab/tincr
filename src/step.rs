use crate::defaults;
use crate::gradients;
use crate::gradients::get_gradients;
use crate::internal_coordinates::*;
use crate::scc_routine;
use crate::solver::get_exc_energies;
use crate::Molecule;
use approx::AbsDiffEq;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use std::ops::Deref;

pub fn get_delta_prime(
    v0: f64,
    cart_coords: Array1<f64>,
    gradient_ic: Array1<f64>,
    hessian_ic: Array2<f64>,
    internal_coords: &InternalCoordinates,
    bool_rfo: bool,
) {
    // Return the internal coordinate step given a parameter "v".
    // "v" refers to the multiple of the identity added to the Hessian
    // in trust-radius Newton Raphson (TRM), and the multiple of the
    // identity on the RHS matrix in rational function optimization (RFO).
    // Note that reasonable default values are v = 0.0 in TRM and 1.0 in RFO.

    if bool_rfo {
        // do rfo
    } else {
        // do trust radius Newton-Raphson
        // get_delta_prime_trm(v,coords,grad,hess,internal_coords)
    }
}

pub fn get_delta_prime_trm(
    v0: f64,
    cart_coords: Array1<f64>,
    gradient_ic: Array1<f64>,
    hessian_ic: Array2<f64>,
    internal_coords: &InternalCoordinates,
) -> (Array1<f64>, f64, f64) {
    // Returns the Newton-Raphson step given a multiple of the diagonal
    // added to the Hessian, the expected decrease in the energy, and
    // the derivative of the step length w/r.t. v.

    // if constraints use method
    let gc: Array1<f64> = gradient_ic.clone();
    let hc: Array2<f64> = hessian_ic.clone();

    let mut ht: Array2<f64> = hc.clone() + v0 * Array::eye(hc.dim().0);
    // invert svd for ht
    let h_i: Array2<f64> = invert_svd(ht);

    let dy: Array1<f64> = -1.0 * h_i.dot(&gc.t());
    let d_prime: Array1<f64> = -1.0 * h_i.dot(&dy.t());
    let dy_prime: f64 = dy.dot(&d_prime) / dy.clone().to_vec().norm();
    let sol: f64 = 0.5 * dy.dot(&hessian_ic.dot(&dy)) + dy.dot(&gradient_ic);

    return (dy, sol, dy_prime);
}

pub fn invert_svd(mat: Array2<f64>) -> Array2<f64> {
    let (u, s, vh) = mat.svd(true, true).unwrap();
    let ut: Array2<f64> = u.unwrap().reversed_axes();
    let s: Array1<f64> = s;
    let v: Array2<f64> = vh.unwrap().reversed_axes();

    let mut s_inv: Array1<f64> = Array::zeros((s.dim()));

    for (ival, value) in s.iter().enumerate() {
        if value.abs() > 1.0e-12 {
            s_inv[ival] = 1.0 / value;
        }
    }
    let s_inv_2d: Array2<f64> = Array::from_diag(&s_inv);
    let inv: Array2<f64> = v.dot(&s_inv_2d.dot(&ut));

    return inv;
}
