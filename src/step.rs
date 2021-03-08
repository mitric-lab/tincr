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

pub fn get_delta_prime(v0:f64,cart_coords:Array1<f64>,gradient_ic:Array2<f64>,hessian_ic:Array2<f64>,internal_coords:&InternalCoordinates,bool_rfo:bool){
    // Return the internal coordinate step given a parameter "v".
    // "v" refers to the multiple of the identity added to the Hessian
    // in trust-radius Newton Raphson (TRM), and the multiple of the
    // identity on the RHS matrix in rational function optimization (RFO).
    // Note that reasonable default values are v = 0.0 in TRM and 1.0 in RFO.

    if bool_rfo{
        // do rfo
    }
    else{
        // do trust radius Newton-Raphson
        // get_delta_prime_trm(v,coords,grad,hess,internal_coords)
    }

}

pub fn get_delta_prime_trm(v0:f64,cart_coords:Array1<f64>,gradient_ic:Array2<f64>,hessian_ic:Array2<f64>,internal_coords:&InternalCoordinates){
    // Returns the Newton-Raphson step given a multiple of the diagonal
    // added to the Hessian, the expected decrease in the energy, and
    // the derivative of the step length w/r.t. v.

}