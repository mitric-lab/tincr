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

pub fn calc_drms_dmax(x_new:Array1<f64>,x_old:Array1<f64>)->(f64,f64){
    // Align and calculate the RMSD for two geometries
    let n_at: usize = x_old.len() / 3;
    let mut coords_new: Array2<f64> = x_new.into_shape((n_at, 3)).unwrap();
    let mut coords_old: Array2<f64> = x_old.into_shape((n_at, 3)).unwrap();
    coords_old = coords_old.clone() - coords_old.mean_axis(Axis(0)).unwrap();
    coords_new = coords_new.clone() - coords_new.mean_axis(Axis(0)).unwrap();
    // Obtain the rotation
    let u:Array2<f64> = get_rot(coords_new,coords_old);
    let mut x_rot:Array2<f64> = u.dot(&coords_new.t());
    x_rot = x_rot.reversed_axes();
    let x_rot_new:Array1<f64> = x_rot.into_shape((x_rot.dim().0*x_rot.dim().1)).unwrap();
    let x_old_new:Array1<f64> = coords_old.into_shape(x_old.len()).unwrap();
    let difference_arr:Array2<f64> = (x_rot_new-x_old_new).into_shape((n_at, 3)).unwrap();
    let displacement:Array1<f64> = difference_arr.mapv(|val|val.powi(2)).sum_axis(Axis(1)).mapv(|val|val.sqrt());
    let rms_displacement:f64 = displacement.mapv(|val|val.powi(2)).mean().unwrap().sqrt();
    let max_displacement:f64 = displacement.iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
        .unwrap();

    return ( rms_displacement, max_displacement);
}

pub fn get_cartesian_norm(coords:&Array1<f64>,dy:Array1<f64>,internal_coordinates:&InternalCoordinates,dlc_mat:&Array2<f64>)->f64{
    // Get the norm of the optimization step in Cartesian coordinates.
    let x_new:Array1<f64> = cartesian_from_step(coords.clone(),dy,internal_coordinates,dlc_mat.clone());
    let (rmsd,maxd):(f64,f64) = calc_drms_dmax(x_new,coords.clone());
    return rmsd;
}

pub fn get_rot(x:Array2<f64>,y:Array2<f64>)->Array2<f64>{
    // Calculate the rotation matrix that brings x into maximal coincidence with y
    // to minimize the RMSD, following the algorithm in Reference 1.  Mainly
    // used to check the correctness of the quaternion.

    let x_new:Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y_new:Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();
    let n:usize = x.dim().0;
    let temp:(Array1<f64>,f64) = get_quat_rot(&x_new,&y_new);
    let q:Array1<f64> = temp.0;
    let u:Array2<f64> = form_rot(q);
    //let x_r = u.dot(&x_new.t()).t();
    return u;
}

pub fn form_rot(q:Array1<f64>)->Array2<f64>{
    // Given a quaternion p, form a rotation matrix from it.
    let qc:Array1<f64> = get_conj(q.clone());
    let r_4:Array2<f64> = a_l(q).dot(&a_r(qc));
    let returh_val:Array2<f64> = r_4.slice(s![1..,1..]).to_owned();

    return returh_val;
}

pub fn a_l(p:Array1<f64>)->Array2<f64>{
    // Given a quaternion p, return the 4x4 matrix A_L(p)
    // which when multiplied with a column vector q gives
    // the quaternion product pq.
    let a_mat:Array2<f64> = array![[p[0],-p[1],-p[2],-p[3]],
    [p[1],p[0],-p[3],p[2]],
    [p[2],p[3],p[0],-p[1]],
    [p[3],-p[2],p[1],p[0]]];

    return a_mat;
}

pub fn a_r(p:Array1<f64>)->Array2<f64>{
    // Given a quaternion p, return the 4x4 matrix A_R(p)
    // which when multiplied with a column vector q gives
    // the quaternion product qp.
    let a_mat:Array2<f64> = array![[p[0],-p[1],-p[2],-p[3]],
    [p[1],p[0],p[3],-p[2]],
    [p[2],-p[3],p[0],p[1]],
    [p[3],p[2],-p[1],p[0]]];

    return a_mat;
}

pub fn get_conj(q:Array1<f64>)->Array1<f64>{
    // Given a quaternion p, return its conjugate, simply the second
    // through fourth elements changed in sign.
    let mut qc:Array1<f64> = Array::zeros(4);
    qc[0] = q[0];
    qc[1] = -q[1];
    qc[2] = -q[2];
    qc[3] = -q[3];
    return qc;
}

pub fn get_delta_prime(
    v0: f64,
    cart_coords: Array1<f64>,
    gradient_ic: Array1<f64>,
    hessian_ic: Array2<f64>,
    internal_coords: &InternalCoordinates,
    bool_rfo: bool,
) -> (Array1<f64>, f64, f64) {
    // Return the internal coordinate step given a parameter "v".
    // "v" refers to the multiple of the identity added to the Hessian
    // in trust-radius Newton Raphson (TRM), and the multiple of the
    // identity on the RHS matrix in rational function optimization (RFO).
    // Note that reasonable default values are v = 0.0 in TRM and 1.0 in RFO.

    let (dy, sol, dy_prime): (Array1<f64>, f64, f64) =
        get_delta_prime_trm(v0, cart_coords, gradient_ic, hessian_ic, internal_coords);

    return (dy, sol, dy_prime);
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
