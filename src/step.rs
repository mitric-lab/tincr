use crate::defaults;
use crate::gradients;
use crate::gradients::get_gradients;
use crate::internal_coordinates::*;
use crate::scc_routine;
use crate::solver::get_exc_energies;
use crate::Molecule;
use approx::AbsDiffEq;
use itertools::any;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use rand::Rng;
use std::ops::Deref;
use std::ops::Not;

pub fn find_root_brent(
    a: f64,
    b: f64,
    rel: f64,
    conv: Option<f64>,
    v0: f64,
    cart_coords: Array1<f64>,
    gradient_ic: Array1<f64>,
    hessian_ic: Array2<f64>,
    internal_coords: &InternalCoordinates,
    dlc_mat: &Array2<f64>,
) -> (f64, bool,Option<f64>,bool) {
    // Brent's method for finding the root of a function.
    // Parameters
    // ----------
    // a : float
    // One side of the "bracket" to start finding the root
    // b : float
    // The other side of the "bracket"
    // rel : float
    // The denominator used to calculate the fractional error (in our case, the trust radius)
    // conv : float
    // The convergence threshold for the relative error
    let tmp_1: (f64, usize, Option<f64>,Option<f64>,bool) = evaluate_find_root(
        a,
        v0,
        cart_coords.clone(),
        gradient_ic.clone(),
        hessian_ic.clone(),
        internal_coords,
        dlc_mat,
        None,
        None,
    );
    let mut stored_arg:Option<f64> = None;
    if tmp_1.3.is_some(){
        stored_arg = tmp_1.3;
    }
    let tmp_2: (f64, usize, Option<f64>,Option<f64>,bool) = evaluate_find_root(
        b,
        v0,
        cart_coords.clone(),
        gradient_ic.clone(),
        hessian_ic.clone(),
        internal_coords,
        dlc_mat,
        None,
        None,
    );
    if tmp_2.3.is_some(){
        stored_arg = tmp_2.3;
    }
    let mut f_a: f64 = tmp_1.0;
    let mut f_b: f64 = tmp_2.0;
    let conv: f64 = conv.unwrap_or(0.1);

    let mut a: f64 = a;
    let mut b: f64 = b;
    let mut a_new: f64 = a;
    let mut b_new: f64 = b;
    let mut f_a_new: f64 = f_a;
    let mut f_b_new: f64 = f_b;

    if f_a * f_b > 0.0 {
        println!("Error");
    }
    if f_a.abs() < f_b.abs() {
        // Swap if |f(a)| < |f(b)|
        a_new = b;
        b_new = a;
        f_a_new = f_b;
        f_b_new = f_a;
    }
    let mut c: f64 = a_new;
    let mut f_c: f64 = f_a_new;
    let mut mflag: bool = true;
    let delta: f64 = 1e-6;
    let epsilon: f64 = 0.01_f64.min(1.0e-2 * (a_new - b_new).abs());
    let mut d: Option<f64> = None;
    let mut return_value: f64 = 0.0;
    let mut brent_failed: bool = false;
    let mut f_d: f64 = 0.0;
    let mut bork:bool = false;

    while true {
        let mut s: f64 = 0.0;
        if f_a_new != f_c && f_b_new != f_c {
            s = a_new * f_b_new * f_c / ((f_a_new - f_b_new) * (f_a_new - f_c));
            s += b_new * f_a_new * f_c / ((f_b_new - f_a_new) * (f_b_new - f_c));
            s += c * f_a_new * f_b_new / ((f_c - f_a_new) * (f_c - f_b_new));
        } else {
            s = b_new - f_b_new * (b_new - a_new) / (f_b_new - f_a_new);
        }
        // evaluate conditions
        let condition_1: bool = between_floats(s, (3.0 * a_new + b) / 4.0, b).not();
        let condition_2: bool = mflag && ((s - b_new).abs() >= (b_new - c).abs() / 2.0);

        let mut condition_3: bool = false;
        let mut condition_5: bool = false;
        let condition_4: bool = mflag && ((b_new - c).abs() < delta);
        if d.is_some() {
            let d: f64 = d.unwrap();
            let condition_3: bool = mflag.not() && ((s - b_new).abs() >= (c - d).abs() / 2.0);
            let condition_5: bool = mflag.not() && ((c - d).abs() < delta);
        }
        if condition_1 || condition_2 || condition_3 || condition_4 || condition_5 {
            // bisection
            s = (a_new + b_new) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }
        let tmp_1: (f64, usize, Option<f64>,Option<f64>,bool) = evaluate_find_root(
            a,
            v0,
            cart_coords.clone(),
            gradient_ic.clone(),
            hessian_ic.clone(),
            internal_coords,
            dlc_mat,
            None,
            None,
        );
        if tmp_1.3.is_some(){
            stored_arg = tmp_1.3;
        }
        bork = tmp_1.4;
        let f_s: f64 = tmp_1.0;

        if (f_s / rel).abs() <= conv {
            return_value = s;
            break;
        }

        if (a_new - b_new).abs() < epsilon {
            return_value = s;
            brent_failed = true;
            break;
        }
        d = Some(c);
        f_d = f_c;
        c = b_new;
        f_c = f_b_new;

        if f_a_new * f_s < 0.0 {
            b_new = s;
            f_b_new = f_s;
        } else {
            a_new = s;
            f_a_new = f_s;
        }
        if f_a_new.abs() < f_b_new.abs() {
            // Swap if |f(a)| < |f(b)|
            a = b_new;
            b = a_new;
            f_a = f_b_new;
            f_b = f_a_new;
            a_new = a;
            b_new = b;
            f_a_new = f_a;
            f_b_new = f_b;
        }
    }
    return (return_value, brent_failed,stored_arg,bork);
}

pub fn between_floats(s: f64, a: f64, b: f64) -> bool {
    let mut return_val_1: bool = false;
    let mut return_val_2: bool = false;
    if a < b {
        return_val_1 = s > a;
        return_val_2 = s < b;
    } else if a > b {
        return_val_1 = s > b;
        return_val_2 = s < a;
    } else {
        println!("Error in between, a and b must be different");
    }
    let mut return_final: bool = false;
    if return_val_1 == true && return_val_2 == true {
        return_final = true;
    }
    return return_final;
}

pub fn evaluate_find_root(
    trial: f64,
    v0: f64,
    cart_coords: Array1<f64>,
    gradient_ic: Array1<f64>,
    hessian_ic: Array2<f64>,
    internal_coords: &InternalCoordinates,
    dlc_mat: &Array2<f64>,
    counter: Option<usize>,
    stored_val: Option<f64>,
) -> (f64, usize, Option<f64>,Option<f64>,bool) {
    let trust: f64 = 0.1;
    let target: f64 = 0.1;
    let mut return_value: f64 = 0.0;
    let mut cnorm: f64 = 0.0;
    let mut counter: usize = counter.unwrap_or(0);
    let mut stored_val: Option<f64> = stored_val;
    let mut stored_arg: Option<f64> = None;
    let mut bork:bool = false;

    if trial == 0.0 {
        return_value = -1.0 * trust;
    } else {
        let (dy, sol): (Array1<f64>, f64) = trust_step(
            target,
            v0,
            cart_coords.clone(),
            gradient_ic.clone(),
            hessian_ic.clone(),
            internal_coords,
        );
        let tmp:(f64,bool) = get_cartesian_norm(&cart_coords, dy.clone(), internal_coords, dlc_mat);
        let c_norm: f64 = tmp.0;
        bork = tmp.1;
        counter += 1;
    }
    if cnorm - target < 0.0 {
        if stored_val.is_none() || cnorm > stored_val.unwrap() {
            stored_val = Some(cnorm);
            stored_arg = Some(trial);
        }
    }
    let return_val: f64 = cnorm - target;

    return (return_val, counter, stored_val,stored_arg,bork);
}

pub fn trust_step(
    target: f64,
    v0: f64,
    cart_coords: Array1<f64>,
    gradient_ic: Array1<f64>,
    hessian_ic: Array2<f64>,
    internal_coords: &InternalCoordinates,
) -> (Array1<f64>, f64) {
    let (dy, sol, dy_prime): (Array1<f64>, f64, f64) = get_delta_prime(
        v0,
        cart_coords.clone(),
        gradient_ic.clone(),
        hessian_ic.clone(),
        internal_coords,
        false,
    );
    let ndy: f64 = dy.clone().to_vec().norm();
    let mut return_dy: Array1<f64> = Array::zeros((dy.dim()));
    let mut return_sol: f64 = 0.0;
    let mut do_while: bool = true;

    if ndy < target {
        return_dy = dy.clone();
        return_sol = sol;
        do_while = false;
    }
    let mut v: f64 = v0;
    let mut niter: usize = 0;
    let mut ndy_last: f64 = 0.0;
    let mut m_ndy: f64 = 0.0;
    let mut m_dy: Array1<f64> = dy.clone();
    let mut m_sol: f64 = sol;

    while do_while {
        v += (1.0 - ndy / target) * (ndy / dy_prime);
        let (dy, sol, dy_prime): (Array1<f64>, f64, f64) = get_delta_prime(
            v0,
            cart_coords.clone(),
            gradient_ic.clone(),
            hessian_ic.clone(),
            internal_coords,
            false,
        );
        let ndy: f64 = dy.clone().to_vec().norm();

        if ((ndy - target) / target).abs() < 0.001 {
            return_dy = dy.clone();
            return_sol = sol;
            break;
        } else if niter > 10 && ((ndy_last - ndy) / ndy).abs() < 0.001 {
            return_dy = dy.clone();
            return_sol = sol;
            break;
        }
        niter += 1;
        ndy_last = ndy;
        if ndy < m_ndy {
            m_ndy = ndy;
            m_dy = dy.clone();
            m_sol = sol;
        }
        if niter % 100 == 99 {
            let rng: f64 = rand::random::<f64>();
            v += rng * (niter as f64) / 100.0;
        }
        if niter % 100 == 999 {
            return_dy = dy.clone();
            return_sol = sol;
            break;
        }
    }
    return (return_dy, return_sol);
}

pub fn calc_drms_dmax(x_new: Array1<f64>, x_old: Array1<f64>) -> (f64, f64) {
    // Align and calculate the RMSD for two geometries
    let n_at: usize = x_old.len() / 3;
    let mut coords_new: Array2<f64> = x_new.into_shape((n_at, 3)).unwrap();
    let mut coords_old: Array2<f64> = x_old.clone().into_shape((n_at, 3)).unwrap();
    coords_old = coords_old.clone() - coords_old.mean_axis(Axis(0)).unwrap();
    coords_new = coords_new.clone() - coords_new.mean_axis(Axis(0)).unwrap();
    // Obtain the rotation
    let u: Array2<f64> = get_rot(coords_new.clone(), coords_old.clone());
    let mut x_rot: Array2<f64> = u.dot(&coords_new.t());
    x_rot = x_rot.reversed_axes();
    let x_rot_new: Array1<f64> = x_rot
        .clone()
        .into_shape((x_rot.dim().0 * x_rot.dim().1))
        .unwrap();
    let x_old_new: Array1<f64> = coords_old.into_shape(x_old.len()).unwrap();
    let difference_arr: Array2<f64> = ((x_rot_new - x_old_new) / 1.8897261278504418)
        .into_shape((n_at, 3))
        .unwrap();
    let displacement: Array1<f64> = difference_arr
        .mapv(|val| val.powi(2))
        .sum_axis(Axis(1))
        .mapv(|val| val.sqrt());
    let rms_displacement: f64 = displacement.mapv(|val| val.powi(2)).mean().unwrap().sqrt();
    let max_displacement: f64 = displacement
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
        .unwrap();

    return (rms_displacement, max_displacement);
}

pub fn get_cartesian_norm(
    coords: &Array1<f64>,
    dy: Array1<f64>,
    internal_coordinates: &InternalCoordinates,
    dlc_mat: &Array2<f64>,
) -> (f64,bool) {
    // Get the norm of the optimization step in Cartesian coordinates.
    let tmp:(Array1<f64>, bool) = cartesian_from_step(
        coords.clone(),
        dy,
        internal_coordinates,
        dlc_mat.clone(),
    );
    let x_new: Array1<f64> = tmp.0;
    let bork:bool = tmp.1;
    let (rmsd, maxd): (f64, f64) = calc_drms_dmax(x_new, coords.clone());
    return (rmsd,bork);
}

pub fn get_rot(x: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
    // Calculate the rotation matrix that brings x into maximal coincidence with y
    // to minimize the RMSD, following the algorithm in Reference 1.  Mainly
    // used to check the correctness of the quaternion.

    let x_new: Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y_new: Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();
    let n: usize = x.dim().0;
    let temp: (Array1<f64>, f64) = get_quat_rot(&x_new, &y_new);
    let q: Array1<f64> = temp.0;
    let u: Array2<f64> = form_rot(q);
    //let x_r = u.dot(&x_new.t()).t();
    return u;
}

pub fn form_rot(q: Array1<f64>) -> Array2<f64> {
    // Given a quaternion p, form a rotation matrix from it.
    let qc: Array1<f64> = get_conj(q.clone());
    let r_4: Array2<f64> = a_l(q).dot(&a_r(qc));
    let returh_val: Array2<f64> = r_4.slice(s![1.., 1..]).to_owned();

    return returh_val;
}

pub fn a_l(p: Array1<f64>) -> Array2<f64> {
    // Given a quaternion p, return the 4x4 matrix A_L(p)
    // which when multiplied with a column vector q gives
    // the quaternion product pq.
    let a_mat: Array2<f64> = array![
        [p[0], -p[1], -p[2], -p[3]],
        [p[1], p[0], -p[3], p[2]],
        [p[2], p[3], p[0], -p[1]],
        [p[3], -p[2], p[1], p[0]]
    ];

    return a_mat;
}

pub fn a_r(p: Array1<f64>) -> Array2<f64> {
    // Given a quaternion p, return the 4x4 matrix A_R(p)
    // which when multiplied with a column vector q gives
    // the quaternion product qp.
    let a_mat: Array2<f64> = array![
        [p[0], -p[1], -p[2], -p[3]],
        [p[1], p[0], p[3], -p[2]],
        [p[2], -p[3], p[0], p[1]],
        [p[3], p[2], -p[1], p[0]]
    ];

    return a_mat;
}

pub fn get_conj(q: Array1<f64>) -> Array1<f64> {
    // Given a quaternion p, return its conjugate, simply the second
    // through fourth elements changed in sign.
    let mut qc: Array1<f64> = Array::zeros(4);
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
