use crate::defaults;
use crate::gradients;
use crate::gradients::get_gradients;
use crate::scc_routine;
use crate::solver::get_exc_energies;
use crate::Molecule;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use std::ops::Deref;

// solve the following optimization problem:
// minimize f(x)      subject to  c_i(x) > 0   for  i=1,...,m
// where f(x) is a scalar function, x is a real vector of size n, and c_i(x) are the
// m strict inequality constraints. The feasible space {x|C(x) > 0} is assumed to be convex.
// The constraints are enforced by minimizing an auxiliary function f(x)+nu*B(x).
// B(x) is the log-barrier
// B(x) = - sum_i log(c_i(x))
// and `nu` is a small adjustable number.
// References
// ----------
// [1] J. Nocedal, S. Wright, 'Numerical Optimization', Springer, 2006

pub fn geometry_optimization(state: Option<usize>, coord_system: Option<String>, mol:&Molecule) {
    // defaults
    let state: usize = state.unwrap_or(0);
    let coord_system: String = coord_system.unwrap_or(String::from("internal"));
    let mut cart_coord: bool = false;

    let mut final_coord: Array1<f64> = Array::zeros(3*mol.n_atoms);
    let mut final_grad: Array1<f64> = Array::zeros(3*mol.n_atoms));

    if coord_system == "cartesian" {
        cart_coord = true;
        // flatten cartesian coordinates
        let coords: Array1<f64> = mol.positions.clone().into_shape(3 * mol.n_atoms).unwrap();
        // start the geometry optimization
        let tmp: (Array1<f64>, Array1<f64>, usize) = minimize(
            &coords, cart_coord, state, mol, None, None, None, None, None,);
        final_coord = tmp.0;
        final_grad = tmp.1;

    } else {
        // transform to internal

        // and start optimization
    }
}

pub fn get_energy_and_gradient_s0(x: &Array1<f64>, mol: &Molecule) -> (f64, Array1<f64>) {
    let coords: Array2<f64> = x.clone().into_shape((mol.n_atoms, 3)).unwrap();
    let mut molecule:Molecule = mol.clone();
    molecule.positions = coords;
    let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
        scc_routine::run_scc(&mol, None, None, None);
    let (grad_e0, grad_vrep, grad_exc): (Array1<f64>, Array1<f64>, Array1<f64>) =
        get_gradients(&orbe, &orbs, &s, &mol, &None, &None, None, &None);
    return (energy, grad_e0);
}

pub fn get_energies_and_gradient(
    x: &Array1<f64>,
    mol: &Molecule,
    ex_state: usize,
) -> (Array1<f64>, Array1<f64>) {
    let coords: Array2<f64> = x.clone().into_shape((mol.n_atoms, 3)).unwrap();
    let mut molecule:Molecule = mol.clone();
    molecule.positions = coords;
    let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
        scc_routine::run_scc(&mol, None, None, None);
    let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
        get_exc_energies(&f, &mol, None, &s, &orbe, &orbs, None);
    let omega: Array1<f64> = tmp.0.clone();
    let (grad_e0, grad_vrep, grad_exc): (Array1<f64>, Array1<f64>, Array1<f64>) = get_gradients(
        &orbe,
        &orbs,
        &s,
        &mol,
        &Some(tmp.2),
        &Some(tmp.3),
        Some(ex_state),
        &Some(tmp.0),
    );
    let grad_tot: Array1<f64> = grad_e0 + grad_vrep + grad_exc;
    let energy_tot: Array1<f64> = omega + energy;
    return (energy_tot, grad_tot);
}

pub fn objective_cart(x: &Array1<f64>, state: usize, mol: &Molecule) -> (f64, Array1<f64>) {
    let mut energy: f64 = 0.0;
    let mut gradient: Array1<f64> = Array::zeros(3 * mol.n_atoms);
    if state == 0 {
        let (en, grad): (f64, Array1<f64>) = get_energy_and_gradient_s0(x, mol);
        energy = en;
        gradient = grad;
    } else {
        let (en, grad): (Array1<f64>, Array1<f64>) = get_energies_and_gradient(x, mol, state - 1);
        energy = en[state - 1];
        gradient = grad;
    }
    return (energy, gradient);
}

// pub fn objective_intern(x: Array1<f64>,, state: usize, mol: &mut Molecule) -> (f64, Array1<f64>){
//     //transform to cartesian
//     x_new = internal2cartesian(&x);
//     // transform back to internal
//     x_intern = cartesian2internal(&x_new);
//     let energy, grad = objective_cart(&x_new,state,mol);
//     // transform gradient to internal
//
// }

// pub fn internal2cartesian(){
//
// }

pub fn minimize(
    x0: &Array1<f64>,
    cart_coord: bool,
    state: usize,
    mol: &Molecule,
    method: Option<String>,
    line_search: Option<String>,
    maxiter: Option<usize>,
    gtol: Option<f64>,
    ftol: Option<f64>,
) -> (Array1<f64>, Array1<f64>, usize) {
    // minimize a scalar function ``objfunc``(x) possibly subject to constraints.
    // The minimization is converged if
    //      * |df/dx| < gtol and
    //      * |f(k+1)-f(k)| < ftol

    // set defaults
    let maxiter: usize = maxiter.unwrap_or(100000);
    let gtol: f64 = gtol.unwrap_or(1.0e-6);
    let ftol: f64 = ftol.unwrap_or(1.0e-8);
    let method: String = method.unwrap_or(String::from("BFGS"));
    let line_search: String = line_search.unwrap_or(String::from("Wolfe"));

    let n: usize = x0.len();
    let mut xk: Array1<f64> = x0.clone();
    let mut fk: f64 = 0.0;
    let mut grad_fk: Array1<f64> = Array::zeros(n);

    if cart_coord {
        let tmp: (f64, Array1<f64>) = objective_cart(&xk, state, &mol);
        fk = tmp.0;
        grad_fk = tmp.1;
    }
    // else {
    //     let tmp: (f64, Array1<f64>) = objective_intern(xk);
    //     fk = tmp.0;
    //     grad_fk = tmp.1;
    // }
    let mut converged: bool = false;
    //smallest representable positive number such that 1.0+eps != 1.0.
    let epsilon: f64 = 1.0 + 1.0e-16;

    let mut pk: Array1<f64> = Array::zeros(n);
    let mut x_kp1: Array1<f64> = Array::zeros(n);
    let mut iter_index: usize = 0;
    let mut sk: Array1<f64> = Array::zeros(n);
    let mut yk: Array1<f64> = Array::zeros(n);

    for k in 0..maxiter {
        if method == "BFGS" {
            let mut inv_hk: Array2<f64> = Array::zeros((n, n));
            if k == 0 {
                inv_hk = Array::eye(n);
            } else {
                if yk.dot(&sk) <= 0.0 {
                    println!("Warning: positive definiteness of Hessian approximation lost in BFGS update, since yk.sk <= 0!")
                }
                inv_hk = bfgs_update(&inv_hk, &sk, &yk, k);
            }
            pk = inv_hk.dot(&-&grad_fk);
        } else if method == "Steepest Descent" {
            pk = -grad_fk.clone();
        }
        if line_search == "Armijo" {
            x_kp1 = line_search_backtracking(
                &xk, fk, &grad_fk, &pk, None, None, None, None, cart_coord, state, mol,
            );
        }
        if line_search == "Wolfe" {
            x_kp1 = line_search_wolfe(
                &xk, fk, &grad_fk, &pk, None, None, None, None, None, cart_coord, state, mol,
            );
        }
        let mut f_kp1: f64 = 0.0;
        let mut grad_f_kp1: Array1<f64> = Array::zeros(n);
        if cart_coord {
            let tmp: (f64, Array1<f64>) = objective_cart(&x_kp1, state, mol);
            f_kp1 = tmp.0;
            grad_f_kp1 = tmp.1;
        }
        // else {
        //     let tmp: (f64, Array1<f64>) = objective_intern(&x_kp1);
        //     f_kp1 = tmp.0;
        //     grad_f_kp1 = tmp.1;
        // }
        let f_change: f64 = (f_kp1 - fk).abs();
        let gnorm: f64 = grad_f_kp1.norm();
        if f_change < ftol && gnorm < gtol {
            converged = true;
        }
        if f_change < epsilon {
            println!("WARNING: |f(k+1) - f(k)| < epsilon  (numerical precision) !");
            converged = true;
        }
        // step vector
        sk = &x_kp1 - &xk;
        // gradient difference vector
        yk = &grad_f_kp1 - &grad_fk;
        // new variables for step k become old ones for step k+1
        xk = x_kp1.clone();
        fk = f_kp1;
        grad_fk = grad_f_kp1;
        if converged {
            break;
        }
        iter_index += 1;
    }
    return (xk, grad_fk, iter_index);
}

pub fn bfgs_update(
    inv_hk: &Array2<f64>,
    sk: &Array1<f64>,
    yk: &Array1<f64>,
    k: usize,
) -> (Array2<f64>) {
    // update the inverse Hessian invH_(k+1) based on Algorithm 6.1 in Ref.[1]
    let n: usize = sk.len();
    let id: Array2<f64> = Array::eye(n);
    let mut inv_hkp1: Array2<f64> = Array::zeros((n, n));

    assert!(k >= 1);
    if k == 1 {
        inv_hkp1 = yk.dot(sk) / yk.dot(yk) * &id;
    } else {
        let rk: f64 = 1.0 / yk.dot(sk);
        let u: Array2<f64> = &id
            - &einsum("i,j->ij", &[sk, yk])
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap();
        let v: Array2<f64> = &id
            - &einsum("i,j->ij", &[yk, sk])
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap();
        let w: Array2<f64> = rk
            * einsum("i,j->ij", &[sk, sk])
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap();
        inv_hkp1 = u.dot(&inv_hk.dot(&v)) + w;
    }
    return inv_hkp1;
}

pub fn line_search_backtracking(
    xk: &Array1<f64>,
    fk: f64,
    grad_fk: &Array1<f64>,
    pk: &Array1<f64>,
    a0: Option<f64>,
    rho: Option<f64>,
    c: Option<f64>,
    lmax: Option<usize>,
    cart_coord: bool,
    state: usize,
    mol: &Molecule,
) -> Array1<f64> {
    // set defaults
    let mut a: f64 = a0.unwrap_or(1.0);
    let rho: f64 = rho.unwrap_or(0.3);
    let c: f64 = c.unwrap_or(0.0001);
    let lmax: usize = lmax.unwrap_or(100);

    let n: usize = xk.len();
    // directional derivative
    let df: f64 = grad_fk.dot(pk);

    assert!(df <= 0.0, "pk = {} not a descent direction", &pk);

    let mut x_interp: Array1<f64> = Array::zeros(xk.len());

    if cart_coord {
        for i in 0..lmax {
            x_interp = xk + &(a * pk);

            if objective_cart(&x_interp, state, mol).0 <= fk + c * a * df {
                break;
            } else {
                a = a * rho;
            }
        }
    }
    // else {
    //     for i in 0..lmax {
    //         x_interp = xk + &(a * pk);
    //
    //         if objective_intern(&x_interp) <= fk + c * a * df {
    //             break;
    //         }
    //         else {
    //             a = a * rho;
    //         }
    //     }
    // }
    return x_interp;
}

pub fn line_search_wolfe(
    xk: &Array1<f64>,
    fk: f64,
    grad_fk: &Array1<f64>,
    pk: &Array1<f64>,
    a0: Option<f64>,
    amax: Option<f64>,
    c1: Option<f64>,
    c2: Option<f64>,
    lmax: Option<usize>,
    cart_coord: bool,
    state: usize,
    mol: &Molecule,
) -> (Array1<f64>) {
    // find step size `a`` that satisfies the strong Wolfe conditions:
    //     __
    // 1) sufficient decrease condition   f(xk + a pk) <= f(xk) + c1 a \/f(xk).pk
    // __                        __
    // 2) curvature condition            |\/f(xk + a pk).pk| <= c2 |\/f(xk).pk|

    // set defaults
    let mut a0: f64 = a0.unwrap_or(1.0);
    let mut amax: f64 = amax.unwrap_or(50.0);
    let c1: f64 = c1.unwrap_or(0.0001);
    let c2: f64 = c2.unwrap_or(0.9);
    let lmax: usize = lmax.unwrap_or(100);

    assert!((0.0 < c1) < (c2 < 1.0));

    let s0: f64 = fk;
    let ds0: f64 = grad_fk.dot(pk);

    // Find the largest feasible step length.
    // Not for cartesian coords
    if cart_coord == false {
    }
    //     // call max steplen if coords are internal
    //     // function does not exist
    //     amax = max_steplen(xk, pk);
    // }
    else {
        // for constraints check if amax is feasible
    }
    // The initial guess for the step length should satisfy `a0` < `amax`.
    if a0 >= amax {
        a0 = 0.5 * amax;
    }

    let mut aim1: f64 = 0.0;
    let mut sim1: f64 = s0;
    let mut ai: f64 = a0;
    let mut a_wolfe: f64 = 0.0;
    let mut x_wolfe: Array1<f64> = Array::zeros(xk.len());

    for i in 1..lmax {
        let (si, dsi): (f64, f64) = s_wolfe(ai, cart_coord, xk, pk, state, mol);
        if (si > (s0 + c1 * ai * ds0)) || ((si >= sim1) && i > 1) {
            a_wolfe = zoom(
                lmax, aim1, ai, sim1, si, cart_coord, &xk, &pk, s0, c1, c2, ds0, state, mol,
            );
            break;
        }
        if dsi.abs() <= (c2 * ds0.abs()) {
            a_wolfe = ai;
            break;
        }
        if dsi >= 0.0 {
            a_wolfe = zoom(
                lmax, ai, aim1, si, sim1, cart_coord, &xk, &pk, s0, c1, c2, ds0, state, mol,
            );
            break;
        }
        aim1 = ai;
        sim1 = si;

        ai = 2.0 * aim1;
        if ai >= amax {
            println!("Warning: end of search interval reached");
            ai = 0.5 * (aim1 + amax);
        }
    }
    x_wolfe = xk + &(a_wolfe * pk);
    return x_wolfe;
}

pub fn s_wolfe(
    a: f64,
    cart_coord: bool,
    xk: &Array1<f64>,
    pk: &Array1<f64>,
    state: usize,
    mol: &Molecule,
) -> (f64, f64) {
    // computes scalar function s: a -> f(xk + a*pk) and its derivative ds/da
    let mut fx: f64 = 0.0;
    let mut dsda: f64 = 0.0;
    if cart_coord {
        let tmp: (f64, Array1<f64>) = objective_cart(&(xk + &(a * pk)), state, mol);
        fx = tmp.0;
        let dfdx: Array1<f64> = tmp.1;
        dsda = dfdx.dot(pk);
    }
    // else {
    //     let tmp: (f64, Array1<f64>) = objective_intern(xk + &(a * pk));
    //     fx = tmp.0;
    //     let dfdx: Array1<f64> = tmp.1;
    //     dsda = dfdx.dot(pk);
    // }
    return (fx, dsda);
}

pub fn zoom(
    lmax: usize,
    alo: f64,
    ahi: f64,
    slo: f64,
    shi: f64,
    cart_coord: bool,
    xk: &Array1<f64>,
    pk: &Array1<f64>,
    s0: f64,
    c1: f64,
    c2: f64,
    ds0: f64,
    state: usize,
    mol: &Molecule,
) -> f64 {
    // find a step length a that satisfies Wolfe's conditions by bisection inside in the interval [alo,ali]
    let mut ahi: f64 = ahi;
    let mut shi: f64 = shi;
    let mut alo: f64 = alo;
    let mut slo: f64 = slo;
    let mut a_wolfe: f64 = 0.0;

    for j in 0..lmax {
        let aj: f64 = 0.5 * (ahi + alo);
        let (sj, dsj): (f64, f64) = s_wolfe(aj, cart_coord, xk, pk, state, mol);

        if (sj > (s0 + c1 * aj * ds0)) || sj >= slo {
            ahi = aj;
            shi = sj;
        } else {
            if dsj.abs() <= (c2 * ds0.abs()) {
                a_wolfe = aj;
                break;
            }
            if (dsj * (ahi - alo)) >= 0.0 {
                ahi = alo;
                shi = slo;
            }
            alo = aj;
            slo = sj;
        }
    }
    return a_wolfe;
}
