use ndarray::{Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Ix2, Ix3, Ix4};
use rayon::{into_par_iter};
use rayon::prelude::IntoParallelIterator;
use std::collections::HashMap;
use crate::initialization::parameters::RepulsivePotentialTable;
use ndarray_einsum_beta::tensordot;

pub fn get_outer_product(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> (Array2<f64>) {
    let mut matrix: Array2<f64> = Array::zeros((v1.len(), v2.len()));
    for (i, i_value) in v1.outer_iter().enumerate() {
        for (j, j_value) in v2.outer_iter().enumerate() {
            matrix[[i, j]] = (&i_value * &j_value).into_scalar();
        }
    }
    return matrix;
}

pub fn f_v_new(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let vp: Array2<f64> = v.to_owned() + v.t().to_owned();
    let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_ao.dot(&sv);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb, n_orb));

    let mut f_return: Vec<_> = (0..3 * n_atoms)
        .into_par_iter()
        .map(|nc| {
            let ds: Array2<f64> = grad_s.slice(s![nc, .., ..]).to_owned();
            let dg: Array2<f64> = g1_ao.slice(s![nc, .., ..]).to_owned();

            let gdsv: Array1<f64> = g0_ao.dot(&(&ds * &vp).sum_axis(Axis(0)));
            let dgsv: Array1<f64> = dg.dot(&sv);
            //let mut d_f:Array2<f64> = Array2::zeros((n_orb,n_orb));
            let mut d_f: Vec<f64> = Vec::new();

            for b in 0..n_orb {
                for a in 0..n_orb {
                    d_f.push(
                        ds[[a, b]] * (gsv[a] + gsv[b])
                            + s[[a, b]] * (dgsv[a] + gdsv[a] + dgsv[b] + gdsv[b]),
                    );
                }
            }
            (Array::from(d_f) * 0.25).to_vec()
        })
        .collect();
    let mut f_result: Vec<f64> = Vec::new();

    for vec in f_return.iter_mut() {
        f_result.append(&mut *vec);
    }

    //for i in 0..f_return.len(){
    //    for j in 0..f_return[i].len(){
    //        f_result.push(f_return[i][j]);
    //    }
    //}
    let f_result_temp: Array1<f64> = Array::from(f_result);
    let f_return: Array3<f64> = f_result_temp
        .into_shape((3 * n_atoms, n_orb, n_orb))
        .unwrap();

    return f_return;
}

pub fn f_lr_new(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let sv: Array2<f64> = s.dot(&v);
    let v_t: Array2<f64> = v.t().to_owned();
    let sv_t: Array2<f64> = s.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_a0 * &v;

    let t_sv: Array2<f64> = sv.t().to_owned();
    let svg_t: Array2<f64> = (&sv * &g0_lr_a0).t().to_owned();
    let sgv_t: Array2<f64> = s.dot(&gv).t().to_owned();

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb, n_orb));

    let mut f_return: Vec<_> = (0..3 * n_atoms)
        .into_par_iter()
        .map(|nc| {
            let d_s: Array2<f64> = grad_s.slice(s![nc, .., ..]).to_owned();
            let d_g: Array2<f64> = g1_lr_ao.slice(s![nc, .., ..]).to_owned();

            let d_sv_t: Array2<f64> = d_s.dot(&v_t);
            let d_sv: Array2<f64> = d_s.dot(&v);
            let d_gv: Array2<f64> = d_g.clone() * v;

            let mut d_f: Array2<f64> = Array2::zeros((n_orb, n_orb));
            // 1st term
            d_f = d_f + g0_lr_a0.to_owned() * d_s.dot(&t_sv);
            // 2nd term
            d_f = d_f + (&d_sv_t * &g0_lr_a0).dot(&s);
            // 3rd term
            d_f = d_f + d_s.dot(&svg_t);
            // 4th term
            d_f = d_f + d_s.dot(&sgv_t);
            // 5th term
            d_f = d_f + g0_lr_a0.to_owned() * s.dot(&d_sv.t());
            // 6th term
            d_f = d_f + (sv_t.clone() * g0_lr_a0).dot(&d_s.t());
            // 7th term
            d_f = d_f + s.dot(&(&d_sv * &g0_lr_a0).t());
            // 8th term
            d_f = d_f + s.dot(&(d_s.dot(&gv)).t());
            // 9th term
            d_f = d_f + d_g.clone() * s.dot(&t_sv);
            // 10th term
            d_f = d_f + (sv_t.clone() * d_g.clone()).dot(&s);
            // 11th term
            d_f = d_f + s.dot(&(&sv * &d_g).t());
            // 12th term
            d_f = d_f + s.dot(&(s.dot(&d_gv)).t());
            d_f = d_f * 0.25;

            d_f.into_shape(n_orb * n_orb).unwrap().to_vec()
        })
        .collect();
    let mut f_result: Vec<f64> = Vec::new();

    for vec in f_return.iter_mut() {
        f_result.append(&mut *vec);
    }

    let f_result_temp: Array1<f64> = Array::from(f_result);
    let f_return: Array3<f64> = f_result_temp
        .into_shape((3 * n_atoms, n_orb, n_orb))
        .unwrap();

    return f_return;
}

pub fn h_minus(
    g0_lr_a0: ArrayView2<f64>,
    q_ps: ArrayView3<f64>,
    q_qr: ArrayView3<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> (Array2<f64>) {
    // term 1
    let tmp: Array3<f64> = tensordot(&q_qr, &v_rs, &[Axis(2)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let tmp2: Array3<f64> = tensordot(&g0_lr_a0, &tmp, &[Axis(1)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let mut h_minus_pq: Array2<f64> =
        tensordot(&q_ps, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
            .into_dimensionality::<Ix2>()
            .unwrap();
    // term 2
    let tmp: Array3<f64> = tensordot(&q_qs, &v_rs, &[Axis(2)], &[Axis(1)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let tmp2: Array3<f64> = tensordot(&g0_lr_a0, &tmp, &[Axis(1)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    h_minus_pq = h_minus_pq
        - tensordot(&q_pr, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    return h_minus_pq;
}

pub fn h_plus_lr(
    g0_ao: ArrayView2<f64>,
    g0_lr_a0: ArrayView2<f64>,
    q_pq: ArrayView3<f64>,
    q_rs: ArrayView3<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
    q_ps: ArrayView3<f64>,
    q_qr: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> (Array2<f64>) {
    // term 1
    let tmp: Array1<f64> = tensordot(&q_rs, &v_rs, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
        .into_dimensionality::<Ix1>()
        .unwrap();
    let tmp2: Array1<f64> = g0_ao.to_owned().dot(&tmp);
    let mut hplus_pq: Array2<f64> = 4.0
        * tensordot(&q_pq, &tmp2, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    // term 2
    let tmp: Array3<f64> = tensordot(&q_qs, &v_rs, &[Axis(2)], &[Axis(1)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let tmp2: Array3<f64> = tensordot(&g0_lr_a0, &tmp, &[Axis(1)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    hplus_pq = hplus_pq
        - tensordot(&q_pr, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    // term 3
    let tmp: Array3<f64> = tensordot(&q_qr, &v_rs, &[Axis(2)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let tmp2: Array3<f64> = tensordot(&g0_lr_a0, &tmp, &[Axis(1)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    hplus_pq = hplus_pq
        - tensordot(&q_ps, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    return hplus_pq;
}

pub fn h_plus_no_lr(
    g0: ArrayView2<f64>,
    q_pq: ArrayView3<f64>,
    q_rs: ArrayView3<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
    q_ps: ArrayView3<f64>,
    q_qr: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> (Array2<f64>) {
    // term 1
    let tmp: Array1<f64> = tensordot(&q_rs, &v_rs, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
        .into_dimensionality::<Ix1>()
        .unwrap();
    let tmp2: Array1<f64> = g0.to_owned().dot(&tmp);
    let hplus_pq: Array2<f64> = 4.0
        * tensordot(&q_pq, &tmp2, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    return hplus_pq;
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
fn gradient_v_rep(
    atomic_numbers: &[u8],
    distances: ArrayView2<f64>,
    directions: ArrayView3<f64>,
    v_rep: &HashMap<(u8, u8), RepulsivePotentialTable>,
) -> Array1<f64> {
    let n_atoms: usize = atomic_numbers.len();
    let mut grad: Array1<f64> = Array1::zeros([3 * n_atoms]);
    for (i, z_i) in atomic_numbers.iter().enumerate() {
        let mut grad_i: Array1<f64> = Array::zeros([3]);
        for (j, z_j) in atomic_numbers.iter().enumerate() {
            if i != j {
                let (z_1, z_2): (u8, u8) = if z_i > z_j {
                    (*z_j, *z_i)
                } else {
                    (*z_i, *z_j)
                };
                let r_ij: f64 = distances[[i, j]];
                let v_ij_deriv: f64 = v_rep[&(z_1, z_2)].spline_deriv(r_ij);
                grad_i = &grad_i + &directions.slice(s![i, j, ..]).map(|x| x * v_ij_deriv);
            }
        }
        grad.slice_mut(s![i * 3..i * 3 + 3]).assign(&grad_i);
    }
    return grad;
}