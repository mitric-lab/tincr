use ndarray::{Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Ix2, Ix3, Ix4,s};
use rayon::iter::*;
use std::collections::HashMap;
use crate::initialization::parameters::{RepulsivePotentialTable, RepulsivePotential};
use ndarray_einsum_beta::tensordot;
use crate::initialization::Atom;
use nalgebra::Vector3;
use ndarray_linalg::krylov::qr;

pub fn get_outer_product(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> (Array2<f64>) {
    let mut matrix: Array2<f64> = Array::zeros((v1.len(), v2.len()));
    for (i, i_value) in v1.outer_iter().enumerate() {
        for (j, j_value) in v2.outer_iter().enumerate() {
            matrix[[i, j]] = (&i_value * &j_value).into_scalar();
        }
    }
    return matrix;
}

pub fn f_v(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let vp: Array2<f64> = &v + &(v.t());
    let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_ao.dot(&sv);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb, n_orb));

    for nc in 0..3*n_atoms{
       let ds:ArrayView2<f64> = grad_s.slice(s![nc,..,..]);
       let dg:ArrayView2<f64> = g1_ao.slice(s![nc,..,..]);

       let gdsv:Array1<f64> = g0_ao.dot(&(&ds*&vp).sum_axis(Axis(0)));
       let dgsv:Array1<f64> = dg.dot(&sv);
       let mut d_f:Array2<f64> = Array2::zeros((n_orb,n_orb));

       for b in 0..n_orb{
           for a in 0..n_orb{
               d_f[[a,b]] = ds[[a,b]]*(gsv[a]+gsv[b]) + s[[a,b]]*(dgsv[a]+gdsv[a]+dgsv[b]+gdsv[b]);
           }
       }
       d_f = d_f * 0.25;
       f_return.slice_mut(s![nc,..,..]).assign(&d_f);
    }

    return f_return;
}

pub fn f_v_par(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let vp: Array2<f64> = &v + &(v.t());
    let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_ao.dot(&sv);

    let mut f_return: Vec<_> = (0..3 * n_atoms)
        .into_par_iter()
        .map(|nc| {
            let ds:ArrayView2<f64> = grad_s.slice(s![nc,..,..]);
            let dg:ArrayView2<f64> = g1_ao.slice(s![nc,..,..]);

            let gdsv: Array1<f64> = g0_ao.dot(&(&ds * &vp).sum_axis(Axis(0)));
            let dgsv: Array1<f64> = dg.dot(&sv);

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
    let f_result_temp: Array1<f64> = Array::from(f_result);
    let f_return: Array3<f64> = f_result_temp
        .into_shape((3 * n_atoms, n_orb, n_orb))
        .unwrap();

    return f_return;
}

fn f_lr(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let sv: Array2<f64> = s.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_a0 * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_a0).reversed_axes();
    let sgv_t: Array2<f64> = s.dot(&gv).reversed_axes();

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb, n_orb));

    for nc in 0..3*n_atoms{
        let d_s:ArrayView2<f64> = grad_s.slice(s![nc,..,..]);
        let d_g:ArrayView2<f64> = g1_lr_ao.slice(s![nc,..,..]);

        let d_sv_t:Array2<f64> = d_s.dot(&v_t);
        let d_sv:Array2<f64> = d_s.dot(&v);
        let d_gv:Array2<f64> = &d_g * &v;

        let mut d_f:Array2<f64> = Array2::zeros((n_orb,n_orb));
        // 1st term
        d_f = d_f + &g0_lr_a0*&(d_s.dot(&t_sv));
        // 2nd term
        d_f = d_f + (&d_sv_t*&g0_lr_a0).dot(&s);
        // 3rd term
        d_f = d_f + d_s.dot(&svg_t);
        // 4th term
        d_f = d_f + d_s.dot(&sgv_t);
        // 5th term
        d_f = d_f + &g0_lr_a0 * &(s.dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t*&g0_lr_a0).dot(&d_s.t());
        // 7th term
        d_f = d_f + s.dot(&(&d_sv*&g0_lr_a0).t());
        // 8th term
        d_f = d_f + s.dot(&(d_s.dot(&gv)).t());
        // 9th term
        d_f = d_f + &d_g*&(s.dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t*&d_g).dot(&s);
        // 11th term
        d_f = d_f + s.dot(&(&sv*&d_g).t());
        // 12th term
        d_f = d_f + s.dot(&(s.dot(&d_gv)).t());
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc,..,..]).assign(&d_f);
    }
    return f_return;
}

pub fn f_lr_par(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let sv: Array2<f64> = s.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_a0 * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_a0).reversed_axes();
    let sgv_t: Array2<f64> = s.dot(&gv).reversed_axes();

    let mut f_return: Vec<_> = (0..3 * n_atoms)
        .into_par_iter()
        .map(|nc| {
            let d_s:ArrayView2<f64> = grad_s.slice(s![nc,..,..]);
            let d_g:ArrayView2<f64> = g1_lr_ao.slice(s![nc,..,..]);

            let d_sv_t:Array2<f64> = d_s.dot(&v_t);
            let d_sv:Array2<f64> = d_s.dot(&v);
            let d_gv:Array2<f64> = &d_g * &v;

            let mut d_f:Array2<f64> = Array2::zeros((n_orb,n_orb));
            // 1st term
            d_f = d_f + &g0_lr_a0*&(d_s.dot(&t_sv));
            // 2nd term
            d_f = d_f + (&d_sv_t*&g0_lr_a0).dot(&s);
            // 3rd term
            d_f = d_f + d_s.dot(&svg_t);
            // 4th term
            d_f = d_f + d_s.dot(&sgv_t);
            // 5th term
            d_f = d_f + &g0_lr_a0 * &(s.dot(&d_sv.t()));
            // 6th term
            d_f = d_f + (&sv_t*&g0_lr_a0).dot(&d_s.t());
            // 7th term
            d_f = d_f + s.dot(&(&d_sv*&g0_lr_a0).t());
            // 8th term
            d_f = d_f + s.dot(&(d_s.dot(&gv)).t());
            // 9th term
            d_f = d_f + &d_g*&(s.dot(&t_sv));
            // 10th term
            d_f = d_f + (&sv_t*&d_g).dot(&s);
            // 11th term
            d_f = d_f + s.dot(&(&sv*&d_g).t());
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
    g0_lr: ArrayView2<f64>,
    q_ps: ArrayView3<f64>,
    q_qr: ArrayView3<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> (Array2<f64>) {
    // term 1
    let n_at:usize = q_ps.dim().0;
    let n_virt:usize = q_ps.dim().2;
    let n_occ:usize = q_qr.dim().2;
    let qr_dim_1:usize = q_qr.dim().1;

    let tmp: Array3<f64> = q_qr.into_shape((n_at*qr_dim_1,n_occ)).unwrap().dot(&v_rs).into_shape((n_at,qr_dim_1,n_virt)).unwrap();
    // let tmp: Array3<f64> = tensordot(&q_qr, &v_rs, &[Axis(2)], &[Axis(0)])
    //     .into_dimensionality::<Ix3>()
    //     .unwrap();
    let tmp2:Array3<f64> = g0_lr.dot(&(tmp.into_shape((n_at,qr_dim_1*n_virt)).unwrap())).into_shape((n_at,qr_dim_1,n_virt)).unwrap();
    // let tmp2: Array3<f64> = tensordot(&g0_lr, &tmp, &[Axis(1)], &[Axis(0)])
    //     .into_dimensionality::<Ix3>()
    //     .unwrap();
    let q_ps_swapped = q_ps.permuted_axes([1,0,2]).as_standard_layout().into_shape((qr_dim_1,n_at*n_virt)).unwrap().to_owned();
    let tmp2_swapped = tmp2.permuted_axes([0,2,1]).as_standard_layout().into_shape((n_virt*n_at,qr_dim_1)).unwrap().to_owned();
    let mut h_minus_pq:Array2<f64> = q_ps_swapped.dot(&tmp2_swapped);
    // let mut h_minus_pq: Array2<f64> =
    //     tensordot(&q_ps, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
    //         .into_dimensionality::<Ix2>()
    //         .unwrap();
    // term 2
    let tmp:Array3<f64> = q_qs.into_shape((n_at*qr_dim_1,n_virt)).unwrap().dot(&v_rs.t()).into_shape((n_at,qr_dim_1,n_occ)).unwrap();
    // let tmp: Array3<f64> = tensordot(&q_qs, &v_rs, &[Axis(2)], &[Axis(1)])
    //     .into_dimensionality::<Ix3>()
    //     .unwrap();
    let tmp2:Array3<f64> = g0_lr.dot(&(tmp.into_shape((n_at,qr_dim_1*n_occ)).unwrap())).into_shape((n_at,qr_dim_1,n_occ)).unwrap();
    // let tmp2: Array3<f64> = tensordot(&g0_lr, &tmp, &[Axis(1)], &[Axis(0)])
    //     .into_dimensionality::<Ix3>()
    //     .unwrap();
    let q_pr_swapped = q_pr.permuted_axes([1,0,2]).as_standard_layout().into_shape((qr_dim_1,n_at*n_occ)).unwrap().to_owned();
    let tmp2_swapped = tmp2.permuted_axes([0,2,1]).as_standard_layout().into_shape((n_virt*n_occ,qr_dim_1)).unwrap().to_owned();
    h_minus_pq = h_minus_pq - q_pr_swapped.dot(&tmp2_swapped);
    // h_minus_pq = h_minus_pq
    //     - tensordot(&q_pr, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
    //     .into_dimensionality::<Ix2>()
    //     .unwrap();
    return h_minus_pq;
}

pub fn h_plus_lr(
    g0_ao: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    q_pq: ArrayView3<f64>,
    q_rs: ArrayView3<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
    q_ps: ArrayView3<f64>,
    q_qr: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> (Array2<f64>) {
    // set dimensions
    let v_rs_dim_0:usize = v_rs.dim().0;
    let v_rs_dim_1:usize = v_rs.dim().1;
    let n_at:usize = q_pq.dim().0;
    let q_rs_dim_1:usize = q_rs.dim().1;
    let q_rs_dim_2:usize = q_rs.dim().2;
    let q_pq_dim_1:usize = q_pq.dim().1;
    let q_pq_dim_2:usize = q_pq.dim().2;
    let q_qs_dim_1:usize = q_qs.dim().1;
    let q_qs_dim_2:usize = q_qs.dim().2;
    let q_pr_dim_1:usize = q_pr.dim().1;
    let q_pr_dim_2:usize = q_pr.dim().2;
    let q_qr_dim_1:usize = q_qr.dim().1;
    let q_qr_dim_2:usize = q_qr.dim().2;
    let q_ps_dim_1:usize = q_ps.dim().1;
    let q_ps_dim_2:usize = q_ps.dim().2;

    // term 1
    let tmp:Array1<f64> = q_rs.into_shape((n_at,q_rs_dim_1*q_rs_dim_2)).unwrap().dot(&v_rs.into_shape(v_rs_dim_0*v_rs_dim_1).unwrap());
    // let tmp: Array1<f64> = tensordot(&q_rs, &v_rs, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
    //     .into_dimensionality::<Ix1>()
    //     .unwrap();
    let tmp2: Array1<f64> = g0_ao.to_owned().dot(&tmp);
    let mut hplus_pq:Array2<f64> = 4.0 * tmp2.dot(&q_pq.into_shape((n_at,q_pq_dim_1*q_pq_dim_2)).unwrap()).into_shape((q_pq_dim_1,q_pq_dim_2)).unwrap();
    // let mut hplus_pq: Array2<f64> = 4.0
    //     * tensordot(&q_pq, &tmp2, &[Axis(0)], &[Axis(0)])
    //     .into_dimensionality::<Ix2>()
    //     .unwrap();
    // term 2
    let tmp:Array3<f64> = q_qs.into_shape((n_at*q_qs_dim_1,q_qs_dim_2)).unwrap().dot(&v_rs.t()).into_shape((n_at,q_qs_dim_1,v_rs_dim_0)).unwrap();
    // let tmp: Array3<f64> = tensordot(&q_qs, &v_rs, &[Axis(2)], &[Axis(1)])
    //     .into_dimensionality::<Ix3>()
    //     .unwrap();
    let tmp2:Array3<f64> = g0_lr.dot(&tmp.into_shape((n_at,q_qs_dim_1*v_rs_dim_0)).unwrap()).into_shape((n_at,q_qs_dim_1,v_rs_dim_0)).unwrap();
    // let tmp2: Array3<f64> = tensordot(&g0_lr, &tmp, &[Axis(1)], &[Axis(0)])
    //     .into_dimensionality::<Ix3>()
    //     .unwrap();
    let tmp2_swapped = tmp2.permuted_axes([0,2,1]).as_standard_layout().into_shape((n_at*v_rs_dim_0,q_qs_dim_1)).unwrap().to_owned();
    let q_pr_swapped = q_pr.permuted_axes([1,0,2]).as_standard_layout().into_shape((q_pr_dim_1,n_at*q_pr_dim_2)).unwrap().to_owned();
    hplus_pq = hplus_pq - q_pr_swapped.dot(&tmp2_swapped);
    // hplus_pq = hplus_pq
    //     - tensordot(&q_pr, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
    //     .into_dimensionality::<Ix2>()
    //     .unwrap();
    // term 3
    let tmp:Array3<f64> = q_qr.into_shape((n_at*q_qr_dim_1,q_qr_dim_2)).unwrap().dot(&v_rs).into_shape((n_at,q_qr_dim_1,v_rs_dim_1)).unwrap();
    // let tmp: Array3<f64> = tensordot(&q_qr, &v_rs, &[Axis(2)], &[Axis(0)])
    //     .into_dimensionality::<Ix3>()
    //     .unwrap();
    let tmp2:Array3<f64> = g0_lr.dot(&tmp.into_shape((n_at,q_qr_dim_1*v_rs_dim_1)).unwrap()).into_shape((n_at,q_qr_dim_1,v_rs_dim_1)).unwrap();
    // let tmp2: Array3<f64> = tensordot(&g0_lr, &tmp, &[Axis(1)], &[Axis(0)])
    //     .into_dimensionality::<Ix3>()
    //     .unwrap();
    let tmp2_swapped = tmp2.permuted_axes([0,2,1]).as_standard_layout().into_shape((n_at*v_rs_dim_1,q_qr_dim_1)).unwrap().to_owned();
    let q_ps_swapped = q_ps.permuted_axes([1,0,2]).as_standard_layout().into_shape((q_ps_dim_1,n_at*q_ps_dim_2)).unwrap().to_owned();
    hplus_pq = hplus_pq - q_ps_swapped.dot(&tmp2_swapped);
    // hplus_pq = hplus_pq
    //     - tensordot(&q_ps, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
    //     .into_dimensionality::<Ix2>()
    //     .unwrap();
    return hplus_pq;
}

pub fn h_plus_no_lr(
    g0: ArrayView2<f64>,
    q_pq: ArrayView3<f64>,
    q_rs: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> (Array2<f64>) {
    // term 1
    let n_at:usize = q_pq.dim().0;
    let v_rs_dim_0:usize = v_rs.dim().0;
    let v_rs_dim_1:usize = v_rs.dim().1;
    let q_rs_dim_1:usize = q_rs.dim().1;
    let q_rs_dim_2:usize = q_rs.dim().2;
    let q_pq_dim_1:usize = q_pq.dim().1;
    let q_pq_dim_2:usize = q_pq.dim().2;

    let tmp:Array1<f64> = q_rs.into_shape((n_at,q_rs_dim_1*q_rs_dim_2)).unwrap().dot(&v_rs.into_shape(v_rs_dim_0*v_rs_dim_1).unwrap());
    // let tmp: Array1<f64> = tensordot(&q_rs, &v_rs, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
    //     .into_dimensionality::<Ix1>()
    //     .unwrap();
    let tmp2: Array1<f64> = g0.dot(&tmp);
    let hplus_pq:Array2<f64> = 4.0 * tmp2.dot(&q_pq.into_shape((n_at,q_pq_dim_1*q_pq_dim_2)).unwrap()).into_shape((q_pq_dim_1,q_pq_dim_2)).unwrap();
    // let hplus_pq: Array2<f64> = 4.0
    //     * tensordot(&q_pq, &tmp2, &[Axis(0)], &[Axis(0)])
    //     .into_dimensionality::<Ix2>()
    //     .unwrap();
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
pub fn gradient_v_rep(atoms: &[Atom], v_rep: &RepulsivePotential) -> Array1<f64> {
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

pub fn zvector_lc(
    a_diag: ArrayView2<f64>,
    b_matrix: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array2<f64>) {
    // Parameters:
    // ===========
    // A: linear operator, such that A(X) = A.X
    // Adiag: diagonal elements of A-matrix, with dimension (nocc,nvirt)
    // B: right hand side of equation, (nocc,nvirt, k)
    let maxiter: usize = 10000;
    let conv:f64 = 1.0e-16;

    let n_occ: usize = b_matrix.dim().0;
    let n_virt: usize = b_matrix.dim().1;
    let n_at:usize = qtrans_ov.dim().0;
    let kmax: usize = n_occ * n_virt;

    // bs are expansion vectors
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let mut bs: Array2<f64> = &a_inv * &b_matrix;

    let mut rhs_2: Array1<f64> = Array::zeros((kmax));
    let mut rkm1: Array1<f64> = Array::zeros((kmax));
    let mut pkm1: Array1<f64> = Array::zeros((kmax));
    let rhs:Array1<f64> = b_matrix.into_shape(kmax).unwrap().to_owned();

    // create new arrays for transition charges of specific shapes,
    // which are required by the mult_apb_v_routine
    let tmp_q_vv: ArrayView2<f64> = qtrans_vv
        .into_shape((n_virt * n_at, n_virt))
        .unwrap();
    let tmp_q_oo: ArrayView2<f64> = qtrans_oo
        .into_shape((n_at * n_occ, n_occ))
        .unwrap();
    let tmp_q_ov_swapped: ArrayView3<f64> = qtrans_ov.permuted_axes([0,2,1]);
    let tmp_q_ov_shape_1: Array2<f64> =
        tmp_q_ov_swapped.as_standard_layout().to_owned().into_shape((n_at * n_virt, n_occ)).unwrap();
    let mut tmp_q_ov_swapped_2: ArrayView3<f64> = qtrans_ov.permuted_axes([1,0,2]);
    let tmp_q_ov_shape_2: Array2<f64> =
        tmp_q_ov_swapped_2.as_standard_layout().to_owned().into_shape((n_occ, n_at * n_virt))
            .unwrap();

    let apbv:Array2<f64> = mult_apb_v(
        g0,
        g0_lr,
        qtrans_oo,
        qtrans_vv,
        qtrans_ov,
        tmp_q_oo.view(),
        tmp_q_vv.view(),
        tmp_q_ov_shape_1.view(),
        tmp_q_ov_shape_2.view(),
        a_diag,
        bs.view(),
        n_occ,
        n_virt,
        multiplicity,
        spin_couplings,
    );

    rkm1 = apbv.into_shape(kmax).unwrap();
    rhs_2 = bs.into_shape(kmax).unwrap();
    rkm1 = rhs - rkm1;
    pkm1 = rkm1.clone();

    for it in 0..maxiter {
        let apbv:Array2<f64> = mult_apb_v(
            g0,
            g0_lr,
            qtrans_oo,
            qtrans_vv,
            qtrans_ov,
            tmp_q_oo.view(),
            tmp_q_vv.view(),
            tmp_q_ov_shape_1.view(),
            tmp_q_ov_shape_2.view(),
            a_diag,
            pkm1.view().into_shape((n_occ,n_virt)).unwrap(),
            n_occ,
            n_virt,
            multiplicity,
            spin_couplings,
        );
        let apk:Array1<f64> = apbv.into_shape((kmax)).unwrap();

        let tmp1:f64 = rkm1.dot(&rkm1);
        let tmp2:f64 = pkm1.dot(&apk);

        rhs_2 = rhs_2 + (tmp1/tmp2) * &pkm1;
        rkm1 = rkm1 - (tmp1/tmp2) * &apk;

        let tmp2:f64 = rkm1.dot(&rkm1);

        if tmp2 <= conv{
            break;
        }
        pkm1 = (tmp2/tmp1) * &pkm1 + &rkm1;
    }

    let out:Array2<f64> = rhs_2.into_shape((n_occ,n_virt)).unwrap();
    return out;
}

pub fn zvector_no_lc(
    a_diag: ArrayView2<f64>,
    b_matrix: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array2<f64>) {
    // Parameters:
    // ===========
    // A: linear operator, such that A(X) = A.X
    // Adiag: diagonal elements of A-matrix, with dimension (nocc,nvirt)
    // B: right hand side of equation, (nocc,nvirt, k)
    let maxiter: usize = 10000;
    let conv:f64 = 1.0e-16;

    let n_occ: usize = b_matrix.dim().0;
    let n_virt: usize = b_matrix.dim().1;
    let kmax: usize = n_occ * n_virt;
    let n_at:usize = qtrans_ov.dim().0;

    // bs are expansion vectors
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let mut bs: Array2<f64> = &a_inv * &b_matrix;

    let mut rhs_2: Array1<f64> = Array::zeros((kmax));
    let mut rkm1: Array1<f64> = Array::zeros((kmax));
    let mut pkm1: Array1<f64> = Array::zeros((kmax));
    let rhs:Array1<f64> = b_matrix.into_shape(kmax).unwrap().to_owned();

    let apbv:Array2<f64> = mult_apb_v_no_lc(
        g0,
        qtrans_ov,
        a_diag,
        bs.view(),
        n_occ,
        n_virt,
        multiplicity,
        spin_couplings,
    );

    rkm1 = apbv.into_shape(kmax).unwrap();
    rhs_2 = bs.into_shape(kmax).unwrap();
    rkm1 = rhs - rkm1;
    pkm1 = rkm1.clone();

    for it in 0..maxiter {
        let apbv:Array2<f64> = mult_apb_v_no_lc(
            g0,
            qtrans_ov,
            a_diag,
            pkm1.view().into_shape((n_occ,n_virt)).unwrap(),
            n_occ,
            n_virt,
            multiplicity,
            spin_couplings,
        );
        let apk:Array1<f64> = apbv.into_shape((kmax)).unwrap();

        let tmp1:f64 = rkm1.dot(&rkm1);
        let tmp2:f64 = pkm1.dot(&apk);

        rhs_2 = rhs_2 + (tmp1/tmp2) * &pkm1;
        rkm1 = rkm1 - (tmp1/tmp2) * &apk;

        let tmp2:f64 = rkm1.dot(&rkm1);

        if tmp2 <= conv{
            break;
        }
        pkm1 = (tmp2/tmp1) * &pkm1 + &rkm1;
    }

    let out:Array2<f64> = rhs_2.into_shape((n_occ,n_virt)).unwrap();
    return out;
}

fn mult_apb_v(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
    qtrans_oo_reshaped:ArrayView2<f64>,
    qtrans_vv_reshaped:ArrayView2<f64>,
    qtrans_ov_reshaped_1:ArrayView2<f64>,
    qtrans_ov_reshaped_2:ArrayView2<f64>,
    omega: ArrayView2<f64>,
    vs: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array2<f64>) {
    let n_at:usize = qtrans_ov.dim().0;

    let gamma_equiv: Array2<f64> = if multiplicity == 1 {
        gamma.to_owned()
    } else if multiplicity == 3 {
        Array2::from_diag(&spin_couplings)
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
    };

    // 1st term - KS orbital energy differences
    let mut u_l: Array2<f64> = &omega * &vs;

    // 2nd term - Coulomb
    let mut tmp21: Array1<f64> = Array1::zeros(n_at);
    for at in (0..n_at){
        let tmp: Array2<f64> = &qtrans_ov.slice(s![at, .., ..]) * &vs;
        tmp21[at] = tmp.sum()
    }

    let tmp22: Array1<f64> = 4.0 * gamma_equiv.dot(&tmp21);
    for at in (0..n_at).into_iter() {
        u_l = u_l + &qtrans_ov.slice(s![at, .., ..]) * tmp22[at];
    }

    // 3rd term - Exchange
    let tmp31: Array3<f64> = qtrans_vv_reshaped
        .dot(&vs.t())
        .into_shape((n_at, n_virt, n_occ))
        .unwrap();

    let tmp31_reshaped: Array2<f64> = tmp31.into_shape((n_at, n_virt * n_occ)).unwrap();
    let mut tmp32: Array3<f64> = gamma_lr
        .dot(&tmp31_reshaped)
        .into_shape((n_at, n_virt, n_occ))
        .unwrap();
    tmp32.swap_axes(1, 2);
    let tmp32 = tmp32.as_standard_layout();

    let tmp33: Array2<f64> = qtrans_oo_reshaped
        .t()
        .dot(&tmp32.into_shape((n_at * n_occ, n_virt)).unwrap());
    u_l = u_l - tmp33;

    // 4th term - Exchange
    let tmp41: Array3<f64> = qtrans_ov_reshaped_1
        .dot(&vs)
        .into_shape((n_at, n_virt, n_virt))
        .unwrap();
    let tmp41_reshaped: Array2<f64> = tmp41.into_shape((n_at, n_virt * n_virt)).unwrap();
    let mut tmp42: Array3<f64> = gamma_lr
        .dot(&tmp41_reshaped)
        .into_shape((n_at, n_virt, n_virt))
        .unwrap();
    tmp42.swap_axes(1, 2);
    let tmp42 = tmp42.as_standard_layout();

    let tmp43: Array2<f64> =
        qtrans_ov_reshaped_2.dot(&tmp42.into_shape((n_at * n_virt, n_virt)).unwrap());
    u_l = u_l - tmp43;

    return u_l;
}

fn mult_apb_v_no_lc(
    gamma: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    vs: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array2<f64>) {
    let n_at:usize = qtrans_ov.dim().0;
    let gamma_equiv: Array2<f64> = if multiplicity == 1 {
        gamma.to_owned()
    } else if multiplicity == 3 {
        Array2::from_diag(&spin_couplings)
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
    };

    // 1st term - KS orbital energy differences
    let mut u_l: Array2<f64> = &omega * &vs;

    // 2nd term - Coulomb
    let mut tmp21: Array1<f64> = Array1::zeros(n_at);
    for at in (0..n_at){
        let tmp: Array2<f64> = &qtrans_ov.slice(s![at, .., ..]) * &vs;
        tmp21[at] = tmp.sum()
    }
    let tmp22: Array1<f64> = 4.0 * gamma_equiv.dot(&tmp21);
    for at in (0..n_at).into_iter() {
        u_l = u_l + &qtrans_ov.slice(s![at, .., ..]) * tmp22[at];
    }

    return u_l;
}