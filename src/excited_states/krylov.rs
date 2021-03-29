use ndarray::{Array, Array1, Axis, Array2, Array3, ArrayView1, ArrayView3, ArrayView2};
use ndarray_einsum_beta::tensordot;
use crate::excited_states::a_and_b::{get_apbv_fortran, get_apbv_fortran_no_lc};
use ndarray_linalg::{Solve, QR};
use crate::excited_states::helpers::norm_special;

pub fn krylov_solver_zvector(
    a_diag: ArrayView2<f64>,
    b_matrix: ArrayView3<f64>,
    x_0: Option<Array3<f64>>,
    maxiter: Option<usize>,
    conv: Option<f64>,
    g0: ArrayView2<f64>,
    g0_lr: Option<ArrayView2<f64>>,
    qtrans_oo: Option<ArrayView3<f64>>,
    qtrans_vv: Option<ArrayView3<f64>>,
    qtrans_ov: ArrayView3<f64>,
    lc: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    // Parameters:
    // ===========
    // A: linear operator, such that A(X) = A.X
    // Adiag: diagonal elements of A-matrix, with dimension (nocc,nvirt)
    // B: right hand side of equation, (nocc,nvirt, k)
    // X0: initial guess vectors or None

    let maxiter: usize = maxiter.unwrap_or(1000);
    let conv: f64 = conv.unwrap_or(1.0e-14);

    let n_occ: usize = b_matrix.dim().0;
    let n_virt: usize = b_matrix.dim().1;
    let k: usize = b_matrix.dim().2;
    // number of vectors
    let kmax: usize = n_occ * n_virt;
    let mut l: usize = k;

    // bs are expansion vectors
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let mut bs: Array3<f64> = Array::zeros((n_occ, n_virt, k));

    if x_0.is_none() {
        for i in 0..k {
            bs.slice_mut(s![.., .., i])
                .assign(&(&a_inv * &b_matrix.slice(s![.., .., i])));
        }
    } else {
        bs = x_0.unwrap();
    }

    let mut x_matrix: Array3<f64> = Array::zeros((n_occ, n_virt, k));
    let mut temp_old: Array3<f64> = bs.clone();

    for it in 0..maxiter {
        // representation of A in the basis of expansion vectors
        let mut temp: Array3<f64> = Array3::zeros((n_occ, n_virt, l));
        if it == 0 {
            // temp = get_apbv(
            //     &g0,
            //     &g0_lr,
            //     &qtrans_oo,
            //     &qtrans_vv,
            //     &qtrans_ov,
            //     &a_diag,
            //     &bs,
            //     lc,
            //     multiplicity,
            //     spin_couplings,
            // );
            if lc == 1 {
                temp = get_apbv_fortran(
                    &g0,
                    &g0_lr.clone().unwrap(),
                    &qtrans_oo.clone().unwrap(),
                    &qtrans_vv.clone().unwrap(),
                    &qtrans_ov,
                    &a_diag,
                    &bs,
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    l,
                    multiplicity,
                    spin_couplings,
                );
            } else {
                temp = get_apbv_fortran_no_lc(
                    &g0,
                    &qtrans_ov,
                    &a_diag,
                    &bs,
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    l,
                    multiplicity,
                    spin_couplings,
                );
            }
        } else {
            //let temp_new_vec_alt: Array3<f64> = get_apbv(
            //    &g0,
            //    &g0_lr,
            //    &qtrans_oo,
            //    &qtrans_vv,
            //    &qtrans_ov,
            //    &a_diag,
            //    &bs.slice(s![.., .., l - 2..l]).to_owned(),
            //    lc,
            //    multiplicity,
            //    spin_couplings,
            //);
            let mut temp_new_vec: Array3<f64> = Array3::zeros((n_occ, n_virt, (l - 2..l).len()));
            if lc == 1 {
                temp_new_vec = get_apbv_fortran(
                    &g0,
                    &g0_lr.clone().unwrap(),
                    &qtrans_oo.clone().unwrap(),
                    &qtrans_vv.clone().unwrap(),
                    &qtrans_ov,
                    &a_diag,
                    &bs.slice(s![.., .., l - 2..l]).to_owned(),
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    (l - 2..l).len(),
                    multiplicity,
                    spin_couplings,
                );
            } else {
                temp_new_vec = get_apbv_fortran_no_lc(
                    &g0,
                    &qtrans_ov,
                    &a_diag,
                    &bs.slice(s![.., .., l - 2..l]).to_owned(),
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    (l - 2..l).len(),
                    multiplicity,
                    spin_couplings,
                );
            }
            // println!("Temp new alt {}",temp_new_vec_alt);
            // println!("Temp vec new {}",temp_new_vec)
            temp.slice_mut(s![.., .., ..l - 1]).assign(&temp_old);
            temp.slice_mut(s![.., .., l - 2..l]).assign(&temp_new_vec);
        }
        // let temp:Array3<f64> = get_apbv(
        //         &g0,
        //         &g0_lr,
        //         &qtrans_oo,
        //         &qtrans_vv,
        //         &qtrans_ov,
        //         &a_diag,
        //         &bs,
        //         lc,
        //         multiplicity,
        //         spin_couplings,
        //     );

        temp_old = temp.clone();

        // println!("Temp {}",temp);
        // println!("Bs {}",bs);

        let a_b: Array2<f64> = tensordot(&bs, &temp, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();

        // let a_b: Array2<f64> = tensordot(
        //     &bs,
        //     &get_apbv(
        //         &g0,
        //         &g0_lr,
        //         &qtrans_oo,
        //         &qtrans_vv,
        //         &qtrans_ov,
        //         &a_diag,
        //         &bs,
        //         lc,
        //         multiplicity,
        //         spin_couplings,
        //     ),
        //     &[Axis(0), Axis(1)],
        //     &[Axis(0), Axis(1)],
        // )
        // .into_dimensionality::<Ix2>()
        // .unwrap();
        // RHS in basis of expansion vectors
        let b_b: Array2<f64> = tensordot(&bs, &b_matrix, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();

        // solve
        let mut x_b: Array2<f64> = Array2::zeros((k, l));
        for i in 0..k {
            x_b.slice_mut(s![i, ..])
                .assign((&a_b.solve(&b_b.slice(s![.., i])).unwrap()));
        }
        x_b = x_b.reversed_axes();

        // transform solution vector back into canonical basis
        x_matrix = tensordot(&bs, &x_b, &[Axis(2)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();
        // residual vectors
        let mut w_res: Array3<f64> = Array3::zeros((x_matrix.raw_dim()));
        if lc == 1 {
            w_res = get_apbv_fortran(
                &g0,
                &g0_lr.clone().unwrap(),
                &qtrans_oo.clone().unwrap(),
                &qtrans_vv.clone().unwrap(),
                &qtrans_ov,
                &a_diag,
                &x_matrix,
                qtrans_ov.dim().0,
                n_occ,
                n_virt,
                x_matrix.dim().2,
                multiplicity,
                spin_couplings,
            );
        } else {
            w_res = get_apbv_fortran_no_lc(
                &g0,
                &qtrans_ov,
                &a_diag,
                &x_matrix,
                qtrans_ov.dim().0,
                n_occ,
                n_virt,
                x_matrix.dim().2,
                multiplicity,
                spin_couplings,
            );
        }
        w_res = &w_res - &b_matrix;
        // let w_res: Array3<f64> = &get_apbv(
        //     &g0,
        //     &g0_lr,
        //     &qtrans_oo,
        //     &qtrans_vv,
        //     &qtrans_ov,
        //     &a_diag,
        //     &x_matrix,
        //     lc,
        //     multiplicity,
        //     spin_couplings,
        // ) - &b_matrix;

        let mut norms: Array1<f64> = Array::zeros(k);
        for i in 0..k {
            norms[i] = norm_special(&w_res.slice(s![.., .., i]).to_owned());
        }
        // check if all values of the norms are under the convergence criteria
        let indices_norms: Array1<usize> = norms
            .indexed_iter()
            .filter_map(|(index, &item)| if item < conv { Some(index) } else { None })
            .collect();
        if indices_norms.len() == norms.len() {
            break;
        }

        // # enlarge dimension of subspace by dk vectors
        // # At most k new expansion vectors are added
        let dkmax = (kmax - l).min(k);
        // # count number of non-converged vectors
        // # residual vectors that are zero cannot be used as new expansion vectors
        //1.0e-16
        let eps = 0.01 * conv;
        // version for nc = np.sum(norms > eps)
        let indices_norm_over_eps: Array1<usize> = norms
            .indexed_iter()
            .filter_map(|(index, &item)| if item > eps { Some(index) } else { None })
            .collect();
        // let mut norms_over_eps: Array1<f64> = Array::zeros(indices_norm_over_eps.len());
        // for i in 0..indices_norm_over_eps.len() {
        //     norms_over_eps[i] = norms[indices_norm_over_eps[i]];
        // }
        //let nc: f64 = norms_over_eps.sum();
        let nc: usize = indices_norm_over_eps.len();
        let dk: usize = dkmax.min(nc);

        let mut Qs: Array3<f64> = Array::zeros((n_occ, n_virt, dk));
        let mut nb: i32 = 0;

        for i in 0..dkmax {
            if norms[i] > eps {
                Qs.slice_mut(s![.., .., nb])
                    .assign(&((&a_inv) * &w_res.slice(s![.., .., i])));
                nb += 1;
            }
        }

        assert!(nb as usize == dk);
        // new expansion vectors are bs + Qs
        let mut bs_new: Array3<f64> = Array::zeros((n_occ, n_virt, l + dk));
        bs_new.slice_mut(s![.., .., ..l]).assign(&bs);
        bs_new.slice_mut(s![.., .., l..]).assign(&Qs);

        // QR decomposition as in hermitian davidson
        // to receive orthogonalized vectors
        // alternative: implement gram schmidt orthogonalization
        // Alexander also uses this method in hermitian davidson

        let nvec: usize = l + dk;
        let bs_flat: Array2<f64> = bs_new.into_shape((n_occ * n_virt, nvec)).unwrap();
        let (Q, R): (Array2<f64>, Array2<f64>) = bs_flat.qr().unwrap();
        bs = Q.into_shape((n_occ, n_virt, nvec)).unwrap();
        l = bs.dim().2;
    }
    return x_matrix;
}