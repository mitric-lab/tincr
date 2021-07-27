use ndarray::prelude::*;
use ndarray_einsum_beta::tensordot;
use ndarray_linalg::{Eigh, UPLO, SymmetricSqrt, QR};
use crate::excited_states::helpers::{initial_expansion_vectors, norm_special, matrix_v_product, matrix_v_product_fortran, reorder_vectors_lambda2};
use crate::excited_states::a_and_b::{get_apbv_fortran, get_ambv_fortran};

pub fn hermitian_davidson(
    gamma: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    omega_shift: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
    XmYguess: Option<ArrayView3<f64>>,
    XpYguess: Option<ArrayView3<f64>>,
    Oia: ArrayView2<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
    nstates: Option<usize>,
    ifact: Option<usize>,
    maxiter: Option<usize>,
    conv: Option<f64>,
    l2_treshold: Option<f64>,
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    // f A-B is diagonal the TD-DFT equations can be made hermitian
    //       (A-B)^(1/2).(A+B).(A-B)^(1/2).T = Omega^2 T
    //                    R               .T = Omega^2 T

    let nstates: usize = nstates.unwrap_or(4);
    let ifact: usize = ifact.unwrap_or(1);
    let maxiter: usize = maxiter.unwrap_or(10);
    let conv: f64 = conv.unwrap_or(1.0e-8);
    let l2_treshold: f64 = l2_treshold.unwrap_or(0.5);

    let omega2: Array2<f64> = omega.map(|omega| ndarray_linalg::Scalar::powi(omega, 2));
    let omega_sq: Array2<f64> = omega.map(|omega| ndarray_linalg::Scalar::sqrt(omega));
    let omega_sq_inv: Array2<f64> = 1.0 / &omega_sq;
    let wq_ov: Array3<f64> = &qtrans_ov * &omega_sq;
    //# diagonal elements of R
    let om: Array2<f64> = omega2 + &omega * &omega_shift * 2.0;
    // initial number of expansion vectors
    // at most there are nocc*nvirt excited states
    let kmax = &n_occ * &n_virt;
    let lmax = (&ifact * &nstates).min(kmax);

    let mut bs: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    let mut bs_first: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    if XpYguess.is_none() {
        let omega_guess: Array2<f64> = om.map(|om| ndarray_linalg::Scalar::sqrt(om));
        bs = initial_expansion_vectors(omega_guess.clone(), lmax);
    } else {
        for i in 0..lmax {
            //bs.slice_mut(s![.., .., i]).assign(&(&omega_sq_inv * &XpYguess.unwrap().slice(s![i, .., ..])));
            let tmp_array: Array2<f64> = (&omega_sq_inv * &XpYguess.unwrap().slice(s![i, .., ..]));
            let norm_temp: f64 = norm_special(&tmp_array);
            //tmp_array = tmp_array / norm_special(&tmp_array);
            bs.slice_mut(s![.., .., i])
                .assign(&(&tmp_array / norm_temp));
        }
    }
    let mut l: usize = lmax;
    let k: usize = nstates;
    let mut w: Array1<f64> = Array::zeros(lmax);
    let mut T_new: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    let mut r_bs_first: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    let mut r_bs_old: Array3<f64> = Array::zeros(bs.raw_dim());

    for it in 0..maxiter {
        let lmax: usize = bs.dim().2;
        let mut r_bs: Array3<f64> = Array3::zeros(bs.raw_dim());
        let r_bs_alt: Array3<f64> = matrix_v_product(
            &bs,
            lmax,
            n_occ,
            n_virt,
            &om,
            &wq_ov,
            &gamma,
            multiplicity,
            spin_couplings,
        );
        if it < 2 {
            r_bs = matrix_v_product_fortran(
                &bs,
                lmax,
                n_occ,
                n_virt,
                &om,
                &wq_ov,
                &gamma,
                multiplicity,
                spin_couplings,
            );
        } else {
            let r_bs_new_vec: Array3<f64> = matrix_v_product_fortran(
                &bs.slice(s![.., .., l - (2 * nstates)..l]).to_owned(),
                (2 * nstates),
                n_occ,
                n_virt,
                &om,
                &wq_ov,
                &gamma,
                multiplicity,
                spin_couplings,
            );
            r_bs.slice_mut(s![.., .., ..l - (2 * nstates)])
                .assign(&r_bs_old.slice(s![.., .., ..l - (2 * nstates)]));
            r_bs.slice_mut(s![.., .., l - (2 * nstates)..l])
                .assign(&r_bs_new_vec);
            r_bs.slice_mut(s![.., .., 0..nstates])
                .assign(&(r_bs_old.slice(s![.., .., 0..nstates]).to_owned() * (-1.0)));
        }
        r_bs_old = r_bs.clone();
        // shape of Hb: (lmax, lmax)

        let Hb: Array2<f64> = tensordot(&bs, &r_bs, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();
        let (w2, Tb): (Array1<f64>, Array2<f64>) = Hb.eigh(UPLO::Upper).unwrap();
        // shape of T : (n_occ,n_virt,lmax)
        let T: Array3<f64> = tensordot(&bs, &Tb, &[Axis(2)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();

        // In DFTBaby a selector of symmetry could be used here
        let temp: (Array1<f64>, Array3<f64>);

        if l2_treshold > 0.0 {
            temp = reorder_vectors_lambda2(&Oia, &w2, &T, l2_treshold);
        } else {
            temp = (Array::zeros(lmax), Array::zeros((n_occ, n_virt, lmax)));
        }
        let (w2_new, T_temp): (Array1<f64>, Array3<f64>) = temp;
        T_new = T_temp;

        w = w2_new.mapv(f64::sqrt);
        //residual vectors

        let W_res: Array3<f64> = matrix_v_product_fortran(
            &T,
            lmax,
            n_occ,
            n_virt,
            &om,
            &wq_ov,
            &gamma,
            multiplicity,
            spin_couplings,
        ) - &T * &w2_new;

        // let W_res: Array3<f64> = matrix_v_product(&T, lmax, n_occ, n_virt, &om, &wq_ov, &gamma, multiplicity, spin_couplings)
        //     - einsum("k,ijk->ijk", &[&w2_new, &T])
        //         .unwrap()
        //         .into_dimensionality::<Ix3>()
        //         .unwrap();
        // println!("einsum: {}", now.elapsed().as_micros());

        let mut norms_res: Array1<f64> = Array::zeros(k);
        for i in 0..k {
            norms_res[i] = norm_special(&W_res.slice(s![.., .., i]).to_owned());
        }
        // check if all norms are below the convergence criteria
        // maybe there is a faster method
        let indices_norms: Array1<usize> = norms_res
            .indexed_iter()
            .filter_map(|(index, &item)| if item < conv { Some(index) } else { None })
            .collect();
        if indices_norms.len() == norms_res.len() {
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

        let indices_norm_over_eps: Array1<usize> = norms_res
            .indexed_iter()
            .filter_map(|(index, &item)| if item > eps { Some(index) } else { None })
            .collect();

        let nc: usize = indices_norm_over_eps.len();
        let dk: usize = dkmax.min(nc as usize);

        let mut Qs: Array3<f64> = Array::zeros((n_occ, n_virt, dk));
        let mut nb: i32 = 0;

        // # select new expansion vectors among the residual vectors
        for i in 0..dkmax {
            let wD: Array2<f64> = w[i] - &omega.to_owned();
            // quite the ugly method in order to reproduce
            // indx = abs(wD) < 1.0e-6
            // wD[indx] = 1.0e-6 * omega[indx]
            // from numpy

            let temp: Array2<f64> = wD.map(|wD| if wD < &1.0e-6 { 1.0e-6 } else { 0.0 });
            let temp_2: Array2<f64> = wD.map(|&wD| if wD < 1.0e-6 { 0.0 } else { wD });
            let mut wD_new: Array2<f64> = &temp * &omega.to_owned();
            wD_new = wD_new + temp_2;

            if norms_res[i] > eps {
                Qs.slice_mut(s![.., .., nb])
                    .assign(&((1.0 / &wD_new) * W_res.slice(s![.., .., i])));
                nb += 1;
            }
        }

        // new expansion vectors are bs + Qs
        let mut bs_new: Array3<f64> = Array::zeros((n_occ, n_virt, l + dk));
        bs_new.slice_mut(s![.., .., ..l]).assign(&bs);
        bs_new.slice_mut(s![.., .., l..]).assign(&Qs);

        //QR decomposition
        let nvec: usize = l + dk;

        let bs_flat: Array2<f64> = bs_new.into_shape((n_occ * n_virt, nvec)).unwrap();
        let (Q, R): (Array2<f64>, Array2<f64>) = bs_flat.qr().unwrap();
        bs = Q.into_shape((n_occ, n_virt, nvec)).unwrap();
        l = bs.dim().2;
    }
    let mut Omega: Vec<f64> = w.to_vec();
    Omega.sort_by(|&i, &j| i.partial_cmp(&j).unwrap());
    let Omega: Array1<f64> = Array::from(Omega).slice(s![..k]).to_owned();
    let mut XpY: Array3<f64> = Array::zeros((n_occ, n_virt, k));
    let mut XmY: Array3<f64> = Array::zeros((n_occ, n_virt, k));
    let mut c_matrix: Array3<f64> = Array::zeros((n_occ, n_virt, k));

    for i in 0..k {
        let temp_T: Array2<f64> = T_new.slice(s![.., .., i]).to_owned();
        // # X+Y = 1/sqrt(Omega)*(A-B)^(1/2).T
        XpY.slice_mut(s![.., .., i])
            .assign(&(&(&omega_sq / Omega[i].sqrt()) * &temp_T));
        // # X-Y = sqrt(Omega)*(A-B)^(-1).(X+Y)
        XmY.slice_mut(s![.., .., i])
            .assign(&(Omega[i].sqrt() * &omega_sq_inv * &temp_T));
        // # C = (A-B)^(-1/2).(X+Y) * sqrt(Omega)
        c_matrix.slice_mut(s![.., .., i]).assign(&temp_T);
    }
    // # XmY, XpY and C have shape (nocc,nvirt, nstates)
    // # bring the last axis to the front
    XpY.swap_axes(1, 2);
    XpY.swap_axes(0, 1);
    XmY.swap_axes(1, 2);
    XmY.swap_axes(0, 1);
    c_matrix.swap_axes(1, 2);
    c_matrix.swap_axes(0, 1);

    return (Omega, c_matrix, XmY, XpY);
}

pub fn non_hermitian_davidson(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
    XmYguess: Option<ArrayView3<f64>>,
    XpYguess: Option<ArrayView3<f64>>,
    w_guess: Option<ArrayView1<f64>>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
    nstates: Option<usize>,
    ifact: Option<usize>,
    maxiter: Option<usize>,
    conv: Option<f64>,
    l2_treshold: Option<f64>,
    lc: Option<usize>,
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    // set values or defaults
    let nstates: usize = nstates.unwrap_or(4);
    let ifact: usize = ifact.unwrap_or(1);
    let maxiter: usize = maxiter.unwrap_or(100);
    let conv: f64 = conv.unwrap_or(1.0e-5);
    let l2_treshold: f64 = l2_treshold.unwrap_or(0.5);
    let lc: usize = lc.unwrap_or(1);

    // at most there are nocc*nvirt excited states
    let kmax = n_occ * n_virt;
    // To achieve fast convergence the solution vectors from a nearby geometry
    // should be used as initial expansion vectors

    let lmax: usize = (ifact * nstates).min(kmax);
    let mut bs: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));

    if XpYguess.is_none() {
        bs = initial_expansion_vectors(omega.to_owned(), lmax);
    } else {
        // For the expansion vectors we use X+Y
        let lmaxp: usize = nstates;
        // and X-Y if the vectors space has not been exhausted yet
        let lmaxm: usize = nstates.min(kmax - nstates);
        let lmax: usize = lmaxp + lmaxm;

        bs = Array::zeros((n_occ, n_virt, lmax));
        // bring axis with state indeces to the back
        let mut XmY_temp: Array3<f64> = XmYguess.unwrap().to_owned();
        XmY_temp.swap_axes(0, 1);
        XmY_temp.swap_axes(1, 2);
        let mut XpY_temp: Array3<f64> = XpYguess.unwrap().to_owned();
        XpY_temp.swap_axes(0, 1);
        XpY_temp.swap_axes(1, 2);

        bs.slice_mut(s![.., .., ..lmaxp]).assign(&XmY_temp);
        bs.slice_mut(s![.., .., lmaxp..])
            .assign(&XpY_temp.slice(s![.., .., ..lmaxm]));

        for i in 0..lmax {
            let temp: f64 = norm_special(&bs.slice(s![.., .., i]).to_owned());
            let tmp_array: Array2<f64> = bs.slice(s![.., .., i]).to_owned();
            bs.slice_mut(s![.., .., i]).assign(&(tmp_array / temp));
        }
    }
    let mut l: usize = lmax;
    let k: usize = nstates;

    let mut w: Array1<f64> = Array::zeros(lmax);
    let mut Tb: Array2<f64> = Array::zeros((lmax, lmax));

    let mut l_canon: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    let mut r_canon: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));

    let mut bp_old: Array3<f64> = bs.clone();
    let mut bm_old: Array3<f64> = bs.clone();
    let mut l_prev: usize = l;

    for it in 0..maxiter {
        let lmax: usize = bs.dim().2;
        println!("Iteration {}", it);
        // println!("BS {}",bs.slice(s![0,5,..]));

        if XpYguess.is_none() || it > 0 {
            let mut bp: Array3<f64> = Array3::zeros((n_occ, n_virt, l));
            let mut bm: Array3<f64> = Array3::zeros((n_occ, n_virt, l));

            if it < 2 {
                bp = get_apbv_fortran(
                    &gamma,
                    &gamma_lr,
                    &qtrans_oo,
                    &qtrans_vv,
                    &qtrans_ov,
                    &omega,
                    &bs,
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    l,
                    multiplicity,
                    spin_couplings,
                );
                bm = get_ambv_fortran(
                    &gamma,
                    &gamma_lr,
                    &qtrans_oo,
                    &qtrans_vv,
                    &qtrans_ov,
                    &omega,
                    &bs,
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    l,
                );
            } else {
                let bp_new_vec: Array3<f64> = get_apbv_fortran(
                    &gamma,
                    &gamma_lr,
                    &qtrans_oo,
                    &qtrans_vv,
                    &qtrans_ov,
                    &omega,
                    &bs.slice(s![.., .., l - (3 * nstates)..l]).to_owned(),
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    (3 * nstates),
                    multiplicity,
                    spin_couplings,
                );
                bp.slice_mut(s![.., .., ..l - (3 * nstates)])
                    .assign(&bp_old.slice(s![.., .., ..l - (3 * nstates)]));
                bp.slice_mut(s![.., .., l - (3 * nstates)..l])
                    .assign(&bp_new_vec);
                bp.slice_mut(s![.., .., 0..nstates])
                    .assign(&(bp_old.slice(s![.., .., 0..nstates]).to_owned() * (-1.0)));
                let bm_new_vec: Array3<f64> = get_ambv_fortran(
                    &gamma,
                    &gamma_lr,
                    &qtrans_oo,
                    &qtrans_vv,
                    &qtrans_ov,
                    &omega,
                    &bs.slice(s![.., .., l - (3 * nstates)..l]).to_owned(),
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    (3 * nstates),
                );
                bm.slice_mut(s![.., .., ..l - (3 * nstates)])
                    .assign(&bm_old.slice(s![.., .., ..l - (3 * nstates)]));
                bm.slice_mut(s![.., .., l - (3 * nstates)..l])
                    .assign(&bm_new_vec);
                bm.slice_mut(s![.., .., 0..nstates])
                    .assign(&(bm_old.slice(s![.., .., 0..nstates]).to_owned() * (-1.0)));
            }

            bp_old = bp.clone();
            bm_old = bm.clone();

            // XINCHENG PLEASE CHECK THIS
            //             let bp_alt: Array3<f64> = get_apbv_fortran(
            //                 &gamma,
            //                 &gamma_lr,
            //                 &qtrans_oo,
            //                 &qtrans_vv,
            //                 &qtrans_ov,
            //                 &omega,
            //                 &bs,
            //                 qtrans_ov.dim().0,
            //                 n_occ,
            //                 n_virt,
            //                 l,
            //                 multiplicity,
            //                 spin_couplings,
            //             );
            // XINCHENG PLEASE CHECK THIS

            //             let bm_alt: Array3<f64> = get_ambv_fortran(
            //                 &gamma,
            //                 &gamma_lr,
            //                 &qtrans_oo,
            //                 &qtrans_vv,
            //                 &qtrans_ov,
            //                 &omega,
            //                 &bs,
            //                 qtrans_ov.dim().0,
            //                 n_occ,
            //                 n_virt,
            //                 l,
            //             );

            // # evaluate (A+B).b and (A-B).b
            //let bp_alt: Array3<f64> = get_apbv(
            //     &gamma,
            //     &Some(gamma_lr),
            //     &Some(qtrans_oo),
            //     &Some(qtrans_vv),
            //     &qtrans_ov,
            //     &omega,
            //     &bs,
            //     lc,
            //     multiplicity,
            //     spin_couplings,
            // );
            //let bm_alt: Array3<f64> = get_ambv(
            //    &gamma, &gamma_lr, &qtrans_oo, &qtrans_vv, &qtrans_ov, &omega, &bs, lc,
            //);

            // # M^+ = (b_i, (A+B).b_j)
            let mp: Array2<f64> = tensordot(&bs, &bp, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
                .into_dimensionality::<Ix2>()
                .unwrap();
            // # M^- = (b_i, (A-B).b_j)
            let mm: Array2<f64> = tensordot(&bs, &bm, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
                .into_dimensionality::<Ix2>()
                .unwrap();
            let mmsq: Array2<f64> = mm.ssqrt(UPLO::Upper).unwrap();

            // # Mh is the analog of (A-B)^(1/2).(A+B).(A-B)^(1/2)
            // # in the reduced subspace spanned by the expansion vectors bs
            let mh: Array2<f64> = mmsq.dot(&mp.dot(&mmsq));
            let mh_reversed: Array2<f64> = mh.clone().reversed_axes();
            // check that Mh is hermitian
            let mut subst: Array2<f64> = mh.clone() - mh_reversed;
            subst = subst.map(|subst| subst.abs());
            let err: f64 = subst.sum();

            if err > 1.0e-10 {
                panic!(
                    "Hmm... It seems that Mh is not hermitian. The currect error is {:e}\n\
                        and should be lower than 1.0e-10. If you know what you are doing, try\n\
                        lowering the tolerance. Otherwise check you input!",
                    err
                );
            }

            let tmp: (Array1<f64>, Array2<f64>) = mh.eigh(UPLO::Upper).unwrap();
            let w2: Array1<f64> = tmp.0;
            Tb = tmp.1;
            //In DFTBaby check for selector(symmetry checker)
            w = w2.mapv(f64::sqrt);
            let wsq: Array1<f64> = w.mapv(f64::sqrt);

            // approximate right R = (X+Y) and left L = (X-Y) eigenvectors
            // in the basis bs
            // (X+Y) = (A-B)^(1/2).T / sqrt(w)
            let rb: Array2<f64> = mmsq.dot(&Tb) / wsq;
            // L = (X-Y) = 1/w * (A+B).(X+Y)
            let lb: Array2<f64> = mp.dot(&rb) / &w;
            // check that (Lb^T, Rb) = 1 is fulfilled
            let temp_eye: Array2<f64> = Array::eye(lmax);
            let temp: Array2<f64> = lb.clone().reversed_axes().dot(&rb) - temp_eye;
            let err: f64 = temp.sum();
            if err > 1.0e-3 {
                panic!("Hmm, it seems that (X+Y) and (X-Y) vectors are not orthonormal. The error\n\
                        is {:e} and should be smaller than 1.0e-3. Maybe your molecule just doesn't like you?" , err);
            }
            // transform to the canonical basis Lb -> L, Rb -> R
            l_canon = tensordot(&bs, &lb, &[Axis(2)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            r_canon = tensordot(&bs, &rb, &[Axis(2)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
        } else {
            // bring axis with state indeces to the back
            let mut XmY_temp: Array3<f64> = XmYguess.unwrap().to_owned();
            XmY_temp.swap_axes(0, 1);
            XmY_temp.swap_axes(1, 2);
            let mut XpY_temp: Array3<f64> = XpYguess.unwrap().to_owned();
            XpY_temp.swap_axes(0, 1);
            XpY_temp.swap_axes(1, 2);

            r_canon = XpY_temp;
            l_canon = XmY_temp;
            w = w_guess.unwrap().to_owned();
        }
        // residual vectors
        //let wl = get_apbv(
        //    &gamma,
        //    &Some(gamma_lr),
        //    &Some(qtrans_oo),
        //    &Some(qtrans_vv),
        //    &qtrans_ov,
        //    &omega,
        //    &r_canon,
        //    lc,
        //    multiplicity,
        //    spin_couplings,
        //) - &l_canon * &w;
        let wl = get_apbv_fortran(
            &gamma,
            &gamma_lr,
            &qtrans_oo,
            &qtrans_vv,
            &qtrans_ov,
            &omega,
            &r_canon,
            qtrans_ov.dim().0,
            n_occ,
            n_virt,
            l,
            multiplicity,
            spin_couplings,
        ) - &l_canon * &w;
        //let wr = get_ambv(
        //    &gamma, &gamma_lr, &qtrans_oo, &qtrans_vv, &qtrans_ov, &omega, &l_canon, lc,
        //) - &r_canon * &w;
        let wr = get_ambv_fortran(
            &gamma,
            &gamma_lr,
            &qtrans_oo,
            &qtrans_vv,
            &qtrans_ov,
            &omega,
            &l_canon,
            qtrans_ov.dim().0,
            n_occ,
            n_virt,
            l,
        ) - &r_canon * &w;

        //norms
        let mut norms: Array1<f64> = Array::zeros(k);
        let mut norms_l: Array1<f64> = Array::zeros(k);
        let mut norms_r: Array1<f64> = Array::zeros(k);
        for i in 0..k {
            norms_l[i] = norm_special(&wl.slice(s![.., .., i]).to_owned());
            norms_r[i] = norm_special(&wr.slice(s![.., .., i]).to_owned());
            norms[i] = norms_l[i] + norms_r[i];
        }
        // check for convergence
        let indices_norms: Array1<usize> = norms
            .indexed_iter()
            .filter_map(|(index, &item)| if item < conv { Some(index) } else { None })
            .collect();
        println!("Norms davidson {}", norms);
        if indices_norms.len() == norms.len() && it > 0 {
            break;
        }

        //  enlarge dimension of subspace by dk vectors
        //  At most 2*k new expansion vectors are added
        let dkmax = (kmax - l).min(2 * k);
        // # count number of non-converged vectors
        // # residual vectors that are zero cannot be used as new expansion vectors
        // 1.0e-16
        let eps = 0.01 * conv;

        let indices_norm_r_over_eps: Array1<usize> = norms_r
            .indexed_iter()
            .filter_map(|(index, &item)| if item > eps { Some(index) } else { None })
            .collect();
        let indices_norm_l_over_eps: Array1<usize> = norms_l
            .indexed_iter()
            .filter_map(|(index, &item)| if item > eps { Some(index) } else { None })
            .collect();

        let nc_l: usize = indices_norm_r_over_eps.len();
        let nc_r: usize = indices_norm_l_over_eps.len();
        // Half the new expansion vectors should come from the left residual vectors
        // the other half from the right residual vectors.
        let dk_r: usize = ((dkmax as f64 / 2.0) as usize).min(nc_l);
        let dk_l: usize = (dkmax - dk_r).min(nc_r);
        let dk: usize = dk_r + dk_l;

        let mut Qs: Array3<f64> = Array::zeros((n_occ, n_virt, dk));
        let mut nb: usize = 0;
        // select new expansion vectors among the non-converged left residual vectors
        for i in 0..k {
            if nb == dk {
                //got enough new expansion vectors
                break;
            }
            let wD: Array2<f64> = w[i] - &omega.to_owned();
            // quite the ugly method in order to reproduce
            // indx = abs(wD) < 1.0e-6
            // wD[indx] = 1.0e-6 * omega[indx]
            // from numpy
            let temp: Array2<f64> = wD.map(|wD| if wD < &1.0e-6 { 1.0e-6 } else { 0.0 });
            let temp_2: Array2<f64> = wD.map(|&wD| if wD < 1.0e-6 { 0.0 } else { wD });
            let mut wD_new: Array2<f64> = &temp * &omega.to_owned();

            wD_new = wD_new + temp_2;
            if norms_l[i] > eps {
                Qs.slice_mut(s![.., .., nb])
                    .assign(&((1.0 / &wD_new) * wl.slice(s![.., .., i])));
                nb += 1;
            }
        }
        for i in 0..k {
            if nb == dk {
                //got enough new expansion vectors
                break;
            }
            let wD: Array2<f64> = w[i] - &omega.to_owned();
            // quite the ugly method in order to reproduce
            // indx = abs(wD) < 1.0e-6
            // wD[indx] = 1.0e-6 * omega[indx]
            // from numpy
            let temp: Array2<f64> = wD.map(|wD| if wD < &1.0e-6 { 1.0e-6 } else { 0.0 });
            let temp_2: Array2<f64> = wD.map(|&wD| if wD < 1.0e-6 { 0.0 } else { wD });
            let mut wD_new: Array2<f64> = &temp * &omega.to_owned();
            wD_new = wD_new + temp_2;
            if norms_r[i] > eps {
                Qs.slice_mut(s![.., .., nb])
                    .assign(&((1.0 / &wD_new) * wr.slice(s![.., .., i])));
                nb += 1;
            }
        }
        l_prev = l;
        // new expansion vectors are bs + Qs
        let mut bs_new: Array3<f64> = Array::zeros((n_occ, n_virt, l + dk));
        bs_new.slice_mut(s![.., .., ..l]).assign(&bs);
        bs_new.slice_mut(s![.., .., l..]).assign(&Qs);

        //QR decomposition
        let nvec: usize = l + dk;
        let bs_flat: Array2<f64> = bs_new.into_shape((n_occ * n_virt, nvec)).unwrap();
        let (Q, R): (Array2<f64>, Array2<f64>) = bs_flat.qr().unwrap();
        bs = Q.into_shape((n_occ, n_virt, nvec)).unwrap();
        l = bs.dim().2;
    }
    let mut Omega: Vec<f64> = w.to_vec();
    Omega.sort_by(|&i, &j| i.partial_cmp(&j).unwrap());
    let Omega: Array1<f64> = Array::from(Omega).slice(s![..k]).to_owned();
    let mut XpY: Array3<f64> = r_canon.slice(s![.., .., ..k]).to_owned();
    let mut XmY: Array3<f64> = l_canon.slice(s![.., .., ..k]).to_owned();

    let t_matrix: Array3<f64> = tensordot(&bs, &Tb, &[Axis(2)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let mut c_matrix: Array3<f64> = t_matrix.slice(s![.., .., ..k]).to_owned();

    XpY.swap_axes(1, 2);
    XpY.swap_axes(0, 1);
    XmY.swap_axes(1, 2);
    XmY.swap_axes(0, 1);
    c_matrix.swap_axes(1, 2);
    c_matrix.swap_axes(0, 1);

    return (Omega, c_matrix, XmY, XpY);
}