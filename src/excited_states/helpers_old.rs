use ndarray::prelude::*;
use ndarray_einsum_beta::tensordot;
use rayon::prelude::*;





pub fn reorder_vectors_lambda2(
    Oia: &ArrayView2<f64>,
    w2: &Array1<f64>,
    T: &Array3<f64>,
    l2_treshold: f64,
) -> (Array1<f64>, Array3<f64>) {
    // reorder the expansion vectors so that those with Lambda2 values
    // above a certain threshold come first

    let n_occ: usize = T.dim().0;
    let n_virt: usize = T.dim().1;
    let n_st: usize = T.dim().2;
    let mut l2: Array1<f64> = Array::zeros(n_st);

    for i in 0..n_st {
        let T_temp: Array2<f64> = T
            .slice(s![.., .., i])
            .to_owned()
            .map(|T| ndarray_linalg::Scalar::powi(T, 2));
        l2[i] = tensordot(&T_temp, &Oia, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix0>()
            .unwrap()
            .into_scalar();
    }
    //get indeces
    let over_l2: Array1<_> = l2
        .indexed_iter()
        .filter_map(|(index, &item)| {
            if item > l2_treshold {
                Some(index)
            } else {
                None
            }
        })
        .collect();
    let under_l2: Array1<_> = l2
        .indexed_iter()
        .filter_map(|(index, &item)| {
            if item < l2_treshold {
                Some(index)
            } else {
                None
            }
        })
        .collect();

    let mut T_new: Array3<f64> = Array::zeros((n_occ, n_virt, n_st));
    let mut w2_new: Array1<f64> = Array::zeros(n_st);

    //construct new matrices
    for i in 0..over_l2.len() {
        T_new
            .slice_mut(s![.., .., i])
            .assign(&T.slice(s![.., .., over_l2[i]]));
        w2_new[i] = w2[over_l2[i]];
    }
    let len_over_l2: usize = over_l2.len();
    for i in 0..under_l2.len() {
        T_new
            .slice_mut(s![.., .., i + len_over_l2])
            .assign(&T.slice(s![.., .., under_l2[i]]));
        w2_new[i + len_over_l2] = w2[under_l2[i]];
    }

    return (w2_new, T_new);
}

pub fn norm_special(array: &Array2<f64>) -> (f64) {
    let v: f64 = tensordot(&array, &array, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
        .into_dimensionality::<Ix0>()
        .unwrap()
        .into_scalar();
    return v.sqrt();
}

pub fn matrix_v_product(
    vs: &Array3<f64>,
    lmax: usize,
    n_occ: usize,
    n_virt: usize,
    om: &Array2<f64>,
    wq_ov: &Array3<f64>,
    gamma: &ArrayView2<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let mut us: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    for i in 0..lmax {
        let v: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // # matrix product u = sum_jb (A-B)^(1/2).(A+B).(A-B)^(1/2).v
        // # 1st term in (A+B).v  - KS orbital energy differences
        let mut u: Array2<f64> = Array::zeros((n_occ, n_virt));
        u = om * &v;

        let tmp: Array1<f64> = tensordot(&wq_ov, &v, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix1>()
            .unwrap();

        if multiplicity == 1 {
            let tmp_2: Array1<f64> = gamma.dot(&tmp);
            let u_singlet: Array2<f64> = 4.0
                * tensordot(&wq_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                .into_dimensionality::<Ix2>()
                .unwrap();
            u = u + u_singlet;
        } else if multiplicity == 3 {
            let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings);
            let tmp_2: Array1<f64> = spin_couplings_diag.dot(&tmp);
            let u_triplet: Array2<f64> = 4.0
                * tensordot(&wq_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                .into_dimensionality::<Ix2>()
                .unwrap();
            u = u + u_triplet;
        } else {
            panic!("Currently only singlets and triplets are supported, you wished a multiplicity of {}!", multiplicity);
        }

        // let tmp2: Array1<f64> = gamma.dot(&tmp);
        // u = u + 4.0
        //     * tensordot(&wq_ov, &tmp2, &[Axis(0)], &[Axis(0)])
        //         .into_dimensionality::<Ix2>()
        //         .unwrap();

        us.slice_mut(s![.., .., i]).assign(&u);
    }
    return us;
}

pub fn matrix_v_product_fortran(
    vs: &Array3<f64>,
    n_vec: usize,
    n_occ: usize,
    n_virt: usize,
    om: &Array2<f64>,
    wq_ov: &Array3<f64>,
    gamma: &ArrayView2<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());
    let n_at: usize = wq_ov.dim().0;

    let gamma_equiv: Array2<f64> = if multiplicity == 1 {
        gamma.to_owned()
    } else if multiplicity == 3 {
        Array2::from_diag(&spin_couplings)
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
        Array::zeros(gamma.raw_dim())
    };

    for i in (0..n_vec) {
        let vl: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // 1st term - KS orbital energy differences
        let mut u_l: Array2<f64> = om * &vl;

        // 2nd term - Coulomb
        let mut tmp21: Array1<f64> = Array1::zeros(n_at);

        //for at in (0..n_at) {
        //    let tmp:Array2<f64> = qtrans_ov.clone().slice(s![at, .., ..]).to_owned() * vl.clone();
        //    tmp21[at] = tmp.sum();
        //}
        let tmp21: Vec<f64> = (0..n_at)
            .into_par_iter()
            .map(|at| {
                let tmp: Array2<f64> = &wq_ov.slice(s![at, .., ..]) * &vl;
                tmp.sum()
            })
            .collect();
        let tmp21: Array1<f64> = Array::from(tmp21);

        let tmp22: Array1<f64> = 4.0 * gamma_equiv.dot(&tmp21);

        // for at in (0..n_at).into_iter() {
        //     u_l = u_l + qtrans_ov.slice(s![at, .., ..]).to_owned() * tmp22[at];
        // }
        let mut tmp: Vec<Array2<f64>> = (0..n_at)
            .into_par_iter()
            .map(|at| wq_ov.slice(s![at, .., ..]).to_owned() * tmp22[at])
            .collect();
        for i in tmp.iter() {
            u_l = u_l + i;
        }
        //u_l = u_l + tmp;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}