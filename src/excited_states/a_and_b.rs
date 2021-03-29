use ndarray::{ArrayView2, ArrayView3, ArrayView1, Array2, Array4, Axis, Array3, Array, Array1};
use ndarray_einsum_beta::tensordot;
use rayon::prelude::*;

pub fn build_a_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array2<f64>) {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;

    // Calculate k_a_lr
    // K_A^{lr} = K_{ov,o'v'}°{lr} = - sum_{A,B} q_A^{oo'} \gamma_{AB}^{lr} q_B^{vv'}
    //                             = - sum_{A,B} q_A^{ik} \gamma_{AB}^{lr} q_B^{jl}
    // o = i, o' = k, v = j, v' = l
    // equivalent to einsum("aik,bjl,ab->ijkl", &[&q_trans_oo, &q_trans_vv, &gamma_lr])
    let mut k_a: Array4<f64> = -1.0
        * tensordot(
        &q_trans_oo,
        &tensordot(&gamma_lr, &q_trans_vv, &[Axis(1)], &[Axis(0)]), // ab,bjl->ajl
        &[Axis(0)],
        &[Axis(0)],
    ) // aik,ajl->ikjl
        .into_dimensionality::<Ix4>()
        .unwrap();
    k_a.swap_axes(1, 2); // ikjl->ijkl
    k_a = k_a.as_standard_layout().to_owned();

    if multiplicity == 1 {
        // calculate coulomb integrals for singlets
        // 2 * K_A =  2 K_{ov,o'v'} = 2 sum_{A,B} q_A^{ov} \gamma_{AB} q_B^{o'v'}
        //                          = 2 sum_{A,B} q_A^{ij} \gamma_{AB} q_B^{kl}
        // equivalent to einsum("aij,bkl,ab->ijkl", &[&q_trans_ov, &q_trans_ov, &gamma])
        let k_singlet: Array4<f64> = 2.0
            * tensordot(
            &q_trans_ov,
            &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkl->akl
            &[Axis(0)],
            &[Axis(0)],
        ) // aij,akl->ijkl
            .into_dimensionality::<Ix4>()
            .unwrap();
        k_a = k_a + k_singlet;
    } else if multiplicity == 3 {
        // calculate magnetic corrections for triplets
        // K_A^{magn} = 2 sum_{A,B} q_A^{ov} \delta_{AB} W_A q_B^{o'v'}
        //            = 2 sum_{A,B} q_A^{ij} \delta_{AB} W_A q_B^{kl}
        // equivalent to einsum("aij,bkl,ab,a->ijkl", &[&q_trans_ov, &q_trans_ov, &delta_ab, &spin_couplings])
        let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings); // ab,a->ab
        let k_triplet: Array4<f64> = 2.0
            * tensordot(
            &q_trans_ov,
            &tensordot(&spin_couplings_diag, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkl->akl
            &[Axis(0)],
            &[Axis(0)],
        ) // aij,akl->ijkl
            .into_dimensionality::<Ix4>()
            .unwrap();

        k_a = k_a + k_triplet;
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
    }

    let k_coupling: Array2<f64> = k_a.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();

    // println!("{:?}", k_coupling);

    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.mapv(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(&omega) + &df_half.dot(&k_coupling.dot(&df_half));
}


pub fn build_b_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array2<f64>) {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;

    // Calculate k_b_lr
    // K_B^{lr} = K_{ov,v'o'}°{lr} = - sum_{A,B} q_A^{ov'} \gamma_{AB}^{lr} q_B^{o'v}
    //            K_{ijlk}^{lr}    = - sum_{A,B} q_A^{il} \gamma_{AB}^{lr} q_B^{kj}
    // o = i, o' = k, v = j, v' = l
    // equivalent to einsum("ail,bkj,ab->ijkl", &[&q_trans_ov, &q_trans_ov, &gamma_lr])
    let mut k_b: Array4<f64> = -1.0
        * tensordot(
        &q_trans_ov,
        &tensordot(&gamma_lr, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkj->akj
        &[Axis(0)],
        &[Axis(0)],
    ) // ail,akj->ilkj
        .into_dimensionality::<Ix4>()
        .unwrap();
    k_b.swap_axes(1, 3); // ilkj->ijkl, ijkl is the correct order for all matrices
    k_b = k_b.as_standard_layout().to_owned();

    if multiplicity == 1 {
        // calculate coulomb integrals for singlets
        // 2 * K_B =  2 K_{ov,v'o'} = 2 sum_{A,B} q_A^{ov} \gamma_{AB} q_B^{o'v'} = 2 * K_A
        //                          = 2 sum_{A,B} q_A^{ij} \gamma_{AB} q_B^{kl}
        // equivalent to einsum("aij,bkl,ab->ijkl", &[&q_trans_ov, &q_trans_ov, &gamma])
        let k_singlet: Array4<f64> = 2.0
            * tensordot(
            &q_trans_ov,
            &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkl->akl
            &[Axis(0)],
            &[Axis(0)],
        ) // aij,akl->ijkl
            .into_dimensionality::<Ix4>()
            .unwrap();

        k_b = k_b + k_singlet;
    } else if multiplicity == 3 {
        // calculate magnetic corrections for triplets
        // K_B^{magn} = 2 sum_{A,B} q_A^{ov} \delta_{AB} W_A q_B^{o'v'} = K_A^{magn}
        //            = 2 sum_{A,B} q_A^{ij} \delta_{AB} W_A q_B^{kl}
        // equivalent to einsum("aij,bkl,ab,a->ijkl", &[&q_trans_ov, &q_trans_ov, &delta_ab, &spin_couplings])
        let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings); // ab,a->ab
        let k_triplet: Array4<f64> = 2.0
            * tensordot(
            &q_trans_ov,
            &tensordot(&spin_couplings_diag, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkl->akl
            &[Axis(0)],
            &[Axis(0)],
        ) // aij,akl->ijkl
            .into_dimensionality::<Ix4>()
            .unwrap();

        k_b = k_b + k_triplet;
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
    }

    let mut k_coupling: Array2<f64> = k_b.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();

    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(&k_coupling.dot(&df_half));
}


pub fn get_apbv(
    gamma: &ArrayView2<f64>,
    gamma_lr: &Option<ArrayView2<f64>>,
    qtrans_oo: &Option<ArrayView3<f64>>,
    qtrans_vv: &Option<ArrayView3<f64>>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    lc: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let lmax: usize = vs.dim().2;
    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());

    for i in 0..lmax {
        let v: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // # matrix product u_ia = sum_jb (A+B)_(ia,jb) v_jb
        // # 1st term in (A+B).v: KS orbital energy differences
        let mut u: Array2<f64> = omega * &v;

        // 2nd term Coulomb
        let tmp: Array1<f64> = tensordot(&qtrans_ov, &v, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix1>()
            .unwrap();

        if multiplicity == 1 {
            let tmp_2: Array1<f64> = gamma.dot(&tmp);
            let u_singlet: Array2<f64> = 4.0
                * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                .into_dimensionality::<Ix2>()
                .unwrap();
            u = u + u_singlet;
            //println!("Iteration {} for the tensordot routine",i);
            //println!("Value of u after 2nd term {}",u);
        } else if multiplicity == 3 {
            let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings);
            let tmp_2: Array1<f64> = spin_couplings_diag.dot(&tmp);
            let u_triplet: Array2<f64> = 4.0 * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                .into_dimensionality::<Ix2>()
                .unwrap();
            u = u + u_triplet;
        } else {
            panic!("Currently only singlets and triplets are supported, you wished a multiplicity of {}!", multiplicity);
        }

        if lc == 1 {
            // 3rd term - Exchange
            let tmp: Array3<f64> = tensordot(&qtrans_vv.unwrap(), &v, &[Axis(2)], &[Axis(1)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            let tmp_2: Array3<f64> = tensordot(&gamma_lr.unwrap(), &tmp, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            u = u - tensordot(
                &qtrans_oo.unwrap(),
                &tmp_2,
                &[Axis(0), Axis(2)],
                &[Axis(0), Axis(2)],
            )
                .into_dimensionality::<Ix2>()
                .unwrap();

            //4th term - Exchange
            let tmp: Array3<f64> = tensordot(&qtrans_ov, &v, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            let tmp_2: Array3<f64> = tensordot(&gamma_lr.unwrap(), &tmp, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            u = u - tensordot(&qtrans_ov, &tmp_2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
                .into_dimensionality::<Ix2>()
                .unwrap();
        }

        us.slice_mut(s![.., .., i]).assign(&u);
    }
    return us;
}

pub fn get_apbv_fortran(
    gamma: &ArrayView2<f64>,
    gamma_lr: &ArrayView2<f64>,
    qtrans_oo: &ArrayView3<f64>,
    qtrans_vv: &ArrayView3<f64>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    n_at: usize,
    n_occ: usize,
    n_virt: usize,
    n_vec: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let tmp_q_vv: Array2<f64> = qtrans_vv
        .to_owned()
        .into_shape((n_virt * n_at, n_virt))
        .unwrap();
    let tmp_q_oo: Array2<f64> = qtrans_oo
        .to_owned()
        .into_shape((n_at * n_occ, n_occ))
        .unwrap();
    let mut tmp_q_ov_swapped: Array3<f64> = qtrans_ov.to_owned();
    tmp_q_ov_swapped.swap_axes(1, 2);
    tmp_q_ov_swapped = tmp_q_ov_swapped.as_standard_layout().to_owned();
    let tmp_q_ov_shape_1: Array2<f64> =
        tmp_q_ov_swapped.into_shape((n_at * n_virt, n_occ)).unwrap();
    let mut tmp_q_ov_swapped_2: Array3<f64> = qtrans_ov.to_owned();
    tmp_q_ov_swapped_2.swap_axes(0, 1);
    tmp_q_ov_swapped_2 = tmp_q_ov_swapped_2.as_standard_layout().to_owned();
    let tmp_q_ov_shape_2: Array2<f64> = tmp_q_ov_swapped_2
        .into_shape((n_occ, n_at * n_virt))
        .unwrap();
    //let tmp_q_oo: Array2<f64> = qtrans_oo
    //    .to_owned()
    //    .into_shape((n_at * n_occ, n_occ))
    //    .unwrap();
    let tmp_q_ov_shape_1_new: Array2<f64> = qtrans_ov
        .to_owned()
        .into_shape((n_occ, n_at * n_virt))
        .unwrap()
        .reversed_axes();
    let tmp_q_ov_shape_2_new: Array2<f64> = qtrans_ov
        .to_owned()
        .into_shape((n_at * n_virt, n_occ))
        .unwrap()
        .reversed_axes();

    println!("qtrans ov{}", qtrans_ov.clone());
    println!("Compare shapes");
    println!("Old q_ov {:?}", tmp_q_ov_shape_1);
    println!("New q_ov {:?}", tmp_q_ov_shape_1_new);

    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());

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
        let mut u_l: Array2<f64> = omega * &vl;

        // 2nd term - Coulomb
        let mut tmp21: Array1<f64> = Array1::zeros(n_at);

        //for at in (0..n_at) {
        //    let tmp:Array2<f64> = qtrans_ov.clone().slice(s![at, .., ..]).to_owned() * vl.clone();
        //    tmp21[at] = tmp.sum();
        //}
        let tmp21: Vec<f64> = (0..n_at)
            .into_par_iter()
            .map(|at| {
                let tmp: Array2<f64> = &qtrans_ov.slice(s![at, .., ..]).to_owned() * &vl;
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
            .map(|at| qtrans_ov.slice(s![at, .., ..]).to_owned() * tmp22[at])
            .collect();
        for i in tmp.iter() {
            u_l = u_l + i;
        }
        //u_l = u_l + tmp;

        // 3rd term - Exchange
        let tmp31: Array3<f64> = tmp_q_vv
            .dot(&vl.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();

        let tmp31_reshaped: Array2<f64> = tmp31.into_shape((n_at, n_virt * n_occ)).unwrap();
        let mut tmp32: Array3<f64> = gamma_lr
            .dot(&tmp31_reshaped)
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        tmp32.swap_axes(1, 2);
        tmp32 = tmp32.as_standard_layout().to_owned();

        let tmp33: Array2<f64> = tmp_q_oo
            .t()
            .dot(&tmp32.into_shape((n_at * n_occ, n_virt)).unwrap());
        u_l = u_l - tmp33;

        // 4th term - Exchange
        let tmp41: Array3<f64> = tmp_q_ov_shape_1
            .dot(&vl)
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp41_reshaped: Array2<f64> = tmp41.into_shape((n_at, n_virt * n_virt)).unwrap();
        let mut tmp42: Array3<f64> = gamma_lr
            .dot(&tmp41_reshaped)
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        tmp42.swap_axes(1, 2);
        tmp42 = tmp42.as_standard_layout().to_owned();

        let tmp43: Array2<f64> =
            tmp_q_ov_shape_2.dot(&tmp42.into_shape((n_at * n_virt, n_virt)).unwrap());
        u_l = u_l - tmp43;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}

pub fn get_apbv_fortran_no_lc(
    gamma: &ArrayView2<f64>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    n_at: usize,
    n_occ: usize,
    n_virt: usize,
    n_vec: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());

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
        let mut u_l: Array2<f64> = omega * &vl;

        // 2nd term - Coulomb
        let mut tmp21: Array1<f64> = Array1::zeros(n_at);

        //for at in (0..n_at) {
        //    let tmp:Array2<f64> = qtrans_ov.clone().slice(s![at, .., ..]).to_owned() * vl.clone();
        //    tmp21[at] = tmp.sum();
        //}
        let tmp21: Vec<f64> = (0..n_at)
            .into_par_iter()
            .map(|at| {
                let tmp: Array2<f64> = &qtrans_ov.slice(s![at, .., ..]) * &vl;
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
            .map(|at| qtrans_ov.slice(s![at, .., ..]).to_owned() * tmp22[at])
            .collect();
        for i in tmp.iter() {
            u_l = u_l + i;
        }
        //u_l = u_l + tmp;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}

pub fn get_ambv_fortran(
    gamma: &ArrayView2<f64>,
    gamma_lr: &ArrayView2<f64>,
    qtrans_oo: &ArrayView3<f64>,
    qtrans_vv: &ArrayView3<f64>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    n_at: usize,
    n_occ: usize,
    n_virt: usize,
    n_vec: usize,
) -> (Array3<f64>) {
    let tmp_q_vv: Array2<f64> = qtrans_vv
        .to_owned()
        .into_shape((n_virt * n_at, n_virt))
        .unwrap();
    let mut tmp_q_oo_swapped: Array3<f64> = qtrans_oo.to_owned();
    tmp_q_oo_swapped.swap_axes(0, 1);
    tmp_q_oo_swapped = tmp_q_oo_swapped.as_standard_layout().to_owned();
    let tmp_q_oo: Array2<f64> = tmp_q_oo_swapped.into_shape((n_occ, n_at * n_occ)).unwrap();
    let mut tmp_q_ov_swapped: Array3<f64> = qtrans_ov.to_owned();
    tmp_q_ov_swapped.swap_axes(1, 2);
    tmp_q_ov_swapped = tmp_q_ov_swapped.as_standard_layout().to_owned();
    let tmp_q_ov_shape_1: Array2<f64> =
        tmp_q_ov_swapped.into_shape((n_at * n_virt, n_occ)).unwrap();
    let mut tmp_q_ov_swapped_2: Array3<f64> = qtrans_ov.to_owned();
    //println!("Before swap {}",tmp_q_ov_swapped_2.clone().into_shape((n_occ,n_at,n_virt)).unwrap());
    tmp_q_ov_swapped_2.swap_axes(0, 1);
    //let mut tmp_q_ov_swapped_3:Array3<f64> = tmp_q_ov_swapped_2.to_owned();
    //let tmp_q_ov_shape_3: Array3<f64> = ArrayView::from_shape((n_occ,n_at,n_virt),&tmp_q_ov_swapped_3.clone().as_slice_memory_order().unwrap()).unwrap().to_owned();//.into_shape((n_occ, n_at * n_virt)).unwrap().to_owned();
    //tmp_q_ov_swapped_2 = tmp_q_ov_swapped_3.as_standard_layout().to_owned();
    tmp_q_ov_swapped_2 = tmp_q_ov_swapped_2.as_standard_layout().to_owned();
    let tmp_q_ov_shape_2: Array2<f64> = tmp_q_ov_swapped_2
        .clone()
        .into_shape((n_occ, n_at * n_virt))
        .unwrap();

    // println!("shape after swap {}",tmp_q_ov_swapped_3);
    // println!("shape after standard layout {}",tmp_q_ov_swapped_2);
    // println!("shape Array::from_shape() {}",tmp_q_ov_shape_3);

    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());

    for i in (0..n_vec) {
        let vl: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // 1st term - KS orbital energy differences
        let mut u_l: Array2<f64> = omega * &vl;
        // 2nd term - Coulomb
        let tmp21: Array3<f64> = tmp_q_ov_shape_1
            .dot(&vl)
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();

        let mut tmp22: Array3<f64> = gamma_lr
            .dot(&tmp21.into_shape((n_at, n_virt * n_virt)).unwrap())
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        tmp22.swap_axes(1, 2);
        tmp22 = tmp22.as_standard_layout().to_owned();

        let tmp23: Array2<f64> =
            tmp_q_ov_shape_2.dot(&tmp22.into_shape((n_at * n_virt, n_virt)).unwrap());
        u_l = u_l + tmp23;

        // 3rd term - Exchange
        let tmp31: Array3<f64> = tmp_q_vv
            .dot(&vl.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let mut tmp32: Array3<f64> = gamma_lr
            .dot(&tmp31.into_shape((n_at, n_virt * n_occ)).unwrap())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        tmp32.swap_axes(1, 2);
        tmp32 = tmp32.as_standard_layout().to_owned();

        let tmp33: Array2<f64> = tmp_q_oo.dot(&tmp32.into_shape((n_at * n_occ, n_virt)).unwrap());

        u_l = u_l - tmp33;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}

// pub fn get_apbv_single_vector(
//     gamma: &ArrayView2<f64>,
//     gamma_lr: &Option<ArrayView2<f64>>,
//     qtrans_oo: &Option<ArrayView3<f64>>,
//     qtrans_vv: &Option<ArrayView3<f64>>,
//     qtrans_ov: &ArrayView3<f64>,
//     omega: &ArrayView2<f64>,
//     vs: &Array2<f64>,
//     lc: usize,
//     multiplicity: u8,
//     spin_couplings: ArrayView1<f64>,
// ) -> (Array2<f64>) {
//
//     let v: Array2<f64> = vs.clone();
//     // # matrix product u_ia = sum_jb (A+B)_(ia,jb) v_jb
//     // # 1st term in (A+B).v: KS orbital energy differences
//     let mut u: Array2<f64> = omega * &v;
//
//     // 2nd term Coulomb
//     let tmp: Array1<f64> = tensordot(&qtrans_ov, &v, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//
//     if multiplicity == 1 {
//         let tmp_2: Array1<f64> = gamma.dot(&tmp);
//         let u_singlet: Array2<f64> = 4.0
//             * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
//             .into_dimensionality::<Ix2>()
//             .unwrap();
//         u = u + u_singlet;
//     } else if multiplicity == 3 {
//         let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings);
//         let tmp_2: Array1<f64> = spin_couplings_diag.dot(&tmp);
//         let u_triplet: Array2<f64> = 4.0
//             * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
//             .into_dimensionality::<Ix2>()
//             .unwrap();
//         u = u + u_triplet;
//     } else {
//         panic!("Currently only singlets and triplets are supported, you wished a multiplicity of {}!", multiplicity);
//     }
//
//     if lc == 1 {
//         // 3rd term - Exchange
//         let tmp: Array3<f64> = tensordot(&qtrans_vv.unwrap(), &v, &[Axis(2)], &[Axis(1)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         let tmp_2: Array3<f64> = tensordot(&gamma_lr.unwrap(), &tmp, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         u = u - tensordot(
//             &qtrans_oo.unwrap(),
//             &tmp_2,
//             &[Axis(0), Axis(2)],
//             &[Axis(0), Axis(2)],
//         )
//             .into_dimensionality::<Ix2>()
//             .unwrap();
//
//         //4th term - Exchange
//         let tmp: Array3<f64> = tensordot(&qtrans_ov, &v, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         let tmp_2: Array3<f64> = tensordot(&gamma_lr.unwrap(), &tmp, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         u = u - tensordot(&qtrans_ov, &tmp_2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
//             .into_dimensionality::<Ix2>()
//             .unwrap();
//     }
//
//     return u;
// }

pub fn get_ambv(
    gamma: &ArrayView2<f64>,
    gamma_lr: &ArrayView2<f64>,
    qtrans_oo: &ArrayView3<f64>,
    qtrans_vv: &ArrayView3<f64>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    lc: usize,
) -> (Array3<f64>) {
    let lmax: usize = vs.dim().2;
    let mut us: Array3<f64> = Array::zeros((vs.shape())).into_dimensionality().unwrap();

    for i in 0..lmax {
        let v: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // # matrix product u_ia = sum_jb (A-B)_(ia,jb) v_jb
        // # 1st term, differences in orbital energies
        let mut u: Array2<f64> = omega * &v;

        if lc == 1 {
            // 2nd term - Exchange
            let tmp: Array3<f64> = tensordot(&qtrans_ov, &v, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            let tmp_2: Array3<f64> = tensordot(&gamma_lr, &tmp, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            u = u + tensordot(&qtrans_ov, &tmp_2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
                .into_dimensionality::<Ix2>()
                .unwrap();

            //3rd term - Exchange
            let tmp: Array3<f64> = tensordot(&qtrans_vv, &v, &[Axis(2)], &[Axis(1)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            let tmp_2: Array3<f64> = tensordot(&gamma_lr, &tmp, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            u = u - tensordot(&qtrans_oo, &tmp_2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
                .into_dimensionality::<Ix2>()
                .unwrap();
        }

        us.slice_mut(s![.., .., i]).assign(&u);
    }
    return us;
}