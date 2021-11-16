use ndarray::prelude::*;

pub fn f_le_ct_coulomb(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_pair_ao: ArrayView2<f64>,
    g1_pair_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j: usize,
    bool_ij: bool,
) -> Array3<f64> {
    // The pair indices are IJ -> I < J
    let (s_i, s_ij, g_i, g_ij) = if bool_ij {
        let s_i: ArrayView2<f64> = s.slice(s![..n_orb_i, ..n_orb_i]);
        let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
        let g_i: ArrayView2<f64> = g0_pair_ao.slice(s![..n_orb_i, ..n_orb_i]);
        let g_ij: ArrayView2<f64> = g0_pair_ao.slice(s![..n_orb_i, n_orb_i..]);

        (s_i, s_ij, g_i, g_ij)
    } else {
        // The pair indices are JI -> J < I
        let s_i: ArrayView2<f64> = s.slice(s![n_orb_i.., n_orb_i..]);
        let s_ij: ArrayView2<f64> = s.slice(s![n_orb_i.., ..n_orb_i]);
        let g_i: ArrayView2<f64> = g0_pair_ao.slice(s![n_orb_i.., n_orb_i..]);
        let g_ij: ArrayView2<f64> = g0_pair_ao.slice(s![n_orb_i.., ..n_orb_i]);

        (s_i, s_ij, g_i, g_ij)
    };

    let v:Array2<f64> = &v +&v.t();
    let si_v: Array1<f64> = (&s_i * &v).sum_axis(Axis(1));
    let gi_sv: Array1<f64> = g_i.dot(&si_v);
    let gij_sv: Array1<f64> = g_ij.t().dot(&si_v);

    let mut f_return: Array3<f64>;
    if bool_ij{
        f_return = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_j));;
    }else{
        f_return = Array3::zeros((3 * n_atoms, n_orb_j, n_orb_i));
    }

    for nc in 0..3 * n_atoms {
        let (ds_i, ds_ij, dg_i, dg_ij) = if bool_ij {
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_pair_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_pair_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i, ds_ij, dg_i, dg_ij)
        } else {
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., n_orb_i..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., ..n_orb_i]);
            let dg_i: ArrayView2<f64> = g1_pair_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
            let dg_ij: ArrayView2<f64> = g1_pair_ao.slice(s![nc, n_orb_i.., ..n_orb_i]);

            (ds_i, ds_ij, dg_i, dg_ij)
        };

        let gi_dsv: Array1<f64> = g_i.dot(&(&ds_i * &v).sum_axis(Axis(1)));
        let gij_dsv: Array1<f64> = g_ij.t().dot(&(&ds_i * &v).sum_axis(Axis(1)));
        let dgi_sv: Array1<f64> = dg_i.dot(&si_v);
        let dgij_sv: Array1<f64> = dg_ij.t().dot(&si_v);

        let mut d_f: Array2<f64>;
        if bool_ij{
            d_f = Array2::zeros((n_orb_i, n_orb_j));;
        }else{
            d_f = Array2::zeros((n_orb_j, n_orb_i));
        }

        for b in 0..n_orb_i {
            for a in 0..n_orb_j {
                d_f[[b, a]] += ds_ij[[b, a]] * (gi_sv[b] + gij_sv[a])
                    + s_ij[[b, a]] * (gi_dsv[b] + gij_dsv[a] + dgi_sv[b] + dgij_sv[a]);
            }
        }
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }

    return f_return;
}

pub fn f_lr_le_ct_exchange_hole_i(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j: usize,
    bool_ij: bool,
) -> Array3<f64> {
    let (s_i, s_ij, g_i, g_ij) = if bool_ij {
        let s_i: ArrayView2<f64> = s.slice(s![..n_orb_i, ..n_orb_i]);
        let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
        let g_i: ArrayView2<f64> = g0_lr_a0.slice(s![..n_orb_i, ..n_orb_i]);
        let g_ij: ArrayView2<f64> = g0_lr_a0.slice(s![..n_orb_i, n_orb_i..]);

        (s_i, s_ij, g_i, g_ij)
    } else {
        let s_i: ArrayView2<f64> = s.slice(s![n_orb_i.., n_orb_i..]);
        let s_ij: ArrayView2<f64> = s.slice(s![n_orb_i.., ..n_orb_i]);
        let g_i: ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_i.., n_orb_i..]);
        let g_ij: ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_i.., ..n_orb_i]);

        (s_i, s_ij, g_i, g_ij)
    };

    // for term 1
    let gi_v: Array2<f64> = &g_i * &v;
    // for term 1
    let gi_v_sij: Array2<f64> = gi_v.dot(&s_ij);
    // for term 2
    let v_si: Array2<f64> = v.dot(&s_i);
    // for term 4, 10
    let v_sij: Array2<f64> = v.dot(&s_ij);
    // for term 5
    let gi_v_si: Array2<f64> = s_i.t().dot(&gi_v);
    // for term 7, 11, 12
    let vt_si: Array2<f64> = v.t().dot(&s_i);
    // for term 7
    let gi_vt_si: Array2<f64> = &g_i * &vt_si;
    // for term 8
    let si_v: Array2<f64> = s_i.t().dot(&v);
    // for term 12
    let vt_si_t_sij: Array2<f64> = vt_si.t().dot(&s_ij);

    let mut f_return: Array3<f64>;
    if bool_ij{
        f_return = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_j));
    }else{
        f_return = Array3::zeros((3 * n_atoms, n_orb_j, n_orb_i));
    }

    for nc in 0..3 * n_atoms {
        let (ds_i, ds_ij, dg_i, dg_ij) = if bool_ij {
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i, ds_ij, dg_i, dg_ij)
        } else {
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., n_orb_i..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., ..n_orb_i]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_i.., ..n_orb_i]);

            (ds_i, ds_ij, dg_i, dg_ij)
        };

        let mut d_f: Array2<f64>;
        if bool_ij{
            d_f = Array2::zeros((n_orb_i, n_orb_j));
        }else{
            d_f = Array2::zeros((n_orb_j, n_orb_i));
        }
        // 1st term
        d_f = d_f + ds_i.t().dot(&gi_v_sij);
        // 2nd term
        d_f = d_f + (&v_si * &ds_i).t().dot(&g_ij);
        // 3rd term
        d_f = d_f + (&ds_i.t().dot(&v) * &g_i.t()).dot(&s_ij);
        // 4th term
        d_f = d_f + &ds_i.t().dot(&v_sij) * &g_ij;
        // 5th term
        d_f = d_f + gi_v_si.dot(&ds_ij);
        // 6th term
        d_f = d_f + s_i.t().dot(&(&v.dot(&ds_ij) * &g_ij));
        // 7th term
        d_f = d_f + gi_vt_si.t().dot(&ds_ij);
        // 8th term
        d_f = d_f + &si_v.dot(&ds_ij) * &g_ij;
        // 9th term
        d_f = d_f + s_i.t().dot(&(&dg_i * &v).dot(&s_ij));
        // 10th term
        d_f = d_f + s_i.t().dot(&(&v_sij * &dg_ij));
        // 11th term
        d_f = d_f + (&vt_si * &dg_i).t().dot(&s_ij);
        // 12th term
        d_f = d_f + &vt_si_t_sij * &dg_ij;
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

pub fn f_lr_le_ct_exchange_hole_j(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j: usize,
    bool_ij: bool,
) -> Array3<f64> {
    let (s_i, s_ij, g_i, g_ij) = if bool_ij {
        let s_i: ArrayView2<f64> = s.slice(s![..n_orb_i, ..n_orb_i]);
        let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
        let g_i: ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j.., n_orb_j..]);
        let g_ij: ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j.., ..n_orb_j]);

        (s_i, s_ij, g_i, g_ij)
    } else {
        let s_i: ArrayView2<f64> = s.slice(s![n_orb_j.., n_orb_j..]);
        let s_ij: ArrayView2<f64> = s.slice(s![n_orb_j.., ..n_orb_j]);
        let g_i: ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j.., n_orb_j..]);
        let g_ij: ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j.., ..n_orb_j]);

        (s_i, s_ij, g_i, g_ij)
    };

    // for term 1
    let gi_v: Array2<f64> = &g_i * &v;
    // for term 1
    let gi_v_si: Array2<f64> = gi_v.dot(&s_i);
    // for term 4,10
    let v_si: Array2<f64> = v.dot(&s_i);
    // for term 2
    let v_sij: Array2<f64> = v.dot(&s_ij);
    // for term 5
    let gi_v_sij: Array2<f64> = gi_v.t().dot(&s_ij);
    // for term 7, 11, 12
    let vt_sij: Array2<f64> = v.t().dot(&s_ij);
    // for term 7
    let gij_vt_sij: Array2<f64> = &g_ij * &vt_sij;
    // for term 8
    let sij_v: Array2<f64> = s_ij.t().dot(&v);
    // for term 12
    let si_t_vt_sij: Array2<f64> = s_i.t().dot(&vt_sij);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let (ds_i, ds_ij, dg_i, dg_ij) = if bool_ij {
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i, ds_ij, dg_i, dg_ij)
        } else {
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j.., n_orb_j..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j.., ..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j.., n_orb_j..]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j.., ..n_orb_j]);

            (ds_i, ds_ij, dg_i, dg_ij)
        };

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_j));
        // 1st term
        d_f = d_f + gi_v_si.t().dot(&ds_ij);
        // 2nd term
        d_f = d_f + g_i.dot(&(&v_sij * &ds_ij));
        // 3rd term
        d_f = d_f + s_i.dot(&(&v.t().dot(&ds_ij) * &g_ij));
        // 4th term
        d_f = d_f + v_si.t().dot(&ds_ij) * &g_ij;
        // 5th term
        d_f = d_f + ds_i.t().dot(&gi_v_sij);
        // 6th term
        d_f = d_f + (&v.dot(&ds_i) * &g_i).t().dot(&s_ij);
        // 7th term
        d_f = d_f + ds_i.t().dot(&gij_vt_sij);
        // 8th term
        d_f = d_f + &ds_i.t().dot(&sij_v.t()) * &g_ij;
        // 9th term
        d_f = d_f + (&dg_i * &v).dot(&s_i).t().dot(&s_ij);
        // 10th term
        d_f = d_f + (&v_si * &dg_i).t().dot(&s_ij);
        // 11th term
        d_f = d_f + s_i.t().dot(&(&vt_sij * &dg_ij));
        // 12th term
        d_f = d_f + &si_t_vt_sij * &dg_ij;
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

pub fn f_le_le_coulomb(
    v: ArrayView2<f64>,
    s_i: ArrayView2<f64>,
    s_j: ArrayView2<f64>,
    grad_s_i: ArrayView3<f64>,
    grad_s_j: ArrayView3<f64>,
    g0_pair_ao: ArrayView2<f64>,
    g1_pair_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
) -> Array3<f64> {
    let vp: Array2<f64> = &v + &(v.t());
    let s_j_v: Array1<f64> = (&s_j * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_pair_ao.slice(s![..n_orb_i, n_orb_i..]).dot(&s_j_v);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let ds_i: ArrayView2<f64> = grad_s_i.slice(s![nc, .., ..]);
        let ds_j: ArrayView2<f64> = grad_s_j.slice(s![nc, .., ..]);
        let dg: ArrayView2<f64> = g1_pair_ao.slice(s![nc, .., ..]);

        let gdsv: Array1<f64> = g0_pair_ao
            .slice(s![..n_orb_i, n_orb_i..])
            .dot(&(&ds_j * &vp).sum_axis(Axis(0)));
        let dgsv: Array1<f64> = dg.slice(s![..n_orb_i, n_orb_i..]).dot(&s_j_v);
        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_i));

        for b in 0..n_orb_i {
            for a in 0..n_orb_i {
                d_f[[a, b]] = ds_i[[a, b]] * (gsv[a] + gsv[b])
                    + s_i[[a, b]] * (dgsv[a] + gdsv[a] + dgsv[b] + gdsv[b]);
            }
        }
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }

    return f_return;
}

pub fn f_lr_le_le_exchange(
    v: ArrayView2<f64>,
    s_ij: ArrayView2<f64>,
    grad_pair_s: ArrayView3<f64>,
    g0_pair_lr_a0: ArrayView2<f64>,
    g1_pair_lr_ao: ArrayView3<f64>,
    n_atoms_i: usize,
    n_atoms_j: usize,
    n_orb_i: usize,
) -> Array3<f64> {
    let g0_lr_ao_i: ArrayView2<f64> = g0_pair_lr_a0.slice(s![..n_orb_i, ..n_orb_i]);
    let g0_lr_ao_j: ArrayView2<f64> = g0_pair_lr_a0.slice(s![n_orb_i.., n_orb_i..]);
    let g0_lr_ao_ij: ArrayView2<f64> = g0_pair_lr_a0.slice(s![..n_orb_i, n_orb_i..]);
    let s_ij_outer: ArrayView2<f64> = s_ij.slice(s![..n_orb_i, n_orb_i..]);
    let n_atoms: usize = n_atoms_i + n_atoms_j;

    let sv: Array2<f64> = s_ij_outer.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s_ij_outer.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_ao_j * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_ao_ij).reversed_axes();
    let sgv_t: Array2<f64> = s_ij_outer.dot(&gv).reversed_axes();

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let d_s: ArrayView2<f64> = grad_pair_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
        let d_g_i: ArrayView2<f64> = g1_pair_lr_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
        let d_g_j: ArrayView2<f64> = g1_pair_lr_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
        let d_g_ij: ArrayView2<f64> = g1_pair_lr_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

        let d_sv_t: Array2<f64> = d_s.dot(&v_t);
        let d_sv: Array2<f64> = d_s.dot(&v);
        let d_gv: Array2<f64> = &d_g_j * &v;

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_i));
        // 1st term
        d_f = d_f + &g0_lr_ao_i * &(d_s.dot(&t_sv));
        // 2nd term
        d_f = d_f + (&d_sv_t * &g0_lr_ao_ij).dot(&s_ij_outer.t());
        // 3rd term
        d_f = d_f + d_s.dot(&svg_t);
        // 4th term
        d_f = d_f + d_s.dot(&sgv_t);
        // 5th term
        d_f = d_f + &g0_lr_ao_i * &(s_ij_outer.dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t * &g0_lr_ao_ij).dot(&d_s.t());
        // 7th term
        d_f = d_f + s_ij_outer.dot(&(&d_sv * &g0_lr_ao_ij).t());
        // 8th term
        d_f = d_f + s_ij_outer.dot(&(d_s.dot(&gv)).t());
        // 9th term
        d_f = d_f + &d_g_i * &(s_ij_outer.dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t * &d_g_ij).dot(&s_ij_outer.t());
        // 11th term
        d_f = d_f + s_ij_outer.dot(&(&sv * &d_g_ij).t());
        // 12th term
        d_f = d_f + s_ij_outer.dot(&(s_ij_outer.dot(&d_gv)).t());
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

pub fn f_coulomb_loop(
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
)->Array5<f64>{
    let mut integral:Array5<f64> = Array5::zeros([3*n_atoms,n_orb_i,n_orb_i,n_orb_j,n_orb_j]);

    let s_i: ArrayView2<f64> = s.slice(s![..n_orb_i, ..n_orb_i]);
    let s_j: ArrayView2<f64> = s.slice(s![n_orb_i.., n_orb_i..]);
    let g_ij: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, n_orb_i..]);

    for nc in 0..3 * n_atoms {
        let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
        let ds_j: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., n_orb_i..]);
        let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

        for mu in 0..n_orb_i{
            for la in 0..n_orb_i{
                for nu in 0..n_orb_j{
                    for sig in 0..n_orb_j{
                        integral[[nc,mu,la,nu,sig]] = 0.25 * ((ds_i[[mu,la]]*s_j[[nu,sig]] + s_i[[mu,la]] * ds_j[[nu,sig]]) *
                            (g_ij[[mu,nu]] + g_ij[[mu,sig]] + g_ij[[la,nu]] + g_ij[[la,sig]]) +
                            s_i[[mu,la]] * s_j[[nu,sig]] * (dg_ij[[mu,nu]] + dg_ij[[mu,sig]] + dg_ij[[la,nu]] + dg_ij[[la,sig]]));
                    }
                }
            }
        }
    }
    return integral;
}

pub fn f_le_ct_coulomb_loop(
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
)->Array5<f64>{
    let mut integral:Array5<f64> = Array5::zeros([3*n_atoms,n_orb_i,n_orb_i,n_orb_i,n_orb_j]);

    let s_i: ArrayView2<f64> = s.slice(s![..n_orb_i, ..n_orb_i]);
    let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
    let g_i: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, ..n_orb_i]);
    let g_ij: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, n_orb_i..]);

    for nc in 0..3 * n_atoms {
        let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
        let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
        let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
        let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

        for mu in 0..n_orb_i{
            for nu in 0..n_orb_i{
                for la in 0..n_orb_i{
                    for sig in 0..n_orb_j{
                        integral[[nc,mu,nu,la,sig]] = 0.25 * ((ds_i[[mu,nu]]*s_ij[[la,sig]] + s_i[[mu,nu]] * ds_ij[[la,sig]]) *
                            (g_i[[mu,la]] + g_ij[[mu,sig]] + g_i[[nu,la]] + g_ij[[nu,sig]]) +
                            s_i[[mu,nu]] * s_ij[[la,sig]] * (dg_i[[mu,la]] + dg_ij[[mu,sig]] + dg_i[[nu,la]] + dg_ij[[nu,sig]]));
                    }
                }
            }
        }
    }
    return integral;
}

pub fn f_exchange_loop(
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
)->Array5<f64>{
    let mut integral:Array5<f64> = Array5::zeros([3*n_atoms,n_orb_i,n_orb_i,n_orb_j,n_orb_j]);

    let g_i: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, ..n_orb_i]);
    let g_j: ArrayView2<f64> = g0_ao.slice(s![n_orb_i.., n_orb_i..]);
    let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
    let g_ij: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, n_orb_i..]);

    for nc in 0..3 * n_atoms {
        let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
        let dg_j: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
        let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);
        let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);

        for mu in 0..n_orb_i{
            for la in 0..n_orb_i{
                for nu in 0..n_orb_j{
                    for sig in 0..n_orb_j{
                        integral[[nc,mu,nu,la,sig]] = 0.25 * ((ds_ij[[mu,nu]]*s_ij[[la,sig]] + s_ij[[mu,nu]] * ds_ij[[la,sig]]) *
                            (g_i[[mu,la]] + g_ij[[mu,sig]] + g_ij[[la,nu]] + g_j[[nu,sig]]) +
                            s_ij[[mu,nu]] * s_ij[[la,sig]] * (dg_i[[mu,la]] + dg_ij[[mu,sig]] + dg_ij[[la,nu]] + dg_j[[nu,sig]]));
                    }
                }
            }
        }
    }
    return integral;
}

pub fn f_le_ct_exchange_loop(
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
)->Array5<f64>{
    let mut integral:Array5<f64> = Array5::zeros([3*n_atoms,n_orb_i,n_orb_i,n_orb_i,n_orb_j]);

    let s_i: ArrayView2<f64> = s.slice(s![..n_orb_i, ..n_orb_i]);
    let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
    let g_i: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, ..n_orb_i]);
    let g_ij: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, n_orb_i..]);

    for nc in 0..3 * n_atoms {
        let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
        let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
        let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
        let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

        for mu in 0..n_orb_i{
            for nu in 0..n_orb_i{
                for la in 0..n_orb_i{
                    for sig in 0..n_orb_j{
                        integral[[nc,mu,nu,la,sig]] = 0.25 * ((ds_i[[mu,la]]*s_ij[[nu,sig]] + s_i[[mu,la]] * ds_ij[[nu,sig]]) *
                            (g_i[[mu,nu]] + g_ij[[mu,sig]] + g_i[[la,nu]] + g_ij[[la,sig]]) +
                            s_i[[mu,la]] * s_ij[[nu,sig]] * (dg_i[[mu,nu]] + dg_ij[[mu,sig]] + dg_i[[la,nu]] + dg_ij[[la,sig]]));
                    }
                }
            }
        }
    }
    return integral;
}

pub fn f_coulomb_ct_ct_ijij(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j: usize,
    order_ij: bool,
) -> Array3<f64> {
    /// This function calculates the gradient of the coulomb integral
    /// for the CT-CT coupling type IJIJ
    /// The shape of the input array v is [orbs_I,orbs_I]
    /// The shape of the output array is [3*n_atoms,orbs_J,orbs_J]
    // slice the overlap and gamma matrix
    let (s_i, s_j, g_ij) = if order_ij {
        let s_i: ArrayView2<f64> = s.slice(s![..n_orb_i, ..n_orb_i]);
        let s_j: ArrayView2<f64> = s.slice(s![n_orb_i.., n_orb_i..]);
        let g_ij: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, n_orb_i..]);

        (s_i, s_j, g_ij)
    } else {
        let s_i: ArrayView2<f64> = s.slice(s![n_orb_i.., n_orb_i..]);
        let s_j: ArrayView2<f64> = s.slice(s![..n_orb_i, ..n_orb_i]);
        let g_ij: ArrayView2<f64> = g0_ao.slice(s![n_orb_i.., ..n_orb_i]);

        (s_i, s_j, g_ij)
    };

    // calculate specific terms
    let vp = &v + &v.t();
    let v_s_0: Array1<f64> = (&vp * &s_i).sum_axis(Axis(0));
    let g_ji_vs_0: Array1<f64> = g_ij.t().dot(&v_s_0);

    let mut f_return: Array3<f64>;
    if order_ij{
        f_return = Array3::zeros((3 * n_atoms, n_orb_j, n_orb_j));
    }
    else{
        f_return = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));
    }

    for nc in 0..3 * n_atoms {
        // slice the gradient of the overlap and the gradient of the gamma matrix
        let (ds_i, ds_j, dg_ij) = if order_ij {
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_j: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., n_orb_i..]);
            let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i, ds_j, dg_ij)
        } else {
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., n_orb_i..]);
            let ds_j: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_i.., ..n_orb_i]);

            (ds_i, ds_j, dg_ij)
        };

        // calculate specific terms
        let dg_ji_vs_0: Array1<f64> = dg_ij.t().dot(&v_s_0);
        let v_ds_1: Array1<f64> = (&vp * &ds_i).sum_axis(Axis(1));
        let g_ji_vds_1: Array1<f64> = g_ij.t().dot(&v_ds_1);

        let mut df:Array2<f64>;
        if order_ij{
            df = Array2::zeros((n_orb_j, n_orb_j));
            for mu in 0..n_orb_j{
                for nu in 0..n_orb_j{
                    df[[mu,nu]] += ds_j[[mu,nu]] *(g_ji_vs_0[mu] + g_ji_vs_0[nu])
                        +s_j[[mu,nu]] * (dg_ji_vs_0[mu] + dg_ji_vs_0[nu]
                        + g_ji_vds_1[mu] + g_ji_vds_1[nu]);
                }
            }
        }
        else{
            df = Array2::zeros((n_orb_i, n_orb_i));
            for mu in 0..n_orb_i{
                for nu in 0..n_orb_i{
                    df[[mu,nu]] += ds_j[[mu,nu]] *(g_ji_vs_0[mu] + g_ji_vs_0[nu])
                        +s_j[[mu,nu]] * (dg_ji_vs_0[mu] + dg_ji_vs_0[nu]
                        + g_ji_vds_1[mu] + g_ji_vds_1[nu]);
                }
            }
        }
        df = df * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&df);
    }
    return f_return;
}

pub fn f_exchange_ct_ct_ijij(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j: usize,
    order_ij: bool,
) -> Array3<f64> {
    // The pair indices are IJ -> I < J
    let (s_ij, g_i, g_j, g_ij) = if order_ij {
        let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
        let g_i: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, ..n_orb_i]);
        let g_j: ArrayView2<f64> = g0_ao.slice(s![n_orb_i.., n_orb_i..]);
        let g_ij: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, n_orb_i..]);

        (s_ij, g_i, g_j, g_ij)
    } else {
        // The pair indices are JI -> J < I
        let s_ij: ArrayView2<f64> = s.slice(s![n_orb_i.., ..n_orb_i]);
        let g_i: ArrayView2<f64> = g0_ao.slice(s![n_orb_i.., n_orb_i..]);
        let g_j: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, ..n_orb_i]);
        let g_ij: ArrayView2<f64> = g0_ao.slice(s![n_orb_i.., ..n_orb_i]);

        (s_ij, g_i, g_j, g_ij)
    };

    // calculate specific terms
    let sv: Array2<f64> = s_ij.t().dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s_ij.t().dot(&v_t);
    let gv: Array2<f64> = &g_i * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g_ij.t()).reversed_axes();
    let sgv_t: Array2<f64> = s_ij.t().dot(&gv).reversed_axes();

    let mut f_return: Array3<f64>;
    if order_ij{
        f_return = Array3::zeros((3 * n_atoms, n_orb_j, n_orb_j));
    }
    else{
        f_return = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));
    }

    for nc in 0..3 * n_atoms {
        let (ds_ij, dg_i, dg_j, dg_ij) = if order_ij {
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_j: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
            let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_ij, dg_i, dg_j, dg_ij)
        } else {
            // The pair indices are JI -> J < I
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., ..n_orb_i]);
            let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
            let dg_j: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_i.., ..n_orb_i]);

            (ds_ij, dg_i, dg_j, dg_ij)
        };

        let d_sv_t: Array2<f64> = ds_ij.t().dot(&v_t);
        let d_sv: Array2<f64> = ds_ij.t().dot(&v);
        let d_gv: Array2<f64> = &dg_i * &v;

        let mut d_f:Array2<f64>;
        if order_ij{
            d_f = Array2::zeros((n_orb_j, n_orb_j));
        }else{
            d_f = Array2::zeros((n_orb_i, n_orb_i));
        }
        // 1st term
        d_f = d_f + &g_j * &(ds_ij.t().dot(&t_sv));
        // 2nd term
        d_f = d_f + (&d_sv_t * &g_ij.t()).dot(&s_ij);
        // 3rd term
        d_f = d_f + ds_ij.t().dot(&svg_t);
        // 4th term
        d_f = d_f + ds_ij.t().dot(&sgv_t);
        // 5th term
        d_f = d_f + &g_j * &(s_ij.t().dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t * &g_ij.t()).dot(&ds_ij);
        // 7th term
        d_f = d_f + s_ij.t().dot(&(&d_sv * &g_ij.t()).t());
        // 8th term
        d_f = d_f + s_ij.t().dot(&(ds_ij.t().dot(&gv)).t());
        // 9th term
        d_f = d_f + &dg_j * &(s_ij.t().dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t * &dg_ij.t()).dot(&s_ij);
        // 11th term
        d_f = d_f + s_ij.t().dot(&(&sv * &dg_ij.t()).t());
        // 12th term
        d_f = d_f + s_ij.t().dot(&(s_ij.t().dot(&d_gv)).t());
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

pub fn f_coul_ct_ct_ijji(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j: usize,
    order_ij: bool,
) -> Array3<f64> {
    /// This function calculates the gradient of the coulomb integral
    /// for the CT-CT coupling type IJJI
    /// The shape of the input array v is [orbs_i,orbs_j]
    /// The shape of the output array is [3*n_atoms,orbs_j,orbs_i]
    // slice the overlap and gamma matrix
    let (s_ij, g_i, g_j, g_ij) = if order_ij {
        let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
        let g_i: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, ..n_orb_i]);
        let g_j: ArrayView2<f64> = g0_ao.slice(s![n_orb_i.., n_orb_i..]);
        let g_ij: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, n_orb_i..]);

        (s_ij, g_i, g_j, g_ij)
    } else {
        let s_ij: ArrayView2<f64> = s.slice(s![n_orb_j.., ..n_orb_j]);
        let g_i: ArrayView2<f64> = g0_ao.slice(s![n_orb_j.., n_orb_j..]);
        let g_j: ArrayView2<f64> = g0_ao.slice(s![..n_orb_j, ..n_orb_j]);
        let g_ij: ArrayView2<f64> = g0_ao.slice(s![n_orb_j.., ..n_orb_j]);

        (s_ij, g_i, g_j, g_ij)
    };

    // calculate specific terms
    let v_s_0: Array1<f64> = (&v * &s_ij).sum_axis(Axis(0));
    let v_s_1: Array1<f64> = (&v * &s_ij).sum_axis(Axis(1));
    let g_ji_vs: Array1<f64> = g_ij.t().dot(&v_s_1);
    let g_j_vs: Array1<f64> = g_j.dot(&v_s_0);
    let g_i_vs: Array1<f64> = g_i.dot(&v_s_1);
    let g_ij_vs: Array1<f64> = g_ij.dot(&v_s_0);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        // slice the gradient of the overlap and the gradient of the gamma matrix
        let (ds_ij, dg_i, dg_j, dg_ij) = if order_ij {
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_j: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
            let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_ij, dg_i, dg_j, dg_ij)
        } else {
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j.., ..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_j.., n_orb_j..]);
            let dg_j: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_j, ..n_orb_j]);
            let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_j.., ..n_orb_j]);

            (ds_ij, dg_i, dg_j, dg_ij)
        };

        // calculate specific terms
        let dg_ji_vs: Array1<f64> = dg_ij.t().dot(&v_s_1);
        let dg_j_vs: Array1<f64> = dg_j.dot(&v_s_0);
        let dg_i_vs: Array1<f64> = dg_i.dot(&v_s_1);
        let dg_ij_vs: Array1<f64> = dg_ij.dot(&v_s_0);
        let v_ds_0: Array1<f64> = (&v * &ds_ij).sum_axis(Axis(0));
        let v_ds_1: Array1<f64> = (&v * &ds_ij).sum_axis(Axis(1));
        let g_ji_vds: Array1<f64> = g_ij.t().dot(&v_ds_1);
        let g_j_vds: Array1<f64> = g_j.dot(&v_ds_0);
        let g_i_vds: Array1<f64> = g_i.dot(&v_s_1);
        let g_ij_vds: Array1<f64> = g_ij.dot(&v_ds_0);

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_j, n_orb_i));

        for j in 0..n_orb_j {
            for i in 0..n_orb_i {
                d_f[[j, i]] += ds_ij[[i, j]] * (g_ji_vs[j] + g_j_vs[j] + g_i_vs[i] + g_ij_vs[i])
                    + s_ij[[i, j]]
                        * (dg_ji_vs[j]
                            + dg_j_vs[j]
                            + dg_i_vs[i]
                            + dg_ij_vs[i]
                            + g_ji_vds[j]
                            + g_j_vds[j]
                            + g_i_vds[i]
                            + g_ij_vds[i]);
            }
        }
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

pub fn f_exchange_ct_ct_ijji(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j: usize,
    order_ij: bool,
) -> Array3<f64> {
    /// This function calculates the gradient of the exchange integral
    /// for the CT-CT coupling type IJJI
    /// The shape of the input array v is [orbs_i,orbs_j]
    /// The shape of the output array is [3*n_atoms,orbs_j,orbs_i]
    // slice the overlap and gamma matrix
    let (s_ij, g_i, g_j, g_ij) = if order_ij {
        let s_ij: ArrayView2<f64> = s.slice(s![..n_orb_i, n_orb_i..]);
        let g_i: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, ..n_orb_i]);
        let g_j: ArrayView2<f64> = g0_ao.slice(s![n_orb_i.., n_orb_i..]);
        let g_ij: ArrayView2<f64> = g0_ao.slice(s![..n_orb_i, n_orb_i..]);

        (s_ij, g_i, g_j, g_ij)
    } else {
        let s_ij: ArrayView2<f64> = s.slice(s![n_orb_j.., ..n_orb_j]);
        let g_i: ArrayView2<f64> = g0_ao.slice(s![n_orb_j.., n_orb_j..]);
        let g_j: ArrayView2<f64> = g0_ao.slice(s![..n_orb_j, ..n_orb_j]);
        let g_ij: ArrayView2<f64> = g0_ao.slice(s![n_orb_j.., ..n_orb_j]);

        (s_ij, g_i, g_j, g_ij)
    };

    // calculate specific terms
    // for term 1, 3, 9, 11
    let sij_v: Array2<f64> = s_ij.dot(&v.t());
    // for term 3
    let gi_v_sij: Array2<f64> = &g_i * &sij_v.t();
    // for term 4, 8
    let g_ij_v: Array2<f64> = &g_ij * &v;
    // for term 4
    let gij_v_sij: Array2<f64> = g_ij_v.dot(&s_ij.t());
    // for term 6, 10
    let sji_v: Array2<f64> = s_ij.t().dot(&v);
    // for term 6
    let gj_sji_v: Array2<f64> = &sji_v * &g_j;
    // for term 9
    let sji_v_sij: Array2<f64> = s_ij.t().dot(&sij_v.t());

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_j, n_orb_i));

    for nc in 0..3 * n_atoms {
        // slice the gradient of the overlap and the gradient of the gamma matrix
        let (ds_ij, dg_i, dg_j, dg_ij) = if order_ij {
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_j: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
            let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_ij, dg_i, dg_j, dg_ij)
        } else {
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j.., ..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_j.., n_orb_j..]);
            let dg_j: ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_j, ..n_orb_j]);
            let dg_ij: ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_j.., ..n_orb_j]);

            (ds_ij, dg_i, dg_j, dg_ij)
        };

        // calculate specific terms
        let dsij_v: Array2<f64> = ds_ij.dot(&v.t());

        // add the twelve different contributions
        let mut df: Array2<f64> = Array2::zeros((n_orb_j, n_orb_i));
        // term 1
        df = df + &ds_ij.t().dot(&sij_v.t()) * &g_ij.t();
        // term 2
        df = df + (&ds_ij.t().dot(&v) * &g_j).dot(&s_ij.t());
        // term 3
        df = df + ds_ij.t().dot(&gi_v_sij);
        // term 4
        df = df + ds_ij.t().dot(&gij_v_sij);
        // term 5
        df = df + &s_ij.t().dot(&dsij_v) * &g_ij.t();
        // term 6
        df = df + gj_sji_v.dot(&ds_ij.t());
        // term 7
        df = df + ds_ij.t().dot(&(&g_i * &dsij_v.t()));
        // term 8
        df = df + ds_ij.t().dot(&g_ij_v.dot(&ds_ij.t()));
        // term 9
        df = df + &sji_v_sij * &dg_ij.t();
        // term 10
        df = df + (&sji_v * &dg_j).dot(&s_ij.t());
        // term 11
        df = df + s_ij.t().dot(&(&dg_i * &sij_v.t()));
        // term 12
        df = df + s_ij.t().dot(&(&dg_ij * &v).dot(&s_ij.t()));

        df = df * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&df);
    }
    return f_return;
}
