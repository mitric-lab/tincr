use ndarray::prelude::*;

pub fn f_le_ct_coulomb(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_pair_ao: ArrayView2<f64>,
    g1_pair_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
    bool_ij:bool,
) -> Array3<f64> {
    // The pair indices are IJ -> I < J
    let (s_i,s_ij,g_i,g_ij) = if bool_ij{
        let s_i:ArrayView2<f64> = s.slice(s![..n_orb_i,..n_orb_i]);
        let s_ij:ArrayView2<f64> = s.slice(s![..n_orb_i,n_orb_i..]);
        let g_i:ArrayView2<f64> = g0_pair_ao.slice(s![..n_orb_i,..n_orb_i]);
        let g_ij:ArrayView2<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]);

        (s_i,s_ij,g_i,g_ij)
    }else{
        // The pair indices are JI -> J < I
        let s_i:ArrayView2<f64> = s.slice(s![n_orb_j..,n_orb_j..]);
        let s_ij:ArrayView2<f64> = s.slice(s![n_orb_j..,..n_orb_j]);
        let g_i:ArrayView2<f64> = g0_pair_ao.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_pair_ao.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    };

    let si_v: Array1<f64> = (&s_i * &v).sum_axis(Axis(1));
    let gi_sv:Array1<f64> = g_i.dot(&si_v);
    let gij_sv: Array1<f64> = g_ij.dot(&si_v);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_j));

    for nc in 0..3 * n_atoms {
        let (ds_i,ds_ij,dg_i,dg_ij) = if bool_ij{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_pair_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij:ArrayView2<f64> = g1_pair_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i,ds_ij,dg_i,dg_ij)
        }else{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,n_orb_j..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_pair_ao.slice(s![nc, n_orb_j..,n_orb_j..]);
            let dg_ij:ArrayView2<f64> = g1_pair_ao.slice(s![nc, n_orb_j..,..n_orb_j]);

            (ds_i,ds_ij,dg_i,dg_ij)
        };

        let gi_dsv:Array1<f64> = g_i.dot(&(&ds_i * &v).sum_axis(Axis(1)));
        let gij_dsv:Array1<f64> = g_ij.t().dot(&(&ds_i * &v).sum_axis(Axis(1)));
        let dgi_sv:Array1<f64> = dg_i.dot(&si_v);
        let dgij_sv:Array1<f64> = dg_ij.dot(&si_v);

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_j));

        for b in 0..n_orb_i {
            for a in 0..n_orb_j {
                d_f[[b, a]] = 2.0* ds_ij[[b, a]]  * (gi_sv[b] + gij_sv[a])
                    + 2.0 * s_ij[[b, a]] * (gi_dsv[b] + gij_dsv[a] + dgi_sv[b] + dgij_sv[a]);
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
    n_orb_j:usize,
    bool_ij:bool,
) -> Array3<f64> {
    let (s_i,s_ij,g_i,g_ij) = if bool_ij{
        let s_i:ArrayView2<f64> = s.slice(s![..n_orb_i,..n_orb_i]);
        let s_ij:ArrayView2<f64> = s.slice(s![..n_orb_i,n_orb_i..]);
        let g_i:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    }
    else{
        let s_i:ArrayView2<f64> = s.slice(s![n_orb_j..,n_orb_j..]);
        let s_ij:ArrayView2<f64> = s.slice(s![n_orb_j..,..n_orb_j]);
        let g_i:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    };

    // for term 1
    let gi_v: Array2<f64> = &g_i * &v;
    // for term 1
    let gi_v_sij:Array2<f64> = gi_v.dot(&s_ij);
    // for term 2
    let v_si:Array2<f64> = v.dot(&s_i);
    // for term 4, 10
    let v_sij:Array2<f64> = v.dot(&s_ij);
    // for term 5
    let gi_v_si:Array2<f64> = s_i.dot(&gi_v);
    // for term 7, 11, 12
    let vt_si:Array2<f64> = v.t().dot(&s_i);
    // for term 7
    let gi_vt_si:Array2<f64> = &g_i * &vt_si;
    // for term 8
    let si_v:Array2<f64> = s_i.dot(&v);
    // for term 12
    let vt_si_t_sij:Array2<f64> = vt_si.t().dot(&s_ij);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let (ds_i,ds_ij,dg_i,dg_ij) = if bool_ij{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i,ds_ij,dg_i,dg_ij)
        }else{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,n_orb_j..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j..,n_orb_j..]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j..,..n_orb_j]);

            (ds_i,ds_ij,dg_i,dg_ij)
        };

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_j));
        // 1st term
        d_f = d_f + ds_i.dot(&gi_v_sij);
        // 2nd term
        d_f = d_f + (&v_si * &ds_i).t().dot(&g_ij);
        // 3rd term
        d_f = d_f + (&ds_i.dot(&v) * &g_i).dot(&s_ij);
        // 4th term
        d_f = d_f + &ds_i.dot(&v_sij) *&g_ij;
        // 5th term
        d_f = d_f + gi_v_si.dot(&ds_ij);
        // 6th term
        d_f = d_f + s_i.dot(&(&v.dot(&ds_ij) *&g_ij));
        // 7th term
        d_f = d_f + gi_vt_si.t().dot(&ds_ij);
        // 8th term
        d_f = d_f + &si_v.dot(&ds_ij) * &g_ij;
        // 9th term
        d_f = d_f + s_i.dot(&(&dg_i*&v).dot(&s_ij));
        // 10th term
        d_f = d_f + s_i.dot(&(&v_sij * &dg_ij));
        // 11th term
        d_f = d_f + (&vt_si*&dg_i).t().dot(&s_ij);
        // 12th term
        d_f = d_f + &vt_si_t_sij*&dg_ij;
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
    n_orb_j:usize,
    bool_ij:bool,
) -> Array3<f64> {
    let (s_i,s_ij,g_i,g_ij) = if bool_ij{
        let s_i:ArrayView2<f64> = s.slice(s![..n_orb_i,..n_orb_i]);
        let s_ij:ArrayView2<f64> = s.slice(s![..n_orb_i,n_orb_i..]);
        let g_i:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    }
    else{
        let s_i:ArrayView2<f64> = s.slice(s![n_orb_j..,n_orb_j..]);
        let s_ij:ArrayView2<f64> = s.slice(s![n_orb_j..,..n_orb_j]);
        let g_i:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    };

    // for term 1
    let gi_v: Array2<f64> = &g_i * &v;
    // for term 1
    let gi_v_si:Array2<f64> = gi_v.dot(&s_i);
    // for term 4,10
    let v_si:Array2<f64> = v.dot(&s_i);
    // for term 2
    let v_sij:Array2<f64> = v.dot(&s_ij);
    // for term 5
    let gi_v_sij:Array2<f64> = gi_v.t().dot(&s_ij);
    // for term 7, 11, 12
    let vt_sij:Array2<f64> = v.t().dot(&s_ij);
    // for term 7
    let gij_vt_sij:Array2<f64> = &g_ij * &vt_sij;
    // for term 8
    let sij_v:Array2<f64> = s_ij.t().dot(&v);
    // for term 12
    let si_t_vt_sij:Array2<f64> = s_i.t().dot(&vt_sij);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let (ds_i,ds_ij,dg_i,dg_ij) = if bool_ij{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i,ds_ij,dg_i,dg_ij)
        }else{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,n_orb_j..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j..,n_orb_j..]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j..,..n_orb_j]);

            (ds_i,ds_ij,dg_i,dg_ij)
        };

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_j));
        // 1st term
        d_f = d_f + gi_v_si.t().dot(&ds_ij);
        // 2nd term
        d_f = d_f + g_i.dot(&(&v_sij * &ds_ij));
        // 3rd term
        d_f = d_f + s_i.dot(&(&v.t().dot(&ds_ij) * &g_ij));
        // 4th term
        d_f = d_f + v_si.t().dot(&ds_ij) *&g_ij;
        // 5th term
        d_f = d_f + ds_i.t().dot(&gi_v_sij);
        // 6th term
        d_f = d_f + (&v.dot(&ds_i) *&g_i).t().dot(&s_ij);
        // 7th term
        d_f = d_f + ds_i.t().dot(&gij_vt_sij);
        // 8th term
        d_f = d_f + &ds_i.t().dot(&sij_v.t()) * &g_ij;
        // 9th term
        d_f = d_f + (&dg_i*&v).dot(&s_i).t().dot(&s_ij);
        // 10th term
        d_f = d_f + (&v_si * &dg_i).t().dot(&s_ij);
        // 11th term
        d_f = d_f + s_i.t().dot(&(&vt_sij*&dg_ij));
        // 12th term
        d_f = d_f + &si_t_vt_sij*&dg_ij;
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
    let gsv: Array1<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]).dot(&s_j_v);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let ds_i: ArrayView2<f64> = grad_s_i.slice(s![nc, .., ..]);
        let ds_j: ArrayView2<f64> = grad_s_j.slice(s![nc, .., ..]);
        let dg: ArrayView2<f64> = g1_pair_ao.slice(s![nc, .., ..]);

        let gdsv: Array1<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]).dot(&(&ds_j * &vp).sum_axis(Axis(0)));
        let dgsv: Array1<f64> = dg.slice(s![..n_orb_i,n_orb_i..]).dot(&s_j_v);
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
    let s_ij_outer:ArrayView2<f64> = s_ij.slice(s![..n_orb_i,n_orb_i..]);
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

pub fn f_coulomb_ct_ct_ijij(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
    order_ij:bool,
) -> Array3<f64> {
    // The pair indices are IJ -> I < J
    let (s_i,s_j,g_ij) = if order_ij{
        let s_i:ArrayView2<f64> = s.slice(s![..n_orb_i,..n_orb_i]);
        let s_j:ArrayView2<f64> = s.slice(s![n_orb_i..,n_orb_i..]);
        let g_ij:ArrayView2<f64> = g0_ao.slice(s![..n_orb_i,n_orb_i..]);

        (s_i,s_j,g_ij)
    }else{
        // The pair indices are JI -> J < I
        let s_i:ArrayView2<f64> = s.slice(s![n_orb_j..,n_orb_j..]);
        let s_j:ArrayView2<f64> = s.slice(s![..n_orb_j,..n_orb_j]);
        let g_ij:ArrayView2<f64> = g0_ao.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_j,g_ij)
    };

    let vp: Array2<f64> = &v + &(v.t());
    let s_j_v: Array1<f64> = (&s_j * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g_ij.dot(&s_j_v);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let (ds_i,ds_j,dg_ij) = if order_ij{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_j: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_i.., n_orb_i..]);
            let dg_ij:ArrayView2<f64> = g1_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i,ds_j,dg_ij)
        }else{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,n_orb_j..]);
            let ds_j: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_j,..n_orb_j]);
            let dg_ij:ArrayView2<f64> = g1_ao.slice(s![nc, n_orb_j..,..n_orb_j]);

            (ds_i,ds_j,dg_ij)
        };

        let gdsv: Array1<f64> = g_ij.dot(&(&ds_j * &vp).sum_axis(Axis(0)));
        let dgsv: Array1<f64> = dg_ij.dot(&s_j_v);
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

pub fn f_exchange_ct_ct_ijij(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
    order_ij:bool,
) -> Array3<f64> {
    // The pair indices are IJ -> I < J
    let (s_ij,g_i,g_j,g_ij) = if order_ij{
        let s_ij:ArrayView2<f64> = s.slice(s![..n_orb_i,n_orb_i..]);
        let g_i:ArrayView2<f64> = g0_ao.slice(s![..n_orb_i,..n_orb_i]);
        let g_j:ArrayView2<f64> = g0_ao.slice(s![n_orb_i..,n_orb_i..]);
        let g_ij:ArrayView2<f64> = g0_ao.slice(s![..n_orb_i,n_orb_i..]);

        (s_ij,g_i,g_j,g_ij)
    }else{
        // The pair indices are JI -> J < I
        let s_ij:ArrayView2<f64> = s.slice(s![n_orb_j..,..n_orb_j]);
        let g_i:ArrayView2<f64> = g0_ao.slice(s![n_orb_j..,n_orb_j..]);
        let g_j:ArrayView2<f64> = g0_ao.slice(s![..n_orb_j,..n_orb_j]);
        let g_ij:ArrayView2<f64> = g0_ao.slice(s![n_orb_j..,..n_orb_j]);

        (s_ij,g_i,g_j,g_ij)
    };

    let sv: Array2<f64> = s_ij.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s_ij.dot(&v_t);
    let gv: Array2<f64> = &g_j * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g_ij).reversed_axes();
    let sgv_t: Array2<f64> = s_ij.dot(&gv).reversed_axes();

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let (ds_ij,dg_i,dg_j,dg_ij) = if order_ij{
            let ds_ij:ArrayView2<f64> = grad_s.slice(s![nc,..n_orb_i,n_orb_i..]);
            let dg_i:ArrayView2<f64> = g1_ao.slice(s![nc,..n_orb_i,..n_orb_i]);
            let dg_j:ArrayView2<f64> = g1_ao.slice(s![nc,n_orb_i..,n_orb_i..]);
            let dg_ij:ArrayView2<f64> = g1_ao.slice(s![nc,..n_orb_i,n_orb_i..]);

            (ds_ij,dg_i,dg_j,dg_ij)
        }else{
            // The pair indices are JI -> J < I
            let ds_ij:ArrayView2<f64> = grad_s.slice(s![nc,n_orb_j..,..n_orb_j]);
            let dg_i:ArrayView2<f64> = g1_ao.slice(s![nc,n_orb_j..,n_orb_j..]);
            let dg_j:ArrayView2<f64> = g1_ao.slice(s![nc,..n_orb_j,..n_orb_j]);
            let dg_ij:ArrayView2<f64> = g1_ao.slice(s![nc,n_orb_j..,..n_orb_j]);

            (ds_ij,dg_i,dg_j,dg_ij)
        };

        let d_sv_t: Array2<f64> = ds_ij.dot(&v_t);
        let d_sv: Array2<f64> = ds_ij.dot(&v);
        let d_gv: Array2<f64> = &dg_j * &v;

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_i));
        // 1st term
        d_f = d_f + &g_i * &(ds_ij.dot(&t_sv));
        // 2nd term
        d_f = d_f + (&d_sv_t * &g_ij).dot(&s_ij.t());
        // 3rd term
        d_f = d_f + ds_ij.dot(&svg_t);
        // 4th term
        d_f = d_f + ds_ij.dot(&sgv_t);
        // 5th term
        d_f = d_f + &g_i * &(s_ij.dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t * &g_ij).dot(&ds_ij.t());
        // 7th term
        d_f = d_f + s_ij.dot(&(&d_sv * &g_ij).t());
        // 8th term
        d_f = d_f + s_ij.dot(&(ds_ij.dot(&gv)).t());
        // 9th term
        d_f = d_f + &dg_i * &(s_ij.dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t * &dg_ij).dot(&s_ij.t());
        // 11th term
        d_f = d_f + s_ij.dot(&(&sv * &dg_ij).t());
        // 12th term
        d_f = d_f + s_ij.dot(&(s_ij.dot(&d_gv)).t());
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}