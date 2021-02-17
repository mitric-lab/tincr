#[macro_use(array)]
use ndarray::prelude::*;
use crate::calculator::get_gamma_gradient_matrix;
use crate::h0_and_s::h0_and_s_gradients;
use crate::molecule::{distance_matrix, Molecule};
use crate::parameters::*;
use crate::scc_routine::density_matrix_ref;
use crate::slako_transformations::*;
use crate::solver::*;
use approx::AbsDiffEq;
use ndarray::{array, Array2, Array3, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::AddAssign;

// only ground state
pub fn gradient_lc_gs(
    molecule: &Molecule,
    orbe_occ: Array1<f64>,
    orbe_virt: Array1<f64>,
    orbs_occ: Array2<f64>,
    s: Array2<f64>,
    lc: Option<usize>,
) -> (Array1<f64>, Array1<f64>) {
    let (g0, g1, g0_ao, g1_ao): (Array2<f64>, Array3<f64>, Array2<f64>, Array3<f64>) =
        get_gamma_gradient_matrix(
            &molecule.atomic_numbers,
            molecule.n_atoms,
            molecule.calculator.n_orbs,
            molecule.distance_matrix.view(),
            molecule.directions_matrix.view(),
            &molecule.calculator.hubbard_u,
            &molecule.calculator.valorbs,
            Some(0.0),
        );

    let (g0lr, g1lr, g0lr_ao, g1lr_ao): (Array2<f64>, Array3<f64>, Array2<f64>, Array3<f64>) =
        get_gamma_gradient_matrix(
            &molecule.atomic_numbers,
            molecule.n_atoms,
            molecule.calculator.n_orbs,
            molecule.distance_matrix.view(),
            molecule.directions_matrix.view(),
            &molecule.calculator.hubbard_u,
            &molecule.calculator.valorbs,
            None,
        );
    let n_at: usize = g0.dim().0;
    let n_orb: usize = g0_ao.dim().0;

    let (grad_s, grad_h0): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
        &molecule.atomic_numbers,
        molecule.positions.view(),
        molecule.calculator.n_orbs,
        &molecule.calculator.valorbs,
        molecule.proximity_matrix.view(),
        &molecule.calculator.skt,
        &molecule.calculator.orbital_energies,
    );

    let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
    let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

    // density matrix
    let d = orbs_occ.dot(&orbs_occ.t());
    // reference density matrix
    let d_ref: Array2<f64> = density_matrix_ref(&molecule);

    let diff_d: Array2<f64> = &d - &d_ref;
    // computing F(D-D0)
    let fdmd0: Array3<f64> = f_v(
        diff_d.view(),
        s.view(),
        grad_s.view(),
        g0_ao.view(),
        g1_ao.view(),
        molecule.n_atoms,
        molecule.calculator.n_orbs,
    );
    let mut flr_dmd0: Array3<f64> = Array::zeros((3 * n_at, n_orb, n_orb));
    if lc.unwrap() == 1 {
        flr_dmd0 = f_lr(
            diff_d.view(),
            s.view(),
            grad_s.view(),
            g0_ao.view(),
            g0lr_ao.view(),
            g1_ao.view(),
            g1lr_ao.view(),
            n_at,
            n_orb,
        );
    }

    // energy weighted density matrix
    let d_en = 2.0 * orbs_occ.dot(&ei.dot(&orbs_occ.t()));

    // at the time of writing the code there is no tensordot/einsum functionality availaible
    // in ndarray or other packages. therefore we use indexed loops at the moment
    // tensordot grad_h0, d, axes=([1,2], [0,1])
    // tensordot fdmd0, diff_d, axes=([1,2], [0,1])
    // tensordot grad_s, d_en, axes=([1,2], [0,1])
    let mut grad_e0: Array1<f64> = Array1::zeros([3 * molecule.n_atoms]);
    for i in 0..(3 * molecule.n_atoms) {
        grad_e0[i] += (&grad_h0.slice(s![i, .., ..]) * &d).sum();
        grad_e0[i] += 0.5 * (&fdmd0.slice(s![i, .., ..]) * &diff_d).sum();
        grad_e0[i] -= (&grad_s.slice(s![i, .., ..]) * &d_en).sum();
    }
    if lc.unwrap() == 1 {
        grad_e0 = grad_e0
            - 0.25
                * tensordot(&flr_dmd0, &diff_d, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
                    .into_dimensionality::<Ix1>()
                    .unwrap();
    }
    let grad_v_rep: Array1<f64> = gradient_v_rep(
        &molecule.atomic_numbers,
        molecule.distance_matrix.view(),
        molecule.directions_matrix.view(),
        &molecule.calculator.v_rep,
    );

    return (grad_e0, grad_v_rep);
}

// linear operators
fn f_v(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let mut f: Array3<f64> = Array3::zeros([3 * n_atoms, n_orb, n_orb]);
    let vp: Array2<f64> = &v + &v.t();
    let sv: Array1<f64> = (&s * &vp).sum_axis(Axis(1));
    let gsv: Array1<f64> = g0_ao.dot(&sv);
    let mut gdsv: Array2<f64> = Array2::zeros([3 * n_atoms, n_orb]);
    let mut dgsv: Array2<f64> = Array2::zeros([3 * n_atoms, n_orb]);
    for nc in 0..(3 * n_atoms) {
        gdsv.slice_mut(s![nc, ..])
            .assign(&g0_ao.dot(&(&grad_s.slice(s![nc, .., ..]) * &vp).sum_axis(Axis(1))));
        dgsv.slice_mut(s![nc, ..])
            .assign(&g1_ao.slice(s![nc, .., ..]).dot(&sv));
    }
    // a and b are AOs
    for a in 0..n_orb {
        for b in 0..n_orb {
            let gsv_ab: f64 = gsv[a] + gsv[b];
            let s_ab: f64 = s[[a, b]];
            f.slice_mut(s![.., a, b]).assign(
                //&grad_s.slice(s![.., a, b]) * (gsv.slice(s![a, ..]) + gsv.slice(s![b, ..]))
                &(&((&dgsv.slice(s![.., a])
                    + &dgsv.slice(s![.., b])
                    + &gdsv.slice(s![.., a])
                    + &gdsv.slice(s![.., b]))
                    .map(|x| x * s_ab))
                    + &grad_s.slice(s![.., a, b]).map(|x| x * s_ab)),
            );
        }
    }
    f *= 0.25;
    return f;
}

fn gradients_nolc_ex(
    state: usize,
    g0: ArrayView2<f64>,
    g1: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    g0lr: ArrayView2<f64>,
    g1lr: ArrayView3<f64>,
    g0lr_ao: ArrayView2<f64>,
    g1lr_ao: ArrayView3<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    grad_h0: ArrayView3<f64>,
    XmY: ArrayView3<f64>,
    XpY: ArrayView3<f64>,
    omega: ArrayView1<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
    orbe_occ: Array1<f64>,
    orbe_virt: Array1<f64>,
    orbs_occ: ArrayView2<f64>,
    orbs_virt: ArrayView2<f64>,
    f_dmd0: ArrayView3<f64>,
    f_lrdmd0: ArrayView3<f64>,
    check_z_vec: Option<usize>,
) -> (Array1<f64>) {
    let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
    let ea: Array2<f64> = Array2::from_diag(&orbe_virt);
    let n_occ: usize = orbe_occ.len();
    let n_virt: usize = orbe_virt.len();
    let n_at: usize = g0.dim().0;
    let n_orb: usize = g0_ao.dim().0;

    //select state in XpY and XmY
    let XmY_state: Array2<f64> = XmY.slice(s![state, .., ..]).to_owned();
    let XpY_state: Array2<f64> = XpY.slice(s![state, .., ..]).to_owned();
    let omega_state: f64 = omega[state];

    // vectors U, V and T
    let u_ab: Array2<f64> = tensordot(&XpY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&XmY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    let u_ij: Array2<f64> = tensordot(&XpY_state, &XmY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&XmY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    let v_ab: Array2<f64> = tensordot(&(ei.dot(&XpY_state)), &XpY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&(ei.dot(&XmY_state)), &XmY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    let v_ij: Array2<f64> = tensordot(&(XpY_state.dot(&ea)), &XpY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&(XmY_state.dot(&ea)), &XmY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    let t_ab: Array2<f64> = 0.5
        * tensordot(&XpY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&XmY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    let t_ij: Array2<f64> = 0.5
        * tensordot(&XpY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&XmY_state, &XmY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap();

    // H^+_ij[T_ab]
    let h_pij_tab: Array2<f64> = h_plus_no_lr(
        g0_ao,
        qtrans_oo,
        qtrans_vv,
        qtrans_ov,
        qtrans_ov,
        qtrans_ov,
        qtrans_ov,
        t_ab.view(),
    );
    // H^+_ij[T_ij]
    let h_pij_tij: Array2<f64> = h_plus_no_lr(
        g0_ao,
        qtrans_oo,
        qtrans_oo,
        qtrans_oo,
        qtrans_oo,
        qtrans_oo,
        qtrans_oo,
        t_ij.view(),
    );
    let g_ij: Array2<f64> = h_pij_tab - h_pij_tij;

    // build Q
    let mut qtrans_vo: Array3<f64> = qtrans_ov.to_owned();
    qtrans_vo.swap_axes(1, 2);
    // q_ij
    let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

    // q_ia
    let mut q_ia = tensordot(
        &XpY_state,
        &h_plus_no_lr(
            g0_ao,
            qtrans_vv,
            qtrans_ov,
            qtrans_vo.view(),
            qtrans_vv,
            qtrans_vv,
            qtrans_vo.view(),
            XpY_state.view(),
        ),
        &[Axis(1)],
        &[Axis(1)],
    )
        .into_dimensionality::<Ix2>()
        .unwrap();
    q_ia = q_ia
        + h_plus_no_lr(
        g0_ao,
        qtrans_ov,
        qtrans_vv,
        qtrans_ov,
        qtrans_vv,
        qtrans_ov,
        qtrans_vv,
        t_ab.view(),
    );
    q_ia = q_ia
        - h_plus_no_lr(
        g0_ao,
        qtrans_ov,
        qtrans_oo,
        qtrans_oo,
        qtrans_vo.view(),
        qtrans_oo,
        qtrans_vo.view(),
        t_ij.view(),
    );
    // q_ai
    let q_ai: Array2<f64> = tensordot(
        &XpY_state,
        &h_plus_no_lr(
            g0_ao,
            qtrans_oo,
            qtrans_ov,
            qtrans_oo,
            qtrans_ov,
            qtrans_ov,
            qtrans_oo,
            XpY_state.view(),
        ),
        &[Axis(0)],
        &[Axis(0)],
    )
        .into_dimensionality::<Ix2>()
        .unwrap();
    //q_ab
    let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

    // right hand side
    let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

    // solve z-vector equation
    // build omega
    //let omega_input: Array2<f64> =
    //    get_outer_product(&Array::ones(orbe_occ.len()).view(), &orbe_virt.view())
    //        - get_outer_product(&orbe_occ.view(), &Array::ones(orbe_virt.len()).view());
    let omega_input: Array2<f64> = einsum("i,j->ij", &[&Array::ones(orbe_occ.len()), &orbe_virt])
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap()
        - einsum("i,j->ij", &[&orbe_occ, &Array::ones(orbe_virt.len())])
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let b_matrix_input: Array3<f64> = r_ia.clone().into_shape((n_occ, n_virt, 1)).unwrap();
    let z_ia: Array3<f64> = krylov_solver_zvector(
        omega_input.view(),
        b_matrix_input.view(),
        None,
        None,
        None,
        g0,
        None,
        None,
        None,
        qtrans_ov,
    );
    let z_ia_transformed: Array2<f64> = z_ia.into_shape((n_occ, n_virt)).unwrap();

    if check_z_vec.unwrap() == 1 {
        // compare with full solution
        let gq_ov: Array3<f64> = tensordot(&g0, &qtrans_ov, &[Axis(1)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();

        // build (A+B)_(ia,jb)
        let omega_temp: Array1<f64> = omega_input.into_shape((n_occ * n_virt)).unwrap();
        let tmp: Array2<f64> = Array::from_diag(&omega_temp);
        let mut apb: Array4<f64> = tmp.into_shape((n_occ, n_virt, n_occ, n_virt)).unwrap();
        apb = apb
            + 4.0
            * tensordot(&qtrans_ov, &gq_ov, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix4>()
            .unwrap();

        let apb_transformed: Array2<f64> =
            apb.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
        let err_1: Array1<f64> = apb_transformed
            .dot(&XpY_state.clone().into_shape(n_occ * n_virt).unwrap())
            - omega_state * XmY_state.clone().into_shape(n_occ * n_virt).unwrap();
        let err_sum: f64 = err_1.mapv(|err_1| err_1.abs()).sum();
        assert!(err_sum < 1.0e-5);

        let r_ia_flat: Array1<f64> = r_ia.into_shape(n_occ * n_virt).unwrap();
        // solve for Z
        let z_matrix: Array1<f64> = apb_transformed.solve(&r_ia_flat).unwrap();
        let z_ia_full: Array2<f64> = z_matrix.into_shape((n_occ, n_virt)).unwrap();

        // compare with iterative solution
        let z_diff: Array2<f64> = z_ia_transformed.clone() - z_ia_full;
        let err: f64 = z_diff.mapv(|z_diff| z_diff.abs()).sum();
        assert!(err < 1e-10);
    }
    // build w
    let mut w_ij: Array2<f64> = q_ij
        + h_plus_no_lr(
        g0,
        qtrans_oo,
        qtrans_ov,
        qtrans_oo,
        qtrans_ov,
        qtrans_ov,
        qtrans_oo,
        z_ia_transformed.view(),
    );
    for i in 0..w_ij.dim().0 {
        w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
    }
    let w_ia: Array2<f64> = &q_ai.t() - &ei.dot(&z_ia_transformed);
    let w_ai: Array2<f64> = w_ia.clone().reversed_axes();
    let mut w_ab: Array2<f64> = q_ab;
    for i in 0..w_ab.dim().0 {
        w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
    }
    let length: usize = w_ab.dim().0;
    let mut w_matrix: Array2<f64> = Array::zeros((2 * length, 2 * length));
    for i in 0..length {
        w_matrix
            .slice_mut(s![i, ..length])
            .assign(&w_ij.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![i, length..])
            .assign(&w_ia.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![length + i, ..length])
            .assign(&w_ai.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![length + i, length..])
            .assign(&w_ab.slice(s![i, ..]));
    }
    // assemble gradient

    //dh/dr
    let grad_h: Array3<f64> = &grad_h0 + &f_dmd0;

    // transform vectors to a0 basis
    let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
    let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
    let z_ao: Array2<f64> = orbs_occ.dot(&z_ia_transformed.dot(&orbs_virt.t()));
    let mut orbs: Array2<f64> = Array::zeros((orbs_occ.dim().0 * 2, orbs_occ.dim().1 * 2));
    let length: usize = orbs_occ.dim().0 * 2;
    for i in 0..(length) {
        orbs.slice_mut(s![..length, i])
            .assign(&orbs_occ.slice(s![.., i]));
        orbs.slice_mut(s![length.., i])
            .assign(&orbs_virt.slice(s![.., i]));
    }
    let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
    let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));

    let XpY_ao = orbs_occ.dot(&XpY_state.dot(&orbs_virt.t()));
    let XmY_ao = orbs_occ.dot(&XmY_state.dot(&orbs_virt.t()));

    let mut gradExc: Array1<f64> = Array::zeros(3 * n_at);
    let f: Array3<f64> = f_v(XpY_ao.view(), s, grad_s, g0_ao, g1_ao, n_at, n_orb);
    gradExc = gradExc
        + tensordot(
        &grad_h,
        &(t_vv - t_oo + z_ao),
        &[Axis(1), Axis(2)],
        &[Axis(0), Axis(1)],
    )
        .into_dimensionality::<Ix1>()
        .unwrap();
    gradExc = gradExc
        - tensordot(&grad_s, &w_ao, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
        .into_dimensionality::<Ix1>()
        .unwrap();
    gradExc = gradExc
        + 2.0
        * tensordot(&XpY_ao, &f, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
        .into_dimensionality::<Ix1>()
        .unwrap();

    return gradExc;
}

fn gradients_lc_ex(
    state: usize,
    g0: ArrayView2<f64>,
    g1: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    g0lr: ArrayView2<f64>,
    g1lr: ArrayView3<f64>,
    g0lr_ao: ArrayView2<f64>,
    g1lr_ao: ArrayView3<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    grad_h0: ArrayView3<f64>,
    XmY: ArrayView3<f64>,
    XpY: ArrayView3<f64>,
    omega: ArrayView1<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
    orbe_occ: Array1<f64>,
    orbe_virt: Array1<f64>,
    orbs_occ: ArrayView2<f64>,
    orbs_virt: ArrayView2<f64>,
    f_dmd0: ArrayView3<f64>,
    f_lrdmd0: ArrayView3<f64>,
    check_z_vec: Option<usize>,
) -> (Array1<f64>) {
    let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
    let ea: Array2<f64> = Array2::from_diag(&orbe_virt);
    let n_occ: usize = orbe_occ.len();
    let n_virt: usize = orbe_virt.len();
    let n_at: usize = g0.dim().0;
    let n_orb: usize = g0_ao.dim().0;

    //select state in XpY and XmY
    let XmY_state: Array2<f64> = XmY.slice(s![state, .., ..]).to_owned();
    let XpY_state: Array2<f64> = XpY.slice(s![state, .., ..]).to_owned();
    let omega_state: f64 = omega[state];

    // vectors U, V and T
    let u_ab: Array2<f64> = tensordot(&XpY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&XmY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix2>()
            .unwrap();
    let u_ij: Array2<f64> = tensordot(&XpY_state, &XmY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&XmY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();
    let v_ab: Array2<f64> = tensordot(&(ei.dot(&XpY_state)), &XpY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&(ei.dot(&XmY_state)), &XmY_state, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix2>()
            .unwrap();
    let v_ij: Array2<f64> = tensordot(&(XpY_state.dot(&ea)), &XpY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&(XmY_state.dot(&ea)), &XmY_state, &[Axis(1)], &[Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();
    let t_ab: Array2<f64> = 0.5
        * tensordot(&XpY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix2>()
            .unwrap()
        + tensordot(&XmY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix2>()
            .unwrap();
    let t_ij: Array2<f64> = 0.5
        * tensordot(&XpY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap()
        + tensordot(&XmY_state, &XmY_state, &[Axis(1)], &[Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();

    // H^+_ij[T_ab]
    let h_pij_tab: Array2<f64> = h_plus_lr(
        g0_ao,
        g0lr_ao,
        qtrans_oo,
        qtrans_vv,
        qtrans_ov,
        qtrans_ov,
        qtrans_ov,
        qtrans_ov,
        t_ab.view(),
    );
    // H^+_ij[T_ij]
    let h_pij_tij: Array2<f64> = h_plus_lr(
        g0_ao,
        g0lr_ao,
        qtrans_oo,
        qtrans_oo,
        qtrans_oo,
        qtrans_oo,
        qtrans_oo,
        qtrans_oo,
        t_ij.view(),
    );
    let g_ij: Array2<f64> = h_pij_tab - h_pij_tij;

    // build Q
    let mut qtrans_vo: Array3<f64> = qtrans_ov.to_owned();
    qtrans_vo.swap_axes(1, 2);
    // q_ij
    let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

    // q_ia
    let mut q_ia = tensordot(
        &XpY_state,
        &h_plus_lr(
            g0_ao,
            g0lr_ao,
            qtrans_vv,
            qtrans_ov,
            qtrans_vo.view(),
            qtrans_vv,
            qtrans_vv,
            qtrans_vo.view(),
            XpY_state.view(),
        ),
        &[Axis(1)],
        &[Axis(1)],
    )
    .into_dimensionality::<Ix2>()
    .unwrap();
    q_ia = q_ia
        + tensordot(
            &XmY_state,
            &h_minus(
                g0lr_ao,
                qtrans_vv,
                qtrans_vo.view(),
                qtrans_vo.view(),
                qtrans_vv,
                XmY_state.view(),
            ),
            &[Axis(1)],
            &[Axis(1)],
        )
        .into_dimensionality::<Ix2>()
        .unwrap();
    q_ia = q_ia
        + h_plus_lr(
            g0_ao,
            g0lr_ao,
            qtrans_ov,
            qtrans_vv,
            qtrans_ov,
            qtrans_vv,
            qtrans_ov,
            qtrans_vv,
            t_ab.view(),
        );
    q_ia = q_ia
        - h_plus_lr(
            g0_ao,
            g0lr_ao,
            qtrans_ov,
            qtrans_oo,
            qtrans_oo,
            qtrans_vo.view(),
            qtrans_oo,
            qtrans_vo.view(),
            t_ij.view(),
        );
    // q_ai
    let q_ai: Array2<f64> = tensordot(
        &XpY_state,
        &h_plus_lr(
            g0_ao,
            g0lr_ao,
            qtrans_oo,
            qtrans_ov,
            qtrans_oo,
            qtrans_ov,
            qtrans_ov,
            qtrans_oo,
            XpY_state.view(),
        ),
        &[Axis(0)],
        &[Axis(0)],
    )
    .into_dimensionality::<Ix2>()
    .unwrap()
        + tensordot(
            &XmY_state,
            &h_minus(
                g0lr_ao,
                qtrans_ov,
                qtrans_oo,
                qtrans_oo,
                qtrans_ov,
                XmY_state.view(),
            ),
            &[Axis(0)],
            &[Axis(0)],
        )
        .into_dimensionality::<Ix2>()
        .unwrap();

    //q_ab
    let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

    // right hand side
    let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

    // solve z-vector equation
    // build omega
    //let omega_input: Array2<f64> =
    //    get_outer_product(&Array::ones(orbe_occ.len()).view(), &orbe_virt.view())
    //        - get_outer_product(&orbe_occ.view(), &Array::ones(orbe_virt.len()).view());
    let omega_input: Array2<f64> = einsum("i,j->ij", &[&Array::ones(orbe_occ.len()), &orbe_virt])
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap()
        - einsum("i,j->ij", &[&orbe_occ, &Array::ones(orbe_virt.len())])
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
    let b_matrix_input: Array3<f64> = r_ia.clone().into_shape((n_occ, n_virt, 1)).unwrap();
    let z_ia: Array3<f64> = krylov_solver_zvector(
        omega_input.view(),
        b_matrix_input.view(),
        None,
        None,
        None,
        g0,
        Some(g0lr),
        Some(qtrans_oo),
        Some(qtrans_vv),
        qtrans_ov,
    );
    let z_ia_transformed: Array2<f64> = z_ia.into_shape((n_occ, n_virt)).unwrap();

    if check_z_vec.unwrap() == 1 {
        // compare with full solution
        let gq_ov: Array3<f64> = tensordot(&g0, &qtrans_ov, &[Axis(1)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();
        let gq_lr_oo: Array3<f64> = tensordot(&g0lr, &qtrans_oo, &[Axis(1)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();
        let gq_lr_ov: Array3<f64> = tensordot(&g0lr, &qtrans_ov, &[Axis(1)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();
        let gq_lr_vv: Array3<f64> = tensordot(&g0lr, &qtrans_vv, &[Axis(1)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();

        // build (A+B)_(ia,jb)
        let omega_temp: Array1<f64> = omega_input.into_shape((n_occ * n_virt)).unwrap();
        let tmp: Array2<f64> = Array::from_diag(&omega_temp);
        let mut apb: Array4<f64> = tmp.into_shape((n_occ, n_virt, n_occ, n_virt)).unwrap();
        apb = apb
            + 4.0
                * tensordot(&qtrans_ov, &gq_ov, &[Axis(0)], &[Axis(0)])
                    .into_dimensionality::<Ix4>()
                    .unwrap();
        let mut tmp: Array4<f64> = tensordot(&qtrans_oo, &gq_lr_vv, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix4>()
            .unwrap();
        tmp.swap_axes(1, 2);
        apb = apb - tmp;
        let mut tmp: Array4<f64> = tensordot(&qtrans_ov, &gq_lr_ov, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix4>()
            .unwrap();
        tmp.swap_axes(1, 3);
        apb = apb - tmp;

        let apb_transformed: Array2<f64> =
            apb.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
        let err_1: Array1<f64> = apb_transformed
            .dot(&XpY_state.clone().into_shape(n_occ * n_virt).unwrap())
            - omega_state * XmY_state.clone().into_shape(n_occ * n_virt).unwrap();
        let err_sum: f64 = err_1.mapv(|err_1| err_1.abs()).sum();
        assert!(err_sum < 1.0e-5);

        let r_ia_flat: Array1<f64> = r_ia.into_shape(n_occ * n_virt).unwrap();
        // solve for Z
        let z_matrix: Array1<f64> = apb_transformed.solve(&r_ia_flat).unwrap();
        let z_ia_full: Array2<f64> = z_matrix.into_shape((n_occ, n_virt)).unwrap();

        // compare with iterative solution
        let z_diff: Array2<f64> = z_ia_transformed.clone() - z_ia_full;
        let err: f64 = z_diff.mapv(|z_diff| z_diff.abs()).sum();
        assert!(err < 1e-10);
    }
    // build w
    let mut w_ij: Array2<f64> = q_ij
        + h_plus_lr(
            g0,
            g0lr,
            qtrans_oo,
            qtrans_ov,
            qtrans_oo,
            qtrans_ov,
            qtrans_ov,
            qtrans_oo,
            z_ia_transformed.view(),
        );
    for i in 0..w_ij.dim().0 {
        w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
    }
    let w_ia: Array2<f64> = &q_ai.t() - &ei.dot(&z_ia_transformed);
    let w_ai: Array2<f64> = w_ia.clone().reversed_axes();
    let mut w_ab: Array2<f64> = q_ab;
    for i in 0..w_ab.dim().0 {
        w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
    }
    let length: usize = w_ab.dim().0;
    let mut w_matrix: Array2<f64> = Array::zeros((2 * length, 2 * length));
    for i in 0..length {
        w_matrix
            .slice_mut(s![i, ..length])
            .assign(&w_ij.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![i, length..])
            .assign(&w_ia.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![length + i, ..length])
            .assign(&w_ai.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![length + i, length..])
            .assign(&w_ab.slice(s![i, ..]));
    }
    // assemble gradient

    //dh/dr
    let grad_h: Array3<f64> = &grad_h0 + &f_dmd0 - 0.5 * &f_lrdmd0;

    // transform vectors to a0 basis
    let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
    let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
    let z_ao: Array2<f64> = orbs_occ.dot(&z_ia_transformed.dot(&orbs_virt.t()));
    let mut orbs: Array2<f64> = Array::zeros((orbs_occ.dim().0 * 2, orbs_occ.dim().1 * 2));
    let length: usize = orbs_occ.dim().0 * 2;
    for i in 0..(length) {
        orbs.slice_mut(s![..length, i])
            .assign(&orbs_occ.slice(s![.., i]));
        orbs.slice_mut(s![length.., i])
            .assign(&orbs_virt.slice(s![.., i]));
    }
    let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
    let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));

    let XpY_ao = orbs_occ.dot(&XpY_state.dot(&orbs_virt.t()));
    let XmY_ao = orbs_occ.dot(&XmY_state.dot(&orbs_virt.t()));

    let mut gradExc: Array1<f64> = Array::zeros(3 * n_at);
    let f: Array3<f64> = f_v(XpY_ao.view(), s, grad_s, g0_ao, g1_ao, n_at, n_orb);
    let flr_p = f_lr(
        (&XpY_ao + &XpY_ao.t()).view(),
        s,
        grad_s,
        g0_ao,
        g0lr_ao,
        g1_ao,
        g1lr_ao,
        n_at,
        n_orb,
    );
    let flr_m = -f_lr(
        (&XmY_ao - &XmY_ao.t()).view(),
        s,
        grad_s,
        g0_ao,
        g0lr_ao,
        g1_ao,
        g1lr_ao,
        n_at,
        n_orb,
    );
    gradExc = gradExc
        + tensordot(
            &grad_h,
            &(t_vv - t_oo + z_ao),
            &[Axis(1), Axis(2)],
            &[Axis(0), Axis(1)],
        )
        .into_dimensionality::<Ix1>()
        .unwrap();
    gradExc = gradExc
        - tensordot(&grad_s, &w_ao, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix1>()
            .unwrap();
    gradExc = gradExc
        + 2.0
            * tensordot(&XpY_ao, &f, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
                .into_dimensionality::<Ix1>()
                .unwrap();
    gradExc = gradExc
        - 0.5
            * tensordot(&XpY_ao, &flr_p, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
                .into_dimensionality::<Ix1>()
                .unwrap();
    gradExc = gradExc
        - 0.5
            * tensordot(&XmY_ao, &flr_m, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
                .into_dimensionality::<Ix1>()
                .unwrap();

    return gradExc;
}

fn get_outer_product(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> (Array2<f64>) {
    let mut matrix: Array2<f64> = Array::zeros((v1.len(), v2.len()));
    for (i, i_value) in v1.outer_iter().enumerate() {
        for (j, j_value) in v2.outer_iter().enumerate() {
            matrix[[i, j]] = (&i_value * &j_value).into_scalar();
        }
    }
    return matrix;
}

fn f_lr(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> (Array3<f64>) {
    let sv: Array2<f64> = s.dot(&v);
    let svt: Array2<f64> = s.dot(&v.t());
    let gv: Array2<f64> = &g0_lr_a0 * &v;
    let dsv24: Array3<f64> = tensordot(&grad_s, &v, &[Axis(2)], &[Axis(1)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let dsv23: Array3<f64> = tensordot(&grad_s, &v, &[Axis(2)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let mut dgv: Array3<f64> = Array::zeros((3 * n_atoms, n_orb, n_orb));

    let mut flr: Array3<f64> = Array::zeros((3 * n_atoms, n_orb, n_orb));
    let tmp1: Array3<f64> = tensordot(&grad_s, &sv, &[Axis(2)], &[Axis(1)])
        .into_dimensionality::<Ix3>()
        .unwrap();

    let mut tmp2: Array3<f64> = Array::zeros((3 * n_atoms, n_orb, n_orb));
    let mut tmp7: Array3<f64> = Array::zeros((3 * n_atoms, n_orb, n_orb));
    let mut tmp10: Array3<f64> = Array::zeros((3 * n_atoms, n_orb, n_orb));
    let mut tmp11: Array3<f64> = Array::zeros((3 * n_atoms, n_orb, n_orb));

    //for nc in 0..(3*n_atoms){
    //    dgv.slice_mut(s![nc,..,..]).assign(&(g1_lr_ao.slice(s![nc,..,..]).to_owned()*&v));
    //    flr.slice_mut(s![nc,..,..]).add_assign(&(&g0_lr_a0*&tmp1.slice(s![nc,..,..])));
    //    tmp2.slice_mut(s![nc,..,..]).assign(&(&dsv24.slice(s![nc,..,..])*&g0_lr_a0));
    //    tmp7.slice_mut(s![nc,..,..]).assign(&(&dsv23.slice(s![nc,..,..])*&g0_lr_a0));
    //    tmp10.slice_mut(s![nc,..,..]).assign(&(&svt*&g1_lr_ao.slice(s![nc,..,..])));
    //    tmp11.slice_mut(s![nc,..,..]).assign(&(&sv*&g1_lr_ao.slice(s![nc,..,..])));
    //}

    // replace loop with einsums
    dgv = einsum("ijk,jk->ijk", &[&g1_lr_ao, &v])
        .unwrap()
        .into_dimensionality::<Ix3>()
        .unwrap();
    flr = flr
        + einsum("jk,ijk->ijk", &[&g0_lr_a0, &tmp1])
            .unwrap()
            .into_dimensionality::<Ix3>()
            .unwrap();
    tmp2 = einsum("ijk,jk->ijk", &[&dsv24, &g0_lr_a0])
        .unwrap()
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp7 = einsum("ijk,jk->ijk", &[&dsv23, &g0_lr_a0])
        .unwrap()
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp10 = einsum("jk,ijk->ijk", &[&svt, &g1_lr_ao])
        .unwrap()
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp11 = einsum("jk,ijk->ijk", &[&sv, &g1_lr_ao])
        .unwrap()
        .into_dimensionality::<Ix3>()
        .unwrap();

    flr = flr
        + tensordot(&tmp2, &s, &[Axis(2)], &[Axis(1)])
            .into_dimensionality::<Ix3>()
            .unwrap();
    flr = flr + tensordot(&grad_s, &(&sv * &g0_lr_a0), &[Axis(2)], &[Axis(1)]);
    flr = flr + tensordot(&grad_s, &s.dot(&gv), &[Axis(2)], &[Axis(1)]);

    let mut tmp5: Array3<f64> = tensordot(&s, &dsv23, &[Axis(1)], &[Axis(2)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp5.swap_axes(0, 1);

    for nc in 0..(3 * n_atoms) {
        flr.slice_mut(s![nc, .., ..])
            .add_assign(&(&g0_lr_a0 * &tmp5.slice(s![nc, .., ..])));
    }
    let mut tmp_6: Array3<f64> = tensordot(&(&svt * &g0_lr_a0), &grad_s, &[Axis(1)], &[Axis(2)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp_6.swap_axes(0, 1);

    flr = flr + tmp_6;

    let mut tmp_7: Array3<f64> = tensordot(&s, &tmp7, &[Axis(1)], &[Axis(2)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp_7.swap_axes(0, 1);

    flr = flr + tmp_7;

    let mut tmp8: Array3<f64> = tensordot(&grad_s, &gv, &[Axis(2)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp8.swap_axes(0, 1);

    flr = flr + tmp8;

    let tmp9: Array2<f64> = s.dot(&sv.t());

    for nc in 0..(3 * n_atoms) {
        flr.slice_mut(s![nc, .., ..])
            .add_assign(&(&g1_lr_ao.slice(s![nc, .., ..]) * &tmp9));
    }

    flr = flr
        + tensordot(&tmp10, &s, &[Axis(2)], &[Axis(1)])
            .into_dimensionality::<Ix3>()
            .unwrap();

    let mut tmp_11: Array3<f64> = tensordot(&s, &tmp11, &[Axis(1)], &[Axis(2)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp_11.swap_axes(0, 1);
    flr = flr + tmp_11;

    let mut tmp12: Array3<f64> = tensordot(&dgv, &s, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp12.swap_axes(0, 1);

    flr = flr + tmp12;

    flr = flr * 0.25;

    return flr;
}

fn h_minus(
    g0_lr_a0: ArrayView2<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
    q_ps: ArrayView3<f64>,
    q_qr: ArrayView3<f64>,
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
            .into_dimensionality::<Ix3>()
            .unwrap();
    return h_minus_pq;
}

fn h_plus_lr(
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

fn h_plus_no_lr(
    g0_ao: ArrayView2<f64>,
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
