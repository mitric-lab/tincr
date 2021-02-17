#[macro_use(array)]
use ndarray::prelude::*;
use crate::defaults;
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
use ndarray::Data;
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
    r_lc: Option<f64>,
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
    if r_lc.unwrap_or(defaults::LONG_RANGE_RADIUS)>0.0 {
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
    if r_lc.unwrap_or(defaults::LONG_RANGE_RADIUS)>0.0 {
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
    let t_ab: Array2<f64> = 0.5*(
         tensordot(&XpY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&XmY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap());
    let t_ij: Array2<f64> = 0.5 *
        (tensordot(&XpY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap()
        + tensordot(&XmY_state, &XmY_state, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix2>()
        .unwrap());

    // H^+_ij[T_ab]
    let h_pij_tab: Array2<f64> = h_plus_no_lr(
        g0,
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
        g0,
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
            g0,
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
        g0,
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
        g0,
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
            g0,
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
        0
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
        // doesnt work
        //let r_ia_flat: Array1<f64> = r_ia.clone().into_shape((n_occ * n_virt)).unwrap();
        //working alternative
        //let my_vec: Vec<f64> = Vec::from(r_ia.as_slice_memory_order().unwrap());
        //let my_arr2: Array2<f64> = Array::from_shape_vec((n_virt, n_occ), my_vec.clone()).unwrap();
        //println!("{:?}", my_arr2.t().iter().cloned().collect::<Vec<f64>>());

        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape((n_occ * n_virt)).unwrap();

        // solve for Z
        let z_matrix: Array1<f64> = apb_transformed.solve(&r_ia_flat).unwrap();
        let z_ia_full: Array2<f64> = z_matrix.into_shape((n_occ, n_virt)).unwrap();
        // compare with iterative solution
        let z_diff: Array2<f64> = z_ia_transformed.clone() - z_ia_full;
        let err: f64 = z_diff.mapv(|z_diff| z_diff.abs()).sum();
        assert!(err < 1e-10);
    }
    // build w

    // Continue here with error checking !!!!!!!!!!!!!!!!

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
    println!("w ab {:?}",w_ab);
    println!("w ij {:?}",w_ij);
    println!("w ia {:?}",w_ia);
    println!("w ai {:?}",w_ai);
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
    println!("Test 1234556");

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
        1
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

#[test]
fn exc_gradient_no_lc_routine(){
    let  orbs: Array2<f64> = array![
       [-8.6192475509337374e-01, -1.2183336763751098e-06,
        -2.9726029578790070e-01, -1.0617173035797705e-16,
         4.3204846337269176e-05,  6.5350381550742609e-01],
       [ 2.6757349898771515e-03, -2.0080763751606209e-01,
        -3.6133415221394610e-01,  8.4834397825097296e-01,
         2.8113666974634488e-01, -2.8862829713723015e-01],
       [ 4.2873984947983486e-03, -3.2175920488669046e-01,
        -5.7897493816920131e-01, -5.2944545948125077e-01,
         4.5047246429977195e-01, -4.6247649015464443e-01],
       [ 3.5735716506930821e-03,  5.3637887951745156e-01,
        -4.8258577602132369e-01, -1.0229037571655944e-16,
        -7.5102755864519533e-01, -3.8540254135808827e-01],
       [-1.7925680721591991e-01, -3.6380671959263217e-01,
         2.3851969138617155e-01,  2.2055761208820838e-16,
        -7.2394310468946377e-01, -7.7069773456574175e-01],
       [-1.7925762167314863e-01,  3.6380634174056053e-01,
         2.3851841819041375e-01,  2.5181078277624383e-17,
         7.2383801990121355e-01, -7.7079988940061905e-01]
];
    let active_occupied_orbs: Vec<usize> = vec![2, 3];
    let active_virtual_orbs: Vec<usize> = vec![4, 5];
let  S: Array2<f64> = array![
       [ 1.0000000000000000,  0.0000000000000000,  0.0000000000000000,
         0.0000000000000000,  0.3074918525690681,  0.3074937992389065],
       [ 0.0000000000000000,  1.0000000000000000,  0.0000000000000000,
         0.0000000000000000,  0.0000000000000000, -0.1987769748092704],
       [ 0.0000000000000000,  0.0000000000000000,  1.0000000000000000,
         0.0000000000000000,  0.0000000000000000, -0.3185054221819456],
       [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
         1.0000000000000000, -0.3982160222204482,  0.1327383036929333],
       [ 0.3074918525690681,  0.0000000000000000,  0.0000000000000000,
        -0.3982160222204482,  1.0000000000000000,  0.0268024699984349],
       [ 0.3074937992389065, -0.1987769748092704, -0.3185054221819456,
         0.1327383036929333,  0.0268024699984349,  1.0000000000000000]
];
    let  qtrans_ov: Array3<f64> = array![
       [[-2.8128180967215699e-05, -4.2556726168188097e-01],
        [-5.5841428391588377e-01,  3.9062161363123682e-05],
        [ 2.6230140964894622e-05,  3.7065692971610431e-01],
        [ 8.3798119168991755e-17, -4.1770228089087565e-16]],

       [[ 1.9941574838614257e-01,  2.1276907315399221e-01],
        [ 2.7922766432687390e-01,  2.9822390189614878e-01],
        [-1.7348148207962488e-01, -1.8531671657043469e-01],
        [ 1.3911327941665211e-16,  1.4864632621237686e-16]],

       [[-1.9938762020517536e-01,  2.1279818852788895e-01],
        [ 2.7918661958901014e-01, -2.9826296405751196e-01],
        [ 1.7345525193865990e-01, -1.8534021314566926e-01],
        [-1.2791804822463816e-16,  1.3646231144806550e-16]]
];
    let  qtrans_oo: Array3<f64> = array![
       [[ 8.3848164205032949e-01,  6.1966198783812432e-07,
          1.6942360993663974e-01, -1.6726043633702958e-16],
        [ 6.1966198783877484e-07,  5.8696925273219525e-01,
          6.3044315520843774e-07,  5.9439073263148753e-17],
        [ 1.6942360993663974e-01,  6.3044315520843774e-07,
          8.3509720626502215e-01,  3.0309041988910044e-16],
        [-1.6856147894401218e-16,  5.9439073263148753e-17,
          3.0309041988910044e-16,  9.9999999999999978e-01]],

       [[ 8.0758771317081243e-02,  1.3282878666703324e-01,
         -8.4711785129572156e-02,  7.0739481847429051e-17],
        [ 1.3282878666703324e-01,  2.0651544310932970e-01,
         -1.3057860118005465e-01,  1.0763019545169189e-16],
        [-8.4711785129572156e-02, -1.3057860118005465e-01,
          8.2451766756952941e-02, -6.7819462121347612e-17],
        [ 7.0739481847429051e-17,  1.0763019545169189e-16,
         -6.7819462121347612e-17,  5.5604455852449739e-32]],

       [[ 8.0759586632588587e-02, -1.3282940632902107e-01,
         -8.4711824807067571e-02,  4.7886503144290639e-17],
        [-1.3282940632902107e-01,  2.0651530415847527e-01,
          1.3057797073689964e-01, -8.1137763619098240e-17],
        [-8.4711824807067571e-02,  1.3057797073689964e-01,
          8.2451026978025518e-02, -5.1969527145539535e-17],
        [ 4.7886503144290639e-17, -8.1137763619098240e-17,
         -5.1969527145539535e-17,  2.7922029245655617e-32]]
];
    let  qtrans_vv: Array3<f64> = array![
       [[ 4.1303074685724855e-01, -5.9780600290271213e-06],
        [-5.9780600290271213e-06,  3.2642115209520417e-01]],

       [[ 2.9352824890377344e-01,  3.1440112368021861e-01],
        [ 3.1440112368021861e-01,  3.3674576991286248e-01]],

       [[ 2.9344100423897757e-01, -3.1439514562018966e-01],
        [-3.1439514562018966e-01,  3.3683307799193302e-01]]
];

let  orbe: Array1<f64> = array![
       -0.8688942612301258, -0.4499991998360209, -0.3563323833222918,
       -0.2833072445491910,  0.3766541361485015,  0.4290384545096518
];
let  df: Array2<f64> = array![
       [2.0000000000000000, 2.0000000000000000],
       [2.0000000000000000, 2.0000000000000000]
];
let  omega0: Array2<f64> = array![
       [0.7329867988448483, 0.7853711745345131],
       [0.6599617692239994, 0.7123461449136643]
];
let  gamma0_lr: Array2<f64> = array![
       [0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
       [0.2692279296946004, 0.2923649998054588, 0.2429686492032624],
       [0.2692280400920803, 0.2429686492032624, 0.2923649998054588]
];
let  gamma0_lr_ao: Array2<f64> = array![
       [0.2860554418243039, 0.2860554418243039, 0.2860554418243039,
        0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
       [0.2860554418243039, 0.2860554418243039, 0.2860554418243039,
        0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
       [0.2860554418243039, 0.2860554418243039, 0.2860554418243039,
        0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
       [0.2860554418243039, 0.2860554418243039, 0.2860554418243039,
        0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
       [0.2692279296946004, 0.2692279296946004, 0.2692279296946004,
        0.2692279296946004, 0.2923649998054588, 0.2429686492032624],
       [0.2692280400920803, 0.2692280400920803, 0.2692280400920803,
        0.2692280400920803, 0.2429686492032624, 0.2923649998054588]
];
let  gamma1_lr: Array3<f64> = array![
       [[ 0.0000000000000000,  0.0203615522580970, -0.0067871192953180],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000, -0.0000000000000000,  0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000, -0.0000000000000000,  0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [-0.0203615522580970,  0.0000000000000000, -0.0225742810190271],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0084512391474741],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0135416362745717],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0067871192953180,  0.0225742810190271,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [-0.0101637809408346, -0.0084512391474741,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [-0.0162856857170278, -0.0135416362745717,  0.0000000000000000]]
];
let  gamma1_lr_ao: Array3<f64> = array![
       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0203615522580970, -0.0067871192953180],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0203615522580970, -0.0067871192953180],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0203615522580970, -0.0067871192953180],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0203615522580970, -0.0067871192953180],
        [ 0.0203615522580970,  0.0203615522580970,  0.0203615522580970,
          0.0203615522580970,  0.0000000000000000,  0.0000000000000000],
        [-0.0067871192953180, -0.0067871192953180, -0.0067871192953180,
         -0.0067871192953180,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0101637809408346,  0.0101637809408346,  0.0101637809408346,
          0.0101637809408346,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0162856857170278,  0.0162856857170278,  0.0162856857170278,
          0.0162856857170278,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0203615522580970,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0203615522580970,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0203615522580970,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0203615522580970,  0.0000000000000000],
        [-0.0203615522580970, -0.0203615522580970, -0.0203615522580970,
         -0.0203615522580970,  0.0000000000000000, -0.0225742810190271],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0225742810190271,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0084512391474741],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0084512391474741,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0135416362745717],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0135416362745717,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0067871192953180],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0067871192953180],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0067871192953180],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0067871192953180],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0225742810190271],
        [ 0.0067871192953180,  0.0067871192953180,  0.0067871192953180,
          0.0067871192953180,  0.0225742810190271,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0101637809408346],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0084512391474741],
        [-0.0101637809408346, -0.0101637809408346, -0.0101637809408346,
         -0.0101637809408346, -0.0084512391474741,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0162856857170278],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0135416362745717],
        [-0.0162856857170278, -0.0162856857170278, -0.0162856857170278,
         -0.0162856857170278, -0.0135416362745717,  0.0000000000000000]]
];
let  gamma0: Array2<f64> = array![
       [0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
       [0.3863557889890281, 0.4720158398964135, 0.3084885848056254],
       [0.3863561531176491, 0.3084885848056254, 0.4720158398964135]
];
let  gamma1: Array3<f64> = array![
       [[ 0.0000000000000000,  0.0671593223694436, -0.0223862512902948],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000, -0.0000000000000000,  0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000, -0.0000000000000000,  0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [-0.0671593223694436,  0.0000000000000000, -0.0573037072665056],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0214530568542981],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0343747807663729],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0223862512902948,  0.0573037072665056,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [-0.0335236415187203, -0.0214530568542981,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [-0.0537157867768206, -0.0343747807663729,  0.0000000000000000]]
];
let  gamma0_AO: Array2<f64> = array![
       [0.4467609798860577, 0.4467609798860577, 0.4467609798860577,
        0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
       [0.4467609798860577, 0.4467609798860577, 0.4467609798860577,
        0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
       [0.4467609798860577, 0.4467609798860577, 0.4467609798860577,
        0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
       [0.4467609798860577, 0.4467609798860577, 0.4467609798860577,
        0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
       [0.3863557889890281, 0.3863557889890281, 0.3863557889890281,
        0.3863557889890281, 0.4720158398964135, 0.3084885848056254],
       [0.3863561531176491, 0.3863561531176491, 0.3863561531176491,
        0.3863561531176491, 0.3084885848056254, 0.4720158398964135]
];
let  gamma1_AO: Array3<f64> = array![
       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0671593223694436, -0.0223862512902948],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0671593223694436, -0.0223862512902948],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0671593223694436, -0.0223862512902948],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0671593223694436, -0.0223862512902948],
        [ 0.0671593223694436,  0.0671593223694436,  0.0671593223694436,
          0.0671593223694436,  0.0000000000000000,  0.0000000000000000],
        [-0.0223862512902948, -0.0223862512902948, -0.0223862512902948,
         -0.0223862512902948,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0335236415187203,  0.0335236415187203,  0.0335236415187203,
          0.0335236415187203,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0537157867768206,  0.0537157867768206,  0.0537157867768206,
          0.0537157867768206,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0671593223694436,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0671593223694436,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0671593223694436,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0671593223694436,  0.0000000000000000],
        [-0.0671593223694436, -0.0671593223694436, -0.0671593223694436,
         -0.0671593223694436,  0.0000000000000000, -0.0573037072665056],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0573037072665056,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0214530568542981],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0214530568542981,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0343747807663729],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0343747807663729,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0223862512902948],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0223862512902948],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0223862512902948],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0223862512902948],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0573037072665056],
        [ 0.0223862512902948,  0.0223862512902948,  0.0223862512902948,
          0.0223862512902948,  0.0573037072665056,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0335236415187203],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0214530568542981],
        [-0.0335236415187203, -0.0335236415187203, -0.0335236415187203,
         -0.0335236415187203, -0.0214530568542981,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0537157867768206],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0343747807663729],
        [-0.0537157867768206, -0.0537157867768206, -0.0537157867768206,
         -0.0537157867768206, -0.0343747807663729,  0.0000000000000000]]
];
let  gradS: Array3<f64> = array![
       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.3590399304938401, -0.1196795358000320],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0918870323801337],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.1472329381678249],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.3350012527450977,  0.1558859507933732],
        [ 0.3590399304938401,  0.0000000000000000,  0.0000000000000000,
         -0.3350012527450977,  0.0000000000000000,  0.0000000000000000],
        [-0.1196795358000320,  0.0918870323801337,  0.1472329381678249,
          0.1558859507933732,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.1792213355983593],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.2172441846869481,  0.0796440422386294],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.2204828389926055],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0918870323801337],
        [ 0.0000000000000000,  0.2172441846869481,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.1792213355983593,  0.0796440422386294, -0.2204828389926055,
          0.0918870323801337,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.2871709221530289],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.2204828389926055],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.2172441846869481, -0.1360394644282639],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.1472329381678249],
        [ 0.0000000000000000,  0.0000000000000000,  0.2172441846869481,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.2871709221530289, -0.2204828389926055, -0.1360394644282639,
          0.1472329381678249,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.3590399304938401,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.3350012527450977,  0.0000000000000000],
        [-0.3590399304938401,  0.0000000000000000,  0.0000000000000000,
          0.3350012527450977,  0.0000000000000000, -0.0493263812570877],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0493263812570877,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.2172441846869481,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000, -0.2172441846869481,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0184665480123938],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0184665480123938,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.2172441846869481,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000, -0.2172441846869481,
          0.0000000000000000,  0.0000000000000000,  0.0295894213933693],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0295894213933693,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.1196795358000320],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0918870323801337],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.1472329381678249],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.1558859507933732],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0493263812570877],
        [ 0.1196795358000320, -0.0918870323801337, -0.1472329381678249,
         -0.1558859507933732,  0.0493263812570877,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.1792213355983593],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0796440422386294],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.2204828389926055],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0918870323801337],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0184665480123938],
        [-0.1792213355983593, -0.0796440422386294,  0.2204828389926055,
         -0.0918870323801337, -0.0184665480123938,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.2871709221530289],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.2204828389926055],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.1360394644282639],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.1472329381678249],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0295894213933693],
        [-0.2871709221530289,  0.2204828389926055,  0.1360394644282639,
         -0.1472329381678249, -0.0295894213933693,  0.0000000000000000]]
];
let  gradH0: Array3<f64> = array![
       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.4466562325187367,  0.1488849546574482],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0859724600043996],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.1377558678312508],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.3151837170301471, -0.1441050567093689],
        [-0.4466562325187367,  0.0000000000000000,  0.0000000000000000,
          0.3151837170301471,  0.0000000000000000,  0.0000000000000000],
        [ 0.1488849546574482, -0.0859724600043996, -0.1377558678312508,
         -0.1441050567093689,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.2229567506745117],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.2015137919014903, -0.0727706772643434],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.2062908287050795],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0859724600043996],
        [ 0.0000000000000000, -0.2015137919014903,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [-0.2229567506745117, -0.0727706772643434,  0.2062908287050795,
         -0.0859724600043996,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.3572492944418646],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.2062908287050795],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.2015137919014903,  0.1290297419048926],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.1377558678312508],
        [ 0.0000000000000000,  0.0000000000000000, -0.2015137919014903,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [-0.3572492944418646,  0.2062908287050795,  0.1290297419048926,
         -0.1377558678312508,  0.0000000000000000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.4466562325187367,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.3151837170301471,  0.0000000000000000],
        [ 0.4466562325187367,  0.0000000000000000,  0.0000000000000000,
         -0.3151837170301471,  0.0000000000000000,  0.0738575792762484],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0738575792762484,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.2015137919014903,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.2015137919014903,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0276504073281890],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0276504073281890,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.2015137919014903,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
        [ 0.0000000000000000,  0.0000000000000000,  0.2015137919014903,
          0.0000000000000000,  0.0000000000000000, -0.0443049536699000],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000, -0.0443049536699000,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.1488849546574482],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0859724600043996],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.1377558678312508],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.1441050567093689],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.0738575792762484],
        [-0.1488849546574482,  0.0859724600043996,  0.1377558678312508,
          0.1441050567093689, -0.0738575792762484,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.2229567506745117],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0727706772643434],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.2062908287050795],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0859724600043996],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0276504073281890],
        [ 0.2229567506745117,  0.0727706772643434, -0.2062908287050795,
          0.0859724600043996,  0.0276504073281890,  0.0000000000000000]],

       [[ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.3572492944418646],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.2062908287050795],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000, -0.1290297419048926],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.1377558678312508],
        [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
          0.0000000000000000,  0.0000000000000000,  0.0443049536699000],
        [ 0.3572492944418646, -0.2062908287050795, -0.1290297419048926,
          0.1377558678312508,  0.0443049536699000,  0.0000000000000000]]
];
let  orbe_occ: Array1<f64> = array![
       -0.8688942612301258, -0.4499991998360209, -0.3563323833222918,
       -0.2833072445491910
];
let  orbe_virt: Array1<f64> = array![
       0.3766541361485015, 0.4290384545096518
];
let  orbs_occ: Array2<f64> = array![
       [-8.6192454822475639e-01, -1.2183272343139559e-06,
        -2.9726068852089849e-01,  2.6222307203584133e-16],
       [ 2.6757514101551499e-03, -2.0080751179749709e-01,
        -3.6133406147264924e-01,  8.4834397825097341e-01],
       [ 4.2874248054290296e-03, -3.2175900344462377e-01,
        -5.7897479277210717e-01, -5.2944545948124977e-01],
       [ 3.5735935812255637e-03,  5.3637854372423877e-01,
        -4.8258565481599014e-01,  3.4916084620212056e-16],
       [-1.7925702667910837e-01, -3.6380704327437935e-01,
         2.3851989294050652e-01, -2.0731761365694774e-16],
       [-1.7925784113431714e-01,  3.6380666541125695e-01,
         2.3851861974976313e-01, -9.2582148396003538e-17]
];
let  orbs_virt: Array2<f64> = array![
       [ 4.3204927822809713e-05,  6.5350390970909367e-01],
       [ 2.8113675949215844e-01, -2.8862841063399913e-01],
       [ 4.5047260810178097e-01, -4.6247667201341525e-01],
       [-7.5102779853473878e-01, -3.8540269278994982e-01],
       [-7.2394294209812204e-01, -7.7069762107665973e-01],
       [ 7.2383785715168458e-01, -7.7079977605735461e-01]
];
let  nocc: usize = 4;
let  nvirt: usize = 2;
let  gradVrep: Array1<f64> = array![
        0.1578504879797087,  0.1181937590058072,  0.1893848779393944,
       -0.2367773309532266,  0.0000000000000000,  0.0000000000000000,
        0.0789268429735179, -0.1181937590058072, -0.1893848779393944
];
let  gradE0: Array1<f64> = array![
       -0.1198269660296263, -0.0897205271709892, -0.1437614915530440,
        0.1981679566666738, -0.0068989246413182, -0.0110543231055452,
       -0.0783409906370475,  0.0966194518123075,  0.1548158146585892
];
let  gradExc: Array1<f64> = array![
        0.3607539392221090,  0.2702596932404471,  0.4330440071185614,
       -0.7696026181183455,  0.0854981908865757,  0.1369959343140749,
        0.4088486788962364, -0.3557578841270227, -0.5700399414326363
];
let  omega: Array1<f64> = array![
       0.6599613806976925, 0.7123456990588429, 0.7456810724193919,
       0.7930925652350215, 0.8714866033195531, 0.9348736014087142,
       1.2756452171931041, 1.3231856682450711
];
let  XmY: Array3<f64> = array![
       [[ 1.2521414892619180e-17, -3.0731988457045628e-18],
        [-2.0314998941993035e-17,  6.1984001001408129e-17],
        [-1.1949222929340009e-16,  1.8444992477011119e-17],
        [-9.9999999999999978e-01, -8.5678264450787104e-18]],

       [[-5.6554497734654670e-33, -1.6132260882333921e-17],
        [-1.4165350977779408e-16, -1.2680008989475773e-17],
        [ 1.8003860860678714e-17,  1.5123402272838473e-16],
        [-7.1090761021361151e-18,  9.9999999999999967e-01]],

       [[ 2.1571381149267578e-02, -3.0272950933503227e-07],
        [ 2.9991274783090719e-05,  1.4821884853203227e-01],
        [ 9.9507889372419056e-01,  3.2471746769845221e-06],
        [-1.1917605251843160e-16, -4.4289966266035466e-17]],

       [[ 1.1109010931003951e-06, -1.2585514282188778e-02],
        [-3.2069605724216654e-01,  2.8289343609923937e-05],
        [ 9.1191885302142574e-06, -9.4937779050842874e-01],
        [-1.2601161265688548e-17,  1.0521007298293138e-16]],

       [[ 9.1609974782184502e-06,  6.0701873653873452e-02],
        [-9.6788599240172246e-01,  1.0919490802280248e-05],
        [ 2.9445559547715203e-05,  3.4280398643331461e-01],
        [ 3.1783829909643794e-17, -2.1447832161459006e-16]],

       [[-1.0110298012913056e-01,  9.9804621433036054e-06],
        [ 1.5702586864624103e-05,  1.0113993758544988e+00],
        [-1.7694318737762263e-01,  2.6156157292985052e-05],
        [ 1.0459553569289232e-16,  5.5045412199529119e-18]],

       [[ 1.0046988096296818e+00,  9.6514695327939874e-06],
        [ 1.5280031987729607e-05,  1.3343212650518221e-01],
        [-6.0845305188051618e-02,  1.2489695961642065e-07],
        [ 4.2508419260228061e-17,  2.0822376030340786e-18]],

       [[ 9.3758930989162589e-06, -1.0067757866575049e+00],
        [-8.1915557416596632e-02,  1.5848191461915825e-05],
        [-1.1132783884542197e-06,  5.1182023937152175e-02],
        [ 9.2945083674210509e-18, -5.1592764951007866e-17]]
];
let  XpY: Array3<f64> = array![
       [[ 2.3631728626191835e-17, -6.0439980808443620e-18],
        [-2.5446127779063410e-17,  8.2559786740300318e-17],
        [-1.3271411906099627e-16,  2.1950010553683349e-17],
        [-9.9999999999999989e-01, -7.9377675442458068e-18]],

       [[ 0.0000000000000000e+00, -2.9394105266280956e-17],
        [-1.6438371325218934e-16, -1.0269716205837895e-17],
        [ 1.7497024539465256e-17,  1.6673760312079509e-16],
        [-7.6733577656998602e-18,  9.9999999999999989e-01]],

       [[ 3.6031757025210380e-02, -5.2693108210990112e-07],
        [ 3.3247977274044169e-05,  1.7472610444661485e-01],
        [ 9.7813856605377092e-01,  3.4200094262767256e-06],
        [-1.0544215582531297e-16, -4.5125435583086571e-17]],

       [[ 1.7446652973543630e-06, -2.0596777031218809e-02],
        [-3.3426673906312654e-01,  3.1354975875389786e-05],
        [ 8.4280732845994917e-06, -9.4013443503872962e-01],
        [-1.0485887863620174e-17,  9.4498355264689231e-17]],

       [[ 1.3093105142292459e-05,  9.0405231040797618e-02],
        [-9.1809349842431853e-01,  1.1014103424022125e-05],
        [ 2.4765955236613057e-05,  3.0892988258425691e-01],
        [ 2.4069331860443143e-17, -1.7531274647429604e-16]],

       [[-1.3470126301596952e-01,  1.3856384771384507e-05],
        [ 1.3884867211514623e-05,  9.5099287606168215e-01],
        [-1.3873209262146027e-01,  2.1973326807706339e-05],
        [ 7.3838432359937210e-17,  4.0866099935227155e-18]],

       [[ 9.8099453932498049e-01,  9.8200956604003138e-06],
        [ 9.9018827855576371e-06,  9.1947088357024670e-02],
        [-3.4961749454179805e-02,  7.6894757644272947e-08],
        [ 2.1531794612538988e-17,  1.1603416683232019e-18]],

       [[ 8.8257671645045328e-06, -9.8756150574886925e-01],
        [-5.1176316697317654e-02,  1.0528497536484679e-05],
        [-6.1670714154333164e-07,  3.0378857620766524e-02],
        [ 4.6357942979609401e-18, -2.7775304023765253e-17]]
];
let  FDmD0: Array3<f64> = array![
       [[ 1.2859460550231747e-01,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          5.5169274310242961e-02,  3.1687451887156513e-02],
        [ 0.0000000000000000e+00,  1.2859460550231747e-01,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00, -2.0270307973454305e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          1.2859460550231747e-01,  0.0000000000000000e+00,
          0.0000000000000000e+00, -3.2479632035038002e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  1.2859460550231747e-01,
         -6.9533131791531380e-02,  1.6734579981076435e-02],
        [ 5.5169274310242961e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00, -6.9533131791531380e-02,
          1.9585661921955511e-01,  3.8169975457847086e-03],
        [ 3.1687451887156513e-02, -2.0270307973454305e-02,
         -3.2479632035038002e-02,  1.6734579981076435e-02,
          3.8169975457847086e-03,  8.8967693316386487e-02]],

       [[ 9.6284803364408963e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          2.8121212967854570e-02,  3.6912553323229229e-02],
        [ 0.0000000000000000e+00,  9.6284803364408963e-02,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          3.1985317138097402e-03, -2.0983424814785771e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          9.6284803364408963e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00, -3.8747434343718246e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  9.6284803364408963e-02,
         -3.6418257831913756e-02,  1.6148135476012040e-02],
        [ 2.8121212967854570e-02,  3.1985317138097402e-03,
          0.0000000000000000e+00, -3.6418257831913756e-02,
          8.6622241052568244e-02,  2.8579623716003246e-03],
        [ 3.6912553323229229e-02, -2.0983424814785771e-02,
         -3.8747434343718246e-02,  1.6148135476012040e-02,
          2.8579623716003246e-03,  1.2663887792394501e-01]],

       [[ 1.5427959890582069e-01,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          4.5059337567588315e-02,  5.9145926691507270e-02],
        [ 0.0000000000000000e+00,  1.5427959890582069e-01,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00, -3.8747434343718211e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          1.5427959890582069e-01,  0.0000000000000000e+00,
          3.1985317138097402e-03, -5.8887429279635008e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  1.5427959890582069e-01,
         -5.8353904404745277e-02,  2.5874569789451261e-02],
        [ 4.5059337567588315e-02,  0.0000000000000000e+00,
          3.1985317138097402e-03, -5.8353904404745277e-02,
          1.3879702859582799e-01,  4.5793860814115897e-03],
        [ 5.9145926691507270e-02, -3.8747434343718211e-02,
         -5.8887429279635008e-02,  2.5874569789451261e-02,
          4.5793860814115897e-03,  2.0291670761423028e-01]],

       [[-2.0274730331707744e-01,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
         -7.4441477127642444e-02, -5.7028858578436432e-02],
        [ 0.0000000000000000e+00, -2.0274730331707744e-01,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  3.6865862053497241e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
         -2.0274730331707744e-01,  0.0000000000000000e+00,
          0.0000000000000000e+00,  5.9071112077830515e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00, -2.0274730331707744e-01,
          9.4491516481241836e-02, -2.4618102764941988e-02],
        [-7.4441477127642444e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00,  9.4491516481241836e-02,
         -2.4705490326907464e-01, -5.4645067468726107e-03],
        [-5.7028858578436432e-02,  3.6865862053497241e-02,
          5.9071112077830515e-02, -2.4618102764941988e-02,
         -5.4645067468726107e-03, -1.6817958184022622e-01]],

       [[ 3.6898758456716607e-03,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          2.9627859141285529e-04,  1.4760992744354974e-04],
        [ 0.0000000000000000e+00,  3.6898758456716607e-03,
          0.0000000000000000e+00,  0.0000000000000000e+00,
         -3.1985317138097402e-03, -9.5421289475329930e-05],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          3.6898758456716607e-03,  0.0000000000000000e+00,
          0.0000000000000000e+00, -1.5289596855293426e-04],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  3.6898758456716607e-03,
         -3.8369433581985279e-04,  6.3719956062823315e-05],
        [ 2.9627859141285529e-04, -3.1985317138097402e-03,
          0.0000000000000000e+00, -3.8369433581985279e-04,
         -1.7628095579799831e-03, -9.7699346036441648e-05],
        [ 1.4760992744354974e-04, -9.5421289475329930e-05,
         -1.5289596855293426e-04,  6.3719956062823315e-05,
         -9.7699346036441648e-05, -2.7297919167670365e-03]],

       [[ 5.9123822824664772e-03,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          4.7473475201023347e-04,  2.3651915572095363e-04],
        [ 0.0000000000000000e+00,  5.9123822824664772e-03,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00, -1.5289596855293350e-04],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          5.9123822824664772e-03,  0.0000000000000000e+00,
         -3.1985317138097402e-03, -2.4498911436093775e-04],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  5.9123822824664772e-03,
         -6.1480323129167410e-04,  1.0210011258435735e-04],
        [ 4.7473475201023347e-04,  0.0000000000000000e+00,
         -3.1985317138097402e-03, -6.1480323129167410e-04,
         -2.8245947652112049e-03, -1.5654615674725339e-04],
        [ 2.3651915572095363e-04, -1.5289596855293350e-04,
         -2.4498911436093775e-04,  1.0210011258435735e-04,
         -1.5654615674725339e-04, -4.3740152890092152e-03]],

       [[ 7.4152697814759902e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          1.9272202817399473e-02,  2.5341406691279916e-02],
        [ 0.0000000000000000e+00,  7.4152697814759902e-02,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00, -1.6595554080042929e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          7.4152697814759902e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00, -2.6591480042792510e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  7.4152697814759902e-02,
         -2.4958384689710456e-02,  7.8835227838655493e-03],
        [ 1.9272202817399473e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00, -2.4958384689710456e-02,
          5.1198284049519552e-02,  1.6475092010879019e-03],
        [ 2.5341406691279916e-02, -1.6595554080042929e-02,
         -2.6591480042792510e-02,  7.8835227838655493e-03,
          1.6475092010879019e-03,  7.9211888523839730e-02]],

       [[-9.9974679210080616e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
         -2.8417491559267427e-02, -3.7060163250672774e-02],
        [ 0.0000000000000000e+00, -9.9974679210080616e-02,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  2.1078846104261098e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
         -9.9974679210080616e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00,  3.8900330312271178e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00, -9.9974679210080616e-02,
          3.6801952167733604e-02, -1.6211855432074859e-02],
        [-2.8417491559267427e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00,  3.6801952167733604e-02,
         -8.4859431494588272e-02, -2.7602630255638833e-03],
        [-3.7060163250672774e-02,  2.1078846104261098e-02,
          3.8900330312271178e-02, -1.6211855432074859e-02,
         -2.7602630255638833e-03, -1.2390908600717797e-01]],

       [[-1.6019198118828715e-01,  0.0000000000000000e+00,
          0.0000000000000000e+00,  0.0000000000000000e+00,
         -4.5534072319598551e-02, -5.9382445847228217e-02],
        [ 0.0000000000000000e+00, -1.6019198118828715e-01,
          0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00,  3.8900330312271136e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
         -1.6019198118828715e-01,  0.0000000000000000e+00,
          0.0000000000000000e+00,  5.9132418393995939e-02],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
          0.0000000000000000e+00, -1.6019198118828715e-01,
          5.8968707636036941e-02, -2.5976669902035621e-02],
        [-4.5534072319598551e-02,  0.0000000000000000e+00,
          0.0000000000000000e+00,  5.8968707636036941e-02,
         -1.3597243383061677e-01, -4.4228399246643360e-03],
        [-5.9382445847228217e-02,  3.8900330312271136e-02,
          5.9132418393995939e-02, -2.5976669902035621e-02,
         -4.4228399246643360e-03, -1.9854269232522104e-01]]
];



    let gradEx_test:Array1<f64> = gradients_nolc_ex(
    1,
    gamma0.view(),
    gamma1.view(),
    gamma0_AO.view(),
    gamma1_AO.view(),
    gamma0_lr.view(),
    gamma1_lr.view(),
    gamma0_lr_ao.view(),
    gamma1_lr_ao.view(),
    S.view(),
    gradS.view(),
    gradH0.view(),
    XmY.view(),
    XpY.view(),
    omega.view(),
    qtrans_oo.view(),
    qtrans_vv.view(),
    qtrans_ov.view(),
    orbe_occ,
    orbe_virt,
    orbs_occ.view(),
    orbs_virt.view(),
    FDmD0.view(),
    Some(1),
    );

    assert!(gradEx_test.abs_diff_eq(&gradExc,1e-10));

}

pub trait ToOwnedF<A, D> {
    fn to_owned_f(&self) -> Array<A, D>;
}
impl<A, S, D> ToOwnedF<A, D> for ArrayBase<S, D>
    where
        A: Copy + Clone,
        S: Data<Elem = A>,
        D: Dimension,
{
    fn to_owned_f(&self) -> Array<A, D> {
        let mut tmp = unsafe { Array::uninitialized(self.dim().f()) };
        tmp.assign(self);
        tmp
    }
}