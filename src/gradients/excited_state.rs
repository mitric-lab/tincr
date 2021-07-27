use crate::gradients::helpers::{h_minus, Hplus, HplusType};
use crate::initialization::*;
use ndarray::{s, Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, Axis};
use ndarray_linalg::{into_col, into_row, IntoTriangular, Solve, UPLO};
use std::time::Instant;

impl System {
    pub fn excited_state_gradient(&mut self, state: usize) {
        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = self.occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = self.virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // take state specific values from the excitation vectors
        let xmy_state: Array2<f64> = self
            .properties
            .take_xmy()
            .unwrap()
            .slice(s![state, .., ..])
            .to_owned();
        let xpy_state: Array2<f64> = self
            .properties
            .take_xpy()
            .unwrap()
            .slice(s![state, .., ..])
            .to_owned();
        // excitation energy of the state
        let omega_state: f64 = self.properties.excited_states().unwrap()[state];

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = xpy_state.t().dot(&xmy_state) + xmy_state.t().dot(&xpy_state);
        let u_ij: Array2<f64> = xpy_state.dot(&xmy_state.t()) + xmy_state.dot(&xpy_state.t());

        let v_ab: Array2<f64> =
            ei.dot(&xpy_state).t().dot(&xpy_state) + ei.dot(&xmy_state).t().dot(&xmy_state);
        let v_ij: Array2<f64> =
            xpy_state.dot(&ea).dot(&xpy_state.t()) + xmy_state.dot(&ea).dot(&xmy_state.t());

        let t_ab: Array2<f64> =
            0.5 * (xpy_state.t().dot(&xpy_state) + xmy_state.t().dot(&xmy_state));
        let t_ij: Array2<f64> =
            0.5 * (xpy_state.dot(&xpy_state.t()) + xmy_state.dot(&xmy_state.t()));

        // get the transition charges
        let qtrans_ov: Array3<f64> = self.properties.take_qtrans_ov().unwrap();
        let qtrans_oo: Array3<f64> = self.properties.take_qtrans_oo().unwrap();
        let qtrans_vv: Array3<f64> = self.properties.take_qtrans_vv().unwrap();
        let qtrans_vo: Array3<f64> = qtrans_ov
            .view()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hplus: Hplus = Hplus::new(&qtrans_ov, &qtrans_vv, &qtrans_oo, &qtrans_vo);

        // set gamma matrices
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> = hplus.compute(g0, g0_lr, t_ab.view(), HplusType::Tab);
        let hplus_tij: Array2<f64> = hplus.compute(g0, g0_lr, t_ij.view(), HplusType::Tij);

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> = xpy_state.dot(
            &hplus
                .compute(g0, g0_lr, xpy_state.view(), HplusType::Qia_Xpy)
                .t(),
        );
        q_ia = q_ia
            + xmy_state.dot(
                &h_minus(
                    g0_lr,
                    qtrans_vv.view(),
                    qtrans_vo.view(),
                    qtrans_vo.view(),
                    qtrans_vv.view(),
                    xmy_state.view(),
                )
                .t(),
            );
        q_ia = q_ia + hplus.compute(g0, g0_lr, t_ab.view(), HplusType::Qia_Tab);
        q_ia = q_ia - hplus.compute(g0, g0_lr, t_ij.view(), HplusType::Qia_Tij);

        // calculate q_ai
        let mut q_ai: Array2<f64> =
            xpy_state
                .t()
                .dot(&hplus.compute(g0, g0_lr, xpy_state.view(), HplusType::Qai));
        q_ai = q_ai
            + xmy_state.t().dot(&h_minus(
                g0_lr,
                qtrans_ov.view(),
                qtrans_oo.view(),
                qtrans_oo.view(),
                qtrans_ov.view(),
                xmy_state.view(),
            ));
    }
}

// pub fn gradients_lc_ex(
//     state: usize,
//     g0: ArrayView2<f64>,
//     g1: ArrayView3<f64>,
//     g0_ao: ArrayView2<f64>,
//     g1_ao: ArrayView3<f64>,
//     g0lr: ArrayView2<f64>,
//     g1lr: ArrayView3<f64>,
//     g0lr_ao: ArrayView2<f64>,
//     g1lr_ao: ArrayView3<f64>,
//     s: ArrayView2<f64>,
//     grad_s: ArrayView3<f64>,
//     grad_h0: ArrayView3<f64>,
//     XmY: ArrayView3<f64>,
//     XpY: ArrayView3<f64>,
//     omega: ArrayView1<f64>,
//     qtrans_oo: ArrayView3<f64>,
//     qtrans_vv: ArrayView3<f64>,
//     qtrans_ov: ArrayView3<f64>,
//     orbe_occ: Array1<f64>,
//     orbe_virt: Array1<f64>,
//     orbs_occ: ArrayView2<f64>,
//     orbs_virt: ArrayView2<f64>,
//     f_dmd0: ArrayView3<f64>,
//     f_lrdmd0: ArrayView3<f64>,
//     multiplicity: u8,
//     spin_couplings: ArrayView1<f64>,
//     check_z_vec: Option<usize>,
//     old_z_vec: Option<Array3<f64>>,
// ) -> (Array1<f64>, Array3<f64>) {
//     let grad_timer = Instant::now();
//
//     let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
//     let ea: Array2<f64> = Array2::from_diag(&orbe_virt);
//     let n_occ: usize = orbe_occ.len();
//     let n_virt: usize = orbe_virt.len();
//     let n_at: usize = g0.dim().0;
//     let n_orb: usize = g0_ao.dim().0;
//
//     //select state in XpY and XmY
//     let XmY_state: Array2<f64> = XmY.slice(s![state, .., ..]).to_owned();
//     let XpY_state: Array2<f64> = XpY.slice(s![state, .., ..]).to_owned();
//     let omega_state: f64 = omega[state];
//
//     // vectors U, V and T
//     let u_ab: Array2<f64> = tensordot(&XpY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
//         .into_dimensionality::<Ix2>()
//         .unwrap()
//         + tensordot(&XmY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
//         .into_dimensionality::<Ix2>()
//         .unwrap();
//     let u_ij: Array2<f64> = tensordot(&XpY_state, &XmY_state, &[Axis(1)], &[Axis(1)])
//         .into_dimensionality::<Ix2>()
//         .unwrap()
//         + tensordot(&XmY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
//         .into_dimensionality::<Ix2>()
//         .unwrap();
//     let v_ab: Array2<f64> = tensordot(&(ei.dot(&XpY_state)), &XpY_state, &[Axis(0)], &[Axis(0)])
//         .into_dimensionality::<Ix2>()
//         .unwrap()
//         + tensordot(&(ei.dot(&XmY_state)), &XmY_state, &[Axis(0)], &[Axis(0)])
//         .into_dimensionality::<Ix2>()
//         .unwrap();
//     let v_ij: Array2<f64> = tensordot(&(XpY_state.dot(&ea)), &XpY_state, &[Axis(1)], &[Axis(1)])
//         .into_dimensionality::<Ix2>()
//         .unwrap()
//         + tensordot(&(XmY_state.dot(&ea)), &XmY_state, &[Axis(1)], &[Axis(1)])
//         .into_dimensionality::<Ix2>()
//         .unwrap();
//     let t_ab: Array2<f64> = 0.5
//         * (tensordot(&XpY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
//         .into_dimensionality::<Ix2>()
//         .unwrap()
//         + tensordot(&XmY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
//         .into_dimensionality::<Ix2>()
//         .unwrap());
//     let t_ij: Array2<f64> = 0.5
//         * (tensordot(&XpY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
//         .into_dimensionality::<Ix2>()
//         .unwrap()
//         + tensordot(&XmY_state, &XmY_state, &[Axis(1)], &[Axis(1)])
//         .into_dimensionality::<Ix2>()
//         .unwrap());
//     // H^+_ij[T_ab]
//     let h_pij_tab: Array2<f64> = h_plus_lr(
//         g0,
//         g0lr,
//         qtrans_oo,
//         qtrans_vv,
//         qtrans_ov,
//         qtrans_ov,
//         qtrans_ov,
//         qtrans_ov,
//         t_ab.view(),
//     );
//     // H^+_ij[T_ij]
//     let h_pij_tij: Array2<f64> = h_plus_lr(
//         g0,
//         g0lr,
//         qtrans_oo,
//         qtrans_oo,
//         qtrans_oo,
//         qtrans_oo,
//         qtrans_oo,
//         qtrans_oo,
//         t_ij.view(),
//     );
//     let g_ij: Array2<f64> = h_pij_tab - h_pij_tij;
//     // build Q
//     let mut qtrans_vo: Array3<f64> = qtrans_ov.to_owned();
//     qtrans_vo.swap_axes(1, 2);
//     // q_ij
//     let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;
//     // q_ia
//     let mut q_ia = tensordot(
//         &XpY_state,
//         &h_plus_lr(
//             g0,
//             g0lr,
//             qtrans_vv,
//             qtrans_ov,
//             qtrans_vo.view(),
//             qtrans_vv,
//             qtrans_vv,
//             qtrans_vo.view(),
//             XpY_state.view(),
//         ),
//         &[Axis(1)],
//         &[Axis(1)],
//     )
//         .into_dimensionality::<Ix2>()
//         .unwrap();
//
//     q_ia = q_ia
//         + tensordot(
//         &XmY_state,
//         &h_minus(
//             g0lr,
//             qtrans_vv,
//             qtrans_vo.view(),
//             qtrans_vo.view(),
//             qtrans_vv,
//             XmY_state.view(),
//         ),
//         &[Axis(1)],
//         &[Axis(1)],
//     )
//         .into_dimensionality::<Ix2>()
//         .unwrap();
//     q_ia = q_ia
//         + h_plus_lr(
//         g0,
//         g0lr,
//         qtrans_ov,
//         qtrans_vv,
//         qtrans_ov,
//         qtrans_vv,
//         qtrans_ov,
//         qtrans_vv,
//         t_ab.view(),
//     );
//     q_ia = q_ia
//         - h_plus_lr(
//         g0,
//         g0lr,
//         qtrans_ov,
//         qtrans_oo,
//         qtrans_oo,
//         qtrans_vo.view(),
//         qtrans_oo,
//         qtrans_vo.view(),
//         t_ij.view(),
//     );
//     // q_ai
//     let q_ai: Array2<f64> = tensordot(
//         &XpY_state,
//         &h_plus_lr(
//             g0,
//             g0lr,
//             qtrans_oo,
//             qtrans_ov,
//             qtrans_oo,
//             qtrans_ov,
//             qtrans_ov,
//             qtrans_oo,
//             XpY_state.view(),
//         ),
//         &[Axis(0)],
//         &[Axis(0)],
//     )
//         .into_dimensionality::<Ix2>()
//         .unwrap()
//         + tensordot(
//         &XmY_state,
//         &h_minus(
//             g0lr,
//             qtrans_ov,
//             qtrans_oo,
//             qtrans_oo,
//             qtrans_ov,
//             XmY_state.view(),
//         ),
//         &[Axis(0)],
//         &[Axis(0)],
//     )
//         .into_dimensionality::<Ix2>()
//         .unwrap();
//
//     //q_ab
//     let q_ab: Array2<f64> = omega_state * u_ab + v_ab;
//
//     info!(
//         "{:>68} {:>8.2} s",
//         "elapsed time for tensor dots in excited gradients:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//     let grad_timer = Instant::now();
//
//     // right hand side
//     let r_ia: Array2<f64> = &q_ai.t() - &q_ia;
//     // solve z-vector equation
//     // build omega
//     //let omega_input: Array2<f64> =
//     //    get_outer_product(&Array::ones(orbe_occ.len()).view(), &orbe_virt.view())
//     //        - get_outer_product(&orbe_occ.view(), &Array::ones(orbe_virt.len()).view());
//     // let omega_input: Array2<f64> = einsum("i,j->ij", &[&Array::ones(orbe_occ.len()), &orbe_virt])
//     //     .unwrap()
//     //     .into_dimensionality::<Ix2>()
//     //     .unwrap()
//     //     - einsum("i,j->ij", &[&orbe_occ, &Array::ones(orbe_virt.len())])
//     //         .unwrap()
//     //         .into_dimensionality::<Ix2>()
//     //         .unwrap();
//     let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
//         .dot(&into_row(orbe_virt.clone()))
//         - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
//     let b_matrix_input: Array3<f64> = r_ia.clone().into_shape((n_occ, n_virt, 1)).unwrap();
//
//     let z_ia: Array3<f64> = krylov_solver_zvector(
//         omega_input.view(),
//         b_matrix_input.view(),
//         old_z_vec,
//         None,
//         None,
//         g0,
//         Some(g0lr),
//         Some(qtrans_oo),
//         Some(qtrans_vv),
//         qtrans_ov,
//         1,
//         multiplicity,
//         spin_couplings,
//     );
//     let z_ia_transformed: Array2<f64> = z_ia.clone().into_shape((n_occ, n_virt)).unwrap();
//
//     info!(
//         "{:>68} {:>8.2} s",
//         "elapsed time for krylov z vector:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//
//     if check_z_vec.is_some() && check_z_vec.unwrap() == 1 {
//         // compare with full solution
//         let gq_ov: Array3<f64> = tensordot(&g0, &qtrans_ov, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         let gq_lr_oo: Array3<f64> = tensordot(&g0lr, &qtrans_oo, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         let gq_lr_ov: Array3<f64> = tensordot(&g0lr, &qtrans_ov, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         let gq_lr_vv: Array3<f64> = tensordot(&g0lr, &qtrans_vv, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//
//         // build (A+B)_(ia,jb)
//         let omega_temp: Array1<f64> = omega_input.into_shape((n_occ * n_virt)).unwrap();
//         let tmp: Array2<f64> = Array::from_diag(&omega_temp);
//         let mut apb: Array4<f64> = tmp.into_shape((n_occ, n_virt, n_occ, n_virt)).unwrap();
//         apb = apb
//             + 4.0
//             * tensordot(&qtrans_ov, &gq_ov, &[Axis(0)], &[Axis(0)])
//             .into_dimensionality::<Ix4>()
//             .unwrap();
//         let mut tmp: Array4<f64> = tensordot(&qtrans_oo, &gq_lr_vv, &[Axis(0)], &[Axis(0)])
//             .into_dimensionality::<Ix4>()
//             .unwrap();
//         tmp.swap_axes(1, 2);
//         apb = apb - tmp;
//         let mut tmp: Array4<f64> = tensordot(&qtrans_ov, &gq_lr_ov, &[Axis(0)], &[Axis(0)])
//             .into_dimensionality::<Ix4>()
//             .unwrap();
//         tmp.swap_axes(1, 3);
//         apb = apb - tmp;
//
//         let apb_transformed: Array2<f64> =
//             apb.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
//         let err_1: Array1<f64> = apb_transformed
//             .dot(&XpY_state.clone().into_shape(n_occ * n_virt).unwrap())
//             - omega_state * XmY_state.clone().into_shape(n_occ * n_virt).unwrap();
//         let err_sum: f64 = err_1.mapv(|err_1| err_1.abs()).sum();
//         assert!(err_sum < 1.0e-5);
//
//         // doesnt work
//         //let r_ia_flat: Array1<f64> = r_ia.clone().into_shape((n_occ * n_virt)).unwrap();
//         //working alternative
//         //let my_vec: Vec<f64> = Vec::from(r_ia.as_slice_memory_order().unwrap());
//         //let my_arr2: Array2<f64> = Array::from_shape_vec((n_virt, n_occ), my_vec.clone()).unwrap();
//         //println!("{:?}", my_arr2.t().iter().cloned().collect::<Vec<f64>>());
//
//         let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape((n_occ * n_virt)).unwrap();
//         // solve for Z
//         let z_matrix: Array1<f64> = apb_transformed.solve(&r_ia_flat).unwrap();
//         let z_ia_full: Array2<f64> = z_matrix.into_shape((n_occ, n_virt)).unwrap();
//
//         // compare with iterative solution
//         let z_diff: Array2<f64> = z_ia_transformed.clone() - z_ia_full;
//         let err: f64 = z_diff.mapv(|z_diff| z_diff.abs()).sum();
//         assert!(err < 1e-10);
//     }
//     let grad_timer = Instant::now();
//
//     // build w
//     let mut w_ij: Array2<f64> = q_ij
//         + h_plus_lr(
//         g0,
//         g0lr,
//         qtrans_oo,
//         qtrans_ov,
//         qtrans_oo,
//         qtrans_ov,
//         qtrans_ov,
//         qtrans_oo,
//         z_ia_transformed.view(),
//     );
//     for i in 0..w_ij.dim().0 {
//         w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
//     }
//
//     let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia_transformed);
//     let w_ai: Array2<f64> = w_ia.clone().reversed_axes();
//     let mut w_ab: Array2<f64> = q_ab;
//     for i in 0..w_ab.dim().0 {
//         w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
//     }
//
//     let length: usize = n_occ + n_virt;
//     let mut w_matrix: Array2<f64> = Array::zeros((length, length));
//     for i in 0..w_ij.dim().0 {
//         w_matrix
//             .slice_mut(s![i, ..w_ij.dim().1])
//             .assign(&w_ij.slice(s![i, ..]));
//         w_matrix
//             .slice_mut(s![i, w_ij.dim().1..])
//             .assign(&w_ia.slice(s![i, ..]));
//     }
//     for i in 0..w_ai.dim().0 {
//         w_matrix
//             .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
//             .assign(&w_ai.slice(s![i, ..]));
//         w_matrix
//             .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
//             .assign(&w_ab.slice(s![i, ..]));
//     }
//     // assemble gradient
//
//     info!(
//         "{:>68} {:>8.2} s",
//         "elapsed time build w matric:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//     let grad_timer = Instant::now();
//
//     //dh/dr
//     let grad_h: Array3<f64> = &grad_h0 + &f_dmd0 - 0.5 * &f_lrdmd0;
//
//     // transform vectors to a0 basis
//     let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
//     let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
//     let z_ao: Array2<f64> = orbs_occ.dot(&z_ia_transformed.dot(&orbs_virt.t()));
//     // numpy hstack
//
//     let mut orbs: Array2<f64> = Array::zeros((length, length));
//
//     for i in 0..length {
//         orbs.slice_mut(s![i, ..orbs_occ.dim().1])
//             .assign(&orbs_occ.slice(s![i, ..]));
//         orbs.slice_mut(s![i, orbs_occ.dim().1..])
//             .assign(&orbs_virt.slice(s![i, ..]));
//     }
//
//     let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
//     let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
//
//     let XpY_ao = orbs_occ.dot(&XpY_state.dot(&orbs_virt.t()));
//     let XmY_ao = orbs_occ.dot(&XmY_state.dot(&orbs_virt.t()));
//
//     let mut gradExc: Array1<f64> = Array::zeros(3 * n_at);
//     let f: Array3<f64> = f_v_new(XpY_ao.view(), s, grad_s, g0_ao, g1_ao, n_at, n_orb);
//
//     let flr_p = f_lr_new(
//         (&XpY_ao + &XpY_ao.t()).view(),
//         s,
//         grad_s,
//         g0lr_ao,
//         g1lr_ao,
//         n_at,
//         n_orb,
//     );
//     let flr_m = -f_lr_new(
//         (&XmY_ao - &XmY_ao.t()).view(),
//         s,
//         grad_s,
//         g0lr_ao,
//         g1lr_ao,
//         n_at,
//         n_orb,
//     );
//     gradExc = gradExc
//         + tensordot(
//         &grad_h,
//         &(t_vv - t_oo + z_ao),
//         &[Axis(1), Axis(2)],
//         &[Axis(0), Axis(1)],
//     )
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//     gradExc = gradExc
//         - tensordot(&grad_s, &w_ao, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//     gradExc = gradExc
//         + 2.0
//         * tensordot(&XpY_ao, &f, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//     gradExc = gradExc
//         - 0.5
//         * tensordot(&XpY_ao, &flr_p, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//     gradExc = gradExc
//         - 0.5
//         * tensordot(&XmY_ao, &flr_m, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//
//     info!(
//         "{:>68} {:>8.2} s",
//         "elapsed time assemble gradient:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//
//     return (gradExc, z_ia);
// }
