#[macro_use(array)]
use ndarray::prelude::*;
use crate::calculator::get_gamma_gradient_matrix;
use crate::calculator::*;
use crate::constants::ATOM_NAMES;
use crate::{defaults, scc_routine};
use crate::h0_and_s::h0_and_s_gradients;
use crate::io::GeneralConfig;
use crate::molecule::{distance_matrix, Molecule};
use crate::parameters::*;
use crate::scc_routine::density_matrix_ref;
use crate::slako_transformations::*;
use crate::solver::*;
use crate::test::get_water_molecule;
use crate::transition_charges::trans_charges;
use crate::molecule::*;
use crate::gradients::*;
use approx::AbsDiffEq;
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::Data;
use ndarray::{array, Array2, Array3, ArrayView2, ArrayView3, Slice};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_stats::{DeviationExt, QuantileExt};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::AddAssign;
use std::time::Instant;

pub fn dftb_numerical_tda_gradients(molecule:&mut Molecule,state:usize)->Array1<f64>{
    let positions:Array2<f64> = molecule.positions.clone();
    let mut gradient:Array1<f64> = Array1::zeros(positions.dim().0*3);
    let h:f64 = 1.0e-4;

    for (ind,coord) in positions.iter().enumerate(){
        let mut ei:Array1<f64> = Array1::zeros(positions.dim().0*3);
        ei[ind] = 1.0;
        let ei:Array2<f64> = ei.into_shape(positions.raw_dim()).unwrap();
        let positions_1:Array2<f64> = &positions + &(h *&ei);
        let positions_2:Array2<f64> = &positions + &(-h *&ei);

        molecule.update_geometry(positions_1);
        let (e_gs, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(molecule);
        molecule.calculator.set_active_orbitals(f.to_vec());
        let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
            get_exc_energies(&f, &molecule, Some(200008), &s, &orbe, &orbs, false, Some(String::from("TDA")));
        let energy_1:f64 = tmp.0[state];

        molecule.update_geometry(positions_2);
        let (e_gs_2, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(molecule);
        molecule.calculator.set_active_orbitals(f.to_vec());
        let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
            get_exc_energies(&f, &molecule, Some(200008), &s, &orbe, &orbs, false, Some(String::from("TDA")));
        let energy_2:f64 = tmp.0[state];

        let grad_temp:f64 = (energy_1 - energy_2)/(2.0*h);
        gradient[ind] = grad_temp;
    }
    return gradient;
}

pub fn dftb_tda_numerical_gradients_4th_order(molecule:&mut Molecule,state:usize)->Array1<f64>{
    let positions:Array2<f64> = molecule.positions.clone();
    let mut gradient:Array1<f64> = Array1::zeros(positions.dim().0*3);
    let h:f64 = 1.0e-4;

    for (ind,coord) in positions.iter().enumerate(){
        let mut ei:Array1<f64> = Array1::zeros(positions.dim().0*3);
        ei[ind] = 1.0;
        let ei:Array2<f64> = ei.into_shape(positions.raw_dim()).unwrap();
        let positions_1:Array2<f64> = &positions + &(2.0*h *&ei);
        let positions_2:Array2<f64> = &positions + &(h *&ei);
        let positions_3:Array2<f64> = &positions + &(-h *&ei);
        let positions_4:Array2<f64> = &positions + &(-2.0*h *&ei);

        molecule.update_geometry(positions_1);
        let (e_gs, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(molecule);
        molecule.calculator.set_active_orbitals(f.to_vec());
        let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
            get_exc_energies(&f, &molecule, Some(200008), &s, &orbe, &orbs, false, Some(String::from("TDA")));
        let energy_1:f64 = tmp.0[state];

        molecule.update_geometry(positions_2);
        let (e_gs_2, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(molecule);
        molecule.calculator.set_active_orbitals(f.to_vec());
        let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
            get_exc_energies(&f, &molecule, Some(200008), &s, &orbe, &orbs, false, Some(String::from("TDA")));
        let energy_2:f64 = tmp.0[state];

        molecule.update_geometry(positions_3);
        let (e_gs, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(molecule);
        molecule.calculator.set_active_orbitals(f.to_vec());
        let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
            get_exc_energies(&f, &molecule, Some(200008), &s, &orbe, &orbs, false, Some(String::from("TDA")));
        let energy_3:f64 = tmp.0[state];

        molecule.update_geometry(positions_4);
        let (e_gs_2, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(molecule);
        molecule.calculator.set_active_orbitals(f.to_vec());
        let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
            get_exc_energies(&f, &molecule, Some(200008), &s, &orbe, &orbs, false, Some(String::from("TDA")));
        let energy_4:f64 = tmp.0[state];

        let grad_temp:f64 = (-energy_1 + 8.0*energy_2 -8.0*energy_3 + energy_4)/(12.0*h);
        gradient[ind] = grad_temp;
    }
    return gradient;
}

pub fn get_tda_gradients(
    orbe: &Array1<f64>,
    orbs: &Array2<f64>,
    s: &Array2<f64>,
    molecule: &mut Molecule,
    x_cis:ArrayView3<f64>,
    exc_state: Option<usize>,
    omega: &Option<Array1<f64>>,
    f_occ:&Vec<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    info!("{:^80}", "");
    info!("{:^80}", "Calculating analytic gradient");
    info!("{:-^80}", "");
    let grad_timer = Instant::now();
    let n_at: usize = molecule.n_atoms;
    let active_occ: Vec<usize> = molecule.calculator.active_occ.clone().unwrap();
    let active_virt: Vec<usize> = molecule.calculator.active_virt.clone().unwrap();
    let full_occ: Vec<usize> = molecule.calculator.full_occ.clone().unwrap();
    let full_virt: Vec<usize> = molecule.calculator.full_virt.clone().unwrap();

    let n_occ: usize = active_occ.len();
    let n_virt: usize = active_virt.len();

    let n_occ_full: usize = full_occ.len();
    let n_virt_full: usize = full_virt.len();

    let r_lr = molecule
        .calculator
        .r_lr
        .unwrap_or(defaults::LONG_RANGE_RADIUS);

    let mut grad_e0: Array1<f64> = Array::zeros((3 * n_at));
    let mut grad_ex: Array1<f64> = Array::zeros((3 * n_at));
    let mut grad_vrep: Array1<f64> = Array::zeros((3 * n_at));

    // check if active space is smaller than full space
    // otherwise this part is unnecessary

    let orbe_occ: Array1<f64> = full_occ.iter().map(|&full_occ| orbe[full_occ]).collect();
    let orbe_virt: Array1<f64> = full_virt.iter().map(|&full_virt| orbe[full_virt]).collect();

    let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ_full));
    let mut orbs_virt: Array2<f64> = Array::zeros((orbs.dim().0, n_virt_full));

    for (i, index) in full_occ.iter().enumerate() {
        orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
    }

    let gs_timer: Instant = Instant::now();
    let (gradE0, grad_v_rep, grad_s, grad_h0, fdmdO, flrdmdO, g1, g1_ao, g1lr, g1lr_ao): (
        Array1<f64>,
        Array1<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
    ) = gradient_lc_gs(molecule, &orbe_occ, &orbe_virt, &orbs_occ, s, Some(r_lr));

    println!("gs gradients time: {:>8.8} s",gs_timer.elapsed().as_secs_f32());
    drop(gs_timer);

    // set values for return of the gradients
    grad_e0 = gradE0;
    grad_vrep = grad_v_rep;

    // if an excited state is specified in the input, calculate gradients for it
    // otherwise just return ground state
    if exc_state.is_some() {
        for (i, index) in full_virt.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        let nstates: usize = omega.as_ref().unwrap().len();
        // get transition charges of the complete range of orbitals
        let (qtrans_ov, qtrans_oo, qtrans_vv): (Array3<f64>, Array3<f64>, Array3<f64>) =
            trans_charges(
                &molecule.atomic_numbers,
                &molecule.calculator.valorbs,
                orbs.view(),
                s.view(),
                &full_occ[..],
                &full_virt[..],
            );

        // let tmp: Array1<f64> = tda_grad_nolc(
        //     exc_state.unwrap(),
        //     (&molecule.g0).view(),
        //     (&molecule.g0_ao).view(),
        //     g1_ao.view(),
        //     s.view(),
        //     grad_s.view(),
        //     grad_h0.view(),
        //     omega.as_ref().unwrap().view(),
        //     qtrans_oo.view(),
        //     qtrans_vv.view(),
        //     qtrans_ov.view(),
        //     orbe_occ,
        //     orbe_virt,
        //     orbs_occ.view(),
        //     orbs_virt.view(),
        //     fdmdO.view(),
        //     molecule.multiplicity,
        //     molecule.calculator.spin_couplings.view(),
        //     x_cis,
        // );

        let tmp: Array1<f64> = tda_grad_lc(
            exc_state.unwrap(),
            (&molecule.g0).view(),
            g1.view(),
            (&molecule.g0_ao).view(),
            g1_ao.view(),
            (&molecule.g0_lr).view(),
            g1lr.view(),
            (&molecule.g0_lr_ao).view(),
            g1lr_ao.view(),
            s.view(),
            grad_s.view(),
            grad_h0.view(),
            omega.as_ref().unwrap().view(),
            qtrans_oo.view(),
            qtrans_vv.view(),
            qtrans_ov.view(),
            orbe_occ,
            orbe_virt,
            orbs_occ.view(),
            orbs_virt.view(),
            fdmdO.view(),
            flrdmdO.view(),
            molecule.multiplicity,
            molecule.calculator.spin_couplings.view(),
            x_cis,
        );
        grad_ex = tmp;
    }

    let total_grad: Array2<f64> = (&grad_e0 + &grad_vrep + &grad_ex)
        .into_shape([molecule.n_atoms, 3])
        .unwrap();
    if log_enabled!(Level::Debug) || molecule.config.jobtype == "force" {
        info!("{: <45} ", "Gradient in atomic units");
        info!(
            "{: <4} {: >18} {: >18} {: >18}",
            "Atom", "dE/dx", "dE/dy", "dE/dz"
        );
        info!("{:-^61} ", "");
        for (grad_xyz, at) in total_grad.outer_iter().zip(molecule.atomic_numbers.iter()) {
            info!(
                "{: <4} {:>18.10e} {:>18.10e} {:>18.10e}",
                ATOM_NAMES[*at as usize], grad_xyz[0], grad_xyz[1], grad_xyz[2]
            );
        }
        info!("{:-^61} ", "");
    }
    info!(
        "{:<25} {:>18.10e}",
        "Max gradient component:",
        total_grad.max().unwrap()
    );
    info!(
        "{:<25} {:>18.10e}",
        "RMS gradient:",
        total_grad.root_mean_sq_err(&(&total_grad * 0.0)).unwrap()
    );
    info!("{:-^80} ", "");
    info!(
        "{:>68} {:>8.2} s",
        "elapsed time:",
        grad_timer.elapsed().as_secs_f32()
    );
    info!("{:^80} ", "");
    drop(grad_timer);

    return (grad_e0, grad_vrep, grad_ex);
}

pub fn tda_grad_lc(
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
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
    x_cis:ArrayView3<f64>,
) -> Array1<f64> {
    let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
    let ea: Array2<f64> = Array2::from_diag(&orbe_virt);
    let n_occ: usize = orbe_occ.len();
    let n_virt: usize = orbe_virt.len();
    let n_at: usize = g0.dim().0;
    let n_orb: usize = g0_ao.dim().0;

    //select state in cis coefficients
    let x_state:Array2<f64> = x_cis.slice(s![state,..,..]).to_owned();
    let omega_state: f64 = omega[state];

    // vectors U, V and T
    let u_ab: Array2<f64> = 2.0 *x_state.t().dot(&x_state);
    let u_ij: Array2<f64> = 2.0 *x_state.dot(&x_state.t());
    let v_ab: Array2<f64> = 2.0 *ei.dot(&x_state).t().dot(&x_state);
    let v_ij: Array2<f64> = 2.0 *x_state.dot(&ea).dot(&x_state.t());
    let t_ab: Array2<f64> = x_state.t().dot(&x_state);
    let t_ij: Array2<f64> = x_state.dot(&x_state.t());

    // H^+_ij[T_ab]
    let h_pij_tab: Array2<f64> = 2.0*h_a_lr(
        g0,
        g0lr,
        qtrans_oo,
        qtrans_vv,
        qtrans_ov,
        qtrans_ov,
        t_ab.view(),
    );
    // H^+_ij[T_ij]
    let h_pij_tij: Array2<f64> = 2.0*h_a_lr(
        g0,
        g0lr,
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
        &x_state,
        &(2.0*h_a_lr(
            g0,
            g0lr,
            qtrans_vv,
            qtrans_ov,
            qtrans_vo.view(),
            qtrans_vv,
            x_state.view(),
        )),
        &[Axis(1)],
        &[Axis(1)],
    )
        .into_dimensionality::<Ix2>()
        .unwrap();
    q_ia = q_ia
        + 2.0*h_a_lr(
        g0,
        g0lr,
        qtrans_ov,
        qtrans_vv,
        qtrans_ov,
        qtrans_vv,
        t_ab.view(),
    );
    q_ia = q_ia
        - 2.0*h_a_lr(
        g0,
        g0lr,
        qtrans_ov,
        qtrans_oo,
        qtrans_oo,
        qtrans_vo.view(),
        t_ij.view(),
    );
    // q_ai
    let q_ai: Array2<f64> = tensordot(
        &x_state,
        &(2.0*h_a_lr(
            g0,
            g0lr,
            qtrans_oo,
            qtrans_ov,
            qtrans_oo,
            qtrans_ov,
            x_state.view(),
        )),
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
    // let omega_input: Array2<f64> =
    //    get_outer_product(&Array::ones(orbe_occ.len()).view(), &orbe_virt.view())
    //        - get_outer_product(&orbe_occ.view(), &Array::ones(orbe_virt.len()).view());
    // let omega_input: Array2<f64> = einsum("i,j->ij", &[&Array::ones(orbe_occ.len()), &orbe_virt])
    //     .unwrap()
    //     .into_dimensionality::<Ix2>()
    //     .unwrap()
    //     - einsum("i,j->ij", &[&orbe_occ, &Array::ones(orbe_virt.len())])
    //         .unwrap()
    //         .into_dimensionality::<Ix2>()
    //         .unwrap();
    let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
        .dot(&into_row(orbe_virt.clone()))
        - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
    let b_matrix_input: Array3<f64> = r_ia.clone().into_shape((n_occ, n_virt, 1)).unwrap();

    let z_ia: Array3<f64> = zvector_tda(
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
        1,
        multiplicity,
        spin_couplings,
    );
    let z_ia_transformed: Array2<f64> = z_ia.clone().into_shape((n_occ, n_virt)).unwrap();

    // build w
    let mut w_ij: Array2<f64> = q_ij
        + 2.0*h_a_lr(
        g0,
        g0lr,
        qtrans_oo,
        qtrans_ov,
        qtrans_oo,
        qtrans_ov,
        z_ia_transformed.view(),
    );
    for i in 0..w_ij.dim().0 {
        w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
    }
    let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia_transformed);
    //     + 2.0*h_a_lr(
    //     g0,
    //     g0lr,
    //     qtrans_ov,
    //     qtrans_ov,
    //     qtrans_oo,
    //     qtrans_vv,
    //     z_ia_transformed.view(),
    // );;

    let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia_transformed).t();
    let mut w_ab: Array2<f64> = q_ab;
    for i in 0..w_ab.dim().0 {
        w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
    }

    let length: usize = n_occ + n_virt;
    let mut w_matrix: Array2<f64> = Array::zeros((length, length));
    for i in 0..w_ij.dim().0 {
        w_matrix
            .slice_mut(s![i, ..w_ij.dim().1])
            .assign(&w_ij.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![i, w_ij.dim().1..])
            .assign(&w_ia.slice(s![i, ..]));
    }
    for i in 0..w_ai.dim().0 {
        w_matrix
            .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
            .assign(&w_ai.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
            .assign(&w_ab.slice(s![i, ..]));
    }
    // assemble gradient
    //dh/dr
    let grad_h: Array3<f64> = &grad_h0 + &f_dmd0 -0.5 * &f_lrdmd0;

    // transform vectors to a0 basis
    let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
    let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
    let z_ao: Array2<f64> = orbs_occ.dot(&z_ia_transformed.dot(&orbs_virt.t()));

    let mut orbs: Array2<f64> = Array::zeros((length, length));

    for i in 0..length {
        orbs.slice_mut(s![i, ..orbs_occ.dim().1])
            .assign(&orbs_occ.slice(s![i, ..]));
        orbs.slice_mut(s![i, orbs_occ.dim().1..])
            .assign(&orbs_virt.slice(s![i, ..]));
    }
    let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
    let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
    let x_ao = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

    let mut gradExc: Array1<f64> = Array::zeros(3 * n_at);
    let f: Array3<f64> = f_v_new(x_ao.view(), s, grad_s, g0_ao, g1_ao, n_at, n_orb);

    let flr_p = f_lr_new(
        x_ao.t(),
        s,
        grad_s,
        g0lr_ao,
        g1lr_ao,
        n_at,
        n_orb,
    );
    gradExc = gradExc
        + tensordot(
        &grad_h,
        &(&t_vv - &t_oo + &z_ao),
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
        * tensordot(&x_ao, &f, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
        .into_dimensionality::<Ix1>()
        .unwrap();
    gradExc = gradExc
        - tensordot(&x_ao, &flr_p, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
        .into_dimensionality::<Ix1>()
        .unwrap();

    return (gradExc);
}

pub fn tda_grad_nolc(
    state: usize,
    g0: ArrayView2<f64>,
    g0_ao: ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    grad_h0: ArrayView3<f64>,
    omega: ArrayView1<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
    orbe_occ: Array1<f64>,
    orbe_virt: Array1<f64>,
    orbs_occ: ArrayView2<f64>,
    orbs_virt: ArrayView2<f64>,
    f_dmd0: ArrayView3<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
    x_cis:ArrayView3<f64>,
) -> Array1<f64> {
    let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
    let ea: Array2<f64> = Array2::from_diag(&orbe_virt);
    let n_occ: usize = orbe_occ.len();
    let n_virt: usize = orbe_virt.len();
    let n_at: usize = g0.dim().0;
    let n_orb: usize = g0_ao.dim().0;

    //select state in cis coefficients
    let x_state:Array2<f64> = x_cis.slice(s![state,..,..]).to_owned();
    let omega_state: f64 = omega[state];

    // vectors U, V and T
    let u_ab: Array2<f64> = 2.0 *x_state.t().dot(&x_state);
    let u_ij: Array2<f64> = 2.0 *x_state.dot(&x_state.t());
    let v_ab: Array2<f64> = 2.0 *ei.dot(&x_state).t().dot(&x_state);
    let v_ij: Array2<f64> = 2.0 *x_state.dot(&ea).dot(&x_state.t());
    let t_ab: Array2<f64> = x_state.t().dot(&x_state);
    let t_ij: Array2<f64> = x_state.dot(&x_state.t());

    // H^+_ij[T_ab]
    let h_pij_tab: Array2<f64> = 2.0*h_a_no_lr(
        g0,
        qtrans_oo,
        qtrans_vv,
        t_ab.view(),
    );
    // H^+_ij[T_ij]
    let h_pij_tij: Array2<f64> = 2.0*h_a_no_lr(
        g0,
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
        &x_state,
        &(2.0*h_a_no_lr(
            g0,
            qtrans_vv,
            qtrans_ov,
            x_state.view(),
        )),
        &[Axis(1)],
        &[Axis(1)],
        )
        .into_dimensionality::<Ix2>()
        .unwrap();
    q_ia = q_ia
        + 2.0*h_a_no_lr(
        g0,
        qtrans_ov,
        qtrans_vv,
        t_ab.view(),
    );
    q_ia = q_ia
        - 2.0*h_a_no_lr(
        g0,
        qtrans_ov,
        qtrans_oo,
        t_ij.view(),
    );
    // q_ai
    let q_ai: Array2<f64> = tensordot(
        &x_state,
        &(2.0*h_a_no_lr(
            g0,
            qtrans_oo,
            qtrans_ov,
            x_state.view(),
        )),
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
    // let omega_input: Array2<f64> =
    //    get_outer_product(&Array::ones(orbe_occ.len()).view(), &orbe_virt.view())
    //        - get_outer_product(&orbe_occ.view(), &Array::ones(orbe_virt.len()).view());
    // let omega_input: Array2<f64> = einsum("i,j->ij", &[&Array::ones(orbe_occ.len()), &orbe_virt])
    //     .unwrap()
    //     .into_dimensionality::<Ix2>()
    //     .unwrap()
    //     - einsum("i,j->ij", &[&orbe_occ, &Array::ones(orbe_virt.len())])
    //         .unwrap()
    //         .into_dimensionality::<Ix2>()
    //         .unwrap();
    let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
        .dot(&into_row(orbe_virt.clone()))
        - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
    let b_matrix_input: Array3<f64> = r_ia.clone().into_shape((n_occ, n_virt, 1)).unwrap();

    let z_ia: Array3<f64> = zvector_tda(
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
        0,
        multiplicity,
        spin_couplings,
    );
    let z_ia_transformed: Array2<f64> = z_ia.clone().into_shape((n_occ, n_virt)).unwrap();

    // build w
    let mut w_ij: Array2<f64> = q_ij
        + 2.0*h_a_no_lr(
        g0,
        qtrans_oo,
        qtrans_ov,
        z_ia_transformed.view(),
    );
    for i in 0..w_ij.dim().0 {
        w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
    }
    let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia_transformed);
    //     +2.0*h_a_no_lr(
    //     g0,
    //     qtrans_ov,
    //     qtrans_ov,
    //     z_ia_transformed.view(),
    // );

    let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia_transformed).t();
    let mut w_ab: Array2<f64> = q_ab;
    for i in 0..w_ab.dim().0 {
        w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
    }

    let length: usize = n_occ + n_virt;
    let mut w_matrix: Array2<f64> = Array::zeros((length, length));
    for i in 0..w_ij.dim().0 {
        w_matrix
            .slice_mut(s![i, ..w_ij.dim().1])
            .assign(&w_ij.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![i, w_ij.dim().1..])
            .assign(&w_ia.slice(s![i, ..]));
    }
    for i in 0..w_ai.dim().0 {
        w_matrix
            .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
            .assign(&w_ai.slice(s![i, ..]));
        w_matrix
            .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
            .assign(&w_ab.slice(s![i, ..]));
    }
    // assemble gradient
    //dh/dr
    let grad_h: Array3<f64> = &grad_h0 + &f_dmd0;

    // transform vectors to a0 basis
    let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
    let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
    let z_ao: Array2<f64> = orbs_occ.dot(&z_ia_transformed.dot(&orbs_virt.t()));

    let mut orbs: Array2<f64> = Array::zeros((length, length));

    for i in 0..length {
        orbs.slice_mut(s![i, ..orbs_occ.dim().1])
            .assign(&orbs_occ.slice(s![i, ..]));
        orbs.slice_mut(s![i, orbs_occ.dim().1..])
            .assign(&orbs_virt.slice(s![i, ..]));
    }
    let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
    let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));

    let x_ao = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

    let mut gradExc: Array1<f64> = Array::zeros(3 * n_at);
    let f: Array3<f64> = f_v_new(x_ao.view(), s, grad_s, g0_ao, g1_ao, n_at, n_orb);
    gradExc = gradExc
        + tensordot(
        &grad_h,
        &(&t_vv - &t_oo + &z_ao),
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
        * tensordot(&x_ao, &f, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
        .into_dimensionality::<Ix1>()
        .unwrap();

    return (gradExc);
}

fn h_a_no_lr(
    g0: ArrayView2<f64>,
    q_pq: ArrayView3<f64>,
    q_rs: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> (Array2<f64>) {
    // term 1
    let tmp: Array1<f64> = tensordot(&q_rs, &v_rs, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
        .into_dimensionality::<Ix1>()
        .unwrap();
    let tmp2: Array1<f64> = g0.to_owned().dot(&tmp);
    let hplus_pq: Array2<f64> = 2.0
        * tensordot(&q_pq, &tmp2, &[Axis(0)], &[Axis(0)])
        .into_dimensionality::<Ix2>()
        .unwrap();
    return hplus_pq;
}

fn h_a_lr(
    g0_ao: ArrayView2<f64>,
    g0_lr_a0: ArrayView2<f64>,
    q_pq: ArrayView3<f64>,
    q_rs: ArrayView3<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
    v_rs: ArrayView2<f64>,
) -> (Array2<f64>) {
    // term 1
    let tmp: Array1<f64> = tensordot(&q_rs, &v_rs, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
        .into_dimensionality::<Ix1>()
        .unwrap();
    let tmp2: Array1<f64> = g0_ao.to_owned().dot(&tmp);
    let mut hplus_pq: Array2<f64> = 2.0
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
        - 1.0*tensordot(&q_pr, &tmp2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
        .into_dimensionality::<Ix2>()
        .unwrap();

    return hplus_pq;
}

pub fn zvector_tda(
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
        println!("iteration {}",it);
        let z_timer = Instant::now();

        let mut temp: Array3<f64> = Array3::zeros((n_occ, n_virt, l));
        if it == 0 {
            if lc == 1 {
                temp = get_av_fortran(
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
                temp = get_av_fortran_no_lc(
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
                temp_new_vec = get_av_fortran(
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
                temp_new_vec = get_av_fortran_no_lc(
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
        println!(
            "{:>68} {:>8.8} s",
            "elapsed time apbv:",
            z_timer.elapsed().as_secs_f32());
        drop(z_timer);
        let z_timer = Instant::now();
        temp_old = temp.clone();

        let a_b: Array2<f64> = tensordot(&bs, &temp, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();

        // RHS in basis of expansion vectors
        let b_b: Array2<f64> = tensordot(&bs, &b_matrix, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();

        println!(
            "{:>68} {:>8.8} s",
            "elapsed time until solve:",
            z_timer.elapsed().as_secs_f32());
        drop(z_timer);
        let z_timer = Instant::now();

        // solve
        let mut x_b: Array2<f64> = Array2::zeros((k, l));
        for i in 0..k {
            x_b.slice_mut(s![i, ..])
                .assign((&a_b.solve(&b_b.slice(s![.., i])).unwrap()));
        }
        x_b = x_b.reversed_axes();

        println!(
            "{:>68} {:>8.8} s",
            "elapsed time solve:",
            z_timer.elapsed().as_secs_f32());
        drop(z_timer);
        let z_timer = Instant::now();

        // transform solution vector back into canonical basis
        x_matrix = tensordot(&bs, &x_b, &[Axis(2)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();
        // residual vectors
        let mut w_res: Array3<f64> = Array3::zeros((x_matrix.raw_dim()));
        if lc == 1 {
            w_res = get_av_fortran(
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
            w_res = get_av_fortran_no_lc(
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
        println!(
            "{:>68} {:>8.8} s",
            "elapsed time residuals:",
            z_timer.elapsed().as_secs_f32());
        drop(z_timer);
        let z_timer = Instant::now();

        let mut norms: Array1<f64> = Array::zeros(k);
        for i in 0..k {
            norms[i] = norm_special(&w_res.slice(s![.., .., i]).to_owned());
        }
        println!("norms z_vector {}",norms);
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
        println!("new subspace {}",l);
        println!(
            "{:>68} {:>8.8} s",
            "elapsed time norms + expansion of subspace:",
            z_timer.elapsed().as_secs_f32());
        drop(z_timer);
    }
    return x_matrix;
}

fn get_av_fortran_no_lc(
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
        let tmp21: Vec<f64> = (0..n_at)
            .into_iter()
            .map(|at| {
                let tmp: Array2<f64> = &qtrans_ov.slice(s![at, .., ..]) * &vl;
                tmp.sum()
            })
            .collect();
        let tmp21: Array1<f64> = Array::from(tmp21);
        let tmp22: Array1<f64> = 4.0 * gamma_equiv.dot(&tmp21);

        for at in (0..n_at).into_iter() {
            u_l = u_l + &qtrans_ov.slice(s![at, .., ..]) * tmp22[at];
        }

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}

fn get_av_fortran(
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
    let tmp_q_vv: ArrayView2<f64> = qtrans_vv
        .into_shape((n_virt * n_at, n_virt))
        .unwrap();
    let tmp_q_oo: ArrayView2<f64> = qtrans_oo
        .into_shape((n_at * n_occ, n_occ))
        .unwrap();
    let tmp_q_ov_swapped: ArrayView3<f64> = qtrans_ov.permuted_axes([0,2,1]);
    // tmp_q_ov_swapped.swap_axes(1, 2);
    let tmp_q_ov_shape_1: Array2<f64> =
        tmp_q_ov_swapped.as_standard_layout().to_owned().into_shape((n_at * n_virt, n_occ)).unwrap();
    let mut tmp_q_ov_swapped_2: ArrayView3<f64> = qtrans_ov.permuted_axes([1,0,2]);
    // tmp_q_ov_swapped_2.swap_axes(0, 1);
    let tmp_q_ov_shape_2: Array2<f64> =
        tmp_q_ov_swapped_2.as_standard_layout().to_owned().into_shape((n_occ, n_at * n_virt))
            .unwrap();
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
    println!("Test");

    for i in (0..n_vec) {
        let vl: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // 1st term - KS orbital energy differences
        let mut u_l: Array2<f64> = omega * &vl;

        // 2nd term - Coulomb
        let tmp21:Array1<f64> = qtrans_ov.into_shape([n_at,n_occ*n_virt]).unwrap().dot(&vl.view().into_shape(n_occ*n_virt).unwrap());
        let tmp22: Array1<f64> = 2.0 * gamma_equiv.dot(&tmp21);
        u_l = u_l + tmp22.dot(&qtrans_ov.into_shape([n_at,n_occ*n_virt]).unwrap()).into_shape([n_occ,n_virt]).unwrap();

        // 3rd term - Exchange
        let tmp31: Array3<f64> = tmp_q_vv
            .dot(&vl.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();

        let tmp31_reshaped: Array2<f64> = tmp31.into_shape((n_at, n_virt * n_occ)).unwrap();
        let mut tmp32: Array3<f64> = 1.0*gamma_lr
            .dot(&tmp31_reshaped)
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        tmp32.swap_axes(1, 2);
        let tmp32 = tmp32.as_standard_layout();

        let tmp33: Array2<f64> = tmp_q_oo
            .t()
            .dot(&tmp32.into_shape((n_at * n_occ, n_virt)).unwrap());
        u_l = u_l - tmp33;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}