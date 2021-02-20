#[macro_use(array)]
use ndarray::prelude::*;
use crate::calculator::get_gamma_gradient_matrix;
use crate::defaults;
use crate::h0_and_s::h0_and_s_gradients;
use crate::molecule::{distance_matrix, Molecule};
use crate::parameters::*;
use crate::scc_routine::density_matrix_ref;
use crate::slako_transformations::*;
use crate::solver::*;
use crate::transition_charges::trans_charges;
use approx::AbsDiffEq;
use ndarray::Data;
use ndarray::{array, Array2, Array3, ArrayView2, ArrayView3, Slice};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::AddAssign;

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

pub fn get_gradients(
    orbe: &Array1<f64>,
    orbs: &Array2<f64>,
    active_occ: &Vec<usize>,
    active_virt: &Vec<usize>,
    full_occ: &Vec<usize>,
    full_virt: &Vec<usize>,
    s: &Array2<f64>,
    molecule: &Molecule,
    XmY: &Option<Array3<f64>>,
    XpY: &Option<Array3<f64>>,
    exc_state: Option<usize>,
    omega: &Option<Array1<f64>>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_at: usize = molecule.n_atoms;
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
    if (n_occ + n_virt) < orbe.len() {
        // set arrays of orbitals energies of the active space
        let orbe_occ: Array1<f64> = active_occ
            .iter()
            .map(|&active_occ| orbe[active_occ])
            .collect();
        let orbe_virt: Array1<f64> = active_virt
            .iter()
            .map(|&active_virt| orbe[active_virt])
            .collect();

        let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((orbs.dim().0, n_virt));

        for index in active_occ.iter() {
            orbs_occ
                .slice_mut(s![.., *index])
                .assign(&orbs.column(*index));
        }

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
        ) = gradient_lc_gs(&molecule, &orbe_occ, &orbe_virt, &orbs_occ, s, Some(r_lr));

        // set values for return of the gradients
        grad_e0 = gradE0;
        grad_vrep = grad_v_rep;

        // if an excited state is specified in the input, calculate gradients for it
        // otherwise just return ground state
        if exc_state.is_some() {
            for index in active_virt.iter() {
                orbs_virt
                    .slice_mut(s![.., *index])
                    .assign(&orbs.column(*index));
            }

            let nstates: usize = XmY.as_ref().unwrap().dim().0;
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

            // construct XY matrices for active space
            let mut XmY_active: Array3<f64> = Array::zeros((nstates, n_occ_full, n_virt_full));
            let mut XpY_active: Array3<f64> = Array::zeros((nstates, n_occ_full, n_virt_full));

            for n in 0..nstates {
                for (i, occ) in active_occ.iter().enumerate() {
                    for (j, virt) in active_virt.iter().enumerate() {
                        XmY_active
                            .slice_mut(s![n, *occ, *virt - n_occ_full])
                            .assign(&XmY.as_ref().unwrap().slice(s![n, i, j]));
                        XpY_active
                            .slice_mut(s![n, *occ, *virt - n_occ_full])
                            .assign(&XpY.as_ref().unwrap().slice(s![n, i, j]));
                    }
                }
            }
            //check for lc correction
            if r_lr > 0.0 {
                let grad_ex: Array1<f64> = gradients_lc_ex(
                    exc_state.unwrap(),
                    (&molecule.calculator.g0).view(),
                    g1.view(),
                    (&molecule.calculator.g0_ao).view(),
                    g1_ao.view(),
                    (&molecule.calculator.g0_lr).view(),
                    g1lr.view(),
                    (&molecule.calculator.g0_lr_ao).view(),
                    g1lr_ao.view(),
                    s.view(),
                    grad_s.view(),
                    grad_h0.view(),
                    XmY_active.view(),
                    XpY_active.view(),
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
                    None,
                );
            } else {
                let grad_ex: Array1<f64> = gradients_nolc_ex(
                    exc_state.unwrap(),
                    (&molecule.calculator.g0).view(),
                    g1.view(),
                    (&molecule.calculator.g0_ao).view(),
                    g1_ao.view(),
                    (&molecule.calculator.g0_lr).view(),
                    g1lr.view(),
                    (&molecule.calculator.g0_lr_ao).view(),
                    g1lr_ao.view(),
                    s.view(),
                    grad_s.view(),
                    grad_h0.view(),
                    XmY_active.view(),
                    XpY_active.view(),
                    omega.as_ref().unwrap().view(),
                    qtrans_oo.view(),
                    qtrans_vv.view(),
                    qtrans_ov.view(),
                    orbe_occ,
                    orbe_virt,
                    orbs_occ.view(),
                    orbs_virt.view(),
                    fdmdO.view(),
                    None,
                );
            }
        }
    } else {
        // no active space, use full range of orbitals

        let orbe_occ: Array1<f64> = full_occ.iter().map(|&full_occ| orbe[full_occ]).collect();
        let orbe_virt: Array1<f64> = full_virt.iter().map(|&full_virt| orbe[full_virt]).collect();

        let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((orbs.dim().0, n_virt));

        for index in full_occ.iter() {
            orbs_occ
                .slice_mut(s![.., *index])
                .assign(&orbs.column(*index));
        }

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
        ) = gradient_lc_gs(&molecule, &orbe_occ, &orbe_virt, &orbs_occ, s, Some(r_lr));

        // set values for return of the gradients
        grad_e0 = gradE0;
        grad_vrep = grad_v_rep;

        if exc_state.is_some() {
            for index in active_virt.iter() {
                orbs_virt
                    .slice_mut(s![.., *index])
                    .assign(&orbs.column(*index));
            }

            let nstates: usize = XmY.as_ref().unwrap().dim().0;
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

            // construct XY matrices for active space
            let mut XmY_active: Array3<f64> = Array::zeros((nstates, n_occ_full, n_virt_full));
            let mut XpY_active: Array3<f64> = Array::zeros((nstates, n_occ_full, n_virt_full));

            for n in 0..nstates {
                for (i, occ) in active_occ.iter().enumerate() {
                    for (j, virt) in active_virt.iter().enumerate() {
                        XmY_active
                            .slice_mut(s![n, *occ, *virt - n_occ_full])
                            .assign(&XmY.as_ref().unwrap().slice(s![n, i, j]));
                        XpY_active
                            .slice_mut(s![n, *occ, *virt - n_occ_full])
                            .assign(&XpY.as_ref().unwrap().slice(s![n, i, j]));
                    }
                }
            }
            if r_lc.unwrap() > 0.0 {
                let grad_ex: Array1<f64> = gradients_lc_ex(
                    exc_state.unwrap(),
                    (&molecule.calculator.g0_ao).view(),
                    g1.view(),
                    (&molecule.calculator.g0_ao).view(),
                    g1_ao.view(),
                    (&molecule.calculator.g0_lr).view(),
                    g1lr.view(),
                    (&molecule.calculator.g0_lr_ao).view(),
                    g1lr_ao.view(),
                    s.view(),
                    grad_s.view(),
                    grad_h0.view(),
                    XmY_active.view(),
                    XpY_active.view(),
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
                    None,
                );
            } else {
                let grad_ex: Array1<f64> = gradients_nolc_ex(
                    exc_state.unwrap(),
                    (&molecule.calculator.g0_ao).view(),
                    g1.view(),
                    (&molecule.calculator.g0_ao).view(),
                    g1_ao.view(),
                    (&molecule.calculator.g0_lr).view(),
                    g1lr.view(),
                    (&molecule.calculator.g0_lr_ao).view(),
                    g1lr_ao.view(),
                    s.view(),
                    grad_s.view(),
                    grad_h0.view(),
                    XmY_active.view(),
                    XpY_active.view(),
                    omega.as_ref().unwrap().view(),
                    qtrans_oo.view(),
                    qtrans_vv.view(),
                    qtrans_ov.view(),
                    orbe_occ,
                    orbe_virt,
                    orbs_occ.view(),
                    orbs_virt.view(),
                    fdmdO.view(),
                    None,
                );
            }
        }
    }

    return (grad_e0, grad_vrep, grad_ex);
}

// only ground state
pub fn gradient_lc_gs(
    molecule: &Molecule,
    orbe_occ: &Array1<f64>,
    orbe_virt: &Array1<f64>,
    orbs_occ: &Array2<f64>,
    s: &Array2<f64>,
    r_lc: Option<f64>,
) -> (
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
) {
    let (g1, g1_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
        &molecule.atomic_numbers,
        molecule.n_atoms,
        molecule.calculator.n_orbs,
        molecule.distance_matrix.view(),
        molecule.directions_matrix.view(),
        &molecule.calculator.hubbard_u,
        &molecule.calculator.valorbs,
        Some(0.0),
    );

    let (g1lr, g1lr_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
        &molecule.atomic_numbers,
        molecule.n_atoms,
        molecule.calculator.n_orbs,
        molecule.distance_matrix.view(),
        molecule.directions_matrix.view(),
        &molecule.calculator.hubbard_u,
        &molecule.calculator.valorbs,
        None,
    );
    let n_at: usize = *&molecule.calculator.g0.dim().0;
    let n_orb: usize = *&molecule.calculator.g0_ao.dim().0;

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
    let d = 2.0 * orbs_occ.dot(&orbs_occ.t());
    // reference density matrix
    let d_ref: Array2<f64> = density_matrix_ref(&molecule);

    let diff_d: Array2<f64> = &d - &d_ref;
    // computing F(D-D0)

    let fdmd0: Array3<f64> = f_v(
        diff_d.view(),
        s.view(),
        grad_s.view(),
        (&molecule.calculator.g0_ao).view(),
        g1_ao.view(),
        molecule.n_atoms,
        molecule.calculator.n_orbs,
    );
    let mut flr_dmd0: Array3<f64> = Array::zeros((3 * n_at, n_orb, n_orb));
    if r_lc.unwrap_or(defaults::LONG_RANGE_RADIUS) > 0.0 {
        println!("Test lc");
        flr_dmd0 = f_lr(
            diff_d.view(),
            s.view(),
            grad_s.view(),
            (&molecule.calculator.g0_ao).view(),
            (&molecule.calculator.g0_lr_ao).view(),
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
    if r_lc.unwrap_or(defaults::LONG_RANGE_RADIUS) > 0.0 {
        println!("Test lc");
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

    return (
        grad_e0, grad_v_rep, grad_s, grad_h0, fdmd0, flr_dmd0, g1, g1_ao, g1lr, g1lr_ao,
    );
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
                    + &grad_s.slice(s![.., a, b]).map(|x| x * gsv_ab)),
            );
        }
    }
    f *= 0.25;
    return f;
}

pub fn gradients_nolc_ex(
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
    let t_ab: Array2<f64> = 0.5
        * (tensordot(&XpY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix2>()
            .unwrap()
            + tensordot(&XmY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
                .into_dimensionality::<Ix2>()
                .unwrap());
    let t_ij: Array2<f64> = 0.5
        * (tensordot(&XpY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
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
        0,
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
    let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia_transformed);

    let w_ai: Array2<f64> = w_ia.clone().reversed_axes();
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

    let XpY_ao = orbs_occ.dot(&XpY_state.dot(&orbs_virt.t()));
    let XmY_ao = orbs_occ.dot(&XmY_state.dot(&orbs_virt.t()));

    let mut gradExc: Array1<f64> = Array::zeros(3 * n_at);
    let f: Array3<f64> = f_v(XpY_ao.view(), s, grad_s, g0_ao, g1_ao, n_at, n_orb);
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
            * tensordot(&XpY_ao, &f, &[Axis(0), Axis(1)], &[Axis(1), Axis(2)])
                .into_dimensionality::<Ix1>()
                .unwrap();

    return gradExc;
}

pub fn gradients_lc_ex(
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
        * (tensordot(&XpY_state, &XpY_state, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix2>()
            .unwrap()
            + tensordot(&XmY_state, &XmY_state, &[Axis(0)], &[Axis(0)])
                .into_dimensionality::<Ix2>()
                .unwrap());
    let t_ij: Array2<f64> = 0.5
        * (tensordot(&XpY_state, &XpY_state, &[Axis(1)], &[Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap()
            + tensordot(&XmY_state, &XmY_state, &[Axis(1)], &[Axis(1)])
                .into_dimensionality::<Ix2>()
                .unwrap());

    // H^+_ij[T_ab]
    let h_pij_tab: Array2<f64> = h_plus_lr(
        g0,
        g0lr,
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
        g0,
        g0lr,
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
            g0,
            g0lr,
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
                g0lr,
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
            g0,
            g0lr,
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
            g0,
            g0lr,
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
            g0,
            g0lr,
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
                g0lr,
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
        1,
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

    let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia_transformed);
    let w_ai: Array2<f64> = w_ia.clone().reversed_axes();
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
    let grad_h: Array3<f64> = &grad_h0 + &f_dmd0 - 0.5 * &f_lrdmd0;

    // transform vectors to a0 basis
    let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
    let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
    let z_ao: Array2<f64> = orbs_occ.dot(&z_ia_transformed.dot(&orbs_virt.t()));
    // numpy hstack

    let mut orbs: Array2<f64> = Array::zeros((length, length));

    for i in 0..length {
        orbs.slice_mut(s![i, ..orbs_occ.dim().1])
            .assign(&orbs_occ.slice(s![i, ..]));
        orbs.slice_mut(s![i, orbs_occ.dim().1..])
            .assign(&orbs_virt.slice(s![i, ..]));
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
    tmp8 = tensordot(&s, &tmp8, &[Axis(1)], &[Axis(2)])
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
    tmp12 = tensordot(&s, &tmp12, &[Axis(1)], &[Axis(1)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    tmp12.swap_axes(0, 1);

    flr = flr + tmp12;

    flr = flr * 0.25;

    return flr;
}

fn h_minus(
    g0_lr_a0: ArrayView2<f64>,
    q_ps: ArrayView3<f64>,
    q_qr: ArrayView3<f64>,
    q_pr: ArrayView3<f64>,
    q_qs: ArrayView3<f64>,
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
            .into_dimensionality::<Ix2>()
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
fn gs_gradients_no_lc_routine() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    let S: Array2<f64> = array![
        [
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.3074918525690681,
            0.3074937992389065
        ],
        [
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.1987769748092704
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.3185054221819456
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            -0.3982160222204482,
            0.1327383036929333
        ],
        [
            0.3074918525690681,
            0.0000000000000000,
            0.0000000000000000,
            -0.3982160222204482,
            1.0000000000000000,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.0000000000000000
        ]
    ];
    let orbe_occ: Array1<f64> = array![
        -0.8688942612301258,
        -0.4499991998360209,
        -0.3563323833222918,
        -0.2833072445491910
    ];
    let orbe_virt: Array1<f64> = array![0.3766541361485015, 0.4290384545096518];
    let orbs_occ: Array2<f64> = array![
        [
            -8.6192454822475639e-01,
            -1.2183272343139559e-06,
            -2.9726068852089849e-01,
            2.6222307203584133e-16
        ],
        [
            2.6757514101551499e-03,
            -2.0080751179749709e-01,
            -3.6133406147264924e-01,
            8.4834397825097341e-01
        ],
        [
            4.2874248054290296e-03,
            -3.2175900344462377e-01,
            -5.7897479277210717e-01,
            -5.2944545948124977e-01
        ],
        [
            3.5735935812255637e-03,
            5.3637854372423877e-01,
            -4.8258565481599014e-01,
            3.4916084620212056e-16
        ],
        [
            -1.7925702667910837e-01,
            -3.6380704327437935e-01,
            2.3851989294050652e-01,
            -2.0731761365694774e-16
        ],
        [
            -1.7925784113431714e-01,
            3.6380666541125695e-01,
            2.3851861974976313e-01,
            -9.2582148396003538e-17
        ]
    ];

    let gradVrep_ref: Array1<f64> = array![
        0.1578504879797087,
        0.1181937590058072,
        0.1893848779393944,
        -0.2367773309532266,
        0.0000000000000000,
        0.0000000000000000,
        0.0789268429735179,
        -0.1181937590058072,
        -0.1893848779393944
    ];
    let gradE0_ref: Array1<f64> = array![
        -0.1198269660296263,
        -0.0897205271709892,
        -0.1437614915530440,
        0.1981679566666738,
        -0.0068989246413182,
        -0.0110543231055452,
        -0.0783409906370475,
        0.0966194518123075,
        0.1548158146585892
    ];

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
    ) = gradient_lc_gs(&mol, &orbe_occ, &orbe_virt, &orbs_occ, &S, Some(0.0));
    println!("gradE0 {}", gradE0);
    println!("gradE0_ref {}", gradE0_ref);
    assert!(gradE0.abs_diff_eq(&gradE0_ref, 1.0e-6));
    assert!(grad_v_rep.abs_diff_eq(&gradVrep_ref, 1.0e-5));
}
#[test]
fn gs_gradients_lc_routine() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    let S: Array2<f64> = array![
        [
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.3074918525690681,
            0.3074937992389065
        ],
        [
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.1987769748092704
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.3185054221819456
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            -0.3982160222204482,
            0.1327383036929333
        ],
        [
            0.3074918525690681,
            0.0000000000000000,
            0.0000000000000000,
            -0.3982160222204482,
            1.0000000000000000,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.0000000000000000
        ]
    ];
    let orbe_occ: Array1<f64> = array![
        -0.8274698453897348,
        -0.4866977301135286,
        -0.4293504173916549,
        -0.3805317623354842
    ];
    let orbe_virt: Array1<f64> = array![0.4597732058522500, 0.5075648555895175];
    let orbs_occ: Array2<f64> = array![
        [
            8.7633817073096332e-01,
            -7.3282333460513933e-07,
            -2.5626946551477814e-01,
            3.5545737574547093e-16
        ],
        [
            1.5609825393248001e-02,
            -1.9781346650256848e-01,
            -3.5949496391504693e-01,
            -8.4834397825097241e-01
        ],
        [
            2.5012021798970618e-02,
            -3.1696156822050980e-01,
            -5.7602795979720633e-01,
            5.2944545948125143e-01
        ],
        [
            2.0847651645094598e-02,
            5.2838144790875974e-01,
            -4.8012913249888561e-01,
            1.3290510512115002e-15
        ],
        [
            1.6641905232447368e-01,
            -3.7146604214648776e-01,
            2.5136102811675498e-01,
            -5.9695075273495377e-16
        ],
        [
            1.6641962261693885e-01,
            3.7146556016201521e-01,
            2.5135992631729770e-01,
            4.7826699854874327e-17
        ]
    ];
    let orbs_virt: Array2<f64> = array![
        [4.4638746430458731e-05, 6.5169208620107999e-01],
        [2.8325035872464788e-01, -2.9051011756322664e-01],
        [4.5385928211929927e-01, -4.6549177907242634e-01],
        [-7.5667687027917274e-01, -3.8791257751172398e-01],
        [-7.2004450606556147e-01, -7.6949321868972742e-01],
        [7.1993590540307117e-01, -7.6959837575446877e-01]
    ];
    let gradVrep_ref: Array1<f64> = array![
        0.1578504879797087,
        0.1181937590058072,
        0.1893848779393944,
        -0.2367773309532266,
        0.0000000000000000,
        0.0000000000000000,
        0.0789268429735179,
        -0.1181937590058072,
        -0.1893848779393944
    ];
    let gradE0_ref: Array1<f64> = array![
        -0.0955096709004110,
        -0.0715133858595269,
        -0.1145877241401038,
        0.1612048707194388,
        -0.0067164109317917,
        -0.0107618767285816,
        -0.0656951998190278,
        0.0782297967913186,
        0.1253496008686854
    ];

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
    ) = gradient_lc_gs(&mol, &orbe_occ, &orbe_virt, &orbs_occ, &S, Some(1.0));
    println!("gradE0 {}", gradE0);
    println!("gradE0_ref {}", gradE0_ref);
    assert!(gradE0.abs_diff_eq(&gradE0_ref, 1.0e-6));
    assert!(grad_v_rep.abs_diff_eq(&gradVrep_ref, 1.0e-5));
}

#[test]
fn exc_gradient_no_lc_routine() {
    let orbs: Array2<f64> = array![
        [
            -8.6192475509337374e-01,
            -1.2183336763751098e-06,
            -2.9726029578790070e-01,
            -1.0617173035797705e-16,
            4.3204846337269176e-05,
            6.5350381550742609e-01
        ],
        [
            2.6757349898771515e-03,
            -2.0080763751606209e-01,
            -3.6133415221394610e-01,
            8.4834397825097296e-01,
            2.8113666974634488e-01,
            -2.8862829713723015e-01
        ],
        [
            4.2873984947983486e-03,
            -3.2175920488669046e-01,
            -5.7897493816920131e-01,
            -5.2944545948125077e-01,
            4.5047246429977195e-01,
            -4.6247649015464443e-01
        ],
        [
            3.5735716506930821e-03,
            5.3637887951745156e-01,
            -4.8258577602132369e-01,
            -1.0229037571655944e-16,
            -7.5102755864519533e-01,
            -3.8540254135808827e-01
        ],
        [
            -1.7925680721591991e-01,
            -3.6380671959263217e-01,
            2.3851969138617155e-01,
            2.2055761208820838e-16,
            -7.2394310468946377e-01,
            -7.7069773456574175e-01
        ],
        [
            -1.7925762167314863e-01,
            3.6380634174056053e-01,
            2.3851841819041375e-01,
            2.5181078277624383e-17,
            7.2383801990121355e-01,
            -7.7079988940061905e-01
        ]
    ];
    let active_occupied_orbs: Vec<usize> = vec![2, 3];
    let active_virtual_orbs: Vec<usize> = vec![4, 5];
    let S: Array2<f64> = array![
        [
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.3074918525690681,
            0.3074937992389065
        ],
        [
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.1987769748092704
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.3185054221819456
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            -0.3982160222204482,
            0.1327383036929333
        ],
        [
            0.3074918525690681,
            0.0000000000000000,
            0.0000000000000000,
            -0.3982160222204482,
            1.0000000000000000,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.0000000000000000
        ]
    ];
    let qtrans_ov: Array3<f64> = array![
        [
            [-2.8128180967215699e-05, -4.2556726168188097e-01],
            [-5.5841428391588377e-01, 3.9062161363123682e-05],
            [2.6230140964894622e-05, 3.7065692971610431e-01],
            [8.3798119168991755e-17, -4.1770228089087565e-16]
        ],
        [
            [1.9941574838614257e-01, 2.1276907315399221e-01],
            [2.7922766432687390e-01, 2.9822390189614878e-01],
            [-1.7348148207962488e-01, -1.8531671657043469e-01],
            [1.3911327941665211e-16, 1.4864632621237686e-16]
        ],
        [
            [-1.9938762020517536e-01, 2.1279818852788895e-01],
            [2.7918661958901014e-01, -2.9826296405751196e-01],
            [1.7345525193865990e-01, -1.8534021314566926e-01],
            [-1.2791804822463816e-16, 1.3646231144806550e-16]
        ]
    ];
    let qtrans_oo: Array3<f64> = array![
        [
            [
                8.3848164205032949e-01,
                6.1966198783812432e-07,
                1.6942360993663974e-01,
                -1.6726043633702958e-16
            ],
            [
                6.1966198783877484e-07,
                5.8696925273219525e-01,
                6.3044315520843774e-07,
                5.9439073263148753e-17
            ],
            [
                1.6942360993663974e-01,
                6.3044315520843774e-07,
                8.3509720626502215e-01,
                3.0309041988910044e-16
            ],
            [
                -1.6856147894401218e-16,
                5.9439073263148753e-17,
                3.0309041988910044e-16,
                9.9999999999999978e-01
            ]
        ],
        [
            [
                8.0758771317081243e-02,
                1.3282878666703324e-01,
                -8.4711785129572156e-02,
                7.0739481847429051e-17
            ],
            [
                1.3282878666703324e-01,
                2.0651544310932970e-01,
                -1.3057860118005465e-01,
                1.0763019545169189e-16
            ],
            [
                -8.4711785129572156e-02,
                -1.3057860118005465e-01,
                8.2451766756952941e-02,
                -6.7819462121347612e-17
            ],
            [
                7.0739481847429051e-17,
                1.0763019545169189e-16,
                -6.7819462121347612e-17,
                5.5604455852449739e-32
            ]
        ],
        [
            [
                8.0759586632588587e-02,
                -1.3282940632902107e-01,
                -8.4711824807067571e-02,
                4.7886503144290639e-17
            ],
            [
                -1.3282940632902107e-01,
                2.0651530415847527e-01,
                1.3057797073689964e-01,
                -8.1137763619098240e-17
            ],
            [
                -8.4711824807067571e-02,
                1.3057797073689964e-01,
                8.2451026978025518e-02,
                -5.1969527145539535e-17
            ],
            [
                4.7886503144290639e-17,
                -8.1137763619098240e-17,
                -5.1969527145539535e-17,
                2.7922029245655617e-32
            ]
        ]
    ];
    let qtrans_vv: Array3<f64> = array![
        [
            [4.1303074685724855e-01, -5.9780600290271213e-06],
            [-5.9780600290271213e-06, 3.2642115209520417e-01]
        ],
        [
            [2.9352824890377344e-01, 3.1440112368021861e-01],
            [3.1440112368021861e-01, 3.3674576991286248e-01]
        ],
        [
            [2.9344100423897757e-01, -3.1439514562018966e-01],
            [-3.1439514562018966e-01, 3.3683307799193302e-01]
        ]
    ];

    let orbe: Array1<f64> = array![
        -0.8688942612301258,
        -0.4499991998360209,
        -0.3563323833222918,
        -0.2833072445491910,
        0.3766541361485015,
        0.4290384545096518
    ];
    let df: Array2<f64> = array![
        [2.0000000000000000, 2.0000000000000000],
        [2.0000000000000000, 2.0000000000000000]
    ];
    let omega0: Array2<f64> = array![
        [0.7329867988448483, 0.7853711745345131],
        [0.6599617692239994, 0.7123461449136643]
    ];
    let gamma0_lr: Array2<f64> = array![
        [0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
        [0.2692279296946004, 0.2923649998054588, 0.2429686492032624],
        [0.2692280400920803, 0.2429686492032624, 0.2923649998054588]
    ];
    let gamma0_lr_ao: Array2<f64> = array![
        [
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2692279296946004,
            0.2692280400920803
        ],
        [
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2692279296946004,
            0.2692280400920803
        ],
        [
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2692279296946004,
            0.2692280400920803
        ],
        [
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2692279296946004,
            0.2692280400920803
        ],
        [
            0.2692279296946004,
            0.2692279296946004,
            0.2692279296946004,
            0.2692279296946004,
            0.2923649998054588,
            0.2429686492032624
        ],
        [
            0.2692280400920803,
            0.2692280400920803,
            0.2692280400920803,
            0.2692280400920803,
            0.2429686492032624,
            0.2923649998054588
        ]
    ];
    let gamma1_lr: Array3<f64> = array![
        [
            [0.0000000000000000, 0.0203615522580970, -0.0067871192953180],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, -0.0000000000000000, 0.0101637809408346],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, -0.0000000000000000, 0.0162856857170278],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0203615522580970, 0.0000000000000000, -0.0225742810190271],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0084512391474741],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0135416362745717],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0067871192953180, 0.0225742810190271, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0101637809408346, -0.0084512391474741, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0162856857170278, -0.0135416362745717, 0.0000000000000000]
        ]
    ];
    let gamma1_lr_ao: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0203615522580970,
                -0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0203615522580970,
                -0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0203615522580970,
                -0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0203615522580970,
                -0.0067871192953180
            ],
            [
                0.0203615522580970,
                0.0203615522580970,
                0.0203615522580970,
                0.0203615522580970,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.0067871192953180,
                -0.0067871192953180,
                -0.0067871192953180,
                -0.0067871192953180,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0101637809408346,
                0.0101637809408346,
                0.0101637809408346,
                0.0101637809408346,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0162856857170278,
                0.0162856857170278,
                0.0162856857170278,
                0.0162856857170278,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0203615522580970,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0203615522580970,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0203615522580970,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0203615522580970,
                0.0000000000000000
            ],
            [
                -0.0203615522580970,
                -0.0203615522580970,
                -0.0203615522580970,
                -0.0203615522580970,
                0.0000000000000000,
                -0.0225742810190271
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0225742810190271,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0084512391474741
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0084512391474741,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0135416362745717
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0135416362745717,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0225742810190271
            ],
            [
                0.0067871192953180,
                0.0067871192953180,
                0.0067871192953180,
                0.0067871192953180,
                0.0225742810190271,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0084512391474741
            ],
            [
                -0.0101637809408346,
                -0.0101637809408346,
                -0.0101637809408346,
                -0.0101637809408346,
                -0.0084512391474741,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0135416362745717
            ],
            [
                -0.0162856857170278,
                -0.0162856857170278,
                -0.0162856857170278,
                -0.0162856857170278,
                -0.0135416362745717,
                0.0000000000000000
            ]
        ]
    ];
    let gamma0: Array2<f64> = array![
        [0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
        [0.3863557889890281, 0.4720158398964135, 0.3084885848056254],
        [0.3863561531176491, 0.3084885848056254, 0.4720158398964135]
    ];
    let gamma1: Array3<f64> = array![
        [
            [0.0000000000000000, 0.0671593223694436, -0.0223862512902948],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, -0.0000000000000000, 0.0335236415187203],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, -0.0000000000000000, 0.0537157867768206],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0671593223694436, 0.0000000000000000, -0.0573037072665056],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0214530568542981],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0343747807663729],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0223862512902948, 0.0573037072665056, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0335236415187203, -0.0214530568542981, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0537157867768206, -0.0343747807663729, 0.0000000000000000]
        ]
    ];
    let gamma0_AO: Array2<f64> = array![
        [
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.3863557889890281,
            0.3863561531176491
        ],
        [
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.3863557889890281,
            0.3863561531176491
        ],
        [
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.3863557889890281,
            0.3863561531176491
        ],
        [
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.3863557889890281,
            0.3863561531176491
        ],
        [
            0.3863557889890281,
            0.3863557889890281,
            0.3863557889890281,
            0.3863557889890281,
            0.4720158398964135,
            0.3084885848056254
        ],
        [
            0.3863561531176491,
            0.3863561531176491,
            0.3863561531176491,
            0.3863561531176491,
            0.3084885848056254,
            0.4720158398964135
        ]
    ];
    let gamma1_AO: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0671593223694436,
                -0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0671593223694436,
                -0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0671593223694436,
                -0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0671593223694436,
                -0.0223862512902948
            ],
            [
                0.0671593223694436,
                0.0671593223694436,
                0.0671593223694436,
                0.0671593223694436,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.0223862512902948,
                -0.0223862512902948,
                -0.0223862512902948,
                -0.0223862512902948,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0335236415187203,
                0.0335236415187203,
                0.0335236415187203,
                0.0335236415187203,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0537157867768206,
                0.0537157867768206,
                0.0537157867768206,
                0.0537157867768206,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0671593223694436,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0671593223694436,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0671593223694436,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0671593223694436,
                0.0000000000000000
            ],
            [
                -0.0671593223694436,
                -0.0671593223694436,
                -0.0671593223694436,
                -0.0671593223694436,
                0.0000000000000000,
                -0.0573037072665056
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0573037072665056,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0214530568542981
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0214530568542981,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0343747807663729
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0343747807663729,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0573037072665056
            ],
            [
                0.0223862512902948,
                0.0223862512902948,
                0.0223862512902948,
                0.0223862512902948,
                0.0573037072665056,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0214530568542981
            ],
            [
                -0.0335236415187203,
                -0.0335236415187203,
                -0.0335236415187203,
                -0.0335236415187203,
                -0.0214530568542981,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0343747807663729
            ],
            [
                -0.0537157867768206,
                -0.0537157867768206,
                -0.0537157867768206,
                -0.0537157867768206,
                -0.0343747807663729,
                0.0000000000000000
            ]
        ]
    ];
    let gradS: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.3590399304938401,
                -0.1196795358000320
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0918870323801337
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1472329381678249
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.3350012527450977,
                0.1558859507933732
            ],
            [
                0.3590399304938401,
                0.0000000000000000,
                0.0000000000000000,
                -0.3350012527450977,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.1196795358000320,
                0.0918870323801337,
                0.1472329381678249,
                0.1558859507933732,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1792213355983593
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2172441846869481,
                0.0796440422386294
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2204828389926055
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0918870323801337
            ],
            [
                0.0000000000000000,
                0.2172441846869481,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.1792213355983593,
                0.0796440422386294,
                -0.2204828389926055,
                0.0918870323801337,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2871709221530289
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2204828389926055
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2172441846869481,
                -0.1360394644282639
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1472329381678249
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.2172441846869481,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.2871709221530289,
                -0.2204828389926055,
                -0.1360394644282639,
                0.1472329381678249,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.3590399304938401,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.3350012527450977,
                0.0000000000000000
            ],
            [
                -0.3590399304938401,
                0.0000000000000000,
                0.0000000000000000,
                0.3350012527450977,
                0.0000000000000000,
                -0.0493263812570877
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0493263812570877,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2172441846869481,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                -0.2172441846869481,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0184665480123938
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0184665480123938,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2172441846869481,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                -0.2172441846869481,
                0.0000000000000000,
                0.0000000000000000,
                0.0295894213933693
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0295894213933693,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1196795358000320
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0918870323801337
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1472329381678249
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1558859507933732
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0493263812570877
            ],
            [
                0.1196795358000320,
                -0.0918870323801337,
                -0.1472329381678249,
                -0.1558859507933732,
                0.0493263812570877,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1792213355983593
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0796440422386294
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2204828389926055
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0918870323801337
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0184665480123938
            ],
            [
                -0.1792213355983593,
                -0.0796440422386294,
                0.2204828389926055,
                -0.0918870323801337,
                -0.0184665480123938,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2871709221530289
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2204828389926055
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1360394644282639
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1472329381678249
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0295894213933693
            ],
            [
                -0.2871709221530289,
                0.2204828389926055,
                0.1360394644282639,
                -0.1472329381678249,
                -0.0295894213933693,
                0.0000000000000000
            ]
        ]
    ];
    let gradH0: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.4466562325187367,
                0.1488849546574482
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0859724600043996
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1377558678312508
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.3151837170301471,
                -0.1441050567093689
            ],
            [
                -0.4466562325187367,
                0.0000000000000000,
                0.0000000000000000,
                0.3151837170301471,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.1488849546574482,
                -0.0859724600043996,
                -0.1377558678312508,
                -0.1441050567093689,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2229567506745117
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2015137919014903,
                -0.0727706772643434
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2062908287050795
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0859724600043996
            ],
            [
                0.0000000000000000,
                -0.2015137919014903,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.2229567506745117,
                -0.0727706772643434,
                0.2062908287050795,
                -0.0859724600043996,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.3572492944418646
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2062908287050795
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2015137919014903,
                0.1290297419048926
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1377558678312508
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                -0.2015137919014903,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.3572492944418646,
                0.2062908287050795,
                0.1290297419048926,
                -0.1377558678312508,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.4466562325187367,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.3151837170301471,
                0.0000000000000000
            ],
            [
                0.4466562325187367,
                0.0000000000000000,
                0.0000000000000000,
                -0.3151837170301471,
                0.0000000000000000,
                0.0738575792762484
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0738575792762484,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2015137919014903,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.2015137919014903,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0276504073281890
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0276504073281890,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2015137919014903,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.2015137919014903,
                0.0000000000000000,
                0.0000000000000000,
                -0.0443049536699000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0443049536699000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1488849546574482
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0859724600043996
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1377558678312508
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1441050567093689
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0738575792762484
            ],
            [
                -0.1488849546574482,
                0.0859724600043996,
                0.1377558678312508,
                0.1441050567093689,
                -0.0738575792762484,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2229567506745117
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0727706772643434
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2062908287050795
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0859724600043996
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0276504073281890
            ],
            [
                0.2229567506745117,
                0.0727706772643434,
                -0.2062908287050795,
                0.0859724600043996,
                0.0276504073281890,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.3572492944418646
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2062908287050795
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1290297419048926
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1377558678312508
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0443049536699000
            ],
            [
                0.3572492944418646,
                -0.2062908287050795,
                -0.1290297419048926,
                0.1377558678312508,
                0.0443049536699000,
                0.0000000000000000
            ]
        ]
    ];
    let orbe_occ: Array1<f64> = array![
        -0.8688942612301258,
        -0.4499991998360209,
        -0.3563323833222918,
        -0.2833072445491910
    ];
    let orbe_virt: Array1<f64> = array![0.3766541361485015, 0.4290384545096518];
    let orbs_occ: Array2<f64> = array![
        [
            -8.6192454822475639e-01,
            -1.2183272343139559e-06,
            -2.9726068852089849e-01,
            2.6222307203584133e-16
        ],
        [
            2.6757514101551499e-03,
            -2.0080751179749709e-01,
            -3.6133406147264924e-01,
            8.4834397825097341e-01
        ],
        [
            4.2874248054290296e-03,
            -3.2175900344462377e-01,
            -5.7897479277210717e-01,
            -5.2944545948124977e-01
        ],
        [
            3.5735935812255637e-03,
            5.3637854372423877e-01,
            -4.8258565481599014e-01,
            3.4916084620212056e-16
        ],
        [
            -1.7925702667910837e-01,
            -3.6380704327437935e-01,
            2.3851989294050652e-01,
            -2.0731761365694774e-16
        ],
        [
            -1.7925784113431714e-01,
            3.6380666541125695e-01,
            2.3851861974976313e-01,
            -9.2582148396003538e-17
        ]
    ];
    let orbs_virt: Array2<f64> = array![
        [4.3204927822809713e-05, 6.5350390970909367e-01],
        [2.8113675949215844e-01, -2.8862841063399913e-01],
        [4.5047260810178097e-01, -4.6247667201341525e-01],
        [-7.5102779853473878e-01, -3.8540269278994982e-01],
        [-7.2394294209812204e-01, -7.7069762107665973e-01],
        [7.2383785715168458e-01, -7.7079977605735461e-01]
    ];
    let nocc: usize = 4;
    let nvirt: usize = 2;
    let gradVrep: Array1<f64> = array![
        0.1578504879797087,
        0.1181937590058072,
        0.1893848779393944,
        -0.2367773309532266,
        0.0000000000000000,
        0.0000000000000000,
        0.0789268429735179,
        -0.1181937590058072,
        -0.1893848779393944
    ];
    let gradE0: Array1<f64> = array![
        -0.1198269660296263,
        -0.0897205271709892,
        -0.1437614915530440,
        0.1981679566666738,
        -0.0068989246413182,
        -0.0110543231055452,
        -0.0783409906370475,
        0.0966194518123075,
        0.1548158146585892
    ];
    let gradExc: Array1<f64> = array![
        0.3607539392221090,
        0.2702596932404471,
        0.4330440071185614,
        -0.7696026181183455,
        0.0854981908865757,
        0.1369959343140749,
        0.4088486788962364,
        -0.3557578841270227,
        -0.5700399414326363
    ];
    let omega: Array1<f64> = array![
        0.6599613806976925,
        0.7123456990588429,
        0.7456810724193919,
        0.7930925652350215,
        0.8714866033195531,
        0.9348736014087142,
        1.2756452171931041,
        1.3231856682450711
    ];
    let XmY: Array3<f64> = array![
        [
            [1.2521414892619180e-17, -3.0731988457045628e-18],
            [-2.0314998941993035e-17, 6.1984001001408129e-17],
            [-1.1949222929340009e-16, 1.8444992477011119e-17],
            [-9.9999999999999978e-01, -8.5678264450787104e-18]
        ],
        [
            [-5.6554497734654670e-33, -1.6132260882333921e-17],
            [-1.4165350977779408e-16, -1.2680008989475773e-17],
            [1.8003860860678714e-17, 1.5123402272838473e-16],
            [-7.1090761021361151e-18, 9.9999999999999967e-01]
        ],
        [
            [2.1571381149267578e-02, -3.0272950933503227e-07],
            [2.9991274783090719e-05, 1.4821884853203227e-01],
            [9.9507889372419056e-01, 3.2471746769845221e-06],
            [-1.1917605251843160e-16, -4.4289966266035466e-17]
        ],
        [
            [1.1109010931003951e-06, -1.2585514282188778e-02],
            [-3.2069605724216654e-01, 2.8289343609923937e-05],
            [9.1191885302142574e-06, -9.4937779050842874e-01],
            [-1.2601161265688548e-17, 1.0521007298293138e-16]
        ],
        [
            [9.1609974782184502e-06, 6.0701873653873452e-02],
            [-9.6788599240172246e-01, 1.0919490802280248e-05],
            [2.9445559547715203e-05, 3.4280398643331461e-01],
            [3.1783829909643794e-17, -2.1447832161459006e-16]
        ],
        [
            [-1.0110298012913056e-01, 9.9804621433036054e-06],
            [1.5702586864624103e-05, 1.0113993758544988e+00],
            [-1.7694318737762263e-01, 2.6156157292985052e-05],
            [1.0459553569289232e-16, 5.5045412199529119e-18]
        ],
        [
            [1.0046988096296818e+00, 9.6514695327939874e-06],
            [1.5280031987729607e-05, 1.3343212650518221e-01],
            [-6.0845305188051618e-02, 1.2489695961642065e-07],
            [4.2508419260228061e-17, 2.0822376030340786e-18]
        ],
        [
            [9.3758930989162589e-06, -1.0067757866575049e+00],
            [-8.1915557416596632e-02, 1.5848191461915825e-05],
            [-1.1132783884542197e-06, 5.1182023937152175e-02],
            [9.2945083674210509e-18, -5.1592764951007866e-17]
        ]
    ];
    let XpY: Array3<f64> = array![
        [
            [2.3631728626191835e-17, -6.0439980808443620e-18],
            [-2.5446127779063410e-17, 8.2559786740300318e-17],
            [-1.3271411906099627e-16, 2.1950010553683349e-17],
            [-9.9999999999999989e-01, -7.9377675442458068e-18]
        ],
        [
            [0.0000000000000000e+00, -2.9394105266280956e-17],
            [-1.6438371325218934e-16, -1.0269716205837895e-17],
            [1.7497024539465256e-17, 1.6673760312079509e-16],
            [-7.6733577656998602e-18, 9.9999999999999989e-01]
        ],
        [
            [3.6031757025210380e-02, -5.2693108210990112e-07],
            [3.3247977274044169e-05, 1.7472610444661485e-01],
            [9.7813856605377092e-01, 3.4200094262767256e-06],
            [-1.0544215582531297e-16, -4.5125435583086571e-17]
        ],
        [
            [1.7446652973543630e-06, -2.0596777031218809e-02],
            [-3.3426673906312654e-01, 3.1354975875389786e-05],
            [8.4280732845994917e-06, -9.4013443503872962e-01],
            [-1.0485887863620174e-17, 9.4498355264689231e-17]
        ],
        [
            [1.3093105142292459e-05, 9.0405231040797618e-02],
            [-9.1809349842431853e-01, 1.1014103424022125e-05],
            [2.4765955236613057e-05, 3.0892988258425691e-01],
            [2.4069331860443143e-17, -1.7531274647429604e-16]
        ],
        [
            [-1.3470126301596952e-01, 1.3856384771384507e-05],
            [1.3884867211514623e-05, 9.5099287606168215e-01],
            [-1.3873209262146027e-01, 2.1973326807706339e-05],
            [7.3838432359937210e-17, 4.0866099935227155e-18]
        ],
        [
            [9.8099453932498049e-01, 9.8200956604003138e-06],
            [9.9018827855576371e-06, 9.1947088357024670e-02],
            [-3.4961749454179805e-02, 7.6894757644272947e-08],
            [2.1531794612538988e-17, 1.1603416683232019e-18]
        ],
        [
            [8.8257671645045328e-06, -9.8756150574886925e-01],
            [-5.1176316697317654e-02, 1.0528497536484679e-05],
            [-6.1670714154333164e-07, 3.0378857620766524e-02],
            [4.6357942979609401e-18, -2.7775304023765253e-17]
        ]
    ];
    let FDmD0: Array3<f64> = array![
        [
            [
                1.2859460550231747e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.5169274310242961e-02,
                3.1687451887156513e-02
            ],
            [
                0.0000000000000000e+00,
                1.2859460550231747e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0270307973454305e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.2859460550231747e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.2479632035038002e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.2859460550231747e-01,
                -6.9533131791531380e-02,
                1.6734579981076435e-02
            ],
            [
                5.5169274310242961e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -6.9533131791531380e-02,
                1.9585661921955511e-01,
                3.8169975457847086e-03
            ],
            [
                3.1687451887156513e-02,
                -2.0270307973454305e-02,
                -3.2479632035038002e-02,
                1.6734579981076435e-02,
                3.8169975457847086e-03,
                8.8967693316386487e-02
            ]
        ],
        [
            [
                9.6284803364408963e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.8121212967854570e-02,
                3.6912553323229229e-02
            ],
            [
                0.0000000000000000e+00,
                9.6284803364408963e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.1985317138097402e-03,
                -2.0983424814785771e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.6284803364408963e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.8747434343718246e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.6284803364408963e-02,
                -3.6418257831913756e-02,
                1.6148135476012040e-02
            ],
            [
                2.8121212967854570e-02,
                3.1985317138097402e-03,
                0.0000000000000000e+00,
                -3.6418257831913756e-02,
                8.6622241052568244e-02,
                2.8579623716003246e-03
            ],
            [
                3.6912553323229229e-02,
                -2.0983424814785771e-02,
                -3.8747434343718246e-02,
                1.6148135476012040e-02,
                2.8579623716003246e-03,
                1.2663887792394501e-01
            ]
        ],
        [
            [
                1.5427959890582069e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                4.5059337567588315e-02,
                5.9145926691507270e-02
            ],
            [
                0.0000000000000000e+00,
                1.5427959890582069e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.8747434343718211e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5427959890582069e-01,
                0.0000000000000000e+00,
                3.1985317138097402e-03,
                -5.8887429279635008e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5427959890582069e-01,
                -5.8353904404745277e-02,
                2.5874569789451261e-02
            ],
            [
                4.5059337567588315e-02,
                0.0000000000000000e+00,
                3.1985317138097402e-03,
                -5.8353904404745277e-02,
                1.3879702859582799e-01,
                4.5793860814115897e-03
            ],
            [
                5.9145926691507270e-02,
                -3.8747434343718211e-02,
                -5.8887429279635008e-02,
                2.5874569789451261e-02,
                4.5793860814115897e-03,
                2.0291670761423028e-01
            ]
        ],
        [
            [
                -2.0274730331707744e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -7.4441477127642444e-02,
                -5.7028858578436432e-02
            ],
            [
                0.0000000000000000e+00,
                -2.0274730331707744e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6865862053497241e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0274730331707744e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9071112077830515e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0274730331707744e-01,
                9.4491516481241836e-02,
                -2.4618102764941988e-02
            ],
            [
                -7.4441477127642444e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.4491516481241836e-02,
                -2.4705490326907464e-01,
                -5.4645067468726107e-03
            ],
            [
                -5.7028858578436432e-02,
                3.6865862053497241e-02,
                5.9071112077830515e-02,
                -2.4618102764941988e-02,
                -5.4645067468726107e-03,
                -1.6817958184022622e-01
            ]
        ],
        [
            [
                3.6898758456716607e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.9627859141285529e-04,
                1.4760992744354974e-04
            ],
            [
                0.0000000000000000e+00,
                3.6898758456716607e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.1985317138097402e-03,
                -9.5421289475329930e-05
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6898758456716607e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.5289596855293426e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6898758456716607e-03,
                -3.8369433581985279e-04,
                6.3719956062823315e-05
            ],
            [
                2.9627859141285529e-04,
                -3.1985317138097402e-03,
                0.0000000000000000e+00,
                -3.8369433581985279e-04,
                -1.7628095579799831e-03,
                -9.7699346036441648e-05
            ],
            [
                1.4760992744354974e-04,
                -9.5421289475329930e-05,
                -1.5289596855293426e-04,
                6.3719956062823315e-05,
                -9.7699346036441648e-05,
                -2.7297919167670365e-03
            ]
        ],
        [
            [
                5.9123822824664772e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                4.7473475201023347e-04,
                2.3651915572095363e-04
            ],
            [
                0.0000000000000000e+00,
                5.9123822824664772e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.5289596855293350e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9123822824664772e-03,
                0.0000000000000000e+00,
                -3.1985317138097402e-03,
                -2.4498911436093775e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9123822824664772e-03,
                -6.1480323129167410e-04,
                1.0210011258435735e-04
            ],
            [
                4.7473475201023347e-04,
                0.0000000000000000e+00,
                -3.1985317138097402e-03,
                -6.1480323129167410e-04,
                -2.8245947652112049e-03,
                -1.5654615674725339e-04
            ],
            [
                2.3651915572095363e-04,
                -1.5289596855293350e-04,
                -2.4498911436093775e-04,
                1.0210011258435735e-04,
                -1.5654615674725339e-04,
                -4.3740152890092152e-03
            ]
        ],
        [
            [
                7.4152697814759902e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.9272202817399473e-02,
                2.5341406691279916e-02
            ],
            [
                0.0000000000000000e+00,
                7.4152697814759902e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6595554080042929e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.4152697814759902e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.6591480042792510e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.4152697814759902e-02,
                -2.4958384689710456e-02,
                7.8835227838655493e-03
            ],
            [
                1.9272202817399473e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.4958384689710456e-02,
                5.1198284049519552e-02,
                1.6475092010879019e-03
            ],
            [
                2.5341406691279916e-02,
                -1.6595554080042929e-02,
                -2.6591480042792510e-02,
                7.8835227838655493e-03,
                1.6475092010879019e-03,
                7.9211888523839730e-02
            ]
        ],
        [
            [
                -9.9974679210080616e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.8417491559267427e-02,
                -3.7060163250672774e-02
            ],
            [
                0.0000000000000000e+00,
                -9.9974679210080616e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.1078846104261098e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -9.9974679210080616e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8900330312271178e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -9.9974679210080616e-02,
                3.6801952167733604e-02,
                -1.6211855432074859e-02
            ],
            [
                -2.8417491559267427e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6801952167733604e-02,
                -8.4859431494588272e-02,
                -2.7602630255638833e-03
            ],
            [
                -3.7060163250672774e-02,
                2.1078846104261098e-02,
                3.8900330312271178e-02,
                -1.6211855432074859e-02,
                -2.7602630255638833e-03,
                -1.2390908600717797e-01
            ]
        ],
        [
            [
                -1.6019198118828715e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -4.5534072319598551e-02,
                -5.9382445847228217e-02
            ],
            [
                0.0000000000000000e+00,
                -1.6019198118828715e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8900330312271136e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6019198118828715e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9132418393995939e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6019198118828715e-01,
                5.8968707636036941e-02,
                -2.5976669902035621e-02
            ],
            [
                -4.5534072319598551e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.8968707636036941e-02,
                -1.3597243383061677e-01,
                -4.4228399246643360e-03
            ],
            [
                -5.9382445847228217e-02,
                3.8900330312271136e-02,
                5.9132418393995939e-02,
                -2.5976669902035621e-02,
                -4.4228399246643360e-03,
                -1.9854269232522104e-01
            ]
        ]
    ];
    let f_matrix_ref: Array3<f64> = array![
        [
            [
                -1.3095118557477233e-18,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.5682785358343564e-17,
                5.4177805777971490e-18
            ],
            [
                0.0000000000000000e+00,
                -1.3095118557477233e-18,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -4.0060849244565283e-18
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.3095118557477233e-18,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -6.4190521632852976e-18
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.3095118557477233e-18,
                1.4898757886224559e-17,
                -4.8620648676854233e-18
            ],
            [
                -1.5682785358343564e-17,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.4898757886224559e-17,
                -3.4694469519536142e-18,
                8.1366029360785854e-20
            ],
            [
                5.4177805777971490e-18,
                -4.0060849244565283e-18,
                -6.4190521632852976e-18,
                -4.8620648676854233e-18,
                8.1366029360785854e-20,
                9.5409791178724390e-18
            ]
        ],
        [
            [
                -1.1834244701821207e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.5155098804056893e-02,
                -3.5155579415381805e-02
            ],
            [
                0.0000000000000000e+00,
                -1.1834244701821207e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -9.0446060927013613e-18,
                2.2726050870467304e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.1834244701821207e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6414531582301686e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.1834244701821207e-01,
                4.5527461913397772e-02,
                -1.5175889687825432e-02
            ],
            [
                -3.5155098804056893e-02,
                -9.0446060927013613e-18,
                0.0000000000000000e+00,
                4.5527461913397772e-02,
                -1.1031466054635622e-01,
                -2.9567278725390581e-03
            ],
            [
                -3.5155579415381805e-02,
                2.2726050870467304e-02,
                3.6414531582301686e-02,
                -1.5175889687825432e-02,
                -2.9567278725390581e-03,
                -1.1031633896314529e-01
            ]
        ],
        [
            [
                7.3856681775322036e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.1940047806782801e-02,
                2.1940347752901667e-02
            ],
            [
                0.0000000000000000e+00,
                7.3856681775322036e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.4183167151272260e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.3856681775322036e-02,
                0.0000000000000000e+00,
                -9.0446060927013613e-18,
                -2.2726050870467297e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.3856681775322036e-02,
                -2.8413366051645416e-02,
                9.4711651108466931e-03
            ],
            [
                2.1940047806782801e-02,
                0.0000000000000000e+00,
                -9.0446060927013613e-18,
                -2.8413366051645416e-02,
                6.8846597179717473e-02,
                1.8452728930367307e-03
            ],
            [
                2.1940347752901667e-02,
                -1.4183167151272260e-02,
                -2.2726050870467297e-02,
                9.4711651108466931e-03,
                1.8452728930367307e-03,
                6.8847644667730454e-02
            ]
        ],
        [
            [
                1.3201164616469479e-18,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5542177658181059e-17,
                3.7894503577446934e-19
            ],
            [
                0.0000000000000000e+00,
                1.3201164616469479e-18,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.4496607091486667e-19
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.3201164616469479e-18,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.9251539023499230e-19
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.3201164616469479e-18,
                -1.4716664479246559e-17,
                1.6358222951507426e-19
            ],
            [
                1.5542177658181059e-17,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.4716664479246559e-17,
                2.5442964874390038e-18,
                1.7607886195210724e-18
            ],
            [
                3.7894503577446934e-19,
                -2.4496607091486667e-19,
                -3.9251539023499230e-19,
                1.6358222951507426e-19,
                1.7607886195210724e-18,
                1.1446163993890481e-18
            ]
        ],
        [
            [
                5.9167066848125033e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.8469145584213149e-02,
                1.6683721986871618e-02
            ],
            [
                0.0000000000000000e+00,
                5.9167066848125033e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.0446060927013613e-18,
                -1.0785062311232586e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9167066848125033e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.7281180720220526e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9167066848125033e-02,
                -2.3918388818785757e-02,
                7.2019954916269068e-03
            ],
            [
                1.8469145584213149e-02,
                9.0446060927013613e-18,
                0.0000000000000000e+00,
                -2.3918388818785757e-02,
                6.0960640145767783e-02,
                1.4782604072106087e-03
            ],
            [
                1.6683721986871618e-02,
                -1.0785062311232586e-02,
                -1.7281180720220526e-02,
                7.2019954916269068e-03,
                1.4782604072106087e-03,
                4.9347134271808736e-02
            ]
        ],
        [
            [
                -3.6925746745026120e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.1526462756557687e-02,
                -1.0412192553553429e-02
            ],
            [
                0.0000000000000000e+00,
                -3.6925746745026120e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.7308808894676652e-03
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.6925746745026120e-02,
                0.0000000000000000e+00,
                9.0446060927013613e-18,
                1.0785062311232569e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.6925746745026120e-02,
                1.4927296807506617e-02,
                -4.4947143022193358e-03
            ],
            [
                -1.1526462756557687e-02,
                0.0000000000000000e+00,
                9.0446060927013613e-18,
                1.4927296807506617e-02,
                -3.8045103118182155e-02,
                -9.2257183476702977e-04
            ],
            [
                -1.0412192553553429e-02,
                6.7308808894676652e-03,
                1.0785062311232569e-02,
                -4.4947143022193358e-03,
                -9.2257183476702977e-04,
                -3.0797196477407474e-02
            ]
        ],
        [
            [
                8.7371839357182344e-19,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                4.7818651247407591e-19,
                -5.2180741153526060e-18
            ],
            [
                0.0000000000000000e+00,
                8.7371839357182344e-19,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8769862351403739e-18
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                8.7371839357182344e-19,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.2121940370700683e-18
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                8.7371839357182344e-19,
                -6.1927341907091090e-19,
                4.9482737510093077e-18
            ],
            [
                4.7818651247407591e-19,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -6.1927341907091090e-19,
                2.2365201931020580e-18,
                -1.7859939646658028e-18
            ],
            [
                -5.2180741153526060e-18,
                3.8769862351403739e-18,
                6.2121940370700683e-18,
                4.9482737510093077e-18,
                -1.7859939646658028e-18,
                -7.8062556418956319e-18
            ]
        ],
        [
            [
                5.9175380170087041e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.6685953219843737e-02,
                1.8471857428510183e-02
            ],
            [
                0.0000000000000000e+00,
                5.9175380170087041e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.1940988559234721e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9175380170087041e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.9133350862081160e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9175380170087041e-02,
                -2.1609073094612016e-02,
                7.9738941961985266e-03
            ],
            [
                1.6685953219843737e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.1609073094612016e-02,
                4.9354020400588429e-02,
                1.4784674653284501e-03
            ],
            [
                1.8471857428510183e-02,
                -1.1940988559234721e-02,
                -1.9133350862081160e-02,
                7.9738941961985266e-03,
                1.4784674653284501e-03,
                6.0969204691336563e-02
            ]
        ],
        [
            [
                -3.6930935030295917e-02,
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                -1.0413585050225112e-02,
                -1.1528155199348239e-02
            ],
            [
                -0.0000000000000000e+00,
                -3.6930935030295917e-02,
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                7.4522862618045978e-03
            ],
            [
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                -3.6930935030295917e-02,
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                1.1940988559234729e-02
            ],
            [
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                -3.6930935030295917e-02,
                1.3486069244138798e-02,
                -4.9764508086273581e-03
            ],
            [
                -1.0413585050225112e-02,
                -0.0000000000000000e+00,
                -0.0000000000000000e+00,
                1.3486069244138798e-02,
                -3.0801494061535310e-02,
                -9.2270105826970053e-04
            ],
            [
                -1.1528155199348239e-02,
                7.4522862618045978e-03,
                1.1940988559234729e-02,
                -4.9764508086273581e-03,
                -9.2270105826970053e-04,
                -3.8050448190322973e-02
            ]
        ]
    ];
    let Tvv_ref: Array2<f64> = array![
        [
            0.4270673600050710,
            -0.1886197948024401,
            -0.3022303133100169,
            -0.2518621665506448,
            -0.5036539085770945,
            -0.5037206672563749
        ],
        [
            -0.1886197948024401,
            0.0833063594251084,
            0.1334839067985333,
            0.1112381666740266,
            0.2224452294507603,
            0.2224747142804766
        ],
        [
            -0.3022303133100169,
            0.1334839067985333,
            0.2138846721566039,
            0.1782397547465046,
            0.3564296709241896,
            0.3564769152196909
        ],
        [
            -0.2518621665506448,
            0.1112381666740266,
            0.1782397547465046,
            0.1485352356097444,
            0.2970289384897529,
            0.2970683092943947
        ],
        [
            -0.5036539085770945,
            0.2224452294507603,
            0.3564296709241896,
            0.2970289384897529,
            0.5939748231332224,
            0.5940535537338250
        ],
        [
            -0.5037206672563749,
            0.2224747142804766,
            0.3564769152196909,
            0.2970683092943947,
            0.5940535537338250,
            0.5941322947700678
        ]
    ];
    let Too_ref: Array2<f64> = array![
        [
            5.5068664451933902e-32,
            1.9900707552507591e-16,
            -1.2419890426831438e-16,
            6.2913248483833764e-32,
            -3.8107890252274779e-32,
            -1.2561714277425054e-32
        ],
        [
            1.9900707552507591e-16,
            7.1968750543468762e-01,
            -4.4915186736323787e-01,
            2.2582946996334598e-16,
            -1.3670319910018115e-16,
            -4.6451264710797351e-17
        ],
        [
            -1.2419890426831438e-16,
            -4.4915186736323781e-01,
            2.8031249456531165e-01,
            -1.4093857038468774e-16,
            8.5315496915969487e-17,
            2.8989904824924182e-17
        ],
        [
            6.2913248483833775e-32,
            2.2582946996334598e-16,
            -1.4093857038468779e-16,
            7.9124012099327148e-32,
            -4.8358033936060716e-32,
            -9.8994324462343965e-33
        ],
        [
            -3.8107890252274779e-32,
            -1.3670319910018115e-16,
            8.5315496915969487e-17,
            -4.8358033936060705e-32,
            2.9578233119356636e-32,
            5.7341294779009836e-33
        ],
        [
            -1.2561714277425054e-32,
            -4.6451264710797351e-17,
            2.8989904824924188e-17,
            -9.8994324462343951e-33,
            5.7341294779009849e-33,
            5.6734148941043009e-33
        ]
    ];
    let Zao_ref: Array2<f64> = array![
        [
            0.0136791622502726,
            -0.0060420535380897,
            -0.0096813366580395,
            -0.0080659898650065,
            -0.0161310400107494,
            -0.0161356269318391
        ],
        [
            -0.0164776282846432,
            0.0128875509833642,
            0.0206500520031876,
            -0.0052673447279338,
            0.0049885293993804,
            0.0338791489240662
        ],
        [
            -0.0264025245297810,
            0.0206500520031876,
            0.0330881055900226,
            -0.0084400009506040,
            0.0079932480305691,
            0.0542854253697066
        ],
        [
            -0.0219987790507901,
            -0.0052665568968952,
            -0.0084387385888147,
            0.0529941334628792,
            0.0645195533406180,
            -0.0126282476892237
        ],
        [
            0.0164656628243572,
            0.0028898482374026,
            0.0046304776183400,
            -0.0368548836543085,
            -0.0455828838865788,
            0.0067432519571227
        ],
        [
            0.0164730347535184,
            -0.0174387985058098,
            -0.0279426321170664,
            0.0174324226766067,
            0.0067402091362492,
            -0.0455970007758543
        ]
    ];
    let gradH_ref: Array3<f64> = array![
        [
            [
                1.2859460550231747e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.9148695820849377e-01,
                1.8057240654460474e-01
            ],
            [
                0.0000000000000000e+00,
                1.2859460550231747e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.0624276797785391e-01
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.2859460550231747e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.7023549986628886e-01
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.2859460550231747e-01,
                2.4565058523861577e-01,
                -1.2737047672829249e-01
            ],
            [
                -3.9148695820849377e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.4565058523861577e-01,
                1.9585661921955511e-01,
                3.8169975457847086e-03
            ],
            [
                1.8057240654460474e-01,
                -1.0624276797785391e-01,
                -1.7023549986628886e-01,
                -1.2737047672829249e-01,
                3.8169975457847086e-03,
                8.8967693316386487e-02
            ]
        ],
        [
            [
                9.6284803364408963e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.8121212967854570e-02,
                -1.8604419735128244e-01
            ],
            [
                0.0000000000000000e+00,
                9.6284803364408963e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.9831526018768053e-01,
                -9.3754102079129209e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.6284803364408963e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.6754339436136123e-01
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.6284803364408963e-02,
                -3.6418257831913756e-02,
                -6.9824324528387563e-02
            ],
            [
                2.8121212967854570e-02,
                -1.9831526018768053e-01,
                0.0000000000000000e+00,
                -3.6418257831913756e-02,
                8.6622241052568244e-02,
                2.8579623716003246e-03
            ],
            [
                -1.8604419735128244e-01,
                -9.3754102079129209e-02,
                1.6754339436136123e-01,
                -6.9824324528387563e-02,
                2.8579623716003246e-03,
                1.2663887792394501e-01
            ]
        ],
        [
            [
                1.5427959890582069e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                4.5059337567588315e-02,
                -2.9810336775035734e-01
            ],
            [
                0.0000000000000000e+00,
                1.5427959890582069e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.6754339436136126e-01
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5427959890582069e-01,
                0.0000000000000000e+00,
                -1.9831526018768053e-01,
                7.0142312625257608e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5427959890582069e-01,
                -5.8353904404745277e-02,
                -1.1188129804179958e-01
            ],
            [
                4.5059337567588315e-02,
                0.0000000000000000e+00,
                -1.9831526018768053e-01,
                -5.8353904404745277e-02,
                1.3879702859582799e-01,
                4.5793860814115897e-03
            ],
            [
                -2.9810336775035734e-01,
                1.6754339436136126e-01,
                7.0142312625257608e-02,
                -1.1188129804179958e-01,
                4.5793860814115897e-03,
                2.0291670761423028e-01
            ]
        ],
        [
            [
                -2.0274730331707744e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.7221475539109428e-01,
                -5.7028858578436432e-02
            ],
            [
                0.0000000000000000e+00,
                -2.0274730331707744e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6865862053497241e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0274730331707744e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9071112077830515e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0274730331707744e-01,
                -2.2069220054890532e-01,
                -2.4618102764941988e-02
            ],
            [
                3.7221475539109428e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.2069220054890532e-01,
                -2.4705490326907464e-01,
                6.8393072529375737e-02
            ],
            [
                -5.7028858578436432e-02,
                3.6865862053497241e-02,
                5.9071112077830515e-02,
                -2.4618102764941988e-02,
                6.8393072529375737e-02,
                -1.6817958184022622e-01
            ]
        ],
        [
            [
                3.6898758456716607e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.9627859141285529e-04,
                1.4760992744354974e-04
            ],
            [
                0.0000000000000000e+00,
                3.6898758456716607e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.9831526018768053e-01,
                -9.5421289475329930e-05
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6898758456716607e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.5289596855293426e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6898758456716607e-03,
                -3.8369433581985279e-04,
                6.3719956062823315e-05
            ],
            [
                2.9627859141285529e-04,
                1.9831526018768053e-01,
                0.0000000000000000e+00,
                -3.8369433581985279e-04,
                -1.7628095579799831e-03,
                -2.7748106674225444e-02
            ],
            [
                1.4760992744354974e-04,
                -9.5421289475329930e-05,
                -1.5289596855293426e-04,
                6.3719956062823315e-05,
                -2.7748106674225444e-02,
                -2.7297919167670365e-03
            ]
        ],
        [
            [
                5.9123822824664772e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                4.7473475201023347e-04,
                2.3651915572095363e-04
            ],
            [
                0.0000000000000000e+00,
                5.9123822824664772e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.5289596855293350e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9123822824664772e-03,
                0.0000000000000000e+00,
                1.9831526018768053e-01,
                -2.4498911436093775e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9123822824664772e-03,
                -6.1480323129167410e-04,
                1.0210011258435735e-04
            ],
            [
                4.7473475201023347e-04,
                0.0000000000000000e+00,
                1.9831526018768053e-01,
                -6.1480323129167410e-04,
                -2.8245947652112049e-03,
                -4.4461499826647245e-02
            ],
            [
                2.3651915572095363e-04,
                -1.5289596855293350e-04,
                -2.4498911436093775e-04,
                1.0210011258435735e-04,
                -4.4461499826647245e-02,
                -4.3740152890092152e-03
            ]
        ],
        [
            [
                7.4152697814759902e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.9272202817399473e-02,
                -1.2354354796616830e-01
            ],
            [
                0.0000000000000000e+00,
                7.4152697814759902e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.9376905924356677e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.4152697814759902e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.1116438778845834e-01
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.4152697814759902e-02,
                -2.4958384689710456e-02,
                1.5198857949323449e-01
            ],
            [
                1.9272202817399473e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.4958384689710456e-02,
                5.1198284049519552e-02,
                -7.2210070075160449e-02
            ],
            [
                -1.2354354796616830e-01,
                6.9376905924356677e-02,
                1.1116438778845834e-01,
                1.5198857949323449e-01,
                -7.2210070075160449e-02,
                7.9211888523839730e-02
            ]
        ],
        [
            [
                -9.9974679210080616e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.8417491559267427e-02,
                1.8589658742383891e-01
            ],
            [
                0.0000000000000000e+00,
                -9.9974679210080616e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.3849523368604540e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -9.9974679210080616e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6739049839280828e-01
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -9.9974679210080616e-02,
                3.6801952167733604e-02,
                6.9760604572324747e-02
            ],
            [
                -2.8417491559267427e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6801952167733604e-02,
                -8.4859431494588272e-02,
                2.4890144302625119e-02
            ],
            [
                1.8589658742383891e-01,
                9.3849523368604540e-02,
                -1.6739049839280828e-01,
                6.9760604572324747e-02,
                2.4890144302625119e-02,
                -1.2390908600717797e-01
            ]
        ],
        [
            [
                -1.6019198118828715e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -4.5534072319598551e-02,
                2.9786684859463636e-01
            ],
            [
                0.0000000000000000e+00,
                -1.6019198118828715e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6739049839280834e-01
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6019198118828715e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -6.9897323510896670e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6019198118828715e-01,
                5.8968707636036941e-02,
                1.1177919792921523e-01
            ],
            [
                -4.5534072319598551e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.8968707636036941e-02,
                -1.3597243383061677e-01,
                3.9882113745235655e-02
            ],
            [
                2.9786684859463636e-01,
                -1.6739049839280834e-01,
                -6.9897323510896670e-02,
                1.1177919792921523e-01,
                3.9882113745235655e-02,
                -1.9854269232522104e-01
            ]
        ]
    ];
    let Wao_ref: Array2<f64> = array![
        [
            0.1105137112438734,
            -0.0863701700313839,
            -0.1383931287623632,
            -0.1153313829033104,
            -0.1935994055737700,
            -0.1936178347717592
        ],
        [
            -0.0809759533770376,
            0.1758048098815216,
            -0.0615578460799648,
            0.0452498868247516,
            0.0970537726009901,
            0.0897220684134011
        ],
        [
            -0.1297498376846011,
            -0.0615578460799650,
            0.1155869164343443,
            0.0725050490411757,
            0.1555117379479668,
            0.1437639611251137
        ],
        [
            -0.1081314768929897,
            0.0452482187891391,
            0.0725023763051235,
            0.0167002549856182,
            0.1148916657571488,
            0.1345180115669712
        ],
        [
            -0.2309769562719880,
            0.0952296320293957,
            0.1525888709336543,
            0.1568118978706647,
            0.2678082241417885,
            0.2545435384390421
        ],
        [
            -0.2310102441154931,
            0.1100445122367659,
            0.1763271317636674,
            0.1172898179179449,
            0.2545446857376794,
            0.2678844886920843
        ]
    ];
    let Wtriangular_ref: Array2<f64> = array![
        [
            -5.7206444388124618e-02,
            -6.1999995690354792e-06,
            -2.4919066060893037e-02,
            1.1268989253182943e-17,
            -3.5113150492664241e-06,
            4.1902583027236470e-02
        ],
        [
            0.0000000000000000e+00,
            -3.8710044025305493e-02,
            5.9126167214951144e-06,
            -1.4482710199435413e-17,
            4.4709793388444857e-02,
            -9.9628540128118933e-06
        ],
        [
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            -5.6957550586124936e-02,
            4.7138675138357234e-17,
            2.0028422314261856e-06,
            -2.4734918293858252e-02
        ],
        [
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            2.1422262498722527e-01,
            -4.9311331996798658e-18,
            -2.2542868783418967e-16
        ],
        [
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            6.1280910433987415e-33,
            -6.3422325805474797e-18
        ],
        [
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            4.2903845450965178e-01
        ]
    ];
    let orbs_ref: Array2<f64> = array![
        [
            -8.6192454822475639e-01,
            -1.2183272343139559e-06,
            -2.9726068852089849e-01,
            2.6222307203584133e-16,
            4.3204927822809713e-05,
            6.5350390970909367e-01
        ],
        [
            2.6757514101551499e-03,
            -2.0080751179749709e-01,
            -3.6133406147264924e-01,
            8.4834397825097341e-01,
            2.8113675949215844e-01,
            -2.8862841063399913e-01
        ],
        [
            4.2874248054290296e-03,
            -3.2175900344462377e-01,
            -5.7897479277210717e-01,
            -5.2944545948124977e-01,
            4.5047260810178097e-01,
            -4.6247667201341525e-01
        ],
        [
            3.5735935812255637e-03,
            5.3637854372423877e-01,
            -4.8258565481599014e-01,
            3.4916084620212056e-16,
            -7.5102779853473878e-01,
            -3.8540269278994982e-01
        ],
        [
            -1.7925702667910837e-01,
            -3.6380704327437935e-01,
            2.3851989294050652e-01,
            -2.0731761365694774e-16,
            -7.2394294209812204e-01,
            -7.7069762107665973e-01
        ],
        [
            -1.7925784113431714e-01,
            3.6380666541125695e-01,
            2.3851861974976313e-01,
            -9.2582148396003538e-17,
            7.2383785715168458e-01,
            -7.7079977605735461e-01
        ]
    ];

    let gradEx_test: Array1<f64> = gradients_nolc_ex(
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

    println!("gradEx_result {}", gradEx_test);
    println!("gradEx_ref {}", gradExc);
    assert!(gradEx_test.abs_diff_eq(&gradExc, 1e-10));
}

#[test]
fn exc_gradient_lc_routine() {
    let orbs: Array2<f64> = array![
        [
            8.7633793350448586e-01,
            -7.3281651176580203e-07,
            -2.5626938751925138e-01,
            -5.1521049207637978e-16,
            4.4638775222205496e-05,
            6.5169243587295766e-01
        ],
        [
            1.5609898461625849e-02,
            -1.9781358246929345e-01,
            -3.5949502670174666e-01,
            -8.4834397825097363e-01,
            2.8325027774717504e-01,
            -2.9051003593100055e-01
        ],
        [
            2.5012138878298080e-02,
            -3.1696175403695270e-01,
            -5.7602806040194532e-01,
            5.2944545948124955e-01,
            4.5385915236702673e-01,
            -4.6549164827102618e-01
        ],
        [
            2.0847749239999245e-02,
            5.2838175770958107e-01,
            -4.8012921634369682e-01,
            -7.4787175447161805e-16,
            -7.5667665393399342e-01,
            -3.8791246851639116e-01
        ],
        [
            1.6641932892259598e-01,
            -3.7146574737227095e-01,
            2.5136085315217793e-01,
            3.8681048614854049e-16,
            -7.2004465814576490e-01,
            -7.6949321601532639e-01
        ],
        [
            1.6641989920310374e-01,
            3.7146526537931795e-01,
            2.5135975138380157e-01,
            8.0313243575659063e-16,
            7.1993605749383982e-01,
            -7.6959837308789769e-01
        ]
    ];
    let active_occupied_orbs: Vec<usize> = vec![2, 3];
    let active_virtual_orbs: Vec<usize> = vec![4, 5];
    let S: Array2<f64> = array![
        [
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.3074918525690681,
            0.3074937992389065
        ],
        [
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.1987769748092704
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.3185054221819456
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            -0.3982160222204482,
            0.1327383036929333
        ],
        [
            0.3074918525690681,
            0.0000000000000000,
            0.0000000000000000,
            -0.3982160222204482,
            1.0000000000000000,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.0000000000000000
        ]
    ];
    let qtrans_ov: Array3<f64> = array![
        [
            [2.7883898248858355e-05, 4.0491876259450360e-01],
            [-5.6013372416463048e-01, 4.0582596496696599e-05],
            [2.8628773640737570e-05, 3.9452416704423321e-01],
            [-4.5537717379370529e-16, -7.6021803873605936e-16]
        ],
        [
            [-1.8878066850507635e-01, -2.0244520931052146e-01],
            [2.8008828225551385e-01, 3.0088595462037382e-01],
            [-1.8358406140825892e-01, -1.9724907532205949e-01],
            [4.8486030440255935e-16, 5.2066920309664093e-16]
        ],
        [
            [1.8875278460682746e-01, -2.0247355328398234e-01],
            [2.8004544190911645e-01, -3.0092653721687068e-01],
            [1.8355543263461827e-01, -1.9727509172217361e-01],
            [-1.5400592101868242e-17, 1.6261706675140976e-17]
        ]
    ];
    let qtrans_oo: Array3<f64> = array![
        [
            [
                8.5619854006969454e-01,
                -4.6848521532277809e-07,
                -1.7025620326228011e-01,
                2.4288654353579825e-16
            ],
            [
                -4.6848521532277809e-07,
                5.7510242562693203e-01,
                5.2738791891937531e-07,
                -5.7325487465525487e-16
            ],
            [
                -1.7025620326228011e-01,
                5.2738791891937531e-07,
                8.1374590562337779e-01,
                1.1937126270737105e-16
            ],
            [
                2.4288654353579825e-16,
                -5.7325487465525477e-16,
                1.1937126270737100e-16,
                9.9999999999999978e-01
            ]
        ],
        [
            [
                7.1900434246837897e-02,
                -1.2783412279680445e-01,
                8.5128073353799044e-02,
                -2.1346373826798261e-16
            ],
            [
                -1.2783412279680445e-01,
                2.1244887047110758e-01,
                -1.4069188288740681e-01,
                3.5933785622400282e-16
            ],
            [
                8.5128073353799044e-02,
                -1.4069188288740681e-01,
                9.3127359519010966e-02,
                -2.3822626839433911e-16
            ],
            [
                -2.1346373826798261e-16,
                3.5933785622400282e-16,
                -2.3822626839433911e-16,
                6.0627396243002842e-31
            ]
        ],
        [
            [
                7.1901025683467984e-02,
                1.2783459128201977e-01,
                8.5128129908480982e-02,
                4.5641731925539990e-18
            ],
            [
                1.2783459128201977e-01,
                2.1244870390196036e-01,
                1.4069135549948786e-01,
                8.0286645911930116e-19
            ],
            [
                8.5128129908480982e-02,
                1.4069135549948786e-01,
                9.3126734857611818e-02,
                1.4846088480627604e-19
            ],
            [
                4.5641731925539990e-18,
                8.0286645911930116e-19,
                1.4846088480627604e-19,
                -3.3150038630800853e-33
            ]
        ]
    ];
    let qtrans_vv: Array3<f64> = array![
        [
            [4.2489757389951999e-01, -6.7121242356138477e-06],
            [-6.7121242355860922e-06, 3.3005555478047682e-01]
        ],
        [
            [2.8759575994652431e-01, 3.1037546092365087e-01],
            [3.1037546092365087e-01, 3.3492757581651916e-01]
        ],
        [
            [2.8750666615395581e-01, -3.1036874879941534e-01],
            [-3.1036874879941534e-01, 3.3501686940300440e-01]
        ]
    ];
    let orbe: Array1<f64> = array![
        -0.8274698453897348,
        -0.4866977301135286,
        -0.4293504173916549,
        -0.3805317623354842,
        0.4597732058522500,
        0.5075648555895175
    ];
    let df: Array2<f64> = array![
        [2.0000000000000000, 2.0000000000000000],
        [2.0000000000000000, 2.0000000000000000]
    ];
    let omega0: Array2<f64> = array![
        [0.8891235178682032, 0.9369152042028386],
        [0.8403048164870012, 0.8880965028216365]
    ];
    let gamma0_lr: Array2<f64> = array![
        [0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
        [0.2692279296946004, 0.2923649998054588, 0.2429686492032624],
        [0.2692280400920803, 0.2429686492032624, 0.2923649998054588]
    ];
    let gamma0_lr_ao: Array2<f64> = array![
        [
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2692279296946004,
            0.2692280400920803
        ],
        [
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2692279296946004,
            0.2692280400920803
        ],
        [
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2692279296946004,
            0.2692280400920803
        ],
        [
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2860554418243039,
            0.2692279296946004,
            0.2692280400920803
        ],
        [
            0.2692279296946004,
            0.2692279296946004,
            0.2692279296946004,
            0.2692279296946004,
            0.2923649998054588,
            0.2429686492032624
        ],
        [
            0.2692280400920803,
            0.2692280400920803,
            0.2692280400920803,
            0.2692280400920803,
            0.2429686492032624,
            0.2923649998054588
        ]
    ];
    let gamma1_lr: Array3<f64> = array![
        [
            [0.0000000000000000, 0.0203615522580970, -0.0067871192953180],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, -0.0000000000000000, 0.0101637809408346],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, -0.0000000000000000, 0.0162856857170278],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0203615522580970, 0.0000000000000000, -0.0225742810190271],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0084512391474741],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0135416362745717],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0067871192953180, 0.0225742810190271, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0101637809408346, -0.0084512391474741, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0162856857170278, -0.0135416362745717, 0.0000000000000000]
        ]
    ];
    let gamma1_lr_ao: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0203615522580970,
                -0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0203615522580970,
                -0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0203615522580970,
                -0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0203615522580970,
                -0.0067871192953180
            ],
            [
                0.0203615522580970,
                0.0203615522580970,
                0.0203615522580970,
                0.0203615522580970,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.0067871192953180,
                -0.0067871192953180,
                -0.0067871192953180,
                -0.0067871192953180,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0101637809408346,
                0.0101637809408346,
                0.0101637809408346,
                0.0101637809408346,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0162856857170278,
                0.0162856857170278,
                0.0162856857170278,
                0.0162856857170278,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0203615522580970,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0203615522580970,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0203615522580970,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0203615522580970,
                0.0000000000000000
            ],
            [
                -0.0203615522580970,
                -0.0203615522580970,
                -0.0203615522580970,
                -0.0203615522580970,
                0.0000000000000000,
                -0.0225742810190271
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0225742810190271,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0084512391474741
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0084512391474741,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0135416362745717
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0135416362745717,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0067871192953180
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0225742810190271
            ],
            [
                0.0067871192953180,
                0.0067871192953180,
                0.0067871192953180,
                0.0067871192953180,
                0.0225742810190271,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0101637809408346
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0084512391474741
            ],
            [
                -0.0101637809408346,
                -0.0101637809408346,
                -0.0101637809408346,
                -0.0101637809408346,
                -0.0084512391474741,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0162856857170278
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0135416362745717
            ],
            [
                -0.0162856857170278,
                -0.0162856857170278,
                -0.0162856857170278,
                -0.0162856857170278,
                -0.0135416362745717,
                0.0000000000000000
            ]
        ]
    ];
    let gamma0: Array2<f64> = array![
        [0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
        [0.3863557889890281, 0.4720158398964135, 0.3084885848056254],
        [0.3863561531176491, 0.3084885848056254, 0.4720158398964135]
    ];
    let gamma1: Array3<f64> = array![
        [
            [0.0000000000000000, 0.0671593223694436, -0.0223862512902948],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, -0.0000000000000000, 0.0335236415187203],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, -0.0000000000000000, 0.0537157867768206],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0671593223694436, 0.0000000000000000, -0.0573037072665056],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0214530568542981],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0343747807663729],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0223862512902948, 0.0573037072665056, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0335236415187203, -0.0214530568542981, 0.0000000000000000]
        ],
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-0.0537157867768206, -0.0343747807663729, 0.0000000000000000]
        ]
    ];
    let gamma0_AO: Array2<f64> = array![
        [
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.3863557889890281,
            0.3863561531176491
        ],
        [
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.3863557889890281,
            0.3863561531176491
        ],
        [
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.3863557889890281,
            0.3863561531176491
        ],
        [
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.4467609798860577,
            0.3863557889890281,
            0.3863561531176491
        ],
        [
            0.3863557889890281,
            0.3863557889890281,
            0.3863557889890281,
            0.3863557889890281,
            0.4720158398964135,
            0.3084885848056254
        ],
        [
            0.3863561531176491,
            0.3863561531176491,
            0.3863561531176491,
            0.3863561531176491,
            0.3084885848056254,
            0.4720158398964135
        ]
    ];
    let gamma1_AO: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0671593223694436,
                -0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0671593223694436,
                -0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0671593223694436,
                -0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0671593223694436,
                -0.0223862512902948
            ],
            [
                0.0671593223694436,
                0.0671593223694436,
                0.0671593223694436,
                0.0671593223694436,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.0223862512902948,
                -0.0223862512902948,
                -0.0223862512902948,
                -0.0223862512902948,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0335236415187203,
                0.0335236415187203,
                0.0335236415187203,
                0.0335236415187203,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0537157867768206,
                0.0537157867768206,
                0.0537157867768206,
                0.0537157867768206,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0671593223694436,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0671593223694436,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0671593223694436,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0671593223694436,
                0.0000000000000000
            ],
            [
                -0.0671593223694436,
                -0.0671593223694436,
                -0.0671593223694436,
                -0.0671593223694436,
                0.0000000000000000,
                -0.0573037072665056
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0573037072665056,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0214530568542981
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0214530568542981,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0343747807663729
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0343747807663729,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0223862512902948
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0573037072665056
            ],
            [
                0.0223862512902948,
                0.0223862512902948,
                0.0223862512902948,
                0.0223862512902948,
                0.0573037072665056,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0335236415187203
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0214530568542981
            ],
            [
                -0.0335236415187203,
                -0.0335236415187203,
                -0.0335236415187203,
                -0.0335236415187203,
                -0.0214530568542981,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0537157867768206
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0343747807663729
            ],
            [
                -0.0537157867768206,
                -0.0537157867768206,
                -0.0537157867768206,
                -0.0537157867768206,
                -0.0343747807663729,
                0.0000000000000000
            ]
        ]
    ];
    let gradS: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.3590399304938401,
                -0.1196795358000320
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0918870323801337
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1472329381678249
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.3350012527450977,
                0.1558859507933732
            ],
            [
                0.3590399304938401,
                0.0000000000000000,
                0.0000000000000000,
                -0.3350012527450977,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.1196795358000320,
                0.0918870323801337,
                0.1472329381678249,
                0.1558859507933732,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1792213355983593
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2172441846869481,
                0.0796440422386294
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2204828389926055
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0918870323801337
            ],
            [
                0.0000000000000000,
                0.2172441846869481,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.1792213355983593,
                0.0796440422386294,
                -0.2204828389926055,
                0.0918870323801337,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2871709221530289
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2204828389926055
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2172441846869481,
                -0.1360394644282639
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1472329381678249
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.2172441846869481,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.2871709221530289,
                -0.2204828389926055,
                -0.1360394644282639,
                0.1472329381678249,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.3590399304938401,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.3350012527450977,
                0.0000000000000000
            ],
            [
                -0.3590399304938401,
                0.0000000000000000,
                0.0000000000000000,
                0.3350012527450977,
                0.0000000000000000,
                -0.0493263812570877
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0493263812570877,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2172441846869481,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                -0.2172441846869481,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0184665480123938
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0184665480123938,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2172441846869481,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                -0.2172441846869481,
                0.0000000000000000,
                0.0000000000000000,
                0.0295894213933693
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0295894213933693,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1196795358000320
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0918870323801337
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1472329381678249
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1558859507933732
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0493263812570877
            ],
            [
                0.1196795358000320,
                -0.0918870323801337,
                -0.1472329381678249,
                -0.1558859507933732,
                0.0493263812570877,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1792213355983593
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0796440422386294
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2204828389926055
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0918870323801337
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0184665480123938
            ],
            [
                -0.1792213355983593,
                -0.0796440422386294,
                0.2204828389926055,
                -0.0918870323801337,
                -0.0184665480123938,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2871709221530289
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2204828389926055
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1360394644282639
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1472329381678249
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0295894213933693
            ],
            [
                -0.2871709221530289,
                0.2204828389926055,
                0.1360394644282639,
                -0.1472329381678249,
                -0.0295894213933693,
                0.0000000000000000
            ]
        ]
    ];
    let gradH0: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.4466562325187367,
                0.1488849546574482
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0859724600043996
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1377558678312508
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.3151837170301471,
                -0.1441050567093689
            ],
            [
                -0.4466562325187367,
                0.0000000000000000,
                0.0000000000000000,
                0.3151837170301471,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.1488849546574482,
                -0.0859724600043996,
                -0.1377558678312508,
                -0.1441050567093689,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2229567506745117
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2015137919014903,
                -0.0727706772643434
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2062908287050795
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0859724600043996
            ],
            [
                0.0000000000000000,
                -0.2015137919014903,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.2229567506745117,
                -0.0727706772643434,
                0.2062908287050795,
                -0.0859724600043996,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.3572492944418646
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2062908287050795
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2015137919014903,
                0.1290297419048926
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1377558678312508
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                -0.2015137919014903,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                -0.3572492944418646,
                0.2062908287050795,
                0.1290297419048926,
                -0.1377558678312508,
                0.0000000000000000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.4466562325187367,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.3151837170301471,
                0.0000000000000000
            ],
            [
                0.4466562325187367,
                0.0000000000000000,
                0.0000000000000000,
                -0.3151837170301471,
                0.0000000000000000,
                0.0738575792762484
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0738575792762484,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2015137919014903,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.2015137919014903,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0276504073281890
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0276504073281890,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2015137919014903,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.2015137919014903,
                0.0000000000000000,
                0.0000000000000000,
                -0.0443049536699000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0443049536699000,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1488849546574482
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0859724600043996
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1377558678312508
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1441050567093689
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.0738575792762484
            ],
            [
                -0.1488849546574482,
                0.0859724600043996,
                0.1377558678312508,
                0.1441050567093689,
                -0.0738575792762484,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.2229567506745117
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0727706772643434
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2062908287050795
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0859724600043996
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0276504073281890
            ],
            [
                0.2229567506745117,
                0.0727706772643434,
                -0.2062908287050795,
                0.0859724600043996,
                0.0276504073281890,
                0.0000000000000000
            ]
        ],
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.3572492944418646
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.2062908287050795
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.1290297419048926
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.1377558678312508
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0443049536699000
            ],
            [
                0.3572492944418646,
                -0.2062908287050795,
                -0.1290297419048926,
                0.1377558678312508,
                0.0443049536699000,
                0.0000000000000000
            ]
        ]
    ];
    let orbe_occ: Array1<f64> = array![
        -0.8274698453897348,
        -0.4866977301135286,
        -0.4293504173916549,
        -0.3805317623354842
    ];
    let orbe_virt: Array1<f64> = array![0.4597732058522500, 0.5075648555895175];
    let orbs_occ: Array2<f64> = array![
        [
            8.7633817073096332e-01,
            -7.3282333460513933e-07,
            -2.5626946551477814e-01,
            3.5545737574547093e-16
        ],
        [
            1.5609825393248001e-02,
            -1.9781346650256848e-01,
            -3.5949496391504693e-01,
            -8.4834397825097241e-01
        ],
        [
            2.5012021798970618e-02,
            -3.1696156822050980e-01,
            -5.7602795979720633e-01,
            5.2944545948125143e-01
        ],
        [
            2.0847651645094598e-02,
            5.2838144790875974e-01,
            -4.8012913249888561e-01,
            1.3290510512115002e-15
        ],
        [
            1.6641905232447368e-01,
            -3.7146604214648776e-01,
            2.5136102811675498e-01,
            -5.9695075273495377e-16
        ],
        [
            1.6641962261693885e-01,
            3.7146556016201521e-01,
            2.5135992631729770e-01,
            4.7826699854874327e-17
        ]
    ];
    let orbs_virt: Array2<f64> = array![
        [4.4638746430458731e-05, 6.5169208620107999e-01],
        [2.8325035872464788e-01, -2.9051011756322664e-01],
        [4.5385928211929927e-01, -4.6549177907242634e-01],
        [-7.5667687027917274e-01, -3.8791257751172398e-01],
        [-7.2004450606556147e-01, -7.6949321868972742e-01],
        [7.1993590540307117e-01, -7.6959837575446877e-01]
    ];
    let nocc: usize = 4;
    let nvirt: usize = 2;
    let gradVrep: Array1<f64> = array![
        0.1578504879797087,
        0.1181937590058072,
        0.1893848779393944,
        -0.2367773309532266,
        0.0000000000000000,
        0.0000000000000000,
        0.0789268429735179,
        -0.1181937590058072,
        -0.1893848779393944
    ];
    let gradE0: Array1<f64> = array![
        -0.0955096709004110,
        -0.0715133858595269,
        -0.1145877241401038,
        0.1612048707194388,
        -0.0067164109317917,
        -0.0107618767285816,
        -0.0656951998190278,
        0.0782297967913186,
        0.1253496008686854
    ];
    let gradExc: Array1<f64> = array![
        0.3386045165303381,
        0.2536652454776656,
        0.4064542997561643,
        -0.7230154291658203,
        0.0804978899187990,
        0.1289838243988970,
        0.3844109126354822,
        -0.3341631353964647,
        -0.5354381241550613
    ];
    let omega: Array1<f64> = array![
        0.5639270376740081,
        0.6133146373942775,
        0.6284386094838754,
        0.6699518396823455,
        0.7221984798804741,
        0.7797834247427743,
        1.0399489219631464,
        1.0845186330829202
    ];
    let XmY: Array3<f64> = array![
        [
            [-6.7160620047438017e-17, -5.1301838632504444e-18],
            [6.1827987228826222e-16, 1.7499314216344848e-16],
            [1.6069268971051589e-16, -3.1898570182056540e-17],
            [-9.9999999999555755e-01, 2.9807538842552423e-06]
        ],
        [
            [-1.0254775188457787e-18, 6.9809135673006465e-17],
            [-4.4763166261423607e-16, -5.8978588162873293e-16],
            [3.3092655011635419e-16, -1.2150083884089285e-16],
            [2.9807538842880051e-06, 9.9999999999555755e-01]
        ],
        [
            [-3.4096336997069134e-02, 6.1258454571076908e-07],
            [3.9780204906517530e-05, 1.7222258226335382e-01],
            [9.9024005503008095e-01, 2.9144620365419810e-06],
            [1.6960068405972428e-16, -2.2260684017288400e-16]
        ],
        [
            [-2.5364612085089851e-06, 1.4682256488048218e-02],
            [-4.6763207818765606e-01, 3.7130880372279206e-05],
            [1.5649698450732982e-05, -8.8583534829327848e-01],
            [-2.6041562351418946e-16, -3.0910995580813334e-16]
        ],
        [
            [-1.0628036269263018e-05, -7.1701437719352515e-02],
            [-9.0792672280396047e-01, 8.2541765540584952e-06],
            [3.6384675265926511e-05, 4.8569044549211882e-01],
            [-5.8698143078261842e-16, -3.9708414343434237e-16]
        ],
        [
            [-1.1896987769742669e-01, 1.1788386439345313e-05],
            [-1.8212172831982295e-05, -1.0063064903016539e+00],
            [2.0293601397976366e-01, -3.2261881487866900e-05],
            [-1.8936682753514791e-16, -6.6611717841939922e-16]
        ],
        [
            [1.0016339682724578e+00, 6.6106320059117291e-06],
            [-1.6702215607300192e-05, -1.5033047374961833e-01],
            [7.9992335306747098e-02, -4.7224233614213065e-07],
            [-1.1373390545893086e-16, -1.1723927149645725e-16]
        ],
        [
            [-6.0223763823537746e-06, 1.0057436303032681e+00],
            [-8.3630812238129307e-02, 1.7129089350739801e-05],
            [-1.2248116993918996e-06, 6.7510772476669936e-02],
            [-6.4349356797468524e-17, -1.3199057068888888e-16]
        ]
    ];
    let XpY: Array3<f64> = array![
        [
            [-9.1528601848405032e-17, -9.3198834085742693e-18],
            [6.2729042247126871e-16, 2.2499612349354311e-16],
            [1.2896383402827113e-16, -3.7312414589145561e-17],
            [-9.9999999999555778e-01, 2.9807538842549949e-06]
        ],
        [
            [1.3248482112914848e-18, 9.6234920152849223e-17],
            [-4.9186879695308368e-16, -5.9242524234884815e-16],
            [3.3409029139138233e-16, -8.7714798259291981e-17],
            [2.9807538842665349e-06, 9.9999999999555755e-01]
        ],
        [
            [-4.7420378519548285e-02, 8.2900487714134389e-07],
            [4.2985498975294977e-05, 2.0184926468961656e-01],
            [9.7311771078766207e-01, 3.1099380170612475e-06],
            [1.9490709105851324e-16, -1.9673182994695335e-16]
        ],
        [
            [-3.5195127520993807e-06, 1.9781980905011955e-02],
            [-4.7198640639973560e-01, 4.0288485623254992e-05],
            [1.4408580149043882e-05, -8.7938867144101562e-01],
            [-2.6386129153271599e-16, -3.1948377538542627e-16]
        ],
        [
            [-1.4654734885296861e-05, -1.0108421929429864e-01],
            [-8.5334248772724131e-01, 7.8848241640237460e-06],
            [3.1146410007646910e-05, 4.4880369482940324e-01],
            [-5.4139568288584693e-16, -3.2040082761685790e-16]
        ],
        [
            [-1.5428166166672055e-01, 1.5821073295078862e-05],
            [-1.6325778041880442e-05, -9.4347361543836172e-01],
            [1.5877667515112676e-01, -2.7419455420427776e-05],
            [-1.2923664157452835e-16, -6.1208737779577347e-16]
        ],
        [
            [9.7842954534177307e-01, 6.8180504172591540e-06],
            [-1.1049986933891715e-05, -1.0519074149796337e-01],
            [5.1984450448914178e-02, -3.4611368124568511e-07],
            [-7.3075625394683845e-17, -7.8447521815669574e-17]
        ],
        [
            [-5.5872568625835780e-06, 9.8679389249228922e-01],
            [-5.3946210431269737e-02, 1.1600954869661929e-05],
            [-7.8528545029687771e-07, 4.4833771912454586e-02],
            [-3.9844404876807111e-17, -8.7580662633292646e-17]
        ]
    ];
    let FlrDmD0: Array3<f64> = array![
        [
            [
                -4.0738398288130556e-03,
                5.4903447790611380e-03,
                8.7973196264669613e-03,
                -4.6556028719722461e-02,
                -5.6972962473002452e-02,
                2.0487221159291830e-02
            ],
            [
                5.4903447790611371e-03,
                -1.1285849399834234e-02,
                -1.8083604659066468e-02,
                -4.8819575480307468e-03,
                4.8623878260349558e-03,
                -3.5654362816992754e-03
            ],
            [
                8.7973196264669613e-03,
                -1.8083604659066464e-02,
                -2.8975821480500489e-02,
                -7.8224852320137282e-03,
                7.7911281667789305e-03,
                -5.7129895917526503e-03
            ],
            [
                -4.6556028719722455e-02,
                -4.8819575480307459e-03,
                -7.8224852320137282e-03,
                9.3145197462136925e-02,
                3.6184775020237585e-02,
                -2.0075629792977731e-03
            ],
            [
                -5.6972962473002445e-02,
                4.8623878260349532e-03,
                7.7911281667789340e-03,
                3.6184775020237585e-02,
                7.2785773814337235e-02,
                -2.9368678263062310e-02
            ],
            [
                2.0487221159291830e-02,
                -3.5654362816992745e-03,
                -5.7129895917526477e-03,
                -2.0075629792977705e-03,
                -2.9368678263062310e-02,
                -2.9979431341325535e-02
            ]
        ],
        [
            [
                -3.0504203568801611e-03,
                -1.3832996742714640e-02,
                -1.5845793884760977e-02,
                5.4903426180716378e-03,
                8.4020364950677104e-04,
                -2.8159079164154513e-02
            ],
            [
                -1.3832996742714642e-02,
                -1.0200064166785910e-02,
                5.3682803426542778e-03,
                -2.8535652344704270e-02,
                1.2871624253655465e-02,
                2.3577591691672575e-02
            ],
            [
                -1.5845793884760977e-02,
                5.3682803426542769e-03,
                4.3391590643130966e-02,
                -1.6725496875181059e-02,
                -8.7453897719416401e-03,
                8.4087204336778451e-03
            ],
            [
                5.4903426180716386e-03,
                -2.8535652344704260e-02,
                -1.6725496875181059e-02,
                6.4044158061542155e-03,
                3.5836019647354192e-03,
                -2.2867147603989843e-03
            ],
            [
                8.4020364950677018e-04,
                1.2871624253655465e-02,
                -8.7453897719416383e-03,
                3.5836019647354183e-03,
                -3.2109692296644768e-03,
                -2.1989510885454630e-02
            ],
            [
                -2.8159079164154506e-02,
                2.3577591691672572e-02,
                8.4087204336778416e-03,
                -2.2867147603989834e-03,
                -2.1989510885454627e-02,
                3.5261441946498016e-02
            ]
        ],
        [
            [
                -4.8877664253254615e-03,
                -1.5845793884760984e-02,
                -2.9333869312215093e-02,
                8.7973161638582337e-03,
                1.3462797608311519e-03,
                -4.5120011540771179e-02
            ],
            [
                -1.5845793884760984e-02,
                2.7080383941710924e-02,
                2.9841220791968645e-02,
                -1.6725496875181066e-02,
                -8.7453897719416418e-03,
                8.4087204336779440e-03
            ],
            [
                -2.9333869312215086e-02,
                2.9841220791968645e-02,
                2.6103246422697325e-02,
                -4.4897079080987369e-02,
                4.3165986480372563e-03,
                3.1803276366288109e-02
            ],
            [
                8.7973161638582355e-03,
                -1.6725496875181059e-02,
                -4.4897079080987376e-02,
                1.0261958972487295e-02,
                5.7420969295125281e-03,
                -3.6640614481099356e-03
            ],
            [
                1.3462797608311528e-03,
                -8.7453897719416383e-03,
                4.3165986480372546e-03,
                5.7420969295125281e-03,
                -5.1450179835406569e-03,
                -3.5234354757971584e-02
            ],
            [
                -4.5120011540771172e-02,
                8.4087204336779423e-03,
                3.1803276366288116e-02,
                -3.6640614481099373e-03,
                -3.5234354757971577e-02,
                5.6500308774141403e-02
            ]
        ],
        [
            [
                6.2113865086159273e-03,
                1.0809474247735734e-03,
                1.7320296589397603e-03,
                4.0874400348748306e-02,
                6.1548198025732595e-02,
                1.2109848046809491e-02
            ],
            [
                1.0809474247735734e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.0185729898407885e-03,
                -5.1208859059130541e-03,
                1.3380437013933301e-04
            ],
            [
                1.7320296589397606e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6320855092177499e-03,
                -8.2053262404934366e-03,
                2.1439815875006426e-04
            ],
            [
                4.0874400348748299e-02,
                -1.0185729898407885e-03,
                -1.6320855092177497e-03,
                -8.2785739513022039e-02,
                -4.2588101820418840e-02,
                -5.2763759420740739e-03
            ],
            [
                6.1548198025732595e-02,
                -5.1208859059130533e-03,
                -8.2053262404934383e-03,
                -4.2588101820418840e-02,
                -6.9506154298119016e-02,
                2.2268859151741888e-02
            ],
            [
                1.2109848046809490e-02,
                1.3380437013933301e-04,
                2.1439815875006429e-04,
                -5.2763759420740739e-03,
                2.2268859151741885e-02,
                3.2819697729205865e-03
            ]
        ],
        [
            [
                -3.7644153389846622e-05,
                1.9840087023916841e-03,
                1.9496111787931564e-05,
                1.6250374616467047e-05,
                -6.7842244584777334e-03,
                -3.0247245850300707e-04
            ],
            [
                1.9840087023916841e-03,
                1.2746747925423034e-03,
                1.0212219834135874e-03,
                2.5427821022893489e-02,
                -1.6386520692992748e-02,
                6.7888312504343811e-03
            ],
            [
                1.9496111787931567e-05,
                1.0212219834135874e-03,
                0.0000000000000000e+00,
                -2.5248357054310708e-05,
                1.4068483929400346e-02,
                -7.7121760014770460e-05
            ],
            [
                1.6250374616467047e-05,
                2.5427821022893489e-02,
                -2.5248357054310708e-05,
                2.1044690940980066e-05,
                -5.9036840802718744e-03,
                -1.9277900843462953e-03
            ],
            [
                -6.7842244584777351e-03,
                -1.6386520692992752e-02,
                1.4068483929400346e-02,
                -5.9036840802718744e-03,
                1.9818458000688747e-03,
                8.1552306935161658e-03
            ],
            [
                -3.0247245850300713e-04,
                6.7888312504343793e-03,
                -7.7121760014770460e-05,
                -1.9277900843462953e-03,
                8.1552306935161710e-03,
                -1.2273694138129165e-03
            ]
        ],
        [
            [
                -6.0318188158459924e-05,
                1.9496111787931574e-05,
                2.0030804350308481e-03,
                2.6038390174711645e-05,
                -1.0870536073898224e-02,
                -4.8465934339905406e-04
            ],
            [
                1.9496111787931574e-05,
                0.0000000000000000e+00,
                6.3733739627115172e-04,
                -2.5248357054310715e-05,
                1.4068483929400348e-02,
                -7.7121760014774336e-05
            ],
            [
                2.0030804350308481e-03,
                6.3733739627115172e-04,
                2.0424439668271749e-03,
                2.5403122256259618e-02,
                -2.6242707090760918e-03,
                6.7133882285814843e-03
            ],
            [
                2.6038390174711649e-05,
                -2.5248357054310719e-05,
                2.5403122256259618e-02,
                3.3720445636507502e-05,
                -9.4596237427400798e-03,
                -3.0889472751159386e-03
            ],
            [
                -1.0870536073898223e-02,
                1.4068483929400348e-02,
                -2.6242707090760910e-03,
                -9.4596237427400816e-03,
                3.1755621286424628e-03,
                1.3067334370702939e-02
            ],
            [
                -4.8465934339905401e-04,
                -7.7121760014774336e-05,
                6.7133882285814834e-03,
                -3.0889472751159386e-03,
                1.3067334370702942e-02,
                -1.9666453506236711e-03
            ]
        ],
        [
            [
                -2.1375466798028726e-03,
                -6.5712922038347107e-03,
                -1.0529349285406721e-02,
                5.6816283709741587e-03,
                -4.5752355527301468e-03,
                -3.2597069206101316e-02
            ],
            [
                -6.5712922038347115e-03,
                1.1285849399834234e-02,
                1.8083604659066468e-02,
                5.9005305378715357e-03,
                2.5849807987809819e-04,
                3.4316319115599422e-03
            ],
            [
                -1.0529349285406721e-02,
                1.8083604659066464e-02,
                2.8975821480500489e-02,
                9.4545707412314779e-03,
                4.1419807371450499e-04,
                5.4985914330025858e-03
            ],
            [
                5.6816283709741596e-03,
                5.9005305378715357e-03,
                9.4545707412314779e-03,
                -1.0359457949114879e-02,
                6.4033268001812632e-03,
                7.2839389213718506e-03
            ],
            [
                -4.5752355527301477e-03,
                2.5849807987809851e-04,
                4.1419807371450472e-04,
                6.4033268001812632e-03,
                -3.2796195162182488e-03,
                7.0998191113204297e-03
            ],
            [
                -3.2597069206101309e-02,
                3.4316319115599413e-03,
                5.4985914330025840e-03,
                7.2839389213718454e-03,
                7.0998191113204297e-03,
                2.6697461568404950e-02
            ]
        ],
        [
            [
                3.0880645102700078e-03,
                1.1848988040322956e-02,
                1.5826297772973046e-02,
                -5.5065929926881047e-03,
                5.9440208089709644e-03,
                2.8461551622657517e-02
            ],
            [
                1.1848988040322958e-02,
                8.9253893742436054e-03,
                -6.3895023260678654e-03,
                3.1078313218107800e-03,
                3.5148964393372810e-03,
                -3.0366422942106956e-02
            ],
            [
                1.5826297772973043e-02,
                -6.3895023260678646e-03,
                -4.3391590643130966e-02,
                1.6750745232235367e-02,
                -5.3230941574587062e-03,
                -8.3315986736630723e-03
            ],
            [
                -5.5065929926881056e-03,
                3.1078313218107787e-03,
                1.6750745232235367e-02,
                -6.4254604970951958e-03,
                2.3200821155364556e-03,
                4.2145048447452781e-03
            ],
            [
                5.9440208089709652e-03,
                3.5148964393372801e-03,
                -5.3230941574587062e-03,
                2.3200821155364561e-03,
                1.2291234295956027e-03,
                1.3834280191938463e-02
            ],
            [
                2.8461551622657510e-02,
                -3.0366422942106969e-02,
                -8.3315986736630706e-03,
                4.2145048447452798e-03,
                1.3834280191938459e-02,
                -3.4034072532685089e-02
            ]
        ],
        [
            [
                4.9480846134839218e-03,
                1.5826297772973053e-02,
                2.7330788877184240e-02,
                -8.8233545540329448e-03,
                9.5242563130670688e-03,
                4.5604670884170230e-02
            ],
            [
                1.5826297772973053e-02,
                -2.7080383941710924e-02,
                -3.0478558188239798e-02,
                1.6750745232235374e-02,
                -5.3230941574587045e-03,
                -8.3315986736631660e-03
            ],
            [
                2.7330788877184237e-02,
                -3.0478558188239795e-02,
                -2.8145690389524497e-02,
                1.9493956824727751e-02,
                -1.6923279389611625e-03,
                -3.8516664594869611e-02
            ],
            [
                -8.8233545540329465e-03,
                1.6750745232235367e-02,
                1.9493956824727751e-02,
                -1.0295679418123803e-02,
                3.7175268132275526e-03,
                6.7530087232258746e-03
            ],
            [
                9.5242563130670688e-03,
                -5.3230941574587054e-03,
                -1.6923279389611629e-03,
                3.7175268132275526e-03,
                1.9694558548981936e-03,
                2.2167020387268629e-02
            ],
            [
                4.5604670884170223e-02,
                -8.3315986736631660e-03,
                -3.8516664594869611e-02,
                6.7530087232258746e-03,
                2.2167020387268633e-02,
                -5.4533663423517724e-02
            ]
        ]
    ];

    let FDmD0: Array3<f64> = array![
        [
            [
                1.2895439603332617e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.4551664955240511e-02,
                3.1896810825290445e-02
            ],
            [
                0.0000000000000000e+00,
                1.2895439603332617e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0418366493227764e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.2895439603332617e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.2716869981703256e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.2895439603332617e-01,
                -6.8847147171129136e-02,
                1.6643149085710365e-02
            ],
            [
                5.4551664955240511e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -6.8847147171129136e-02,
                1.9352532012311063e-01,
                3.7900443203262332e-03
            ],
            [
                3.1896810825290445e-02,
                -2.0418366493227764e-02,
                -3.2716869981703256e-02,
                1.6643149085710365e-02,
                3.7900443203262332e-03,
                8.9287743042153617e-02
            ]
        ],
        [
            [
                9.6554182299140187e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.8123169086191932e-02,
                3.6604907768036161e-02
            ],
            [
                0.0000000000000000e+00,
                9.6554182299140187e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.0082385354721721e-03,
                -2.0955801994294795e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.6554182299140187e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.8398249953977337e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.6554182299140187e-02,
                -3.6420791094687342e-02,
                1.6002611599989034e-02
            ],
            [
                2.8123169086191932e-02,
                3.0082385354721721e-03,
                0.0000000000000000e+00,
                -3.6420791094687342e-02,
                8.6365585175836868e-02,
                2.8377812843105180e-03
            ],
            [
                3.6604907768036161e-02,
                -2.0955801994294795e-02,
                -3.8398249953977337e-02,
                1.6002611599989034e-02,
                2.8377812843105180e-03,
                1.2538962130149364e-01
            ]
        ],
        [
            [
                1.5471123165109152e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                4.5062471905946644e-02,
                5.8652978363195267e-02
            ],
            [
                0.0000000000000000e+00,
                1.5471123165109152e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.8398249953977365e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5471123165109152e-01,
                0.0000000000000000e+00,
                3.0082385354721721e-03,
                -5.8518222896465216e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5471123165109152e-01,
                -5.8357963516370251e-02,
                2.5641393167185410e-02
            ],
            [
                4.5062471905946644e-02,
                0.0000000000000000e+00,
                3.0082385354721721e-03,
                -5.8357963516370251e-02,
                1.3838578233125345e-01,
                4.5470494099560744e-03
            ],
            [
                5.8652978363195267e-02,
                -3.8398249953977365e-02,
                -5.8518222896465216e-02,
                2.5641393167185410e-02,
                4.5470494099560744e-03,
                2.0091499183035150e-01
            ]
        ],
        [
            [
                -2.0365105882487339e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -7.4061526517149284e-02,
                -5.7278976029705761e-02
            ],
            [
                0.0000000000000000e+00,
                -2.0365105882487339e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.7027548534438903e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0365105882487339e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9330186454643281e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0365105882487339e-01,
                9.4113310812033571e-02,
                -2.4726072962349757e-02
            ],
            [
                -7.4061526517149284e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.4113310812033571e-02,
                -2.4572542808438536e-01,
                -5.4623382852136276e-03
            ],
            [
                -5.7278976029705761e-02,
                3.7027548534438903e-02,
                5.9330186454643281e-02,
                -2.4726072962349757e-02,
                -5.4623382852136276e-03,
                -1.6890263930809610e-01
            ]
        ],
        [
            [
                3.8261749328971431e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.9028764003359775e-04,
                2.3425297869790289e-04
            ],
            [
                0.0000000000000000e+00,
                3.8261749328971431e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.0082385354721721e-03,
                -1.5143101604286901e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8261749328971431e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.4264178354889410e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8261749328971431e-03,
                -5.0544035634594777e-04,
                1.0112185385311725e-04
            ],
            [
                3.9028764003359775e-04,
                -3.0082385354721721e-03,
                0.0000000000000000e+00,
                -5.0544035634594777e-04,
                -1.2876514775744868e-03,
                -8.3375408545420415e-05
            ],
            [
                2.3425297869790289e-04,
                -1.5143101604286901e-04,
                -2.4264178354889410e-04,
                1.0112185385311725e-04,
                -8.3375408545420415e-05,
                -2.3025475994178056e-03
            ]
        ],
        [
            [
                6.1307777901020719e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.2536785098255779e-04,
                3.7534952903441748e-04
            ],
            [
                0.0000000000000000e+00,
                6.1307777901020719e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.4264178354886830e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.1307777901020719e-03,
                0.0000000000000000e+00,
                -3.0082385354721721e-03,
                -3.8879112524157567e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.1307777901020719e-03,
                -8.0987998856615982e-04,
                1.6203012841003648e-04
            ],
            [
                6.2536785098255779e-04,
                0.0000000000000000e+00,
                -3.0082385354721721e-03,
                -8.0987998856615982e-04,
                -2.0632368405929195e-03,
                -1.3359454596706925e-04
            ],
            [
                3.7534952903441748e-04,
                -2.4264178354886830e-04,
                -3.8879112524157567e-04,
                1.6203012841003648e-04,
                -1.3359454596706925e-04,
                -3.6894308103355587e-03
            ]
        ],
        [
            [
                7.4696662791547241e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.9509861561908780e-02,
                2.5382165204415306e-02
            ],
            [
                0.0000000000000000e+00,
                7.4696662791547241e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6609182041211132e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.4696662791547241e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.6613316472940025e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.4696662791547241e-02,
                -2.5266163640904438e-02,
                8.0829238766393936e-03
            ],
            [
                1.9509861561908780e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.5266163640904438e-02,
                5.2200107961274729e-02,
                1.6722939648873942e-03
            ],
            [
                2.5382165204415306e-02,
                -1.6609182041211132e-02,
                -2.6613316472940025e-02,
                8.0829238766393936e-03,
                1.6722939648873942e-03,
                7.9614896265942495e-02
            ]
        ],
        [
            [
                -1.0038035723203732e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.8513456726225527e-02,
                -3.6839160746734066e-02
            ],
            [
                0.0000000000000000e+00,
                -1.0038035723203732e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.1107233010337664e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.0038035723203732e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8640891737526231e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.0038035723203732e-01,
                3.6926231451033284e-02,
                -1.6103733453842153e-02
            ],
            [
                -2.8513456726225527e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6926231451033284e-02,
                -8.5077933698262387e-02,
                -2.7544058757650982e-03
            ],
            [
                -3.6839160746734066e-02,
                2.1107233010337664e-02,
                3.8640891737526231e-02,
                -1.6103733453842153e-02,
                -2.7544058757650982e-03,
                -1.2308707370207586e-01
            ]
        ],
        [
            [
                -1.6084200944119359e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -4.5687839756929208e-02,
                -5.9028327892229684e-02
            ],
            [
                0.0000000000000000e+00,
                -1.6084200944119359e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8640891737526238e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6084200944119359e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.8907014021706797e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6084200944119359e-01,
                5.9167843504936417e-02,
                -2.5803423295595446e-02
            ],
            [
                -4.5687839756929208e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9167843504936417e-02,
                -1.3632254549066050e-01,
                -4.4134548639890043e-03
            ],
            [
                -5.9028327892229684e-02,
                3.8640891737526238e-02,
                5.8907014021706797e-02,
                -2.5803423295595446e-02,
                -4.4134548639890043e-03,
                -1.9722556102001593e-01
            ]
        ]
    ];

    let FDmD0: Array3<f64> = array![
        [
            [
                1.2895439603332617e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.4551664955240511e-02,
                3.1896810825290445e-02
            ],
            [
                0.0000000000000000e+00,
                1.2895439603332617e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0418366493227764e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.2895439603332617e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.2716869981703256e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.2895439603332617e-01,
                -6.8847147171129136e-02,
                1.6643149085710365e-02
            ],
            [
                5.4551664955240511e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -6.8847147171129136e-02,
                1.9352532012311063e-01,
                3.7900443203262332e-03
            ],
            [
                3.1896810825290445e-02,
                -2.0418366493227764e-02,
                -3.2716869981703256e-02,
                1.6643149085710365e-02,
                3.7900443203262332e-03,
                8.9287743042153617e-02
            ]
        ],
        [
            [
                9.6554182299140187e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.8123169086191932e-02,
                3.6604907768036161e-02
            ],
            [
                0.0000000000000000e+00,
                9.6554182299140187e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.0082385354721721e-03,
                -2.0955801994294795e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.6554182299140187e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.8398249953977337e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.6554182299140187e-02,
                -3.6420791094687342e-02,
                1.6002611599989034e-02
            ],
            [
                2.8123169086191932e-02,
                3.0082385354721721e-03,
                0.0000000000000000e+00,
                -3.6420791094687342e-02,
                8.6365585175836868e-02,
                2.8377812843105180e-03
            ],
            [
                3.6604907768036161e-02,
                -2.0955801994294795e-02,
                -3.8398249953977337e-02,
                1.6002611599989034e-02,
                2.8377812843105180e-03,
                1.2538962130149364e-01
            ]
        ],
        [
            [
                1.5471123165109152e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                4.5062471905946644e-02,
                5.8652978363195267e-02
            ],
            [
                0.0000000000000000e+00,
                1.5471123165109152e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.8398249953977365e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5471123165109152e-01,
                0.0000000000000000e+00,
                3.0082385354721721e-03,
                -5.8518222896465216e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.5471123165109152e-01,
                -5.8357963516370251e-02,
                2.5641393167185410e-02
            ],
            [
                4.5062471905946644e-02,
                0.0000000000000000e+00,
                3.0082385354721721e-03,
                -5.8357963516370251e-02,
                1.3838578233125345e-01,
                4.5470494099560744e-03
            ],
            [
                5.8652978363195267e-02,
                -3.8398249953977365e-02,
                -5.8518222896465216e-02,
                2.5641393167185410e-02,
                4.5470494099560744e-03,
                2.0091499183035150e-01
            ]
        ],
        [
            [
                -2.0365105882487339e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -7.4061526517149284e-02,
                -5.7278976029705761e-02
            ],
            [
                0.0000000000000000e+00,
                -2.0365105882487339e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.7027548534438903e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0365105882487339e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9330186454643281e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.0365105882487339e-01,
                9.4113310812033571e-02,
                -2.4726072962349757e-02
            ],
            [
                -7.4061526517149284e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                9.4113310812033571e-02,
                -2.4572542808438536e-01,
                -5.4623382852136276e-03
            ],
            [
                -5.7278976029705761e-02,
                3.7027548534438903e-02,
                5.9330186454643281e-02,
                -2.4726072962349757e-02,
                -5.4623382852136276e-03,
                -1.6890263930809610e-01
            ]
        ],
        [
            [
                3.8261749328971431e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.9028764003359775e-04,
                2.3425297869790289e-04
            ],
            [
                0.0000000000000000e+00,
                3.8261749328971431e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -3.0082385354721721e-03,
                -1.5143101604286901e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8261749328971431e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.4264178354889410e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8261749328971431e-03,
                -5.0544035634594777e-04,
                1.0112185385311725e-04
            ],
            [
                3.9028764003359775e-04,
                -3.0082385354721721e-03,
                0.0000000000000000e+00,
                -5.0544035634594777e-04,
                -1.2876514775744868e-03,
                -8.3375408545420415e-05
            ],
            [
                2.3425297869790289e-04,
                -1.5143101604286901e-04,
                -2.4264178354889410e-04,
                1.0112185385311725e-04,
                -8.3375408545420415e-05,
                -2.3025475994178056e-03
            ]
        ],
        [
            [
                6.1307777901020719e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.2536785098255779e-04,
                3.7534952903441748e-04
            ],
            [
                0.0000000000000000e+00,
                6.1307777901020719e-03,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.4264178354886830e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.1307777901020719e-03,
                0.0000000000000000e+00,
                -3.0082385354721721e-03,
                -3.8879112524157567e-04
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                6.1307777901020719e-03,
                -8.0987998856615982e-04,
                1.6203012841003648e-04
            ],
            [
                6.2536785098255779e-04,
                0.0000000000000000e+00,
                -3.0082385354721721e-03,
                -8.0987998856615982e-04,
                -2.0632368405929195e-03,
                -1.3359454596706925e-04
            ],
            [
                3.7534952903441748e-04,
                -2.4264178354886830e-04,
                -3.8879112524157567e-04,
                1.6203012841003648e-04,
                -1.3359454596706925e-04,
                -3.6894308103355587e-03
            ]
        ],
        [
            [
                7.4696662791547241e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                1.9509861561908780e-02,
                2.5382165204415306e-02
            ],
            [
                0.0000000000000000e+00,
                7.4696662791547241e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6609182041211132e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.4696662791547241e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.6613316472940025e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                7.4696662791547241e-02,
                -2.5266163640904438e-02,
                8.0829238766393936e-03
            ],
            [
                1.9509861561908780e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.5266163640904438e-02,
                5.2200107961274729e-02,
                1.6722939648873942e-03
            ],
            [
                2.5382165204415306e-02,
                -1.6609182041211132e-02,
                -2.6613316472940025e-02,
                8.0829238766393936e-03,
                1.6722939648873942e-03,
                7.9614896265942495e-02
            ]
        ],
        [
            [
                -1.0038035723203732e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -2.8513456726225527e-02,
                -3.6839160746734066e-02
            ],
            [
                0.0000000000000000e+00,
                -1.0038035723203732e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                2.1107233010337664e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.0038035723203732e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8640891737526231e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.0038035723203732e-01,
                3.6926231451033284e-02,
                -1.6103733453842153e-02
            ],
            [
                -2.8513456726225527e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.6926231451033284e-02,
                -8.5077933698262387e-02,
                -2.7544058757650982e-03
            ],
            [
                -3.6839160746734066e-02,
                2.1107233010337664e-02,
                3.8640891737526231e-02,
                -1.6103733453842153e-02,
                -2.7544058757650982e-03,
                -1.2308707370207586e-01
            ]
        ],
        [
            [
                -1.6084200944119359e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -4.5687839756929208e-02,
                -5.9028327892229684e-02
            ],
            [
                0.0000000000000000e+00,
                -1.6084200944119359e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                3.8640891737526238e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6084200944119359e-01,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.8907014021706797e-02
            ],
            [
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                -1.6084200944119359e-01,
                5.9167843504936417e-02,
                -2.5803423295595446e-02
            ],
            [
                -4.5687839756929208e-02,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                5.9167843504936417e-02,
                -1.3632254549066050e-01,
                -4.4134548639890043e-03
            ],
            [
                -5.9028327892229684e-02,
                3.8640891737526238e-02,
                5.8907014021706797e-02,
                -2.5803423295595446e-02,
                -4.4134548639890043e-03,
                -1.9722556102001593e-01
            ]
        ]
    ];

    let (gradEx_test) = gradients_lc_ex(
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
        FlrDmD0.view(),
        Some(1),
    );

    println!("gradEx_result {}", gradEx_test);
    println!("gradEx_ref {}", gradExc);
    assert!(gradEx_test.abs_diff_eq(&gradExc, 1e-10));
}
