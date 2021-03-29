use ndarray::{Array1, Array3, Array2, Array, Axis};
use std::time::Instant;
use crate::calculator::get_gamma_gradient_matrix;
use crate::initialization::Molecule;
use crate::h0_and_s::h0_and_s_gradients;
use crate::gradients::helpers::{f_v_new, f_lr_new};
use crate::defaults;
use ndarray_einsum_beta::tensordot;

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
    let grad_timer = Instant::now();
    let (g1, g1_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
        &molecule.atomic_numbers.unwrap(),
        molecule.n_atoms,
        molecule.calculator.n_orbs,
        molecule.distance_matrix.view(),
        molecule.directions_matrix.view(),
        &molecule.calculator.hubbard_u,
        &molecule.calculator.valorbs,
        Some(0.0),
    );

    let (g1lr, g1lr_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
        &molecule.atomic_numbers.unwrap(),
        molecule.n_atoms,
        molecule.calculator.n_orbs,
        molecule.distance_matrix.view(),
        molecule.directions_matrix.view(),
        &molecule.calculator.hubbard_u,
        &molecule.calculator.valorbs,
        None,
    );
    let n_at: usize = *&molecule.g0.dim().0;
    let n_orb: usize = *&molecule.g0_ao.dim().0;

    info!(
        "{:>65} {:>8.3} s",
        "elapsed time for gammas:",
        grad_timer.elapsed().as_secs_f32()
    );
    drop(grad_timer);

    let grad_timer = Instant::now();
    let (grad_s, grad_h0): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
        &molecule.atomic_numbers.unwrap(),
        molecule.positions.view(),
        molecule.calculator.n_orbs,
        &molecule.calculator.valorbs,
        molecule.proximity_matrix.view(),
        &molecule.calculator.skt,
        &molecule.calculator.orbital_energies,
    );

    info!(
        "{:>65} {:>8.3} s",
        "elapsed time for h0andS gradients:",
        grad_timer.elapsed().as_secs_f32()
    );
    drop(grad_timer);
    let grad_timer = Instant::now();
    let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
    let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

    // density matrix
    let d = 2.0 * orbs_occ.dot(&orbs_occ.t());
    // reference density matrix
    let d_ref: Array2<f64> = density_matrix_ref(&molecule);

    let diff_d: Array2<f64> = &d - &d_ref;
    // computing F(D-D0)

    let fdmd0: Array3<f64> = f_v_new(
        diff_d.view(),
        s.view(),
        grad_s.view(),
        (&molecule.g0_ao).view(),
        g1_ao.view(),
        molecule.n_atoms,
        molecule.calculator.n_orbs,
    );

    info!(
        "{:>65} {:>8.3} s",
        "elapsed time for f:",
        grad_timer.elapsed().as_secs_f32()
    );
    drop(grad_timer);
    let grad_timer = Instant::now();

    let mut flr_dmd0: Array3<f64> = Array::zeros((3 * n_at, n_orb, n_orb));
    if r_lc.unwrap_or(defaults::LONG_RANGE_RADIUS) > 0.0 {
        flr_dmd0 = f_lr_new(
            diff_d.view(),
            s.view(),
            grad_s.view(),
            (&molecule.g0_lr_ao).view(),
            g1lr_ao.view(),
            n_at,
            n_orb,
        );
    }
    info!(
        "{:>65} {:>8.3} s",
        "elapsed time for f_lr:",
        grad_timer.elapsed().as_secs_f32()
    );
    drop(grad_timer);
    let grad_timer = Instant::now();
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
        grad_e0 = grad_e0
            - 0.25
            * tensordot(&flr_dmd0, &diff_d, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix1>()
            .unwrap();
    }

    info!(
        "{:>65} {:>8.3} s",
        "time tensordots gradients:",
        grad_timer.elapsed().as_secs_f32()
    );
    drop(grad_timer);
    let grad_timer = Instant::now();
    let grad_v_rep: Array1<f64> = gradient_v_rep(
        &molecule.atomic_numbers,
        molecule.distance_matrix.view(),
        molecule.directions_matrix.view(),
        &molecule.calculator.v_rep,
    );

    info!(
        "{:>65} {:>8.3} s",
        "time grad_v_rep gradients:",
        grad_timer.elapsed().as_secs_f32()
    );
    drop(grad_timer);

    return (
        grad_e0, grad_v_rep, grad_s, grad_h0, fdmd0, flr_dmd0, g1, g1_ao, g1lr, g1lr_ao,
    );
