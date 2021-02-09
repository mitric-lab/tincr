#[macro_use(array)]
use ndarray::prelude::*;
use crate::calculator::get_gamma_gradient_matrix;
use crate::h0_and_s::h0_and_s_gradients;
use crate::molecule::{distance_matrix, Molecule};
use crate::parameters::*;
use crate::scc_routine::density_matrix_ref;
use crate::slako_transformations::*;
use approx::AbsDiffEq;
use ndarray::{array, Array2, Array3, ArrayView2, ArrayView3};
use std::collections::HashMap;

// only ground state
pub fn gradient_nolc(
    molecule: &Molecule,
    orbe_occ: Array1<f64>,
    orbe_virt: Array1<f64>,
    orbs_occ: Array2<f64>,
    s: Array2<f64>,
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

    let grad_v_rep: Array1<f64> = gradient_v_rep(
        &molecule.atomic_numbers,
        molecule.distance_matrix.view(),
        molecule.directions_matrix.view(),
        &molecule.calculator.v_rep,
    );

    return (grad_e0, grad_v_rep);
}

pub fn gradient_lc(
    molecule: &Molecule,
    orbe_occ: Array1<f64>,
    orbe_virt: Array1<f64>,
    orbs_occ: Array2<f64>,
    s: Array2<f64>,
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
