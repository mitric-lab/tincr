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
use std::cmp::Ordering;
use std::ops::AddAssign;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;

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

fn f_lr(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_ao: ArrayView2<f64>,
    g0_lr_a0:ArrayView2<f64>,
    g1_ao: ArrayView3<f64>,
    g1_lr_ao:ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) ->(Array3<f64>){
    let sv:Array2<f64> = s.dot(&v);
    let svt:Array2<f64> = s.dot(&v.t());
    let gv:Array2<f64> = &g0_lr_a0 * &v;
    let dsv24:Array3<f64> = tensordot(&grad_s,&v,&[Axis(2)],&[Axis(1)]).into_dimensionality::<Ix3>().unwrap();
    let dsv23:Array3<f64> = tensordot(&grad_s,&v,&[Axis(2)],&[Axis(0)]).into_dimensionality::<Ix3>().unwrap();
    let mut dgv:Array3<f64> = Array::zeros((3*n_atoms,n_orb,n_orb));

    let mut flr:Array3<f64> = Array::zeros((3*n_atoms,n_orb,n_orb));
    let tmp1:Array3<f64> = tensordot(&grad_s,&sv,&[Axis(2)],&[Axis(1)]).into_dimensionality::<Ix3>().unwrap();

    let mut tmp2:Array3<f64> = Array::zeros((3*n_atoms,n_orb,n_orb));
    let mut tmp7:Array3<f64> = Array::zeros((3*n_atoms,n_orb,n_orb));
    let mut tmp10:Array3<f64> = Array::zeros((3*n_atoms,n_orb,n_orb));
    let mut tmp11:Array3<f64> = Array::zeros((3*n_atoms,n_orb,n_orb));

    //for nc in 0..(3*n_atoms){
    //    dgv.slice_mut(s![nc,..,..]).assign(&(g1_lr_ao.slice(s![nc,..,..]).to_owned()*&v));
    //    flr.slice_mut(s![nc,..,..]).add_assign(&(&g0_lr_a0*&tmp1.slice(s![nc,..,..])));
    //    tmp2.slice_mut(s![nc,..,..]).assign(&(&dsv24.slice(s![nc,..,..])*&g0_lr_a0));
    //    tmp7.slice_mut(s![nc,..,..]).assign(&(&dsv23.slice(s![nc,..,..])*&g0_lr_a0));
    //    tmp10.slice_mut(s![nc,..,..]).assign(&(&svt*&g1_lr_ao.slice(s![nc,..,..])));
    //    tmp11.slice_mut(s![nc,..,..]).assign(&(&sv*&g1_lr_ao.slice(s![nc,..,..])));
    //}

    // replace loop with einsums
    dgv = einsum("ijk,jk->ijk",&[&g1_lr_ao,&v]).unwrap().into_dimensionality::<Ix3>().unwrap();
    flr =  flr + einsum("jk,ijk->ijk",&[&g0_lr_a0,&tmp1]).unwrap().into_dimensionality::<Ix3>().unwrap();
    tmp2 = einsum("ijk,jk->ijk",&[&dsv24,&g0_lr_a0]).unwrap().into_dimensionality::<Ix3>().unwrap();
    tmp7 = einsum("ijk,jk->ijk",&[&dsv23,&g0_lr_a0]).unwrap().into_dimensionality::<Ix3>().unwrap();
    tmp10 = einsum("jk,ijk->ijk",&[&svt,&g1_lr_ao]).unwrap().into_dimensionality::<Ix3>().unwrap();
    tmp11 = einsum("jk,ijk->ijk",&[&sv,&g1_lr_ao]).unwrap().into_dimensionality::<Ix3>().unwrap();

    flr = flr + tensordot(&tmp2,&s,&[Axis(2)],&[Axis(1)]).into_dimensionality::<Ix3>().unwrap();
    flr = flr + tensordot(&grad_s,&(&sv*&g0_lr_a0),&[Axis(2)],&[Axis(1)]);
    flr = flr + tensordot(&grad_s, &s.dot(&gv),&[Axis(2)],&[Axis(1)]);

    let mut tmp5:Array3<f64> = tensordot(&s, &dsv23, &[Axis(1)],&[Axis(2)]).into_dimensionality::<Ix3>().unwrap();
    tmp5.swap_axes(0,1);

    for nc in 0.. (3*n_atoms){
        flr.slice_mut(s![nc,..,..]).add_assign(&(&g0_lr_a0*&tmp5.slice(s![nc,..,..])));
    }
    let mut tmp_6:Array3<f64> = tensordot(&(&svt*&g0_lr_a0),&grad_s,&[Axis(1)],&[Axis(2)]).into_dimensionality::<Ix3>().unwrap();
    tmp_6.swap_axes(0,1);

    flr = flr + tmp_6;

    let mut tmp_7:Array3<f64> = tensordot(&s,&tmp7,&[Axis(1)],&[Axis(2)]).into_dimensionality::<Ix3>().unwrap();
    tmp_7.swap_axes(0,1);

    flr = flr +tmp_7;

    let mut tmp8:Array3<f64> = tensordot(&grad_s,&gv, &[Axis(2)],&[Axis(0)]).into_dimensionality::<Ix3>().unwrap();
    tmp8.swap_axes(0,1);

    flr = flr + tmp8;

    let tmp9:Array2<f64> = s.dot(&sv.t());

    for nc in 0..(3*n_atoms){
        flr.slice_mut(s![nc,..,..]).add_assign(&(&g1_lr_ao.slice(s![nc,..,..])*&tmp9));
    }

    flr = flr + tensordot(&tmp10, &s, &[Axis(2)],&[Axis(1)]).into_dimensionality::<Ix3>().unwrap();

    let mut tmp_11:Array3<f64> = tensordot(&s,&tmp11, &[Axis(1)],&[Axis(2)]).into_dimensionality::<Ix3>().unwrap();
    tmp_11.swap_axes(0,1);
    flr = flr + tmp_11;

    let mut tmp12:Array3<f64> = tensordot(&dgv,&s,&[Axis(1)],&[Axis(1)]).into_dimensionality::<Ix3>().unwrap();
    tmp12.swap_axes(0,1);

    flr = flr + tmp12;

    flr = flr * 0.25;

    return flr;
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
