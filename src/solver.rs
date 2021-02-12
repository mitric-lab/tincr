use ndarray::prelude::*;
use ndarray::{Array, Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use std::ops::AddAssign;

pub fn build_a_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
) -> () {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;
    let mut k_lr_a: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
    // K_lr_A = np.tensordot(qtrans_oo, np.tensordot(gamma_lr, qtrans_vv, axes=(1,0)),axes=(0,0))
    // K_lr_A = np.swapaxes(K_lr_A, 1, 2)
    // K_A = - K_lr_A
    // electrostatic Coulomb interaction 2*(ov|o'v'), only needed for Singlets
    if multiplicity == 1 {
        for i in 0..n_occ {
            for j in 0..n_occ {
                for a in 0..n_virt {
                    for b in 0..n_virt {
                        k_lr_a.slice_mut(s![i, a, j, b]).add_assign(
                            -1.0 * (&q_trans_oo.slice(s![.., i, j])
                                * &gamma_lr.dot(&q_trans_vv.slice(s![.., a, b])))
                                .sum(),
                        );
                        k_lr_a.slice_mut(s![i, a, j, b]).add_assign(
                            2.0 * (&q_trans_ov.slice(s![.., i, j])
                                * &gamma.dot(&q_trans_ov.slice(s![.., a, b])))
                                .sum(),
                        );
                    }
                }
            }
        }
    } else if multiplicity == 3 {
        for i in 0..n_occ {
            for j in 0..n_occ {
                for a in 0..n_virt {
                    for b in 0..n_virt {
                        k_lr_a.slice_mut(s![i, a, j, b]).add_assign(
                            -1.0 * (&q_trans_oo.slice(s![.., i, j])
                                * &gamma_lr.dot(&q_trans_vv.slice(s![.., a, b])))
                                .sum(),
                        );
                    }
                }
            }
        }
    }
    let mut k_a: Array2<f64> = k_lr_a.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(&omega) + &df_half.dot(k_a.dot(&df_half));
}


pub fn build_b_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
) -> () {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;
    let mut k_lr_b: Array4<f64> = Array4::zeros([n_occ, n_virt, n_virt, n_occ]);
    // ... and for B matrix, (ia|jb)
    // K_lr_B = np.tensordot(qtrans_ov, np.tensordot(gamma_lr, qtrans_ov, axes=(1,0)),axes=(0,0))
    //  got K_ia_jb but we need K_ib_ja
    //K_lr_B = np.swapaxes(K_lr_B, 1, 3)
    // electrostatic Coulomb interaction 2*(ov|o'v'), only needed for Singlets
    if multiplicity == 1 {
        for i in 0..n_occ {
            for j in 0..n_occ {
                for a in 0..n_virt {
                    for b in 0..n_virt {
                        k_lr_b.slice_mut(s![i, b, a, j]).add_assign(
                            -1.0 * (&q_trans_ov.slice(s![.., i, j])
                                * &gamma_lr.dot(&q_trans_ov.slice(s![.., a, b])))
                                .sum(),
                        );
                        k_lr_b.slice_mut(s![i, b, a, j]).add_assign(
                            2.0 * (&q_trans_ov.slice(s![.., i, j])
                                * &gamma.dot(&q_trans_ov.slice(s![.., a, b])))
                                .sum(),
                        );
                    }
                }
            }
        }
    } else if multiplicity == 3 {
        for i in 0..n_occ {
            for j in 0..n_occ {
                for a in 0..n_virt {
                    for b in 0..n_virt {
                        k_lr_b.slice_mut(s![i, b, a, j]).add_assign(
                            -1.0 * (&q_trans_ov.slice(s![.., i, j])
                                * &gamma_lr.dot(&q_trans_ov.slice(s![.., a, b])))
                                .sum(),
                        );
                    }
                }
            }
        }
    }
    let mut k_b: Array2<f64> = k_lr_b.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(k_b.dot(&df_half));
}


fn get_orbital_en_diff(
    orbe: ArrayView1<f64>,
    n_occ: usize,
    n_virt: usize,
    active_occupied_orbs: &[usize],
    active_virtual_orbs: &[usize],
) -> Array2<f64> {
    // energy difference between occupied and virtual Kohn-Sham orbitals
    // omega_ia = omega_a - omega_i
    let mut omega: Array2<f64> = Array2::zeros([n_occ, n_virt]);
    for (i, occ_i) in active_occupied_orbs.iter().enumerate() {
        for (a, virt_a) in active_virtual_orbs.iter().enumerate() {
            omega[[i, a]] = orbe[*virt_a] - orbe[*occ_i];
        }
    }
    return omega;
}

fn get_orbital_occ_diff(
    f: ArrayView1<f64>,
    n_occ: usize,
    n_virt: usize,
    active_occupied_orbs: &[usize],
    active_virtual_orbs: &[usize],
) -> Array2<f64> {
    // occupation difference between occupied and virtual Kohn-Sham orbitals
    // f_ia = f_a - f_i
    let mut df: Array2<f64> = Array2::zeros([n_occ, n_virt]);
    for (i, occ_i) in active_occupied_orbs.iter().enumerate() {
        for (a, virt_a) in active_virtual_orbs.iter().enumerate() {
            df[[i, a]] = f[*virt_a] - f[*occ_i];
        }
    }
    return df;
}
