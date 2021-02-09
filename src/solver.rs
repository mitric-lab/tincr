use ndarray::{Array2, ArrayView2, ArrayView3};

pub fn build_a_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    multiplicity: u8,
) -> () {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;

    // K_lr_A = np.tensordot(qtrans_oo, np.tensordot(gamma_lr, qtrans_vv, axes=(1,0)),axes=(0,0))
    // K_lr_A = np.swapaxes(K_lr_A, 1, 2)
    // K_A = - K_lr_A
    if multiplicity == 1 {
        let mut k_lr_a: Array2<f64> = Array2::zeros([n_occ, n_virt, n_occ, n_occ]);
        for i in 0..n_occ {
            for j in 0..n_occ {
                for a in 0..n_virt {
                    for b in 0..n_virt {
                        k_lr_a.slice_mut(s![i, a, j, b]).assign(
                            -1.0 * (&qtrans_oo.slice(s![.., i, j])
                                * &gamma_lr.dot(&qtrans_vv.slice(s![.., a, b])).sum()),
                        );
                        k_lr_a.slice_mut(s![i, a, j, b]).add_assign(
                            2.0 * (&qtrans_ov.slice(s![.., i, j])
                                * &gamma.dot(&qtrans_ov.slice(s![.., a, b])).sum()),
                        );
                    }
                }
            }
        }
    }
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
            omega[[i, a]] = orbe[virt_a] - orbe[occ_i];
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
            df[[i, a]] = f[virt_a] - f[occ_i];
        }
    }
    return omega;
}
