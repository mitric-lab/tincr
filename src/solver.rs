use ndarray::{ArrayView3, Array2, ArrayView2};

pub fn build_a_matrix(gamma_lr: ArrayView2<f64>, qtrans_ov: ArrayView3<f64>, qtrans_oo: ArrayView3<f64>, qtrans_vv: ArrayView3<f64> ) -> () {
    let n_occ: usize;
    let n_virt: usize;

    // K_lr_A = np.tensordot(qtrans_oo, np.tensordot(gamma_lr, qtrans_vv, axes=(1,0)),axes=(0,0))
    let mut k_lr_a: Array2<f64> = Array2::zeros([n_occ, n_virt, n_occ, n_occ]);
    for i in 0..n_occ {
        for a in 0..n_virt {
            for j in 0..n_occ {
                for b in 0..n_virt {
                    k_lr_a.slice_mut(s![i, a, j, b]).assign(
                        (&qtrans_oo.slice(s![.., i, j]) * &gamma_lr.dot(&qtrans_vv.slice(s![.., a, b])).sum()
                    );
                }
            }
        }
    }


}