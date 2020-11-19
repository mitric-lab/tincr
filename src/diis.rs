use crate::defaults;
use ndarray::*;
use ndarray_linalg::Solve;
use ndarray_stats::QuantileExt;

// Psi4NumPy implementation is used as a reference
pub fn diis(
    h: ArrayView2<f64>,
    fock_list: &[Array2<f64>],
    diis_error: &[Array1<f64>],
) -> Array2<f64> {

    let diis_count: usize = fock_list.len();
    // build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
    let mut b: Array2<f64> = Array2::zeros((diis_count + 1, diis_count + 1));
    b.slice_mut(s![diis_count, ..]).fill(-1.0);
    b.slice_mut(s![.., diis_count]).fill(-1.0);
    b[[diis_count, diis_count]] = 0.0;
    for (idx1, e1) in diis_error.iter().enumerate() {
        for (idx2, e2) in diis_error.iter().enumerate() {
            if idx2 <= idx1 {
                let val: f64 = e1.dot(e2);
                b[[idx1, idx2]] = val;
                b[[idx2, idx1]] = val;
            }
        }
    }

    // normalize
    // calculate the maximal element of the array slice
    let max: f64 = *b
        .slice(s![0..diis_count - 1, 0..diis_count - 1])
        .map(|x| x.abs())
        .max()
        .unwrap();
    b.slice_mut(s![0..diis_count - 1, 0..diis_count - 1]).map(|x| x / max);

    // build residual vector, [Pulay:1980:393], Eqn. 6, RHS
    let mut resid: Array1<f64> = Array1::zeros((diis_count + 1));
    resid[diis_count] = -1.0;

    // Solve Pulay equations, [Pulay:1980:393], Eqn. 6
    let ci: Array1<f64> = b.solve_into(resid).unwrap();

    // calculate new fock matrix as linear combination of previous fock matrices
    let mut fock: Array2<f64> = Array2::zeros(h.raw_dim());
    for (idx, coeff) in ci.slice(s![0..diis_count - 1]).iter().enumerate() {
        fock += &fock_list[idx].map(|x| x * *coeff);
    }

    return fock;
}
