use ndarray::prelude::*;
use crate::AtomSlice;

/// Calculate Mulliken charges according to:
///       ⎲  ⎲  P   S
/// q  =  ⎳  ⎳   µν  νµ
///  A    µ∈A  ν
pub fn mulliken(
    p: ArrayView2<f64>,
    s: ArrayView2<f64>,
    atoms: &AtomSlice,
) -> Array1<f64> {
    let mut q: Array1<f64> = Array1::<f64>::zeros(atoms.len());
    let q_ao: Array1<f64> = s.dot(&p).diag().to_owned();

    let mut mu = 0;
    for (mut q_i, n_orbs_i) in q.iter_mut().zip(atoms.n_orbs.iter()) {
        for _ in 0..*n_orbs_i {
            *q_i += q_ao[mu];
            mu += 1;
        }
    }
    q
}
