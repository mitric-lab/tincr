use ndarray::prelude::*;
use crate::initialization::Atom;
use itertools::Itertools;

/// Calculate Mulliken charges according to:
///       ⎲  ⎲  P   S
/// q  =  ⎳  ⎳   µν  νµ
///  A    µ∈A  ν
pub fn mulliken(
    p: ArrayView2<f64>,
    s: ArrayView2<f64>,
    atoms: &[Atom],
) -> Array1<f64> {
    let mut q: Array1<f64> = Array1::<f64>::zeros(atoms.len());
    let q_ao: Array1<f64> = s.dot(&p).diag().to_owned();

    let mut mu = 0;
    for (mut q_i, atomi) in q.iter_mut().zip(atoms.iter()) {
        for _ in 0..atomi.n_orbs {
            *q_i += q_ao[mu];
            mu += 1;
        }
    }
    q
}
