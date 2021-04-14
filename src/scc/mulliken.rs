use ndarray::prelude::*;
use crate::initialization::Atom;

/// Calculate the Mulliken charges from the density matrix and the overlap integrals
pub fn mulliken(
    p: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    s: ArrayView2<f64>,
    atoms: &[Atom],
    n_atoms: usize,
) -> (Array1<f64>, Array1<f64>) {
    let dp = &p - &p0;

    let mut q: Array1<f64> = Array1::<f64>::zeros(n_atoms);
    let mut dq: Array1<f64> = Array1::<f64>::zeros(n_atoms);

    // iterate over atoms A
    let mut mu = 0;
    // inside the loop
    for (i, atomi) in atoms.iter().enumerate() {
        // iterate over orbitals on atom A
        for _ in 0..atomi.n_orbs {
            let mut nu = 0;
            // iterate over atoms B
            for (_, atomj) in atoms.iter().enumerate() {
                // iterate over orbitals on atom B
                for _ in 0..atomj.n_orbs {
                    q[i] = q[i] + (&p[[mu, nu]] * &s[[mu, nu]]);
                    dq[i] = dq[i] + (&dp[[mu, nu]] * &s[[mu, nu]]);
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    (q, dq)
}
