use approx::AbsDiffEq;
use ndarray::{array, Array, Array1, Array2, ArrayView2};

// Mulliken Charges
pub fn mulliken(
    p: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    s: ArrayView2<f64>,
    orbs_per_atom: &[usize],
    n_atom: usize,
) -> (Array1<f64>, Array1<f64>) {
    let dp = &p - &p0;

    let mut q: Array1<f64> = Array1::<f64>::zeros(n_atom);
    let mut dq: Array1<f64> = Array1::<f64>::zeros(n_atom);

    // iterate over atoms A
    let mut mu = 0;
    // inside the loop
    for a in 0..n_atom {
        // iterate over orbitals on atom A
        for _mu_a in 0..orbs_per_atom[a] {
            let mut nu = 0;
            // iterate over atoms B
            for b in 0..n_atom {
                // iterate over orbitals on atom B
                for _nu_b in 0..orbs_per_atom[b] {
                    q[a] = q[a] + (&p[[mu, nu]] * &s[[mu, nu]]);
                    dq[a] = dq[a] + (&dp[[mu, nu]] * &s[[mu, nu]]);
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    (q, dq)
}
