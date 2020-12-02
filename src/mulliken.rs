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

#[test]
fn mulliken_charges() {
    use crate::molecule::*;
    //use crate::scc_routine::*;
    let p0: Array2<f64> = array![
        [2., 0., 0., 0., 0., 0.],
        [0., 1.3333333333333333, 0., 0., 0., 0.],
        [0., 0., 1.3333333333333333, 0., 0., 0.],
        [0., 0., 0., 1.3333333333333333, 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]
    ];
    let p: Array2<f64> = array![
        [
            1.6672853597938484,
            0.2116144144027609,
            0.3390751794256264,
            0.282623268095985,
            0.162846907840508,
            0.1628473832190555
        ],
        [
            0.2116144144027609,
            1.776595917657724,
            -0.357966065395007,
            0.1368169475860144,
            -0.0285685423132669,
            -0.3224914900258041
        ],
        [
            0.3390751794256264,
            -0.357966065395007,
            1.426421833339374,
            0.2192252885141318,
            -0.0457761047995666,
            -0.5167363487612707
        ],
        [
            0.282623268095985,
            0.1368169475860144,
            0.2192252885141318,
            1.020291038750183,
            -0.6269841692414225,
            0.1581194812138258
        ],
        [
            0.162846907840508,
            -0.0285685423132669,
            -0.0457761047995666,
            -0.6269841692414225,
            0.4577294196704165,
            -0.0942187608833704
        ],
        [
            0.1628473832190555,
            -0.3224914900258041,
            -0.5167363487612707,
            0.1581194812138258,
            -0.0942187608833704,
            0.4577279753425148
        ]
    ];
    let s: Array2<f64> = array![
        [1., 0., 0., 0., 0.3074918525690681, 0.3074937992389065],
        [0., 1., 0., 0., 0., -0.1987769748092704],
        [0., 0., 1., 0., 0., -0.3185054221819456],
        [0., 0., 0., 1., -0.3982160222204482, 0.1327383036929333],
        [
            0.3074918525690681,
            0.,
            0.,
            -0.3982160222204482,
            1.,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.
        ]
    ];
    let orbs_per_atom: Vec<usize> = vec![4, 1, 1];
    let (q, dq): (Array1<f64>, Array1<f64>) = mulliken(
        p.view(),
        p0.view(),
        s.view(),
        &orbs_per_atom[..],
        orbs_per_atom.len(),
    );
    let q_ref: Array1<f64> = array![6.4900936727759640, 0.7549533634060839, 0.7549529638179489];
    let dq_ref: Array1<f64> = array![0.4900936727759634, -0.2450466365939161, -0.2450470361820512];
    assert!(q.abs_diff_eq(&q_ref, 1e-15));
    assert!(dq.abs_diff_eq(&dq_ref, 1e-15));
}
