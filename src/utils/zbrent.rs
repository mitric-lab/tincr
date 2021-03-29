use std::f64::consts::PI;

/// In the original DFTBaby code the Fermi energy is searched using the bisection
/// method, as it is necessary to find the root of (sum_a fa - n_elec).
/// Since the bisection method can be slow, we use the Brent's method.
/// Using Brent's method, find the root of a function known to lie between `x1 ` and
/// `x2`. The root will be refined until its accuracy is `tol`
///
/// The code is based on:
/// Numerical Recipes in C: The Art of Scientific Computing. W. H. Press,
/// S. A. Teukolsky, W. T. Vetterling, B. P. Flannery. Cambridge University Press 1992
pub fn zbrent<F: Fn(f64) -> f64>(func: F, x1: f64, x2: f64, tol: f64, maxiter: usize) -> f64 {
    let eps: f64 = 2.220446049250313e-016_f64.sqrt();
    let iter: usize;
    let mut a: f64 = x1;
    let mut b: f64 = x2;
    let mut c: f64 = x2;
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut min1: f64;
    let mut min2: f64;

    let mut root: f64 = 0.0;

    let mut fa: f64 = func(a);
    let mut fb: f64 = func(b);
    let mut fc: f64;
    let mut p: f64;
    let mut q: f64;
    let mut r: f64;
    let mut s: f64;
    let mut tol1: f64;
    let mut xm: f64;

    assert!(
        (fa > 0.0 && fb < 0.0) || (fa < 0.0 && fb > 0.0),
        "Root must be bracketed in zbrent"
    );
    fc = fb;
    'main_loop: for iter in 0..maxiter {
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            // rename a, b, c and adjust the bounding interval d
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        // convergence check
        tol1 = 2.0 * eps * b.abs() + 0.5 * tol;
        xm = 0.5 * (c - b);

        if xm.abs() <= tol1 || fb == 0.0 {
            root = b;
            break 'main_loop;
        }

        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // attempt inverse quadratic interpolation
            s = fb / fa;
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) {
                // check whether in bounds
                q = -q;
            }
            p = p.abs();
            min1 = 3.0 * xm * q - (tol1 * q).abs();
            min2 = (e * q).abs();
            if (2.0 * p) < min1.min(min2) {
                // accept interpolation
                e = d;
                d = p / q;
            } else {
                // interpolation failed, use bisection.
                d = xm;
                e = d;
            }
        } else {
            // bounds decreasing to slowly, use bisection.
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b = b + d;
        } else if xm > 0.0 {
            b = b + tol1;
        } else {
            b = b - tol1;
        }
        fb = func(b);
    }
    return root;
}