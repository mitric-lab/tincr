use tincr::utils::zbrent::zbrent;

/// The tests are taken from John Burkardts Python version of the Brent's method
/// See https://people.sc.fsu.edu/~jburkardt/py_src/brent/zero.py
#[test]
fn zero_test() {
    let eps: f64 = 2.220446049250313e-016;
    let maxiter: usize = 100;
    let t: f64 = 10.0 * (eps).sqrt();

    // F_01 evaluates sin ( x ) - x / 2.
    fn f_01(x: f64) -> f64 {
        (x).sin() - 0.5 * x
    }
    let a: f64 = 1.0;
    let b: f64 = 2.0;
    let x: f64 = zbrent(f_01, a, b, t, maxiter);
    assert!(f_01(x) <= t);

    // F_02 evaluates 2*x-exp(-x).
    fn f_02(x: f64) -> f64 {
        2.0 * x - (-x).exp()
    }
    let a: f64 = 0.0;
    let b: f64 = 1.0;
    let x: f64 = zbrent(f_02, a, b, t, maxiter);
    assert!(f_02(x) <= t);

    // F_03 evaluates x*exp(-x).
    fn f_03(x: f64) -> f64 {
        x * (-x).exp()
    }
    let a: f64 = -1.0;
    let b: f64 = 0.5;
    let x: f64 = zbrent(f_03, a, b, t, maxiter);
    assert!(f_03(x) <= t);

    // F_04 evaluates exp(x) - 1 / (100*x*x).
    fn f_04(x: f64) -> f64 {
        (x).exp() - 1.0 / 100.0 / x / x
    }
    let a: f64 = 0.0001;
    let b: f64 = 20.0;
    let x: f64 = zbrent(f_04, a, b, t, maxiter);
    assert!(f_04(x) <= t);

    // F_05 evaluates (x+3)*(x-1)*(x-1).
    fn f_05(x: f64) -> f64 {
        (x + 3.0) * (x - 1.0) * (x - 1.0)
    }
    let a: f64 = -5.0;
    let b: f64 = 2.0;
    let x: f64 = zbrent(f_05, a, b, t, maxiter);
    assert!(f_05(x) <= t);
}
