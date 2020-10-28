use peroxide::structure::matrix::r_matrix;

pub fn curfit(
    iopt: i8,
    m: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    xb: f64,
    xe: f64,
    k: u8,
    s: f64,
    nest: usize,
    n: usize,
    t: Vec<f64>,
    c: Vec<f64>,
    fp: f64,
    wrk: Vec<f64>,
    lwrk: usize,
    iwrk: Vec<usize>,
    ier: i8,
) -> Option<f64> {
    let maxit: usize = 20;
    let tol: f64 = 1e-3;
    assert!(k >= 0 && k <= 5, "the degree k must be within 1 and 5");
    let k1: u8 = k + 1;
    let k2: u8 = k1 + 1;
    assert!(iopt >= -1 && iopt <= 1, "iopt must be within -1 and 1");
    let nmin: usize = 2 * k1 as usize;
    assert!(
        m >= k1 as usize && nest >= nmin,
        "number of data points (m) must be greater than k"
    );
    let lwest: usize = m * k1 as usize + nest * (7 + 3 * k as usize);
    assert!(lwrk >= lwest, "lwrk is to small");
    assert!(xb <= x[1] && xe >= x[m]);
    //assert!(x.is_sorted(), "x has to be sorted in ascending order");
    if iopt >= 0 {
        assert!(s >= 0.0, "smoothing factor cannot be negative");
        if s == 0 && nest <= m + k1 {
            panic!()
        }
    }
    assert!(
        n >= nmin && ng <= nest,
        "total number of knots must be within nmin and nest"
    );

    let mut j: usize = n - 1;
    let mut t: Vec<f64> = t;
    for i in 0..k1 {
        t[i as usize] = xb;
        t[j] = xe;
        j = j - 1;
    }

    // verify the number and position of knots
    let mut ier: u8 = check_knots(&x, &t, k, m, n);
    assert_eq!(ier, 0, "The knots dont fullfill all five conditions");

    // we partition the working space and determine the spline approximation
    let ifp: usize = 0;
    let iz: usize = ifp + nest;
    let ia: usize = iz + nest;
    let ib: usize = ia + nest * k1;
    let ig: usize = ib + nest * k2;
    let iq: usize = ig + nest * k2;


    return Some(1.0);
}

///  function check_knots verifies the number and the position of the knots
///  t[j],j=0,1,2,...,n-1 of a spline of degree k, in relation to the number
///  and the position of the data points x[i],i=0,1,2,...,m-1. if all of the
///  following conditions are fulfilled, the function will return zero.
///  if one of the conditions is violated the function returns 1.
///      1) k+1 <= n-k-1 <= m
///      2) t[0] <= t[1] <= ... <= t[k]
///         t[n-k] <= t[n-k+1] <= ... <= t[n-1]
///      3) t[k] < t[k+1] < ... < t[n-k-1]
///      4) t[k] <= x[i] <= t[n-k-1]
///      5) the conditions specified by schoenberg and whitney must hold
///         for at least one subset of data points, i.e. there must be a
///         subset of data points y(j) such that
///             t[j] < y[j] < t[j+k+1], j=0,1,2,...,n-k-1
///
///  The original subroutine in FITPACK by Paul Dierckx is named fpcheck
fn check_knots(x: &Vec<f64>, t: &Vec<f64>, k: u8, m: usize, n: usize) -> u8 {
    let k1: usize = k as usize + 1;
    let nk1: usize = n - k1;
    // check condition no 1
    if nk1 < k1 || nk1 > m {
        return 10;
    }
    // check condition no 2
    let mut j: usize = n - 1;
    for i in 0..k {
        if t[i as usize] > t[i as usize + 1] {
            return 10;
        }
        if t[j] < t[j - 1] {
            return 10;
        }
        j = j - 1;
    }
    // check condition no 3
    for i in k1..nk1 {
        if t[i] <= t[i - 1] {
            return 10;
        }
    }
    // check condition no 4
    if x[0] < t[k as usize] || x[m - 1] > t[nk1] {
        return 10;
    }
    //check condition no 5
    if x[0] > t[k1] || x[m - 1] <= t[nk1 - 1] {
        return 10;
    }
    let mut i: usize = 0;
    let mut l: usize = k1;
    let nk3: usize = nk1 - 1;
    if nk3 < 2 {
        return 0;
    }
    for j in 1..nk3 - 1 {
        l = l + 1;
        i = i + 1;
        if i > m {
            return 10;
        }
        while x[i] <= t[j] {
            i = i + 1;
            if i > m {
                return 10;
            }
        }
        if x[i] >= t[l] {
            return 10;
        }
    }
    return 0;
}
