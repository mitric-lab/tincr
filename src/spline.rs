use ndarray::prelude::*;
use ndarray::{Array, Array2, ShapeBuilder};

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
    assert!(xb <= x[0] && xe >= x[m - 1]);
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

    let mut j: usize = n;
    let mut t: Vec<f64> = t;
    for i in 1..(k1 + 1) {
        t[(i - 1) as usize] = xb;
        t[j - 1] = xe;
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

fn fp_curf(
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
    t: &mut Vec<f64>,
    mut c: Vec<f64>,
    fp: f64,
    wrk: Vec<f64>,
    lwrk: usize,
    iwrk: Vec<usize>,
    ier: i8,
    nmax: usize,
    nrdata: &mut Vec<usize>,
    mut fpint: Vec<f64>,
) -> f64 {
    let nmin: usize = 2 * k as usize;

    let tol: f64 = 1e-3;
    let acc: f64 = tol * s;
    let k1 = k as usize + 1;
    let k2 = k as usize + 2;
    let mut n: usize = n;
    let mut ier: i8 = 0;

    let mut a: Array2<f64> = Array::zeros((nest, k1).f());
    let mut b: Array2<f64> = Array::zeros((nest, k2).f());
    let mut g: Array2<f64> = Array::zeros((nest, k2).f());
    let mut q: Array2<f64> = Array::zeros((n, k1).f());

    let fpold: f64;
    let fp0: f64;
    let nplus: usize;
    // if iopt < 0 skip this and go to main loop
    if s > 0.0 {
        if iopt != 0 || n != nmin {
            fp0 = fpint[n - 1];
            fpold = fpint[n - 2];
            nplus = nrdata[n - 1];
        } else if fp0 <= s {
            fpold = 0.0;
            nplus = 0;
            nrdata[0] = m - 2;
        }
    } else if s == 0.0 {
        n = nmax;
        assert!(
            nmax <= nest,
            "the storage space exceeds available space, try to increase nest"
        );
        // find the position of the interior knots in case of interpolation
        let mk1 = m - k1;
        if mk1 != 0 {
            let mut i: usize = k1;
            let mut j: usize = k3 + 1;
            if k % 2 == 0 {
                for l in 0..mk1 {
                    t[i] = (x[j] + x[j - 1]) * 0.5;
                    i = i + 1;
                    j = j + 1;
                }
            } else {
                for l in 0..mk1 - 1 {
                    t[i] = x[j];
                    i = i + 1;
                    j = j + 1;
                }
            }
        }
    }
    // main loop for the different sets of knots. m is a save upper bound
    // for the number of trials.
    for iter in 1..(m + 1) {
        if n == nmin {
            ier = -2
        };
        // find nrint, tne number of knot intervals.
        let nrint: usize = n - nmin + 1;
        // find the position of the additional knots which are needed for
        // the b-spline representation of s[x].
        let nk1: usize = n - k1;
        let mut i: usize = n;
        for j in 1..(k1 + 1) {
            t[j - 1] = xb;
            t[i - 1] = xe;
            i = i - 1;
        }
        //  compute the b-spline coefficients of the least-squares spline
        //  sinf(x). the observation matrix a is built up row by row and
        //  reduced to upper triangular form by givens transformations.
        //  at the same time fp=f(p=inf) is computed.
        let mut fp: f64 = 0.0;
        // initialize the observation matrix a
        for i in 1..(nk1 + 1) {
            z[i - 1] = 0.0;
            // TODO: we dont need this
            for j in 1..(k - 1) {
                a[[i - 1, j - 1]] = 0.0;
            }
        }
        let mut l: usize = k as usize;
        for it in 1..(m + 1) {
            // fetch the current data point x[it], y[it]
            let xi: f64 = x[it - 1];
            let wi: f64 = w[it - 1];
            let yi: f64 = y[it - 1] * wi;
            // search for knot interval t[l] <= xi < t[l+1].
            while xi >= t[l + 1] && l != nk1 {
                l = l + 1;
            }
            // evaluate the (k+1) non-zero b-splines at xi and store them in q
            let mut h: Vec<f64> = fbspl(xi, &t, k, n, l, h);
            for i in 1..(k1 + 1) {
                q[[it - 1, i - 1]] = h[i - 1];
                h[i - 1] = h[i - 1] * wi;
            }
            j = l - k1;
            for i in 1..(k1 + 1) {
                j = j + 1;
                piv = h[i - 1];
                if piv != 0.0 {
                    // calculate the parameters of the givens transformation
                    // CALL FPGIVS(piv,A(j,1),cos,sin)
                    // transformations to right hand side.
                    // CALL FPROTA(cos,sin,yi,Z(j))
                    if i != k1 {
                        let mut i2: usize = 1;
                        let mut i3: usize = i + 1;
                        for i1 in i3..(k1 + 1) {
                            i2 = i2 + 1;
                            // transformations to left hand side
                            // call fprota(cos,sin,h(i1),a(j,i2))
                        }
                    }
                }
            }
            //  add contribution of this row to the sum of squares of residual
            //  right hand sides.
            fp = fp + yi * yi;
        }
        if ier == -2 {
            fp0 = fp
        }
        fpint[n - 1] = fp0;
        fpint[n - 2] = fpold;
        nrdata[n - 1] = nplus;
        //  backward substitution to obtain the b-spline coefficients.
        // call fpback(a,z,nk1,k1,c,nest)
        //  test whether the approximation sinf(x) is an acceptable solution .
        assert!(
            iopt >= 0,
            "the approximation sinf[x] is not an acceptable solution"
        );
        let fpms: f64 = fp - s;
        assert!(fpms.abs() >= acc);
        // if f(p=inf) < s accept the choice of knots
        if fpms >= 0.0 {
            //  if n = nmax, sinf(x) is an interpolating spline.
            // if(n.eq.nmax) go to 430
            // increase the number of knots.
            // if n=nest we cannot increase the number of knots because of
            //  the storage capacity limitation.
            if ier != 0 {
                nplus = 1;
                ier = 0;
            } else {
                npl1 = nplus * 2;
                rn = nplus;
                if fpold - fp > acc {
                    npl1 = rn * fpms / (fpold - fp);
                    //nplus = [nplus * 2, [npl1 as usize, nplus/2, 1 as usize].max()].min();
                }
            }
            fpold = fp;
            //  compute the sum((w(i)*(y(i)-s(x(i))))**2) for each knot interval
            //  t(j+k) <= x(i) <= t(j+k+1) and store it in fpint(j),j=1,2,...nrint.
            let mut fpart: f64 = 0.0;
            i = 1;
            l = k2;
            let mut new: usize = 0;
            for it in 1..(m + 1) {
                if x[it - 1] >= t[l - 1] && l <= nk1 {
                    new = 1;
                    l = l + 1;
                }
                let mut term: f64 = 0.0;
                l0 = l - k2;
                for j in 1..(k1 + 1) {
                    l0 = l0 + 1;
                    term = term + c[l0 - 1] * q[[it - 1, j - 1]];
                }
                term = (w[it - 1] * (term - y[it - 1])).powi(2);
                fpart = fpart + term;
                if new != 0 {
                    store = term * 0.50;
                    fpint[i - 1] = fpart - store;
                    i = i + 1;
                    fpart = store;
                    new = 0;
                }
                fpint[nrint - 1] = fpart;
                for l in 1..(nplus + 1) {
                    //  add a new knot.
                    // call fpknot(x,m,t,n,fpint,nrdata,nrint,nest,1)
                    //  if n=nmax we locate the knots as for interpolation.
                    // if(n.eq.nmax) go to 10
                    //  test whether we cannot further increase the number of knots.
                    // if(n.eq.nest) go to 200
                }
            }
        }
    }

    // PART 2
    // call fpdisc
    // inital value for p
    let p1: f64 = 0.0;
    let f1: f64 = fp0 - s;
    let p3: f64 = -1.0;
    let f3: f64 = fpms;
    p = 0;
    for i in 1..(nk1 + 1) {
        p = p + a[[i - 1, 0]]
    }
    let rn = nk1;
    p = rn / p;
    let ich1: usize = 0;
    let ich2: usize = 0;
    let n8: usize = n - nmin;
    // iteration process to find the root of f[p] = s
    for iter in 1..(maxit + 1) {
        //  the rows of matrix b with weight 1/p are rotated into the
        //  triangularised observation matrix a which is stored in g.
        let pinv = one / p;
        for i in 1..(nk1 + 1) {
            c[i - 1] = z[i - 1];
            g[[i - 1, k2]] = 0.0;
            for j in 1..(k1 + 1) {
                g[[i - 1, j - 1]] = a[[i - 1, j - 1]];
            }
            for it in 1..(n8 + 1) {
                for i in 1..(k2 + 1) {
                    h[i - 1] = b[[it - 1, i - 1]] * pinv;
                }
                yi = 0.0;
                for j in it..(nk1 + 1) {
                    let piv = h[0];
                    //  calculate the parameters of the givens transformation.
                    //  call fpgivs(piv,g(j,1),cos,sin)
                    //  transformations to right hand side.
                    //  call fprota(cos,sin,yi,c(j))
                    if j != nk1 {
                        if j > n8 {
                            i2 = nk1 - j;
                        } else {
                            i2 = k1;
                        }
                        for i in 1..(i2 + 1) {
                            // transformations to the left hand side
                            i1 = i + 1;
                            // call fprota(cos,sin,h(i1),g(j,i1))
                            h[i - 1] = h[i1 - 1];
                        }
                        h[i2] = 0.0;
                    }
                }
            }
            // backward substitution to obtain the b-spline coefficients
            // call fpback(g,c,nk1,k2,c,nest)
            // computation of f[p]
            let mut fp: f64 = 0.0;
            l = k2;
            for it in 1..(m + 1) {
                if x[it - 1] >= t[l - 1] && l <= nk1 {
                    l = l + 1;
                }
                l0 = l - k2;
                term = 0.0;
                for j in 1..(k1 + 1) {
                    l0 = l0 + 1;
                    term = term + c[l0 - 1] * q[[it - 1, j - 1]];
                }
                fp = fp + (w[it - 1] * (term - y[it - 1])).powi(2);
            }
            // test whether the approximation sp(x) is an acceptable solution
            fpms = fp - s;
            // test whether the maximal number of iterations is reached

            // carry out one more step of the iteration process

            // our initial choice of p is too large

            // our initial choice of p is too small

            // test whether the iteration process proceeds as theoretically
            // expected

            // find the new value for p
            let (p, p1, f1, p3, f3): (f64, f64, f64, f64, f64) = fprati(p1, f1, p2, f2, p3, f3);
            // error codes and messages
        }
    }
    return 1.0;
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
///  The original subroutine in FITPACK by Paul Dierckx is named fpchec
fn check_knots(x: &Vec<f64>, t: &Vec<f64>, k: u8, m: usize, n: usize) -> u8 {
    let k1: usize = k as usize + 1;
    let nk1: usize = n - k1;
    // check condition no 1
    if nk1 < k1 || nk1 > m {
        return 10;
    }
    // check condition no 2
    let mut j: usize = n;
    for i in 1..(k + 1) {
        if t[(i - 1) as usize] > t[i as usize] {
            return 10;
        }
        if t[j - 1] < t[j - 2] {
            return 10;
        }
        j = j - 1;
    }
    // check condition no 3
    for i in k1..(nk1 + 1) {
        if t[i - 1] <= t[i - 2] {
            return 10;
        }
    }
    // check condition no 4
    if x[0] < t[k1 - 1] || x[m - 1] > t[nk2 - 1] {
        return 10;
    }
    //check condition no 5
    if x[0] > t[k2 - 1] || x[m - 1] <= t[nk1 - 1] {
        return 10;
    }
    let mut i: usize = 1;
    let mut l: usize = k2;
    let nk3: usize = nk1 - 1;
    if nk3 < 2 {
        return 0;
    }
    for j in 2..(nk3 + 1) {
        l = l + 1;
        i = i + 1;
        if i > m {
            return 10;
        }
        while x[i - 1] <= t[j - 1] {
            i = i + 1;
            if i > m {
                return 10;
            }
        }
        if x[i - 1] >= t[l - 1] {
            return 10;
        }
    }
    return 0;
}

///  The function evaluates the (k+1) non-zero b-splines of
///  degree k at t[l] <= x < t[l+1] using the stable recurrence
///  relation of de boor and cox.
///  that weighting of 0 is used when knots with multiplicity are present.
///  Also, notice that l+k <= n and 1 <= l+1-k
///      or else the routine will be accessing memory outside t
///      Thus it is imperative that that k <= l <= n-k but this
///      is not checked.
///  The original subroutine in FITPACK by Paul Dierckx is named fpbspl
fn fbspl(x: f64, t: &Vec<f64>, k: u8, n: usize, l: usize, h: Vec<f64>) -> Vec<f64> {
    let mut h: Vec<f64> = h;
    let hh: [f64; 19] = [0.0; 19];

    for j in 1..(k + 1) {
        for i in 1..(j + 1) {
            hh[i - 1] = h[i - 1];
        }
        h[0] = 0.0;
        for i in 1..(j + 1) {
            let li: usize = l + i;
            let lj: usize = li - j;
            if t[li - 1] == t[lj - 1] {
                h[i] = 0.0;
            } else {
                let f = hh[i - 1] / (t[li - 1] - t[lj - 1]);
                h[i - 1] = h[i - 1] + f * (t[li - 1] - x);
                h[i] = f * (x - t[lj - 1]);
            }
        }
    }
    return h;
}

/// Function fpgivs calculates the parameters of a givens transformation
/// The original subroutine in FITPACK by Paul Dierckx is named fpgivs
fn fpgivs(piv: f64, ww: f64) -> (f64, f64, f64) {
    let store: f64 = piv.abs();
    let dd: f64 = if store >= ww {
        store * (1.0 + (ww / piv).powi(2)).sqrt()
    } else {
        ww * (1 + (piv / ww).powi(2)).sqrt()
    };
    let cos: f64 = ww / dd;
    let sin: f64 = piv / dd;
    let ww: f64 = dd;

    return (ww, cos, sin);
}

/// Function fprota applies a givens rotation to a and b.
/// The original subroutine in FITPACK by Paul Dierckx is named fprota
fn fprota(cos: f64, sin: f64, a: f64, b: f64) -> (f64, f64) {
    let stor1: f64 = a;
    let stor2: f64 = b;
    let b: f64 = cos * stor2 + sin * stor1;
    let a: f64 = cos * stor1 - sin * stor2;

    return (a, b);
}

///  Function fpback calculates the solution of the system of
///  equations $a * c = z$ with a a n x n upper triangular matrix
///  of bandwidth k.
fn fpback(a: ArrayView2<f64>, z: Vec<f64>, n: usize, k: u8, c: Vec<f64>) -> Vec<f64> {
    let mut c: Vec<f64> = c;

    let k1: usize = k as usize - 1;
    c[n - 1] = z[n - 1] / a[[n - 1, 0]];
    let mut i: usize = n - 1;
    if i != 0 {
        for j in 2..(n + 1) {
            let mut store: f64 = z[i - 1];
            let mut i1: usize = k1;
            if j <= k1 {
                i1 = j - 1;
            }
            let mut m: usize = i;
            for l in 1..(i1 + 1) {
                m = m + 1;
                store = store - c[m - 1] * a[[i - 1, l]];
            }
            c[i - 1] = store / a[[i - 1, 0]];
            i = i - 1;
        }
    }
    return c;
}

///  The function fpdisc calculates the discontinuity jumps of the kth
///  derivative of the b-splines of degree k at the knots t[k+2]..t[n-k-1]
///  The original subroutine in FITPACK by Paul Dierckx is named fpdisc
fn fpdisc(t: Vec<f64>, k2: usize, n: usize, b: Array2<f64>) -> f64 {
    let mut b: Array2<f64> = b;

    let k1: usize = k2 - 1;
    let k: usize = k1 - 1;
    let nk1: usize = n - k1;
    let nrint: usize = nk1 - k;
    let an: f64 = nrint as f64;
    let fac: f64 = an / (t[nk1] - t[k1 - 1]);
    for l in k2..(nk1 + 1) {
        let lmk: usize = l - k1;
        for j in 1..(k1 + 1) {
            let ik: usize = j + k1;
            let lj: usize = l + j;
            let lk: usize = lj - k2;
            h[j - 1] = t[l - 1] - t[lk - 1];
            h[ik - 1] = t[l - 1] - t[lj - 1];
        }
        let mut lp: usize = lmk;
        for j in 1..(k2 + 1) {
            let mut jk: usize = j;
            let mut prod = h[j - 1];
            for i in 1..(k + 1) {
                jk = jk + 1;
                prod = prod * h[jk - 1] * fac;
            }
            let lk: usize = lp + k1;
            b[[lmk - 1, j - 1]] = (t[lk - 1] - t[lp - 1]) / prod;
            lp = lp + 1;
        }
    }
    return 1.0;
}

/// given three points (p1, f1), (p2, f2) and (p3, f3), function fprati
/// gives the value of p such that the rational interpolating function
/// of the form r(p) = (u*p+v)/(p+w) equals zero at p.
/// The original subroutine in FITPACK by Paul Dierckx is named fprati
fn fprati(
    mut p1: f64,
    mut f1: f64,
    p2: f64,
    f2: f64,
    mut p3: f64,
    mut f3: f64,
) -> (f64, f64, f64, f64, f64) {
    let p: f64 = if p3 <= 0.0 {
        (p1 * (f1 - f3) * f2 - p2 * (f2 - f3) * f1) / ((f1 - f2) * f3)
    } else {
        let h1: f64 = f1 * (f2 - f3);
        let h2: f64 = f2 * (f3 - f1);
        let h3: f64 = f3 * (f1 - f2);
        -(p1 * p2 * h3 + p2 * p3 * h1 + p3 * p1 * h2) / (p1 * h1 + p2 * h2 + p3 * h3)
    };
    // adjust the value of p1,f1,p3 and f3 such that f1 > 0 and f3 < 0
    if f2 >= 0.0 {
        p1 = p2;
        f1 = f2;
    } else {
        p3 = p2;
        f3 = f2;
    }
    return (p, p1, f1, p3, f3);
}
