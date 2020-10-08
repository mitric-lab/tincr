use ndarray::ArrayView1;

/// This function assumes that the knots t are spaced uniformly
/// so that the interval t_i <= x < t_(i+1) can be found without iterating over all knots
/// subroutine splev evaluates a spline s(x) of degree k, given in its b-spline representation,
/// at position x
///
///  calling sequence:
///     call splev_uniform(t,n,c,k,x,y,ier)
///
///  input parameters:
///    t    : array,length n, which contains the position of the knots.
///    n    : integer, giving the total number of knots of s(x).
///    c    : array,length n, which contains the b-spline coefficients.
///    k    : integer, giving the degree of s(x).
///    x    : contains the point where s(x) must
///           be evaluated.
///
///  output parameter:
///    y    : gives the value of s(x)
///
///    ier  : error flag
///      ier = 0 : normal return
fn splev_uniform(
    t: ArrayView1<f64>,
    n: usize,
    c: ArrayView1<f64>,
    k: usize,
    x: usize,
) -> (f64, u8) {
    let ier: u8 = 0;
    // fetch tb and te, the boundaries of the approximation interval.
    // let k1: usize = k + 1;
    let nk: usize = n - k;
    // first and last nodes
    let tb: f64 = t[k1];
    let te: f64 = t[nk];
    let arg: f64;
    let mut l: usize;
    if (x as f64 <= tb) {
        arg = tb;
        l = k;
    } else if (x as f64 >= te) {
        arg = te;
        l = nk;
    } else {
        arg = x as f64;
    }
    // find interval such that t(l) <= x < t(l+1)
    let dt: f64 = t[k + 2] - t[k + 1]; // uniform distance between knots
    l = ((x - t[0]) / dt) as usize + 1;
    // If l < k, we divide by zero because the interpolating points t(1:k) = 0.0
    if (l < k) {
        l = k;
    }
    // evaluate the non-zero b-splines at x.
    let h: [f64; 6] = fpbspl(t, n, k, arg, l);
    //  find the value of s(x) at x
    let mut sp: f64 = 0.0;
    let mut ll: usize = l - k;
    for j in 1..k1 {
        ll = ll + 1;
        // linear combination of b-splines
        sp = sp + (c[ll] * h[j]);
    }
    let y: f64 = sp;
}

/// Function fpbspl evaluates the (k+1) non-zero b-splines of
//  degree k at t[l] <= x < t[l+1] using the stable recurrence
//  relation of de boor and cox.
fn fpbspl(t: ArrayView1<f64>, n: usize, k: usize, x: f64, l: usize) -> ([f64; 6]) {
    let one: f64 = 1.0;
    let mut h: [f64; 6] = [one, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut hh: [f64; 5] = [0.0; 5];
    for j in 0..k {
        for i in 0..j {
            hh[i] = h[i];
        }
        h[0] = 0.0;
        for i in 0..j {
            let li: usize = l + i;
            let lj: usize = li - j;
            let f: f64 = hh[i] / (t[li] - t[lj]);
            h[i] = h[i] + f * (t[li] - x);
            h[i + 1] = f * (x - t[lj]);
        }
    }
    return h;
}
