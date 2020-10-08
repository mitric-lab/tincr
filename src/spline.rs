use ndarray::ArrayView1

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
fn splev_uniform(t: ArrayView1<f64>, n: usize, c: f64, k: usize, x: usize) -> (f64, usize) {
    let ier: i32 = 0;
    //  fetch tb and te, the boundaries of the approximation interval.
    let k1: usize = k + 1;
    let nk1: usize = n - k1;
    // first and last nodes
    let tb = t(k1);
    let te = t(nk1 + 1);
    if (x <= tb) {
        let arg = tb;
        let l = k1;
    } else if (x >= te) {
        arg = te;
        l = nk1;
    } else {
        let arg = x;
    }
    // find interval such that t(l) <= x < t(l+1)
    //dt = t(k1+2)-t(k1+1) ! uniform distance between knots
    //l = INT((x-t(1))/dt)+1

    // If l < k, we divide by zero because the interpolating points t(1:k) = 0.0
    if (l < k) {
        let l = k1;
    }
    // evaluate the non-zero b-splines at x.
    // call fpbspl(t,n,k,arg,l,h)
    //  find the value of s(x) at x
    sp = 0.d0;
    ll = l - k1;
    for j in 1..k1 {
        ll = ll + 1;
        // linear combination of b-splines
        let sp = sp + c(ll) * h(j);
    }
    let y = sp;
}