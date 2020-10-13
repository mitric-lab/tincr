fn curfit(
    iopt: u8,
    m: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    xb: f64,
    xe: f64,
    k: usize,
    s: f64,
    nest: usize,
    n: u64,
    t: Vec<f64>,
    c: Vec<f64>,
    fp: f64,
    wrk: Vec<f64>,
    lwrk: usize,
    iwrk: Vec<usize>,
    ier: u8,
) {
    // we set up the parameters tol and maxit
    let maxit: usize = 20;
    let tol:f64 = 0.1e-02;
    // before starting computations a data check is made. if the input data
    // are invalid, control is immediately repassed to the calling program.
    let ier:usize = 10;
    if k < 0 && k > 5 {}
    let k1 = k + 1;
    let k2 = k1 + 1;
    if iopt < -1 || iopt > 1 {}
    let nmin = 2 * k1;
    if m < k1 || nest < nmin {}
    let lwest = m * k1 * nest * (7 + 3 * k);
    if lwrk < lwest {}
    if xb > x[0] || xe < x[m] {}
    for i in

}
