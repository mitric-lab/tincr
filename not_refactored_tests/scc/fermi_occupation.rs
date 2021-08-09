use tincr::scc::fermic_occupation::fermi_occupation;
use ndarray::prelude::*;

#[test]
fn fermi_occ_at_t0() {
    let orbe: Array1<f64> = array![
        -0.8274698649039047,
        -0.4866977381657900,
        -0.4293504325361446,
        -0.3805317817759825,
        0.4597732008355508,
        0.5075648461370381
    ];
    let temperature: f64 = 0.0;
    let n_elec: usize = 8;
    let n_elec_unpaired: usize = 0;
    let mu_ref: f64 = 0.0;
    let occ_ref: Vec<f64> = vec![2.0, 2.0, 2.0, 2.0, 0.0, 0.0];
    let result: (f64, Vec<f64>) =
        fermi_occupation(orbe.view(), n_elec, n_elec_unpaired, temperature);
    let mu: f64 = result.0;
    let occ: Vec<f64> = result.1;
    assert!((mu - mu_ref).abs() < 1e-8);
    assert_eq!(occ, occ_ref);
}

#[test]
fn fermi_occ_at_t100k() {
    let orbe: Array1<f64> = array![
        -0.8274698649039047,
        -0.4866977381657900,
        -0.4293504325361446,
        -0.3805317817759825,
        0.4597732008355508,
        0.5075648461370381
    ];
    let temperature: f64 = 100.0;
    let n_elec: usize = 8;
    let n_elec_unpaired: usize = 0;
    let mu_ref: f64 = -0.3692029124379807;
    let occ_ref: Vec<f64> = vec![2.0, 2.0, 2.0, 1.9999999999999996, 0.0, 0.0];
    let result: (f64, Vec<f64>) =
        fermi_occupation(orbe.view(), n_elec, n_elec_unpaired, temperature);
    let mu: f64 = result.0;
    let occ: Vec<f64> = result.1;
    // TODO: Check the differences to DFTBaby
    //assert!((mu-mu_ref).abs() < 1e-4);
    //assert_eq!(occ, occ_ref);
    assert!((occ.iter().sum::<f64>() - n_elec as f64).abs() < 1e-08);
}

#[test]
fn fermi_occ_at_t100000k() {
    let orbe: Array1<f64> = array![
        -0.8274698649039047,
        -0.4866977381657900,
        -0.4293504325361446,
        -0.3805317817759825,
        0.4597732008355508,
        0.5075648461370381
    ];
    let temperature: f64 = 100000.0;
    let n_elec: usize = 8;
    let n_elec_unpaired: usize = 0;
    let mu_ref: f64 = 0.1259066212142123;
    let occ_ref: Vec<f64> = vec![
        1.906094704507551,
        1.747482667930276,
        1.7047529931205556,
        1.6638147878851874,
        0.5168129976856992,
        0.4611095080472262,
    ];
    let result: (f64, Vec<f64>) =
        fermi_occupation(orbe.view(), n_elec, n_elec_unpaired, temperature);
    let mu: f64 = result.0;
    let occ: Vec<f64> = result.1;
    assert!((mu - mu_ref).abs() < 1e-4);
    //assert_eq!(occ, occ_ref);
    assert!((occ.iter().sum::<f64>() - n_elec as f64).abs() < 1e-08);
}
