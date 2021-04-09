use super::tests::*;
use tincr::param::gamma_approximation::*;

/// Test of Gaussian decay function on a water molecule. The xyz geometry of the
/// water molecule is
/// ```no_run
/// 3
//
// O          0.34215        1.17577        0.00000
// H          1.31215        1.17577        0.00000
// H          0.01882        1.65996        0.77583
///```
///
///
#[test]
fn test_gaussian_decay() {
    let mut u: HashMap<u8, f64> = HashMap::new();
    u.insert(1, 0.4720158398964136);
    u.insert(8, 0.4467609798860577);

    let mut ref_sigmas: HashMap<u8, f64> = HashMap::new();
    ref_sigmas.insert(1, 1.1952768018792987);
    ref_sigmas.insert(8, 1.2628443596207704);

    let sigmas: HashMap<u8, f64> = gaussian_decay(&u);
    assert_eq!(ref_sigmas, sigmas);
}

#[test]
fn test_gamma_gaussian() {
    let mut u: HashMap<u8, f64> = HashMap::new();
    u.insert(1, 0.4720158398964136);
    u.insert(8, 0.4467609798860577);
    let sigmas: HashMap<u8, f64> = gaussian_decay(&u);
    let new_c: HashMap<(u8, u8), f64> = HashMap::new();
    let mut gfunc = GammaFunction::Gaussian {
        sigma: sigmas,
        c: new_c,
        r_lr: 3.03,
    };
    gfunc.initialize();
    assert_eq!(gfunc.eval(1.0, 1, 1), 0.2859521722011254);
    assert_eq!(gfunc.eval(2.0, 1, 1), 0.26817515355018845);
    assert_eq!(gfunc.eval(3.0, 1, 1), 0.24278403726022513);
    assert_eq!(gfunc.eval(1.0, 1, 8), 0.2829517673247839);
    assert_eq!(gfunc.eval(2.0, 1, 8), 0.26571666152876605);
    assert_eq!(gfunc.eval(3.0, 1, 8), 0.2410200913795066);
    assert_eq!(gfunc.eval_limit0(1), 0.2923649998054588);
    assert_eq!(gfunc.eval_limit0(8), 0.28605544182430387);
}

#[test]
fn gamma_ao_matrix() {
    // test gamma matrix with and without long range correction
    let mol: Molecule = get_water_molecule();
    // get gamma matrix without LRC
    let hubbard_u: HashMap<u8, f64>;
    let mut sigma: HashMap<u8, f64> = HashMap::new();
    sigma.insert(1, 1.1952768018792987);
    sigma.insert(8, 1.2628443596207704);
    let mut c: HashMap<(u8, u8), f64> = HashMap::new();
    let r_lr: f64 = 0.0;
    let mut gf = GammaFunction::Gaussian { sigma, c, r_lr };
    gf.initialize();
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let (gm, gm_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
        gf,
        &atomic_numbers,
        mol.n_atoms,
        mol.calculator.n_orbs,
        mol.distance_matrix.view(),
        &mol.calculator.valorbs,
    );
    let gamma_ref: Array2<f64> = array![
        [0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
        [0.3863557889890281, 0.4720158398964135, 0.3084885848056254],
        [0.3863561531176491, 0.3084885848056254, 0.4720158398964135]
    ];
    assert!(gm.abs_diff_eq(&gamma_ref, 1e-06));
    // test gamma matrix with long range correction
    //
    let mut sigma: HashMap<u8, f64> = HashMap::new();
    sigma.insert(1, 1.1952768018792987);
    sigma.insert(8, 1.2628443596207704);
    // get gamma matrix with LRC
    let mut c: HashMap<(u8, u8), f64> = HashMap::new();
    let r_lr: f64 = 3.03;
    let mut gf = GammaFunction::Gaussian { sigma, c, r_lr };
    gf.initialize();
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let (gm_lrc, gm_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
        gf,
        &atomic_numbers,
        mol.n_atoms,
        mol.calculator.n_orbs,
        mol.distance_matrix.view(),
        &mol.calculator.valorbs,
    );
    let gamma_lrc_ref: Array2<f64> = array![
        [0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
        [0.2692279296946004, 0.2923649998054588, 0.24296864292032624],
        [0.2692280400920803, 0.2429686492032624, 0.2923649998054588]
    ];
    assert!(gm_lrc.abs_diff_eq(&gamma_lrc_ref, 1e-08));
}
