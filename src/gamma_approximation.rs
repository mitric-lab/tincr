use crate::calculator::get_gamma_matrix;
use crate::molecule::{distance_matrix, Molecule};
use approx::AbsDiffEq;
use libm;
use ndarray::{array, Array1, Array2, Array3, ArrayView2};
use std::collections::HashMap;
use std::f64::consts::PI;

const PI_SQRT: f64 = 1.7724538509055159;

/// The decay constants for the gaussian charge fluctuations
/// are determined from the requirement d^2 E_atomic/d n^2 = U_H.
///
/// In the DFTB approximations with long-range correction one has
///
/// U_H = gamma_AA - 1/2 * 1/(2*l+1) gamma^lr_AA
///
/// where l is the angular momentum of the highest valence orbital
///
/// see "Implementation and benchmark of a long-range corrected functional
///      in the DFTB method" by V. Lutsker, B. Aradi and Th. Niehaus
///
/// Here, this equation is solved for sigmaA, the decay constant
/// of a gaussian.
pub fn gaussian_decay(hubbard_u: &HashMap<u8, f64>) -> HashMap<u8, f64> {
    let mut sigmas: HashMap<u8, f64> = HashMap::new();
    for (z, u) in hubbard_u.iter() {
        sigmas.insert(*z, 1.0 / (*u * PI_SQRT));
    }
    return sigmas;
}

// TODO: DO WE NEED THIS?
enum SwitchingFunction {
    // f(R) = erf(R/Rlr)
    ErrorFunction,
    // f(R) = erf(R/Rlr) - 2/(sqrt(pi)*Rlr) * R * exp(-1/3 * (R/Rlr)**2)
    ErrorFunctionGaussian,
    // f(R) = 0
    NoSwitching,
}

impl SwitchingFunction {
    fn eval(&self, r: f64, r_lr: f64) -> f64 {
        let result: f64 = match *self {
            SwitchingFunction::ErrorFunction => libm::erf(r / r_lr),
            SwitchingFunction::ErrorFunctionGaussian => {
                if r < 1.0e-8 {
                    0.0
                } else {
                    libm::erf(r / r_lr) / r
                        - 2.0 / (PI_SQRT * r_lr) * (-1.0 / 3.0 * (r / r_lr).powi(2)).exp()
                }
            }
            SwitchingFunction::NoSwitching => 0.0,
        };
        return result;
    }

    fn eval_deriv(&self, r: f64, r_lr: f64) -> f64 {
        let result: f64 = match *self {
            SwitchingFunction::ErrorFunction => {
                2.0 / (PI_SQRT * r_lr) * (-(r / r_lr).powi(2)).exp()
            }
            SwitchingFunction::ErrorFunctionGaussian => {
                if r < 1.0e-8 {
                    0.0
                } else {
                    let r2: f64 = (r / r_lr).powi(2);
                    4.0 / (3.0 * PI_SQRT * r_lr.powi(3)) * (-r2 / 3.0).exp() * r
                        + 2.0 / (PI_SQRT * r_lr) * -r2.exp() / r
                        - libm::erf(r / r_lr) / r.powi(2)
                }
            }
            SwitchingFunction::NoSwitching => 0.0,
        };
        return result;
    }
}

/// ## Gamma Function
/// gamma_AB = int F_A(r-RA) * 1/|RA-RB| * F_B(r-RB) d^3r
pub enum GammaFunction {
    Slater {
        tau: HashMap<u8, f64>,
    },
    Gaussian {
        sigma: HashMap<u8, f64>,
        c: HashMap<(u8, u8), f64>,
        r_lr: f64,
    },
}

impl GammaFunction {
    pub(crate) fn initialize(&mut self) {
        match *self {
            GammaFunction::Gaussian {
                ref sigma,
                ref mut c,
                ref r_lr,
            } => {
                // Construct the C_AB matrix
                for z_a in sigma.keys() {
                    for z_b in sigma.keys() {
                        c.insert(
                            (*z_a, *z_b),
                            1.0 / (2.0
                                * (sigma[z_a].powi(2) + sigma[z_b].powi(2) + 0.5 * r_lr.powi(2)))
                            .sqrt(),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    fn eval(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Gaussian {
                ref sigma,
                ref c,
                ref r_lr,
            } => {
                assert!(r > 0.0);
                libm::erf(c[&(z_a, z_b)] * r) / r
            }
            GammaFunction::Slater { ref tau } => {
                let t_a: f64 = tau[&z_a];
                let t_b: f64 = tau[&z_b];
                if r.abs() < 1.0e-5 {
                    // R -> 0 limit
                    t_a * t_b * (t_a.powi(2) + 3.0 * t_a * t_b + t_b.powi(2))
                        / (2.0 * (t_a + t_b).powi(3))
                } else if (t_a - t_b).abs() < 1.0e-5 {
                    // t_A == t_b limit
                    let x: f64 = t_a * r;
                    (1.0 / r)
                        * (1.0
                            - (-t_a * r).exp() * (48.0 + 33.0 * x + 9.0 * x.powi(2) + x.powi(3))
                                / 48.0)
                } else {
                    // general case R != 0 and t_a != t_b
                    let denom_ab: f64 = t_b.powi(4)
                        * (t_b.powi(2) * (2.0 + t_a * r) - t_a.powi(2) * (6.0 + t_a * r));
                    let denom_ba: f64 = t_a.powi(4)
                        * (t_a.powi(2) * (2.0 + t_b * r) - t_b.powi(2) * (6.0 + t_b * r));
                    let num: f64 = 2.0 * (t_a.powi(2) - t_b.powi(2)).powi(3);
                    (1.0 / r)
                        * (1.0 + ((-t_a * r).exp() * denom_ab - (-t_b * r).exp() * denom_ba) / num)
                }
            }
        };
        return result;
    }

    fn eval_limit0(&self, z: u8) -> f64 {
        let result: f64 = match *self {
            GammaFunction::Gaussian {
                ref sigma,
                ref c,
                ref r_lr,
            } => 1.0 / (PI * (sigma[&z].powi(2) + 0.25 * r_lr.powi(2))).sqrt(),
            GammaFunction::Slater { ref tau } => (5.0 / 16.0) * tau[&z],
        };
        return result;
    }
}

fn gamma_atomwise(
    gamma_func: GammaFunction,
    atomic_numbers: &[u8],
    n_atoms: usize,
    distances: ArrayView2<f64>,
) -> (Array2<f64>) {
    let mut g0 = Array2::zeros((n_atoms, n_atoms));
    for (i, z_i) in atomic_numbers.iter().enumerate() {
        for (j, z_j) in atomic_numbers.iter().enumerate() {
            if i == j {
                g0[[i, j]] = gamma_func.eval_limit0(*z_i);
            } else if i < j {
                g0[[i, j]] = gamma_func.eval(distances[[i, j]], *z_i, *z_j);
            } else {
                g0[[i, j]] = g0[[j, i]];
            }
        }
    }
    return g0;
}

pub fn gamma_ao_wise(
    gamma_func: GammaFunction,
    atomic_numbers: &[u8],
    n_atoms: usize,
    n_orbs: usize,
    distances: ArrayView2<f64>,
    valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
) -> (Array2<f64>, Array2<f64>) {
    let g0: Array2<f64> = gamma_atomwise(gamma_func, atomic_numbers, n_atoms, distances);
    let mut g0_a0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, z_i) in atomic_numbers.iter().enumerate() {
        for _ in &valorbs[z_i] {
            nu = 0;
            for (j, z_j) in atomic_numbers.iter().enumerate() {
                for _ in &valorbs[z_j] {
                    g0_a0[[mu, nu]] = g0[[i, j]];
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (g0, g0_a0);
}

//TODO: DERIVATIVE OF GAMMA MATRIX

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
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers.clone(), positions, charge, multiplicity);
    // get gamma matrix without LRC
    let hubbard_u: HashMap<u8, f64>;
    let mut sigma: HashMap<u8, f64> = HashMap::new();
    sigma.insert(1, 1.1952768018792987);
    sigma.insert(8, 1.2628443596207704);
    let mut c: HashMap<(u8, u8), f64> = HashMap::new();
    let r_lr: f64 = 0.0;
    let mut gf = GammaFunction::Gaussian { sigma, c, r_lr };
    gf.initialize();
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
