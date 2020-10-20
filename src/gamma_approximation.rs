use libm;
use std::collections::HashMap;
use std::collections::hash_map::RandomState;

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
pub fn gaussian_decay(
    hubbard_u: HashMap<u8, f64>,
    valorbs: HashMap<u8, (u8, u8, u8)>,
    r_lr: f64,
    lc_flag: bool,
) -> HashMap<u8, f64> {
    let mut sigmas: HashMap<u8, f64> = HashMap::new();

    // ATTENTION LC_FLAG IS OVERWRITTEN HERE
    let lc_flag = true;
    if lc_flag == false {
        for (z, u) in hubbard_u.iter() {
            sigmas.insert(*z, 1.0 / (*u * PI_SQRT));
        }
    } else {
        // do something else
        // but it don't understand the dftbaby code
        // at this place, as it always returns sigmas0
        // so this if-else conditions might be superfluous
    }
    return sigmas;
}

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
        let result: f64 = match self {
            SwitchingFunction::ErrorFunction => libm::erf(r / r_lr),
            SwitchingFunction::ErrorFunctionGaussian => {
                if r < 1.0e-8 {
                    0.0
                } else {
                    libm::erf(r / lr) / r
                        - 2.0 / (PI_SQRT * r_lr) * (-1.0 / 3.0 * (r / r_lr).pow(2)).exp()
                }
            }
            SwitchingFunction::NoSwitching => 0.0,
        };
        return result;
    }

    fn eval_deriv(&self, r: f64, r_lr: f64) -> f64 {
        let result: f64 = match self {
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
enum GammaFunction {
    // spherical charge fluctuations are modelled as Gaussians
    // FA(|r-RA|) = 1/(2 pi sigmaA^2)^(3/2) * exp( - (r-RA)^2/(2*sigmaA^2) )
    Gaussian{sigma:HashMap<u8, f64>, c:HashMap<(u8, u8), f64>},
    // spherical charge fluctuation around an atom A are modelled as Slater functions
    // FA(|r-RA|) = tA^3/(8 pi)*exp(-tA*|r-RA|)
    Slater{tau:HashMap<u8, f64>},
    // spherical charge fluctuations are modelled as Gaussians
    // FA(|r-RA|) = 1/(2 pi sigmaA^2)^(3/2) * exp( - (r-RA)^2/(2*sigmaA^2) )
    // long-range part of the Coulomb potential
    // 1/r  ->  erf(r/Rlr)/r
    GaussianLC{sigma:HashMap<u8, f64>, c:HashMap<(u8, u8), f64>, r_lr:f64},
    // aproximate LC integral by taking switching function out of the integral
    //ApproxLC{gf: , sw:},
    // the gamma function are read from a table and interpolated
    Numerical{atompairs: Vec<(u8, u8)>},
}


impl GammaFunction {

    fn initialize (&self, sigmas: HashMap<u8, f64>) -> HashMap<(u8, u8), f64> {
        let result: HashMap<(u8, u8), f64> = HashMap::new();
        let bla:f64 = match self {
            GammaFunction::Gaussian => {1.0}
            GammaFunction::Slater => {1.0}
            GammaFunction::GaussianLC => {1.0}
            GammaFunction::ApproxLC => {1.0}
            GammaFunction::Numerical => {1.0}
        };
        return result;
    }

    fn eval(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match self {
            GammaFunction::Gaussian => {
                assert!(r > 0.0);
                libm::erf()
            }


        },
        return result;
    }

}

//pub fn gamma_function (sigmas: HashMap<u8, f64>)
