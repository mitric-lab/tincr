use crate::molecule::*;
use libm;
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

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
pub fn gaussian_decay(hubbard_u: HashMap<u8, f64>) -> HashMap<u8, f64> {
    let mut sigmas: HashMap<u8, f64> = HashMap::new();
    for (z, u) in hubbard_u.iter() {
        sigmas.insert(*z, 1.0 / (*u * PI_SQRT));
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
    fn initialize(&self) {
        match self {
            GammaFunction::Gaussian(sigmas, ref mut c, r_lr) => {
                // Construct the C_AB matrix
                for z_a in sigmas.keys() {
                    for z_b in sigmas.keys() {
                        c.insert(
                            (z_a, z_b),
                            1.0 / (2.0
                                * (sigmas[z_a].powi(2) + sigmas[z_b].powi(2) + 0.5 * r_lr.powi(2)))
                            .sqrt(),
                        )
                    }
                }
            }
            GammaFunction::Slater => {}
        }
    }

    fn eval(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let result: f64 = match self {
            GammaFunction::Gaussian(_, &c, _) => {
                assert!(r > 0.0);
                libm::erf(c[(z_a, z_b)] * r) / r
            }
            GammaFunction::Slater(&tau) => {
                let t_a = tau[z_a];
                let t_b = tau[z_b];
                if r.abs() < 1.0e-5 {
                    // R -> 0 limit
                    t_a * t_b * (t_a.powi(2) + 3 * t_a * t_b + t_b.powi(2))
                        / (2.0 * (t_a + t_b).powi(3))
                } else if (t_a - t_b).abs() < 1.0e-5 {
                    // t_A == t_b limit
                    let x = t_a * r;
                    (1.0 / r)
                        * (1.0
                            - (-t_a * r).exp() * (48 + 33 * x + 9 * x.powi(2) + x.powi(3)) / 48.0)
                } else {
                    // general case R != 0 and t_a != t_b
                    let denom_ab =
                        t_b.powi(4) * (t_b.powi(2) * (2 + t_a * r) - t_a.powi(2) * (6 + t_a * r));
                    let denom_ba =
                        t_a.powi(4) * (t_a.powi(2) * (2 + t_b * r) - t_b.powi(2) * (6 + t_b * r));
                    let num = 2 * (t_a.powi(2) - t_b.powi(2)).powi(3);
                    (1.0 / r)
                        * (1.0 + ((-t_a * r).exp() * denom_ab - (-t_b * r).exp() * denom_ba) / num)
                }
            }
        };
        return result;
    }
}

fn gamma_atomwise(
    gamma_func: GammaFunction,
    mol: &Molecule,
    distances: ArrayView2<f64>,
) -> (Array2<f64>) {
    let mut g0 = Array2::zeros((mol.n_atoms, mol.n_atoms));
    for (i, (z_i, pos_i)) in mol.iter_atomlist().enumerate() {
        for (j, (z_j, pos_j)) in mol.iter_atomlist().enumerate() {
            if i == j {
                g0[[i, j]] = gamma_func.eval();
            } else if i < j {
                g0[[i, j]] = gamma_func.eval(distances[[i, j]], *z_i, *z_j);
            } else {
                g0[[i, j]] = g0[[j, i]];
            }
        }
    }
    return g0;
}

fn gamma_ao_wise(
    gamma_func: GammaFunction,
    mol: Molecule,
    distances: ArrayView2<f64>,
    directions: ArrayView2<f64>,
) -> Array2<f64> {
    let g0: Array2<f64> = gamma_atomwise(gamma_func, &mol, distances);
    let mut g0_a0: Array2<f64> = Array2::zeros((mol.n_orbs, mol.norbs));
    let mu: u32 = 0;
    for (i, (z_i, pos_i)) in mol.iter_atomlist().enumerate() {
        for (n_i, l_i, m_i) in mol.valorbs[*z_i] {
            let nu: u32 = 0;
            for (j, (z_j, pos_j)) in mol.iter_atomlist().enumerate() {
                for (n_j, l_j, m_j) in mol.valorbs[*z_j] {
                    g0_A0[[mu, nu]] = g0[[i, j]];
                }
            }
        }
    }

    return g0_a0;
}
