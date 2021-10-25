use crate::AtomSlice;
use hashbrown::HashMap;
use itertools::iproduct;
use libm;
use nalgebra::Vector3;
use ndarray::prelude::*;
use soa_derive::soa_zip;
use std::f64::consts::PI;

const PI_SQRT: f64 = 1.7724538509055159;

/// Type of Gamma function that is used for the approximate computation of the +
/// two electron integrals.
#[derive(Copy, Clone)]
pub enum FunctionType {
    Slater,
    Gaussian,
}

pub trait GammaFunction {
    fn eval(&self, r: f64, z_a: u8, z_b: u8) -> f64;
    fn eval_limit0(&self, z: u8) -> f64;
    fn deriv(&self, r: f64, z_a: u8, z_b: u8) -> f64;

    /// Compute the Gamma matrix between all atoms of one sets of atoms
    fn gamma_atomwise(&self, atoms: AtomSlice) -> Array2<f64> {
        let mut g0 = Array2::zeros((atoms.len(), atoms.len()));
        for (i, (xyz_i, number_i)) in soa_zip!(atoms, [xyz, number]).enumerate() {
            for (j, (xyz_j, number_j)) in soa_zip!(atoms, [xyz, number]).enumerate() {
                if i == j {
                    g0[[i, j]] = self.eval_limit0(*number_i);
                } else if i < j {
                    g0[[i, j]] = self.eval((xyz_i - xyz_j).norm(), *number_i, *number_j);
                } else {
                    g0[[i, j]] = g0[[j, i]];
                }
            }
        }
        g0
    }

    /// Compute the Gamma matrix between two sets of atoms.
    fn gamma_atomwise_ab(&self, atoms_a: AtomSlice, atoms_b: AtomSlice) -> Array2<f64> {
        let g = iproduct!(
            soa_zip!(atoms_a, [xyz, number]),
            soa_zip!(atoms_b, [xyz, number])
        )
        .map(|((xyz_i, number_i), (xyz_j, number_j))| {
            self.eval((xyz_i - xyz_j).norm(), *number_i, *number_j)
        })
        .collect::<Vec<f64>>();
        Array2::from_shape_vec((atoms_a.len(), atoms_b.len()), g).unwrap()
    }

    fn gamma_ao_wise_from_atomwise(
        g0: ArrayView2<f64>,
        atoms: AtomSlice,
        n_orbs: usize,
    ) -> Array2<f64> {
        let mut g0_a0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
        let mut mu: usize = 0;
        let mut nu: usize;
        for (n_orbs_i, g0_i) in atoms.n_orbs.iter().zip(g0.outer_iter()) {
            for _ in 0..*n_orbs_i {
                nu = 0;
                for (n_orbs_j, g0_ij) in atoms.n_orbs.iter().zip(g0_i.iter()) {
                    for _ in 0..*n_orbs_j {
                        if mu <= nu {
                            g0_a0[[mu, nu]] = *g0_ij;
                            g0_a0[[nu, mu]] = *g0_ij;
                        }
                        nu += 1;
                    }
                }
                mu += 1;
            }
        }
        g0_a0
    }

    fn gamma_ao_wise(&self, atoms: AtomSlice, n_orbs: usize) -> (Array2<f64>, Array2<f64>) {
        let g0: Array2<f64> = self.gamma_atomwise(atoms);
        let g0_ao: Array2<f64> =
            GammaGaussian::gamma_ao_wise_from_atomwise(g0.view(), atoms, n_orbs);
        (g0, g0_ao)
    }

    fn gamma_gradients_atomwise(&self, atoms: AtomSlice) -> Array3<f64> {
        let mut g1_val: Array2<f64> = Array2::zeros((atoms.len(), atoms.len()));
        let mut g1: Array3<f64> = Array3::zeros((3 * atoms.len(), atoms.len(), atoms.len()));
        for (i, (xyz_i, number_i)) in soa_zip!(atoms, [xyz, number]).enumerate() {
            for (j, (xyz_j, number_j)) in soa_zip!(atoms, [xyz, number]).enumerate() {
                if i < j {
                    let r = xyz_i - xyz_j;
                    let r_ij: f64 = r.norm();
                    let e_ij: Vector3<f64> = r / r_ij;
                    g1_val[[i, j]] = self.deriv(r_ij, *number_i, *number_j);
                    g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
                        .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
                } else if j < i {
                    g1_val[[i, j]] = g1_val[[j, i]];
                    let r = xyz_i - xyz_j;
                    let e_ij: Vector3<f64> = r / r.norm();
                    g1.slice_mut(s![(3 * i)..(3 * i + 3), i, j])
                        .assign(&Array::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
                }
            }
        }
        g1
    }

    fn gamma_gradients_atomwise_2d(&self, atoms: AtomSlice) -> Array2<f64> {
        let mut g1_val: Array2<f64> = Array2::zeros((atoms.len(), atoms.len()));
        let mut g1: Array2<f64> = Array2::zeros((3 * atoms.len(), atoms.len()));
        for (i, (xyz_i, number_i)) in soa_zip!(atoms, [xyz, number]).enumerate() {
            for (j, (xyz_j, number_j)) in soa_zip!(atoms, [xyz, number]).enumerate() {
                if i < j {
                    let r = xyz_i - xyz_j;
                    let r_ij: f64 = r.norm();
                    let e_ij: Vector3<f64> = r / r_ij;
                    g1_val[[i, j]] = self.deriv(r_ij, *number_i, *number_j);
                    g1.slice_mut(s![(3 * i)..(3 * i + 3), j])
                        .assign(&Array1::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
                } else if j < i {
                    g1_val[[i, j]] = g1_val[[j, i]];
                    let r = xyz_i - xyz_j;
                    let e_ij: Vector3<f64> = r / r.norm();
                    g1.slice_mut(s![(3 * i)..(3 * i + 3), j])
                        .assign(&Array::from_iter((e_ij * g1_val[[i, j]]).iter().cloned()));
                }
            }
        }
        g1
    }

    fn gamma_grads_ao_wise_from_atomwise(
        g1: ArrayView3<f64>,
        atoms: AtomSlice,
        n_orbs: usize,
    ) -> Array3<f64> {
        let mut g1_a0: Array3<f64> = Array3::zeros((3 * atoms.len(), n_orbs, n_orbs));
        let mut mu: usize = 0;
        let mut nu: usize;

        for (i, n_orbs_i) in atoms.n_orbs.iter().enumerate() {
            for _ in 0..*n_orbs_i {
                nu = 0;
                for (j, (n_orbs_j, g_ij)) in atoms
                    .n_orbs
                    .iter()
                    .zip(g1.slice(s![(3 * i)..(3 * i + 3), i, ..]).axis_iter(Axis(1)))
                    .enumerate()
                {
                    for _ in 0..*n_orbs_j {
                        if i != j {
                            g1_a0
                                .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
                                .assign(&g_ij);
                            g1_a0
                                .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
                                .assign(&g_ij);
                        }
                        nu += 1;
                    }
                }
                mu += 1;
            }
        }
        g1_a0
    }

    fn gamma_gradients_ao_wise(
        &self,
        atoms: AtomSlice,
        n_orbs: usize,
    ) -> (Array3<f64>, Array3<f64>) {
        let g1: Array3<f64> = self.gamma_gradients_atomwise(atoms);
        let g1_ao: Array3<f64> =
            GammaGaussian::gamma_grads_ao_wise_from_atomwise(g1.view(), atoms, n_orbs);
        (g1, g1_ao)
    }
}

pub struct GammaGaussian {
    /// Decay constant of the Gaussian function.
    sigma: HashMap<u8, f64>,
    /// Element-pairwise coefficients.
    c: HashMap<(u8, u8), f64>,
    /// Radius of long-range correction. If r_lr == 0, no LC is used.
    r_lr: f64,
}

impl GammaGaussian {
    /// The decay constants for the gaussian charge fluctuations are determined from the requirement
    /// d^2 E_atomic
    /// ------------- = U_H
    /// d n^2
    ///
    /// In the tight-binding approximations with long-range correction one has
    ///
    /// U_H = gamma_AA - 1/2 * 1/(2*l+1) gamma^lr_AA
    ///
    /// where l is the angular momentum of the highest valence orbital.
    ///
    /// for further information see:
    /// "Implementation and benchmark of a long-range corrected functional in the DFTB method"
    ///  by V. Lutsker, B. Aradi and T. Niehaus
    ///
    /// Here, this equation is solved for sigmaA, the decay constant of a Gaussian.
    pub fn new(unique_atoms: AtomSlice, r_lr: f64) -> Self {
        let mut sigma: HashMap<u8, f64> = HashMap::with_capacity(unique_atoms.len());

        for (number, hubbard) in soa_zip!(unique_atoms, [number, hubbard]) {
            sigma.insert(*number, 1.0 / (hubbard * PI_SQRT));
        }

        let mut c: HashMap<(u8, u8), f64> = HashMap::with_capacity(unique_atoms.len().pow(2));
        // Construct the C_AB matrix
        for z_a in sigma.keys() {
            for z_b in sigma.keys() {
                c.insert(
                    (*z_a, *z_b),
                    1.0 / (2.0 * (sigma[z_a].powi(2) + sigma[z_b].powi(2) + 0.5 * r_lr.powi(2)))
                        .sqrt(),
                );
            }
        }

        Self { sigma, c, r_lr }
    }
}

impl GammaFunction for GammaGaussian {
    fn eval(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        libm::erf(self.c[&(z_a, z_b)] * r) / r
    }

    fn eval_limit0(&self, z: u8) -> f64 {
        1.0 / (PI * (self.sigma[&z].powi(2) + 0.25 * self.r_lr.powi(2))).sqrt()
    }

    fn deriv(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let c_v: f64 = self.c[&(z_a, z_b)];
        2.0 * c_v / PI_SQRT * (-(c_v * r).powi(2)).exp() / r - libm::erf(c_v * r) / r.powi(2)
    }
}

pub struct GammaSlater {
    tau: HashMap<u8, f64>,
}

impl GammaFunction for GammaSlater {
    fn eval(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let t_a: f64 = self.tau[&z_a];
        let t_b: f64 = self.tau[&z_b];
        if r.abs() < 1.0e-5 {
            // R -> 0 limit
            return t_a * t_b * (t_a.powi(2) + 3.0 * t_a * t_b + t_b.powi(2))
                / (2.0 * (t_a + t_b).powi(3));
        }
        if (t_a - t_b).abs() < 1.0e-5 {
            // t_A == t_B limit
            let x: f64 = t_a * r;
            (1.0 / r)
                * (1.0 - (-t_a * r).exp() * (48.0 + 33.0 * x + 9.0 * x.powi(2) + x.powi(3)) / 48.0)
        } else {
            // general case R != 0 and t_a != t_b
            let denom_ab: f64 =
                t_b.powi(4) * (t_b.powi(2) * (2.0 + t_a * r) - t_a.powi(2) * (6.0 + t_a * r));
            let denom_ba: f64 =
                t_a.powi(4) * (t_a.powi(2) * (2.0 + t_b * r) - t_b.powi(2) * (6.0 + t_b * r));
            let num: f64 = 2.0 * (t_a.powi(2) - t_b.powi(2)).powi(3);
            (1.0 / r) * (1.0 + ((-t_a * r).exp() * denom_ab - (-t_b * r).exp() * denom_ba) / num)
        }
    }

    fn eval_limit0(&self, z: u8) -> f64 {
        (5.0 / 16.0) * self.tau[&z]
    }

    fn deriv(&self, r: f64, z_a: u8, z_b: u8) -> f64 {
        let t_a: f64 = self.tau[&z_a];
        let t_b: f64 = self.tau[&z_b];
        if r.abs() < 1.0e-5 {
            // R -> 0 limit
            return 0.0;
        }
        if (t_a - t_b).abs() < 1.0e-5 {
            // t_A == t_b limit
            let x: f64 = t_a * r;
            -1.0 / r.powi(2)
                * (1.0 - (-x).exp() * (1.0 + 1.0 / 48.0 * (x * (4.0 + x) * (12.0 + x * (3.0 + x)))))
        } else {
            // general case R != 0 and t_a != t_b
            let t_a_r: f64 = t_a * r;
            let t_b_r: f64 = t_b * r;
            let t_a2: f64 = t_a.powi(2);
            let t_b2: f64 = t_b.powi(2);
            let denom: f64 = 2.0 * (t_a2 - t_b2).powi(3);
            let f_b: f64 =
                (2.0 + t_b_r * (2.0 + t_b_r)) * t_a2 - (6.0 + t_b_r * (6.0 + t_b_r)) * t_b2;
            let f_a: f64 =
                (2.0 + t_a_r * (2.0 + t_a_r)) * t_b2 - (6.0 + t_a_r * (6.0 + t_a_r)) * t_a2;
            -1.0 / r.powi(2)
                * (1.0
                    - 1.0 / denom
                        * (t_a2.powi(2) * f_b * (-t_b_r).exp()
                            - t_b2.powi(2) * f_a * (-t_a_r).exp()))
        }
    }
}
