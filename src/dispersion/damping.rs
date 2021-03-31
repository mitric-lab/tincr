use crate::dispersion::model::Molecule;
use ndarray::prelude::*;
use std::ops::AddAssign;

const EPSILON: f64 = 2.2204460492503131E-016;
const S6_DEFAULT: f64 = 1.0;
const S9_DEFAULT: f64 = 1.0;
const ALP_DEFAULT: f64 = 16.0;

pub struct AtmDispersionGradientsResult {
    pub energy: Array1<f64>, // dispersion energy input
    pub dEdcn: Option<Array1<f64>>, // derivative of the energy w.r.t the coordination number
    pub dEdq: Option<Array1<f64>>, // derivative of the energy w.r.t the partial charges
    pub gradient: Option<Array2<f64>>, // dispersion gradient
    pub sigma: Option<Array2<f64>>, // dispersion virial
}

// Evaluation of the dispersion energy expression
pub fn get_atm_dispersion(
    mol: &Molecule, // molecular structure data
    trans: ArrayView2<f64>, // lattice points
    cutoff: f64, // real space cutoff
    s9: f64, // scaling for dispersion coefficients
    a1: f64, // scaling parameter for critical radius
    a2: f64, // offset parameter for critical radius
    alp: f64, // exponent of zero damping function
    r4r2: ArrayView1<f64>, // expectation values for r4 over r4 operator
    c6: ArrayView2<f64>, // C6 coefficients for all atom pairs
    dc6dcn: Option<ArrayView2<f64>>, // derivative of the C6 w.r.t. the coordination number
    dc6dq: Option<ArrayView2<f64>>, // derivative of the C6 w.r.t. the partial charges
    energy_in: Array1<f64>, // dispersion energy input
    dEdcn_in: Option<Array1<f64>>, // derivative of the energy w.r.t the coordination number
    dEdq_in: Option<Array1<f64>>, // derivative of the energy w.r.t the partial charges
    gradient_in: Option<Array2<f64>>, // dispersion gradient
    sigma_in: Option<Array2<f64>>, // dispersion virial
) -> AtmDispersionGradientsResult {


    if s9.abs() < EPSILON {
        let res = AtmDispersionGradientsResult {
            energy: energy_in,
            dEdcn: dEdcn_in,
            dEdq: dEdq_in,
            gradient: gradient_in,
            sigma: sigma_in,
        };
        return res;
    }

    let grad = dc6dcn.is_some() && dEdcn_in.is_some() && dc6dq.is_some()
                    && dEdq_in.is_some() && gradient_in.is_some() && sigma_in.is_some();
    let cutoff2 = cutoff * cutoff;

    if grad {
        let mut energy: Array1<f64> = energy_in.clone();
        let mut dEdcn: Array1<f64> = dEdcn_in.unwrap().clone();
        let mut dEdq: Array1<f64> = dEdq_in.unwrap().clone();
        let mut gradient: Array2<f64> = gradient_in.unwrap().clone();
        let mut sigma: Array2<f64> = sigma_in.unwrap().clone();

        let dc6dcn = dc6dcn.unwrap();
        let dc6dq = dc6dq.unwrap();

        // parallelise this loop
        for iat in 0..mol.n_atoms {
            let izp = mol.id[iat];
            for jat in 0..mol.n_atoms {
                let jzp = mol.id[jat];
                let c6ij = c6[[iat, jat]];
                let r0ij = a1 * (3.0*r4r2[jzp]*r4r2[izp]).sqrt() + a2;
                for jtr in 0..trans.len_of(Axis(0)) {
                    let vij: Array1<f64> = &mol.positions.slice(s![jat, ..])
                                           + &trans.slice(s![jtr, ..])
                                           - &mol.positions.slice(s![iat, ..]);
                    let r2ij = vij[0]*vij[0] + vij[1]*vij[1] + vij[2]*vij[2];
                    if r2ij > cutoff2 || r2ij < EPSILON {
                        continue;
                    }
                    for kat in 0..jat{
                        let kzp = mol.id[kat];
                        let c6ik = c6[[iat, kat]];
                        let c6jk = c6[[jat, kat]];
                        let c9 = -s9 * ((c6ij*c6ik*c6jk).abs()).sqrt();
                        let r0ik = a1 * (3.0*r4r2[kzp]*r4r2[izp]).sqrt() + a2;
                        let r0jk = a1 * (3.0*r4r2[kzp]*r4r2[jzp]).sqrt() + a2;
                        let r0 = r0ij * r0ik * r0jk;
                        let triple = triple_scale(iat, jat, kat);
                        for ktr in 0..trans.len_of(Axis(0)) {
                            let vik: Array1<f64> = &mol.positions.slice(s![kat, ..])
                                + &trans.slice(s![ktr, ..])
                                - &mol.positions.slice(s![iat, ..]);
                            let r2ik = vik[0]*vik[0] + vik[1]*vik[1] + vik[2]*vik[2];
                            if r2ik > cutoff2 || r2ik < EPSILON {
                                continue;
                            }
                            let vjk: Array1<f64> = &mol.positions.slice(s![kat, ..])
                                + &trans.slice(s![ktr, ..])
                                - &mol.positions.slice(s![jat, ..])
                                - &trans.slice(s![jtr, ..]);
                            let r2jk = vjk[0]*vjk[0] + vjk[1]*vjk[1] + vjk[2]*vjk[2];
                            if r2jk > cutoff2 || r2jk < EPSILON {
                                continue;
                            }
                            let r2 = r2ij*r2ik*r2jk;
                            let r1 = r2.sqrt();
                            let r3 = r2 * r1;
                            let r5 = r3 * r2;

                            let fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1).powf(alp / 3.0));
                            let ang = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik)
                                                 * (-r2ij + r2jk + r2ik) / r5 + (1.0 / r3);

                            let rr = ang * fdmp;

                            let dfdmp = -2.0 * alp * (r0 / r1).powf(alp / 3.0) * fdmp.powi(2);

                            // d/drij
                            let dang = -0.375 * (r2ij.powi(3) + r2ij.powi(2) * (r2jk + r2ik)
                                       + r2ij * (3.0 * r2jk.powi(2) + 2.0 * r2jk * r2ik + 3.0 * r2ik.powi(2))
                                       - 5.0 * (r2jk - r2ik).powi(2) * (r2jk + r2ik)) / r5;
                            let dGij: Array1<f64> = c9 * (-dang*fdmp + ang*dfdmp) / r2ij * &vij;

                            // d/drik
                            let dang = -0.375 * (r2ik.powi(3) + r2ik.powi(2) * (r2jk + r2ij)
                                       + r2ik * (3.0 * r2jk.powi(2) + 2.0 * r2jk * r2ij + 3.0 * r2ij.powi(2))
                                       - 5.0 * (r2jk - r2ij).powi(2) * (r2jk + r2ij)) / r5;
                            let dGik: Array1<f64> = c9 * (-dang * fdmp + ang * dfdmp) / r2ik * &vik;

                            // d/drjk
                            let dang = -0.375 * (r2jk.powi(3) + r2jk.powi(2) * (r2ik + r2ij)
                                     + r2jk * (3.0 * r2ik.powi(2) + 2.0 * r2ik * r2ij + 3.0 * r2ij.powi(2))
                                     - 5.0 * (r2ik - r2ij).powi(2) * (r2ik + r2ij)) / r5;
                            let dGjk: Array1<f64> = c9 * (-dang * fdmp + ang * dfdmp) / r2jk * &vjk;

                            let dE = rr * c9 * triple;
                            energy[iat] = energy[iat] - dE/3.0;
                            energy[jat] = energy[jat] - dE/3.0;
                            energy[kat] = energy[kat] - dE/3.0;

                            gradient.slice_mut(s![iat, ..]).add_assign(&(-(&dGij + &dGik)));
                            gradient.slice_mut(s![jat, ..]).add_assign(&(&dGij - &dGjk));
                            gradient.slice_mut(s![kat, ..]).add_assign(&(&dGik + &dGjk));

                            let dS: Array2<f64> = spread(&dGij, 1, 3) * spread(&vij, 0, 3)
                                                + spread(&dGik, 1, 3) * spread(&vik, 0, 3)
                                                + spread(&dGjk, 1, 3) * spread(&vjk, 0, 3);
                            sigma += &(dS * triple);

                            dEdcn[iat] = dEdcn[iat] - dE * 0.5 * (dc6dcn[[jat, iat]] / c6ij + dc6dcn[[kat, iat]] / c6ik);
                            dEdcn[jat] = dEdcn[jat] - dE * 0.5 * (dc6dcn[[iat, jat]] / c6ij + dc6dcn[[kat, jat]] / c6jk);
                            dEdcn[kat] = dEdcn[kat] - dE * 0.5 * (dc6dcn[[iat, kat]] / c6ik + dc6dcn[[jat, kat]] / c6jk);

                            dEdq[iat] = dEdq[iat] - dE * 0.5 * (dc6dq[[jat, iat]] / c6ij + dc6dq[[kat, iat]] / c6ik);
                            dEdq[jat] = dEdq[jat] - dE * 0.5 * (dc6dq[[iat, jat]] / c6ij + dc6dq[[kat, jat]] / c6jk);
                            dEdq[kat] = dEdq[kat] - dE * 0.5 * (dc6dq[[iat, kat]] / c6ik + dc6dq[[jat, kat]] / c6jk);
                        }
                    }
                }
            }
        }

        let res = AtmDispersionGradientsResult {
            energy: energy,
            dEdcn: Some(dEdcn),
            dEdq: Some(dEdq),
            gradient: Some(gradient),
            sigma: Some(sigma),
        };
        res
    } else {
        let mut energy: Array1<f64> = energy_in.clone();

        // parallelise this loop
        for iat in 0..mol.n_atoms {
            let izp = mol.id[iat];
            for jat in 0..mol.n_atoms {
                let jzp = mol.id[jat];
                let c6ij = c6[[iat, jat]];
                let r0ij = a1 * (3.0*r4r2[jzp]*r4r2[izp]).sqrt() + a2;
                for jtr in 0..trans.len_of(Axis(0)) {
                    let vij: Array1<f64> = &mol.positions.slice(s![jat, ..])
                        + &trans.slice(s![jtr, ..])
                        - &mol.positions.slice(s![iat, ..]);
                    let r2ij = vij[0]*vij[0] + vij[1]*vij[1] + vij[2]*vij[2];
                    if r2ij > cutoff2 || r2ij < EPSILON {
                        continue;
                    }
                    for kat in 0..jat{
                        let kzp = mol.id[kat];
                        let c6ik = c6[[iat, kat]];
                        let c6jk = c6[[jat, kat]];
                        let c9 = -s9 * ((c6ij*c6ik*c6jk).abs()).sqrt();
                        let r0ik = a1 * (3.0*r4r2[kzp]*r4r2[izp]).sqrt() + a2;
                        let r0jk = a1 * (3.0*r4r2[kzp]*r4r2[jzp]).sqrt() + a2;
                        let r0 = r0ij * r0ik * r0jk;
                        let triple = triple_scale(iat, jat, kat);
                        for ktr in 0..trans.len_of(Axis(0)) {
                            let vik: Array1<f64> = &mol.positions.slice(s![kat, ..])
                                + &trans.slice(s![ktr, ..])
                                - &mol.positions.slice(s![iat, ..]);
                            let r2ik = vik[0]*vik[0] + vik[1]*vik[1] + vik[2]*vik[2];
                            if r2ik > cutoff2 || r2ik < EPSILON {
                                continue;
                            }
                            let vjk: Array1<f64> = &mol.positions.slice(s![kat, ..])
                                + &trans.slice(s![ktr, ..])
                                - &mol.positions.slice(s![jat, ..])
                                - &trans.slice(s![jtr, ..]);
                            let r2jk = vjk[0]*vjk[0] + vjk[1]*vjk[1] + vjk[2]*vjk[2];
                            if r2jk > cutoff2 || r2jk < EPSILON {
                                continue;
                            }
                            let r2 = r2ij*r2ik*r2jk;
                            let r1 = r2.sqrt();
                            let r3 = r2 * r1;
                            let r5 = r3 * r2;

                            let fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1).powf(alp / 3.0));
                            let ang = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik)
                                * (-r2ij + r2jk + r2ik) / r5 + (1.0 / r3);

                            let rr = ang * fdmp;

                            let dE = rr * c9 * triple;
                            energy[iat] = energy[iat] - dE/3.0;
                            energy[jat] = energy[jat] - dE/3.0;
                            energy[kat] = energy[kat] - dE/3.0;
                        }
                    }
                }
            }
        }

        let res = AtmDispersionGradientsResult {
            energy: energy,
            dEdcn: None,
            dEdq: None,
            gradient: None,
            sigma: None,
        };
        res
    }
}

// Logic exercise to distribute a triple energy to atomwise energies.
fn triple_scale(ii: usize, jj: usize, kk: usize) -> f64 {
    // atom indices: ii, jj, kk
    // fraction of energy: triple
    if ii == jj {
        if ii == kk {
            // i,i',i'' -> 1/6
            1.0/6.0
        } else {
            // i,i',j -> 1/2
            0.5
        }
    } else {
        if ii != kk && jj != kk {
            // i,j,k -> 1 (full)
            1.0
        } else {
            // i,j,j' and i,j,i' -> 1/2
            0.5
        }
    }
}

fn spread(source: &Array1<f64>, axis: u8, copies: usize) -> Array2<f64> {
    let mut a: Array2<f64> = Array::zeros((copies, source.len()));
    for i in 0..copies {
        a.slice_mut(s![i, ..]).assign(&source);
    }
    if axis == 0 {
        a.reversed_axes()
    } else {
        a
    }
}

pub struct RationalDampingParam {
    pub s6: f64,
    pub s8: f64,
    pub s9: f64,
    pub a1: f64,
    pub a2: f64,
    pub alp: f64,
}