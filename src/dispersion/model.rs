use crate::dispersion::data::{
    get_covalent_rad, get_effective_charge, get_electronegativity, get_hardness, get_r4r2_val,
};
use crate::dispersion::reference::*;
use ndarray::{array, s, Array, Array1, Array2, Array3, Array4};
use std::f64::consts;

const GA_DEFAULT: f64 = 3.0;
const GC_DEFAULT: f64 = 2.0;
const WF_DEFAULT: f64 = 6.0;

const THOPI: f64 = 3.0 / consts::PI;

const FREQ: [f64; TRAPZD_POINTS] = [
    0.000001, 0.050000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
    0.800000, 0.900000, 1.000000, 1.200000, 1.400000, 1.600000, 1.800000, 2.000000, 2.500000,
    3.000000, 4.000000, 5.000000, 7.500000, 10.00000,
];

#[derive(Clone)]
pub struct Molecule {
    pub atomic_numbers: Vec<u8>,
    pub positions: Array2<f64>,
    pub charge: i8,
    pub multiplicity: u8,
    pub periodic: [bool; 3],
    pub lattice: [[f64; 3]; 3],
    pub n_atoms: usize,
    pub nid: usize,           // number of unique atoms
    pub id: Array1<usize>,    // identifier of atoms
    pub num: Vec<u8>,         // list of unique atomic numbers
}

impl Molecule {
    pub fn new(
        atomic_numbers: Vec<u8>,
        positions: Array2<f64>,
        charge: Option<i8>,
        multiplicity: Option<u8>,
        periodic: Option<[bool; 3]>,
        lattice: Option<[[f64; 3]; 3]>,
    ) -> Molecule {
        let n_atoms: usize = atomic_numbers.len();
        let charge: i8 = charge.unwrap_or(0);
        let multiplicity: u8 = multiplicity.unwrap_or(1);

        // Obtain nid and id
        let mut nid: usize = 0;
        let mut id: Array1<usize> = Array::zeros((n_atoms));
        let mut num: Vec<u8> = Vec::new();
        for iat in 0..n_atoms {
            if !num.iter().any(|&i| i == atomic_numbers[iat]) {
                nid = nid + 1;
                num.push(atomic_numbers[iat]);
            }
            id[iat] = nid - 1;
        }

        let periodic = if periodic.is_some() {
            periodic.unwrap()
        } else {
            if lattice.is_some() {
                [true; 3]
            } else {
                [false; 3]
            }
        };

        let lattice = if lattice.is_some() {
            lattice.unwrap()
        } else {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        };

        let mol = Molecule {
            atomic_numbers,
            positions,
            charge,
            multiplicity,
            periodic,
            lattice,
            n_atoms,
            nid,
            id,
            num,
        };
        return mol;
    }
}

pub struct ModelResult1  {
    pub quantity: Array1<f64>, // wished quantity, can be weight_reference or C6
    pub quantity_dcn: Option<Array1<f64>>, // derivative w.r.t the coordination number
    pub quantity_dq: Option<Array1<f64>>, // derivative w.r.t the partial charges
}

pub struct ModelResult2  {
    pub quantity: Array2<f64>, // wished quantity, can only be alpha
    pub quantity_dcn: Option<Array2<f64>>, // derivative w.r.t the coordination number
    pub quantity_dq: Option<Array2<f64>>, // derivative w.r.t the partial charges
}

pub struct D4Model {
    pub ga: f64,             // charge scaling height
    pub gc: f64,             // charge scaling steepness
    pub wf: f64,             // weighting factor for CN interpolation
    pub zeff: Array1<f64>,   // effective nuclear charges
    pub eta: Array1<f64>,    // chemical hardness
    pub en: Array1<f64>,     // electronegativity
    pub rcov: Array1<f64>,   // covalent radii for coordination number
    pub r4r2: Array1<f64>,   // expectation values for C8 extrapolation
    pub ref_: Array1<usize>, // number of reference systems
    pub ngw: Array2<usize>,  // number of Gaussian weights for each reference
    pub cn: Array2<f64>,     // reference coordination number
    pub q: Array2<f64>,      // reference partial charges
    pub aiw: Array3<f64>,    // reference dynamic polarizabilities
    pub c6: Array4<f64>,     // reference C6 coefficients
}

impl D4Model {
    pub fn from_molecule(
        mol: &Molecule,
        ga: Option<f64>,
        gc: Option<f64>,
        wf: Option<f64>,
    ) -> (D4Model) {
        // Get parameters
        let ga: f64 = ga.unwrap_or(GA_DEFAULT);
        let gc: f64 = gc.unwrap_or(GC_DEFAULT);
        let wf: f64 = wf.unwrap_or(WF_DEFAULT);

        let mut rcov: Array1<f64> = Array::zeros(mol.nid);
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            rcov[isp] = get_covalent_rad(izp);
        }

        let mut en: Array1<f64> = Array::zeros(mol.nid);
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            en[isp] = get_electronegativity(izp);
        }

        let mut zeff: Array1<f64> = Array::zeros(mol.nid);
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            zeff[isp] = get_effective_charge(izp);
        }

        let mut eta: Array1<f64> = Array::zeros(mol.nid);
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            eta[isp] = get_hardness(izp);
        }

        let mut r4r2: Array1<f64> = Array::zeros(mol.nid);
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            r4r2[isp] = get_r4r2_val(izp);
        }

        let mut ref_: Array1<usize> = Array::zeros(mol.nid);
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            ref_[isp] = get_nref(izp);
        }

        let mut cn: Array2<f64> = Array::zeros((mol.nid, MAX_NREF));
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            cn.slice_mut(s![isp, ..])
                .assign(&Array::from(get_refcn(izp).to_vec()));
        }

        let mut q: Array2<f64> = Array::zeros((mol.nid, MAX_NREF));
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            q.slice_mut(s![isp, ..])
                .assign(&Array::from(get_refq(izp).to_vec()));
        }

        let mut ngw: Array2<usize> = Array::zeros((mol.nid, MAX_NREF));
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            ngw.slice_mut(s![isp, ..])
                .assign(&Array::from(get_refgw(izp).to_vec()));
        }

        let mut aiw: Array3<f64> = Array::zeros((mol.nid, MAX_NREF, TRAPZD_POINTS));
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            let atmp: [[f64; TRAPZD_POINTS]; MAX_NREF] = get_refalpha(ga, gc, izp);
            for iref in 0..MAX_NREF {
                aiw.slice_mut(s![isp, iref, ..])
                    .assign(&Array::from(atmp[iref].to_vec()));
            }
        }

        let mut c6: Array4<f64> = Array::zeros((mol.nid, mol.nid, MAX_NREF, MAX_NREF));
        for isp in 0..mol.nid {
            let izp = mol.num[isp];
            for jsp in 0..isp {
                let jzp = mol.num[jsp];
                for iref in 0..ref_[isp] {
                    for jref in 0..ref_[isp] {
                        let atmp: Array1<f64> = &aiw.slice(s![isp, iref, ..])
                                                * &aiw.slice(s![jsp, jref, ..]);
                        let c6tmp: f64 = THOPI * trapzd(atmp);
                        c6[[isp, jsp, iref, jref]] = c6tmp;
                        c6[[jsp, isp, jref, iref]] = c6tmp;
                    }
                }
            }
        }

        let model = D4Model {
            ga: ga,
            gc: gc,
            wf: wf,
            zeff: zeff,
            eta: eta,
            en: en,
            rcov: rcov,
            r4r2: r4r2,
            ref_: ref_,
            ngw: ngw,
            cn: cn,
            q: q,
            aiw: aiw,
            c6: c6,
        };

        return model;
    }

    /// Calculate the weights of the reference system and the derivatives w.r.t.
    /// coordination number for later use.
    pub fn weight_references_derivatives(
        &self,
        mol: &Molecule,
        cn: Array1<f64>,
        q: Array1<f64>,
        derivative: bool,
    ) -> ModelResult2 {
        if derivative {
            let mut gwvec: Array2<f64> = Array::zeros((mol.n_atoms, MAX_NREF));
            let mut gwdcn: Array2<f64> = Array::zeros((mol.n_atoms, MAX_NREF));
            let mut gwdq: Array2<f64> = Array::zeros((mol.n_atoms, MAX_NREF));

            // parallelise this loop
            for iat in 0..mol.n_atoms {
                let izp = mol.id[iat] as usize;
                let zi = self.zeff[izp];
                let gi = self.eta[izp] * self.gc;
                let mut norm = 0.0;
                let mut dnorm = 0.0;
                for iref in 0..self.ref_[izp] {
                    for igw in 0..self.ngw[[izp, iref]] {
                        let wf = igw as f64 * self.wf;
                        let gw = weight_cn(wf, cn[iat], self.cn[[izp, iref]]);
                        norm = norm + gw;
                        dnorm = dnorm + 2.0 * wf * (self.cn[[izp, iref]] - cn[iat]) * gw;
                    }
                }
                norm = 1.0 / norm;
                for iref in 0..self.ref_[izp] {
                    let mut expw = 0.0;
                    let mut expd = 0.0;
                    for igw in 0..self.ngw[[izp, iref]] {
                        let wf = igw as f64 * self.wf;
                        let gw = weight_cn(wf, cn[iat], self.cn[[izp, iref]]);
                        expw = expw + gw;
                        expd = expd + 2.0 * wf * (self.cn[[izp, iref]] - cn[iat]) * gw;
                    }
                    let mut gwk = expw * norm;
                    if !gwk.is_finite() {
                        if (self.cn.slice(s![izp, ..self.ref_[izp]]))
                            .iter()
                            .cloned()
                            .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
                            .unwrap()
                            == self.cn[[izp, iref]]
                        {
                            gwk = 1.0;
                        } else {
                            gwk = 0.0;
                        }
                    }
                    gwvec[[iat, iref]] = gwk * zeta(self.ga, gi, self.q[[izp, iref]] + zi, q[iat] + zi);
                    gwdq[[iat, iref]] = gwk * dzeta(self.ga, gi, self.q[[izp, iref]] + zi, q[iat] + zi);

                    let mut dgwk = norm * (expd - expw * dnorm * norm);
                    if !dgwk.is_finite() {
                        dgwk = 0.0;
                    }
                    gwdcn[[iat, iref]] =
                        dgwk * zeta(self.ga, gi, self.q[[izp, iref]] + zi, q[iat] + zi);
                }
            }
            ModelResult2 {
                quantity: gwvec,
                quantity_dcn: Some(gwdcn),
                quantity_dq: Some(gwdq),
            }
        } else {
            let mut gwvec: Array2<f64> = Array::zeros((mol.n_atoms, MAX_NREF));

            // parallelise this loop
            for iat in 0..mol.n_atoms {
                let izp = mol.id[iat] as usize;
                let zi = self.zeff[izp];
                let gi = self.eta[izp] * self.gc;
                let mut norm = 0.0;
                for iref in 0..self.ref_[izp] {
                    for igw in 0..self.ngw[[izp, iref]] {
                        let wf = igw as f64 * self.wf;
                        let gw = weight_cn(wf, cn[iat], self.cn[[izp, iref]]);
                        norm = norm + gw;
                    }
                }
                norm = 1.0 / norm;
                for iref in 0..self.ref_[izp] {
                    let mut expw = 0.0;
                    for igw in 0..self.ngw[[izp, iref]] {
                        let wf = igw as f64 * self.wf;
                        expw = expw + weight_cn(wf, cn[iat], self.cn[[izp, iref]]);
                    }
                    let mut gwk = expw * norm;
                    if !gwk.is_finite() {
                        if (self.cn.slice(s![izp, ..self.ref_[izp]]))
                            .iter()
                            .cloned()
                            .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
                            .unwrap()
                            == self.cn[[izp, iref]]
                        {
                            gwk = 1.0;
                        } else {
                            gwk = 0.0;
                        }
                    }
                    gwvec[[iat, iref]] = gwk * zeta(self.ga, gi, self.q[[izp, iref]] + zi, q[iat] + zi);
                }
            }

            ModelResult2 {
                quantity: gwvec,
                quantity_dcn: None,
                quantity_dq: None,
            }
        }
    }

    /// Calculate atomic dispersion coefficients and their derivatives w.r.t.
    /// the coordination numbers and atomic partial charges.
    pub fn get_atomic_c6_derivatives(
        &self,
        mol: &Molecule,
        gw_inp: ModelResult2,
    ) -> ModelResult2 {
        if gw_inp.quantity_dcn.is_some() && gw_inp.quantity_dq.is_some() {
            let gwvec: Array2<f64> = gw_inp.quantity;
            let gwdcn: Array2<f64> = gw_inp.quantity_dcn.unwrap();
            let gwdq: Array2<f64> = gw_inp.quantity_dq.unwrap();

            let mut c6: Array2<f64> = Array::zeros((mol.n_atoms, mol.n_atoms));
            let mut dc6dcn: Array2<f64> = Array::zeros((mol.n_atoms, mol.n_atoms));
            let mut dc6dq: Array2<f64> = Array::zeros((mol.n_atoms, mol.n_atoms));

            // paralellise this loop
            for iat in 0..mol.n_atoms {
                let izp = mol.id[iat] as usize;
                for jat in 0..iat+1 {
                    let jzp = mol.id[jat] as usize;
                    let mut dc6 = 0.0;
                    let mut dc6dcni = 0.0;
                    let mut dc6dcnj = 0.0;
                    let mut dc6dqi = 0.0;
                    let mut dc6dqj = 0.0;
                    for iref in 0..self.ref_[izp] {
                        for jref in 0..self.ref_[jzp] {
                            let refc6 = self.c6[[jzp, izp, jref, iref]];
                            dc6 = dc6 + gwvec[[iat, iref]] * gwvec[[jat, jref]] * refc6;
                            dc6dcni = dc6dcni + gwdcn[[iat, iref]] * gwvec[[jat, jref]] * refc6;
                            dc6dcnj = dc6dcnj + gwvec[[iat, iref]] * gwdcn[[jat, jref]] * refc6;
                            dc6dqi = dc6dqi + gwdq[[iat, iref]] * gwvec[[jat, jref]] * refc6;
                            dc6dqj = dc6dqj + gwvec[[iat, iref]] * gwdq[[jat, jref]] * refc6;
                        }
                    }
                    c6[[jat, iat]] = dc6;
                    c6[[iat, jat]] = dc6;
                    dc6dcn[[jat, iat]] = dc6dcnj;
                    dc6dcn[[iat, jat]] = dc6dcni;
                    dc6dq[[jat, iat]] = dc6dqj;
                    dc6dq[[iat, jat]] = dc6dqi;
                }
            }

            ModelResult2{
                quantity: c6,
                quantity_dcn: Some(dc6dcn),
                quantity_dq: Some(dc6dq),
            }
        } else {
            let gwvec: Array2<f64> = gw_inp.quantity;
            let mut c6: Array2<f64> = Array::zeros((mol.n_atoms, mol.n_atoms));

            // paralellise this loop
            for iat in 0..mol.n_atoms {
                let izp = mol.id[iat] as usize;
                for jat in 0..iat+1 {
                    let jzp = mol.id[jat] as usize;
                    let mut dc6 = 0.0;
                    for iref in 0..self.ref_[izp] {
                        for jref in 0..self.ref_[jzp] {
                            let refc6 = self.c6[[jzp, izp, jref, iref]];
                            dc6 = dc6 + gwvec[[iat, iref]] * gwvec[[jat, jref]] * refc6;
                        }
                    }
                    c6[[jat, iat]] = dc6;
                    c6[[iat, jat]] = dc6;
                }
            }

            ModelResult2{
                quantity: c6,
                quantity_dcn: None,
                quantity_dq: None,
            }
        }
    }

    /// Calculate atomic polarizibilities and their derivatives w.r.t.
    /// the coordination numbers and atomic partial charges.
    pub fn get_polarizabilities_derivatives(
        &self,
        mol: &Molecule,
        gw_inp: ModelResult2,
    ) -> ModelResult1 {
        if gw_inp.quantity_dcn.is_some() && gw_inp.quantity_dq.is_some() {
            let gwvec: Array2<f64> = gw_inp.quantity;
            let gwdcn: Array2<f64> = gw_inp.quantity_dcn.unwrap();
            let gwdq: Array2<f64> = gw_inp.quantity_dq.unwrap();

            let mut alpha: Array1<f64> = Array::zeros(mol.n_atoms);
            let mut dadcn: Array1<f64> = Array::zeros(mol.n_atoms);
            let mut dadq: Array1<f64> = Array::zeros(mol.n_atoms);

            // parallelise this loop
            for iat in 0..mol.n_atoms {
                let izp = mol.id[iat] as usize;
                let mut da = 0.0;
                let mut dadcni = 0.0;
                let mut dadqi = 0.0;
                for iref in 0..self.ref_[izp] {
                    let refa = self.aiw[[izp, iref, 0]];
                    da = da + gwvec[[iat, iref]] * refa;
                    dadcni = dadcni + gwdcn[[iat, iref]] * refa;
                    dadqi = dadqi + gwdq[[iat, iref]] * refa;
                }
                alpha[iat] = da;
                dadcn[iat] = dadcni;
                dadq[iat] = dadqi;
            }

            ModelResult1{
                quantity: alpha,
                quantity_dcn: Some(dadcn),
                quantity_dq: Some(dadq),
            }
        } else {
            let gwvec: Array2<f64> = gw_inp.quantity;
            let mut alpha: Array1<f64> = Array::zeros(mol.n_atoms);

            // parallelise this loop
            for iat in 0..mol.n_atoms {
                let izp = mol.id[iat] as usize;
                let mut da = 0.0;
                for iref in 0..self.ref_[izp] {
                    da = da + gwvec[[iat, iref]] * self.aiw[[izp, iref, 0]];
                }
                alpha[iat] = da;
            }

            ModelResult1{
                quantity: alpha,
                quantity_dcn: None,
                quantity_dq: None,
            }
        }
    }
}

fn weight_cn(wf: f64, cn: f64, cnref: f64) -> f64 {
    // cngw
    (-wf * (cn - cnref).powi(2)).exp()
}

fn dzeta(a: f64, c: f64, qref: f64, qmod: f64) -> f64 {
    if qmod < 0.0 {
        0.0
    } else {
        -a * c * (c * (1.0 - qref / qmod)).exp() * zeta(a, c, qref, qmod) * qref / (qmod.powi(2))
    }
}

// numerical Casimir--Polder integration
fn trapzd(pol: Array1<f64>) -> f64 {
    let weights: Array1<f64> = 0.5
        * array![
            (FREQ[1] - FREQ[0]),
            (FREQ[1] - FREQ[0]) + (FREQ[2] - FREQ[1]),
            (FREQ[2] - FREQ[2]) + (FREQ[3] - FREQ[2]),
            (FREQ[3] - FREQ[3]) + (FREQ[4] - FREQ[3]),
            (FREQ[4] - FREQ[4]) + (FREQ[5] - FREQ[4]),
            (FREQ[5] - FREQ[5]) + (FREQ[6] - FREQ[5]),
            (FREQ[6] - FREQ[6]) + (FREQ[7] - FREQ[6]),
            (FREQ[7] - FREQ[7]) + (FREQ[8] - FREQ[7]),
            (FREQ[8] - FREQ[8]) + (FREQ[9] - FREQ[8]),
            (FREQ[9] - FREQ[9]) + (FREQ[10] - FREQ[9]),
            (FREQ[10] - FREQ[10]) + (FREQ[11] - FREQ[10]),
            (FREQ[11] - FREQ[11]) + (FREQ[12] - FREQ[11]),
            (FREQ[12] - FREQ[12]) + (FREQ[13] - FREQ[12]),
            (FREQ[13] - FREQ[13]) + (FREQ[14] - FREQ[13]),
            (FREQ[14] - FREQ[14]) + (FREQ[15] - FREQ[14]),
            (FREQ[15] - FREQ[15]) + (FREQ[16] - FREQ[15]),
            (FREQ[16] - FREQ[16]) + (FREQ[17] - FREQ[16]),
            (FREQ[17] - FREQ[17]) + (FREQ[18] - FREQ[17]),
            (FREQ[18] - FREQ[18]) + (FREQ[19] - FREQ[18]),
            (FREQ[19] - FREQ[19]) + (FREQ[20] - FREQ[19]),
            (FREQ[20] - FREQ[20]) + (FREQ[21] - FREQ[20]),
            (FREQ[21] - FREQ[21]) + (FREQ[22] - FREQ[21]),
            (FREQ[22] - FREQ[22]),
        ];

    // trapzd
    pol.dot(&weights)
}