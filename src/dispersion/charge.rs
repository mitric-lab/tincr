use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2, Array3};
use ndarray_linalg::{Inverse, Determinant, Norm};
use std::f64::consts::PI;
use crate::dispersion::model::Molecule;
use crate::dispersion::cutoff::get_lattice_points_rep_3d;
use crate::dispersion::wignerseitz::WignerSeitzCell;
use crate::dispersion::auxliary_functions::{lat_to_array, spread};
use libm::{erf, sin, cos};
use std::ops::{AddAssign, SubAssign};


const TWOPI: f64 = 2.0 * PI;
const SQRTPI: f64 = 1.77245385090551602729816748334114518;
const SQRT2PI: f64 = 0.79788456080286535587989211986876373;
const EPS: f64 = 1.4901161193847656E-008; // sqrt(EPSILON)
const REG: f64 = 1.0000000000000000E-014;

pub struct MchrgModel {
    rad: Array1<f64>,
    chi: Array1<f64>,
    eta: Array1<f64>,
    kcn: Array1<f64>,
}

pub struct VrhsResult {
    xvec: Array1<f64>,
    dxdcn: Option<Array1<f64>>,
}

impl MchrgModel {
    pub fn new(
        chi: Array1<f64>,
        rad: Array1<f64>,
        eta: Array1<f64>,
        kcn: Array1<f64>,
    ) -> MchrgModel {
        let chrg_model = MchrgModel {
            rad: rad,
            chi: chi,
            eta: eta,
            kcn: kcn,
        };

        return chrg_model;
    }

    pub fn get_vrhs(
        &self,
        mol: &Molecule,
        cn: Array1<f64>,
        derivative: bool,
    ) -> VrhsResult {
        if derivative {
            let mut xvec: Array1<f64> = Array::zeros(mol.n_atoms + 1);
            let mut dxdcn: Array1<f64> = Array::zeros(mol.n_atoms + 1);

            // Parallise this loop
            for iat in 0..mol.n_atoms {
                let izp = mol.id[iat];
                let tmp = self.kcn[izp] / (cn[iat] + REG).sqrt();
                xvec[iat] = -self.chi[izp] + tmp*cn[iat];
                dxdcn[iat] = 0.5*tmp;
            }
            xvec[mol.n_atoms] = mol.charge as f64;
            // dxdcn[mol.n_atoms] = 0.0

            VrhsResult {
                xvec: xvec,
                dxdcn: Some(dxdcn),
            }
        } else {
            let mut xvec: Array1<f64> = Array::zeros(mol.n_atoms);

            // Parallise this loop
            for iat in 0..mol.n_atoms {
                let izp = mol.id[iat];
                let tmp = self.kcn[izp] / (cn[iat] + REG).sqrt();
                xvec[iat] = -self.chi[izp] + tmp*cn[iat];
            }
            xvec[mol.n_atoms] = mol.charge as f64;

            VrhsResult {
                xvec: xvec,
                dxdcn: None,
            }
        }
    }

    fn get_amat_0d(
        &self,
        mol: &Molecule,
    ) -> Array2<f64> {
        let mut amat: Array2<f64> = Array::zeros((mol.n_atoms + 1, mol.n_atoms + 1));

        // parallise this loop
        for iat in 0..mol.n_atoms {
            let izp = mol.id[iat];
            for jat in 0..iat {
                let jzp = mol.id[jat];
                let vec: Array1<f64> = &mol.positions.slice(s![jat, ..]) - &mol.positions.slice(s![iat, ..]);
                let r2 = vec[0].powi(2) + vec[1].powi(2) + vec[2].powi(2);
                let gam = 1.0/(self.rad[izp].powi(2) + self.rad[jzp].powi(2));
                let tmp = erf((r2*gam).sqrt()) / r2.sqrt();
                amat[[iat, jat]] = amat[[iat, jat]] + tmp;
                amat[[jat, iat]] = amat[[jat, iat]] + tmp;
            }
            let tmp = self.eta[izp] + SQRT2PI / self.rad[izp];
            amat[[iat, iat]] = amat[[iat, iat]] + tmp;
        }

        amat.slice_mut(s![.., mol.n_atoms]).fill(1.0);
        amat.slice_mut(s![mol.n_atoms, ..]).fill(1.0);
        amat[[mol.n_atoms, mol.n_atoms]] = 0.0;

        return amat;
    }

    fn get_amat_3d(
        &self,
        mol: &Molecule,
        wsc: &WignerSeitzCell,
        alpha: f64,
    ) -> Array2<f64> {
        let mut amat: Array2<f64> = Array::zeros((mol.n_atoms + 1, mol.n_atoms + 1));

        let lat_a: Array2<f64> = lat_to_array(&(mol.lattice));
        let vol = lat_a.det().unwrap().abs();
        let dtrans: Array2<f64> = get_dir_trans(lat_a.view());
        let rtrans: Array2<f64> = get_rec_trans(lat_a.view());

        // parallise this loop
        for iat in 0..mol.n_atoms {
            let izp = mol.id[iat];
            for jat in 0..iat {
                let jzp = mol.id[jat];
                let gam = 1.0/(self.rad[izp].powi(2) + self.rad[jzp].powi(2));
                let wsw = 1.0 / (wsc.nimg[[iat, jat]] as f64);
                for img in 0..wsc.nimg[[iat, jat]] {
                    let vec: Array1<f64> = &mol.positions.slice(s![iat, ..])
                                         - &mol.positions.slice(s![jat, ..])
                                         - &wsc.trans.slice(s![wsc.tridx[[iat, jat, img]], ..]);
                    let dtmp = get_amat_dir_3d(vec.view(), gam, alpha, dtrans.view());
                    let rtmp = get_amat_rec_3d(vec.view(), vol, alpha, rtrans.view());
                    amat[[iat, jat]] = amat[[iat, jat]] + (dtmp + rtmp) * wsw;
                    amat[[jat, iat]] = amat[[jat, iat]] + (dtmp + rtmp) * wsw;
                }

                let dtmp = self.eta[izp] + SQRT2PI / self.rad[izp] - 2.0 * alpha / SQRTPI;
                amat[[iat, iat]] = amat[[iat, iat]] + dtmp;
            }
        }

        amat.slice_mut(s![.., mol.n_atoms]).fill(1.0);
        amat.slice_mut(s![mol.n_atoms, ..]).fill(1.0);
        amat[[mol.n_atoms, mol.n_atoms]] = 0.0;

        return amat;
    }

    fn get_damat_0d(
        &self,
        mol: &Molecule,
        qvec: Array1<f64>,
    ) -> (Array3<f64>, Array3<f64>, Array2<f64>) {
        let mut atrace: Array2<f64> = Array::zeros((mol.n_atoms, 3));
        let mut dadr: Array3<f64> = Array::zeros((mol.n_atoms, mol.n_atoms, 3));
        let mut dadL: Array3<f64> = Array::zeros((mol.n_atoms, mol.n_atoms, 3));

        // Parallelise this loop
        for iat in 0..mol.n_atoms{
            let izp = mol.id[iat];
            for jat in 0..iat {
                let jzp = mol.id[jat];
                let vec: Array1<f64> = &mol.positions.slice(s![iat, ..]) - &mol.positions.slice(s![jat, ..]);
                let r2 = vec[0].powi(2) + vec[1].powi(2) + vec[2].powi(2);
                let gam = 1.0/(self.rad[izp].powi(2) + self.rad[jzp].powi(2)).sqrt();
                let arg = gam * gam * r2;
                let dtmp: f64 = 2.0*gam*(-arg).exp()/(SQRTPI*r2) - erf(arg.sqrt())/(r2*r2.sqrt());
                let dG: Array1<f64> = dtmp * &vec;
                let dS: Array2<f64> = &spread(&dG, 0, 3) * &spread(&vec, 1, 3);
                atrace.slice_mut(s![iat, ..]).add_assign(&(&dG*qvec[jat]));
                atrace.slice_mut(s![jat, ..]).sub_assign(&(&dG*qvec[iat]));
                dadr.slice_mut(s![jat, iat, ..]).assign(&(&dG*qvec[iat]));
                dadr.slice_mut(s![iat, jat, ..]).assign(&(-(&dG*qvec[jat])));
                dadL.slice_mut(s![jat, .., ..]).add_assign(&(&dS*qvec[iat]));
                dadL.slice_mut(s![iat, .., ..]).add_assign(&(&dS*qvec[jat]));
            }
        }

        return (dadr, dadL, atrace);
    }

    fn get_damat_3d(
        &self,
        mol: &Molecule,
        wsc: &WignerSeitzCell,
        alpha: f64,
        qvec: ArrayView1<f64>,
    ) -> (Array3<f64>, Array3<f64>, Array2<f64>) {
        let mut atrace: Array2<f64> = Array::zeros((mol.n_atoms, 3));
        let mut dadr: Array3<f64> = Array::zeros((mol.n_atoms, mol.n_atoms, 3));
        let mut dadL: Array3<f64> = Array::zeros((mol.n_atoms, mol.n_atoms, 3));

        let lat_a: Array2<f64> = lat_to_array(&(mol.lattice));
        let vol: f64 = lat_a.det().unwrap().abs();
        let dtrans: Array2<f64> = get_dir_trans(lat_a.view());
        let rtrans: Array2<f64> = get_rec_trans(lat_a.view());

        // parallelise this loop
        for iat in 0..mol.n_atoms{
            let izp = mol.id[iat];
            for jat in 0..iat {
                let jzp = mol.id[jat];
                let mut dG: Array1<f64> = Array::zeros(3);
                let mut dS: Array2<f64> = Array::zeros((3, 3));
                let gam = 1.0 / (self.rad[izp].powi(2) + self.rad[jzp].powi(2)).sqrt();
                let wsw = 1.0 / (wsc.nimg[[iat, jat]] as f64);
                for img in 0..wsc.nimg[[iat, jat]] {
                    let vec: Array1<f64> = &mol.positions.slice(s![iat, ..])
                                         - &mol.positions.slice(s![jat, ..])
                                         - &wsc.trans.slice(s![wsc.tridx[[iat, jat, img]], ..]);
                    let (dGd, dSd): (Array1<f64>, Array2<f64>)
                        = get_damat_dir_3d(vec.view(), gam, alpha, dtrans.view());
                    let (dGr, dSr): (Array1<f64>, Array2<f64>)
                        = get_damat_rec_3d(vec.view(), vol, alpha, rtrans.view());
                    dG += &((&dGd + &dGr) * wsw);
                    dS += &((&dSd + &dSr) * wsw);
                }
                atrace.slice_mut(s![iat, ..]).add_assign(&(&dG*qvec[jat]));
                atrace.slice_mut(s![jat, ..]).sub_assign(&(&dG*qvec[iat]));
                dadr.slice_mut(s![jat, iat, ..]).add_assign(&(&dG*qvec[iat]));
                dadr.slice_mut(s![iat, jat, ..]).sub_assign(&(&dG*qvec[jat]));
                dadL.slice_mut(s![jat, .., ..]).add_assign(&(&dS*qvec[iat]));
                dadL.slice_mut(s![iat, .., ..]).add_assign(&(&dS*qvec[jat]));
            }

            let mut dS: Array2<f64> = Array::zeros((3, 3));
            let gam = 1.0 / (2.0 * self.rad[izp].powi(2)).sqrt();
            let wsw = 1.0 / (wsc.nimg[[iat, iat]] as f64);
            for img in 0..wsc.nimg[[iat, iat]] {
                let vec: Array1<f64> = wsc.trans.slice(s![wsc.tridx[[iat, iat, img]], ..]).to_owned();
                let (dGd, dSd): (Array1<f64>, Array2<f64>)
                    = get_damat_dir_3d(vec.view(), gam, alpha, dtrans.view());
                let (dGr, dSr): (Array1<f64>, Array2<f64>)
                    = get_damat_rec_3d(vec.view(), vol, alpha, rtrans.view());
                dS += &((&dSd + &dSr) * wsw);
            }
            dadL.slice_mut(s![iat, .., ..]).add_assign(&(&dS*qvec[iat]));
        }

        return (dadr, dadL, atrace);
    }

    // fn solve(
    //     &self,
    //     mol: &Molecule,
    //
    // ) -> () {
    //     let ndim = mol.n_atoms + 1;
    //     let (wsc, alpha): (Option<WignerSeitzCell>, Option<f64>)
    //         = if mol.periodic.iter().any(|&x| x) {
    //             let wsc_tmp: WignerSeitzCell = WignerSeitzCell::new(mol);
    //             let alpha_tmp =
    //     }
    // }
}

fn get_dir_trans(
    lattice: ArrayView2<f64>,
) -> Array2<f64> {
    let rep: [usize; 3] = [2, 2, 2];

    let trans: Array2<f64> = get_lattice_points_rep_3d(lattice, &rep, true);

    return trans;
}

fn get_rec_trans(
    lattice: ArrayView2<f64>,
) -> Array2<f64> {
    let rep: [usize; 3] = [2, 2, 2];
    let rec_lat: Array2<f64> = TWOPI * lattice.inv().unwrap().reversed_axes();

    let trans: Array2<f64> = get_lattice_points_rep_3d(rec_lat.view(), &rep, false);

    return trans;
}

fn get_amat_dir_3d(
    rij: ArrayView1<f64>,
    gam: f64,
    alp: f64,
    trans: ArrayView2<f64>,
) -> f64 {
    let mut amat: f64 = 0.0;

    for itr in 0..trans.dim().0 {
        let vec: Array1<f64> = &rij + &trans.slice(s![itr, ..]);
        let r1 = vec.norm();
        if r1 < EPS {
            continue;
        }
        let tmp = erf(gam*r1)/r1 - erf(alp*r1)/r1;
        amat = amat + tmp;
    }

    return amat;
}

fn get_amat_rec_3d(
    rij: ArrayView1<f64>,
    vol: f64,
    alp: f64,
    trans: ArrayView2<f64>
) -> f64 {
    let mut amat: f64 = 0.0;
    let fac = 4.0*PI/vol;

    for itr in 0..trans.dim().0 {
        let vec: Array1<f64> = trans.slice(s![itr, ..]).to_owned();
        let g2: f64 = vec.dot(&vec);
        if g2 < EPS {
            continue;
        }
        let tmp = cos(rij.dot(&vec)) * fac * (-0.25*g2/(alp*alp)).exp()/g2;
        amat = amat + tmp;
    }

    return amat;
}

fn get_damat_dir_3d(
    rij: ArrayView1<f64>,
    gam: f64,
    alp: f64,
    trans: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let mut dg: Array1<f64> = Array::zeros(3);
    let mut ds: Array2<f64> = Array::zeros((3, 3));

    let gam2 = gam*gam;
    let alp2 = alp*alp;

    for itr in 0..trans.dim().0 {
        let vec: Array1<f64> = &rij + &trans.slice(s![itr, ..]);
        let r1: f64 = vec.norm();
        if r1 < EPS {
            continue;
        }
        let r2 = r1 * r1;
        let gtmp = 2.0*gam*(-r2*gam2).exp()/(SQRTPI*r2) - erf(r1*gam)/(r2*r1);
        let atmp = -2.0*alp*(-r2*alp2).exp()/(SQRTPI*r2) + erf(r1*alp)/(r2*r1);
        dg += &((gtmp + atmp) * &vec);
        ds += &((gtmp + atmp)
            * &(&spread(&vec, 0, 3)
              * &spread(&vec, 1, 3)));
    }

    return (dg, ds);
}

fn get_damat_rec_3d(
    rij: ArrayView1<f64>,
    vol: f64,
    alp: f64,
    trans: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let mut dg: Array1<f64> = Array::zeros(3);
    let mut ds: Array2<f64> = Array::zeros((3, 3));

    let unity: Array2<f64> = Array::eye(3);
    let fac = 4.0*PI/vol;
    let alp2 = alp*alp;

    for itr in 0..trans.dim().0 {
        let vec: Array1<f64> = trans.slice(s![itr, ..]).to_owned();
        let g2: f64 = vec.dot(&vec);
        if g2 < EPS {
            continue;
        }
        let gv: f64 = rij.dot( &vec);
        let etmp = fac * (-0.25*g2/alp2).exp()/g2;
        let dtmp = -sin(gv) * etmp;
        dg += &(dtmp * &vec);
        ds += &(etmp * cos(gv) *
            &((2.0/g2 + 0.5/alp2)
                * &spread(&vec, 0, 3) * &spread(&vec, 1, 3)
            - &unity));
    }

    return (dg, ds);
}

