use ndarray::{Array, Array1, Array2, ArrayView2};
use std::f64::consts::PI;
use crate::dispersion::model::Molecule;
use crate::dispersion::cutoff::get_lattice_points_rep_3d;
use ndarray_linalg::Inverse;

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