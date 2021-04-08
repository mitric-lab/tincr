use ndarray::{s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2};
use crate::dispersion::model::Molecule;
use crate::dispersion::cutoff::get_lattice_points_cutoff;
use crate::dispersion::auxliary_functions::argminf;

const THR: f64 = 1.4901161193847656E-008; // sqrt(EPSILON)
const TOL: f64 = 0.01;

pub struct WignerSeitzCell {
    pub nimg: Array2<usize>,
    pub tridx: Array3<usize>,
    pub trans: Array2<f64>,
}

impl WignerSeitzCell {
    pub fn new(
        mol: &Molecule,
    ) -> WignerSeitzCell {
        let trans: Array2<f64> = get_lattice_points_cutoff(&mol.periodic, &mol.lattice, THR);
        let ntr = trans.dim().0;

        let mut nimg: Array2<usize> = Array::zeros((mol.n_atoms, mol.n_atoms));
        let mut tridx: Array3<usize> = Array::zeros((mol.n_atoms, mol.n_atoms, ntr));

        // parallise this loop
        for iat in 0..mol.n_atoms {
            for jat in 0..mol.n_atoms {
                let vec: Array1<f64> = &mol.positions.slice(s![iat, ..]) - &mol.positions.slice(s![jat, ..]);
                let (nimg_tmp, tridx_tmp) = get_pairs(trans.view(), vec.view());
                nimg[[iat, jat]] = nimg_tmp;
                tridx.slice_mut(s![iat, jat, ..]).assign(&tridx_tmp);
            }
        }

        WignerSeitzCell {
            nimg: nimg,
            tridx: tridx,
            trans: trans,
        }

    }
}

fn get_pairs(
    trans: ArrayView2<f64>,
    rij: ArrayView1<f64>,
) -> (usize, Array1<usize>) {
    let mut iws: usize = 0;
    let mut img: usize = 0;
    let mut dist: Array1<f64> = Array::zeros(trans.dim().0);
    let mut list: Array1<usize> = Array::from_elem(trans.dim().0, 0);
    let mut mask: Array1<bool> = Array::from_elem(trans.dim().0, true);

    for itr in 0..trans.dim().0 {
        let vec: Array1<f64> = &rij - &trans.slice(s![itr, ..]);
        let r2 = vec[0].powi(2) + vec[1].powi(2) + vec[2].powi(2);
        if r2 < THR {
            continue;
        }
        dist[img] = r2;
        img = img + 1;
    }

    if img == 0 {
        return (iws, list);
    }

    let (pos, _) = argminf(dist.slice(s![0..img]));
    let pos = pos.unwrap();

    let r2 = dist[pos];
    mask[pos] = false;

    list[iws] = pos;
    iws = 1;
    if img <= iws {
        return (iws, list);
    }

    loop {
        let dist_tmp: Array1<f64> = Array::from(dist.iter()
            .zip(&mask).filter(|(a, b)| **b)
            .unzip::<&f64, &bool, Vec<f64>, Vec<bool>>().0);
        let (pos, _) = argminf(dist_tmp.view());

        if pos.is_none() {
            break;
        }

        let pos = pos.unwrap();
        if (dist[pos] - r2).abs() > TOL {
            break;
        }
        mask[pos] = false;
        list[iws] = pos;
        iws = iws + 1;
    }

    return (iws, list);
}