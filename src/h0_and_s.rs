#[macro_use(array)]
use ndarray::prelude::*;
use crate::molecule::Molecule;
use crate::parameters::*;
use crate::slako_transformations::*;
use approx::AbsDiffEq;
use ndarray::{array, Array2, ArrayView2};
use std::collections::HashMap;

///
/// compute Hamiltonian and overlap matrix elements between two sets of atoms. If the sets
/// A and B contain exactly the same structure AisB should be set to True to ensure that
/// the diagonal elements of the Hamiltonian are replaced by the correct on-site energies.
///
/// Parameters:
/// ===========
/// dim_a: number of orbitals of all the atoms in set A
/// dim_b:  ''                                 in set B
/// atomlist_a, atomlist_b: list of (Zi,(xi,yi,zi)) for each atom
///
///
pub fn h0_and_s(
    atomic_numbers: &[u8],
    positions: ArrayView2<f64>,
    n_orbs: usize,
    valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
    proximity_matrix: ArrayView2<bool>,
    skt: &HashMap<(u8, u8), SlaterKosterTable>,
    orbital_energies: &HashMap<u8, HashMap<(i8, i8), f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let mut h0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    let mut s: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    // iterate over atoms
    let mut mu: usize = 0;
    for (i, (zi, posi)) in atomic_numbers
        .iter()
        .zip(positions.outer_iter())
        .enumerate()
    {
        // iterate over orbitals on center i
        for (ni, li, mi) in &valorbs[zi] {
            // iterate over atoms
            let mut nu: usize = 0;
            for (j, (zj, posj)) in atomic_numbers
                .iter()
                .zip(positions.outer_iter())
                .enumerate()
            {
                // iterate over orbitals on center j
                for (nj, lj, mj) in &valorbs[zj] {
                    if proximity_matrix[[i, j]] {
                        if mu < nu {
                            if zi <= zj {
                                if i != j {
                                    let (r, x, y, z): (f64, f64, f64, f64) =
                                        directional_cosines(posi, posj);
                                    s[[mu, nu]] = slako_transformation(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &skt[&(*zi, *zj)].s_spline,
                                        *li,
                                        *mi,
                                        *lj,
                                        *mj,
                                    );
                                    h0[[mu, nu]] = slako_transformation(
                                        r,
                                        x,
                                        y,
                                        z,
                                        &skt[&(*zi, *zj)].h_spline,
                                        *li,
                                        *mi,
                                        *lj,
                                        *mj,
                                    );
                                }
                            } else {
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(posj, posi);
                                s[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt[&(*zj, *zi)].s_spline,
                                    *lj,
                                    *mj,
                                    *li,
                                    *mi,
                                );
                                h0[[mu, nu]] = slako_transformation(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt[&(*zj, *zi)].h_spline,
                                    *lj,
                                    *mj,
                                    *li,
                                    *mi,
                                );
                            }
                        } else if mu == nu {
                            assert_eq!(zi, zj);
                            h0[[mu, nu]] = orbital_energies[zi][&(*ni, *li)];
                            s[[mu, nu]] = 1.0;
                        } else {
                            s[[mu, nu]] = s[[nu, mu]];
                            h0[[mu, nu]] = h0[[nu, mu]];
                        }
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (s, h0);
}

/// Test of H0 and S construction for a water molecule. The xyz geometry of the
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
fn test_h0_and_s() {
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
    let mol: Molecule = Molecule::new(
        atomic_numbers.clone(),
        positions.clone(),
        charge,
        multiplicity,
    );
    let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(
        &atomic_numbers,
        positions.view(),
        mol.calculator.n_orbs,
        &mol.calculator.valorbs,
        mol.proximity_matrix.view(),
        &mol.calculator.skt,
        &mol.calculator.orbital_energies,
    );
    let h0_ref: Array2<f64> = array![
        [-0.84692807, 0., 0., 0., -0.40019001, -0.40019244],
        [0., -0.31478407, 0., 0., -0., 0.18438378],
        [0., 0., -0.31478407, 0., -0., 0.29544284],
        [0., 0., 0., -0.31478407, 0.36938167, -0.12312689],
        [-0.40019001, -0., -0., 0.36938167, -0.21807977, -0.05387315],
        [
            -0.40019244,
            0.18438378,
            0.29544284,
            -0.12312689,
            -0.05387315,
            -0.21807977
        ]
    ];
    let s_ref: Array2<f64> = array![
        [1., 0., 0., 0., 0.30749185, 0.3074938],
        [0., 1., 0., 0., 0., -0.19877697],
        [0., 0., 1., 0., 0., -0.31850542],
        [0., 0., 0., 1., -0.39821602, 0.1327383],
        [0.30749185, 0., 0., -0.39821602, 1., 0.02680247],
        [
            0.3074938,
            -0.19877697,
            -0.31850542,
            0.1327383,
            0.02680247,
            1.
        ]
    ];
    println!("s {}", s);
    assert!(s.abs_diff_eq(&s_ref, 1e-05));
    assert!(h0.abs_diff_eq(&h0_ref, 1e-05));
}
