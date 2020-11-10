#[macro_use(array)]
use ndarray::prelude::*;
use ndarray::{ArrayView2, Array2, array};
use crate::parameters::*;
use crate::slako_transformations::*;
use std::collections::HashMap;
use crate::molecule::Molecule;

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
pub fn h0_and_s_ab(
    molecule_a: &Molecule,
    molecule_b: &Molecule,
) -> (Array2<f64>, Array2<f64>) {
    let mut h0: Array2<f64> = Array2::zeros((molecule_a.n_orbs, molecule_b.n_orbs));
    let mut s: Array2<f64> = Array2::zeros((molecule_a.n_orbs, molecule_b.n_orbs));
    // iterate over atoms
    let mut mu: usize = 0;
    for (i, (zi, posi)) in molecule_a.iter_atomlist().enumerate() {
        // iterate over orbitals on center i
        for (ni, li, mi) in &molecule_a.valorbs[zi] {
            // iterate over atoms
            let mut nu: usize = 0;
            for (j, (zj, posj)) in molecule_b.iter_atomlist().enumerate() {
                // iterate over orbitals on center j
                for (nj, lj, mj) in &molecule_b.valorbs[zj] {
                    if molecule_a.proximity_matrix[[i, j]] {
                        println!("HALLO mu {}, nu {}, zi {} zj {}", mu, nu, zi, zj);
                        if mu < nu {
                            if zi <= zj {
                                if i != j {
                                    let (r, x, y, z): (f64, f64, f64, f64) = directional_cosines(posi, posj);
                                    s[[mu, nu]] = slako_transformation(r, x, y, z, &molecule_a.skt[&(*zi, *zj)].s_spline, *li, *mi, *lj, *mj);
                                    h0[[mu, nu]] = slako_transformation(r, x, y, z, &molecule_a.skt[&(*zi, *zj)].h_spline, *li, *mi, *lj, *mj);
                                }
                            } else {
                                let (r, x, y, z): (f64, f64, f64, f64) = directional_cosines(posj, posi);
                                s[[mu, nu]] = slako_transformation(r, x, y, z, &molecule_a.skt[&(*zj, *zi)].s_spline, *lj, *mj, *li, *mi);
                                h0[[mu, nu]] = slako_transformation(r, x, y, z, &molecule_a.skt[&(*zj, *zi)].h_spline, *lj, *mj, *li, *mi);
                            }
                        } else if mu == nu {
                            assert_eq!(zi, zj);
                            h0[[mu, nu]] = molecule_a.orbital_energies[zi][&(*ni, *li)];
                            s[[mu, nu]] = 1.0;
                        } else {
                            s[[mu, nu]] = s[[nu, mu]];
                            h0[[mu, nu]] = h0[[nu, mu]];
                        }
                    }
                    nu= nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (s, h0);
}


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
fn test_h0_and_s() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![[0.34215, 1.17577, 0.00000],
                                        [1.31215, 1.17577, 0.00000],
                                        [0.01882, 1.65996, 0.77583]];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers, positions, charge, multiplicity);
    let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s_ab(&mol, &mol);
    let h0_ref: Array2<f64> = array![
    [-0.84692807,  0.        ,  0.        ,  0.        , -0.40019001, -0.40019244],
    [ 0.        , -0.31478407,  0.        ,  0.        , -0.        ,  0.18438378],
    [ 0.        ,  0.        , -0.31478407,  0.        , -0.        ,  0.29544284],
    [ 0.        ,  0.        ,  0.        , -0.31478407,  0.36938167, -0.12312689],
    [-0.40019001, -0.        , -0.        ,  0.36938167, -0.21807977, -0.05387315],
    [-0.40019244,  0.18438378,  0.29544284, -0.12312689, -0.05387315, -0.21807977]];
    let s_ref: Array2<f64> = array![
    [ 1.        ,  0.        ,  0.        ,  0.        ,  0.30749185,  0.3074938 ],
    [ 0.        ,  1.        ,  0.        ,  0.        ,  0.        , -0.19877697],
    [ 0.        ,  0.        ,  1.        ,  0.        ,  0.        , -0.31850542],
    [ 0.        ,  0.        ,  0.        ,  1.        , -0.39821602,  0.1327383 ],
    [ 0.30749185,  0.        ,  0.        , -0.39821602,  1.        ,  0.02680247],
    [ 0.3074938 , -0.19877697, -0.31850542,  0.1327383 ,  0.02680247,  1.        ]];
    println!("s {}", s);
    assert!(s.all_close(&s_ref, 1e-05));
    assert!(h0.all_close(&h0_ref, 1e-05));
}