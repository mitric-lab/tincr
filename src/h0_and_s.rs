use ndarray::{ArrayView2, Array};
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
    molecule_a: Molecule,
    molecule_b: Molecule,
    valorbs: HashMap<u8, (u8, u8, u8)>,
    skt: HashMap<(u8, u8), SlaterKosterTable>,
    orbital_energies: HashMap<u8, HashMap<(u8, u8), f64>>,
    m_proximity: ArrayView2<u8>,
) -> (Array<f64, Ix2>, Array<f64, Ix2>) {
    let mut h0: Array<f64, Ix2> = Array::zeros((dim_a, dim_b));
    let mut s: Array<f64, Ix2> = Array::zeros((dim_a, dim_b));
    // iterate over atoms
    let mu: u32 = 0;
    for (i, (zi, posi)) in molecule_a.iter_atomlist().enumerate() {
        // iterate over orbitals on center i
        for (ni, li, mi) in valorbs[zi] {
            // iterate over atoms
            let nu: u32 = 0;
            for (j, (zj, posj)) in molecule_b.iter_atomlist().enumerate() {
                // iterate over orbitals on center j
                for (nj, lj, mj) in valorbs[zj] {
                    if m_proximity[[i, j]] == 1 {
                        if mu < nu {
                            if zi <= zj {
                                if i != j {
                                    let (r, x, y, z): (f64, f64, f64, f64) = directional_cosines(&posi, &posj);
                                    s[[mu, nu]] = slako_transformation(r, x, y, z, skt[(zi, zj)].s_spline, li, mi, lj, mj);
                                    h0[[mu, nu]] = slako_transformation(r, x, y, z, skt[(zi, zj)].h_spline, li, mi, lj, mj);
                                }
                            } else {
                                let (r, x, y, z): (f64, f64, f64, f64) = directional_cosines(&posj, &posi);
                                s[[mu, nu]] = slako_transformation(r, x, y, z, skt[(zj, zi)].s_spline, lj, mj, li, mi);
                                h0[[mu, nu]] = slako_transformation(r, x, y, z, skt[(zj, zi)].h_spline, lj, mj, li, mi);
                            }
                        } else if mu == nu {
                            assert_eq!(zi, zj);
                            s[[mu, nu]] = orbital_energies[zi][(ni, li)];
                            h0[[mu, nu]] = 1.0;
                        } else {
                            s[[mu, nu]] = s[[nu, mu]];
                            h0[[mu, nu]] = h0[[nu, mu]];
                        }
                    }
                    let nu: u32 = nu + 1;
                }
            }
            let mu: u32 = mu + 1;
        }
    }
    return (s, h0);
}