#[macro_use(array)]
use ndarray::prelude::*;
use crate::molecule::Molecule;
use crate::parameters::*;
use crate::slako_transformations::*;
use approx::AbsDiffEq;
use ndarray::{array, Array2, Array3, ArrayView2, ArrayView3};
use std::collections::HashMap;

///
/// gradients of overlap matrix S and 0-order hamiltonian matrix H0
/// using Slater-Koster Rules
///
/// Parameters:
/// ===========
/// atomlist: list of tuples (Zi,[xi,yi,zi]) of atom types and positions
/// valorbs: list of valence orbitals with quantum numbers (ni,li,mi)
/// SKT: Slater Koster table
/// Mproximity: M[i,j] == 1, if the atoms i and j are close enough
/// so that the gradients for matrix elements
/// between orbitals on i and j should be computed
///
///
pub fn h0_and_s_gradients(
    atomic_numbers: &[u8],
    positions: ArrayView2<f64>,
    n_orbs: usize,
    valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
    proximity_matrix: ArrayView2<bool>,
    skt: &HashMap<(u8, u8), SlaterKosterTable>,
    orbital_energies: &HashMap<u8, HashMap<(i8, i8), f64>>,
) -> (Array3<f64>, Array3<f64>) {
    let n_atoms: usize = atomic_numbers.len();
    let mut grad_h0: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut grad_s: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    let mut s_deriv: Array1<f64> = Array1::zeros((3));
    let mut h0_deriv: Array1<f64> = Array1::zeros((3));
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
                    if proximity_matrix[[i, j]] && mu != nu {
                        if zi <= zj {
                            if i != j {
                                // the hardcoded Slater-Koster rules compute the gradient
                                // with respect to r = posj - posi
                                // but we want the gradient with respect to posi, so an additional
                                // minus sign is introduced
                                let (r, x, y, z): (f64, f64, f64, f64) =
                                    directional_cosines(posi.view(), posj.view());
                                s_deriv = slako_transformation_gradients(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt[&(*zi, *zj)].s_spline,
                                    *li,
                                    *mi,
                                    *lj,
                                    *mj,
                                )
                                .mapv(|x| x * -1.0);

                                h0_deriv = slako_transformation_gradients(
                                    r,
                                    x,
                                    y,
                                    z,
                                    &skt[&(*zi, *zj)].h_spline,
                                    *li,
                                    *mi,
                                    *lj,
                                    *mj,
                                )
                                .mapv(|x| x * -1.0);
                            }
                        } else {
                            // swap atoms if Zj > Zi, since posi and posj are swapped, the gradient
                            // with respect to r = posi - posj equals the gradient with respect to
                            // posi, so no additional minus sign is needed.
                            let (r, x, y, z): (f64, f64, f64, f64) =
                                directional_cosines(posj.view(), posi.view());
                            s_deriv = slako_transformation_gradients(
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
                            h0_deriv = slako_transformation_gradients(
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

                        grad_s
                            .slice_mut(s![3 * i..3 * (i + 1), mu, nu])
                            .assign(&s_deriv);
                        grad_h0
                            .slice_mut(s![3 * i..3 * (i + 1), mu, nu])
                            .assign(&h0_deriv);
                        // S and H0 are hermitian/symmetric
                        grad_s
                            .slice_mut(s![3 * i..3 * (i + 1), nu, mu])
                            .assign(&s_deriv);
                        grad_h0
                            .slice_mut(s![3 * i..3 * (i + 1), nu, mu])
                            .assign(&h0_deriv);
                    }
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return (grad_s, grad_h0);
}
