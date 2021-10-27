use crate::parametrization::slako::SlaterKoster;
use crate::parametrization::slako_transformations::*;
use crate::{AtomSlice, SorH, PROXIMITY_CUTOFF};
use itertools::{iproduct, multizip};
use nalgebra::{Vector3, Vector4};
use ndarray::prelude::*;
use rayon::prelude::*;
use soa_derive::soa_zip;
use std::cmp::Ordering;

fn directions(atoms_a: AtomSlice, atoms_b: AtomSlice) -> Vec<Vector3<f64>> {
    iproduct!(atoms_a.xyz, atoms_b.xyz)
        .map(|(xyz_i, xyz_j)| xyz_i - xyz_j)
        .collect::<Vec<Vector3<f64>>>()
}

fn directions_to_proximities(directions: &[Vector3<f64>]) -> Vec<bool> {
    directions
        .par_iter()
        .map(|dxyz| dxyz.norm() < PROXIMITY_CUTOFF)
        .collect::<Vec<bool>>()
}

/// Computes the directional Cosines for the difference vector.
fn directional_cosines(dxyz: &Vector3<f64>) -> Vector4<f64> {
    let r: f64 = dxyz.norm();
    // directional cosines
    match r {
        r if r > 0.0 => Vector4::new(dxyz.x / r, dxyz.y / r, dxyz.z / r, r),
        _ => Vector4::new(0.0, 0.0, 1.0, r),
    }
}

impl SlaterKoster {
    pub fn get_h0_and_s(&self, atoms: AtomSlice, n_orbs: usize) -> (Array2<f64>, Array2<f64>) {
        let dirs = directions(atoms, atoms);
        let get_proxs = || directions_to_proximities(&dirs);
        let get_cosines = || {
            dirs.iter()
                .map(directional_cosines)
                .collect::<Vec<Vector4<f64>>>()
        };

        let (proxs, cosines) = rayon::join(get_proxs, get_cosines);

        let get_olap = || {
            Array2::from_shape_vec(
                (n_orbs, n_orbs),
                self.get_one_elec_int((atoms, atoms), &proxs, &cosines, SorH::S),
            )
            .unwrap()
        };
        let get_h0 = || {
            Array2::from_shape_vec(
                (n_orbs, n_orbs),
                self.get_one_elec_int((atoms, atoms), &proxs, &cosines, SorH::H0),
            )
            .unwrap()
        };

        rayon::join(get_olap, get_h0)
    }

    fn get_one_elec_int(
        &self,
        atoms: (AtomSlice, AtomSlice),
        proxs: &[bool],
        cosines: &[Vector4<f64>],
        s_or_h: SorH,
    ) -> Vec<f64> {
        let atoms_a_iter = soa_zip!(atoms.0, [kind, valorbs]);
        let atoms_b_iter = soa_zip!(atoms.1, [kind, valorbs]);
        let iter = multizip((iproduct!(atoms_a_iter, atoms_b_iter), proxs, cosines));

        iter.map(|(((ki, vi), (kj, vj)), pij, cos)| {
            iproduct!(vi, vj).map(move |(orbi, orbj)| match (pij, ki.cmp(kj)) {
                (true, Ordering::Greater) => {
                    slako_transformation(cos, self.get(*kj, *ki).get_splines(s_or_h), orbj, orbi)
                }
                (true, _) => {
                    slako_transformation(cos, self.get(*ki, *kj).get_splines(s_or_h), orbi, orbj)
                }
                _ => 0.0,
            })
        })
        .flatten()
        .collect()
    }
}

// // gradients of overlap matrix S and 0-order hamiltonian matrix H0
// // using Slater-Koster Rules
// //
// // Parameters:
// // ===========
// // atomlist: list of tuples (Zi,[xi,yi,zi]) of atom types and positions
// // valorbs: list of valence orbitals with quantum numbers (ni,li,mi)
// // SKT: Slater Koster table
// // Mproximity: M[i,j] == 1, if the atoms i and j are close enough
// // so that the gradients for matrix elements
// // between orbitals on i and j should be computed
// pub fn h0_and_s_gradients(&self, atoms: &[Atom], n_orbs: usize) -> (Array3<f64>, Array3<f64>) {
//     let n_atoms: usize = atoms.len();
//     let mut grad_h0: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
//     let mut grad_s: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
//     // iterate over atoms
//     let mut mu: usize = 0;
//     for (i, atomi) in atoms.iter().enumerate() {
//         // iterate over orbitals on center i
//         for orbi in atomi.valorbs.iter() {
//             // iterate over atoms
//             let mut nu: usize = 0;
//             for (j, atomj) in atoms.iter().enumerate() {
//                 // iterate over orbitals on center j
//                 for orbj in atomj.valorbs.iter() {
//                     if (atomi-atomj).norm() < PROXIMITY_CUTOFF && mu != nu {
//                         let mut s_deriv: Array1<f64> = Array1::zeros([3]);
//                         let mut h0_deriv: Array1<f64> = Array1::zeros([3]);
//                         if atomi <= atomj {
//                             if i != j {
//                                 // the hardcoded Slater-Koster rules compute the gradient
//                                 // with respect to r = posj - posi
//                                 // but we want the gradient with respect to posi, so an additional
//                                 // minus sign is introduced
//                                 let (r, x, y, z): (f64, f64, f64, f64) =
//                                     directional_cosines(&atomi.xyz, &atomj.xyz);
//                                 s_deriv = -1.0 * slako_transformation_gradients(
//                                     r,
//                                     x,
//                                     y,
//                                     z,
//                                     &self.get(atomi.kind, atomj.kind).s_spline,
//                                     orbi.l,
//                                     orbi.m,
//                                     orbj.l,
//                                     orbj.m,
//                                 );
//                                 h0_deriv = -1.0 * slako_transformation_gradients(
//                                     r,
//                                     x,
//                                     y,
//                                     z,
//                                     &self.get(atomi.kind, atomj.kind).h_spline,
//                                     orbi.l,
//                                     orbi.m,
//                                     orbj.l,
//                                     orbj.m,
//                                 );
//                             }
//                         } else {
//                             // swap atoms if Zj > Zi, since posi and posj are swapped, the gradient
//                             // with respect to r = posi - posj equals the gradient with respect to
//                             // posi, so no additional minus sign is needed.
//                             let (r, x, y, z): (f64, f64, f64, f64) =
//                                 directional_cosines(&atomj.xyz, &atomi.xyz);
//                             s_deriv = slako_transformation_gradients(
//                                 r,
//                                 x,
//                                 y,
//                                 z,
//                                 &self.get(atomi.kind, atomj.kind).s_spline,
//                                 orbj.l,
//                                 orbj.m,
//                                 orbi.l,
//                                 orbi.m,
//                             );
//                             h0_deriv = slako_transformation_gradients(
//                                 r,
//                                 x,
//                                 y,
//                                 z,
//                                 &self.get(atomi.kind, atomj.kind).h_spline,
//                                 orbj.l,
//                                 orbj.m,
//                                 orbi.l,
//                                 orbi.m,
//                             );
//                         }
//
//                         grad_s
//                             .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
//                             .assign(&s_deriv);
//                         grad_h0
//                             .slice_mut(s![(3 * i)..(3 * i + 3), mu, nu])
//                             .assign(&h0_deriv);
//                         // S and H0 are hermitian/symmetric
//                         grad_s
//                             .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
//                             .assign(&s_deriv);
//                         grad_h0
//                             .slice_mut(s![(3 * i)..(3 * i + 3), nu, mu])
//                             .assign(&h0_deriv);
//                     }
//                     nu = nu + 1;
//                 }
//             }
//             mu = mu + 1;
//         }
//     }
//     return (grad_s, grad_h0);
// }
