use crate::parametrization::slako::SlaterKoster;
use crate::parametrization::slako_transformations::*;
use crate::{AtomSlice, PROXIMITY_CUTOFF};
use itertools::{iproduct, multizip};
use nalgebra::Vector3;
use ndarray::prelude::*;
use soa_derive::soa_zip;

fn directions(atoms_a: AtomSlice, atoms_b: AtomSlice) -> Vec<Vector3<f64>> {
    atoms_a
        .xyz
        .iter()
        .map(|xyz_i| atoms_b.xyz.iter().map(move |xyz_j| xyz_i - xyz_j))
        .flatten()
        .collect::<Vec<Vector3<f64>>>()
}

fn directions_to_proximities(directions: &[Vector3<f64>]) -> Vec<bool> {
    directions
        .iter()
        .map(|dxyz| dxyz.norm() < PROXIMITY_CUTOFF)
        .collect::<Vec<bool>>()
}

/// Computes the directional Cosines for the difference vector.
fn directional_cosines(dxyz: &Vector3<f64>) -> (f64, f64, f64, f64) {
    let r: f64 = dxyz.norm();
    // directional cosines
    let (x, y, z): (f64, f64, f64) = match r {
        r if r > 0.0 => (dxyz.x / r, dxyz.y / r, dxyz.z / r),
        _ => (0.0, 0.0, 1.0),
    };
    (r, x, y, z)
}

impl SlaterKoster {
    pub fn core_hamiltonian_ab(
        &self,
        atoms_a: AtomSlice,
        atoms_b: AtomSlice,
        n_orbs_a: usize,
        n_orbs_b: usize,
    ) -> Array2<f64> {
        let dirs = directions(atoms_a, atoms_b);
        let proxs = directions_to_proximities(&dirs);
        let cosines = dirs
            .iter()
            .map(directional_cosines)
            .collect::<Vec<(f64, f64, f64, f64)>>();
        let atoms_a_iter = soa_zip!(atoms_a, [kind, valorbs, number]);
        let atoms_b_iter = soa_zip!(atoms_b, [kind, valorbs, number]);
        let iter = multizip((iproduct!(atoms_a_iter, atoms_b_iter), &proxs, &cosines));

        let h: Vec<f64> = iter
            .map(|(((ki, vi, ni), (kj, vj, nj)), pij, cos)| {
                iproduct!(vi, vj).map(move |(orbi, orbj)| match pij {
                    true if ni <= nj => slako_transformation(
                        cos.0,
                        cos.1,
                        cos.2,
                        cos.3,
                        &self.get(*ki, *kj).h_spline,
                        orbi.l,
                        orbi.m,
                        orbj.l,
                        orbj.m,
                    ),
                    true => slako_transformation(
                        cos.0,
                        cos.1,
                        cos.2,
                        cos.3,
                        &self.get(*kj, *ki).h_spline,
                        orbj.l,
                        orbj.m,
                        orbi.l,
                        orbi.m,
                    ),
                    false => 0.0,
                })
            })
            .flatten()
            .collect();
        Array2::from_shape_vec((n_orbs_a, n_orbs_b), h).unwrap()
    }
}
//
// /// Computes the H0 and S outer diagonal block for two sets of atoms
// pub fn h0_and_s_ab_old(
//     &self,
//     n_orbs_a: usize,
//     n_orbs_b: usize,
//     atoms_a: AtomSlice,
//     atoms_b: AtomSlice,
// ) -> (Array2<f64>, Array2<f64>) {
//     let mut h0: Array2<f64> = Array2::zeros((n_orbs_a, n_orbs_b));
//     let mut s: Array2<f64> = Array2::zeros((n_orbs_a, n_orbs_b));
//     // Iteration over atoms in set A.
//     let mut mu: usize = 0;
//     for (i, (xyz_i, kind_i, valorbs_i, number_i)) in soa_zip!(atoms_a, [xyz, kind, valorbs, number]).enumerate() {
//         // Iteration over orbitals on atom I.
//         for orbi in valorbs_i.iter() {
//             // Iteration over atoms in set B.
//             let mut nu: usize = 0;
//             for (j, (xyz_j, kind_j, valorbs_j, number_j)) in soa_zip!(atoms_b, [xyz, kind, valorbs, number]).enumerate() {
//                 // Iteration over orbitals on atom J.
//                 for orbj in valorbs_j.iter() {
//                     //if geometry.proximities.as_ref().unwrap()[[i, j]] {
//                     if (xyz_i - xyz_j).norm() < PROXIMITY_CUTOFF {
//                         if number_i <= number_j {
//                             let (r, x, y, z): (f64, f64, f64, f64) =
//                                 directional_cosines(xyz_i, xyz_j);
//                             s[[mu, nu]] = slako_transformation(
//                                 r,
//                                 x,
//                                 y,
//                                 z,
//                                 &self.get(kind_i, kind_j).s_spline,
//                                 orbi.l,
//                                 orbi.m,
//                                 orbj.l,
//                                 orbj.m,
//                             );
//                             h0[[mu, nu]] = slako_transformation(
//                                 r,
//                                 x,
//                                 y,
//                                 z,
//                                 &self.get(kind_i, kind_j).h_spline,
//                                 orbi.l,
//                                 orbi.m,
//                                 orbj.l,
//                                 orbj.m,
//                             );
//                         } else {
//                             let (r, x, y, z): (f64, f64, f64, f64) =
//                                 directional_cosines(xyz_j, xyz_i);
//                             s[[mu, nu]] = slako_transformation(
//                                 r,
//                                 x,
//                                 y,
//                                 z,
//                                 &self.get(kind_j, kind_i).s_spline,
//                                 orbj.l,
//                                 orbj.m,
//                                 orbi.l,
//                                 orbi.m,
//                             );
//                             h0[[mu, nu]] = slako_transformation(
//                                 r,
//                                 x,
//                                 y,
//                                 z,
//                                 &self.get(kind_j, kind_i).h_spline,
//                                 orbj.l,
//                                 orbj.m,
//                                 orbi.l,
//                                 orbi.m,
//                             );
//                         }
//                     }
//                     nu = nu + 1;
//                 }
//             }
//             mu = mu + 1;
//         }
//     }
//     return (s, h0);
// }
//
// /// Computes the H0 and S matrix elements for a single molecule.
// pub fn h0_and_s(&self, n_orbs: usize, atoms: AtomSlice) -> (Array2<f64>, Array2<f64>) {
//     let mut h0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
//     let mut s: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
//     // iterate over atoms
//     let mut mu: usize = 0;
//     for (i, xyz_i) in atoms.iter().enumerate() {
//         // iterate over orbitals on center i
//         for orbi in atomi.valorbs.iter() {
//             // iterate over atoms
//             let mut nu: usize = 0;
//             for (j, atomj) in atoms.iter().enumerate() {
//                 // iterate over orbitals on center j
//                 for orbj in atomj.valorbs.iter() {
//                     //if geometry.proximities.as_ref().unwrap()[[i, j]] {
//                     if (atomi-atomj).norm() < PROXIMITY_CUTOFF {
//                         if mu < nu {
//                             if atomi <= atomj {
//                                 if i != j {
//                                     let (r, x, y, z): (f64, f64, f64, f64) =
//                                         directional_cosines(&atomi.xyz, &atomj.xyz);
//                                     s[[mu, nu]] = slako_transformation(
//                                         r,
//                                         x,
//                                         y,
//                                         z,
//                                         &self.get(atomi.kind, atomj.kind).s_spline,
//                                         orbi.l,
//                                         orbi.m,
//                                         orbj.l,
//                                         orbj.m,
//                                     );
//                                     h0[[mu, nu]] = slako_transformation(
//                                         r,
//                                         x,
//                                         y,
//                                         z,
//                                         &self.get(atomi.kind, atomj.kind).h_spline,
//                                         orbi.l,
//                                         orbi.m,
//                                         orbj.l,
//                                         orbj.m,
//                                     );
//                                 }
//                             } else {
//                                 let (r, x, y, z): (f64, f64, f64, f64) =
//                                     directional_cosines(&atomj.xyz, &atomi.xyz);
//                                 s[[mu, nu]] = slako_transformation(
//                                     r,
//                                     x,
//                                     y,
//                                     z,
//                                     &self.get(atomj.kind, atomi.kind).s_spline,
//                                     orbj.l,
//                                     orbj.m,
//                                     orbi.l,
//                                     orbi.m,
//                                 );
//                                 h0[[mu, nu]] = slako_transformation(
//                                     r,
//                                     x,
//                                     y,
//                                     z,
//                                     &self.get(atomj.kind, atomi.kind).h_spline,
//                                     orbj.l,
//                                     orbj.m,
//                                     orbi.l,
//                                     orbi.m,
//                                 );
//                             }
//
//                         } else if mu == nu {
//                             assert_eq!(atomi.number, atomj.number);
//                             h0[[mu, nu]] = orbi.energy;
//                             s[[mu, nu]] = 1.0;
//                         } else {
//                             s[[mu, nu]] = s[[nu, mu]];
//                             h0[[mu, nu]] = h0[[nu, mu]];
//                         }
//                     }
//                     nu = nu + 1;
//                 }
//             }
//             mu = mu + 1;
//         }
//     }
//     return (s, h0);
// }
//
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
