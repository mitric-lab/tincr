use ndarray::{ArrayView2, ArrayView3, ArrayView1, Array1, Array3, Array2};
use ndarray_linalg::{Eigh, UPLO};
use crate::excited_states::solvers::davidson::DavidsonEngine;
use crate::initialization::System;

// impl DavidsonEngine for System {
//
//
//     // """Build the combinations
//     //     Singlet:
//     //        A X = [(Ea - Ei) + 2 J - K] X
//     //     Triplet:
//     //        A X = [(Ea - Ei) - K] X
//     //     """
//     // Ax = []
//     // if Kx is not None:
//     // for Fxi, Jxi, Kxi in zip(Fx, Jx, Kx):
//     //   Ax_so = self.vector_scale(-1.0, self.vector_copy(Kxi))
//     //   if self.singlet:
//     //     Ax_so = self.vector_axpy(2.0, Jxi, Ax_so)
//     //   Ax.append(self.vector_axpy(1.0, Fxi, self._so_to_mo(Ax_so)))
//     //
//     // else:
//     // for Fxi, Jxi in zip(Fx, Jx):
//     // if self.singlet:
//     // Ax.append(self.vector_axpy(1.0, Fxi, self._so_to_mo(self.vector_scale(2.0, Jxi))))
//     // else:
//     // Ax.append(self.vector_copy(Fxi))
//     // return Ax
//
//     fn compute_products(&self, x: _) -> _ {
//
//     }
//
//     /// The preconditioner and a shift are applied to the residual vectors.
//     /// The energy difference of the virtual and occupied orbitals is used as a preconditioner.
//     fn precondition(&self, r_k: ArrayView1<f64>, w_k: f64) -> Array1<f64> {
//         // The denominator is build from the orbital energy differences and the shift value.
//         let mut denom: Array1<f64> = w_k - &self.properties.orbe_diffs().unwrap();
//         // Values smaller than 0.0001 are replaced by 1.0.
//         denom.mapv_inplace(|&x| if x < 0.0001 {1.0} else {x});
//         r_k / denom
//     }
//
//     fn get_size(&self) -> usize {
//         self.occ_indices.len() * self.virt_indices.len()
//     }
// }