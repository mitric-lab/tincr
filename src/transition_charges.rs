use ndarray::prelude::*;
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use std::ops::AddAssign;

// computes the Mulliken transition charges between occupied-occupied
// occupied-virtual and virtual-virtual molecular orbitals.
// Point charge approximation of transition densities according to formula (14)
// in Heringer, Niehaus  J Comput Chem 28: 2589-2601 (2007)
pub fn trans_charges(
    atomic_numbers: &[u8],
    valorbs: HashMap<u8, Vec<(i8, i8, i8)>>,
    orbs: Array2<f64>,
    s: Array2<f64>,
    n_atoms: usize,
    active_occupied_orbs: Array1<usize>,
    active_virtual_orbs: Array1<usize>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let dim_o: usize = active_occupied_orbs.len();
    let dim_v: usize = active_virtual_orbs.len();
    // transition charges between occupied and virutal orbitals
    let mut q_trans_ov: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_v]);
    // transition charges between occupied and occupied orbitals
    let mut q_trans_oo: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_o]);
    // transition charges between virtual and virtual orbitals
    let mut q_trans_vv: Array3<f64> = Array3::zeros([n_atoms, dim_v, dim_v]);

    let s_c: Array2<f64> = s.dot(&orbs);

    let mut mu: usize = 0;
    for (atom_a, z_a) in atomic_numbers.iter().enumerate() {
        for _ in valorbs[z_a].iter() {
            // occupied - virtuals
            for (i, occi) in active_occupied_orbs.iter().enumerate() {
                for (a, virta) in active_virtual_orbs.iter().enumerate() {
                    q_trans_ov.slice_mut(s![atom_a, i, a]).add_assign(
                        0.5 * (orbs[[mu, *occi]] * s_c[[mu, *virta]]
                            + orbs[[mu, *virta]] * s_c[[mu, *occi]]),
                    );
                }
            }
            // occupied - occupied
            for (i, occi) in active_occupied_orbs.iter().enumerate() {
                for (j, occj) in active_occupied_orbs.iter().enumerate() {
                    q_trans_oo.slice_mut(s![atom_a, i, j]).add_assign(
                        0.5 * (orbs[[mu, *occi]] * s_c[[mu, *occj]]
                            + orbs[[mu, *occj]] * s_c[[mu, *occi]]),
                    );
                }
            }
            // virtual - virtual
            for (a, virta) in active_virtual_orbs.iter().enumerate() {
                for (b, virtb) in active_virtual_orbs.iter().enumerate() {
                    q_trans_vv.slice_mut(s![atom_a, a, b]).add_assign(
                        0.5 * (orbs[[mu, *virta]] * s_c[[mu, *virtb]]
                            + orbs[[mu, *virtb]] * s_c[[mu, *virta]]),
                    );
                }
            }
            mu += 1;
        }
    }
    return (q_trans_ov, q_trans_oo, q_trans_vv);
}
