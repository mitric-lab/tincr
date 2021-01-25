use ndarray::{Array1, Array2, Array3};

// computes the Mulliken transition charges between occupied-occupied
// occupied-virtual and virtual-virtual molecular orbitals.
// Point charge approximation of transition densities according to formula (14)
// in Heringer, Niehaus  J Comput Chem 28: 2589-2601 (2007)
pub fn trans_charges(
    orbs: Array2<f64>,
    s: Array2<f64>,
    n_atoms: usize,
    active_occupied_orbs: Array1<f64>,
    active_virtual_orbs: Array1<f64>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    // transition charges between occupied and virutal orbitals
    let mut qtrans_ov: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_v]);
    // transition charges between occupied and occupied orbitals
    let mut qtrans_oo: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_o]);
    // transition charges between virtual and virtual orbitals
    let mut qtrans_vv: Array3<f64> = Array3::zeros([n_atoms, dim_v, dim_v]);

    let s_c: Array2<f64> = s.dot(&orbs);

    let mut mu: usize = 0;
    for (atom_a, z_a) in atomic_numbers.iter().enumerate() {
        for _ in valorbs[z_a] {
            // occupied - virtuals
            for (i, occi) in active_occupied_orbs.iter().enumerate() {
                for (a, virta) in active_virtual_orbs.iter().enumerate() {
                    *q_trans_ov.slice_mut(s![atom_a, i, a]) += 0.5
                        * (orbs[[mu, occi]] * s_c[[mu, virta]]
                            + orbs[[mu, virta]] * s_c[[mu, occi]]);
                }
            }
            // occupied - occupied
            for (i, occi) in active_occupied_orbs.iter().enumerate() {
                for (j, occj) in active_occupied_orbs.iter().enumerate() {
                    *q_trans_oo.slice_mut(s![atom_a, i, j]) += 0.5
                        * (orbs[[mu, occi]] * s_c[[mu, occj]] + orbs[[mu, occj]] * s_c[[mu, occi]]);
                }
            }
            // virtual - virtual
            for (a, virta) in active_virtual_orbs.iter().enumerate() {
                for (b, virtb) in active_virtual_orbs.iter().enumerate() {
                    *q_trans_vv.slice_mut(s![atom_a, a, b]) += 0.5(orbs[[mu, virta]]
                        * s_c[[mu, virtb]]
                        + orbs[[mu, virtb]] * s_c[[mu, virta]]);
                }
            }
            mu += 1;
        }
    }
    return (qtrans_ov, qtrans_oo, qtrans_vv);
}
