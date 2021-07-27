use ndarray::prelude::*;
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use std::ops::AddAssign;
use crate::initialization::Atom;

/// Computes the Mulliken transition charges between occupied-occupied
/// occupied-virtual and virtual-virtual molecular orbitals.
/// Point charge approximation of transition densities according to formula (14)
/// in Heringer, Niehaus  J Comput Chem 28: 2589-2601 (2007)
pub fn trans_charges(
    n_atoms: usize,
    atoms: &[Atom],
    orbs: ArrayView2<f64>,
    s: ArrayView2<f64>,
    occ_indices: &[usize],
    virt_indices: &[usize],
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let dim_o: usize = occ_indices.len();
    let dim_v: usize = virt_indices.len();
    // transition charges between occupied and virutal orbitals
    let mut q_trans_ov: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_v]);
    // transition charges between occupied and occupied orbitals
    let mut q_trans_oo: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_o]);
    // transition charges between virtual and virtual orbitals
    let mut q_trans_vv: Array3<f64> = Array3::zeros([n_atoms, dim_v, dim_v]);

    let s_c: Array2<f64> = s.dot(&orbs);

    let mut mu: usize = 0;
    for (n, atom) in atoms.iter().enumerate() {
        for _ in 0..(atom.n_orbs) {
            // occupied - virtuals
            for (i, occi) in occ_indices.iter().enumerate() {
                for (a, virta) in virt_indices.iter().enumerate() {
                    q_trans_ov.slice_mut(s![n, i, a]).add_assign(
                        0.5 * (orbs[[mu, *occi]] * s_c[[mu, *virta]]
                            + orbs[[mu, *virta]] * s_c[[mu, *occi]]),
                    );
                }
            }
            // occupied - occupied
            for (i, occi) in occ_indices.iter().enumerate() {
                for (j, occj) in occ_indices.iter().enumerate() {
                    q_trans_oo.slice_mut(s![n, i, j]).add_assign(
                        0.5 * (orbs[[mu, *occi]] * s_c[[mu, *occj]]
                            + orbs[[mu, *occj]] * s_c[[mu, *occi]]),
                    );
                }
            }
            // virtual - virtual
            for (a, virta) in virt_indices.iter().enumerate() {
                for (b, virtb) in virt_indices.iter().enumerate() {
                    q_trans_vv.slice_mut(s![n, a, b]).add_assign(
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

/// Computes the Mulliken transition charges between occupied-occupied,
/// occupied-virtual and virtual-virtual molecular orbitals.
/// Point charge approximation of transition densities according to formula (14)
/// in Heringer, Niehaus  J Comput Chem 28: 2589-2601 (2007)
pub fn trans_charges_fast(
    n_atoms: usize,
    atoms: &[Atom],
    orbs: ArrayView2<f64>,
    s: ArrayView2<f64>,
    occ_indices: &[usize],
    virt_indices: &[usize],
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    // Number of occupied orbitals.
    let dim_o: usize = occ_indices.len();
    // Number of virtual orbitals.
    let dim_v: usize = virt_indices.len();
    // Initial index of occupied orbitals.
    let i_o: usize = occ_indices[0];
    // Final index of occupied orbitals.
    let f_o: usize = occ_indices[occ_indices.len() - 1] + 1;
    // Initial index of virtual orbitals.
    let i_v: usize = virt_indices[0];
    // Final index of virtual orbitals.
    let f_v: usize = virt_indices[virt_indices.len() - 1] + 1;
    // The transition charges between occupied and virtual orbitals are initialized.
    let mut q_trans_ov: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_v]);
    // The transition charges between occupied and occupied orbitals are initialized.
    let mut q_trans_oo: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_o]);
    // The transition charges between virtual and virtual orbitals are initialized.
    let mut q_trans_vv: Array3<f64> = Array3::zeros([n_atoms, dim_v, dim_v]);
    // Matrix product of overlap matrix with the MO coefficients.
    let s_c: Array2<f64> = s.dot(&orbs);
    let mut mu: usize = 0;
    for (n, atom) in atoms.iter().enumerate() {
        // Iteration over the atomic orbitals Mu.
        for _ in 0..atom.n_orbs {
            // Iteration over occupied orbital I.
            for (i, (orb_mu_i, s_c_mu_i)) in orbs
                .slice(s![mu, i_o..f_o])
                .iter()
                .zip(s_c.slice(s![mu, i_o..f_o]).iter())
                .enumerate()
            {
                // Iteration over virtual orbital A.
                for (a, (orb_mu_a, s_c_mu_a)) in orbs
                    .slice(s![mu, i_v..f_v])
                    .iter()
                    .zip(s_c.slice(s![mu, i_v..f_v]).iter())
                    .enumerate()
                {   // The index to determine the pair of MOs is computed.
                    // let idx: usize = (i * dim_v) + a;
                    // The occupied - virtual transition charge is computed.
                    q_trans_ov[[n, i,a]] += 0.5 * (orb_mu_i * s_c_mu_a + orb_mu_a * s_c_mu_i);
                }
                // Iteration over occupied orbital J.
                for (j, (orb_mu_j, s_c_mu_j)) in orbs
                    .slice(s![mu, i_o..f_o])
                    .iter()
                    .zip(s_c.slice(s![mu, i_o..f_o]).iter())
                    .enumerate()
                {
                    // The index is computed.
                    // let idx: usize = (i * dim_o) + j;
                    // The occupied - occupied transition charge is computed.
                    q_trans_oo[[n,i,j]] += 0.5 * (orb_mu_i * s_c_mu_j + orb_mu_j * s_c_mu_i);
                }
            }
            // Iteration over virtual orbital A.
            for (a, (orb_mu_a, s_c_mu_a)) in orbs
                .slice(s![mu, i_v..f_v])
                .iter()
                .zip(s_c.slice(s![mu, i_v..f_v]).iter())
                .enumerate()
            {   // Iteration over virtual orbital B.
                for (b, (orb_mu_b, s_c_mu_b)) in orbs
                    .slice(s![mu, i_v..f_v])
                    .iter()
                    .zip(s_c.slice(s![mu, i_v..f_v]).iter())
                    .enumerate()
                {   // The index is computed.
                    // let idx: usize = (a * dim_v) + b;
                    // The virtual - virtual transition charge is computed.
                    q_trans_vv[[n, a,b]] += 0.5 * (orb_mu_a * s_c_mu_b + orb_mu_b * s_c_mu_a);
                }
            }
            mu += 1;
        }
    }
    (q_trans_ov.into_shape([n_atoms,dim_o*dim_v]).unwrap(),
     q_trans_oo.into_shape([n_atoms,dim_o*dim_o]).unwrap(),
     q_trans_vv.into_shape([n_atoms,dim_v*dim_v]).unwrap())
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::initialization::Properties;
//     use crate::initialization::System;
//     use crate::utils::*;
//     use approx::AbsDiffEq;
//
//     pub const EPSILON: f64 = 1e-15;
//     // n_atoms: usize,
//     // atoms: &[Atom],
//     // orbs: ArrayView2<f64>,
//     // s: ArrayView2<f64>,
//     // occ_indeces: &[usize],
//     // virt_indeces: &[usize],
//
//     fn test_transition_charges(molecule_and_properties: (&str, System, Properties)) {
//         let name = molecule_and_properties.0;
//         let molecule = molecule_and_properties.1;
//         let props = molecule_and_properties.2;
//
//         let s_ref: Array2<f64> = props.get("S").unwrap().as_array2().unwrap().to_owned();
//         let orbs_ref: Array2<f64> = props.get("orbs_after_scc").unwrap().as_array2().unwrap().to_owned();
//         //let (q_ov, q_oo, q_vv) = trans_charges(molecule.n_atoms, &molecule.atoms, orbs_ref.view, s.view, );
//         let q_ov_ref: Array2<f64> = props.get("q_ov").unwrap().as_array3().unwrap().to_owned();
//         let q_oo_ref: Array2<f64> = props.get("q_oo").unwrap().as_array3().unwrap().to_owned();
//         let q_vv_ref: Array2<f64> = props.get("q_vv").unwrap().as_array3().unwrap().to_owned();
//         assert!(q_ov_ref.abs_diff_eq(&q_ov, EPSILON), "Molecule: {} qov (ref): {} qov {}", name, q_ov_ref, q_ov);
//         assert!(q_oo_ref.abs_diff_eq(&q_oo, EPSILON), "Molecule: {} qoo (ref): {} qoo {}", name, q_oo_ref, q_oo);
//         assert!(q_vv_ref.abs_diff_eq(&q_vv, EPSILON), "Molecule: {} qvv (ref): {} qvv {}", name, q_vv_ref, q_vv);
//         }
//
//
//     fn transition_charges() {
//         let names = AVAILAIBLE_MOLECULES;
//         for molecule in names.iter() {
//             test_transition_charges(get_molecule(molecule, "no_lc_gs"));
//         }
//     }
// }