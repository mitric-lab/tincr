use core::::Atom;
use ndarray::prelude::*;

/// Computes the Mulliken transition charges between occupied-occupied,
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
    let mut q_trans_ov: Array2<f64> = Array2::zeros([n_atoms, dim_o * dim_v]);
    // The transition charges between occupied and occupied orbitals are initialized.
    let mut q_trans_oo: Array2<f64> = Array2::zeros([n_atoms, dim_o * dim_o]);
    // The transition charges between virtual and virtual orbitals are initialized.
    let mut q_trans_vv: Array2<f64> = Array2::zeros([n_atoms, dim_v * dim_v]);
    // Matrix product of overlap matrix with the MO coefficients.
    let s_c: Array2<f64> = s.dot(&orbs);

    let mut mu: usize = 0;
    for (n, atom) in atoms.iter().enumerate() {
        // Iteration over the atomic orbitals Mu.
        for _ in 0..atom.n_orbs() {
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
                {
                    // The index to determine the pair of MOs is computed.
                    let idx: usize = (i * dim_v) + a;
                    // The occupied - virtual transition charge is computed.
                    q_trans_ov[[n, idx]] += 0.5 * (orb_mu_i * s_c_mu_a + orb_mu_a * s_c_mu_i);
                }
                // Iteration over occupied orbital J.
                for (j, (orb_mu_j, s_c_mu_j)) in orbs
                    .slice(s![mu, i_o..f_o])
                    .iter()
                    .zip(s_c.slice(s![mu, i_o..f_o]).iter())
                    .enumerate()
                {
                    // The index is computed.
                    let idx: usize = (i * dim_o) + j;
                    // The occupied - occupied transition charge is computed.
                    q_trans_oo[[n, idx]] += 0.5 * (orb_mu_i * s_c_mu_j + orb_mu_j * s_c_mu_i);
                }
            }
            // Iteration over virtual orbital A.
            for (a, (orb_mu_a, s_c_mu_a)) in orbs
                .slice(s![mu, i_v..f_v])
                .iter()
                .zip(s_c.slice(s![mu, i_v..f_v]).iter())
                .enumerate()
            {
                // Iteration over virtual orbital B.
                for (b, (orb_mu_b, s_c_mu_b)) in orbs
                    .slice(s![mu, i_v..f_v])
                    .iter()
                    .zip(s_c.slice(s![mu, i_v..f_v]).iter())
                    .enumerate()
                {
                    // The index is computed.
                    let idx: usize = (a * dim_v) + b;
                    // The virtual - virtual transition charge is computed.
                    q_trans_vv[[n, idx]] += 0.5 * (orb_mu_a * s_c_mu_b + orb_mu_b * s_c_mu_a);
                }
            }
            mu += 1;
        }
    }

    (q_trans_ov, q_trans_oo, q_trans_vv)
}
