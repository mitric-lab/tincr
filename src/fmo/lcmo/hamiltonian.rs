use crate::fmo::{SuperSystem, Monomer};
use ndarray::prelude::*;
use hashbrown::HashMap;
use crate::scc::h0_and_s::h0_and_s_ab;
use crate::initialization::Atom;
use itertools::chain;

impl SuperSystem {

    pub fn build_lcmo_hamiltonian(&self) -> Array2<f64> {
        // TODO: READ THIS FROM THE INPUT FILE
        // Number of active orbitals per monomer
        let n_occ_m: usize = 1;
        let n_virt_m: usize= 1;
        let n_active_m: usize = n_occ_m + n_virt_m;

        // Reference to the atoms.
        let atoms: &[Atom] = &self.atoms;

        // The dimension of the Hamiltonian.
        let dim: usize = self.n_mol * n_active_m;

        let mut h: Array2<f64> = Array2::zeros([dim, dim]);

        // The diagonal elements are set.
        for (i, mol) in self.monomers.iter().enumerate() {
            // Reference to the occupied orbitals.
            let occ_indices: &[usize] = mol.properties.occ_indices().unwrap();
            // Reference to the virtual orbitals.
            let virt_indices: &[usize] = mol.properties.virt_indices().unwrap();
            // Reference to the orbital energies.
            let orbe: ArrayView1<f64> = mol.properties.orbe().unwrap();
            // Iteration through the active occupied orbitals.
            for occ_i in 0..n_occ_m {
                // Index to the matrix element in the Hamiltonian that will be set.
                let idx: usize = i * n_active_m + occ_i;
                // Index of the active occupied orbital.
                let occ_idx: usize = occ_indices[occ_indices.len() - (occ_i + 1)];
                // The matrix element in the Hamiltonian is set.
                h[[idx, idx]] = orbe[occ_idx];
            }
            // Iteration through the active virtual orbitals.
            for virt_i in 0..n_virt_m {
                // Index to the matrix element in H.
                let idx: usize = i * n_active_m + n_occ_m + virt_i;
                // Index of the active virtual orbital.
                let virt_idx: usize = virt_indices[virt_i];
                // The matrix element in the Hamiltonian is set.
                h[[idx, idx]] = orbe[virt_idx];
            }
        }

        // The off-diagonal elements are set.
        for pair in self.pairs.iter() {
            // Reference to monomer I.
            let m_i: &Monomer = &self.monomers[pair.i];
            // Reference to monomer J.
            let m_j: &Monomer = &self.monomers[pair.j];
            // Compute the overlap matrix and H0 matrix elements between both fragments.
            let (s_ab, h0_ab): (Array2<f64>, Array2<f64>) =
                h0_and_s_ab(m_i.n_orbs, m_j.n_orbs, &atoms[0..m_i.n_atoms], &atoms[m_i.n_atoms..], &m_i.slako);
            // Reference to the MO coefficients of monomer I.
            let orbs_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            // Reference to the MO coefficients of monomer J.
            let orbs_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
            // Reference to the indices of occupied orbitals of both monomers.
            let occ_indices_i: &[usize] = m_i.properties.occ_indices().unwrap();
            let occ_indices_j: &[usize] = m_j.properties.occ_indices().unwrap();
            // Reference to the indices of the virtual orbitals of both monomers.
            let virt_indices_i: &[usize] = m_i.properties.virt_indices().unwrap();
            let virt_indices_j: &[usize] = m_j.properties.virt_indices().unwrap();

            // The list with indices of occupied and virtual orbitals of monomer I is created.
            let indices_i: Vec<usize> = (0..n_occ_m)
                .map(|i| occ_indices_i[occ_indices_i.len() - (i+1)])
                .chain((0..n_virt_m)
                .map(|i| virt_indices_i[i]))
                .collect::<Vec<usize>>();

            // The same list for monomer J is createsd.
            let indices_j: Vec<usize> = (0..n_occ_m)
                .map(|j| occ_indices_j[occ_indices_j.len() - (j+1)])
                .chain((0..n_virt_m)
                    .map(|j| virt_indices_j[j]))
                .collect::<Vec<usize>>();

            // Iterate trough the active orbitals of monomer I and J and set the coupling elements.
            for (i, occ_i) in indices_i.iter().enumerate() {
                for (j, occ_j) in indices_j.iter().enumerate() {
                    // Index to the row of the Hamiltonian.
                    let row: usize = pair.i * n_active_m + i;
                    // Index to the column of the Hamiltonian.
                    let column: usize = pair.j * n_active_m + j;
                    // Contract the H0 matrix with the MO coefficients of both monomers.
                    h[[row, column]] = orbs_i.slice(s![0.., *occ_i]).t().dot(&h0_ab.dot(&orbs_j.slice(s![0.., *occ_j])));
                    h[[row, column]] = h[[column, row]];
                }
            }
        }
        // The LCMO Hamiltonian is returned.
        h
    }


}