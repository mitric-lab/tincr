use crate::fmo::{Monomer, SuperSystem};
use crate::initialization::Atom;
use crate::scc::h0_and_s::h0_and_s;
use hashbrown::HashMap;
use itertools::chain;
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use std::ops::AddAssign;
use crate::fmo::helpers::get_pair_slice;

impl SuperSystem {
    pub fn build_lcmo_fock_matrix(&mut self) -> Array2<f64> {
        // TODO: READ THIS FROM THE INPUT FILE
        // Number of active orbitals per monomer
        let n_occ_m: usize = 1;
        let n_virt_m: usize = 1;
        let n_active_m: usize = n_occ_m + n_virt_m;

        // Reference to the atoms.
        let atoms: &[Atom] = &self.atoms;

        // The dimension of the Fock matrix.
        let dim: usize = self.monomers.iter().map(|mol| mol.n_orbs).sum();

        // Initialize the Fock matrix.
        let mut fock: Array2<f64> = Array2::zeros([dim, dim]);
        // Initialize the overlap matrix.
        let mut s_total: Array2<f64> = Array2::eye(dim);

        // The diagonal elements are set.
        for (i, mol) in self.monomers.iter().enumerate() {
            // The diagonal Fock matrix of the monomer.
            let f_i: Array2<f64> = Array2::from_diag(&mol.properties.orbe().unwrap());
            // Fill the diagonal block of the Fock matrix.
            fock.slice_mut(s![mol.slice.orb, mol.slice.orb])
                .assign(&f_i);
        }

        // The off-diagonal elements are set.
        for pair in self.pairs.iter_mut() {
            // Reference to monomer I.
            let m_i: &Monomer = &self.monomers[pair.i];
            // Reference to monomer J.
            let m_j: &Monomer = &self.monomers[pair.j];
            //
            let pair_atoms: Vec<Atom> = atoms[m_i.slice.atom_as_range()]
                .iter()
                .chain(atoms[m_j.slice.atom_as_range()].iter())
                .map(From::from)
                .collect::<Vec<Atom>>();
            // Compute the overlap matrix and H0 matrix elements between both fragments.
            let (s, h0): (Array2<f64>, Array2<f64>) =
                h0_and_s(pair.n_orbs, &pair_atoms, &m_i.slako);
            // Reference to the MO coefficients of monomer I.
            let orbs_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            // Reference to the MO coefficients of monomer J.
            let orbs_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
            // Reference to the MO coefficients of the pair IJ.
            let orbs_ij: ArrayView2<f64> = pair.properties.orbs().unwrap();

            // Overlap between orbitals of monomer I and dimer IJ.
            let s_pr: Array2<f64> = (orbs_i.t().dot(&s.slice(s![0..m_i.n_orbs, ..]))).dot(&orbs_ij);
            // Overlap between orbitals of monomer J and dimer IJ.
            let s_qr: Array2<f64> = (orbs_j.t().dot(&s.slice(s![m_i.n_orbs.., ..]))).dot(&orbs_ij);

            // Overlap between orbitals of monomer I and J.
            let s_pq: Array2<f64> =
                (orbs_i.t().dot(&s.slice(s![0..m_i.n_orbs, m_i.n_orbs..]))).dot(&orbs_j);

            // Fill the off-diagonal block of the total overlap matrix
            s_total
                .slice_mut(s![m_i.slice.orb, m_j.slice.orb])
                .assign(&s_pq);
            s_total
                .slice_mut(s![m_j.slice.orb, m_i.slice.orb])
                .assign(&s_pq.t());

            // Reference to the orbital energies of the pair IJ.
            let orbe_ij: ArrayView1<f64> = pair.properties.orbe().unwrap();

            // The four blocks of the Fock submatrix are individually computed
            // according to: Sum_r e_r * <phi_p^I | phi_r^IJ > < phi_r^IJ | phi_q^J >
            let f_aa: Array2<f64> = (&s_pr * &orbe_ij).dot(&s_pr.t());
            let f_bb: Array2<f64> = (&s_qr * &orbe_ij).dot(&s_qr.t());
            let f_ab: Array2<f64> = (&s_pr * &orbe_ij).dot(&s_qr.t());
            let f_ba: Array2<f64> = (&s_qr * &orbe_ij).dot(&s_pr.t());

            // Save overlap between the monomers and the dimer
            pair.properties.set_overlap_i_ij(s_pr);
            pair.properties.set_overlap_j_ij(s_qr);

            // The diagonal Fock matrix of monomer I.
            let f_i: Array2<f64> = Array2::from_diag(&m_i.properties.orbe().unwrap());
            // The diagonal Fock matrix of monomer I.
            let f_j: Array2<f64> = Array2::from_diag(&m_j.properties.orbe().unwrap());

            // The diagonal block for monomer I is set.
            fock.slice_mut(s![m_i.slice.orb, m_i.slice.orb])
                .add_assign(&(&f_aa - &f_i));
            // The diagonal block for monomer J is set.
            fock.slice_mut(s![m_j.slice.orb, m_j.slice.orb])
                .add_assign(&(&f_bb - &f_j));
            // The off-diagonal block for the interaction I-J is set.
            fock.slice_mut(s![m_i.slice.orb, m_j.slice.orb])
                .add_assign(&f_ab);
            // The off-diagonal block for the interaction J-i is set.
            fock.slice_mut(s![m_j.slice.orb, m_i.slice.orb])
                .add_assign(&f_ba);
        }
        for esd_pair in self.esd_pairs.iter_mut() {
            // Reference to monomer I.
            let m_i: &Monomer = &self.monomers[esd_pair.i];
            // Reference to monomer J.
            let m_j: &Monomer = &self.monomers[esd_pair.j];

            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                self.monomers[esd_pair.i].slice.atom_as_range(),
                self.monomers[esd_pair.j].slice.atom_as_range(),
            );
            // do a scc calculation of the ESD pair
            esd_pair.prepare_scc(&pair_atoms, m_i, m_j);
            esd_pair.run_scc_test(&pair_atoms, self.config.scf);

            // get overlap matrix
            let s:ArrayView2<f64> = esd_pair.properties.s().unwrap();
            // Reference to the MO coefficients of monomer I.
            let orbs_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            // Reference to the MO coefficients of monomer J.
            let orbs_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
            // Reference to the MO coefficients of the pair IJ.
            let orbs_ij: ArrayView2<f64> = esd_pair.properties.orbs().unwrap();

            // Overlap between orbitals of monomer I and dimer IJ.
            let s_pr: Array2<f64> = (orbs_i.t().dot(&s.slice(s![0..m_i.n_orbs, ..]))).dot(&orbs_ij);
            // Overlap between orbitals of monomer J and dimer IJ.
            let s_qr: Array2<f64> = (orbs_j.t().dot(&s.slice(s![m_i.n_orbs.., ..]))).dot(&orbs_ij);

            // Save overlap between the monomers and the dimer
            esd_pair.properties.set_overlap_i_ij(s_pr);
            esd_pair.properties.set_overlap_j_ij(s_qr);
        }
        // The Fock matrix needs to be transformed by the Löwdin orthogonalization. Computation of
        // the matrix inverse of the total overlap is computationally demanding. Since the
        // total overlap matrix is almost diagonal, it can be approximated in first order by:
        // S^-1/2 = 1.5 * I - 1/2 Delta
        let x: Array2<f64> = 1.5 * Array::eye(dim) - 0.5 * &s_total;
        // let x2: Array2<f64> = s_total.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // The transformed Fock matrix is returned.
        (x.t().dot(&fock)).dot(&x)
        // fock
    }
}
