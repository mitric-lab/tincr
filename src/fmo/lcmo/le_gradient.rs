use crate::excited_states::tda::*;
use crate::excited_states::ExcitedState;
use crate::fmo::{BasisState, ExcitedStateMonomerGradient, ExcitonStates, LocallyExcited, Monomer, SuperSystem, ChargeTransfer, Particle};
use crate::initialization::{Atom, MO};
use crate::io::settings::LcmoConfig;
use crate::utils::Timer;
use nalgebra::{max, Vector3};
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, UPLO};
use ndarray_npy::write_npy;
use std::fmt::{Display, Formatter};

impl SuperSystem {
    pub fn exciton_le_energy(&mut self, monomer_index: usize, state: usize) -> f64 {
        let lcmo_config: LcmoConfig = self.config.lcmo.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // get the monomer
        let mol = &mut self.monomers[monomer_index];
        // Compute the excited states for the monomer.
        mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
        mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);

        // switch to immutable borrow for the monomer
        let mol = &self.monomers[monomer_index];

        // Calculate transition charges
        let homo: usize = mol.properties.homo().unwrap();
        let q_ov: ArrayView2<f64> = mol.properties.q_ov().unwrap();

        // Create the LE state
        let tdm: ArrayView1<f64> = mol.properties.ci_coefficient(state).unwrap();
        let le_state: BasisState = BasisState::LE(LocallyExcited {
            monomer: mol,
            n: state,
            atoms: &atoms[mol.slice.atom_as_range()],
            q_trans: q_ov.dot(&tdm),
            occs: mol.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
            virts: mol.properties.orbs_slice(homo + 1, None).unwrap(),
            tdm: tdm,
            tr_dipole: mol.properties.tr_dipole(state).unwrap(),
        });

        let val: f64 = self.exciton_coupling(&le_state, &le_state);
        return val;
    }

    pub fn exciton_le_gradient(&mut self, monomer_index: usize, state: usize) -> Array1<f64> {
        let lcmo_config: LcmoConfig = self.config.lcmo.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];

        // parameters for the TDA routine
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // get the monomer
        let mol = &mut self.monomers[monomer_index];
        // Compute the excited states for the monomer.
        mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
        mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);

        // calculate the gradient
        mol.prepare_excited_gradient(&atoms[mol.slice.atom_as_range()]);
        let grad = mol.tda_gradient_lc(state);

        return grad;
    }

    pub fn exciton_le_le_coupling(&mut self,ind_a: usize, ind_b: usize, state_a: usize, state_b:usize)->f64{
        let lcmo_config: LcmoConfig = self.config.lcmo.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // use mutbale borrow for monomer a
        let mol_a = &mut self.monomers[ind_a];

        // Compute the excited states for the monomer.
        mol_a.prepare_tda(&atoms[mol_a.slice.atom_as_range()]);
        mol_a.run_tda(&atoms[mol_a.slice.atom_as_range()], n_le, max_iter, tolerance);

        // use mutbale borrow for monomer b
        let mol_b = &mut self.monomers[ind_b];
        mol_b.prepare_tda(&atoms[mol_b.slice.atom_as_range()]);
        mol_b.run_tda(&atoms[mol_b.slice.atom_as_range()], n_le, max_iter, tolerance);

        // switch to immutable borrow for the monomer a
        let mol_a = &self.monomers[ind_a];

        // Calculate transition charges
        let homo: usize = mol_a.properties.homo().unwrap();
        let q_ov: ArrayView2<f64> = mol_a.properties.q_ov().unwrap();

        // Create the LE state
        let tdm: ArrayView1<f64> = mol_a.properties.ci_coefficient(state_a).unwrap();
        let le_state_a: BasisState = BasisState::LE(LocallyExcited {
            monomer: mol_a,
            n: state_a,
            atoms: &atoms[mol_a.slice.atom_as_range()],
            q_trans: q_ov.dot(&tdm),
            occs: mol_a.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
            virts: mol_a.properties.orbs_slice(homo + 1, None).unwrap(),
            tdm: tdm,
            tr_dipole: mol_a.properties.tr_dipole(state_a).unwrap(),
        });

        // switch to immutable borrow for the monomer b
        let mol_b = &self.monomers[ind_b];

        // Calculate transition charges
        let homo: usize = mol_b.properties.homo().unwrap();
        let q_ov: ArrayView2<f64> = mol_b.properties.q_ov().unwrap();

        // Create the LE state
        let tdm: ArrayView1<f64> = mol_b.properties.ci_coefficient(state_b).unwrap();
        let le_state_b: BasisState = BasisState::LE(LocallyExcited {
            monomer: mol_b,
            n: state_b,
            atoms: &atoms[mol_b.slice.atom_as_range()],
            q_trans: q_ov.dot(&tdm),
            occs: mol_b.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
            virts: mol_b.properties.orbs_slice(homo + 1, None).unwrap(),
            tdm: tdm,
            tr_dipole: mol_b.properties.tr_dipole(state_b).unwrap(),
        });

        let val: f64 = self.exciton_coupling(&le_state_a, &le_state_b);
        return val;
    }

    pub fn exciton_le_ct_coupling(
        &mut self,
        ind_a: usize,
        state_a: usize,
        index_i: usize,
        index_j: usize,
        ct_ind_i: usize,
        ct_ind_j: usize,
        hole_i: bool,
    )->f64{
        let lcmo_config: LcmoConfig = self.config.lcmo.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;

        // calculate lcmo hamiltonian
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // use mutbale borrow for monomer a
        let mol_a = &mut self.monomers[ind_a];

        // Compute the excited states for the monomer.
        mol_a.prepare_tda(&atoms[mol_a.slice.atom_as_range()]);
        mol_a.run_tda(&atoms[mol_a.slice.atom_as_range()], n_le, max_iter, tolerance);

        // switch to immutable borrow for the monomer a
        let mol_a = &self.monomers[ind_a];

        // Calculate transition charges
        let homo: usize = mol_a.properties.homo().unwrap();
        let q_ov: ArrayView2<f64> = mol_a.properties.q_ov().unwrap();

        // Create the LE state
        let tdm: ArrayView1<f64> = mol_a.properties.ci_coefficient(state_a).unwrap();
        let le_state_a: BasisState = BasisState::LE(LocallyExcited {
            monomer: mol_a,
            n: state_a,
            atoms: &atoms[mol_a.slice.atom_as_range()],
            q_trans: q_ov.dot(&tdm),
            occs: mol_a.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
            virts: mol_a.properties.orbs_slice(homo + 1, None).unwrap(),
            tdm: tdm,
            tr_dipole: mol_a.properties.tr_dipole(state_a).unwrap(),
        });

        // initialize a CT state
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];

        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        // get occupied and virtual orbitals
        let mut occs: &[usize];
        let mut virts: &[usize];
        let mut hole: MO;
        let mut elec: MO;

        let ct_state: BasisState = if hole_i {
            // Indices of the occupied orbitals of Monomer J.
            occs = m_i.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_j.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_ind_i];
            let virt: usize = virts[ct_ind_j];

            // create hole and electron
            hole = MO::new(
                m_i.properties.mo_coeff(occ).unwrap(),
                m_i.properties.orbe().unwrap()[occ],
                occ,
                m_i.properties.occupation().unwrap()[occ],
            );
            elec = MO::new(
                m_j.properties.mo_coeff(virt).unwrap(),
                m_j.properties.orbe().unwrap()[virt],
                virt,
                m_j.properties.occupation().unwrap()[virt],
            );

            BasisState::CT(ChargeTransfer {
                // system: &self,
                hole: Particle {
                    idx: m_i.index,
                    atoms: &atoms[m_i.slice.atom_as_range()],
                    monomer: &m_i,
                    mo: hole,
                },
                electron: Particle {
                    idx: m_j.index,
                    atoms: &atoms[m_j.slice.atom_as_range()],
                    monomer: &m_j,
                    mo: elec,
                },
            })
        } else {
            // Indices of the occupied orbitals of Monomer J.
            occs = m_j.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_i.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_ind_j];
            let virt: usize = virts[ct_ind_i];

            // create hole and electron
            hole = MO::new(
                m_j.properties.mo_coeff(occ).unwrap(),
                m_j.properties.orbe().unwrap()[occ],
                occ,
                m_j.properties.occupation().unwrap()[occ],
            );
            elec = MO::new(
                m_i.properties.mo_coeff(virt).unwrap(),
                m_i.properties.orbe().unwrap()[virt],
                virt,
                m_i.properties.occupation().unwrap()[virt],
            );

            BasisState::CT(ChargeTransfer {
                // system: &self,
                hole: Particle {
                    idx: m_j.index,
                    atoms: &atoms[m_j.slice.atom_as_range()],
                    monomer: &m_j,
                    mo: hole,
                },
                electron: Particle {
                    idx: m_i.index,
                    atoms: &atoms[m_i.slice.atom_as_range()],
                    monomer: &m_i,
                    mo: elec,
                },
            })
        };

        let val: f64 = self.exciton_coupling(&le_state_a, &ct_state);
        return val;
    }
}
