use crate::fmo::helpers::get_pair_slice;
use crate::fmo::{
    BasisState, ChargeTransfer, ESDPair, GroundStateGradient, Monomer, Pair, PairType, Particle,
    SuperSystem,
};
use crate::initialization::{Atom, MO};
use crate::scc::gamma_approximation::gamma_atomwise_ab;
use crate::scc::h0_and_s::h0_and_s_ab;
use ndarray::prelude::*;
use std::ops::AddAssign;

impl SuperSystem {
    pub fn ct_gradient_new(
        &mut self,
        index_i: usize,
        index_j: usize,
        ct_ind_i: usize,
        ct_ind_j: usize,
        ct_energy: f64,
        hole_i: bool,
    ) -> Array1<f64> {
        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        // get pair type
        let pair_type: PairType = self.properties.type_of_pair(index_i, index_j);
        let mut ct_gradient: Array1<f64> = Array1::zeros([3 * (m_i.n_atoms + m_j.n_atoms)]);

        if pair_type == PairType::Pair {
            // get pair index
            let pair_index: usize = self.properties.index_of_pair(index_i, index_j);
            // get correct pair from pairs vector
            let pair_ij: &mut Pair = &mut self.pairs[pair_index];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            pair_ij.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
            pair_ij.prepare_ct_state(&pair_atoms, m_i, m_j, ct_ind_i, ct_ind_j, ct_energy, hole_i);
            ct_gradient = pair_ij.tda_gradient_lc(0);
            // reset gradient specific properties
            pair_ij.properties.reset_gradient();
        } else {
            // Do something for ESD pairs
            // get pair index
            let pair_index: usize = self.properties.index_of_esd_pair(index_i, index_j);
            // get correct pair from pairs vector
            let pair_ij: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            // do a scc calculation of the ESD pair
            pair_ij.prepare_scc(&pair_atoms, m_i, m_j);
            pair_ij.run_scc(&pair_atoms, self.config.scf);

            pair_ij.prepare_lcmo_gradient(&pair_atoms);
            pair_ij.prepare_ct_state(&pair_atoms, m_i, m_j, ct_ind_i, ct_ind_j, ct_energy, hole_i);
            ct_gradient = pair_ij.tda_gradient_nolc(0);
            pair_ij.properties.reset();
        }

        return ct_gradient;
    }

    pub fn exciton_ct_energy(
        &mut self,
        index_i: usize,
        index_j: usize,
        ct_ind_i: usize,
        ct_ind_j: usize,
        hole_i: bool,
    ) -> f64 {
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
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

        let state: BasisState = if hole_i {
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
        let val: f64 = self.exciton_coupling(&state, &state);

        return val;
    }

    pub fn exciton_hamiltonian_ct_test(&mut self) -> f64 {
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 50;
        let tolerance: f64 = 1e-4;
        // Number of LE states per monomer.
        let n_le: usize = self.config.lcmo.n_le;
        // Compute the n_le excited states for each monomer.
        for mol in self.monomers.iter_mut() {
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
            mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);
        }

        // Construct the diabatic basis states.
        let states: Vec<BasisState> = self.create_diab_basis();

        let ct_state = &states[2 * n_le];
        let val: f64 = self.exciton_coupling(ct_state, ct_state);

        return val;
    }

    pub fn exciton_ct_ct_coupling(
        &mut self,
        ind_i: usize,
        ind_j: usize,
        ct_i: usize,
        ct_j: usize,
        hole_i: bool,
        ind_a: usize,
        ind_b: usize,
        ct_a: usize,
        ct_b: usize,
        hole_a: bool,
    ) -> f64 {
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];

        // get monomers for the first CT state
        let m_i: &Monomer = &self.monomers[ind_i];
        let m_j: &Monomer = &self.monomers[ind_j];

        // get occupied and virtual orbitals for the first CT state
        let mut occs: &[usize];
        let mut virts: &[usize];
        let mut hole: MO;
        let mut elec: MO;

        let state_1: BasisState = if hole_i {
            // Indices of the occupied orbitals of Monomer J.
            occs = m_i.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_j.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_i];
            let virt: usize = virts[ct_j];

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
            let occ: usize = occs[nocc - 1 - ct_j];
            let virt: usize = virts[ct_i];

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

        // get monomers for the second CT state
        let m_a: &Monomer = &self.monomers[ind_a];
        let m_b: &Monomer = &self.monomers[ind_b];

        // get occupied and virtual orbitals for the second CT state
        let mut occs: &[usize];
        let mut virts: &[usize];
        let mut hole: MO;
        let mut elec: MO;

        let state_2: BasisState = if hole_a {
            // Indices of the occupied orbitals of Monomer J.
            occs = m_a.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_b.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_a];
            let virt: usize = virts[ct_b];

            // create hole and electron
            hole = MO::new(
                m_a.properties.mo_coeff(occ).unwrap(),
                m_a.properties.orbe().unwrap()[occ],
                occ,
                m_a.properties.occupation().unwrap()[occ],
            );
            elec = MO::new(
                m_b.properties.mo_coeff(virt).unwrap(),
                m_b.properties.orbe().unwrap()[virt],
                virt,
                m_b.properties.occupation().unwrap()[virt],
            );

            BasisState::CT(ChargeTransfer {
                // system: &self,
                hole: Particle {
                    idx: m_a.index,
                    atoms: &atoms[m_a.slice.atom_as_range()],
                    monomer: &m_a,
                    mo: hole,
                },
                electron: Particle {
                    idx: m_b.index,
                    atoms: &atoms[m_b.slice.atom_as_range()],
                    monomer: &m_b,
                    mo: elec,
                },
            })
        } else {
            // Indices of the occupied orbitals of Monomer J.
            occs = m_b.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_a.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_b];
            let virt: usize = virts[ct_a];

            // create hole and electron
            hole = MO::new(
                m_b.properties.mo_coeff(occ).unwrap(),
                m_b.properties.orbe().unwrap()[occ],
                occ,
                m_b.properties.occupation().unwrap()[occ],
            );
            elec = MO::new(
                m_a.properties.mo_coeff(virt).unwrap(),
                m_a.properties.orbe().unwrap()[virt],
                virt,
                m_a.properties.occupation().unwrap()[virt],
            );

            BasisState::CT(ChargeTransfer {
                // system: &self,
                hole: Particle {
                    idx: m_b.index,
                    atoms: &atoms[m_b.slice.atom_as_range()],
                    monomer: &m_b,
                    mo: hole,
                },
                electron: Particle {
                    idx: m_a.index,
                    atoms: &atoms[m_a.slice.atom_as_range()],
                    monomer: &m_a,
                    mo: elec,
                },
            })
        };

        let val: f64 = self.exciton_coupling(&state_1, &state_2);

        return val;
    }
}
