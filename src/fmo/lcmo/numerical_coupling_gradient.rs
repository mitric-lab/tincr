use crate::fmo::lcmo::cis_gradient::{
    ReducedBasisState, ReducedCT, ReducedLE, ReducedMO, ReducedParticle,
};
use crate::fmo::{
    BasisState, ChargeTransfer, ExcitedStateMonomerGradient, LocallyExcited, Monomer, Particle,
    SuperSystem,
};
use crate::gradients::assert_deriv;
use crate::initialization::{Atom, MO};
use crate::io::settings::LcmoConfig;
use crate::scc::scc_routine::RestrictedSCC;
use ndarray::prelude::*;

impl SuperSystem {
    pub fn fmo_le_le_coupling_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        self.run_scc();

        let val: f64 = self.exciton_le_le_coupling(0, 1, 6, 6);
        return val;
    }

    pub fn fmo_le_le_coupling_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        let grad: Array1<f64> = self.exciton_le_le_coupling_gradient(0, 1, 6, 6);
        let mol_a: &Monomer = &self.monomers[0];
        let mol_b: &Monomer = &self.monomers[1];

        let mut full_gradient: Array1<f64> = Array1::zeros(self.atoms.len() * 3);
        full_gradient
            .slice_mut(s![mol_a.slice.grad])
            .assign(&grad.slice(s![..mol_a.n_atoms * 3]));
        full_gradient
            .slice_mut(s![mol_b.slice.grad])
            .assign(&grad.slice(s![mol_a.n_atoms * 3..]));
        return full_gradient;
    }

    pub fn test_le_le_coupling_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        assert_deriv(
            self,
            SuperSystem::fmo_le_le_coupling_energy_wrapper,
            SuperSystem::fmo_le_le_coupling_gradient_wrapper,
            self.get_xyz(),
            0.1,
            1e-6,
        );
    }

    pub fn fmo_le_ct_coupling_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        self.run_scc();

        let val: f64 = self.exciton_le_ct_coupling(0, 0, 0, 1, 0, 0, true);
        return val;
    }

    pub fn fmo_le_ct_coupling_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        let grad: Array1<f64> = self.exciton_le_ct_coupling_gradient(0, 0, 0, 1, 0, 0, true);
        let mol_a: &Monomer = &self.monomers[0];
        let mol_b: &Monomer = &self.monomers[1];

        let mut full_gradient: Array1<f64> = Array1::zeros(self.atoms.len() * 3);
        full_gradient
            .slice_mut(s![mol_a.slice.grad])
            .assign(&grad.slice(s![..mol_a.n_atoms * 3]));
        full_gradient
            .slice_mut(s![mol_b.slice.grad])
            .assign(&grad.slice(s![mol_a.n_atoms * 3..]));
        return full_gradient;
    }

    pub fn test_le_ct_coupling_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        assert_deriv(
            self,
            SuperSystem::fmo_le_ct_coupling_energy_wrapper,
            SuperSystem::fmo_le_ct_coupling_gradient_wrapper,
            self.get_xyz(),
            0.1,
            1e-6,
        );
    }

    pub fn fmo_ct_ct_coupling_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        self.run_scc();

        let val: f64 = self.exciton_ct_ct_coupling(0, 1, 0, 0, false, 0, 1, 1, 1, false);
        return val;
    }

    pub fn fmo_ct_ct_coupling_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        let grad: Array1<f64> =
            self.exciton_ct_ct_coupling_gradient(0, 1, 0, 0, false, 0, 1, 1, 1, false);
        let mol_a: &Monomer = &self.monomers[0];
        let mol_b: &Monomer = &self.monomers[1];

        let mut full_gradient: Array1<f64> = Array1::zeros(self.atoms.len() * 3);
        full_gradient
            .slice_mut(s![mol_a.slice.grad])
            .assign(&grad.slice(s![..mol_a.n_atoms * 3]));
        full_gradient
            .slice_mut(s![mol_b.slice.grad])
            .assign(&grad.slice(s![mol_a.n_atoms * 3..]));
        return full_gradient;
    }

    pub fn test_ct_ct_coupling_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut(){
            esd_pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        let val: f64 = self.exciton_ct_ct_coupling(0, 1, 0, 0, false, 0, 1, 0, 0, true);
        println!("Coupling: {}",val);

        assert_deriv(
            self,
            SuperSystem::fmo_ct_ct_coupling_energy_wrapper,
            SuperSystem::fmo_ct_ct_coupling_gradient_wrapper,
            self.get_xyz(),
            0.001,
            1e-6,
        );
    }

    pub fn exciton_le_le_coupling(
        &mut self,
        ind_a: usize,
        ind_b: usize,
        state_a: usize,
        state_b: usize,
    ) -> f64 {
        // Number of LE states per monomer.
        let n_states: usize = self.config.excited.nstates;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // use mutable borrow for monomer a
        let mol_a = &mut self.monomers[ind_a];

        // Compute the excited states for the monomer.
        mol_a.prepare_tda(&atoms[mol_a.slice.atom_as_range()]);
        mol_a.run_tda(
            &atoms[mol_a.slice.atom_as_range()],
            n_states,
            max_iter,
            tolerance,
        );

        // use mutable borrow for monomer b
        let mol_b = &mut self.monomers[ind_b];
        mol_b.prepare_tda(&atoms[mol_b.slice.atom_as_range()]);
        mol_b.run_tda(
            &atoms[mol_b.slice.atom_as_range()],
            n_states,
            max_iter,
            tolerance,
        );

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

    pub fn exciton_le_le_coupling_gradient(
        &mut self,
        ind_a: usize,
        ind_b: usize,
        state_a: usize,
        state_b: usize,
    ) -> Array1<f64> {
        // Number of states per monomer.
        let n_states:usize = self.config.excited.nstates;

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms;
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // use mutable borrow for monomer a
        let mol_a = &mut self.monomers[ind_a];

        // Compute the excited states for the monomer.
        mol_a.prepare_tda(&atoms[mol_a.slice.atom_as_range()]);
        mol_a.run_tda(
            &atoms[mol_a.slice.atom_as_range()],
            n_states,
            max_iter,
            tolerance,
        );

        // use mutable borrow for monomer b
        let mol_b = &mut self.monomers[ind_b];
        mol_b.prepare_tda(&atoms[mol_b.slice.atom_as_range()]);
        mol_b.run_tda(
            &atoms[mol_b.slice.atom_as_range()],
            n_states,
            max_iter,
            tolerance,
        );

        // switch to immutable borrow for the monomer a
        let mol_a = &self.monomers[ind_a];

        // Calculate transition charges
        let homo: usize = mol_a.properties.homo().unwrap();

        // Create the LE state
        let le_state_a = ReducedBasisState::LE(ReducedLE {
            energy: 0.0,
            monomer_index: ind_a,
            state_index: state_a,
            state_coefficient: 0.0,
            homo: homo,
        });

        // switch to immutable borrow for the monomer b
        let mol_b = &self.monomers[ind_b];

        // Calculate transition charges
        let homo: usize = mol_b.properties.homo().unwrap();

        // Create the LE state
        let le_state_b = ReducedBasisState::LE(ReducedLE {
            energy: 0.0,
            monomer_index: ind_b,
            state_index: state_b,
            state_coefficient: 0.0,
            homo: homo,
        });

        let grad: Array1<f64> = self.exciton_coupling_gradient_new(&le_state_a, &le_state_b);
        return grad;
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
    ) -> f64 {
        // Number of states per monomer.
        let n_states: usize = self.config.excited.nstates;

        // calculate lcmo hamiltonian
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // use mutable borrow for monomer a
        let mol_a = &mut self.monomers[ind_a];

        // Compute the excited states for the monomer.
        mol_a.prepare_tda(&atoms[mol_a.slice.atom_as_range()]);
        mol_a.run_tda(
            &atoms[mol_a.slice.atom_as_range()],
            n_states,
            max_iter,
            tolerance,
        );

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

        let ct_state = if hole_i {
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

    pub fn exciton_le_ct_coupling_gradient(
        &mut self,
        ind_a: usize,
        state_a: usize,
        index_i: usize,
        index_j: usize,
        ct_ind_i: usize,
        ct_ind_j: usize,
        hole_i: bool,
    ) -> Array1<f64> {
        // Number of states per monomer.
        let n_states: usize = self.config.excited.nstates;

        // calculate lcmo hamiltonian
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-4;

        // use mutable borrow for monomer a
        let mol_a = &mut self.monomers[ind_a];

        // Compute the excited states for the monomer.
        mol_a.prepare_tda(&atoms[mol_a.slice.atom_as_range()]);
        mol_a.run_tda(
            &atoms[mol_a.slice.atom_as_range()],
            n_states,
            max_iter,
            tolerance,
        );

        // switch to immutable borrow for the monomer a
        let mol_a = &self.monomers[ind_a];

        // Calculate transition charges
        let homo: usize = mol_a.properties.homo().unwrap();

        // Create the LE state
        let le_state_a = ReducedBasisState::LE(ReducedLE {
            energy: 0.0,
            monomer_index: ind_a,
            state_index: state_a,
            state_coefficient: 0.0,
            homo: homo,
        });

        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        // initialize a CT state
        let ct_state = if hole_i {
            // Indices of the occupied orbitals of Monomer J.
            let occs = m_i.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            let virts = m_j.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_ind_i];
            let virt: usize = virts[ct_ind_j];

            // create hole and electron
            ReducedBasisState::CT(ReducedCT {
                energy: 0.0,
                hole: ReducedParticle {
                    m_index: m_i.index,
                    ct_index: ct_ind_i,
                    mo: ReducedMO {
                        c: m_i.properties.mo_coeff(occ).unwrap().to_owned(),
                        index: occ,
                    },
                },
                electron: ReducedParticle {
                    m_index: m_j.index,
                    ct_index: ct_ind_j,
                    mo: ReducedMO {
                        c: m_j.properties.mo_coeff(virt).unwrap().to_owned(),
                        index: virt,
                    },
                },
                state_coefficient: 0.0,
            })
        } else {
            // Indices of the occupied orbitals of Monomer J.
            let occs = m_j.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            let virts = m_i.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_ind_j];
            let virt: usize = virts[ct_ind_i];

            ReducedBasisState::CT(ReducedCT {
                energy: 0.0,
                hole: ReducedParticle {
                    m_index: m_j.index,
                    ct_index: ct_ind_j,
                    mo: ReducedMO {
                        c: m_j.properties.mo_coeff(occ).unwrap().to_owned(),
                        index: occ,
                    },
                },
                electron: ReducedParticle {
                    m_index: m_i.index,
                    ct_index: ct_ind_i,
                    mo: ReducedMO {
                        c: m_i.properties.mo_coeff(virt).unwrap().to_owned(),
                        index: virt,
                    },
                },
                state_coefficient: 0.0,
            })
        };

        let grad: Array1<f64> = self.exciton_coupling_gradient_new(&le_state_a, &ct_state);
        return grad;
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

    pub fn exciton_ct_ct_coupling_gradient(
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
    ) -> Array1<f64> {
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        // Reference to the atoms of the total system.

        // get monomers for the first CT state
        let m_i: &Monomer = &self.monomers[ind_i];
        let m_j: &Monomer = &self.monomers[ind_j];

        // initialize the first CT state
        let state_1 = if hole_i {
            // Indices of the occupied orbitals of Monomer J.
            let occs = m_i.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            let virts = m_j.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_i];
            let virt: usize = virts[ct_j];

            ReducedBasisState::CT(ReducedCT {
                energy: 0.0,
                hole: ReducedParticle {
                    m_index: m_i.index,
                    ct_index: ct_i,
                    mo: ReducedMO {
                        c: m_i.properties.mo_coeff(occ).unwrap().to_owned(),
                        index: occ,
                    },
                },
                electron: ReducedParticle {
                    m_index: m_j.index,
                    ct_index: ct_j,
                    mo: ReducedMO {
                        c: m_j.properties.mo_coeff(virt).unwrap().to_owned(),
                        index: virt,
                    },
                },
                state_coefficient: 0.0,
            })
        } else {
            // Indices of the occupied orbitals of Monomer J.
            let occs = m_j.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            let virts = m_i.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_j];
            let virt: usize = virts[ct_i];

            ReducedBasisState::CT(ReducedCT {
                energy: 0.0,
                hole: ReducedParticle {
                    m_index: m_j.index,
                    ct_index: ct_j,
                    mo: ReducedMO {
                        c: m_j.properties.mo_coeff(occ).unwrap().to_owned(),
                        index: occ,
                    },
                },
                electron: ReducedParticle {
                    m_index: m_i.index,
                    ct_index: ct_i,
                    mo: ReducedMO {
                        c: m_i.properties.mo_coeff(virt).unwrap().to_owned(),
                        index: virt,
                    },
                },
                state_coefficient: 0.0,
            })
        };

        // get monomers for the second CT state
        let m_a: &Monomer = &self.monomers[ind_a];
        let m_b: &Monomer = &self.monomers[ind_b];

        let state_2 = if hole_a {
            // Indices of the occupied orbitals of Monomer J.
            let occs = m_a.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            let virts = m_b.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_a];
            let virt: usize = virts[ct_b];

            ReducedBasisState::CT(ReducedCT {
                energy: 0.0,
                hole: ReducedParticle {
                    m_index: m_a.index,
                    ct_index: ct_a,
                    mo: ReducedMO {
                        c: m_a.properties.mo_coeff(occ).unwrap().to_owned(),
                        index: occ,
                    },
                },
                electron: ReducedParticle {
                    m_index: m_b.index,
                    ct_index: ct_b,
                    mo: ReducedMO {
                        c: m_b.properties.mo_coeff(virt).unwrap().to_owned(),
                        index: virt,
                    },
                },
                state_coefficient: 0.0,
            })
        } else {
            // Indices of the occupied orbitals of Monomer J.
            let occs = m_b.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            let virts = m_a.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_b];
            let virt: usize = virts[ct_a];

            ReducedBasisState::CT(ReducedCT {
                energy: 0.0,
                hole: ReducedParticle {
                    m_index: m_b.index,
                    ct_index: ct_b,
                    mo: ReducedMO {
                        c: m_b.properties.mo_coeff(occ).unwrap().to_owned(),
                        index: occ,
                    },
                },
                electron: ReducedParticle {
                    m_index: m_a.index,
                    ct_index: ct_a,
                    mo: ReducedMO {
                        c: m_a.properties.mo_coeff(virt).unwrap().to_owned(),
                        index: virt,
                    },
                },
                state_coefficient: 0.0,
            })
        };
        let grad: Array1<f64> = self.exciton_coupling_gradient_new(&state_1, &state_2);
        return grad;
    }
}
