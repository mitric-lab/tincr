use crate::fmo::lcmo::cis_gradient::{
    ReducedBasisState, ReducedCT, ReducedLE, ReducedMO, ReducedParticle,
};
use crate::fmo::{BasisState, Monomer, SuperSystem};
use crate::initialization::{Atom, MO};
use crate::scc::scc_routine::RestrictedSCC;
use ndarray::prelude::*;

impl SuperSystem {
    pub fn test_diabatic_overlap(&mut self) {
        // scc routine
        self.prepare_scc();
        self.run_scc();
        let (s_full, s_occ, occ_vec) = self.build_occ_overlap();

        // // excited state preparation
        // let hamiltonian = self.build_lcmo_fock_matrix();
        // self.properties.set_lcmo_fock(hamiltonian);
        //
        // // Reference to the atoms of the total system.
        // let atoms: &[Atom] = &self.atoms[..];
        // let max_iter: usize = 50;
        // let tolerance: f64 = 1e-4;
        // // Number of LE states per monomer.
        // let n_le: usize = self.config.lcmo.n_le;
        // // Compute the n_le excited states for each monomer.
        // for mol in self.monomers.iter_mut() {
        //     mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
        //     mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);
        // }
        //
        // // Construct the diabatic basis states.
        // let states: Vec<BasisState> = self.create_diab_basis();
        // // Construct Reduced dibatic basis states
        // let mut reduced_states:Vec<ReducedBasisState> = Vec::new();
        // for state in states{
        //     match state {
        //         BasisState::LE(ref a) => {
        //             // get index and the Atom vector of the monomer
        //             let new_state = ReducedLE{
        //                 energy: 0.0,
        //                 monomer_index: a.monomer.index,
        //                 state_index: a.n,
        //                 state_coefficient: 0.0,
        //                 homo: a.monomer.properties.homo().unwrap(),
        //             };
        //
        //             reduced_states.push(ReducedBasisState::LE(new_state));
        //             // (vec![le_state], vec![monomer_ind])
        //         }
        //         BasisState::CT(ref a) => {
        //             // get indices
        //             let index_i: usize = a.hole.idx;
        //             let index_j: usize = a.electron.idx;
        //
        //             // get Atom vector and nocc of the monomer I
        //             let mol_i: &Monomer = &self.monomers[index_i];
        //             let nocc_i: usize = mol_i.properties.occ_indices().unwrap().len();
        //             drop(mol_i);
        //
        //             // get Atom vector and nocc of the monomer J
        //             let mol_j: &Monomer = &self.monomers[index_j];
        //             let nocc_j: usize = mol_j.properties.occ_indices().unwrap().len();
        //             drop(mol_j);
        //
        //             // get ct indices of the MOs
        //             let mo_i: usize =
        //                 (a.hole.mo.idx as i32 - (nocc_i - 1) as i32).abs() as usize;
        //             let mo_j: usize = a.electron.mo.idx - nocc_j;
        //
        //             reduced_states.push(ReducedBasisState::CT(ReducedCT{
        //                 energy: 0.0,
        //                 hole:ReducedParticle{
        //                     m_index:a.hole.idx,
        //                     ct_index:mo_i,
        //                     mo:ReducedMO{
        //                         c:a.hole.mo.c.to_owned(),
        //                         index:a.hole.mo.idx,
        //                     }
        //                 },
        //                 electron:ReducedParticle{
        //                     m_index:a.electron.idx,
        //                     ct_index:mo_j,
        //                     mo:ReducedMO{
        //                         c:a.electron.mo.c.to_owned(),
        //                         index:a.electron.mo.idx,
        //                     }
        //                 },
        //                 state_coefficient: 0.0,
        //             }));
        //         }
        //     };
        // }

        // let state_number:usize = reduced_states.len();
        // let mut overlap_matrix:Array2<f64> = Array2::zeros([state_number,state_number]);
        // // calculate the diabatic overlap of all states
        // for (i, state_i) in reduced_states.iter().enumerate() {
        //     // Only the upper triangle is calculated!
        //     for (j, state_j) in reduced_states[i..].iter().enumerate() {
        //         overlap_matrix[[i, j + i]] = self.diabatic_overlap(state_i, state_j);
        //         // if i == (j+i) {
        //         //     overlap_matrix[[i,j+i]] = 1.0;
        //         // }
        //         // else{
        //         //     overlap_matrix[[i, j + i]] = self.diabatic_overlap(state_i, state_j);
        //         // }
        //     }
        // }

        // println!("overlap matrix {}",overlap_matrix);
    }

    pub fn test_diabatic_le_le_overlap(
        &mut self,
        ind_a: usize,
        ind_b: usize,
        state_a: usize,
        state_b: usize,
    ) -> f64 {
        // Number of states per monomer.
        let n_states: usize = self.config.excited.nstates;

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

        let (mut s_full, mut s_occ, mut occ_vec) = self.build_occ_overlap();
        let overlap: f64 = self.diabatic_overlap(
            &le_state_a,
            &le_state_b,
            &mut s_full.view(),
            &mut s_occ.view(),
            occ_vec,
        );
        return overlap;
    }

    pub fn test_diabatic_le_ct_overlap(
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

        let (mut s_full, mut s_occ, mut occ_vec) = self.build_occ_overlap();
        let overlap: f64 = self.diabatic_overlap(
            &le_state_a,
            &ct_state,
            &mut s_full.view(),
            &mut s_occ.view(),
            occ_vec,
        );
        return overlap;
    }

    pub fn test_diabatic_ct_ct_overlap(
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

        let (mut s_full, mut s_occ, mut occ_vec) = self.build_occ_overlap();
        let overlap: f64 = self.diabatic_overlap(
            &state_1,
            &state_2,
            &mut s_full.view(),
            &mut s_occ.view(),
            occ_vec,
        );
        return overlap;
    }
}
