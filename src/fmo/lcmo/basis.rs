use std::any::Any;
use crate::excited_states::tda::*;
use crate::excited_states::ExcitedState;
use crate::fmo::{ESDPair, ExcitonStates, Monomer, Pair, PairType, SuperSystem};
use crate::initialization::{Atom, MO};
use crate::io::settings::LcmoConfig;
use crate::utils::Timer;
use nalgebra::{max, Vector3};
use ndarray::{concatenate, AssignElem};
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, UPLO};
use ndarray_npy::write_npy;
use std::fmt::{Display, Formatter};
use std::time::Instant;
use rayon::prelude::*;
use crate::{Davidson, initial_subspace};
use crate::fmo::lcmo::cis_gradient::{
    ReducedBasisState, ReducedCT, ReducedLE, ReducedMO, ReducedParticle,
};
use crate::properties::Properties;

impl SuperSystem {
    /// The diabatic basis states are constructed, which will be used for the Exciton-Hamiltonian.
    pub fn create_diab_basis(&self) -> Vec<BasisState> {
        let lcmo_config: LcmoConfig = self.config.lcmo.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;
        // Number of occupied orbitals for construction of CT states.
        let n_occ: usize = lcmo_config.n_holes;
        // Number of virtual orbitals for construction of CT states.
        let n_virt: usize = lcmo_config.n_particles;
        // The total number of states is given by: Sum_I n_LE_I + Sum_I Sum_J nocc_I * nvirt_J
        let n_states: usize = n_le * self.n_mol + n_occ * n_virt * self.n_mol * self.n_mol;
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];

        let mut states: Vec<BasisState> = Vec::with_capacity(n_states);
        // Create all LE states.
        for mol in self.monomers.iter() {
            let homo: usize = mol.properties.homo().unwrap();
            let q_ov: ArrayView2<f64> = mol.properties.q_ov().unwrap();
            for n in 0..n_le {
                let tdm: ArrayView1<f64> = mol.properties.ci_coefficient(n).unwrap();
                states.push(BasisState::LE(LocallyExcited {
                    monomer: mol,
                    n: n,
                    atoms: &atoms[mol.slice.atom_as_range()],
                    q_trans: q_ov.dot(&tdm),
                    occs: mol.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
                    virts: mol.properties.orbs_slice(homo + 1, None).unwrap(),
                    tdm: tdm,
                    tr_dipole: mol.properties.tr_dipole(n).unwrap(),
                }))
            }
        }

        // Create all CT states.
        for (idx, m_i) in self.monomers.iter().enumerate() {
            // Indices of the occupied orbitals of Monomer I.
            let occs_i: &[usize] = m_i.properties.occ_indices().unwrap();

            // Indices of the virtual orbitals of Monomer I.
            let virts_i: &[usize] = m_i.properties.virt_indices().unwrap();

            for m_j in self.monomers[idx + 1..].iter() {
                // Indices of the occupied orbitals of Monomer J.
                let occs_j: &[usize] = m_j.properties.occ_indices().unwrap();

                // Indices of the virtual orbitals of Monomer J.
                let virts_j: &[usize] = m_j.properties.virt_indices().unwrap();

                // First create all CT states from I to J.
                for occ in occs_i[occs_i.len() - n_occ..].iter().rev() {
                    for virt in virts_j[0..n_virt].iter() {
                        let mo_hole = MO::new(
                            m_i.properties.mo_coeff(*occ).unwrap(),
                            m_i.properties.orbe().unwrap()[*occ],
                            *occ,
                            m_i.properties.occupation().unwrap()[*occ],
                        );
                        let mo_elec = MO::new(
                            m_j.properties.mo_coeff(*virt).unwrap(),
                            m_j.properties.orbe().unwrap()[*virt],
                            *virt,
                            m_j.properties.occupation().unwrap()[*virt],
                        );
                        states.push(BasisState::CT(ChargeTransfer {
                            // system: &self,
                            hole: Particle {
                                idx: m_i.index,
                                atoms: &atoms[m_i.slice.atom_as_range()],
                                monomer: &m_i,
                                mo: mo_hole,
                            },
                            electron: Particle {
                                idx: m_j.index,
                                atoms: &atoms[m_j.slice.atom_as_range()],
                                monomer: &m_j,
                                mo: mo_elec,
                            },
                        }));
                    }
                }

                // And create all CT states from J to I.
                for occ in occs_j[occs_j.len() - n_occ..].iter().rev() {
                    for virt in virts_i[0..n_virt].iter() {
                        let mo_hole = MO::new(
                            m_j.properties.mo_coeff(*occ).unwrap(),
                            m_j.properties.orbe().unwrap()[*occ],
                            *occ,
                            m_j.properties.occupation().unwrap()[*occ],
                        );
                        let mo_elec = MO::new(
                            m_i.properties.mo_coeff(*virt).unwrap(),
                            m_i.properties.orbe().unwrap()[*virt],
                            *virt,
                            m_i.properties.occupation().unwrap()[*virt],
                        );
                        states.push(BasisState::CT(ChargeTransfer {
                            // system: &self,
                            hole: Particle {
                                idx: m_j.index,
                                atoms: &atoms[m_j.slice.atom_as_range()],
                                monomer: &m_j,
                                mo: mo_hole,
                            },
                            electron: Particle {
                                idx: m_i.index,
                                atoms: &atoms[m_i.slice.atom_as_range()],
                                monomer: &m_i,
                                mo: mo_elec,
                            },
                        }));
                    }
                }
            }
        }

        states
    }

    pub fn create_diab_basis_new(&self,n_ct:usize) -> Vec<BasisState> {
        let lcmo_config: LcmoConfig = self.config.lcmo.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];

        let mut states: Vec<BasisState> = Vec::new();
        // Create all LE states.
        for mol in self.monomers.iter() {
            let homo: usize = mol.properties.homo().unwrap();
            let q_ov: ArrayView2<f64> = mol.properties.q_ov().unwrap();
            for n in 0..n_le {
                let tdm: ArrayView1<f64> = mol.properties.ci_coefficient(n).unwrap();
                states.push(BasisState::LE(LocallyExcited {
                    monomer: mol,
                    n: n,
                    atoms: &atoms[mol.slice.atom_as_range()],
                    q_trans: q_ov.dot(&tdm),
                    occs: mol.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
                    virts: mol.properties.orbs_slice(homo + 1, None).unwrap(),
                    tdm: tdm,
                    tr_dipole: mol.properties.tr_dipole(n).unwrap(),
                }))
            }
        }

        if n_ct > 0{
            // Create all CT states.
            for (idx, m_i) in self.monomers.iter().enumerate() {
                for m_j in self.monomers[idx + 1..].iter() {
                    // get the PairType
                    let type_ij: PairType = self.properties.type_of_pair(m_i.index, m_j.index);

                    // create both CT states
                    let mut state_1 = PairChargeTransfer{
                        m_h:m_i,
                        m_l:m_j,
                        pair_type:type_ij,
                        properties:Properties::new(),
                    };
                    let mut state_2 = PairChargeTransfer{
                        m_h:m_j,
                        m_l:m_i,
                        pair_type:type_ij,
                        properties:Properties::new(),
                    };

                    // prepare the TDA calculation of both states
                    state_1.prepare_ct_tda(
                        self.properties.gamma().unwrap(),
                        self.properties.gamma_lr().unwrap(),
                        self.properties.s().unwrap(),
                        atoms
                    );
                    state_2.prepare_ct_tda(
                        self.properties.gamma().unwrap(),
                        self.properties.gamma_lr().unwrap(),
                        self.properties.s().unwrap(),
                        atoms
                    );
                    // do the TDA calculation using the davidson routine
                    state_1.run_ct_tda(atoms,n_ct,150,1.0e-4);
                    state_2.run_ct_tda(atoms,n_ct,150,1.0e-4);

                    let q_ov_1:ArrayView2<f64> = state_1.properties.q_ov().unwrap();
                    let q_ov_2:ArrayView2<f64> = state_2.properties.q_ov().unwrap();

                    for n in 0..n_ct{
                        let tdm_1:ArrayView1<f64> = state_1.properties.ci_coefficient(n).unwrap();
                        let ct_1 = ChargeTransferPair{
                            m_h:m_i.index,
                            m_l:m_j.index,
                            state_index:n,
                            state_energy:state_1.properties.ci_eigenvalue(n).unwrap(),
                            eigenvectors: state_1.properties.tdm(n).unwrap().to_owned(),
                            q_tr: q_ov_1.dot(&tdm_1),
                            tr_dipole: state_1.properties.tr_dipole(n).unwrap(),
                        };

                        let tdm_2:ArrayView1<f64> = state_2.properties.ci_coefficient(n).unwrap();
                        let ct_2 = ChargeTransferPair{
                            m_h:m_j.index,
                            m_l:m_i.index,
                            state_index:n,
                            state_energy:state_2.properties.ci_eigenvalue(n).unwrap(),
                            eigenvectors:state_2.properties.tdm(n).unwrap().to_owned(),
                            q_tr:q_ov_2.dot(&tdm_2),
                            tr_dipole:state_1.properties.tr_dipole(n).unwrap(),
                        };

                        states.push(BasisState::PairCT(ct_1));
                        states.push(BasisState::PairCT(ct_2));
                    }
                }
            }
        }

        states
    }

    pub fn create_diabatic_hamiltonian(&mut self)->(Array2<f64>){
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 100;
        let tolerance: f64 = 1e-5;
        // Number of LE states per monomer.
        let n_le: usize = self.config.lcmo.n_le;
        // Compute the n_le excited states for each monomer.
        for mol in self.monomers.iter_mut() {
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
            mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);
        }
        // Construct the diabatic basis states.
        let states: Vec<BasisState> = self.create_diab_basis();
        // Dimension of the basis states.
        let dim: usize = states.len();
        let mut h: Array2<f64> = Array2::zeros([dim, dim]);

        let arr:Vec<Array1<f64>> = states.par_iter().enumerate().map(|(i,state_i)|{
            let mut arr:Array1<f64> = Array1::zeros(dim);
            for (j, state_j) in states[i..].iter().enumerate() {
                arr[j+i] = self.exciton_coupling(state_i, state_j);
            }
            arr
        }).collect();

        for (i, arr) in arr.iter().enumerate(){
            h.slice_mut(s![i,..]).assign(&arr);
        }

        // fill the lower triagonal block of the matrix
        h = &h +&h.t() - Array::from_diag(&h.diag());

        // Construct Reduced dibatic basis states
        let mut reduced_states:Vec<ReducedBasisState> = Vec::new();
        for (idx,state) in states.iter().enumerate(){
            match state {
                BasisState::LE(ref a) => {
                    // get index and the Atom vector of the monomer
                    let new_state = ReducedLE{
                        energy: h[[idx,idx]],
                        monomer_index: a.monomer.index,
                        state_index: a.n,
                        state_coefficient: 0.0,
                        homo: a.monomer.properties.homo().unwrap(),
                    };

                    reduced_states.push(ReducedBasisState::LE(new_state));
                    // (vec![le_state], vec![monomer_ind])
                }
                BasisState::CT(ref a) => {
                    // get indices
                    let index_i: usize = a.hole.idx;
                    let index_j: usize = a.electron.idx;

                    // get Atom vector and nocc of the monomer I
                    let mol_i: &Monomer = &self.monomers[index_i];
                    let nocc_i: usize = mol_i.properties.occ_indices().unwrap().len();
                    drop(mol_i);

                    // get Atom vector and nocc of the monomer J
                    let mol_j: &Monomer = &self.monomers[index_j];
                    let nocc_j: usize = mol_j.properties.occ_indices().unwrap().len();
                    drop(mol_j);

                    // get ct indices of the MOs
                    let mo_i: usize =
                        (a.hole.mo.idx as i32 - (nocc_i - 1) as i32).abs() as usize;
                    let mo_j: usize = a.electron.mo.idx - nocc_j;

                    reduced_states.push(ReducedBasisState::CT(ReducedCT{
                        energy: h[[idx,idx]],
                        hole:ReducedParticle{
                            m_index:a.hole.idx,
                            ct_index:mo_i,
                            mo:ReducedMO{
                                c:a.hole.mo.c.to_owned(),
                                index:a.hole.mo.idx,
                            }
                        },
                        electron:ReducedParticle{
                            m_index:a.electron.idx,
                            ct_index:mo_j,
                            mo:ReducedMO{
                                c:a.electron.mo.c.to_owned(),
                                index:a.electron.mo.idx,
                            }
                        },
                        state_coefficient: 0.0,
                    }));
                },
                _ =>{},
            };
        }
        // save the basis in the properties
        self.properties.set_basis_states(reduced_states);

        return (h);
    }

    pub fn create_exciton_hamiltonian(&mut self) {
        let timer = Instant::now();
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        println!("elapsed time 1 {}",timer.elapsed().as_secs_f64());

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 50;
        let tolerance: f64 = 1e-4;
        // Number of LE states per monomer.
        let n_le: usize = self.config.lcmo.n_le;
        // Compute the n_le excited states for each monomer.
        // for mol in self.monomers.iter_mut() {
        //     mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
        //     mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);
        // }
        self.monomers.par_iter_mut().for_each(|mol| {
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
            mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);
        });

        println!("elapsed time 2 {}",timer.elapsed().as_secs_f64());

        // Construct the diabatic basis states.
        let states: Vec<BasisState> = self.create_diab_basis();
        // Dimension of the basis states.
        let dim: usize = states.len();
        println!("Dimension of the Hamiltonian: {}", dim);
        // Initialize the Exciton-Hamiltonian.
        let mut h: Array2<f64> = Array2::zeros([dim, dim]);

        println!("elapsed time 3 {}",timer.elapsed().as_secs_f64());

        let arr:Vec<Array1<f64>> = states.par_iter().enumerate().map(|(i,state_i)|{
            let mut arr:Array1<f64> = Array1::zeros(dim);
            for (j, state_j) in states[i..].iter().enumerate() {
                arr[j+i] = self.exciton_coupling(state_i, state_j);
            }
            arr
        }).collect();

        for (i, arr) in arr.iter().enumerate(){
            h.slice_mut(s![i,..]).assign(&arr);
        }


        // for (i, state_i) in states.iter().enumerate() {
        //     // Only the upper triangle is calculated!
        //     for (j, state_j) in states[i..].iter().enumerate() {
        //         h[[i, j + i]] = self.exciton_coupling(state_i, state_j);
        //     }
        // }
        // write_npy("diabatic_energies.npy",&h.diag());
        // write_npy("diabatic_hamiltonian.npy",&h);
        println!("elapsed time 4 {}",timer.elapsed().as_secs_f64());
        // write_npy("diabatic_hamiltonian.npy", &h);

        // The Hamiltonian is returned. Only the upper triangle is filled, so this has to be
        // considered when using eigh.
        // TODO: If the Hamiltonian gets to big, the Davidson diagonalization should be used.
        let (energies, eigvectors): (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Lower).unwrap();
        // let diag = h.diag();
        // h = &h + &h.t() - Array::from_diag(&diag);
        // let nroots:usize = self.config.excited.nstates;
        // let guess: Array2<f64> = initial_subspace(h.diag(), nroots);
        // let davidson: Davidson = Davidson::new(&mut h, guess, nroots, 1e-4,100).unwrap();
        // let energies = davidson.eigenvalues;
        // let eigvectors = davidson.eigenvectors;

        write_npy("fmo_energies.npy",&energies);

        println!("elapsed time 5 {}",timer.elapsed().as_secs_f64());

        let n_occ: usize = self.monomers.iter().map(|m| m.properties.n_occ().unwrap()).sum();
        let n_virt: usize = self.monomers.iter().map(|m| m.properties.n_virt().unwrap()).sum();
        let n_orbs: usize = n_occ + n_virt;
        let mut occ_orbs: Array2<f64> = Array2::zeros([n_orbs, n_occ]);
        let mut virt_orbs: Array2<f64> = Array2::zeros([n_orbs, n_virt]);

        for mol in self.monomers.iter() {
            let mol_orbs: ArrayView2<f64> = mol.properties.orbs().unwrap();
            let lumo: usize = mol.properties.lumo().unwrap();
            occ_orbs.slice_mut(s![mol.slice.orb, mol.slice.occ_orb]).assign(&mol_orbs.slice(s![.., ..lumo]));
            virt_orbs.slice_mut(s![mol.slice.orb, mol.slice.virt_orb]).assign(&mol_orbs.slice(s![.., lumo..]));
        }

        let orbs: Array2<f64> = concatenate![Axis(1), occ_orbs, virt_orbs];
        // write_npy("/Users/hochej/Downloads/lcmo_energies.npy", &energies.view());
        let exciton = ExcitonStates::new(self.properties.last_energy().unwrap(),
                                         (energies.clone(), eigvectors.clone()), states.clone(),
                                         (n_occ, n_virt), orbs, self.properties.s().unwrap(),&self.atoms);

        exciton.spectrum_to_npy("lcmo_spec.npy");
        exciton.spectrum_to_txt("lcmo_spec.txt");
        // exciton.ntos_to_molden(&self.atoms, 1, "ntos_fmo.molden");
        println!("{}", exciton);
        exciton.print_state_contributions(0);

        // self.properties.set_ci_eigenvalues(energies);
        // self.properties.set_ci_coefficients(eigvectors);
    }

    pub fn create_exciton_hamiltonian_new(&mut self) {
        let timer = Instant::now();
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        println!("elapsed time 1 {}",timer.elapsed().as_secs_f64());

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 50;
        let tolerance: f64 = 1e-4;
        // Number of LE states per monomer.
        let n_le: usize = self.config.lcmo.n_le;

        self.monomers.par_iter_mut().for_each(|mol| {
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
            mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);
        });

        println!("elapsed time 2 {}",timer.elapsed().as_secs_f64());

        // Construct the diabatic basis states.
        let states: Vec<BasisState> = self.create_diab_basis_new(self.config.lcmo.n_holes);

        let dim: usize = states.len();
        println!("Dimension of the Hamiltonian: {}", dim);
        // Initialize the Exciton-Hamiltonian.
        let mut h: Array2<f64> = Array2::zeros([dim, dim]);

        println!("elapsed time 3 {}",timer.elapsed().as_secs_f64());

        let arr:Vec<Array1<f64>> = states.par_iter().enumerate().map(|(i,state_i)|{
            let mut arr:Array1<f64> = Array1::zeros(dim);
            for (j, state_j) in states[i..].iter().enumerate() {
                arr[j+i] = self.exciton_coupling(state_i, state_j);
            }
            arr
        }).collect();

        for (i, arr) in arr.iter().enumerate(){
            h.slice_mut(s![i,..]).assign(&arr);
        }

        println!("elapsed time 4 {}",timer.elapsed().as_secs_f64());

        let (energies, eigvectors): (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Lower).unwrap();

        println!("elapsed time 5 {}",timer.elapsed().as_secs_f64());
        // let diag = h.diag();
        // h = &h + &h.t() - Array::from_diag(&diag);
        // let nroots:usize = self.config.excited.nstates;
        // let guess: Array2<f64> = initial_subspace(h.diag(), nroots);
        // let davidson: Davidson = Davidson::new(&mut h, guess, nroots, 1e-4,100).unwrap();
        // let energies = davidson.eigenvalues;
        // let eigvectors = davidson.eigenvectors;

        write_npy("fmo_energies.npy",&energies);

        let n_occ: usize = self.monomers.iter().map(|m| m.properties.n_occ().unwrap()).sum();
        let n_virt: usize = self.monomers.iter().map(|m| m.properties.n_virt().unwrap()).sum();
        let n_orbs: usize = n_occ + n_virt;
        let mut occ_orbs: Array2<f64> = Array2::zeros([n_orbs, n_occ]);
        let mut virt_orbs: Array2<f64> = Array2::zeros([n_orbs, n_virt]);

        for mol in self.monomers.iter() {
            let mol_orbs: ArrayView2<f64> = mol.properties.orbs().unwrap();
            let lumo: usize = mol.properties.lumo().unwrap();
            occ_orbs.slice_mut(s![mol.slice.orb, mol.slice.occ_orb]).assign(&mol_orbs.slice(s![.., ..lumo]));
            virt_orbs.slice_mut(s![mol.slice.orb, mol.slice.virt_orb]).assign(&mol_orbs.slice(s![.., lumo..]));
        }

        let orbs: Array2<f64> = concatenate![Axis(1), occ_orbs, virt_orbs];
        // write_npy("/Users/hochej/Downloads/lcmo_energies.npy", &energies.view());
        let exciton = ExcitonStates::new(self.properties.last_energy().unwrap(),
                                         (energies.clone(), eigvectors.clone()), states.clone(),
                                         (n_occ, n_virt), orbs, self.properties.s().unwrap(),&self.atoms);

        exciton.spectrum_to_npy("lcmo_spec.npy");
        exciton.spectrum_to_txt("lcmo_spec.txt");
        // exciton.ntos_to_molden(&self.atoms, 1, "ntos_fmo.molden");
        println!("{}", exciton);
        exciton.print_state_contributions(0);
    }
}

/// Different types of diabatic basis states that are used for the FMO-exciton model.
#[derive(Clone, Debug)]
pub enum BasisState<'a> {
    // Locally excited state that is on one monomer.
    LE(LocallyExcited<'a>),
    // Charge transfer state between two different monomers and two MOs.
    CT(ChargeTransfer<'a>),
    PairCT(ChargeTransferPair),
}

impl Display for BasisState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BasisState::LE(state) => write!(f, "{}", state),
            BasisState::CT(state) => write!(f, "{}", state),
            BasisState::PairCT(state) => write!(f,"{}",state),
        }
    }
}

/// Type that holds all the relevant data that characterize a locally excited diabatic basis state.
#[derive(Clone, Debug)]
pub struct LocallyExcited<'a> {
    // Reference to the corresponding monomer.
    pub monomer: &'a Monomer,
    // Number of excited state for the monomer. 1 -> S1, 2 -> S2, ...
    pub n: usize,
    // The atoms corresponding to the monomer of this state.
    pub atoms: &'a [Atom],
    //
    pub q_trans: Array1<f64>,
    //
    pub occs: ArrayView2<'a, f64>,
    //
    pub virts: ArrayView2<'a, f64>,
    //
    pub tdm: ArrayView1<'a, f64>,
    //
    pub tr_dipole: Vector3<f64>,
}

impl Display for LocallyExcited<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LE(S{}) on Frag. {:>4}",
            self.n + 1,
            self.monomer.index + 1
        )
    }
}

impl PartialEq for LocallyExcited<'_> {
    /// Two LE states are considered equal, if it is the same excited state on the same monomer.
    fn eq(&self, other: &Self) -> bool {
        self.monomer.index == other.monomer.index && self.n == other.n
    }
}

#[derive(Clone, Debug)]
pub struct PairChargeTransfer<'a>{
    pub m_h:&'a Monomer,
    pub m_l:&'a Monomer,
    pub pair_type:PairType,
    pub properties:Properties,
}

#[derive(Clone, Debug)]
pub struct ChargeTransferPair{
    pub m_h:usize,
    pub m_l:usize,
    pub state_index:usize,
    pub state_energy:f64,
    pub eigenvectors:Array2<f64>,
    pub q_tr:Array1<f64>,
    pub tr_dipole: Vector3<f64>,
}

impl PartialEq for ChargeTransferPair{
    fn eq(&self,other:&Self) ->bool{
        self.m_h == other.m_h && self.m_l == other.m_l && self.state_index == other.state_index
    }
}

impl Display for ChargeTransferPair{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CT {}: {} -> {}", self.state_index,self.m_h+1, self.m_l+1)
    }
}

/// Type that holds all the relevant data that characterize a charge-transfer diabatic basis state.
#[derive(Copy, Clone, Debug)]
pub struct ChargeTransfer<'a> {
    // // Reference to the total system. This is needed to access the complete Gamma matrix.
    // pub system: &'a SuperSystem,
    // The hole of the CT state.
    pub hole: Particle<'a>,
    // The electron of the CT state.
    pub electron: Particle<'a>,
}

impl Display for ChargeTransfer<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CT: {} -> {}", self.hole, self.electron)
    }
}

impl PartialEq for ChargeTransfer<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.hole == other.hole && self.electron == other.electron
    }
}
#[derive(Copy, Clone, Debug)]
pub struct Particle<'a> {
    /// The index of the corresponding monomer.
    pub idx: usize,
    /// The atoms of the corresponding monomer.
    pub atoms: &'a [Atom],
    /// The corresponding monomer itself.
    pub monomer: &'a Monomer,
    /// The corresponding molecular orbital.
    pub mo: MO<'a>,
}

impl Display for Particle<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Frag. {: >4}", self.monomer.index + 1)
    }
}

// TODO: this definition could lead to mistakes for degenerate orbitals
impl PartialEq for Particle<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.mo.e == other.mo.e
    }
}
