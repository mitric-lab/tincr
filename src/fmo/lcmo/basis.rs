use crate::fmo::{Monomer, SuperSystem, ExcitonStates};
use crate::initialization::{Atom, MO};
use ndarray::prelude::*;
use nalgebra::{max, Vector3};
use crate::excited_states::tda::*;
use ndarray_linalg::{Eigh, UPLO};
use std::fmt::{Display, Formatter};
use crate::excited_states::ExcitedState;
use ndarray::concatenate;
use crate::io::settings::LcmoConfig;
use ndarray_npy::write_npy;
use crate::utils::Timer;

impl SuperSystem {
    /// The diabatic basis states are constructed, which will be used for the Exciton-Hamiltonian.
    pub fn create_diab_basis(&self) -> Vec<BasisState> {
        // TODO: The first three numbers should be read from the input file.
        let max_iter: usize = 50;
        let tolerance: f64 = 1e-4;
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
                    occs: mol.properties.orbs_slice(0, Some(homo+1)).unwrap(),
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

            for m_j in self.monomers[idx+1..].iter() {
                // Indices of the occupied orbitals of Monomer J.
                let occs_j: &[usize] = m_j.properties.occ_indices().unwrap();

                // Indices of the virtual orbitals of Monomer J.
                let virts_j: &[usize] = m_j.properties.virt_indices().unwrap();

                // First create all CT states from I to J.
                for occ in occs_i[occs_i.len() - n_occ..].iter().rev() {
                    for virt in virts_j[0..n_virt].iter() {
                        let mo_hole = MO::new(m_i.properties.mo_coeff(*occ).unwrap(),
                                              m_i.properties.orbe().unwrap()[*occ],
                                             *occ,
                                              m_i.properties.occupation().unwrap()[*occ]);
                        let mo_elec = MO::new(m_j.properties.mo_coeff(*virt).unwrap(),
                                              m_j.properties.orbe().unwrap()[*virt],
                                             *virt,
                                              m_j.properties.occupation().unwrap()[*virt]);
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
                            }
                        }));
                    }
                }

                // And create all CT states from J to I.
                for occ in occs_j[occs_j.len() - n_occ..].iter().rev() {
                    for virt in virts_i[0..n_virt].iter() {
                        let mo_hole = MO::new(m_j.properties.mo_coeff(*occ).unwrap(),
                                              m_j.properties.orbe().unwrap()[*occ],
                                              *occ,
                                              m_j.properties.occupation().unwrap()[*occ]);
                        let mo_elec = MO::new(m_i.properties.mo_coeff(*virt).unwrap(),
                                              m_i.properties.orbe().unwrap()[*virt],
                                              *virt,
                                              m_i.properties.occupation().unwrap()[*virt]);
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
                            }
                        }));
                    }
                }
            }
        }

        states
    }

    pub fn create_exciton_hamiltonian(&mut self){
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
            mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le,  max_iter, tolerance);
        }

        // Construct the diabatic basis states.
        let states: Vec<BasisState> = self.create_diab_basis();
        // Dimension of the basis states.
        let dim: usize = states.len();
        // Initialize the Exciton-Hamiltonian.
        let mut h: Array2<f64> = Array2::zeros([dim, dim]);

        let timer: Timer = Timer::start();

        for (i, state_i) in states.iter().enumerate() {
            // Only the upper triangle is calculated!
            for (j, state_j) in states[i..].iter().enumerate() {
                h[[i, j+i]] = self.exciton_coupling(state_i, state_j);
            }
        }
        println!("time calculate matrix {}",timer);
        // The Hamiltonian is returned. Only the upper triangle is filled, so this has to be
        // considered when using eigh.
        // TODO: If the Hamiltonian gets to big, the Davidson diagonalization should be used.
        let (energies, eigvectors): (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Lower).unwrap();
        println!("Eigh of matrix {}",timer);

        // let n_occ: usize = self.monomers.iter().map(|m| m.properties.n_occ().unwrap()).sum();
        // let n_virt: usize = self.monomers.iter().map(|m| m.properties.n_virt().unwrap()).sum();
        // let n_orbs: usize = n_occ + n_virt;
        // let mut occ_orbs: Array2<f64> = Array2::zeros([n_orbs, n_occ]);
        // let mut virt_orbs: Array2<f64> = Array2::zeros([n_orbs, n_virt]);
        //
        // for mol in self.monomers.iter() {
        //     let mol_orbs: ArrayView2<f64> = mol.properties.orbs().unwrap();
        //     let lumo: usize = mol.properties.lumo().unwrap();
        //     occ_orbs.slice_mut(s![mol.slice.orb, mol.slice.occ_orb]).assign(&mol_orbs.slice(s![.., ..lumo]));
        //     virt_orbs.slice_mut(s![mol.slice.orb, mol.slice.virt_orb]).assign(&mol_orbs.slice(s![.., lumo..]));
        // }
        //
        // let orbs: Array2<f64> = concatenate![Axis(1), occ_orbs, virt_orbs];
        // // write_npy("/Users/hochej/Downloads/lcmo_energies.npy", &energies.view());
        // let exciton = ExcitonStates::new(self.properties.last_energy().unwrap(),
        //                                  (energies.clone(), eigvectors.clone()), states.clone(),
        //                                  (n_occ, n_virt), orbs);

        // exciton.spectrum_to_npy("/Users/hochej/Downloads/lcmo_spec.npy");
        // exciton.spectrum_to_txt("/Users/hochej/Downloads/lcmo_spec.txt");
        // exciton.ntos_to_molden(&self.atoms, 1, "/Users/hochej/Downloads/ntos_fmo.molden");
        // println!("{}", exciton);

        self.properties.set_ci_eigenvalues(energies);
        self.properties.set_ci_coefficients(eigvectors);
    }

    pub fn exciton_ct_energy(
        &mut self,
        index_i:usize,
        index_j:usize,
        ct_ind_i:usize,
        ct_ind_j:usize,
        hole_i:bool,
    ) -> f64
    {
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];

        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        // get occupied and virtual orbitals
        let mut occs:&[usize];
        let mut virts:&[usize];
        let mut hole:MO;
        let mut elec:MO;

        let state:BasisState = if hole_i{
            // Indices of the occupied orbitals of Monomer J.
            occs = m_i.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_j.properties.virt_indices().unwrap();
            // set ct indices
            let nocc:usize = occs.len();
            let occ:usize = occs[nocc-1-ct_ind_i];
            let virt:usize = virts[ct_ind_j];

            // create hole and electron
            hole = MO::new(m_i.properties.mo_coeff(occ).unwrap(),
                                  m_i.properties.orbe().unwrap()[occ],
                                  occ,
                                  m_i.properties.occupation().unwrap()[occ]);
            elec = MO::new(m_j.properties.mo_coeff(virt).unwrap(),
                                  m_j.properties.orbe().unwrap()[virt],
                                  virt,
                                  m_j.properties.occupation().unwrap()[virt]);

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
                }
            })
        } else{
            // Indices of the occupied orbitals of Monomer J.
            occs = m_j.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_i.properties.virt_indices().unwrap();
            // set ct indices
            let nocc:usize = occs.len();
            let occ:usize = occs[nocc-1-ct_ind_j];
            let virt:usize = virts[ct_ind_i];

            // create hole and electron
            hole = MO::new(m_j.properties.mo_coeff(occ).unwrap(),
                                  m_j.properties.orbe().unwrap()[occ],
                                  occ,
                                  m_j.properties.occupation().unwrap()[occ]);
            elec = MO::new(m_i.properties.mo_coeff(virt).unwrap(),
                                  m_i.properties.orbe().unwrap()[virt],
                                  virt,
                                  m_i.properties.occupation().unwrap()[virt]);

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
                }
            })
        };
        let val:f64 = self.exciton_coupling(&state,&state);

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
            mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le,  max_iter, tolerance);
        }

        // Construct the diabatic basis states.
        let states: Vec<BasisState> = self.create_diab_basis();

        let ct_state = &states[2*n_le];
        let val:f64 = self.exciton_coupling(ct_state,ct_state);

        return val;
    }
}

/// Different types of diabatic basis states that are used for the FMO-exciton model.
#[derive(Clone,Debug)]
pub enum BasisState<'a> {
    // Locally excited state that is on one monomer.
    LE(LocallyExcited<'a>),
    // Charge transfer state between two different monomers and two MOs.
    CT(ChargeTransfer<'a>),
}

impl Display for BasisState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BasisState::LE(state) => write!(f, "{}", state),
            BasisState::CT(state) => write!(f, "{}", state),
        }
    }
}


/// Type that holds all the relevant data that characterize a locally excited diabatic basis state.
#[derive(Clone,Debug)]
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
        write!(f, "LE(S{}) on Frag. {:>4}", self.n + 1, self.monomer.index + 1)
    }
}

impl PartialEq for LocallyExcited<'_> {
    /// Two LE states are considered equal, if it is the same excited state on the same monomer.
    fn eq(&self, other: &Self) -> bool {
        self.monomer.index == other.monomer.index && self.n == other.n
    }
}

/// Type that holds all the relevant data that characterize a charge-transfer diabatic basis state.
#[derive(Copy, Clone,Debug)]
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
#[derive(Copy, Clone,Debug)]
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