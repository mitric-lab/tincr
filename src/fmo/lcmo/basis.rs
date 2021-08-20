use crate::fmo::{Monomer, SuperSystem};
use crate::initialization::Atom;
use ndarray::prelude::*;
use nalgebra::max;
use crate::excited_states::tda::TDA;

impl SuperSystem {
    /// The diabatic basis states are constructed, which will be used for the Exciton-Hamiltonian.
    pub fn create_diab_basis(&self) -> Vec<BasisState> {
        // TODO: The first three numbers should be read from the input file.
        let max_iter: usize = 50;
        let tolerance: f64 = 1e-4;
        // Number of LE states per monomer.
        let n_le: usize = 1;
        // Number of occupied orbitals for construction of CT states.
        let n_occ: usize = 1;
        // Number of virtual orbitals for construction of CT states.
        let n_virt: usize = 1;
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
                println!("I {} J {}", m_i.index, m_j.index);
                // First create all CT states from I to J.
                for occ in occs_i[occs_i.len() - n_occ..].iter().rev() {
                    for virt in virts_j[0..n_virt].iter() {
                        states.push(BasisState::CT(ChargeTransfer {
                            system: &self,
                            hole: Particle {
                                idx: m_i.index,
                                orb: m_i.properties.mo_coeff(*occ).unwrap(),
                                atoms: &atoms[m_i.slice.atom_as_range()],
                                monomer: &m_i,
                                energy: m_i.properties.orbe().unwrap()[*occ],
                            },
                            electron: Particle {
                                idx: m_j.index,
                                orb: m_j.properties.mo_coeff(*virt).unwrap(),
                                atoms: &atoms[m_j.slice.atom_as_range()],
                                monomer: &m_j,
                                energy: m_j.properties.orbe().unwrap()[*virt]
                            }
                        }));
                    }
                }

                // And create all CT states from J to I.
                for occ in occs_j[occs_j.len() - n_occ..].iter().rev() {
                    for virt in virts_i[0..n_virt].iter() {
                        states.push(BasisState::CT(ChargeTransfer {
                            system: &self,
                            hole: Particle {
                                idx: m_j.index,
                                orb: m_j.properties.mo_coeff(*occ).unwrap(),
                                atoms: &atoms[m_j.slice.atom_as_range()],
                                monomer: &m_j,
                                energy: m_j.properties.orbe().unwrap()[*occ]
                            },
                            electron: Particle {
                                idx: m_i.index,
                                orb: m_i.properties.mo_coeff(*virt).unwrap(),
                                atoms: &atoms[m_i.slice.atom_as_range()],
                                monomer: &m_i,
                                energy: m_i.properties.orbe().unwrap()[*virt]
                            }
                        }));
                    }
                }
            }
        }
        println!("LEN STATES {}", states.len());
        for state in states.iter() {
            match state {
                BasisState::CT(ct) => println!("hole {} elec {}", ct.hole.idx, ct.electron.idx),
                BasisState::LE(le) => println!("le on {}", le.monomer.index),
            }
        }
        states
    }

    pub fn create_exciton_hamiltonian(&mut self) -> Array2<f64> {
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 50;
        let tolerance: f64 = 1e-4;
        // Number of LE states per monomer.
        let n_le: usize = 1;
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

        for (i, state_i) in states.iter().enumerate() {
            // Only the upper triangle is calculated!
            for (j, state_j) in states[i..].iter().enumerate() {
                h[[i, j+i]] = self.exciton_coupling(state_i, state_j);
            }
        }
        // The Hamiltonian is returned. Only the upper triangle is filled, so this has to be
        // considered when using eigh.
        h
    }
}

/// Different types of diabatic basis states that are used for the FMO-exciton model.
pub enum BasisState<'a> {
    // Locally excited state that is on one monomer.
    LE(LocallyExcited<'a>),
    // Charge transfer state between two different monomers and two MOs.
    CT(ChargeTransfer<'a>),
}

/// Type that holds all the relevant data that characterize a locally excited diabatic basis state.
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
}

impl PartialEq for LocallyExcited<'_> {
    /// Two LE states are considered equal, if it is the same excited state on the same monomer.
    fn eq(&self, other: &Self) -> bool {
        self.monomer.index == other.monomer.index && self.n == other.n
    }
}

/// Type that holds all the relevant data that characterize a charge-transfer diabatic basis state.
pub struct ChargeTransfer<'a> {
    // Reference to the total system. This is needed to access the complete Gamma matrix.
    pub system: &'a SuperSystem,
    // The hole of the CT state.
    pub hole: Particle<'a>,
    // The electron of the CT state.
    pub electron: Particle<'a>,
}

impl PartialEq for ChargeTransfer<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.hole == other.hole && self.electron == other.electron
    }
}

pub struct Particle<'a> {
    // The index of the corresponding monomer.
    pub idx: usize,
    // The MO coefficient of the corresponding orbital.
    pub orb: ArrayView1<'a, f64>,
    // The atoms of the corresponding monomer.
    pub atoms: &'a [Atom],
    // The corresponding monomer itself.
    pub monomer: &'a Monomer,
    // The MO-energy of the corresponding orbital.
    pub energy: f64,
}

// TODO: this definition could lead to mistakes for degenerate orbitals
impl PartialEq for Particle<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.energy == other.energy
    }
}