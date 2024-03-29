use crate::fmo::{BasisState, ChargeTransfer, ChargeTransferPair, LocallyExcited, Monomer, Particle};
use crate::initialization::Atom;
use ndarray::concatenate;
use ndarray::prelude::*;
use std::iter;
use std::io::empty;
use std::ops::AddAssign;
use std::thread::panicking;
use crate::SuperSystem;

/// Type to determine the kind of orbitals that are used for the LE-LE transition charges. `Hole`
/// specifies the use of occupied orbitals, while `Electron` means that virtual orbitals are used.
pub enum ElecHole {
    Hole,
    Electron,
}

/// Computes the Mulliken transition charges between two sets of orbitals. These orbitals
/// are part of two LE states, that are either on one monomer or two different ones. The overlap
/// matrix `s`, is the overlap matrix between the basis functions of the corresponding monomers.
pub fn q_lele<'a>(
    a: &'a LocallyExcited<'a>,
    b: &'a LocallyExcited<'a>,
    kind_a: ElecHole,
    kind_b: ElecHole,
    s: ArrayView2<f64>,
) -> Array3<f64> {
    // Number of atoms.
    let n_atoms_i: usize = a.atoms.len();
    let n_atoms_j: usize = b.atoms.len();
    let n_atoms: usize = n_atoms_i + n_atoms_j;
    // Check if the occupied or virtual orbitals of the first LE state are needed.
    let orbs_i: ArrayView2<f64> = match kind_a {
        ElecHole::Hole => a.occs,
        ElecHole::Electron => a.virts,
    };
    // Check if the occupied or virtual orbitals of the second LE state are needed.
    let orbs_j: ArrayView2<f64> = match kind_b {
        ElecHole::Hole => b.occs,
        ElecHole::Electron => b.virts,
    };
    // Number of molecular orbitals on monomer I.
    let dim_i: usize = orbs_i.ncols();
    // Number of molecular orbitals on monomer J.
    let dim_j: usize = orbs_j.ncols();
    // The transition charges between the two sets of MOs  are initialized.
    let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_i, dim_j]);
    // Matrix product of overlap matrix with the orbitals on I.
    let sc_mu_j: Array2<f64> = s.dot(&orbs_j);
    // Matrix product of overlap matrix with the orbitals on J.
    let sc_mu_i: Array2<f64> = s.t().dot(&orbs_i);
    let mut mu: usize = 0;
    // Iteration over all atoms (I).
    for (atom, mut q_n) in a.atoms.iter().zip(q_trans.slice_mut(s![0..n_atoms_i, .., ..]).axis_iter_mut(Axis(0))) {
        // Iteration over atomic orbitals mu on I.
        for _ in 0..atom.n_orbs {
            // Iteration over orbitals i on monomer I. orb_i -> C_(mu i) (mu on I, i on I)
            for (orb_i, mut q_i) in orbs_i.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                // Iteration over S * C_J on monomer J. sc -> SC_(mu j) (mu on I, j on J)
                for (sc, mut q) in sc_mu_j.row(mu).iter().zip(q_i.iter_mut()) {
                    // The transition charge is computed.
                    *q += orb_i * sc;
                }
            }
            mu += 1;
        }
    }
    mu = 0;
    // Iteration over all atoms J.
    for (atom, mut q_n) in b.atoms.iter().zip(q_trans.slice_mut(s![n_atoms_i.., .., ..]).axis_iter_mut(Axis(0))) {
        // Iteration over atomic orbitals mu on J.
        for _ in 0..atom.n_orbs {
            // Iteration over occupied orbital i. sc -> SC_(mu i) (mu on J, i on I)
            for (sc, mut q_i) in sc_mu_i.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                // Iteration over occupied orbital j. C_(mu j) (mu on J, j on J)
                for (orb_j, mut q) in orbs_j.row(mu).iter().zip(q_i.iter_mut()) {
                    // The transition charge is computed.
                    *q += orb_j * sc;
                }
            }
            mu += 1;
        }
    }
    0.5 * q_trans
}

// /// Computes the Mulliken transition charges between two sets of orbitals. These orbitals
// /// are part of two LE states, that are either on one monomer or two different ones. The overlap
// /// matrix `s`, is the overlap matrix between the basis functions of the corresponding monomers.
// pub fn q_lele<'a>(
//     a: &'a LocallyExcited<'a>,
//     b: &'a LocallyExcited<'a>,
//     kind_a: ElecHole,
//     kind_b: ElecHole,
//     s: ArrayView2<f64>,
// ) -> Array3<f64> {
//     // Check if the transition charges are on the same monomer or not.
//     let inter: bool = a.monomer != b.monomer;
//
//     // Number of atoms.
//     let n_atoms: usize = if inter {
//         a.monomer.n_atoms + b.monomer.n_atoms
//     } else {
//         a.monomer.n_atoms
//     };
//     // Check if the occupied or virtual orbitals of the first LE state are needed.
//     let orbs_i: ArrayView2<f64> = match kind_a {
//         ElecHole::Hole => a.occs,
//         ElecHole::Electron => a.virts,
//     };
//     // Check if the occupied or virtual orbitals of the second LE state are needed.
//     let orbs_j: ArrayView2<f64> = match kind_b {
//         ElecHole::Hole => b.occs,
//         ElecHole::Electron => b.virts,
//     };
//     // Number of molecular orbitals on monomer I.
//     let dim_i: usize = orbs_i.ncols();
//     // Number of molecular orbitals on monomer J.
//     let dim_j: usize = orbs_j.ncols();
//     // The transition charges between the two sets of MOs  are initialized.
//     let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_i, dim_j]);
//
//     // Matrix product of overlap matrix with the orbitals on I.
//     let sc_i: Array2<f64> = s.dot(&orbs_j);
//     // Matrix product of overlap matrix with the orbitals on J.
//     let sc_j: Array2<f64> = s.t().dot(&orbs_i);
//     // Either append or sum the contributions, depending whether the orbitals are on the same monomer.
//     let sc_ij: Array2<f64> = if inter {
//         concatenate![Axis(0), sc_i, sc_j]
//     } else {
//         &orbs_i + &sc_i
//     };
//     let orbs_ij: Array2<f64> = if inter {
//         concatenate![Axis(0), orbs_i, orbs_j]
//     } else {
//         &orbs_j + &sc_j // TODO: CHECK IF CORRECT
//     };
//
//     // Iterator over the atoms.
//     let atom_iter = if inter {
//         a.atoms.iter().chain(b.atoms.iter())
//     } else {
//         let empty_slice: &[Atom] = &[];
//         a.atoms.iter().chain(empty_slice.iter())
//     };
//
//     let mut mu: usize = 0;
//     // Iteration over all atoms (A).
//     for (atom, mut q_n) in atom_iter.zip(q_trans.axis_iter_mut(Axis(0))) {
//         // Iteration over atomic orbitals mu on A.
//         for _ in 0..atom.n_orbs {
//             // Iteration over occupied orbital i.
//             for (orb_i, mut q_j) in sc_ij.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
//                 // Iteration over occupied orbital j.
//                 for (orb_j, mut q) in orbs_ij.row(mu).iter().zip(q_j.iter_mut()) {
//                     // The transition charge is computed.
//                     *q += orb_i * orb_j;
//                 }
//             }
//             mu += 1;
//         }
//     }
//
//     0.5 * q_trans
// }

/// Computes the Mulliken transition charges between two `Particle`s (hole or electron). They can
/// either be located on the same monomer or on two different ones. The overlap matrix has to be the
/// overlap between the basis functions of `a` and `b`.
pub fn q_pp<'a,'b>(a: &'a Particle<'b>, b: &'a Particle<'b>, s: ArrayView2<f64>) -> Array1<f64> {
    // Check if the transition charges are on the same monomer or not.
    let inter: bool = a.monomer != b.monomer;
    // Number of atoms.
    let n_atoms: usize = if inter {
        a.monomer.n_atoms + b.monomer.n_atoms
    } else {
        a.monomer.n_atoms
    };

    // Matrix product of overlap matrix with the orbital on I.
    let c_sc_i: Array1<f64> = &a.mo.c * &s.dot(&b.mo.c);
    // Matrix product of overlap matrix with the orbital on J.
    let c_sc_j: Array1<f64> = &b.mo.c * &s.t().dot(&a.mo.c);
    // Both coefficients are added.
    let c_sc: Array1<f64> = if inter {
        c_sc_i.into_iter().chain(c_sc_j.into_iter()).collect()
    } else {
        &c_sc_i + &c_sc_j
    };
    // Iterator over the atoms.
    let atom_iter = if inter {
        a.atoms.iter().chain(b.atoms)
    } else {
        let empty_slice: &[Atom] = &[];
        a.atoms.iter().chain(empty_slice.iter())
    };

    let mut mu: usize = 0;
    // Iteration over all atoms (A) on I and on J.
    let q_trans: Array1<f64> = atom_iter
        .map(|atom| {
            // Iteration over atomic orbitals mu on A.
            let new_mu: usize = mu + atom.n_orbs;
            // The transition charge is computed.
            let q: f64 = c_sc.slice(s![mu..new_mu]).sum();
            // mu is updated.
            mu = new_mu;
            q
        })
        .collect();

    0.5 * q_trans
}

/// Computes the Mulliken transition charges between a set of orbitals on one monomer
/// and a single orbital on another monomer.
pub fn q_le_p<'a>(
    a: &'a LocallyExcited<'a>,
    b: &'a Particle<'a>,
    s: ArrayView2<f64>,
    kind: ElecHole,
) -> Array2<f64> {
    // Check if the LE state and the particle are on the same monomer.
    let inter: bool = a.monomer != b.monomer;
    // Check if the occupied or virtual orbitals of the LE state are needed.
    let orbs: ArrayView2<f64> = match kind {
        ElecHole::Hole => a.occs,
        ElecHole::Electron => a.virts,
    };
    // Number of AOs and MOs in monomer I.
    let (n_orbs_i, dim): (usize, usize) = orbs.dim();
    // Number of AOs in monomer J.
    let n_orbs_j: usize = s.ncols();
    // Number of atoms.
    let n_atoms: usize = if inter {
        a.monomer.n_atoms + b.monomer.n_atoms
    } else {
        a.monomer.n_atoms
    };
    // The transition charges are initialized to zeros.
    let mut q_trans: Array2<f64> = Array2::zeros([n_atoms, dim]);

    // 1. (2nd term) Matrix-vector product of Overlap matrix S_mu_(on I)_nu_(on J) with C_nu_(on J)_b
    // 2. Element-wise product with C_mu_(on I)_i
    let c_sc_i: Array2<f64> = &orbs * &s.dot(&b.mo.c).broadcast((dim, n_orbs_i)).unwrap().t();
    // 1. Matrix product of Overlap matrix S_nu_(on J)_mu_(on_I) with C_mu_(on I)_i
    // 2. Element-wise product with C_nu_(on J)_b
    let c_sc_j: Array2<f64> = &s.t().dot(&orbs) * &b.mo.c.broadcast((dim, n_orbs_j)).unwrap().t();
    // Both products are added.
    let csc: Array2<f64> = if inter {
        concatenate![Axis(0), c_sc_i, c_sc_j]
    } else {
        &c_sc_i + &c_sc_j
    };
    // Iterator over the atoms.
    let atom_iter = if inter {
        a.atoms.iter().chain(b.atoms.iter())
    } else {
        let empty_slice: &[Atom] = &[];
        a.atoms.iter().chain(empty_slice)
    };

    let mut mu: usize = 0;
    // Iteration over all atoms in monomer I and J as well as the transition charges.
    for (atom, mut q) in atom_iter.zip(q_trans.axis_iter_mut(Axis(0))) {
        for _ in 0..atom.n_orbs {
            q += &csc.slice(s![mu, ..]);
            mu += 1;
        }
    }

    0.5 * q_trans
}

impl SuperSystem{
    pub fn q_lect<'a>(
        &self,
        a: &'a LocallyExcited<'a>,
        b: &ChargeTransferPair,
        kind: ElecHole,
    ) -> Array3<f64> {
        match kind{
            ElecHole::Hole =>{
                // Number of atoms.
                let n_atoms_le: usize = a.atoms.len();
                let n_atoms_ct_occ: usize = self.monomers[b.m_h].n_atoms;
                let n_atoms_occ: usize = n_atoms_le + n_atoms_ct_occ;

                // get the atoms of the hole
                let atoms_h:&[Atom] = &self.atoms[self.monomers[b.m_h].slice.atom_as_range()];

                // get occupied orbitals of the LE state
                // let occs_le: ArrayView2<f64> = a.occs;
                let occs_le = if self.config.lcmo.restrict_active_space{
                    let homo:usize = a.occs.dim().1;
                    let start:usize = homo- self.config.lcmo.active_space_le;
                    a.occs.slice(s![..,start..])
                }
                else{
                    a.occs
                };

                // get orbitals of the ct state
                let occ_indices: &[usize] = self.monomers[b.m_h].properties.occ_indices().unwrap();
                // The index of the HOMO (zero based).
                let homo: usize = occ_indices[occ_indices.len() - 1];
                // let occs_ct:ArrayView2<f64> = self.monomers[b.m_h].properties.orbs_slice(0, Some(homo + 1)).unwrap();
                let occs_ct = if self.config.lcmo.restrict_active_space{
                    self.monomers[b.m_h].properties
                        .orbs_slice((homo+1-self.config.lcmo.active_space_ct), Some(homo + 1)).unwrap()
                }else{
                    self.monomers[b.m_h].properties.orbs_slice(0, Some(homo + 1)).unwrap()
                };

                // slice the overlap matrix
                let s_ij:ArrayView2<f64> = self.properties.s_slice(a.monomer.slice.orb,self.monomers[b.m_h].slice.orb).unwrap();

                // Number of occupied molecular orbitals on I
                let dim_i: usize = occs_le.ncols();
                // Number of molecular orbitals on the hole
                let dim_j: usize = occs_ct.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ij: Array3<f64> = Array3::zeros([n_atoms_occ, dim_i, dim_j]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ij_c_ct_occ: Array2<f64> = s_ij.dot(&occs_ct);
                let s_ij_c_le_occ: Array2<f64> = s_ij.t().dot(&occs_le);

                let mut mu: usize = 0;
                for (atom, mut q_n) in a.atoms.iter().zip(q_trans_ij.slice_mut(s![0..n_atoms_le, .., ..]).axis_iter_mut(Axis(0))) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in occs_le.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                            for (sc, mut q) in s_ij_c_ct_occ.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_h.iter().zip(q_trans_ij.slice_mut(s![n_atoms_le.., .., ..]).axis_iter_mut(Axis(0))) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in s_ij_c_le_occ.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                            for (orb_j, mut q) in occs_ct.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ij
            }
            ElecHole::Electron =>{
                // Number of atoms.
                let n_atoms_le: usize = a.atoms.len();
                let n_atoms_ct_virt:usize = self.monomers[b.m_l].n_atoms;
                let n_atoms_virt: usize = n_atoms_le + n_atoms_ct_virt;

                // get the atoms of the hole
                let atoms_l:&[Atom] = &self.atoms[self.monomers[b.m_l].slice.atom_as_range()];

                // get virtual orbitals of the LE state
                // let virts_le: ArrayView2<f64> = a.virts;
                let virts_le = if self.config.lcmo.restrict_active_space{
                    a.virts.slice(s![..,..self.config.lcmo.active_space_le])
                }else{
                    a.virts
                };

                // get orbitals of the ct state
                let virt_indices: &[usize] = self.monomers[b.m_l].properties.virt_indices().unwrap();
                // The index of the LUMO (zero based).
                let lumo: usize = virt_indices[0];
                // let virts_ct:ArrayView2<f64> = self.monomers[b.m_l].properties.orbs_slice(lumo, None).unwrap();
                let virts_ct = if self.config.lcmo.restrict_active_space{
                    self.monomers[b.m_l].properties
                        .orbs_slice(lumo, Some(lumo+self.config.lcmo.active_space_ct)).unwrap()
                }else{
                    self.monomers[b.m_l].properties.orbs_slice(lumo, None).unwrap()
                };

                // slice the overlap matrix
                let s_ab:ArrayView2<f64> = self.properties
                    .s_slice(a.monomer.slice.orb,self.monomers[b.m_l].slice.orb).unwrap();

                // Number of virtual molecular orbitals on I
                let dim_a: usize = virts_le.ncols();
                // Number of virtual molecular orbitals on the electron
                let dim_b:usize = virts_ct.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ab: Array3<f64> = Array3::zeros([n_atoms_virt, dim_a, dim_b]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ab_c_ct_virt:Array2<f64> = s_ab.dot(&virts_ct);
                let s_ab_c_le_virt:Array2<f64> = s_ab.t().dot(&virts_le);

                let mut mu: usize = 0;
                for (atom, mut q_n) in a.atoms.iter().zip(q_trans_ab.slice_mut(s![0..n_atoms_le, .., ..]).axis_iter_mut(Axis(0))) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in virts_le.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                            for (sc, mut q) in s_ab_c_ct_virt.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_l.iter().zip(q_trans_ab.slice_mut(s![n_atoms_le.., .., ..]).axis_iter_mut(Axis(0))) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in s_ab_c_le_virt.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                            for (orb_j, mut q) in virts_ct.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ab
            }
        }
    }

    pub fn q_ctct<'a>(
        &self,
        a: &ChargeTransferPair,
        b: &ChargeTransferPair,
        kind: ElecHole,
    ) -> Array3<f64> {
        match kind{
            ElecHole::Hole =>{
                // Number of atoms.
                let n_atoms_ct_a: usize = self.monomers[a.m_h].n_atoms;
                let n_atoms_ct_b: usize = self.monomers[b.m_h].n_atoms;
                let n_atoms_occ: usize = n_atoms_ct_a+ n_atoms_ct_b;

                // get the atoms of the hole
                let atoms_a:&[Atom] = &self.atoms[self.monomers[a.m_h].slice.atom_as_range()];
                let atoms_b:&[Atom] = &self.atoms[self.monomers[b.m_h].slice.atom_as_range()];

                // get orbitals of the ct state
                let occ_indices_a: &[usize] = self.monomers[a.m_h].properties.occ_indices().unwrap();
                let occ_indices_b: &[usize] = self.monomers[b.m_h].properties.occ_indices().unwrap();
                // The index of the HOMO (zero based).
                let homo_a: usize = occ_indices_a[occ_indices_a.len() - 1];
                let homo_b: usize = occ_indices_b[occ_indices_b.len() - 1];
                // let occs_ct_a:ArrayView2<f64> = self.monomers[a.m_h].properties.orbs_slice(0, Some(homo_a + 1)).unwrap();
                // let occs_ct_b:ArrayView2<f64> = self.monomers[b.m_h].properties.orbs_slice(0, Some(homo_b + 1)).unwrap();
                let restrict_space:bool = self.config.lcmo.restrict_active_space;
                let active_space:usize = self.config.lcmo.active_space_ct;

                let occs_ct_a = if restrict_space{
                    self.monomers[a.m_h].properties.orbs_slice((homo_a-active_space+1), Some(homo_a + 1)).unwrap()
                }
                else{
                    self.monomers[a.m_h].properties.orbs_slice(0, Some(homo_a + 1)).unwrap()
                };
                let occs_ct_b = if restrict_space{
                    self.monomers[b.m_h].properties.orbs_slice((homo_b-active_space+1), Some(homo_b + 1)).unwrap()
                }else{
                    self.monomers[b.m_h].properties.orbs_slice(0, Some(homo_b + 1)).unwrap()
                };

                // slice the overlap matrix
                let s_ij:ArrayView2<f64> =
                    self.properties.s_slice(self.monomers[a.m_h].slice.orb,self.monomers[b.m_h].slice.orb).unwrap();

                // Number of occupied molecular orbitals on I
                let dim_i: usize = occs_ct_a.ncols();
                // Number of molecular orbitals on the hole
                let dim_j: usize = occs_ct_b.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ij: Array3<f64> = Array3::zeros([n_atoms_occ, dim_i, dim_j]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ij_c_ct_b: Array2<f64> = s_ij.dot(&occs_ct_b);
                let s_ij_c_ct_a: Array2<f64> = s_ij.t().dot(&occs_ct_a);

                let mut mu: usize = 0;
                for (atom, mut q_n) in atoms_a.iter().zip(q_trans_ij.slice_mut(s![0..n_atoms_ct_a, .., ..]).axis_iter_mut(Axis(0))) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in occs_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                            for (sc, mut q) in s_ij_c_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_b.iter().zip(q_trans_ij.slice_mut(s![n_atoms_ct_a.., .., ..]).axis_iter_mut(Axis(0))) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in s_ij_c_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                            for (orb_j, mut q) in occs_ct_a.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ij
            },
            ElecHole::Electron =>{
                // Number of atoms.
                let n_atoms_ct_a: usize =self.monomers[a.m_l].n_atoms;
                let n_atoms_ct_b:usize = self.monomers[b.m_l].n_atoms;
                let n_atoms_virt: usize = n_atoms_ct_a+ n_atoms_ct_b;

                // get the atoms of the hole
                let atoms_a:&[Atom] = &self.atoms[self.monomers[a.m_l].slice.atom_as_range()];
                let atoms_b:&[Atom] = &self.atoms[self.monomers[b.m_l].slice.atom_as_range()];

                // get orbitals of the ct state
                let virt_indices_a: &[usize] = self.monomers[a.m_l].properties.virt_indices().unwrap();
                let virt_indices_b: &[usize] = self.monomers[b.m_l].properties.virt_indices().unwrap();
                // The index of the LUMO (zero based).
                let lumo_a: usize = virt_indices_a[0];
                let lumo_b: usize = virt_indices_b[0];
                // let virts_ct_a:ArrayView2<f64> = self.monomers[a.m_l].properties.orbs_slice(lumo_a, None).unwrap();
                // let virts_ct_b:ArrayView2<f64> = self.monomers[b.m_l].properties.orbs_slice(lumo_b, None).unwrap();
                let restrict_space:bool = self.config.lcmo.restrict_active_space;
                let active_space:usize = self.config.lcmo.active_space_ct;

                let virts_ct_a = if restrict_space{
                    self.monomers[a.m_l].properties.orbs_slice(lumo_a, Some(lumo_a + active_space)).unwrap()
                }
                else{
                    self.monomers[a.m_l].properties.orbs_slice(lumo_a, None).unwrap()
                };
                let virts_ct_b = if restrict_space{
                    self.monomers[b.m_l].properties.orbs_slice(lumo_b, Some(lumo_b + active_space)).unwrap()
                }
                else{
                    self.monomers[b.m_l].properties.orbs_slice(lumo_b, None).unwrap()
                };

                // slice the overlap matrix
                let s_ab:ArrayView2<f64> =
                    self.properties.s_slice(self.monomers[a.m_l].slice.orb,self.monomers[b.m_l].slice.orb).unwrap();

                // Number of virtual molecular orbitals on I
                let dim_a: usize = virts_ct_a.ncols();
                // Number of virtual molecular orbitals on the electron
                let dim_b:usize = virts_ct_b.ncols();

                // The transition charges between the two sets of MOs are initialized.
                let mut q_trans_ab: Array3<f64> = Array3::zeros([n_atoms_virt, dim_a, dim_b]);

                // Matrix products between the overlap matrix and the molecular orbitals
                let s_ab_c_ct_b:Array2<f64> = s_ab.dot(&virts_ct_b);
                let s_ab_c_ct_a:Array2<f64> = s_ab.t().dot(&virts_ct_a);

                let mut mu: usize = 0;
                for (atom, mut q_n) in atoms_a.iter().zip(q_trans_ab.slice_mut(s![0..n_atoms_ct_a, .., ..]).axis_iter_mut(Axis(0))) {
                    for _ in 0..atom.n_orbs {
                        for (orb_i, mut q_i) in virts_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                            for (sc, mut q) in s_ab_c_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_i * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                mu = 0;

                for (atom, mut q_n) in atoms_b.iter().zip(q_trans_ab.slice_mut(s![n_atoms_ct_a.., .., ..]).axis_iter_mut(Axis(0))) {
                    for _ in 0..atom.n_orbs {
                        for (sc, mut q_i) in s_ab_c_ct_a.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                            for (orb_j, mut q) in virts_ct_b.row(mu).iter().zip(q_i.iter_mut()) {
                                *q += orb_j * sc;
                            }
                        }
                        mu += 1;
                    }
                }
                0.5 * q_trans_ab
            },
        }
    }
}
