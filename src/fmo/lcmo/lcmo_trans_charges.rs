use crate::fmo::{BasisState, ChargeTransfer, LocallyExcited, Monomer, Particle};
use crate::initialization::Atom;
use ndarray::concatenate;
use ndarray::prelude::*;
use std::iter;
use std::io::empty;
use std::ops::AddAssign;

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
    // Check if the transition charges are on the same monomer or not.
    let inter: bool = a.monomer != b.monomer;
    // Number of atoms.
    let n_atoms: usize = if inter {
        a.monomer.n_atoms + b.monomer.n_atoms
    } else {
        a.monomer.n_atoms
    };
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
    let sc_i: Array2<f64> = s.dot(&orbs_j);
    // Matrix product of overlap matrix with the orbitals on J.
    let sc_j: Array2<f64> = s.t().dot(&orbs_i);
    // Either append or sum the contributions, depending whether the orbitals are on the same monomer.
    let csc_i: Array2<f64> = if inter {
        concatenate![Axis(0), orbs_i, sc_i]
    } else {
        &orbs_i + &sc_i
    };
    let csc_j: Array2<f64> = if inter {
        concatenate![Axis(0), orbs_j, sc_j]
    } else {
        &orbs_j + &sc_j
    };

    // Iterator over the atoms.
    let atom_iter = if inter {
        a.atoms.iter().chain(b.atoms.iter())
    } else {
        let empty_slice: &[Atom] = &[];
        a.atoms.iter().chain(empty_slice.iter())
    };

    let mut mu: usize = 0;
    // Iteration over all atoms (A).
    for (n, atom) in atom_iter.enumerate() {
        // Iteration over atomic orbitals mu on A.
        for _ in 0..atom.n_orbs {
            // Iteration over occupied orbital i.
            for (i, orb_i) in csc_i.row(mu).iter().enumerate() {
                // Iteration over occupied orbital j.
                for (j, orb_j) in csc_j.row(mu).iter().enumerate() {
                    // The transition charge is computed.
                    q_trans[[n, i, j]] += orb_i * orb_j;
                }
            }
            mu += 1;
        }
    }

    0.5 * q_trans
}

/// Computes the Mulliken transition charges between two `Particle`s (hole or electron). They can
/// either be located on the same monomer or on two different ones. The overlap matrix has to be the
/// overlap between the basis functions of `a` and `b`.
pub fn q_pp<'a>(a: &'a Particle<'a>, b: &'a Particle<'a>, s: ArrayView2<f64>) -> Array1<f64> {
    // Check if the transition charges are on the same monomer or not.
    let inter: bool = a.monomer != b.monomer;
    // Number of atoms.
    let n_atoms: usize = if inter {
        a.monomer.n_atoms + b.monomer.n_atoms
    } else {
        a.monomer.n_atoms
    };

    // Matrix product of overlap matrix with the orbital on I.
    let c_sc_i: Array1<f64> = &a.orb * &s.dot(&b.orb);
    // Matrix product of overlap matrix with the orbital on J.
    let c_sc_j: Array1<f64> = &b.orb * &s.t().dot(&a.orb);
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
    let c_sc_i: Array2<f64> = &orbs * &s.dot(&b.orb).broadcast((dim, n_orbs_i)).unwrap().t();
    // 1. Matrix product of Overlap matrix S_nu_(on J)_mu_(on_I) with C_mu_(on I)_i
    // 2. Element-wise product with C_nu_(on J)_b
    let c_sc_j: Array2<f64> = &s.t().dot(&orbs) * &b.orb.broadcast((dim, n_orbs_j)).unwrap().t();
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
