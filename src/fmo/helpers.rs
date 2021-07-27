use crate::fmo::SuperSystem;
use crate::initialization::Atom;
use ndarray::Slice;
use std::ops::Range;


/// The atoms in the pair are two blocks in the Vec<Atom>, but these two blocks are in general not
/// connected. Since non-contiguous views of data structures are not trivial, the blocks are just
/// sliced individually and then appended afterwards.
pub fn get_pair_slice(atoms: &[Atom], mi_range: Range<usize>, mj_range: Range<usize>) -> Vec<Atom> {
    atoms[mi_range]
        .iter()
        .map(|x| x.clone())
        .chain(atoms[mj_range].iter().map(|x| x.clone()))
        .collect()
}

/// Type that holds different Slices that are frequently used for indexing of molecular subunits
pub struct MolecularSlice {
    /// [Slice](ndarray::prelude::Slice) for the atoms corresponding to the molecular unit
    pub atom: Slice,
    /// Similar to the atom slice, but as an Range. In contrast to the Slice the Range does not
    /// implement the Copy trait
    atom_range: Range<usize>,
    /// Gradient slice, this is the atom slice multiplied by the factor 3
    pub grad: Slice,
    /// [Slice](ndarray::prelude::Slice) for the orbitals corresponding to this molecular unit
    pub orb: Slice,
}

impl MolecularSlice {
    pub fn new(at_index: usize, n_atoms: usize, orb_index: usize, n_orbs: usize) -> Self {
        MolecularSlice {
            atom: Slice::from(at_index..(at_index + n_atoms)),
            atom_range: at_index..(at_index + n_atoms),
            grad: Slice::from((at_index * 3)..(at_index + n_atoms) * 3),
            orb: Slice::from(orb_index..(orb_index + n_orbs)),
        }
    }

    /// Return the range of the atoms corresponding to this molecular unit
    pub fn atom_as_range(&self) -> Range<usize> {
        // since Range does not implement Copy trait, it need to be cloned every time it gets called
        self.atom_range.clone()
    }
}