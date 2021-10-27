use crate::types::atomic_orbital::AtomicOrbital;
use crate::Element;
use nalgebra::Vector3;
use ndarray::prelude::*;
use soa_derive::StructOfArray;
use std::cmp::Ordering;
use std::ops::Sub;

use crate::utils::array_helper::argsort_usize;

/// `Atom` type that contains basic information about the chemical element as well as the
/// data used for the semi-empirical parameters that are used in the DFTB calculations.
/// The `StructofArray` macro automatically generates code that allows to replace Vec<`Atom`>
/// with a struct of arrays. It will generate the `AtomVec` that looks like:
///
/// ```rust
/// use tincr_core::Element;
/// pub struct AtomVec {
///     pub name: Vec<&'static str>,
///     pub number: Vec<u8>,
///     pub kind: Vec<Element>,
///     pub hubbard: Vec<f64>,
///     pub valorbs: Vec<Vec<AtomicOrbital>>,
///     pub n_orbs: Vec<usize>,
///     pub valorbs_occupation: Vec<Vec<f64>>,
///     pub n_elec: Vec<usize>,
/// }
/// ```
/// It will also generate the same functions that a `Vec<Atom>` would have, and a few helper structs:
/// `AtomSlice`, `AtomSliceMut`, `AtomRef` and `AtomRefMut` corresponding respectively
/// to `&[Atom]`, `&mut [Atom]`, `&Atom` and `&mut Atom`.
#[derive(StructOfArray, Clone)]
pub struct Atom {
    /// Name of the chemical element
    pub name: &'static str,
    /// Ordinary number of the element
    pub number: u8,
    /// Element as an enum
    pub kind: Element,
    // /// Mass of the atom in atomic units
    // pub mass: f64,
    /// Hubbard parameter
    pub hubbard: f64,
    /// Vector of the valence orbitals for this atom
    pub valorbs: Vec<AtomicOrbital>,
    /// Number of valence orbitals. This is the length of valorbs
    pub n_orbs: usize,
    /// Occupation number for each valence orbitals
    pub valorbs_occupation: Vec<f64>,
    /// Number of valence electrons
    pub n_elec: usize,
    /// Position of the atom in bohr
    pub xyz: Vector3<f64>,
    pub spin_coupling: f64,
}
//
// impl From<Element> for Atom {
//     /// Create a new [Atom] from the chemical [Element](crate::initialization::elements::Element).
//     /// The parameterization from the parameter files is loaded and the Hubbard parameter
//     /// and the valence orbitals are stored in this type.
//     fn from (element: Element) -> Self {
//         let symbol: &'static str = element.symbol();
//         let confined_atom: PseudoAtom = PseudoAtom::confined_atom(symbol);
//         let free_atom: PseudoAtom = PseudoAtom::free_atom(symbol);
//         let mut valorbs: Vec<AtomicOrbital> = Vec::new();
//         let mut occupation: Vec<f64> = Vec::new();
//         let mut n_elec: usize = 0;
//         for (i, j) in confined_atom.valence_orbitals.iter().zip(free_atom.valence_orbitals.iter()) {
//             let n: i8 = confined_atom.nshell[*i as usize];
//             let l: i8 = confined_atom.angular_momenta[*i as usize];
//             let energy: f64 = free_atom.energies[*j as usize];
//             for m in l.neg()..(l + 1) {
//                 valorbs.push(AtomicOrbital::from(((n - 1, l, m), energy)));
//                 occupation.push(confined_atom.orbital_occupation[*i as usize] as f64 / (2 * l + 1) as f64);
//             }
//             n_elec += confined_atom.orbital_occupation[*i as usize] as usize;
//         }
//         let n_orbs: usize = valorbs.len();
//         let spin_coupling:f64 = constants::SPIN_COUPLING[&element.number()];
//
//         Atom {
//             name: symbol,
//             number: element.number(),
//             kind: element,
//             hubbard: confined_atom.hubbard_u,
//             valorbs: valorbs,
//             n_orbs: n_orbs,
//             valorbs_occupation: occupation,
//             n_elec: n_elec,
//             xyz: Vector3::<f64>::zeros(),
//             spin_coupling:spin_coupling,
//         }
//     }
// }

// impl From<(Element, &SkfHandler)> for Atom {
//     /// Create a new [Atom] from the chemical [Element](crate::initialization::elements::Element) and
//     /// the [SkfHandler](crate::initialization::parameters::SkfHandler).
//     /// The parameterization from the parameter files is loaded and the Hubbard parameter
//     /// and the valence orbitals are stored in this type.
//     fn from (tuple:(Element,&SkfHandler)) -> Self {
//         let element:Element = tuple.0;
//         let symbol: &'static str = element.symbol();
//         let pseudo_atom:PseudoAtomMio = PseudoAtomMio::from(tuple.1);
//         let mut valorbs: Vec<AtomicOrbital> = Vec::new();
//         let mut occupation: Vec<f64> = Vec::new();
//         let mut n_elec: usize = 0;
//         for (i, j) in pseudo_atom.valence_orbitals.iter().zip(pseudo_atom.valence_orbitals.iter()) {
//             let n: i8 = pseudo_atom.nshell[*i as usize];
//             let l: i8 = pseudo_atom.angular_momenta[*i as usize];
//             let energy: f64 = pseudo_atom.energies[*j as usize];
//             for m in l.neg()..(l + 1) {
//                 valorbs.push(AtomicOrbital::from(((n - 1, l, m), energy)));
//                 occupation.push(pseudo_atom.orbital_occupation[*i as usize] as f64 / (2 * l + 1) as f64);
//             }
//             n_elec += pseudo_atom.orbital_occupation[*i as usize] as usize;
//         }
//         let n_orbs: usize = valorbs.len();
//         let spin_coupling:f64 = constants::SPIN_COUPLING[&element.number()];
//
//         Atom {
//             name: symbol,
//             number: element.number(),
//             kind: element,
//             hubbard: pseudo_atom.hubbard_u,
//             valorbs: valorbs,
//             n_orbs: n_orbs,
//             valorbs_occupation: occupation,
//             n_elec: n_elec,
//             xyz: Vector3::<f64>::zeros(),
//             spin_coupling:spin_coupling,
//         }
//     }
// }

impl<'a> AtomRef<'a> {
    pub fn sort_indices_atomic_orbitals(&self) -> Vec<usize> {
        let indices: Array1<usize> = self.valorbs.iter().map(|orb| orb.ord_idx()).collect();
        argsort_usize(indices.view())
    }
}

// impl From<&str> for Atom {
//     /// Create a new [Atom] from the atomic symbol (case insensitive). The parameterization from the
//     /// parameter files is loaded and the Hubbard parameter and the valence orbitals are stored in
//     /// this type.
//     fn from(symbol: &str) -> Self {
//         Self::from(Element::from(symbol))
//     }
// }
//
// impl From<&Atom> for Atom {
//     // Create a new [Atom] from a reference to an [Atom].
//     fn from(atom: &Atom) -> Self {
//         atom.clone()
//     }
// }
//
// impl From<u8> for Atom {
//     /// Create a new [Atom] from the atomic number. The parameterization from the
//     /// parameter files is loaded and the Hubbard parameter and the valence orbitals are stored in
//     /// this type.
//     fn from(number: u8) -> Self {
//         Self::from(Element::from(number))
//     }
// }
impl PartialEq for Atom {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl Sub for Atom {
    type Output = Vector3<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.xyz - rhs.xyz
    }
}

impl Sub for &Atom {
    type Output = Vector3<f64>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.xyz - rhs.xyz
    }
}

impl Eq for Atom {}

impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}
//
// /// The initial atoms are created from the atomic numbers and the coordinates. The unique atoms
// /// and all atoms are returned.
// pub fn create_atoms(at_num: &[u8], coords: ArrayView2<f64>) -> (AtomVec, AtomVec) {
//     // Unique [Atom]s and the HashMap with the mapping from the numbers to the [Atom]s are
//     // constructed
//     let (u_atoms, n_to_at): (AtomVec, HashMap<u8, Atom>) = get_unique_atoms(&at_num);
//
//     // Create the list of all atoms.
//     let mut atoms: AtomVec = AtomVec::with_capacity(at_num.len());
//     at_num.iter().for_each(|num| atoms.push((*n_to_at.get(num).unwrap()).clone()));
//
//     // Positions are set for each atom.
//     coords.outer_iter().zip(atoms.xyz.iter_mut())
//         .for_each(|(pos, xyz)| {
//             *xyz = Vector3::from_iterator(pos.iter().cloned())
//         });
//
//     // (unique atoms, all atoms)
//     (u_atoms, atoms)
// }
//
//
// /// Finds the unique elements in a large list of elements/atoms that are specified by their atomic
// /// numbers. For each of these unique elements a [Atom] is created and stored in a Vec<Atom>.
// /// Furthermore, a HashMap<u8, Atom> is created that links an atomic number to an [Atom] so that
// /// it can be cloned for every atom in the molecule.
// pub fn get_unique_atoms(atomic_numbers: &[u8]) -> (AtomVec, HashMap<u8, Atom>) {
//     let mut unique_numbers: Vec<u8> = atomic_numbers.to_owned();
//     // Sort of atomic numbers
//     unique_numbers.sort_unstable();
//     // Delete duplicates
//     unique_numbers.dedup();
//     // Create the unique Atoms
//     let unique_atoms: AtomVec = unique_numbers
//         .iter()
//         .map(|number| Atom::from(*number))
//         .collect();
//     let mut num_to_atom: HashMap<u8, Atom> = HashMap::with_capacity(unique_numbers.len());
//     // insert the atomic numbers and the reference to atoms in the HashMap
//     for (num, atom) in unique_numbers
//         .into_iter()
//         .zip(unique_atoms.iter())
//     {
//         num_to_atom.insert(num, atom);
//     }
//     return (unique_atoms, num_to_atom);
// }