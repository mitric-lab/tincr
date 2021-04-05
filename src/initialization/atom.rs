use crate::initialization::parameters::{
    get_confined_pseudo_atom, get_free_pseudo_atom, PseudoAtom,
};
use crate::param::elements::Element;
use std::collections::HashMap;
use std::ops::Neg;

/// `Atom` type that contains basic information about the chemical element as well as the
/// data used for the semiempirical parameters that are used in the DFTB calculations.
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
    // Number of valence electrons
    pub n_elec: usize,
}

impl Atom {
    /// Create a new `Atom` from the atomic symbol (case insensitive). The parameterization from the
    /// parameter files is loaded and the Hubbard parameter and the valence orbitals are stored in
    /// this type.
    pub fn new(symbol: &str) -> Self {
        let element: Element = Element::from(symbol);
        let symbol: &'static str = element.symbol();
        let confined_atom: PseudoAtom = PseudoAtom::confined_atom(symbol);
        let free_atom: PseudoAtom = PseudoAtom::free_atom(symbol);
        let mut valorbs: Vec<AtomicOrbital> = Vec::new();
        let mut occupation: Vec<f64> = Vec::new();
        let mut n_elec: usize = 0;
        for i in confined_atom.valence_orbitals {
            let n: i8 = confined_atom.nshell[i as usize];
            let l: i8 = confined_atom.angular_momenta[i as usize];
            let energy: f64 = free_atom.energies[i as usize];
            for m in l.neg()..(l + 1) {
                valorbs.push(AtomicOrbital::from(((n - 1, l, m), energy)));
                occupation.push(atom.orbital_occupation[i as usize] as f64 / (2 * l + 1) as f64);
            }
            n_elec += confined_atom.orbital_occupation[i as usize];
        }
        Atom {
            name: symbol,
            number: element.number(),
            kind: element,
            hubbard: confined_atom.hubbard_u,
            valorbs: valorbs,
            n_orbs: valorbs.len(),
            valorbs_occupation: occupation,
            n_elec: n_elec,
        }
    }
}

/// Type that specifies an atomic orbital by its three quantum numbers and holds its energy
#[derive(Eq)]
pub struct AtomicOrbital {
    pub n: i8,
    pub m: i8,
    pub l: i8,
    pub energy: f64,
}

impl From<(i8, i8, i8)> for AtomicOrbital {
    fn from(qnumber: (i8, i8, i8)) -> Self {
        Self {
            n: qnumber.0,
            m: qnumber.1,
            l: qnumber.2,
            energy: 0.0,
        }
    }
}

impl From<((i8, i8, i8), f64)> for AtomicOrbital {
    fn from(numbers_energy: ((i8, i8, i8), f64)) -> Self {
        Self {
            n: numbers_energy.0 .0,
            m: numbers_energy.0 .1,
            l: numbers_energy.0 .2,
            energy: numbers_energy.1,
        }
    }
}
