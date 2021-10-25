use crate::parametrization::pseudo_atom::{PseudoAtomHomegrown, PseudoAtomSKF};

pub trait PseudoAtom {
    fn z(&self) -> u8;
    fn hubbard_u(&self) -> f64;
    fn n_elec(&self) -> u8;
    fn energies(&self) -> &[f64];
    fn angular_momenta(&self) -> &[i8];
    fn valence_orbitals(&self) -> &[u8];
    fn nshell(&self) -> &[i8];
    fn orbital_occupation(&self) -> &[i8];
}

impl PseudoAtom for PseudoAtomHomegrown {
    fn z(&self) -> u8 {
        self.z
    }

    fn hubbard_u(&self) -> f64 {
        self.hubbard_u
    }

    fn n_elec(&self) -> u8 {
        self.n_elec
    }

    fn energies(&self) -> &[f64] {
        &self.energies
    }

    fn angular_momenta(&self) -> &[i8] {
        &self.angular_momenta
    }

    fn valence_orbitals(&self) -> &[u8] {
        &self.valence_orbitals
    }

    fn nshell(&self) -> &[i8] {
        &self.nshell
    }

    fn orbital_occupation(&self) -> &[i8] {
        &self.orbital_occupation
    }
}

impl PseudoAtom for PseudoAtomSKF {
    fn z(&self) -> u8 {
        self.z
    }

    fn hubbard_u(&self) -> f64 {
        self.hubbard_u
    }

    fn n_elec(&self) -> u8 {
        self.n_elec
    }

    fn energies(&self) -> &[f64] {
        &self.energies
    }

    fn angular_momenta(&self) -> &[i8] {
        &self.angular_momenta
    }

    fn valence_orbitals(&self) -> &[u8] {
        &self.valence_orbitals
    }

    fn nshell(&self) -> &[i8] {
        &self.nshell
    }

    fn orbital_occupation(&self) -> &[i8] {
        &self.orbital_occupation
    }
}
