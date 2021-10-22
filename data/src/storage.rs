use ndarray::prelude::*;
use crate::data::*;
use crate::io::Configuration;
use crate::data::orbitals::OrbType;
use core::::{Atom, AtomSlice};

impl<'a> Storage<'a> {
    pub fn new(config: &Configuration, atoms: AtomSlice<'a>, params: Parametrization<'a>) -> Storage<'a> {
        // Calculate the number of electrons.
        let mut n_elec: usize = atoms.n_elec.iter().sum();
        if !config.fmo {
            n_elec -= config.mol.charge as usize;
        }

        // Calculate the number of atomic orbitals as the sum of the atomic orbitals.
        let n_orbs: usize = atoms.n_orbs().iter().sum();

        let orb_type: OrbType = match config.mol.multiplicity {
            1u8 => OrbType::Restricted,
            2u8 => OrbType::Unrestricted,
            3u8 => OrbType::Unrestricted,
            _ => panic!("The specified multiplicity is not implemented"),
        };

        if n_elec % 2 == 1 {
            assert_eq!(orb_type, OrbType::Unrestricted, "The molecule has an odd number of electrons,\
             so the multiplicity of 1 is invalid.")
        }

        // Construct the orbitals.
        let orbitals = SpatialOrbitals::new(n_orbs, n_elec, OrbType::Restricted);

        Storage {
            gammas: GammaData::new(),
            orbitals,
            charges: ChargeData::new(),
            energies: EnergyData::new(),
            gradients: GradData::new(),
            other: OtherData::new(),
            excited_states: ExcitedStateData::new(),
            parametrization: params,
        }
    }

    pub fn new_with_orbitals(params: Parametrization<'a>, orbitals: SpatialOrbitals) -> Self {
        Storage {
            gammas: GammaData::new(),
            orbitals,
            charges: ChargeData::new(),
            energies: EnergyData::new(),
            gradients: GradData::new(),
            other: OtherData::new(),
            excited_states: ExcitedStateData::new(),
            parametrization: params,
        }
    }


}


