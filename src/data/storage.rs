use ndarray::prelude::*;
use crate::data::*;


impl<'a> Storage<'a> {
    pub fn new() -> Storage<'a> {
        Storage {
            gammas: GammaData::new(),
            orbitals: OrbitalData::new(),
            charges: ChargeData::new(),
            energies: EnergyData::new(),
            gradients: GradData::new(),
            other: OtherData::new(),
            excited_states: ExcitedStateData::new(),
            parametrization: ParamData::new(),
        }
    }

    /// Clear all data without any exceptions.
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}
