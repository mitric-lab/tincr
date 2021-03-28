use crate::initialization::get_fragments;
use crate::initialization::molecule::Molecule;
use crate::initialization::parametrization::Parametrization;
use crate::io::read_file_to_frame;
use crate::io::settings::GeneralConfig;
use chemfiles::Frame;
use std::collections::HashMap;

pub struct Engine {
    pub config: GeneralConfig,
    pub molecules: Vec<Molecule>,
    pub parameters: HashMap<String, Parametrization>,
}

impl Engine {
    pub fn from_geometry_file(filename: &str) -> Engine {
        let settings: GeneralConfig = GeneralConfig::new();
        let main_frame: Frame = read_file_to_frame(filename);

        let molecules: Vec<Molecule> = if settings.fmo {
            let frames: Vec<Frame> = get_fragments(main_frame);
            let mut mols: Vec<Molecule> = Vec::new();
            for frame in frames.into_iter() {
                mols.push(Molecule::from_frame(frame));
            }
            mols
        } else {
            vec![Molecule::from_frame(main_frame)]
        };

        let mut parameters: HashMap<String, Parametrization> = HashMap::new();
        for mol in molecules.iter() {
            let smiles: String = mol.repr.unwrap().clone();
            if !parameters.contains_key(&smiles) {
                parameters.insert(
                    smiles,
                    Parametrization::new(&mol.atomic_numbers.unwrap(), None),
                );
            }
        }
        Engine{config: settings, molecules: molecules, parameters: parameters}
    }
}
