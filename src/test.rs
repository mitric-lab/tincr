use crate::io::GeneralConfig;
use crate::molecule::Molecule;
use ndarray::prelude::*;
use ndarray::Array2;

pub fn get_water_molecule() -> Molecule {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None
    );
    mol
}

pub fn get_benzene_molecule() -> Molecule {
    let atomic_numbers: Vec<u8> = vec![1, 6, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1];
    let mut positions: Array2<f64> = array![
        [1.2194, -0.1652, 2.1600],
        [0.6825, -0.0924, 1.2087],
        [-0.7075, -0.0352, 1.1973],
        [-1.2644, -0.0630, 2.1393],
        [-1.3898, 0.0572, -0.0114],
        [-2.4836, 0.1021, -0.0204],
        [-0.6824, 0.0925, -1.2088],
        [-1.2194, 0.1652, -2.1599],
        [0.7075, 0.0352, -1.1973],
        [1.2641, 0.0628, -2.1395],
        [1.3899, -0.0572, 0.0114],
        [2.4836, -0.1022, 0.0205]
    ];

    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None
    );
    mol
}

pub fn get_ethene_molecule() -> Molecule {
    let atomic_numbers: Vec<u8> = vec![6, 6, 1, 1, 1, 1];
    let mut positions: Array2<f64> = array![
        [-0.75758, 0.00000, -0.00000],
        [0.75758, 0.00000, 0.00000],
        [-1.28092, 0.97850, -0.00000],
        [-1.28092, -0.97850, 0.00000],
        [1.28092, -0.97850, -0.00000],
        [1.28092, 0.97850, 0.00000]
    ];
    // transform coordinates in au
    positions = positions * 1.8897261278504418;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions.clone(),
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None
    );
    mol
}
