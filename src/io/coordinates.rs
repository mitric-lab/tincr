use chemfiles::{Frame, Trajectory};
use ndarray::{Array2};
use crate::constants::{BOHR_TO_ANGS};

pub fn frame_to_coordinates(mut frame: Frame) -> (String, Vec<u8>, Array2<f64>) {
    let mut positions: Array2<f64> = Array2::from_shape_vec(
        (frame.size() as usize, 3),
        frame
            .positions()
            .iter()
            .flat_map(|array| array.iter())
            .cloned()
            .collect(),
    )
        .unwrap();
    // transform the coordinates from angstrom to bohr
    positions = positions / BOHR_TO_ANGS;
    // read the atomic number of each coordinate
    let atomic_numbers: Vec<u8> = (0..frame.size() as usize)
        .map(|i| frame.atom(i).atomic_number() as u8)
        .collect();

    let mut smiles_repr: Trajectory = Trajectory::memory_writer("SMI").unwrap();
    smiles_repr.write(&mut frame).unwrap();
    let smiles: String = smiles_repr.memory_buffer().unwrap().replace('~', "").replace('\n', "");
    return (smiles, atomic_numbers, positions);
}

pub fn read_file_to_frame(filename: &str) -> Frame{
    // read the geometry file
    let mut trajectory = Trajectory::open(filename, 'r').unwrap();
    let mut frame = Frame::new();
    // if multiple geometries are contained in the file, we will only use the first one
    trajectory.read(&mut frame).unwrap();
    return frame;
}