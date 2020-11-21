mod constants;
mod parameters;
mod molecule;
mod h0_and_s;
mod gamma_approximation;
mod slako_transformations;
mod defaults;
mod zbrent;
mod fermi_occupation;
mod scc_routine;
mod diis;
mod mulliken;

use ndarray::*;
use ndarray_linalg::*;
use std::ptr::eq;
use peroxide::fuga::*;
use crate::molecule::Molecule;
use std::env;

fn main() {
    println!("Hello, world!");
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2, "Please provide one xyz-filename");
    println!("Get filename");
    let filename = &args[1];
    println!("Start read-xyz");
    let mol: Molecule = read_xyz(filename);
    println!("Start calculation");
    let energy: f64 = scc_routine::run_scc(&mol, None, None, None);
}


fn read_xyz(path: &str) -> Molecule {
    let mut trajectory = chemfiles::Trajectory::open(path, 'r').unwrap();
    let mut frame = chemfiles::Frame::new();
    trajectory.read(&mut frame).unwrap();
    let natom: usize = frame.size() as usize;
    let mut pos: Vec<f64> = Vec::new();
    for atom in frame.positions().to_vec().iter() {
        for coord in atom.iter() {
            pos.push(*coord);
        }
    }
    println!("natoms {}, pos len {}", natom, pos.len());
    let mut positions: Array2<f64> = Array::from_shape_vec((natom, 3), pos).unwrap();
    let atomnos: Vec<u8> = (0..natom)
        .map(|i| frame.atom(i as u64).atomic_number() as u8)
        .collect();
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomnos, positions, charge, multiplicity);
    return mol;
}
