#![allow(dead_code)]

mod constants;
mod defaults;
mod fermi_occupation;
mod gamma_approximation;
mod h0_and_s;
mod molecule;
mod parameters;
mod slako_transformations;
mod zbrent;
mod broyden;
mod calculator;
mod diis;
mod mulliken;
mod scc_routine;
mod scc_routine_unrestricted;
//mod scc_routine_unrestricted;

use crate::molecule::Molecule;
use ndarray::*;
use ndarray_linalg::*;
use std::env;
use std::ptr::eq;
use std::time::{Duration, Instant};

fn main() {
    println!(
        r#"   _________  ___  ________   ________  ________
  |\___   ___\\  \|\   ___  \|\   ____\|\   __  \
  \|___ \  \_\ \  \ \  \\ \  \ \  \___|\ \  \|\  \
       \ \  \ \ \  \ \  \\ \  \ \  \    \ \   _  _\
        \ \  \ \ \  \ \  \\ \  \ \  \____\ \  \\  \|
         \ \__\ \ \__\ \__\\ \__\ \_______\ \__\\ _\
          \|__|  \|__|\|__| \|__|\|_______|\|__|\|__| "#
    );
    println!("");
    println!("                       R. Mitric");
    println!("            Chair of theoretical Chemistry");
    println!("               University of Wuerzburg");
    println!("");
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2, "Please provide one xyz-filename");
    let filename = &args[1];
    let mol: Molecule = read_xyz(filename);
    println!("Start calculation");
    println!("_______________________________________________________");
    let now = Instant::now();
    //let energy: f64 = scc_routine::run_scc(&mol, None, None, None);
    println!("_______________________________________________________");
    println!("Time elapsed: {:.4} secs", now.elapsed().as_secs_f32());
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
