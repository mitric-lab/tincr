mod constants;
//mod SlakoTransformations;
//mod spline;
//mod slako_transformations_lmax2;
mod parameters;
mod molecule;
mod h0_and_s;
mod gamma_approximation;
mod slako_transformations;
//mod scc_routine;
mod defaults;
mod scc_routine;

use ndarray::*;
use ndarray_linalg::*;
use std::ptr::eq;
use peroxide::fuga::*;


fn main() {
    println!("Hello, world!");
    //let atom = param::slaterkoster::free_pseudo_atom::c;
    let x:Vec<f64> = vec![0.9, 1.3, 1.9, 2.1];
    let y: Vec<f64> = vec![1.3, 1.5, 1.85, 2.1];

    let s:CubicSpline = CubicSpline::from_nodes(x, y);

    for i in 0 .. 4 {
        println!("{}", s.eval(i as f64 / 2.0));
    }
}

struct Atom {
    number: u64,
    coords: [f64; 3],
    charge: f64,
    norb: u8,
    symbol: char,
    valorbs: Vec<usize>,
}

impl Atom {
    fn new(number: u64, coords: [f64; 3]) -> Atom {
        let atom = Atom {
            number,
            coords,
            charge: 0.0,
            norb: 0,
            symbol: 'X',
            valorbs: vec![0, 0, 0],
        };
        return atom;
    }
}





fn read_xyz(path: &str) -> Vec<Atom> {
    let mut trajectory = chemfiles::Trajectory::open(path, 'r').unwrap();
    let mut frame = chemfiles::Frame::new();
    trajectory.read(&mut frame).unwrap();
    let natom: u64 = frame.size();
    let atomcoords = frame.positions();
    let atomnos: Vec<u64> = (0..natom)
        .map(|i: u64| frame.atom(i).atomic_number())
        .collect();
    let mut molecule = Vec::new();
    for i in 0..natom {
        molecule.push(Atom::new(5, [0.0, 0.0, 0.0]));
    }
    return molecule;
}
