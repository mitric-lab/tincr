mod constants;
//mod SlakoTransformations;
//mod spline;
//mod slako_transformations_lmax2;
mod parameters;
mod molecule;

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

fn distance_matrix(
    coordinates: Array<f64, Ix2>,
) -> (Array<f64, Ix2>, Array<f64, Ix3>, Array<usize, Ix2>) {
    let cutoff = 2.0;
    let n_atoms: usize = coordinates.cols();
    let mut dist_matrix: Array<f64, Ix2> = Array::zeros((n_atoms, n_atoms));
    let mut directions_matrix: Array<f64, Ix3> = Array::zeros((n_atoms, n_atoms, 3));
    let mut prox_matrix: Array<usize, Ix2> = Array::zeros((n_atoms, n_atoms));
    for (i, pos_i) in coordinates.outer_iter().enumerate() {
        for (j, pos_j) in coordinates.slice(s![i.., ..]).outer_iter().enumerate() {
            let r = &pos_i - &pos_j;
            let r_ij = r.norm();
            dist_matrix[[i, j]] = r_ij;
            //directions_matrix[[i, j]] = &r/&r_ij;
            if r_ij <= cutoff { prox_matrix[[i, j]] = 1; }
        }
    }
    let dist_matrix = &dist_matrix + &dist_matrix.t();
    let prox_matrix = &prox_matrix + &prox_matrix.t();
    //let directions_matrix = directions_matrix - directions_matrix.t();
    return (dist_matrix, directions_matrix, prox_matrix);
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
