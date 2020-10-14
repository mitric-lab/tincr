mod constants;
//mod SlakoTransformations;
//mod spline;
//mod slako_transformations_lmax2;
mod parameters;

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

fn h0_and_s_ab(
    dim_a: usize,
    dim_b: usize,
    atomlist_a: Vec<Atom>,
    atomlist_b: Vec<Atom>,
    a_is_b: bool,
) -> (Array<f64, Ix2>, Array<f64, Ix2>) {
    /*
    compute Hamiltonian and overlap matrix elements between two sets of atoms. If the sets
    A and B contain exactly the same structure AisB should be set to True to ensure that
    the diagonal elements of the Hamiltonian are replaced by the correct on-site energies.

    Parameters:
    ===========
    dim_a: number of orbitals of all the atoms in set A
    dim_b:  ''                                 in set B
    atomlist_a, atomlist_b: list of (Zi,(xi,yi,zi)) for each atom
    */
    let mut h0: Array<f64, Ix2> = Array::zeros((dim_a, dim_b));
    let mut s: Array<f64, Ix2> = Array::zeros((dim_a, dim_b));
    // iterate over atoms
    let mu = 0;
    for (i, atom_a) in atomlist_a.iter().enumerate() {
        // iterate over orbitals on center i
        for _ in atom_a.valorbs.iter() {
            // iterate over atoms
            let nu = 0;
            for (j, atom_b) in atomlist_b.iter().enumerate() {
                // iterate over orbitals on center j
                for _ in atom_b.valorbs.iter() {
                    if mu == nu && a_is_b == true {
                        assert_eq!(atom_a.number, atom_b.number);
                        // use the true single particle orbitals energies
                        h0[[mu, nu]] = 0.0; //orbital_energies[Zi][(ni,li)];
                                            // orbitals are normalized to 1
                        s[[mu, nu]] = 1.0;
                    } else {
                        // initialize matrix elements of S and H0
                        let mut s_mu_nu: f64 = 0.0;
                        let mut h0_mu_nu: f64 = 0.0;
                        if atom_a.number <= atom_b.number {
                            // the first atom given to getHamiltonian() or getOverlap()
                            // has to be always the one with lower atomic number
                            if i == j && a_is_b == true {
                                assert!(mu != nu);
                                s_mu_nu = 0.0;
                                h0_mu_nu = 0.0;
                            } else {
                                // let s = SKT[(Zi,Zj)].getOverlap(li,mi,posi, lj,mj,posj);
                                // let h0 = SKT[(Zi,Zj)].getHamiltonian0(li,mi,posi, lj,mj,posj);
                                s_mu_nu = 0.0;
                                h0_mu_nu = 0.0;
                            }
                        } else {
                            // swap atoms if Zj > Zi
                            // let s  = SKT[(Zj,Zi)].getOverlap(lj,mj,posj, li,mi,posi);
                            // let h0 = SKT[(Zj,Zi)].getHamiltonian0(lj,mj,posj, li,mi,posi);
                            s_mu_nu = 0.0;
                            h0_mu_nu = 0.0;
                        }
                        h0[[mu, nu]] = h0_mu_nu;
                        s[[mu, nu]] = s_mu_nu;
                    }
                    let nu = nu + 1;
                }
            }
            let mu = mu + 1;
        }
    }
    return (s, h0);
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
