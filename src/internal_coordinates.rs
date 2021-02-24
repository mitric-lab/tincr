use crate::constants;
use crate::defaults;
use crate::gradients;
use crate::Molecule;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use petgraph::algo::*;
use petgraph::data::*;
use petgraph::dot::{Config, Dot};
use petgraph::graph::*;
use petgraph::stable_graph::*;
use crate::molecule::distance_matrix;
use std::f64::consts::PI;

pub fn cartesian_to_internal(coord_vector: &Array1<f64>, mol: &Molecule) {
    let masses: Array1<f64> = build_masses(mol);
    let x_vec: Array1<f64> = shift_to_com(coord_vector, &masses);
}

pub fn shift_to_com(coord_vector: &Array1<f64>, masses: &Array1<f64>) -> (Array1<f64>) {
    // shift center of mass to the origin
    let com: Array1<f64> = Array::from_vec(center_of_mass(coord_vector, masses));
    let mut pos_shifted: Array1<f64> = Array::zeros(coord_vector.len());
    let n_at: usize = coord_vector.len() / 3;

    for i in 0..n_at {
        pos_shifted
            .slice_mut(s![3 * i..3 * i + 3])
            .assign(&(coord_vector.slice(s![3 * i..3 * i + 3]).to_owned() - &com));
    }
    return pos_shifted;
}

pub fn center_of_mass(coord_vector: &Array1<f64>, masses: &Array1<f64>) -> (Vec<f64>) {
    // find the center of mass
    let xm: Array1<f64> = coord_vector * masses;
    let mut com: Vec<f64> = Vec::new();
    com.push(xm.slice(s![0..;3]).sum() / masses.slice(s![0..;3]).sum());
    com.push(xm.slice(s![1..;3]).sum() / masses.slice(s![0..;3]).sum());
    com.push(xm.slice(s![2..;3]).sum() / masses.slice(s![0..;3]).sum());

    return com;
}

pub fn build_masses(mol: &Molecule) -> (Array1<f64>) {
    let mut masses: Vec<f64> = Array::zeros(mol.n_atoms * 3).to_vec();
    for n in mol.atomic_numbers.iter() {
        masses.push(constants::ATOMIC_MASSES[n]);
        masses.push(constants::ATOMIC_MASSES[n]);
        masses.push(constants::ATOMIC_MASSES[n]);
    }
    return Array::from_vec(masses);
}

pub fn internal_to_cartesian(coord_vector: &Array1<f64>) {
    // transform internal coordinates back to cartesians.
    //
    //     Since the internal coordinates are curvilinear the transformation
    // has to be done iteratively and depends on having a closeby point q0
    // for which we know the cartesian coordinates x0. If the displacement
    // dq = q-q0
    // is too large, the iteration will not converge.
    //     Given the initial point
    // x0 ~ q0
    // we wish to find the cartesian coordinate x that corresponds to q
    // x ~ q      q = q0 + dq
}

pub fn wrap_angles() {
    // Bending angles and dihedral angles have to be in the
    // range [0,pi], while inversion angles have to be in the range [-pi/2,pi/2].
    //     Angles outside these ranges are wrapped back to the equivalent
    // angle inside the range.
}

pub fn build_primitive_internal_coords(mol:&Molecule){
    for fragment in mol.sub_graphs.iter(){
        let mut coordinate_matrix:Array2<f64> = Array::zeros((fragment.node_count(),3));
        for (index,z_i) in fragment.node_indices().enumerate(){
            coordinate_matrix.slice_mut(s![index,..]).assign(&mol.positions.slice(s![z_i.index(),..]));
        }
        let coordinate_vector:Array1<f64> = coordinate_matrix.into_shape(3*fragment.node_count()).unwrap();
        // for primitive internal coords
        // first distance
        // then angles
        // then dihedrals
        let mut internal_coords:Vec<_> = Vec::new();

        //distances
        for edge_index in fragment.edge_indices(){
            let (a,b) = fragment.edge_endpoints(edge_index).unwrap();
            //internal_coords.push(mol.distance_matrix[[a.index(),b.index()]]);
            let dist:Distance = Distance::new(a.index(),b.index(),&mol.distance_matrix);
            internal_coords.push(dist);
        }

        //angles
        let linthre:f64= 0.95;
        for b in fragment.node_indices(){
            for a in fragment.neighbors(b){
                for c in fragment.neighbors(b){
                    if a.index() < c.index(){
                        let angl:Angle = Angle::new(a.index(),b.index(),c.index());

                        if  angl.value(&coordinate_vector).cos().abs() < linthre{
                            internal_coords.push(angl);
                        }
                    }
                }
            }
        }
    }
}

// pub fn build_internal_coords(mol:&Molecule){
//     internal_coords:Vec<f64> = Vec::new()
//     for index_vec in mol.n_atoms.combinations(2){
//         let i:usize = index_vec.0;
//         let j:usize = index_vec.1;
//         if mol.connectivity_matrix[[i,j]]{
//
//         }
//     }
// }

pub struct Distance{
    at_a: usize,
    at_b: usize,
    distance: f64,
}
impl Distance{
    pub(crate) fn new(at_a:usize,at_b:usize,dist_matrix:&Array2<f64>) ->Distance {
        let at_a:usize = at_a;
        let at_b:usize = at_b;

        let distance:f64 = dist_matrix[[at_a,at_b]];

        let dist = Distance{
            at_a: at_a,
            at_b: at_b,
            distance: distance,
        };

        return dist;
    }
}

pub struct Angle{
    at_a: usize,
    at_b: usize,
    at_c: usize,
}

impl Angle{
    pub(crate) fn new(at_a:usize,at_b:usize,at_c:usize) ->Angle {
        let at_a:usize = at_a;
        let at_b:usize = at_b;
        let at_c:usize = at_c;

        let angle = Angle{
            at_a: at_a,
            at_b: at_b,
            at_c: at_c,
        };

        return angle;
    }

    pub fn value(self,coord_vector:&Array1<f64>)->f64{
        let a:usize = self.at_a;
        let b:usize = self.at_b;
        let c:usize = self.at_c;

        // vector from first atom to central
        let vec_1:Array1<f64> = coord_vector.slice(s![3*a..3*a+3]).to_owned()- coord_vector.slice(s![3*b..3*b+3]).to_owned();
        // vector from last atom to central
        let vec_2:Array1<f64> = coord_vector.slice(s![3*c..3*c+3]).to_owned()- coord_vector.slice(s![3*b..3*b+3]).to_owned();
        // norm of the vectors
        let norm_1:f64 = vec_1.norm();
        let norm_2:f64 = vec_2.norm();
        let dot:f64 = vec_1.dot(&vec_2);
        let factor:f64 = dot/(norm_1*norm_2);

        let mut return_value:f64 = 0.0;

        if (dot/(norm_1*norm_2)) <= -1.0{
            if ((dot/(norm_1*norm_2)).abs() +1.0) < -1e-6{
                println!("Invalued value in angle");
            }
            return_value = 1.0_f64*PI;
        }
        else if (dot/(norm_1*norm_2)) >= 1.0{
            if ((dot/(norm_1*norm_2)).abs() -1.0) < 1e-6{
                println!("Invalued value in angle");
            }
            return_value = 0.0;
        }
        else {
            return_value = (dot/(norm_1*norm_2)).acos();
        }
        return return_value;
    }
}

