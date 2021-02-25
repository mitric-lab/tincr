use crate::constants;
use crate::defaults;
use crate::gradients;
use crate::Molecule;
use itertools::Itertools;
use nalgebra::*;
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
        let mut internal_coords:Vec<IC> = Vec::new();

        //distances
        for edge_index in fragment.edge_indices(){
            let (a,b) = fragment.edge_endpoints(edge_index).unwrap();
            //internal_coords.push(mol.distance_matrix[[a.index(),b.index()]]);
            let dist:Distance = Distance::new(a.index(),b.index(),&mol.distance_matrix);
            let dist_ic = IC::distance(dist);
            internal_coords.push(dist_ic);
        }

        //angles
        let linthre:f64= 0.95;
        for b in fragment.node_indices(){
            for a in fragment.neighbors(b){
                for c in fragment.neighbors(b){
                    if a.index() < c.index(){
                        let angl:Angle = Angle::new(a.index(),b.index(),c.index());
                        // nnc part doesnt work

                        if angl.clone().value(&coordinate_vector).cos().abs() < linthre{
                            let angl_ic = IC::angle(angl);
                            internal_coords.push(angl_ic);
                        }
                        // cant check for nnc
                    }
                }
            }
        }
        //out of planes
        for b in fragment.node_indices(){
            for a in fragment.neighbors(b){
                for c in fragment.neighbors(b){
                    for d in fragment.neighbors(b){
                        // nc doesnt work
                        let it = vec![a.index(),c.index(),d.index()].into_iter().permutations(3);
                        for index in it.into_iter(){
                            let i = index[0];
                            let j = index[1];
                            let k = index[2];

                            let angl1:Angle = Angle::new(b.index(),i,j);
                            let angl2:Angle = Angle::new(i,j,k);
                            if angl1.value(&coordinate_vector).cos().abs() > linthre{
                                continue
                            }
                            if angl2.value(&coordinate_vector).cos().abs() > linthre{
                                continue
                            }
                            // need normal_vector fn here
                            if (angl1.normal_vector(&coordinate_vector).dot(&angl2.normal_vector(&coordinate_vector))).abs() > linthre{
                                // delete angle i,b,j
                                // for i in (0..internal_coords.len()).rev(){
                                //     // comparison doesnt work
                                //     if internal_coords[i] == IC::angle(Angle::new(i,b.index(),j)){
                                //         internal_coords.remove(i);
                                //     }
                                // }
                                // out of plane bijk
                                let out_of_pl1: Out_of_plane = Out_of_plane::new(b.index(),i,j,k);
                                let out_of_pl_ic = IC::out_of_plane(out_of_pl1);
                                internal_coords.push(out_of_pl_ic);
                            }
                        }
                    }
                }
            }
        }

        //dihedrals

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

//#[derive(Eq,PartialEq,Clone,Copy)]
pub enum IC{
    distance(Distance),
    angle(Angle),
    out_of_plane(Out_of_plane)
}
#[derive(Clone,Copy)]
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
#[derive(Eq,PartialEq,Clone,Copy)]
pub struct Out_of_plane{
    at_a: usize,
    at_b: usize,
    at_c: usize,
    at_d: usize
}
impl Out_of_plane{
    pub(crate) fn new(at_a:usize,at_b:usize,at_c:usize,at_d:usize)->Out_of_plane{
        let at_a:usize = at_a;
        let at_b:usize = at_b;
        let at_c:usize = at_c;
        let at_d:usize = at_d;

        let out_of_plane = Out_of_plane{
            at_a: at_a,
            at_b: at_b,
            at_c: at_c,
            at_d:at_d,
        };

        return out_of_plane;
    }
}


#[derive(Eq,PartialEq,Clone,Copy)]
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
        let vec_1:Vec<f64>= (coord_vector.slice(s![3*a..3*a+3]).to_owned()- coord_vector.slice(s![3*b..3*b+3]).to_owned()).to_vec();
        // vector from last atom to central
        let vec_2:Vec<f64> = (coord_vector.slice(s![3*c..3*c+3]).to_owned()- coord_vector.slice(s![3*b..3*b+3]).to_owned()).to_vec();
        // norm of the vectors
        let norm_1:f64 = vec_1.norm();
        let norm_2:f64 = vec_2.norm();
        let dot:f64 = Array::from_vec(vec_1).dot(&Array::from_vec(vec_2));
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

    pub fn normal_vector(self,coord_vector:&Array1<f64>)->(Array1<f64>){
        let a:usize = self.at_a;
        let b:usize = self.at_b;
        let c:usize = self.at_c;

        // vector from first atom to central
        let vec_1:Vec<f64>= (coord_vector.slice(s![3*a..3*a+3]).to_owned()- coord_vector.slice(s![3*b..3*b+3]).to_owned()).to_vec();
        // vector from last atom to central
        let vec_2:Vec<f64> = (coord_vector.slice(s![3*c..3*c+3]).to_owned()- coord_vector.slice(s![3*b..3*b+3]).to_owned()).to_vec();

        let norm_1:f64 = vec_1.norm();
        let norm_2:f64 = vec_2.norm();

        // need cross product here
        let crs:Vec<f64> = vec_1.cross(&vec_2);
        let crs_2:Array1<f64> = Array::from_vec(crs.clone())/crs.norm();

        return crs_2;
    }
}

