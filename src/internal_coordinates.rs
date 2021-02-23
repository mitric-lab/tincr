use crate::defaults;
use crate::gradients;
use crate::Molecule;
use crate::constants;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;

pub fn cartesian_to_internal(coord_vector:&Array1<f64>, mol: &Molecule){
    let masses:Array1<f64> = build_masses(mol);
    let x_vec:Array1<f64> = shift_to_com(coord_vector,&masses);

}

pub fn shift_to_com(coord_vector:&Array1<f64>, masses:&Array1<f64>)->(Array1<f64>) {
    // shift center of mass to the origin
    let com:Array1<f64> = Array::from_vec(center_of_mass(coord_vector,masses));
    let mut pos_shifted:Array1<f64> = Array::zeros(coord_vector.len());
    let n_at:usize = coord_vector.len()/3;

    for i in 0.. n_at{
        pos_shifted.slice_mut(s![3*i..3*i+3]).assign(&(coord_vector.slice(s![3*i..3*i+3]).to_owned()- &com));
    }
    return pos_shifted;
}

pub fn center_of_mass(coord_vector:&Array1<f64>, masses:&Array1<f64>)->(Vec<f64>){
    // find the center of mass
    let xm: Array1<f64> = coord_vector * masses;
    let mut com: Vec<f64> = Vec::new();
    com.push(xm.slice(s![0..;3]).sum() / masses.slice(s![0..;3]).sum());
    com.push(xm.slice(s![1..;3]).sum() / masses.slice(s![0..;3]).sum());
    com.push(xm.slice(s![2..;3]).sum() / masses.slice(s![0..;3]).sum());

    return com;
}

pub fn build_masses(mol: &Molecule)->(Array1<f64>){
    let mut masses:Vec<f64> = Array::zeros(mol.n_atoms*3).to_vec();
    for n in mol.atomic_numbers.iter(){
        masses.push(constants::ATOMIC_MASSES[n]);
        masses.push(constants::ATOMIC_MASSES[n]);
        masses.push(constants::ATOMIC_MASSES[n]);
    }
    return Array::from_vec(masses);
}

pub fn internal_to_cartesian(coord_vector:&Array1<f64>){
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

pub fn wrap_angles(){
    // Bending angles and dihedral angles have to be in the
    // range [0,pi], while inversion angles have to be in the range [-pi/2,pi/2].
    //     Angles outside these ranges are wrapped back to the equivalent
    // angle inside the range.


}