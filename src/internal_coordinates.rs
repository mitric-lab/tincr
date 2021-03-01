use crate::constants;
use crate::defaults;
use crate::gradients;
use crate::molecule::distance_matrix;
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
use std::f64::consts::PI;
use std::cmp::Ordering;
use std::ops::AddAssign;

pub fn argsort(v: ArrayView1<f64>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(Ordering::Equal));
    idx
}

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

pub fn reorder_primitives(){

}

pub fn build_primitives(mol: &Molecule)->Vec<IC> {
    let coordinate_vector: Array1<f64> = mol
        .positions
        .clone()
        .into_shape(3 * mol.full_graph.node_count())
        .unwrap();
    // for primitive internal coords
    // first distance
    // then angles
    // then out of plane
    // then dihedrals
    let mut internal_coords: Vec<IC> = Vec::new();
    let mut distances_vec:Vec<Distance> = Vec::new();
    let mut angles_vec:Vec<Angle> = Vec::new();
    let mut outofplane_vec:Vec<Out_of_plane> = Vec::new();
    let mut dihedral_vec:Vec<Dihedral> = Vec::new();
    let mut cartesian_x_vec:Vec<CartesianX> = Vec::new();
    let mut cartesian_y_vec:Vec<CartesianY> = Vec::new();
    let mut cartesian_z_vec:Vec<CartesianZ> = Vec::new();
    let mut translation_x_vec:Vec<TranslationX> = Vec::new();
    let mut translation_y_vec:Vec<TranslationY> = Vec::new();
    let mut translation_z_vec:Vec<TranslationZ> = Vec::new();
    let mut rotation_a_Vec: Vec<RotationA> = Vec::new();
    let mut rotation_b_Vec: Vec<RotationB> = Vec::new();
    let mut rotation_c_Vec: Vec<RotationC> = Vec::new();

    for fragment in mol.sub_graphs.clone() {
        if fragment.node_count() >= 2 {
            let mut node_vec: Vec<NodeIndex> = Vec::new();
            for i in fragment.node_indices() {
                node_vec.push(i);
            }
            let trans_x: TranslationX = TranslationX::new(
                node_vec.clone(),
                Array::ones(node_vec.len()) / (node_vec.len() as f64),
            );
            let trans_y: TranslationY = TranslationY::new(
                node_vec.clone(),
                Array::ones(node_vec.len()) / (node_vec.len() as f64),
            );
            let trans_z: TranslationZ = TranslationZ::new(
                node_vec.clone(),
                Array::ones(node_vec.len()) / (node_vec.len() as f64),
            );
            // internal_coords.push(IC::translation_x(trans_x));
            // internal_coords.push(IC::translation_y(trans_y));
            // internal_coords.push(IC::translation_z(trans_z));
            translation_x_vec.push(trans_x);
            translation_y_vec.push(trans_y);
            translation_z_vec.push(trans_z);

            let mut sel: Array2<f64> = coordinate_vector
                .clone()
                .into_shape((mol.n_atoms, 3))
                .unwrap()
                .slice(s![
                    node_vec[0].index()..node_vec.last().unwrap().index(),
                    ..
                ])
                .to_owned();
            sel = &sel - &sel.mean_axis(Axis(0)).unwrap();
            let rg: f64 = sel
                .mapv(|sel| sel.powi(2))
                .sum_axis(Axis(1))
                .mean()
                .unwrap()
                .sqrt();

            // rotations
            let rot_a:RotationA = RotationA::new(node_vec.clone(),coordinate_vector.clone(),rg);
            let rot_b:RotationB = RotationB::new(node_vec.clone(),coordinate_vector.clone(),rg);
            let rot_c:RotationC = RotationC::new(node_vec.clone(),coordinate_vector.clone(),rg);
            // internal_coords.push(IC::rotation_a(rot_a));
            // internal_coords.push(IC::rotation_b(rot_b));
            // internal_coords.push(IC::rotation_c(rot_c));
            rotation_a_Vec.push(rot_a);
            rotation_b_Vec.push(rot_b);
            rotation_c_Vec.push(rot_c);
        }
        else{
            for j in fragment.node_indices(){
                // add cartesian
                let cart_x:CartesianX = CartesianX::new(j.index(),1.0);
                let cart_y:CartesianY = CartesianY::new(j.index(),1.0);
                let cart_z:CartesianZ = CartesianZ::new(j.index(),1.0);
                // internal_coords.push(IC::cartesian_x(cart_x));
                // internal_coords.push(IC::cartesian_y(cart_y));
                // internal_coords.push(IC::cartesian_z(cart_z));
                cartesian_x_vec.push(cart_x);
                cartesian_y_vec.push(cart_y);
                cartesian_z_vec.push(cart_z);
            }
        }
    }

    //distances
    for edge_index in mol.full_graph.edge_indices() {
        let (a, b) = mol.full_graph.edge_endpoints(edge_index).unwrap();
        //internal_coords.push(mol.distance_matrix[[a.index(),b.index()]]);
        let dist: Distance = Distance::new(a.index(), b.index());
        //let dist_ic = IC::distance(dist);
        //internal_coords.push(dist_ic);
        distances_vec.push(dist);
    }

    //angles
    let linthre: f64 = 0.95;
    for b in mol.full_graph.node_indices() {
        for a in mol.full_graph.neighbors(b) {
            for c in mol.full_graph.neighbors(b) {
                if a.index() < c.index() {
                    let angl: Angle = Angle::new(a.index(), b.index(), c.index());
                    // nnc part doesnt work

                    if angl.clone().value(&coordinate_vector).cos().abs() < linthre {
                        //let angl_ic = IC::angle(angl);
                        //internal_coords.push(angl_ic);
                        angles_vec.push(angl);
                    }
                    // cant check for nnc
                }
            }
        }
    }
    //out of planes
    for b in mol.full_graph.node_indices() {
        for a in mol.full_graph.neighbors(b) {
            for c in mol.full_graph.neighbors(b) {
                for d in mol.full_graph.neighbors(b) {
                    // nc doesnt work
                    let it = vec![a.index(), c.index(), d.index()]
                        .into_iter()
                        .permutations(3);
                    for index in it.into_iter() {
                        let i = index[0];
                        let j = index[1];
                        let k = index[2];

                        let angl1: Angle = Angle::new(b.index(), i, j);
                        let angl2: Angle = Angle::new(i, j, k);
                        if angl1.value(&coordinate_vector).cos().abs() > linthre {
                            continue;
                        }
                        if angl2.value(&coordinate_vector).cos().abs() > linthre {
                            continue;
                        }
                        // need normal_vector fn here
                        if (angl1
                            .normal_vector(&coordinate_vector)
                            .dot(&angl2.normal_vector(&coordinate_vector)))
                        .abs()
                            > linthre
                        {
                            // delete angle i,b,j
                            // for i in (0..internal_coords.len()).rev() {
                            //     if internal_coords[i] == IC::angle(Angle::new(i, b.index(), j)) {
                            //         internal_coords.remove(i);
                            //     }
                            // }
                            for i in (0..angles_vec.len()).rev() {
                                if angles_vec[i] == Angle::new(i, b.index(), j){
                                    angles_vec.remove(i);
                                }
                            }
                            // out of plane bijk
                            let out_of_pl1: Out_of_plane = Out_of_plane::new(b.index(), i, j, k);
                            // let out_of_pl_ic = IC::out_of_plane(out_of_pl1);
                            // internal_coords.push(out_of_pl_ic);
                            outofplane_vec.push(out_of_pl1);
                        }
                    }
                }
            }
        }
    }
    // Find groups of atoms that are in straight lines
    let mut atom_lines: Vec<Vec<NodeIndex>> = Vec::new();
    for i in mol.full_graph.edge_indices() {
        let mut aline: Vec<NodeIndex> = Vec::new();
        let (ab, ay) = mol.full_graph.edge_endpoints(i).unwrap();
        aline.push(ab);
        aline.push(ay);
        atom_lines.push(aline);
    }
    let mut aline_new: Vec<NodeIndex> = Vec::new();
    let mut atom_lines_new: Vec<Vec<NodeIndex>> = Vec::new();

    while true {
        atom_lines = atom_lines_new.clone();
        atom_lines_new = Vec::new();
        let mut convergence_1: bool = false;
        let mut convergence_2: bool = false;

        for aline in atom_lines {
            aline_new = aline.clone();
            let ab: NodeIndex = aline[0];
            let ay: NodeIndex = *aline.last().unwrap();

            for aa in mol.full_graph.neighbors(ab) {
                if aa != ab && aa != ay {
                    // If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                    // Add AA to the front of the list
                    let mut val_vector: Vec<f64> = Vec::new();
                    for ac in aline[1..].iter() {
                        if *ac != ab {
                            let angl = Angle::new(aa.index(), ab.index(), ac.index());
                            let val: f64 = angl.value(&coordinate_vector).cos().abs();
                            val_vector.insert(0,val);
                        }
                    }
                    let indices_values: Array1<usize> = Array::from_vec(val_vector.clone())
                        .indexed_iter()
                        .filter_map(
                            |(index, &item)| if item > linthre { Some(index) } else { None },
                        )
                        .collect();
                    if indices_values.len() == val_vector.len() {
                        aline_new.push(aa);
                    } else {
                        convergence_1 = true;
                    }
                }
            }
            for az in mol.full_graph.neighbors(ay) {
                if az != ab && az != ay {
                    let mut val_vector: Vec<f64> = Vec::new();
                    for ax in aline[1..].iter() {
                        if *ax != ay {
                            let angl = Angle::new(ax.index(), ay.index(), az.index());
                            let val: f64 = angl.value(&coordinate_vector).cos().abs();
                            val_vector.push(val);
                        }
                    }
                    let indices_values: Array1<usize> = Array::from_vec(val_vector.clone())
                        .indexed_iter()
                        .filter_map(
                            |(index, &item)| if item > linthre { Some(index) } else { None },
                        )
                        .collect();
                    if indices_values.len() == val_vector.len() {
                        aline_new.push(az);
                    } else {
                        convergence_2 = true;
                    }
                }
            }
            atom_lines_new.push(aline_new.clone());
        }
        if convergence_1 == true && convergence_2 == true {
            break;
        }
    }

    // dihedrals
    for aline in atom_lines_new {
        //Go over ALL pairs of atoms in a line
        for vec in aline.clone().into_iter().combinations(2) {
            let b: NodeIndex = vec[0];
            let c: NodeIndex = vec[1];
            let mut b_new: NodeIndex = b;
            let mut c_new: NodeIndex = c;

            if b.index() > c.index() {
                b_new = c;
                c_new = b;
            }
            for a in mol.full_graph.neighbors(b_new) {
                for d in mol.full_graph.neighbors(c_new) {
                    if aline.contains(&a) == false && aline.contains(&d) == false && a != d {
                        let angl1: Angle = Angle::new(a.index(), b.index(), c.index());
                        let angl2: Angle = Angle::new(b.index(), c.index(), d.index());

                        if angl1.value(&coordinate_vector).cos().abs() > linthre {
                            continue;
                        }
                        if angl2.value(&coordinate_vector).cos().abs() > linthre {
                            continue;
                        }
                        let dihedral: Dihedral =
                            Dihedral::new(a.index(), b.index(), c.index(), d.index());
                        //internal_coords.push(IC::dihedral(dihedral));
                        dihedral_vec.push(dihedral);
                    }
                }
            }
        }
    }
    // reorder internal coordinates
    // One is unable to iterate over enum
    // thus, separate vectors are needed for
    // each operation
    for i in distances_vec{
        let dist = IC::distance(i);
        internal_coords.push(dist);
    }
    for i in angles_vec{
        let angl = IC::angle(i);
        internal_coords.push(angl);
    }
    for i in outofplane_vec{
        let out_of_pl = IC::out_of_plane(i);
        internal_coords.push(out_of_pl);
    }
    for i in dihedral_vec{
        let dihedral = IC::dihedral(i);
        internal_coords.push(dihedral);
    }
    for i in cartesian_x_vec{
        let cart = IC::cartesian_x(i);
        internal_coords.push(cart);
    }
    for i in cartesian_y_vec{
        let cart = IC::cartesian_y(i);
        internal_coords.push(cart);
    }
    for i in cartesian_z_vec{
        let cart = IC::cartesian_z(i);
        internal_coords.push(cart);
    }
    for i in translation_x_vec{
        let trans = IC::translation_x(i);
        internal_coords.push(trans);
    }
    for i in translation_y_vec{
        let trans = IC::translation_y(i);
        internal_coords.push(trans);
    }
    for i in translation_z_vec{
        let trans = IC::translation_z(i);
        internal_coords.push(trans);
    }
    for i in rotation_a_Vec{
        let rot = IC::rotation_a(i);
        internal_coords.push(rot);
    }
    for i in rotation_b_Vec{
        let rot = IC::rotation_b(i);
        internal_coords.push(rot);
    }
    for i in rotation_c_Vec{
        let rot = IC::rotation_c(i);
        internal_coords.push(rot);
    }

    return internal_coords;
}

pub fn build_delocalized_internal_coordinates(coords:Array1<f64>,primitives:Vec<IC>){
    let nprims:usize = primitives.len();
    // build g_matrix
}

pub fn build_g_matrix(coords:Array1<f64>){

}

pub fn check_linearity(x:Array2<f64>,y:Array2<f64>)->bool{
    let x:Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y:Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let f_mat:Array2<f64> = build_f_matrix(&x,&y);

    let n:usize = x.dim().0;
    let tmp:(Array1<f64>,Array2<f64>) = sorted_eigh_linearity(f_mat);
    let l_arr:Array1<f64> = tmp.0;
    let mut bool_return:bool = false;

    if ((l_arr[0]/l_arr[1]) < 1.01) && ((l_arr[0]/l_arr[1]) > 0.0){
        bool_return = true;
    }
    return bool_return;
}

pub fn build_f_matrix(x:&Array2<f64>,y:&Array2<f64>)->Array2<f64>{
    let r_mat:Array2<f64> = build_correlation_for_f_mat(x,y);
    let mut f:Array2<f64> = Array::zeros((4,4));
    let r_11:f64 = r_mat[[0,0]];
    let r_12:f64 = r_mat[[0,1]];
    let r_13:f64 = r_mat[[0,2]];
    let r_21:f64 = r_mat[[1,0]];
    let r_22:f64 = r_mat[[1,1]];
    let r_23:f64 = r_mat[[1,2]];
    let r_31:f64 = r_mat[[2,0]];
    let r_32:f64 = r_mat[[2,1]];
    let r_33:f64 = r_mat[[2,2]];
    f[[0,0]] = r_11 + r_22 + r_33;
    f[[0,1]] = r_23 - r_32;
    f[[0,2]] = r_31 - r_13;
    f[[0,3]] = r_12 - r_21;
    f[[1,0]] = r_23 - r_32;
    f[[1,1]] = r_11 - r_22 - r_33;
    f[[1,2]] = r_12 + r_21;
    f[[1,3]] = r_13 + r_31;
    f[[2,0]] = r_31 - r_13;
    f[[2,1]] = r_12 + r_21;
    f[[2,2]] = r_22 - r_33 - r_11;
    f[[2,3]] = r_23 + r_32;
    f[[3,0]] = r_12 - r_21;
    f[[3,1]] = r_13 + r_31;
    f[[3,2]] = r_23 + r_32;
    f[[3,3]] = r_33 - r_22 - r_11;

    return f;
}

pub fn build_correlation_for_f_mat(x:&Array2<f64>,y:&Array2<f64>)->Array2<f64>{
    let xmat:Array2<f64> = x.clone().reversed_axes();
    let ymat:Array2<f64> = y.clone().reversed_axes();

    let dot:Array2<f64> = xmat.dot(&ymat.t());
    return dot;
}

pub fn sorted_eigh_linearity(f_mat:Array2<f64>)->(Array1<f64>,Array2<f64>){
    // Return eigenvalues and eigenvectors of a symmetric matrix
    // in descending order and associated eigenvectors.
    // This is just a convenience function to get eigenvectors
    // in descending or ascending order as desired.
    let tmp:(Array1<f64>,Array2<f64>) = f_mat.eigh(UPLO::Upper).unwrap();
    let l_vec: Array1<f64> = tmp.0;
    let q_mat: Array2<f64> = tmp.1;

    let idx:Vec<usize> = argsort(l_vec.view());
    let mut l_new:Vec<f64> = Vec::new();
    let mut q_mat_new:Array2<f64> = Array::zeros((q_mat.dim().0,q_mat.dim().1));
    for (i,j) in idx.iter().rev().enumerate(){
        l_new.push(l_vec[*j]);
        q_mat_new.slice_mut(s![..,i]).assign(&q_mat.slice(s![..,*j]));
    }
    let l_arr:Array1<f64> = Array::from(l_new);

    return (l_arr, q_mat_new);
}

pub fn get_quat_rot(x:&Array2<f64>,y:&Array2<f64>)->(Array1<f64>,Array1<f64>){
    let x:Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y:Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let f_mat:Array2<f64> = build_f_matrix(&x,&y);

    let n:usize = x.dim().0;
    let tmp:(Array1<f64>,Array2<f64>) = sorted_eigh_linearity(f_mat);
    let l_vec:Array1<f64> = tmp.0;
    let q_mat:Array2<f64> = tmp.1;

    let mut q_vec:Array1<f64> = q_mat.slice(s![..,0]).to_owned();
    if q_vec[0] < 0.0{
        q_vec = q_vec * (-1.0);
    }
    return (q_vec,l_vec);
}

pub fn get_exmap_rot(x:&Array2<f64>,y:&Array2<f64>)->Array1<f64>{
    let (q_vec,l_vec):(Array1<f64>,Array1<f64>) = get_quat_rot(x,y);
    let (fac,dfac):(f64,f64) = calc_fac_dfac_rot(q_vec[0]);
    let v:Array1<f64> = fac * q_vec.slice(s![1..]).to_owned();

    return v;
}

pub fn calc_fac_dfac_rot(q0:f64)->(f64,f64){
    let qm1 = q0 -1.0;
    let mut fac:f64 = 0.0;
    let mut dfac:f64 = 0.0;

    if qm1.abs() < 1e-8{
        fac = 2.0 - 2.0 *qm1/3.0;
        dfac = -2.0/3.0;
    }
    else{
        let s:f64 = (1.0-q0.powi(2)).sqrt();
        let a:f64 = q0.acos();
        fac = 2.0 * a/s;
        dfac = -2.0/s.powi(2);
        dfac += 2.0*q0*a/s.powi(3);
    }
    return (fac,dfac);
}

pub fn get_r_deriv(x:&Array2<f64>,y:&Array2<f64>)->Array4<f64>{
    // Calculate the derivatives of the correlation matrix with respect
    // to the Cartesian coordinates.
    let x:Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y:Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let mut a_diff_r:Array4<f64> = Array::zeros((x.dim().0,3,3,3));
    for u in 0..x.dim().0{
        for w in 0..3{
            for i in 0..3{
                for j in 0..3{
                    if i == w{
                        a_diff_r[[u,w,i,j]] = y[[u,j]];
                    }
                }
            }
        }
    }
    return a_diff_r;
}

pub fn build_f_matrix_deriv(x:&Array2<f64>,y:&Array2<f64>)->Array4<f64>{
    // Calculate the derivatives of the F-matrix with respect
    // to the Cartesian coordinates.
    let x:Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y:Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let dr:Array4<f64> = get_r_deriv(&x,&y);
    let mut df:Array4<f64> = Array::zeros((x.dim().0,3,4,4));
    for u in 0..x.dim().0 {
        for w in 0..3 {
            let dr_11:f64 = dr[[u,w,0,0]];
            let dr_12:f64 = dr[[u,w,0,1]];
            let dr_13:f64 = dr[[u,w,0,2]];
            let dr_21:f64 = dr[[u,w,1,0]];
            let dr_22:f64 = dr[[u,w,1,1]];
            let dr_23:f64 = dr[[u,w,1,2]];
            let dr_31:f64 = dr[[u,w,2,0]];
            let dr_32:f64 = dr[[u,w,2,1]];
            let dr_33:f64 = dr[[u,w,2,2]];
            df[[u,w,0,0]] = dr_11 + dr_22 + dr_33;
            df[[u,w,0,1]] = dr_23 - dr_32;
            df[[u,w,0,2]] = dr_31 - dr_13;
            df[[u,w,0,3]] = dr_12 - dr_21;
            df[[u,w,1,0]] = dr_23 - dr_32;
            df[[u,w,1,1]] = dr_11 -dr_22 - dr_33;
            df[[u,w,1,2]] = dr_12 + dr_21;
            df[[u,w,1,3]] = dr_13 + dr_31;
            df[[u,w,2,0]] = dr_31 - dr_13;
            df[[u,w,2,1]] = dr_12 + dr_21;
            df[[u,w,2,2]] = dr_22 - dr_33 - dr_11;
            df[[u,w,2,3]] = dr_23 + dr_32;
            df[[u,w,3,0]] = dr_12 - dr_21;
            df[[u,w,3,1]] = dr_13 + dr_31;
            df[[u,w,3,2]] = dr_12 + dr_32;
            df[[u,w,3,3]] = dr_33 - dr_22 - dr_11;
        }
    }
    return df;

}

pub fn get_q_der_rot(x:&Array2<f64>,y:&Array2<f64>){
    let x:Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y:Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let (q_vec,l_vec):(Array1<f64>,Array1<f64>) = get_quat_rot(&x,&y);
    let f_mat:Array2<f64> = build_f_matrix(&x,&y);
    let f_mat_deriv:Array4<f64> = build_f_matrix_deriv(&x,&y);


}

pub fn get_exmap_deriv_rot(x:&Array2<f64>,y:&Array2<f64>){
    // Given trial coordinates x and target coordinates y,
    // return the derivatives of the exponential map that brings
    // x into maximal coincidence (minimum RMSD) with y, with
    // respect to the coordinates of x.
    let (q_vec,l_vec):(Array1<f64>,Array1<f64>) = get_quat_rot(x,y);
    let v:Array1<f64> = get_exmap_rot(x,y);
    let (fac,dfac):(f64,f64) = calc_fac_dfac_rot(q_vec[0]);

    let mut dvdq:Array2<f64> = Array::zeros((4,3));
    dvdq.slice_mut(s![0,..]).assign(&(dfac*q_vec.slice(s![1..]).to_owned()));
    for i in (0..3){
        dvdq[[i+1,i]] = fac;
    }

    //let dqdx =

}

#[derive(PartialEq, Clone)]
pub enum IC {
    distance(Distance),
    angle(Angle),
    out_of_plane(Out_of_plane),
    dihedral(Dihedral),
    cartesian_x(CartesianX),
    cartesian_y(CartesianY),
    cartesian_z(CartesianZ),
    translation_x(TranslationX),
    translation_y(TranslationY),
    translation_z(TranslationZ),
    rotation_a(RotationA),
    rotation_b(RotationB),
    rotation_c(RotationC)
}
#[derive(Clone, PartialEq)]
pub struct CartesianX {
    at_a: usize,
    w_val: f64,
}
impl CartesianX {
    pub(crate) fn new(at_a: usize, w_val:f64) -> CartesianX {
        let at_a:usize =  at_a;
        let w_val:f64 = w_val;

        let cart = CartesianX {
            at_a: at_a,
            w_val: w_val,
        };
        return cart;
    }
    pub fn derivatives(self,coords:Array1<f64>)-> Array2<f64>{
        let n_at:usize = coords.len()/3;
        let coords_new:Array2<f64> = coords.into_shape((n_at,3)).unwrap();
        let mut derivatives:Array2<f64> = Array::zeros((n_at,3));
        derivatives[[self.at_a,0]] = self.w_val;

        return derivatives;
    }
}

#[derive(Clone, PartialEq)]
pub struct CartesianY {
    at_a: usize,
    w_val: f64,
}
impl CartesianY {
    pub(crate) fn new(at_a: usize, w_val:f64) -> CartesianY {
        let at_a:usize =  at_a;
        let w_val:f64 = w_val;

        let cart = CartesianY {
            at_a: at_a,
            w_val: w_val,
        };
        return cart;
    }

    pub fn derivatives(self,coords:Array1<f64>)-> Array2<f64>{
        let n_at:usize = coords.len()/3;
        let coords_new:Array2<f64> = coords.into_shape((n_at,3)).unwrap();
        let mut derivatives:Array2<f64> = Array::zeros((n_at,3));
        derivatives[[self.at_a,1]] = self.w_val;

        return derivatives;
    }
}

#[derive(Clone, PartialEq)]
pub struct CartesianZ {
    at_a: usize,
    w_val: f64,
}
impl CartesianZ {
    pub(crate) fn new(at_a: usize, w_val:f64) -> CartesianZ {
        let at_a:usize =  at_a;
        let w_val:f64 = w_val;

        let cart = CartesianZ {
            at_a: at_a,
            w_val: w_val,
        };
        return cart;
    }
    pub fn derivatives(self,coords:Array1<f64>)-> Array2<f64>{
        let n_at:usize = coords.len()/3;
        let coords_new:Array2<f64> = coords.into_shape((n_at,3)).unwrap();
        let mut derivatives:Array2<f64> = Array::zeros((n_at,3));
        derivatives[[self.at_a,2]] = self.w_val;

        return derivatives;
    }
}

#[derive(Clone, PartialEq)]
pub struct TranslationX {
    nodes: Vec<NodeIndex>,
    w_vec: Array1<f64>,
}
impl TranslationX {
    pub(crate) fn new(nodes: Vec<NodeIndex>, w_vec: Array1<f64>) -> TranslationX {
        let nodes: Vec<NodeIndex> = nodes;
        let w_vec: Array1<f64> = w_vec;

        let trans = TranslationX {
            nodes: nodes,
            w_vec: w_vec,
        };
        return trans;
    }
    pub fn derivatives(self,coords:Array1<f64>)-> Array2<f64>{
        let n_at:usize = coords.len()/3;
        let coords_new:Array2<f64> = coords.into_shape((n_at,3)).unwrap();
        let mut derivatives:Array2<f64> = Array::zeros((n_at,3));

        for (i,a) in self.nodes.iter().enumerate(){
            derivatives[[a.index(),0]] = self.w_vec[i];
        }

        return derivatives;
    }
}

#[derive(Clone, PartialEq)]
pub struct TranslationY {
    nodes: Vec<NodeIndex>,
    w_vec: Array1<f64>,
}
impl TranslationY {
    pub(crate) fn new(nodes: Vec<NodeIndex>, w_vec: Array1<f64>) -> TranslationY {
        let nodes: Vec<NodeIndex> = nodes;
        let w_vec: Array1<f64> = w_vec;

        let trans = TranslationY {
            nodes: nodes,
            w_vec: w_vec,
        };
        return trans;
    }

    pub fn derivatives(self,coords:Array1<f64>)-> Array2<f64>{
        let n_at:usize = coords.len()/3;
        let coords_new:Array2<f64> = coords.into_shape((n_at,3)).unwrap();
        let mut derivatives:Array2<f64> = Array::zeros((n_at,3));

        for (i,a) in self.nodes.iter().enumerate(){
            derivatives[[a.index(),1]] = self.w_vec[i];
        }

        return derivatives;
    }
}

#[derive(Clone, PartialEq)]
pub struct TranslationZ {
    nodes: Vec<NodeIndex>,
    w_vec: Array1<f64>,
}
impl TranslationZ {
    pub(crate) fn new(nodes: Vec<NodeIndex>, w_vec: Array1<f64>) -> TranslationZ {
        let nodes: Vec<NodeIndex> = nodes;
        let w_vec: Array1<f64> = w_vec;

        let trans = TranslationZ {
            nodes: nodes,
            w_vec: w_vec,
        };
        return trans;
    }
    pub fn derivatives(self,coords:Array1<f64>)-> Array2<f64>{
        let n_at:usize = coords.len()/3;
        let coords_new:Array2<f64> = coords.into_shape((n_at,3)).unwrap();
        let mut derivatives:Array2<f64> = Array::zeros((n_at,3));

        for (i,a) in self.nodes.iter().enumerate(){
            derivatives[[a.index(),2]] = self.w_vec[i];
        }

        return derivatives;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Distance {
    at_a: usize,
    at_b: usize,
}
impl Distance {
    pub(crate) fn new(at_a: usize, at_b: usize) -> Distance {
        let at_a: usize = at_a;
        let at_b: usize = at_b;

        let dist = Distance {
            at_a: at_a,
            at_b: at_b,
        };

        return dist;
    }
    pub fn get_distance(self, dist_matrix: &Array2<f64>) -> f64 {
        let distance: f64 = dist_matrix[[self.at_a, self.at_b]];
        return distance;
    }
}
#[derive(Eq, PartialEq, Clone, Copy)]
pub struct Out_of_plane {
    at_a: usize,
    at_b: usize,
    at_c: usize,
    at_d: usize,
}
impl Out_of_plane {
    pub(crate) fn new(at_a: usize, at_b: usize, at_c: usize, at_d: usize) -> Out_of_plane {
        let at_a: usize = at_a;
        let at_b: usize = at_b;
        let at_c: usize = at_c;
        let at_d: usize = at_d;

        let out_of_plane = Out_of_plane {
            at_a: at_a,
            at_b: at_b,
            at_c: at_c,
            at_d: at_d,
        };

        return out_of_plane;
    }
}

#[derive(Eq, PartialEq, Clone, Copy)]
pub struct Angle {
    at_a: usize,
    at_b: usize,
    at_c: usize,
}

impl Angle {
    pub(crate) fn new(at_a: usize, at_b: usize, at_c: usize) -> Angle {
        let at_a: usize = at_a;
        let at_b: usize = at_b;
        let at_c: usize = at_c;

        let angle = Angle {
            at_a: at_a,
            at_b: at_b,
            at_c: at_c,
        };

        return angle;
    }

    pub fn value(self, coord_vector: &Array1<f64>) -> f64 {
        let a: usize = self.at_a;
        let b: usize = self.at_b;
        let c: usize = self.at_c;

        // vector from first atom to central
        let vec_1: Vec<f64> = (coord_vector.slice(s![3 * a..3 * a + 3]).to_owned()
            - coord_vector.slice(s![3 * b..3 * b + 3]).to_owned())
        .to_vec();
        // vector from last atom to central
        let vec_2: Vec<f64> = (coord_vector.slice(s![3 * c..3 * c + 3]).to_owned()
            - coord_vector.slice(s![3 * b..3 * b + 3]).to_owned())
        .to_vec();
        // norm of the vectors
        let norm_1: f64 = vec_1.norm();
        let norm_2: f64 = vec_2.norm();
        let dot: f64 = Array::from_vec(vec_1).dot(&Array::from_vec(vec_2));
        let factor: f64 = dot / (norm_1 * norm_2);

        let mut return_value: f64 = 0.0;

        if (dot / (norm_1 * norm_2)) <= -1.0 {
            if ((dot / (norm_1 * norm_2)).abs() + 1.0) < -1e-6 {
                println!("Invalued value in angle");
            }
            return_value = 1.0_f64 * PI;
        } else if (dot / (norm_1 * norm_2)) >= 1.0 {
            if ((dot / (norm_1 * norm_2)).abs() - 1.0) < 1e-6 {
                println!("Invalued value in angle");
            }
            return_value = 0.0;
        } else {
            return_value = (dot / (norm_1 * norm_2)).acos();
        }
        return return_value;
    }

    pub fn normal_vector(self, coord_vector: &Array1<f64>) -> (Array1<f64>) {
        let a: usize = self.at_a;
        let b: usize = self.at_b;
        let c: usize = self.at_c;

        // vector from first atom to central
        let vec_1: Vec<f64> = (coord_vector.slice(s![3 * a..3 * a + 3]).to_owned()
            - coord_vector.slice(s![3 * b..3 * b + 3]).to_owned())
        .to_vec();
        // vector from last atom to central
        let vec_2: Vec<f64> = (coord_vector.slice(s![3 * c..3 * c + 3]).to_owned()
            - coord_vector.slice(s![3 * b..3 * b + 3]).to_owned())
        .to_vec();

        let norm_1: f64 = vec_1.norm();
        let norm_2: f64 = vec_2.norm();

        // need cross product here
        let crs: Vec<f64> = vec_1.cross(&vec_2);
        let crs_2: Array1<f64> = Array::from_vec(crs.clone()) / crs.norm();

        return crs_2;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Dihedral {
    at_a: usize,
    at_b: usize,
    at_c: usize,
    at_d: usize,
}

impl Dihedral {
    pub(crate) fn new(at_a: usize, at_b: usize, at_c: usize, at_d: usize) -> Dihedral {
        let at_a: usize = at_a;
        let at_b: usize = at_b;
        let at_c: usize = at_c;
        let at_d: usize = at_d;

        let dihedral = Dihedral {
            at_a: at_a,
            at_b: at_b,
            at_c: at_c,
            at_d: at_d,
        };

        return dihedral;
    }
}
#[derive(Clone, PartialEq)]
pub struct RotationA {
    nodes: Vec<NodeIndex>,
    coords: Array1<f64>,
    w_val: f64,
}

impl RotationA {
    pub(crate) fn new(nodes: Vec<NodeIndex>, coords: Array1<f64>, w: f64) -> RotationA {
        let nodes: Vec<NodeIndex> = nodes;
        let coords: Array1<f64> = coords;
        let w:f64 = w;

        let rotation = RotationA {
            nodes: nodes,
            coords: coords,
            w_val: w,
        };

        return rotation;
    }
    pub fn derivatives(self,coords:Array1<f64>)-> Array3<f64>{
        let n_at:usize = coords.len()/3;
        let coords_new:Array2<f64> = coords.into_shape((n_at,3)).unwrap();
        let coords_self:Array2<f64> = self.coords.into_shape((n_at,3)).unwrap();
        let mut derivatives:Array3<f64> = Array::zeros((n_at,3,3));

        let mut x_sel:Array2<f64> = Array::zeros((self.nodes.len(),3));
        let mut y_sel:Array2<f64> = Array::zeros((self.nodes.len(),3));
        for (i,j) in self.nodes.iter().enumerate(){
            x_sel.slice_mut(s![i,..]).assign(&coords_new.slice(s![j.index(),..]));
            y_sel.slice_mut(s![i,..]).assign(&coords_self.slice(s![j.index(),..]));
        }
        let x_mean:Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean:Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear:bool = false;

        if check_linearity(x_sel,y_sel){
            bool_linear = true;
        }
        if bool_linear{
            //
        }
        //let deriv_raw =
        if bool_linear{
            //
        }

        return derivatives;
    }
}
#[derive(Clone, PartialEq)]
pub struct RotationB {
    nodes: Vec<NodeIndex>,
    coords: Array1<f64>,
    w_val: f64,
}

impl RotationB {
    pub(crate) fn new(nodes: Vec<NodeIndex>, coords: Array1<f64>, w: f64) -> RotationB {
        let nodes: Vec<NodeIndex> = nodes;
        let coords: Array1<f64> = coords;
        let w:f64 = w;

        let rotation = RotationB {
            nodes: nodes,
            coords: coords,
            w_val: w,
        };

        return rotation;
    }
}
#[derive(Clone, PartialEq)]
pub struct RotationC {
    nodes: Vec<NodeIndex>,
    coords: Array1<f64>,
    w_val: f64,
}
impl RotationC {
    pub(crate) fn new(nodes: Vec<NodeIndex>, coords: Array1<f64>, w: f64) -> RotationC {
        let nodes: Vec<NodeIndex> = nodes;
        let coords: Array1<f64> = coords;
        let w:f64 = w;

        let rotation = RotationC {
            nodes: nodes,
            coords: coords,
            w_val: w,
        };

        return rotation;
    }
}
