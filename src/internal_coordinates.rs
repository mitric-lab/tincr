use crate::constants;
use crate::defaults;
use crate::gradients;
use crate::molecule::distance_matrix;
use crate::Molecule;
use approx::AbsDiffEq;
use itertools::Itertools;
use nalgebra::*;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::SVD;
use ndarray_linalg::*;
use peroxide::prelude::*;
use petgraph::algo::*;
use petgraph::data::*;
use petgraph::dot::{Config, Dot};
use petgraph::graph::*;
use petgraph::stable_graph::*;
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::ops::{AddAssign, Deref};

pub fn argsort(v: ArrayView1<f64>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(Ordering::Equal));
    idx
}

pub fn build_primitives(mol: &Molecule) -> InternalCoordinates {
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
    let mut distances_vec: Vec<Distance> = Vec::new();
    let mut angles_vec: Vec<Angle> = Vec::new();
    let mut outofplane_vec: Vec<Out_of_plane> = Vec::new();
    let mut dihedral_vec: Vec<Dihedral> = Vec::new();
    let mut cartesian_x_vec: Vec<CartesianX> = Vec::new();
    let mut cartesian_y_vec: Vec<CartesianY> = Vec::new();
    let mut cartesian_z_vec: Vec<CartesianZ> = Vec::new();
    let mut translation_x_vec: Vec<TranslationX> = Vec::new();
    let mut translation_y_vec: Vec<TranslationY> = Vec::new();
    let mut translation_z_vec: Vec<TranslationZ> = Vec::new();
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

            translation_x_vec.push(trans_x);
            translation_y_vec.push(trans_y);
            translation_z_vec.push(trans_z);

            let mut sel: Array2<f64> = coordinate_vector
                .clone()
                .into_shape((mol.n_atoms, 3))
                .unwrap()
                .slice(s![
                    node_vec[0].index()..node_vec.last().unwrap().index() + 1,
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
            let rot_a: RotationA = RotationA::new(node_vec.clone(), coordinate_vector.clone(), rg);
            let rot_b: RotationB = RotationB::new(node_vec.clone(), coordinate_vector.clone(), rg);
            let rot_c: RotationC = RotationC::new(node_vec.clone(), coordinate_vector.clone(), rg);

            rotation_a_Vec.push(rot_a);
            rotation_b_Vec.push(rot_b);
            rotation_c_Vec.push(rot_c);
        } else {
            for j in fragment.node_indices() {
                // add cartesian
                let cart_x: CartesianX = CartesianX::new(j.index(), 1.0);
                let cart_y: CartesianY = CartesianY::new(j.index(), 1.0);
                let cart_z: CartesianZ = CartesianZ::new(j.index(), 1.0);

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
    let mut index: usize = 0;
    let mut index_inner: usize = 0;
    let mut index_vec: Vec<Vec<NodeIndex>> = Vec::new();

    for b in mol.full_graph.node_indices() {
        for a in mol.full_graph.neighbors(b) {
            for c in mol.full_graph.neighbors(b) {
                if a.index() < c.index() {
                    let angl: Angle = Angle::new(a.index(), b.index(), c.index());
                    // nnc part doesnt work
                    println!("Angle for indices");
                    print!("Index 1: {:?}", a);
                    print!("  Index 2: {:?}", b);
                    println!("  Index 3: {:?}", c);
                    println!(
                        "value of angle: {:?}",
                        angl.clone().value(&coordinate_vector)
                    );
                    if angl.clone().value(&coordinate_vector).cos().abs() < linthre {
                        //let angl_ic = IC::angle(angl);
                        //internal_coords.push(angl_ic);
                        angles_vec.insert(index, angl);
                        index_inner += 1;
                        index_vec.insert(index, vec![a, b, c]);
                    }
                    // cant check for nnc
                }
            }
        }
        index += index_inner;
        index_inner = 0;
    }
    println!("Index vec {:?}", index_vec);
    //out of planes
    for b in mol.full_graph.node_indices() {
        for a in mol.full_graph.neighbors(b) {
            for c in mol.full_graph.neighbors(b) {
                for d in mol.full_graph.neighbors(b) {
                    if a.index() < c.index() && c.index() < d.index() {
                        // nc doesnt work
                        let it = vec![a.index(), c.index(), d.index()]
                            .into_iter()
                            .permutations(3);
                        for index in it.into_iter() {
                            let i = index[0];
                            let j = index[1];
                            let k = index[2];

                            println!("Indices Angle 1:");
                            print!("{:?}", b.index());
                            print!("{:?}", i);
                            println!("{:?}", j);

                            println!("Indices Angle 2:");
                            print!("{:?}", i);
                            print!("{:?}", j);
                            println!("{:?}", k);

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
                                let removed_angle: Angle = Angle::new(i, b.index(), j);
                                // delete angle i,b,j
                                for m in (0..angles_vec.len()).rev() {
                                    if (angles_vec[m].at_a == removed_angle.at_a)
                                        && (angles_vec[m].at_b == removed_angle.at_b)
                                        && (angles_vec[m].at_c == removed_angle.at_c)
                                    {
                                        angles_vec.remove(m);
                                    }
                                }
                                // out of plane bijk
                                let out_of_pl1: Out_of_plane =
                                    Out_of_plane::new(b.index(), i, j, k);
                                // let out_of_pl_ic = IC::out_of_plane(out_of_pl1);
                                // internal_coords.push(out_of_pl_ic);
                                outofplane_vec.push(out_of_pl1);
                                break;
                            }
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
    let mut atom_lines_new: Vec<Vec<NodeIndex>> = atom_lines;
    let mut convergence_1: bool = false;
    let mut convergence_2: bool = false;

    while true {
        atom_lines = atom_lines_new.clone();
        atom_lines_new = Vec::new();

        for aline in atom_lines {
            aline_new = aline.clone();
            let ab: NodeIndex = aline[0];
            let ay: NodeIndex = *aline.last().unwrap();

            for aa in mol.full_graph.neighbors(ab) {
                if aline_new.contains(&aa) == false {
                    //if aa != ab && aa != ay {
                    // If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                    // Add AA to the front of the list
                    let mut val_vector: Vec<f64> = Vec::new();
                    for ac in aline[1..].iter() {
                        if *ac != ab {
                            let angl = Angle::new(aa.index(), ab.index(), ac.index());
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
                    if indices_values.len() == val_vector.len() && val_vector.len() != 0 {
                        aline_new.insert(0, aa);
                    } else {
                        convergence_1 = true;
                    }
                } else {
                    convergence_1 = true;
                }
            }
            for az in mol.full_graph.neighbors(ay) {
                if aline_new.contains(&az) == false {
                    //if az != ab && az != ay {
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
                    if indices_values.len() == val_vector.len() && val_vector.len() != 0 {
                        aline_new.push(az);
                    } else {
                        convergence_2 = true;
                    }
                } else {
                    convergence_2 = true;
                }
            }
            atom_lines_new.push(aline_new.clone());
        }
        if convergence_1 == true && convergence_2 == true {
            break;
        }
    }

    let mut index_vec: Vec<Vec<NodeIndex>> = Vec::new();

    // dihedrals
    for aline in atom_lines_new {
        //Go over ALL pairs of atoms in a line
        for vec in aline.clone().into_iter().combinations(2) {
            let b_new: NodeIndex = vec[0];
            let c_new: NodeIndex = vec[1];
            let mut b: NodeIndex = b_new;
            let mut c: NodeIndex = c_new;

            if b.index() > c.index() {
                b = c_new;
                c = b_new;
            }
            println!("Combinations");
            print!("{}", b.index());
            println!("{}", c.index());
            for a in mol.full_graph.neighbors(b) {
                for d in mol.full_graph.neighbors(c) {
                    if aline.contains(&a) == false && aline.contains(&d) == false && a != d {
                        println!("Indices Dihedral");
                        print!("{}", a.index());
                        print!("{}", b.index());
                        print!("{}", c.index());
                        println!("{}", d.index());

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
                        dihedral_vec.insert(0, dihedral);
                        index_vec.insert(0, vec![a, b, c, d]);
                    }
                }
            }
        }
    }

    println!("Indices Dihedrals");
    println!("{:?}", index_vec);

    let internal_coordinates: InternalCoordinates = InternalCoordinates::new(
        distances_vec,
        angles_vec,
        outofplane_vec,
        dihedral_vec,
        cartesian_x_vec,
        cartesian_y_vec,
        cartesian_z_vec,
        translation_x_vec,
        translation_y_vec,
        translation_z_vec,
        rotation_a_Vec,
        rotation_b_Vec,
        rotation_c_Vec,
    );

    return internal_coordinates;
}

pub fn calculate_primitive_values(
    coords: Array1<f64>,
    internal_coords: &InternalCoordinates,
) -> Array1<f64> {
    let mut prim_vals: Vec<f64> = Vec::new();
    for i in &internal_coords.distance {
        let p_val: f64 = i.value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.angle {
        let p_val: f64 = i.value(&coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.out_of_plane {
        let p_val: f64 = i.value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.dihedral {
        let p_val: f64 = i.value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.cartesian_x {
        let p_val: f64 = i.clone().value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.cartesian_y {
        let p_val: f64 = i.clone().value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.cartesian_z {
        let p_val: f64 = i.clone().value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.translation_x {
        let p_val: f64 = i.clone().value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.translation_y {
        let p_val: f64 = i.clone().value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.translation_z {
        let p_val: f64 = i.clone().value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.rotation_a {
        let p_val: f64 = i.value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.rotation_b {
        let p_val: f64 = i.value(coords.clone());
        prim_vals.push(p_val);
    }
    for i in &internal_coords.rotation_c {
        let p_val: f64 = i.value(coords.clone());
        prim_vals.push(p_val);
    }
    let return_val: Array1<f64> = Array::from(prim_vals);

    return return_val;
}

pub fn calculate_internal_coordinate_gradient(
    coords: Array1<f64>,
    gradient: Array1<f64>,
    internal_coord_vector: Array1<f64>,
    internal_coords: &InternalCoordinates,
    dlc_mat: Array2<f64>,
) -> Array1<f64> {
    let g_inv: Array2<f64> = inverse_g_matrix(coords.clone(), internal_coords, dlc_mat.clone());
    let b_mat: Array2<f64> = wilsonB(&coords, internal_coords, true, Some(dlc_mat));
    let gq: Array1<f64> = g_inv.dot(&b_mat.dot(&gradient.t()));

    return gq;
}

pub fn calculate_internal_gradient_norm(grad: Array1<f64>) -> (f64, f64) {
    let grad_2d: Array2<f64> = grad.clone().into_shape((grad.len() / 3, 3)).unwrap();
    let atom_grad: Array1<f64> = grad_2d
        .mapv(|val| val.powi(2))
        .sum_axis(Axis(1))
        .mapv(|val| val.sqrt());
    let rms_gradient: f64 = atom_grad
        .clone()
        .mapv(|val| val.powi(2))
        .mean()
        .unwrap()
        .sqrt();
    let max_gradient: f64 = atom_grad
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
        .unwrap();

    return (rms_gradient, max_gradient);
}

pub fn inverse_g_matrix(
    coords: Array1<f64>,
    internal_coords: &InternalCoordinates,
    dlc_mat: Array2<f64>,
) -> Array2<f64> {
    let n_at: usize = coords.len() / 3;
    let coords_2d: Array2<f64> = coords.clone().into_shape((n_at, 3)).unwrap();

    let g_matrix: Array2<f64> = build_g_matrix(coords, internal_coords, true, Some(dlc_mat));

    let (u, s, vh) = g_matrix.svd(true, true).unwrap();
    let ut: Array2<f64> = u.unwrap().reversed_axes();
    let s: Array1<f64> = s;
    // s is okay
    let v: Array2<f64> = vh.unwrap().reversed_axes();

    let mut large_vals: usize = 0;
    let mut s_inv: Array1<f64> = Array::zeros((s.dim()));

    for (ival, value) in s.iter().enumerate() {
        if value.abs() > 1.0e-6 {
            large_vals += 1;
            s_inv[ival] = 1.0 / value;
        }
    }
    //println!("V matrix from svd {}",v.t());
    //println!("ut matrix from svd {}",ut.t());
    let s_inv_2d: Array2<f64> = Array::from_diag(&s_inv);
    let inv: Array2<f64> = v.dot(&s_inv_2d.dot(&ut));

    return inv;
}

pub fn calculate_internal_coordinate_vector(
    coords: Array1<f64>,
    internal_coords: &InternalCoordinates,
    dlc_mat: &Array2<f64>,
) -> Array1<f64> {
    let prim_values: Array1<f64> = calculate_primitive_values(coords, internal_coords);
    let dlc: Array1<f64> = dlc_mat.t().dot(&prim_values);

    return dlc;
}

pub fn build_delocalized_internal_coordinates(
    coords: Array1<f64>,
    primitives: &InternalCoordinates,
) -> Array2<f64> {
    // Build the delocalized internal coordinates (DLCs) which are linear
    // combinations of the primitive internal coordinates

    let g_matrix: Array2<f64> = build_g_matrix(coords, primitives, false, None);

    let (l_vec, q_mat): (Array1<f64>, Array2<f64>) = g_matrix.eigh(UPLO::Upper).unwrap();

    let mut large_val: usize = 0;
    let mut large_index_vec: Vec<usize> = Vec::new();
    for (ival, value) in l_vec.iter().enumerate() {
        if value.abs() > 1e-6 {
            large_val += 1;
            large_index_vec.push(ival);
        }
    }
    let mut qmat_final: Array2<f64> = Array::zeros((q_mat.dim().0, large_index_vec.len()));
    for (index, val) in large_index_vec.iter().enumerate() {
        qmat_final
            .slice_mut(s![.., index])
            .assign(&q_mat.slice(s![.., *val]));
    }

    return qmat_final;
}

pub fn build_g_matrix(
    coords: Array1<f64>,
    internal_coords: &InternalCoordinates,
    calculated_dlcs: bool,
    dlc_mat: Option<Array2<f64>>,
) -> Array2<f64> {
    // Given Cartesian coordinates xyz, return the G-matrix
    // given by G = BuBt where u is an arbitrary matrix (default to identity)
    let b_mat: Array2<f64> = wilsonB(&coords, internal_coords, calculated_dlcs, dlc_mat);
    let b_ubt: Array2<f64> = b_mat.dot(&b_mat.clone().t());

    return b_ubt;
}

pub fn wilsonB(
    coords: &Array1<f64>,
    internal_coords: &InternalCoordinates,
    calculated_dlcs: bool,
    dlc_mat: Option<Array2<f64>>,
) -> Array2<f64> {
    // Given Cartesian coordinates xyz, return the Wilson B-matrix
    // given by dq_i/dx_j where x is flattened (i.e. x1, y1, z1, x2, y2, z2)
    let derivatives: Vec<Array2<f64>> = get_derivatives(coords, internal_coords);
    let mut deriv_matrix: Array3<f64> = Array::zeros((
        derivatives.len(),
        derivatives[0].dim().0,
        derivatives[0].dim().1,
    ));

    for (index, val) in derivatives.iter().enumerate() {
        deriv_matrix.slice_mut(s![index, .., ..]).assign(val);
    }
    if calculated_dlcs {
        deriv_matrix = tensordot(&dlc_mat.unwrap(), &deriv_matrix, &[Axis(0)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();
    }
    // println!("derivatives");
    // for i in 0..derivatives.len() {
    //     println!("{:?}", derivatives[i]);
    // }
    let mut wilson_b: Array2<f64> = Array::zeros((
        deriv_matrix.dim().0,
        deriv_matrix.dim().1 * deriv_matrix.dim().2,
    ));
    for i in 0..deriv_matrix.dim().0 {
        let deriv_1d: Array1<f64> = deriv_matrix
            .slice(s![i, .., ..])
            .to_owned()
            .clone()
            .into_shape((deriv_matrix.dim().1 * deriv_matrix.dim().2))
            .unwrap();
        wilson_b.slice_mut(s![i, ..]).assign(&deriv_1d);
    }
    return wilson_b;
}

pub fn get_derivatives(
    coords: &Array1<f64>,
    internal_coords: &InternalCoordinates,
) -> Vec<Array2<f64>> {
    let mut derivatives: Vec<Array2<f64>> = Vec::new();
    for i in &internal_coords.distance {
        let deriv: Array2<f64> = i.derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.angle {
        let deriv: Array2<f64> = i.derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.out_of_plane {
        let deriv: Array2<f64> = i.derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.dihedral {
        let deriv: Array2<f64> = i.derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.cartesian_x {
        let deriv: Array2<f64> = i.clone().derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.cartesian_y {
        let deriv: Array2<f64> = i.clone().derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.cartesian_z {
        let deriv: Array2<f64> = i.clone().derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.translation_x {
        let deriv: Array2<f64> = i.clone().derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.translation_y {
        let deriv: Array2<f64> = i.clone().derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.translation_z {
        let deriv: Array2<f64> = i.clone().derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.rotation_a {
        let deriv: Array2<f64> = i.derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.rotation_b {
        let deriv: Array2<f64> = i.derivatives(coords.clone());
        derivatives.push(deriv);
    }
    for i in &internal_coords.rotation_c {
        let deriv: Array2<f64> = i.derivatives(coords.clone());
        derivatives.push(deriv);
    }
    return derivatives;
}

pub fn check_linearity(x: &Array2<f64>, y: &Array2<f64>) -> bool {
    let x: Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y: Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let f_mat: Array2<f64> = build_f_matrix(&x, &y);

    let n: usize = x.dim().0;
    let tmp: (Array1<f64>, Array2<f64>) = sorted_eigh_linearity(f_mat);
    let l_arr: Array1<f64> = tmp.0;
    let mut bool_return: bool = false;

    if ((l_arr[0] / l_arr[1]) < 1.01) && ((l_arr[0] / l_arr[1]) > 0.0) {
        bool_return = true;
    }
    return bool_return;
}

pub fn build_f_matrix(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let r_mat: Array2<f64> = build_correlation_for_f_mat(x, y);
    let mut f: Array2<f64> = Array::zeros((4, 4));
    let r_11: f64 = r_mat[[0, 0]];
    let r_12: f64 = r_mat[[0, 1]];
    let r_13: f64 = r_mat[[0, 2]];
    let r_21: f64 = r_mat[[1, 0]];
    let r_22: f64 = r_mat[[1, 1]];
    let r_23: f64 = r_mat[[1, 2]];
    let r_31: f64 = r_mat[[2, 0]];
    let r_32: f64 = r_mat[[2, 1]];
    let r_33: f64 = r_mat[[2, 2]];
    f[[0, 0]] = r_11 + r_22 + r_33;
    f[[0, 1]] = r_23 - r_32;
    f[[0, 2]] = r_31 - r_13;
    f[[0, 3]] = r_12 - r_21;
    f[[1, 0]] = r_23 - r_32;
    f[[1, 1]] = r_11 - r_22 - r_33;
    f[[1, 2]] = r_12 + r_21;
    f[[1, 3]] = r_13 + r_31;
    f[[2, 0]] = r_31 - r_13;
    f[[2, 1]] = r_12 + r_21;
    f[[2, 2]] = r_22 - r_33 - r_11;
    f[[2, 3]] = r_23 + r_32;
    f[[3, 0]] = r_12 - r_21;
    f[[3, 1]] = r_13 + r_31;
    f[[3, 2]] = r_23 + r_32;
    f[[3, 3]] = r_33 - r_22 - r_11;

    return f;
}

pub fn build_correlation_for_f_mat(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let xmat: Array2<f64> = x.clone().reversed_axes();
    let ymat: Array2<f64> = y.clone().reversed_axes();

    let dot: Array2<f64> = xmat.dot(&ymat.t());
    return dot;
}

pub fn sorted_eigh_linearity(f_mat: Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    // Return eigenvalues and eigenvectors of a symmetric matrix
    // in descending order and associated eigenvectors.
    // This is just a convenience function to get eigenvectors
    // in descending or ascending order as desired.
    let tmp: (Array1<f64>, Array2<f64>) = f_mat.eigh(UPLO::Upper).unwrap();
    let l_vec: Array1<f64> = tmp.0;
    let q_mat: Array2<f64> = tmp.1;

    let idx: Vec<usize> = argsort(l_vec.view());
    let mut l_new: Vec<f64> = Vec::new();
    let mut q_mat_new: Array2<f64> = Array::zeros((q_mat.dim().0, q_mat.dim().1));
    for (i, j) in idx.iter().rev().enumerate() {
        l_new.push(l_vec[*j]);
        q_mat_new
            .slice_mut(s![.., i])
            .assign(&q_mat.slice(s![.., *j]));
    }
    let l_arr: Array1<f64> = Array::from(l_new);

    return (l_arr, q_mat_new);
}

pub fn get_quat_rot(x: &Array2<f64>, y: &Array2<f64>) -> (Array1<f64>, f64) {
    let x: Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y: Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let f_mat: Array2<f64> = build_f_matrix(&x, &y);

    let n: usize = x.dim().0;
    let tmp: (Array1<f64>, Array2<f64>) = sorted_eigh_linearity(f_mat);
    let l_vec: Array1<f64> = tmp.0;
    let q_mat: Array2<f64> = tmp.1;

    let mut q_vec: Array1<f64> = q_mat.slice(s![.., 0]).to_owned();
    if q_vec[0] < 0.0 {
        q_vec = q_vec * (-1.0);
    }
    return (q_vec, l_vec[0]);
}

pub fn get_exmap_rot(x: &Array2<f64>, y: &Array2<f64>) -> Array1<f64> {
    let (q_vec, l_vec): (Array1<f64>, f64) = get_quat_rot(x, y);
    let (fac, dfac): (f64, f64) = calc_fac_dfac_rot(q_vec[0]);
    let v: Array1<f64> = fac * q_vec.slice(s![1..]).to_owned();

    return v;
}

pub fn calc_fac_dfac_rot(q0: f64) -> (f64, f64) {
    let qm1 = q0 - 1.0;
    let mut fac: f64 = 0.0;
    let mut dfac: f64 = 0.0;

    if qm1.abs() < 1e-8 {
        fac = 2.0 - 2.0 * qm1 / 3.0;
        dfac = -2.0 / 3.0;
    } else {
        let s: f64 = (1.0 - q0.powi(2)).sqrt();
        let a: f64 = q0.acos();
        fac = 2.0 * a / s;
        dfac = -2.0 / s.powi(2);
        dfac += 2.0 * q0 * a / s.powi(3);
    }
    return (fac, dfac);
}

pub fn get_r_deriv(x: &Array2<f64>, y: &Array2<f64>) -> Array4<f64> {
    // Calculate the derivatives of the correlation matrix with respect
    // to the Cartesian coordinates.
    let x: Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y: Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let mut a_diff_r: Array4<f64> = Array::zeros((x.dim().0, 3, 3, 3));
    for u in 0..x.dim().0 {
        for w in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    if i == w {
                        a_diff_r[[u, w, i, j]] = y[[u, j]];
                    }
                }
            }
        }
    }
    return a_diff_r;
}

pub fn build_f_matrix_deriv(x: &Array2<f64>, y: &Array2<f64>) -> Array4<f64> {
    // Calculate the derivatives of the F-matrix with respect
    // to the Cartesian coordinates.
    let x: Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y: Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let dr: Array4<f64> = get_r_deriv(&x, &y);
    let mut df: Array4<f64> = Array::zeros((x.dim().0, 3, 4, 4));
    for u in 0..x.dim().0 {
        for w in 0..3 {
            let dr_11: f64 = dr[[u, w, 0, 0]];
            let dr_12: f64 = dr[[u, w, 0, 1]];
            let dr_13: f64 = dr[[u, w, 0, 2]];
            let dr_21: f64 = dr[[u, w, 1, 0]];
            let dr_22: f64 = dr[[u, w, 1, 1]];
            let dr_23: f64 = dr[[u, w, 1, 2]];
            let dr_31: f64 = dr[[u, w, 2, 0]];
            let dr_32: f64 = dr[[u, w, 2, 1]];
            let dr_33: f64 = dr[[u, w, 2, 2]];
            df[[u, w, 0, 0]] = dr_11 + dr_22 + dr_33;
            df[[u, w, 0, 1]] = dr_23 - dr_32;
            df[[u, w, 0, 2]] = dr_31 - dr_13;
            df[[u, w, 0, 3]] = dr_12 - dr_21;
            df[[u, w, 1, 0]] = dr_23 - dr_32;
            df[[u, w, 1, 1]] = dr_11 - dr_22 - dr_33;
            df[[u, w, 1, 2]] = dr_12 + dr_21;
            df[[u, w, 1, 3]] = dr_13 + dr_31;
            df[[u, w, 2, 0]] = dr_31 - dr_13;
            df[[u, w, 2, 1]] = dr_12 + dr_21;
            df[[u, w, 2, 2]] = dr_22 - dr_33 - dr_11;
            df[[u, w, 2, 3]] = dr_23 + dr_32;
            df[[u, w, 3, 0]] = dr_12 - dr_21;
            df[[u, w, 3, 1]] = dr_13 + dr_31;
            df[[u, w, 3, 2]] = dr_12 + dr_32;
            df[[u, w, 3, 3]] = dr_33 - dr_22 - dr_11;
        }
    }
    return df;
}

pub fn invert_svd(x_mat: &Array2<f64>, thresh: Option<f64>) -> Array2<f64> {
    // Invert a matrix using singular value decomposition.
    // @param[in] X The 2-D NumPy array containing the matrix to be inverted
    // @param[in] thresh The SVD threshold; eigenvalues below this are not inverted but set to zero
    // @return Xt The 2-D NumPy array containing the inverted matrix
    let thresh: f64 = thresh.unwrap_or(1e-12);
    let (u, s, vh) = x_mat.clone().svd(true, true).unwrap();
    let uh: Array2<f64> = u.unwrap().reversed_axes();
    let mut s: Array1<f64> = s;
    let v: Array2<f64> = vh.unwrap().reversed_axes();
    for i in 0..s.len() {
        if s[i].abs() > thresh {
            s[i] = 1.0 / s[i];
        } else {
            s[i] = 0.0;
        }
    }
    let si: Array2<f64> = Array::from_diag(&s);
    let x_t: Array2<f64> = v.dot(&si.dot(&uh));
    return x_t;
}

pub fn get_q_der_rot(x: &Array2<f64>, y: &Array2<f64>) -> Array3<f64> {
    let x: Array2<f64> = x.clone() - x.mean_axis(Axis(0)).unwrap();
    let y: Array2<f64> = y.clone() - y.mean_axis(Axis(0)).unwrap();

    let (q_vec, l_val): (Array1<f64>, f64) = get_quat_rot(&x, &y);
    let f_mat: Array2<f64> = build_f_matrix(&x, &y);
    let f_mat_deriv: Array4<f64> = build_f_matrix_deriv(&x, &y);
    let mat: Array2<f64> = &Array::eye(4) * l_val - &f_mat;

    let tmp: Array2<f64> = mat.clone();
    let m_inv: Array2<f64> = invert_svd(&tmp, Some(1e-6));
    let mut dq: Array3<f64> = Array::zeros((x.dim().0, 3, 4));
    for u in 0..x.dim().0 {
        for w in 0..3 {
            let f_temp: Array2<f64> = f_mat_deriv.slice(s![u, w, .., ..]).to_owned();
            let dquw: Array1<f64> = m_inv.dot(&(f_temp.dot(&q_vec.clone().reversed_axes())));
            dq.slice_mut(s![u, w, ..]).assign(&dquw);
        }
    }
    return dq;
}

pub fn get_exmap_deriv_rot(x: &Array2<f64>, y: &Array2<f64>) -> Array3<f64> {
    // Given trial coordinates x and target coordinates y,
    // return the derivatives of the exponential map that brings
    // x into maximal coincidence (minimum RMSD) with y, with
    // respect to the coordinates of x.
    let (q_vec, l_vec): (Array1<f64>, f64) = get_quat_rot(x, y);
    let v: Array1<f64> = get_exmap_rot(x, y);
    let (fac, dfac): (f64, f64) = calc_fac_dfac_rot(q_vec[0]);

    let mut dvdq: Array2<f64> = Array::zeros((4, 3));
    dvdq.slice_mut(s![0, ..])
        .assign(&(dfac * q_vec.slice(s![1..]).to_owned()));
    for i in (0..3) {
        dvdq[[i + 1, i]] = fac;
    }
    let dqdx = get_q_der_rot(&x, &y);
    let mut dvdx: Array3<f64> = Array::zeros((x.dim().0, 3, 3));
    for u in 0..x.dim().0 {
        for w in 0..3 {
            let dqdx_uw: Array1<f64> = dqdx.slice(s![u, w, ..]).to_owned();
            for p in 0..4 {
                dvdx.slice_mut(s![u, w, ..])
                    .add_assign(&(dvdq.slice(s![p, ..]).to_owned() * dqdx[[u, w, p]]));
            }
        }
    }

    return dvdx;
}

pub fn d_cross(vec_1: &Vec<f64>, vec_2: &Vec<f64>) -> Array2<f64> {
    // Given two vectors a and b, return the gradient of the cross product axb w/r.t. a.
    // (Note that the answer is independent of a.)
    // Derivative is on the first axis.
    let mut dcross: Array2<f64> = Array::zeros((3, 3));
    for i in 0..3 {
        let mut ei: Vec<f64> = Array::zeros(3).to_vec();
        ei[i] = 1.0;
        dcross
            .slice_mut(s![i, ..])
            .assign(&Array::from(ei.cross(vec_2)));
    }
    return dcross;
}

pub fn dn_cross(vec_1: &Vec<f64>, vec_2: &Vec<f64>) -> Array1<f64> {
    // Return the gradient of the norm of the cross product w/r.t. a
    let ncross: f64 = (vec_1.cross(vec_2)).norm();
    let term_1: Array1<f64> =
        Array::from(vec_1.clone()) * Array::from(vec_2.clone()).dot(&Array::from(vec_2.clone()));
    let term_2: Array1<f64> = -1.0
        * Array::from(vec_2.clone())
        * Array::from(vec_1.clone()).dot(&Array::from(vec_2.clone()));
    let result: Array1<f64> = (&term_1 + &term_2) / ncross;
    return result;
}

pub fn cartesian_from_step(cart_coords:Array1<f64>,dy:Array1<f64>,internal_coords:&InternalCoordinates,dlc_mat:Array2<f64>){
    let mut microiter:usize = 0;
    let ndqs:Vec<f64> = Vec::new();
    let rmsds:Vec<f64> = Vec::new();
    let damp:f64 = 1.0;
    let mut fail_counter:usize = 0;

    let mut dq:Array1<f64> = dy;
    let mut xyz:Array1<f64> = cart_coords.clone();

    while true{
        microiter += 1;
        let b_mat:Array2<f64> = wilsonB(&cart_coords,internal_coords,true,Some(dlc_mat.clone()));
        let g_inv:Array2<f64> = inverse_g_matrix(cart_coords.clone(),internal_coords,dlc_mat.clone());
        let dxyz:Array1<f64> = damp * b_mat.t().dot(&g_inv.dot(&dq.t()));
        let xyz_2:Array1<f64> = xyz.clone() + dxyz;

        // let dq_actual:Array1<f64> = calc diff between xyz_2 and xyz
        // calcDiff is needed for every internal coordinate
        // let rmsd:f64 = (xyz_2 - xyz).mapv(|val|val.powi(2)).mean().unwrap().sqrt();
        // let ndq:f64 = (dq . dq_actual).to_vec().norm();
    }
}

#[derive(Clone, PartialEq)]
pub struct InternalCoordinates {
    distance: Vec<Distance>,
    angle: Vec<Angle>,
    out_of_plane: Vec<Out_of_plane>,
    dihedral: Vec<Dihedral>,
    cartesian_x: Vec<CartesianX>,
    cartesian_y: Vec<CartesianY>,
    cartesian_z: Vec<CartesianZ>,
    translation_x: Vec<TranslationX>,
    translation_y: Vec<TranslationY>,
    translation_z: Vec<TranslationZ>,
    rotation_a: Vec<RotationA>,
    rotation_b: Vec<RotationB>,
    rotation_c: Vec<RotationC>,
}
impl InternalCoordinates {
    pub(crate) fn new(
        distance: Vec<Distance>,
        angle: Vec<Angle>,
        out_of_plane: Vec<Out_of_plane>,
        dihedral: Vec<Dihedral>,
        cartesian_x: Vec<CartesianX>,
        cartesian_y: Vec<CartesianY>,
        cartesian_z: Vec<CartesianZ>,
        translation_x: Vec<TranslationX>,
        translation_y: Vec<TranslationY>,
        translation_z: Vec<TranslationZ>,
        rotation_a: Vec<RotationA>,
        rotation_b: Vec<RotationB>,
        rotation_c: Vec<RotationC>,
    ) -> InternalCoordinates {
        let internal_coords = InternalCoordinates {
            distance: distance,
            angle: angle,
            out_of_plane: out_of_plane,
            dihedral: dihedral,
            cartesian_x: cartesian_x,
            cartesian_y: cartesian_y,
            cartesian_z: cartesian_z,
            translation_x: translation_x,
            translation_y: translation_y,
            translation_z: translation_z,
            rotation_a: rotation_a,
            rotation_b: rotation_b,
            rotation_c: rotation_c,
        };
        return internal_coords;
    }
}

#[derive(Clone, PartialEq)]
pub struct CartesianX {
    at_a: usize,
    w_val: f64,
}
impl CartesianX {
    pub(crate) fn new(at_a: usize, w_val: f64) -> CartesianX {
        let at_a: usize = at_a;
        let w_val: f64 = w_val;

        let cart = CartesianX {
            at_a: at_a,
            w_val: w_val,
        };
        return cart;
    }

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let diff:f64 = self.value(coords_1) - self.value(coords_2);
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let val: f64 = coords_new[[self.at_a, 0]] * self.w_val;
        return val;
    }
    pub fn derivatives(self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));
        derivatives[[self.at_a, 0]] = self.w_val;

        return derivatives;
    }
}

#[derive(Clone, PartialEq)]
pub struct CartesianY {
    at_a: usize,
    w_val: f64,
}
impl CartesianY {
    pub(crate) fn new(at_a: usize, w_val: f64) -> CartesianY {
        let at_a: usize = at_a;
        let w_val: f64 = w_val;

        let cart = CartesianY {
            at_a: at_a,
            w_val: w_val,
        };
        return cart;
    }

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let diff:f64 = self.value(coords_1) - self.value(coords_2);
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let val: f64 = coords_new[[self.at_a, 1]] * self.w_val;
        return val;
    }

    pub fn derivatives(self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));
        derivatives[[self.at_a, 1]] = self.w_val;

        return derivatives;
    }
}

#[derive(Clone, PartialEq)]
pub struct CartesianZ {
    at_a: usize,
    w_val: f64,
}
impl CartesianZ {
    pub(crate) fn new(at_a: usize, w_val: f64) -> CartesianZ {
        let at_a: usize = at_a;
        let w_val: f64 = w_val;

        let cart = CartesianZ {
            at_a: at_a,
            w_val: w_val,
        };
        return cart;
    }

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let diff:f64 = self.value(coords_1) - self.value(coords_2);
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let val: f64 = coords_new[[self.at_a, 2]] * self.w_val;
        return val;
    }

    pub fn derivatives(self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));
        derivatives[[self.at_a, 2]] = self.w_val;

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

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let diff:f64 = self.value(coords_1) - self.value(coords_2);
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut val: f64 = 0.0;

        for (i, index) in self.nodes.iter().enumerate() {
            val += coords_new[[index.index(), 0]] * self.w_vec[i];
        }
        return val;
    }

    pub fn derivatives(self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));

        for (i, a) in self.nodes.iter().enumerate() {
            derivatives[[a.index(), 0]] = self.w_vec[i];
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

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let diff:f64 = self.value(coords_1) - self.value(coords_2);
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut val: f64 = 0.0;

        for (i, index) in self.nodes.iter().enumerate() {
            val += coords_new[[index.index(), 1]] * self.w_vec[i];
        }
        return val;
    }

    pub fn derivatives(self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));

        for (i, a) in self.nodes.iter().enumerate() {
            derivatives[[a.index(), 1]] = self.w_vec[i];
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

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let diff:f64 = self.value(coords_1) - self.value(coords_2);
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut val: f64 = 0.0;

        for (i, index) in self.nodes.iter().enumerate() {
            val += coords_new[[index.index(), 2]] * self.w_vec[i];
        }
        return val;
    }

    pub fn derivatives(self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));

        for (i, a) in self.nodes.iter().enumerate() {
            derivatives[[a.index(), 2]] = self.w_vec[i];
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

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let diff:f64 = self.value(coords_1) - self.value(coords_2);
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let distance: f64 = (coords_new.slice(s![self.at_a, ..]).to_owned()
            - coords_new.slice(s![self.at_b, ..]).to_owned())
        .mapv(|dist| dist.powi(2))
        .sum()
        .sqrt();
        return distance;
    }

    pub fn derivatives(&self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));

        let m: usize = self.at_a;
        let n: usize = self.at_b;
        let u: Array1<f64> = (coords_new.slice(s![m, ..]).to_owned()
            - coords_new.slice(s![n, ..]).to_owned())
            / (coords_new.slice(s![m, ..]).to_owned() - coords_new.slice(s![n, ..]).to_owned())
                .to_vec()
                .norm();
        derivatives.slice_mut(s![m, ..]).assign(&u);
        derivatives.slice_mut(s![n, ..]).assign(&-&u);

        return derivatives;
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

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let mut diff:f64 = self.value(coords_1) - self.value(coords_2);

        let plus_2_pi:f64 = diff + 2.0_f64 * PI;
        let minus_2_pi:f64 = diff - 2.0_f64 * PI;

        if diff.abs() > plus_2_pi.abs(){
            diff = plus_2_pi;
        }
        if diff.abs() > minus_2_pi.abs(){
            diff = minus_2_pi;
        }
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let a: usize = self.at_a;
        let b: usize = self.at_b;
        let c: usize = self.at_c;
        let d: usize = self.at_d;

        let vec_1: Vec<f64> = (coords_new.slice(s![b, ..]).to_owned()
            - coords_new.slice(s![a, ..]).to_owned())
        .to_vec();
        let vec_2: Vec<f64> = (coords_new.slice(s![c, ..]).to_owned()
            - coords_new.slice(s![b, ..]).to_owned())
        .to_vec();
        let vec_3: Vec<f64> = (coords_new.slice(s![d, ..]).to_owned()
            - coords_new.slice(s![c, ..]).to_owned())
        .to_vec();

        let cross_1: Array1<f64> = Array::from(vec_2.cross(&vec_3));
        let cross_2: Array1<f64> = Array::from(vec_1.cross(&vec_2));

        let arg_1: f64 = (&Array::from(vec_1) * &cross_1).sum()
            * Array::from(vec_2).mapv(|val| val.powi(2)).sum().sqrt();
        let arg_2: f64 = (&cross_1 * &cross_2).sum();

        let return_val: f64 = arg_1.atan2(arg_2);

        return return_val;
    }

    pub fn derivatives(&self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));
        let m: usize = self.at_a;
        let o: usize = self.at_b;
        let p: usize = self.at_c;
        let n: usize = self.at_d;

        // unit displacement vectors
        let u_prime: Array1<f64> =
            coords_new.slice(s![m, ..]).to_owned() - coords_new.slice(s![o, ..]).to_owned();
        let u_norm: f64 = u_prime.clone().to_vec().norm();
        let v_prime: Array1<f64> =
            coords_new.slice(s![n, ..]).to_owned() - coords_new.slice(s![p, ..]).to_owned();
        let v_norm: f64 = v_prime.clone().to_vec().norm();
        let w_prime: Array1<f64> =
            coords_new.slice(s![p, ..]).to_owned() - coords_new.slice(s![o, ..]).to_owned();
        let w_norm: f64 = w_prime.to_vec().norm();
        let u: Array1<f64> = u_prime / u_norm;
        let w: Array1<f64> = w_prime / w_norm;
        let v: Array1<f64> = v_prime / v_norm;

        let mut term_1: Array1<f64> = Array::zeros(3);
        let mut term_2: Array1<f64> = Array::zeros(3);
        let mut term_3: Array1<f64> = Array::zeros(3);
        let mut term_4: Array1<f64> = Array::zeros(3);

        if (1.0 - u.dot(&w).powi(2)) < 1e-6 {
            term_1 = Array::from(u.to_vec().cross(&w.to_vec())) * 0.0;
            term_3 = Array::from(u.to_vec().cross(&w.to_vec())) * 0.0;
        } else {
            term_1 =
                Array::from(u.to_vec().cross(&w.to_vec())) / (u_norm * (1.0 - u.dot(&w).powi(2)));
            term_3 = Array::from(u.to_vec().cross(&w.to_vec())) * u.dot(&w)
                / (w_norm * (1.0 - u.dot(&w).powi(2)));
        }
        if (1.0 - v.dot(&w).powi(2)) < 1e-6 {
            term_2 = Array::from(v.to_vec().cross(&w.to_vec())) * 0.0;
            term_4 = Array::from(v.to_vec().cross(&w.to_vec())) * 0.0;
        } else {
            term_2 =
                Array::from(v.to_vec().cross(&w.to_vec())) / (v_norm * (1.0 - v.dot(&w).powi(2)));
            term_4 = Array::from(v.to_vec().cross(&w.to_vec())) * v.dot(&w)
                / (w_norm * (1.0 - v.dot(&w).powi(2)));
        }

        derivatives.slice_mut(s![m, ..]).assign(&term_1);
        derivatives.slice_mut(s![n, ..]).assign(&(-&term_2));
        derivatives
            .slice_mut(s![o, ..])
            .assign(&(-&term_1 + &term_3 - &term_4));
        derivatives
            .slice_mut(s![p, ..])
            .assign(&(&term_2 - &term_3 + &term_4));

        return derivatives;
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

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let diff:f64 = self.value(&coords_1) - self.value(&coords_2);
        return diff;
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
        // let norm_1: f64 = vec_1.norm();
        // let norm_2: f64 = vec_2.norm();
        let vec_1_new: Array1<f64> = Array::from(vec_1.clone());
        let vec_2_new: Array1<f64> = Array::from(vec_2.clone());
        let norm_1: f64 = (vec_1_new.mapv(|vec_1_new| vec_1_new.powi(2))).sum().sqrt();
        let norm_2: f64 = (vec_2_new.mapv(|vec_2_new| vec_2_new.powi(2))).sum().sqrt();
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

    pub fn derivatives(&self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));
        let m: usize = self.at_a;
        let o: usize = self.at_b;
        let n: usize = self.at_c;

        // unit displacement vectors
        let u_prime: Array1<f64> =
            coords_new.slice(s![m, ..]).to_owned() - coords_new.slice(s![o, ..]).to_owned();
        let u_norm: f64 = u_prime.clone().to_vec().norm();
        let v_prime: Array1<f64> =
            coords_new.slice(s![n, ..]).to_owned() - coords_new.slice(s![o, ..]).to_owned();
        let v_norm: f64 = v_prime.clone().to_vec().norm();
        let u: Array1<f64> = u_prime / u_norm;
        let v: Array1<f64> = v_prime / v_norm;

        let vector_1: Array1<f64> = (array![1.0, -1.0, 1.0] / 3.0.sqrt());
        let vector_2: Array1<f64> = (array![-1.0, 1.0, 1.0] / 3.0.sqrt());

        let mut w_prime: Vec<f64> = Vec::new();
        if (&u + &v).to_vec().norm() < 1e-10 || (&u - &v).to_vec().norm() < 1e-10 {
            if (&u + &vector_1).to_vec().norm() < 1e-10 || (&u - &vector_2).to_vec().norm() < 1e-10
            {
                w_prime = u.to_vec().cross(&vector_2.to_vec());
            } else {
                w_prime = u.to_vec().cross(&vector_1.to_vec());
            }
        } else {
            w_prime = u.to_vec().cross(&v.to_vec());
        }
        let w: Array1<f64> = Array::from(w_prime.clone()) / w_prime.norm();
        let term_1: Array1<f64> = Array::from(u.to_vec().cross(&w.to_vec())) / u_norm;
        let term_2: Array1<f64> = Array::from(w.to_vec().cross(&v.to_vec())) / v_norm;

        derivatives.slice_mut(s![m, ..]).assign(&term_1);
        derivatives.slice_mut(s![n, ..]).assign(&term_2);
        derivatives
            .slice_mut(s![o, ..])
            .assign(&(-(&term_1 + &term_2)));

        return derivatives;
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

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>)->f64{
        let mut diff:f64 = self.value(coords_1) - self.value(coords_2);

        let plus_2_pi:f64 = diff + 2.0_f64 * PI;
        let minus_2_pi:f64 = diff - 2.0_f64 * PI;

        if diff.abs() > plus_2_pi.abs(){
            diff = plus_2_pi;
        }
        if diff.abs() > minus_2_pi.abs(){
            diff = minus_2_pi;
        }
        return diff;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let a: usize = self.at_a;
        let b: usize = self.at_b;
        let c: usize = self.at_c;
        let d: usize = self.at_d;

        let vec_1: Vec<f64> = (coords_new.slice(s![b, ..]).to_owned()
            - coords_new.slice(s![a, ..]).to_owned())
        .to_vec();
        let vec_2: Vec<f64> = (coords_new.slice(s![c, ..]).to_owned()
            - coords_new.slice(s![b, ..]).to_owned())
        .to_vec();
        let vec_3: Vec<f64> = (coords_new.slice(s![d, ..]).to_owned()
            - coords_new.slice(s![c, ..]).to_owned())
        .to_vec();

        let cross_1: Array1<f64> = Array::from(vec_2.cross(&vec_3));
        let cross_2: Array1<f64> = Array::from(vec_1.cross(&vec_2));

        let arg_1: f64 = (&Array::from(vec_1) * &cross_1).sum()
            * Array::from(vec_2).mapv(|val| val.powi(2)).sum().sqrt();
        let arg_2: f64 = (&cross_1 * &cross_2).sum();

        let return_val: f64 = arg_1.atan2(arg_2);

        return return_val;
    }

    pub fn derivatives(&self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array2<f64> = Array::zeros((n_at, 3));
        let m: usize = self.at_a;
        let o: usize = self.at_b;
        let p: usize = self.at_c;
        let n: usize = self.at_d;

        // unit displacement vectors
        let u_prime: Array1<f64> =
            coords_new.slice(s![m, ..]).to_owned() - coords_new.slice(s![o, ..]).to_owned();
        let u_norm: f64 = u_prime.clone().to_vec().norm();
        let v_prime: Array1<f64> =
            coords_new.slice(s![n, ..]).to_owned() - coords_new.slice(s![p, ..]).to_owned();
        let v_norm: f64 = v_prime.clone().to_vec().norm();
        let w_prime: Array1<f64> =
            coords_new.slice(s![p, ..]).to_owned() - coords_new.slice(s![o, ..]).to_owned();
        let w_norm: f64 = w_prime.to_vec().norm();
        let u: Array1<f64> = u_prime / u_norm;
        let w: Array1<f64> = w_prime / w_norm;
        let v: Array1<f64> = v_prime / v_norm;

        let mut term_1: Array1<f64> = Array::zeros(3);
        let mut term_2: Array1<f64> = Array::zeros(3);
        let mut term_3: Array1<f64> = Array::zeros(3);
        let mut term_4: Array1<f64> = Array::zeros(3);

        if (1.0 - u.dot(&w).powi(2)) < 1e-6 {
            term_1 = Array::from(u.to_vec().cross(&w.to_vec())) * 0.0;
            term_3 = Array::from(u.to_vec().cross(&w.to_vec())) * 0.0;
        } else {
            term_1 =
                Array::from(u.to_vec().cross(&w.to_vec())) / (u_norm * (1.0 - u.dot(&w).powi(2)));
            term_3 = Array::from(u.to_vec().cross(&w.to_vec())) * u.dot(&w)
                / (w_norm * (1.0 - u.dot(&w).powi(2)));
        }
        if (1.0 - v.dot(&w).powi(2)) < 1e-6 {
            term_2 = Array::from(v.to_vec().cross(&w.to_vec())) * 0.0;
            term_4 = Array::from(v.to_vec().cross(&w.to_vec())) * 0.0;
        } else {
            term_2 =
                Array::from(v.to_vec().cross(&w.to_vec())) / (v_norm * (1.0 - v.dot(&w).powi(2)));
            term_4 = Array::from(v.to_vec().cross(&w.to_vec())) * v.dot(&w)
                / (w_norm * (1.0 - v.dot(&w).powi(2)));
        }

        derivatives.slice_mut(s![m, ..]).assign(&term_1);
        derivatives.slice_mut(s![n, ..]).assign(&(-&term_2));
        derivatives
            .slice_mut(s![o, ..])
            .assign(&(-&term_1 + &term_3 - &term_4));
        derivatives
            .slice_mut(s![p, ..])
            .assign(&(&term_2 - &term_3 + &term_4));

        return derivatives;
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
        let w: f64 = w;

        let rotation = RotationA {
            nodes: nodes,
            coords: coords,
            w_val: w,
        };

        return rotation;
    }

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>){
        let vec_1:Array1<f64> = self.value_vec(coords_1);
        let vec_2:Array1<f64> = self.value_vec(coords_2);
        let vec_diff:Array1<f64> = calc_rot_vec_diff(vec_1,vec_2);

        let return_val:f64 = vec_diff[0] * self.w_val;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut answer: Array1<f64> = get_exmap_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            answer = get_exmap_rot(&x_sel_new, &y_sel_new);
        }

        let return_val: f64 = answer[0] * self.w_val;

        return return_val;
    }

    pub fn value_vec(&self, coords: Array1<f64>) -> Array1<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut answer: Array1<f64> = get_exmap_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            answer = get_exmap_rot(&x_sel_new, &y_sel_new);
        }

        return answer;
    }

    pub fn calc_e0(&self) -> (Vec<f64>) {
        let n_at: usize = self.coords.clone().len() / 3;
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let vy: Array1<f64> = &y_sel.slice(s![y_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
        let ev: Array1<f64> = &vy / vy.to_vec().norm();
        let ex: Vec<f64> = vec![1.0, 0.0, 0.0];
        let ey: Vec<f64> = vec![0.0, 1.0, 0.0];
        let ez: Vec<f64> = vec![0.0, 0.0, 1.0];
        let e_full: Array1<Vec<f64>> = array![ex, ey, ez];
        let mut dots: Vec<f64> = Vec::new();

        for i in e_full.iter() {
            dots.push((Array::from(i.clone()).dot(&ev)).powi(2));
        }
        let sorted_arr: Vec<usize> = argsort(Array::from(dots).view());
        let min_index: usize = sorted_arr[0];
        let mut e0: Vec<f64> = vy.to_vec().cross(&e_full[min_index]);
        e0 = (Array::from(e0.clone()) / e0.norm()).to_vec();
        return e0;
    }

    pub fn derivatives(&self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array3<f64> = Array::zeros((n_at, 3, 3));

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut deriv_raw: Array3<f64> = get_exmap_deriv_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            deriv_raw = get_exmap_deriv_rot(&x_sel_new, &y_sel_new);

            let draw_dim: usize = deriv_raw.clone().dim().0;

            let nxdum: f64 = xdum.norm();
            let dxdum: Array2<f64> = d_cross(&vx.to_vec(), &e0);
            let dnxdum: Array1<f64> = dn_cross(&vx.to_vec(), &e0);
            let dexdum: Array2<f64> = (dxdum * nxdum
                - einsum("i,j->ij", &[&dnxdum, &Array::from(xdum)])
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap());
            let tmp_slice: Array2<f64> =
                deriv_raw.clone().slice(s![draw_dim - 1, .., ..]).to_owned();
            deriv_raw
                .slice_mut(s![0, .., ..])
                .add_assign(&(-1.0 * dexdum.dot(&tmp_slice.clone())));

            for i in 0..self.nodes.len() {
                deriv_raw.slice_mut(s![i, .., ..]).add_assign(
                    &(Array::eye(3).dot(&tmp_slice.clone()) / (self.nodes.len() as f64)),
                );
            }
            deriv_raw
                .slice_mut(s![draw_dim - 2, .., ..])
                .add_assign(&dexdum.dot(&tmp_slice.clone()));
            let mut deriv_raw_new: Array3<f64> = Array::zeros(deriv_raw.shape())
                .into_dimensionality::<Ix3>()
                .unwrap();

            // change order of first dim of deriv_raw
            for i in 0..draw_dim {
                deriv_raw_new
                    .slice_mut(s![i, .., ..])
                    .assign(&deriv_raw.slice(s![draw_dim - 1 - i, .., ..]));
            }
            deriv_raw = deriv_raw_new;
        }

        for (i, a) in self.nodes.iter().enumerate() {
            derivatives
                .slice_mut(s![a.index(), .., ..])
                .assign(&deriv_raw.slice(s![i, .., ..]));
        }
        let derivatives_new: Array2<f64> = derivatives.slice(s![.., .., 0]).to_owned() * self.w_val;

        return derivatives_new;
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
        let w: f64 = w;

        let rotation = RotationB {
            nodes: nodes,
            coords: coords,
            w_val: w,
        };

        return rotation;
    }

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>){
        let vec_1:Array1<f64> = self.value_vec(coords_1);
        let vec_2:Array1<f64> = self.value_vec(coords_2);
        let vec_diff:Array1<f64> = calc_rot_vec_diff(vec_1,vec_2);

        let return_val:f64 = vec_diff[1] * self.w_val;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut answer: Array1<f64> = get_exmap_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            answer = get_exmap_rot(&x_sel_new, &y_sel_new);
        }

        let return_val: f64 = answer[1] * self.w_val;

        return return_val;
    }

    pub fn value_vec(&self, coords: Array1<f64>) -> Array1<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut answer: Array1<f64> = get_exmap_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            answer = get_exmap_rot(&x_sel_new, &y_sel_new);
        }

        return answer;
    }

    pub fn calc_e0(&self) -> (Vec<f64>) {
        let n_at: usize = self.coords.clone().len() / 3;
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let vy: Array1<f64> = &y_sel.slice(s![y_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
        let ev: Array1<f64> = &vy / vy.to_vec().norm();
        let ex: Vec<f64> = vec![1.0, 0.0, 0.0];
        let ey: Vec<f64> = vec![0.0, 1.0, 0.0];
        let ez: Vec<f64> = vec![0.0, 0.0, 1.0];
        let e_full: Array1<Vec<f64>> = array![ex, ey, ez];
        let mut dots: Vec<f64> = Vec::new();

        for i in e_full.iter() {
            dots.push((Array::from(i.clone()).dot(&ev)).powi(2));
        }
        let sorted_arr: Vec<usize> = argsort(Array::from(dots).view());
        let min_index: usize = sorted_arr[0];
        let mut e0: Vec<f64> = vy.to_vec().cross(&e_full[min_index]);
        e0 = (Array::from(e0.clone()) / e0.norm()).to_vec();
        return e0;
    }

    pub fn derivatives(&self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array3<f64> = Array::zeros((n_at, 3, 3));

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut deriv_raw: Array3<f64> = get_exmap_deriv_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            deriv_raw = get_exmap_deriv_rot(&x_sel_new, &y_sel_new);

            let draw_dim: usize = deriv_raw.clone().dim().0;

            let nxdum: f64 = xdum.norm();
            let dxdum: Array2<f64> = d_cross(&vx.to_vec(), &e0);
            let dnxdum: Array1<f64> = dn_cross(&vx.to_vec(), &e0);
            let dexdum: Array2<f64> = (dxdum * nxdum
                - einsum("i,j->ij", &[&dnxdum, &Array::from(xdum)])
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap());
            let tmp_slice: Array2<f64> =
                deriv_raw.clone().slice(s![draw_dim - 1, .., ..]).to_owned();
            deriv_raw
                .slice_mut(s![0, .., ..])
                .add_assign(&(-1.0 * dexdum.dot(&tmp_slice.clone())));

            for i in 0..self.nodes.len() {
                deriv_raw.slice_mut(s![i, .., ..]).add_assign(
                    &(Array::eye(3).dot(&tmp_slice.clone()) / (self.nodes.len() as f64)),
                );
            }
            deriv_raw
                .slice_mut(s![draw_dim - 2, .., ..])
                .add_assign(&dexdum.dot(&tmp_slice.clone()));
            let mut deriv_raw_new: Array3<f64> = Array::zeros(deriv_raw.shape())
                .into_dimensionality::<Ix3>()
                .unwrap();

            // change order of first dim of deriv_raw
            for i in 0..draw_dim {
                deriv_raw_new
                    .slice_mut(s![i, .., ..])
                    .assign(&deriv_raw.slice(s![draw_dim - 1 - i, .., ..]));
            }
            deriv_raw = deriv_raw_new;
        }

        for (i, a) in self.nodes.iter().enumerate() {
            derivatives
                .slice_mut(s![a.index(), .., ..])
                .assign(&deriv_raw.slice(s![i, .., ..]));
        }
        let derivatives_new: Array2<f64> = derivatives.slice(s![.., .., 1]).to_owned() * self.w_val;

        return derivatives_new;
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
        let w: f64 = w;

        let rotation = RotationC {
            nodes: nodes,
            coords: coords,
            w_val: w,
        };

        return rotation;
    }

    pub fn calc_diff(&self,coords_1:Array1<f64>,coords_2:Array1<f64>){
        let vec_1:Array1<f64> = self.value_vec(coords_1);
        let vec_2:Array1<f64> = self.value_vec(coords_2);
        let vec_diff:Array1<f64> = calc_rot_vec_diff(vec_1,vec_2);

        let return_val:f64 = vec_diff[2] * self.w_val;
    }

    pub fn value(&self, coords: Array1<f64>) -> f64 {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut answer: Array1<f64> = get_exmap_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            answer = get_exmap_rot(&x_sel_new, &y_sel_new);
        }

        let return_val: f64 = answer[2] * self.w_val;

        return return_val;
    }

    pub fn value_vec(&self, coords: Array1<f64>) -> Array1<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut answer: Array1<f64> = get_exmap_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            answer = get_exmap_rot(&x_sel_new, &y_sel_new);
        }

        return answer;
    }

    pub fn calc_e0(&self) -> (Vec<f64>) {
        let n_at: usize = self.coords.clone().len() / 3;
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let vy: Array1<f64> = &y_sel.slice(s![y_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
        let ev: Array1<f64> = &vy / vy.to_vec().norm();
        let ex: Vec<f64> = vec![1.0, 0.0, 0.0];
        let ey: Vec<f64> = vec![0.0, 1.0, 0.0];
        let ez: Vec<f64> = vec![0.0, 0.0, 1.0];
        let e_full: Array1<Vec<f64>> = array![ex, ey, ez];
        let mut dots: Vec<f64> = Vec::new();

        for i in e_full.iter() {
            dots.push((Array::from(i.clone()).dot(&ev)).powi(2));
        }
        let sorted_arr: Vec<usize> = argsort(Array::from(dots).view());
        let min_index: usize = sorted_arr[0];
        let mut e0: Vec<f64> = vy.to_vec().cross(&e_full[min_index]);
        e0 = (Array::from(e0.clone()) / e0.norm()).to_vec();
        return e0;
    }

    pub fn derivatives(&self, coords: Array1<f64>) -> Array2<f64> {
        let n_at: usize = coords.len() / 3;
        let coords_new: Array2<f64> = coords.into_shape((n_at, 3)).unwrap();
        let coords_self: Array2<f64> = self.coords.clone().into_shape((n_at, 3)).unwrap();
        let mut derivatives: Array3<f64> = Array::zeros((n_at, 3, 3));

        let mut x_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        let mut y_sel: Array2<f64> = Array::zeros((self.nodes.len(), 3));
        for (i, j) in self.nodes.iter().enumerate() {
            x_sel
                .slice_mut(s![i, ..])
                .assign(&coords_new.slice(s![j.index(), ..]));
            y_sel
                .slice_mut(s![i, ..])
                .assign(&coords_self.slice(s![j.index(), ..]));
        }
        let x_mean: Array1<f64> = x_sel.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y_sel.mean_axis(Axis(0)).unwrap();

        let mut bool_linear: bool = false;

        if check_linearity(&x_sel, &y_sel) {
            bool_linear = true;
        }
        let mut deriv_raw: Array3<f64> = get_exmap_deriv_rot(&x_sel, &y_sel);

        if bool_linear {
            let vx: Array1<f64> = &x_sel.slice(s![x_sel.dim().0, ..]) - &x_sel.slice(s![0, ..]);
            let vy: Array1<f64> = &y_sel.slice(s![x_sel.dim().0, ..]) - &y_sel.slice(s![0, ..]);
            let e0: Vec<f64> = self.calc_e0();
            let xdum: Vec<f64> = vx.to_vec().cross(&e0);
            let ydum: Vec<f64> = vy.to_vec().cross(&e0);
            let exdum: Array1<f64> = (Array::from(xdum.clone()) / xdum.norm());
            let eydum: Array1<f64> = (Array::from(ydum.clone()) / ydum.norm());

            let mut x_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            let mut y_sel_new: Array2<f64> = Array::zeros((self.nodes.len() + 1, 3));
            // vstacks
            x_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&x_sel);
            y_sel_new
                .slice_mut(s![0..self.nodes.len() - 1, ..])
                .assign(&y_sel);
            x_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(exdum + x_mean));
            y_sel_new
                .slice_mut(s![self.nodes.len(), ..])
                .assign(&(eydum + y_mean));

            deriv_raw = get_exmap_deriv_rot(&x_sel_new, &y_sel_new);

            let draw_dim: usize = deriv_raw.clone().dim().0;

            let nxdum: f64 = xdum.norm();
            let dxdum: Array2<f64> = d_cross(&vx.to_vec(), &e0);
            let dnxdum: Array1<f64> = dn_cross(&vx.to_vec(), &e0);
            let dexdum: Array2<f64> = (dxdum * nxdum
                - einsum("i,j->ij", &[&dnxdum, &Array::from(xdum)])
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap());
            let tmp_slice: Array2<f64> =
                deriv_raw.clone().slice(s![draw_dim - 1, .., ..]).to_owned();
            deriv_raw
                .slice_mut(s![0, .., ..])
                .add_assign(&(-1.0 * dexdum.dot(&tmp_slice.clone())));

            for i in 0..self.nodes.len() {
                deriv_raw.slice_mut(s![i, .., ..]).add_assign(
                    &(Array::eye(3).dot(&tmp_slice.clone()) / (self.nodes.len() as f64)),
                );
            }
            deriv_raw
                .slice_mut(s![draw_dim - 2, .., ..])
                .add_assign(&dexdum.dot(&tmp_slice.clone()));
            let mut deriv_raw_new: Array3<f64> = Array::zeros(deriv_raw.shape())
                .into_dimensionality::<Ix3>()
                .unwrap();

            // change order of first dim of deriv_raw
            for i in 0..draw_dim {
                deriv_raw_new
                    .slice_mut(s![i, .., ..])
                    .assign(&deriv_raw.slice(s![draw_dim - 1 - i, .., ..]));
            }
            deriv_raw = deriv_raw_new;
        }

        for (i, a) in self.nodes.iter().enumerate() {
            derivatives
                .slice_mut(s![a.index(), .., ..])
                .assign(&deriv_raw.slice(s![i, .., ..]));
        }
        let derivatives_new: Array2<f64> = derivatives.slice(s![.., .., 2]).to_owned() * self.w_val;

        return derivatives_new;
    }
}

pub fn calc_rot_vec_diff(vec_1:Array1<f64>,vec_2:Array1<f64>)->Array1<f64>{
    // Calculate the difference in two provided rotation vectors v1_in - v2_in
    let v1:Vec<f64> = vec_1.to_vec();
    let v2:Vec<f64> = vec_2.to_vec();
    let length:usize = vec_1.len();

    let mut return_val:Array1<f64> = Array::zeros(length);

    if v1.norm() < 1e-6 && v2.norm() < 1e-6{
        return_val = &vec_1 - &vec_2;
    }
    let mut va:Array1<f64> = Array::zeros(length);
    let mut vb:Array1<f64> = Array::zeros(length);
    let mut va_is_v1:bool = false;

    if vec_1.dot(&vec_1) > vec_2.dot(&vec_2){
        va = vec_1;
        vb = vec_2;
        va_is_v1 = true;
    }
    else{
        va = vec_2;
        vb = vec_1;
    }
    let vh:Array1<f64> = va.clone() / va.clone().to_vec().norm();
    let mut revcount:usize = 0;

    while true{
        let vd:f64 = (&va-&vb).dot(&(&va-&vb));
        va = va + 2.0_f64 * PI * vh.clone();
        revcount += 1;
        if (&va-&vb).dot(&(&va-&vb)) > vd{
            va = va - 2.0_f64 * PI * vh.clone();
            revcount -= 1;
            break;
        }
    }
    while true{
        let vd:f64 = (&va-&vb).dot(&(&va-&vb));
        va = va - 2.0_f64 * PI * vh.clone();
        revcount -= 1;
        if (&va-&vb).dot(&(&va-&vb)) > vd{
            va = va + 2.0_f64 * PI * vh.clone();
            revcount += 1;
            break;
        }
    }
    let mut return_val:Array1<f64> = Array::zeros(length);
    if va_is_v1{
        return_val = &va - &vb;
    }
    else{
        return_val = &vb - &va;
    }
    return return_val;
}

#[test]
pub fn test_make_primitives() {
    let atomic_numbers: Vec<u8> = vec![6, 6, 1, 1, 1, 1];
    let mut positions: Array2<f64> = array![
        [-0.7575800000, 0.0000000000, -0.0000000000],
        [0.7575800000, 0.0000000000, 0.0000000000],
        [-1.2809200000, 0.9785000000, -0.0000000000],
        [-1.2809200000, -0.9785000000, 0.0000000000],
        [1.2809200000, -0.9785000000, -0.0000000000],
        [1.2809200000, 0.9785000000, 0.0000000000]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule =
        Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    let internal_coordinates: InternalCoordinates = build_primitives(&mol);
}

#[test]
pub fn test_build_gmatrix() {
    let atomic_numbers: Vec<u8> = vec![6, 6, 1, 1, 1, 1];
    let mut positions: Array2<f64> = array![
        [-0.7575800000, 0.0000000000, -0.0000000000],
        [0.7575800000, 0.0000000000, 0.0000000000],
        [-1.2809200000, 0.9785000000, -0.0000000000],
        [-1.2809200000, -0.9785000000, 0.0000000000],
        [1.2809200000, -0.9785000000, -0.0000000000],
        [1.2809200000, 0.9785000000, 0.0000000000]
    ];
    // transform coordinates in au
    positions = positions * 1.8897261278504418;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions.clone(),
        charge,
        multiplicity,
        None,
        None,
    );

    let internal_coordinates: InternalCoordinates = build_primitives(&mol);

    let coordinates_1d: Array1<f64> = positions.clone().into_shape(mol.n_atoms * 3).unwrap();

    // let g_matrix:Array2<f64> = build_g_matrix(coordinates_1d.clone(),&internal_coordinates);
    //
    // println!("Gmatrix");
    // for i in 0..g_matrix.dim().0{
    //     println!("{:?}",g_matrix.slice(s![i,..]));
    // }

    let q_mat: Array2<f64> =
        build_delocalized_internal_coordinates(coordinates_1d.clone(), &internal_coordinates);

    let q_internal: Array1<f64> =
        calculate_internal_coordinate_vector(coordinates_1d, &internal_coordinates, &q_mat);

    println!("q_internal: {:?}", q_internal);

    assert!(1 == 2);
}

#[test]
pub fn test_internal_coordinate_gradient() {
    let atomic_numbers: Vec<u8> = vec![6, 6, 1, 1, 1, 1];
    let mut positions: Array2<f64> = array![
        [-0.7575800000, 0.0000000000, -0.0000000000],
        [0.7575800000, 0.0000000000, 0.0000000000],
        [-1.2809200000, 0.9785000000, -0.0000000000],
        [-1.2809200000, -0.9785000000, 0.0000000000],
        [1.2809200000, -0.9785000000, -0.0000000000],
        [1.2809200000, 0.9785000000, 0.0000000000]
    ];
    // transform coordinates in au
    positions = positions * 1.8897261278504418;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions.clone(),
        charge,
        multiplicity,
        None,
        None,
    );

    let input_gradient: Array1<f64> = array![
        -0.2053378, 0., -0., 0.2053378, -0., 0., -0.0037439, 0.025855, -0., -0.0037439, -0.025855,
        0., 0.0037439, -0.025855, -0., 0.0037439, 0.025855, -0.
    ];

    let gradient_ref: Array1<f64> = array![
        -3.30076111e-17,
        -2.54335801e-16,
        -7.42617391e-17,
        1.97525723e-16,
        -1.74612938e-16,
        5.72171395e-02,
        3.22493572e-16,
        -1.27575900e-01,
        5.11402705e-17,
        -5.32899069e-17,
        9.69124822e-02,
        -4.12639859e-19,
        -6.56895453e-17,
        1.15575994e-17,
        2.50169279e-17,
        7.29185310e-02,
        -1.18933990e-01,
        -1.17062344e-17
    ];

    let coordinates_1d: Array1<f64> = positions.clone().into_shape(mol.n_atoms * 3).unwrap();
    let internal_coordinates: InternalCoordinates = build_primitives(&mol);

    let q_mat: Array2<f64> =
        build_delocalized_internal_coordinates(coordinates_1d.clone(), &internal_coordinates);

    let q_internal_ref: Array1<f64> = array![
        8.88178420e-16,
        4.09397937e-16,
        5.29106653e-16,
        -8.79253055e-16,
        -1.26513229e-15,
        -2.55154855e+00,
        -8.99118363e-18,
        -3.58738579e+00,
        1.32047184e-16,
        3.10862447e-15,
        4.67979294e+00,
        3.14159265e+00,
        -1.77635684e-15,
        2.26651913e-16,
        4.44089210e-16,
        1.51891374e+00,
        -1.68037480e-01,
        -3.32286896e-16
    ];

    //let q_internal: Array1<f64> =
    //    calculate_internal_coordinates(coordinates_1d.clone(), &internal_coordinates, &q_mat);

    let inter_coord_gradient: Array1<f64> = calculate_internal_coordinate_gradient(
        coordinates_1d,
        input_gradient,
        q_internal_ref,
        &internal_coordinates,
        q_mat,
    );

    println!("gradient {:?}", inter_coord_gradient);
    assert!(inter_coord_gradient
        .mapv(|val| val.abs())
        .abs_diff_eq(&gradient_ref.mapv(|val| val.abs()), 1e-7));

    assert!(1 == 2);
}

#[test]
pub fn test_svd() {
    let test_matrix: Array2<f64> = array![[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0],];
    println!("test_matirx {}", test_matrix);

    let (u, s, vh) = test_matrix.svd(true, true).unwrap();
    let u: Array2<f64> = u.unwrap();
    let s: Array1<f64> = s;
    // s is okay
    println!("U matrix from svd {}", u);
    println!("S matrix from svd {}", s);
    let vh: Array2<f64> = vh.unwrap();
    println!("Vh matrix from svd {}", vh);

    println!(
        "eigenvectors of AA.T {}",
        (&test_matrix * &test_matrix.t())
            .eigh(UPLO::Upper)
            .unwrap()
            .1
    );

    let v: Array2<f64> = vh.reversed_axes();
    let ut: Array2<f64> = u.reversed_axes();
    let s_diag: Array2<f64> = Array::from_diag(&(1.0 / s));

    let inv: Array2<f64> = v.dot(&s_diag.dot(&ut));

    println!("inv matrix {}", inv);
    assert!(1 == 2);
}
