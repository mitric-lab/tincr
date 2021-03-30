use crate::calculator::{
    get_gamma_gradient_matrix, get_gamma_matrix, get_only_gamma_matrix_atomwise,
    import_pseudo_atom, Calculator, DFTBCalculator,
};
use crate::constants;
use crate::constants::VDW_RADII;
use crate::defaults;
use crate::gradients::*;
use crate::gradients::{get_gradients, ToOwnedF};
use crate::graph::*;
use crate::h0_and_s::h0_and_s;
use crate::h0_and_s::h0_and_s_gradients;
use crate::internal_coordinates::*;
use crate::io::GeneralConfig;
use crate::molecule::distance_matrix;
use crate::molecule::get_atomtypes;
use crate::parameters::*;
use crate::scc_routine;
use crate::solver::get_exc_energies;
use crate::Molecule;
use approx::AbsDiffEq;
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use petgraph::algo::{is_isomorphic, is_isomorphic_matching};
use petgraph::dot::{Config, Dot};
use petgraph::stable_graph::*;
use petgraph::{Graph, Undirected};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::AddAssign;
use std::time::Instant;

pub fn fmo_gs_gradients(
    fragments: &Vec<Molecule>,
    frag_grad_results: &Vec<frag_grad_result>,
    pair_results: &Vec<pair_grad_result>,
    indices_frags: &Vec<usize>,
    gamma_total: Array2<f64>,
    g1_total: Array3<f64>,
    prox_mat: Array2<bool>,
) -> (Array1<f64>) {
    // sum over all monomer energies
    let mut grad_e0_monomers: Vec<f64> = Vec::new();
    let mut grad_vrep_monomers: Vec<f64> = Vec::new();

    for frag in frag_grad_results.iter() {
        grad_e0_monomers.append(&mut frag.grad_e0.clone().to_vec());
        grad_vrep_monomers.append(&mut frag.grad_vrep.clone().to_vec());
    }

    // get energy term for pairs
    let mut iter: usize = 0;
    let mut pair_energies: f64 = 0.0;
    let mut embedding_potential: f64 = 0.0;

    let proximity_zeros: Array1<f64> = prox_mat
        .iter()
        .filter_map(|&item| if item == true { Some(1.0) } else { Some(0.0) })
        .collect();
    let gamma_zeros: Array2<f64> = proximity_zeros.into_shape((prox_mat.raw_dim())).unwrap();
    //let gamma_tmp: Array2<f64> = gamma_zeros * gamma_total;
    let gamma_tmp: Array2<f64> = 1.0 * gamma_total;
    let mut dimer_gradients: Vec<Array1<f64>> = Vec::new();
    let mut embedding_gradients: Vec<Array1<f64>> = Vec::new();

    for (pair_index, pair) in pair_results.iter().enumerate() {
        if pair.energy_pair.is_some() {
            let dimer_gradient_e0: Array1<f64> = pair.grad_e0.clone().unwrap();
            let dimer_gradient_vrep: Array1<f64> = pair.grad_vrep.clone().unwrap();

            let dimer_pmat: Array2<f64> = pair.p_mat.clone().unwrap();

            let pair_atoms: usize =
                fragments[pair.frag_a_index].n_atoms + fragments[pair.frag_b_index].n_atoms;
            let pair_charges = pair.pair_charges.clone().unwrap();
            let pair_smat: Array2<f64> = pair.s.clone().unwrap();
            let pair_grads: Array3<f64> = pair.grad_s.clone().unwrap();
            let ddq_vec: Vec<f64> = (0..pair_atoms)
                .into_iter()
                .map(|a| {
                    let mut ddq: f64 = 0.0;
                    if a < pair.frag_a_atoms {
                        ddq = pair_charges[a] - fragments[pair.frag_a_index].final_charges[a];
                    } else {
                        ddq = pair_charges[a]
                            - fragments[pair.frag_b_index].final_charges[a - pair.frag_a_atoms];
                    }
                    ddq
                })
                .collect();

            let index_pair_iter: usize = indices_frags[pair.frag_a_index];
            let ddq_arr: Array1<f64> = Array::from(ddq_vec);
            // TODO:Reduce cloning by doing let ... = ...,clone() and using that expression
            let shape_orbs_dimer: usize = pair.grad_s.clone().unwrap().dim().1;
            let shape_orbs_a: usize = frag_grad_results[pair.frag_a_index].grad_s.dim().1;
            let shape_orbs_b: usize = frag_grad_results[pair.frag_b_index].grad_s.dim().1;
            //let w_dimer: Array3<f64> = -0.5
            //    * dimer_pmat
            //        .clone()
            //        .dot(
            //            &pair
            //                .grad_s
            //                .clone()
            //                .unwrap()
            //                .into_shape((3 * pair_atoms * shape_orbs_dimer, shape_orbs_dimer))
            //                .unwrap()
            //                .dot(&dimer_pmat.clone())
            //                .t(),
            //        )
            //        .t()
            //        .to_owned()
            //        .into_shape((3 * pair_atoms, shape_orbs_dimer, shape_orbs_dimer))
            //        .unwrap();

            let w_dimer: Array3<f64> = -0.5*
                &pair
                    .grad_s
                    .clone()
                    .unwrap()
                    .into_shape((3 * pair_atoms * shape_orbs_dimer, shape_orbs_dimer))
                    .unwrap()
                    .dot(&dimer_pmat.clone())
                    .dot(&dimer_pmat
                        .clone())
                .into_shape((3 * pair_atoms, shape_orbs_dimer, shape_orbs_dimer))
                .unwrap();

            let w_mat_a: Array3<f64> = -0.5*
                &frag_grad_results[pair.frag_a_index]
                    .grad_s
                    .clone()
                    .into_shape((
                        3 * fragments[pair.frag_a_index].n_atoms * shape_orbs_a,
                        shape_orbs_a,
                    ))
                    .unwrap()
                    .dot(&fragments[pair.frag_a_index].final_p_matrix)
                    .dot(&fragments[pair.frag_a_index]
                        .final_p_matrix)
                    .into_shape((
                        3 * fragments[pair.frag_a_index].n_atoms,
                        shape_orbs_a,
                        shape_orbs_a,
                    ))
                    .unwrap();

            let w_mat_b: Array3<f64> = -0.5*
                &frag_grad_results[pair.frag_b_index]
                    .grad_s
                    .clone()
                    .into_shape((
                        3 * fragments[pair.frag_b_index].n_atoms * shape_orbs_b,
                        shape_orbs_b,
                    ))
                    .unwrap()
                    .dot(&fragments[pair.frag_b_index].final_p_matrix)
                    .dot(&fragments[pair.frag_b_index]
                        .final_p_matrix)
                    .into_shape((
                        3 * fragments[pair.frag_b_index].n_atoms,
                        shape_orbs_b,
                        shape_orbs_b,
                    ))
                    .unwrap();

            //let w_mat_a: Array3<f64> = -0.5
            //    * fragments[pair.frag_a_index]
            //        .final_p_matrix
            //        .dot(
            //            &frag_grad_results[pair.frag_a_index]
            //                .grad_s
            //                .clone()
            //                .into_shape((
            //                    3 * fragments[pair.frag_a_index].n_atoms * shape_orbs_a,
            //                    shape_orbs_a,
            //                ))
            //                .unwrap()
            //                .dot(&fragments[pair.frag_a_index].final_p_matrix)
            //                .t(),
            //        )
            //        .t()
            //        .to_owned()
            //        .into_shape((
            //            3 * fragments[pair.frag_a_index].n_atoms,
            //            shape_orbs_a,
            //            shape_orbs_a,
            //        ))
            //        .unwrap();

            //let w_mat_b: Array3<f64> = -0.5
            //    * fragments[pair.frag_b_index]
            //        .final_p_matrix
            //        .dot(
            //            &frag_grad_results[pair.frag_b_index]
            //                .grad_s
            //                .clone()
            //                .into_shape((
            //                    3 * fragments[pair.frag_b_index].n_atoms * shape_orbs_b,
            //                    shape_orbs_b,
            //                ))
            //                .unwrap()
            //                .dot(&fragments[pair.frag_b_index].final_p_matrix)
            //                .t(),
            //        )
            //        .t()
            //        .to_owned()
            //        .into_shape((
            //            3 * fragments[pair.frag_b_index].n_atoms,
            //            shape_orbs_b,
            //            shape_orbs_b,
            //        ))
            //        .unwrap();

            // Build delta p_mu,nu^I,J
            let mut dp_direct_sum_monomer: Array2<f64> = Array2::zeros(dimer_pmat.raw_dim());
            let p_dim_monomer: usize = fragments[pair.frag_a_index].final_p_matrix.dim().0;
            dp_direct_sum_monomer
                .slice_mut(s![0..p_dim_monomer, 0..p_dim_monomer])
                .assign(&fragments[pair.frag_a_index].final_p_matrix);
            dp_direct_sum_monomer
                .slice_mut(s![p_dim_monomer.., p_dim_monomer..])
                .assign(&fragments[pair.frag_b_index].final_p_matrix);
            let dp_dimer: Array2<f64> = &dimer_pmat - &dp_direct_sum_monomer;

            // Build delta W_mu,nu^I,J
            let mut dw_dimer: Array3<f64> = Array3::zeros(w_dimer.raw_dim());
            let w_dimer_dim: usize = w_dimer.dim().1;
            let mut dw_dimer_vec: Vec<Array1<f64>> = (0..3 * pair_atoms)
                .into_iter()
                .map(|a| {
                    let mut w_a_dimer: Array2<f64> = Array::zeros((w_dimer_dim, w_dimer_dim));
                    if a < 3 * pair.frag_a_atoms {
                        w_a_dimer
                            .slice_mut(s![0..p_dim_monomer, 0..p_dim_monomer])
                            .assign(&w_mat_a.slice(s![a, .., ..]));
                    } else {
                        w_a_dimer
                            .slice_mut(s![p_dim_monomer.., p_dim_monomer..])
                            .assign(&w_mat_b.slice(s![a - 3 * pair.frag_a_atoms, .., ..]));
                    }
                    let w_return: Array2<f64> = w_dimer.slice(s![a, .., ..]).to_owned() - w_a_dimer;
                    let w_flat: Array1<f64> =
                        w_return.into_shape((w_dimer_dim * w_dimer_dim)).unwrap();
                    w_flat
                })
                .collect();

            // transform dW from flat array to 3d array
            let mut dw_dimer_vec_flat: Vec<f64> = Vec::new();
            for a in dw_dimer_vec.iter() {
                dw_dimer_vec_flat.append(&mut a.to_vec());
            }
            let dw_dimer: Array3<f64> = Array::from(dw_dimer_vec_flat)
                .into_shape((3 * pair_atoms, w_dimer_dim, w_dimer_dim))
                .unwrap();

            let embedding_pot: Vec<Array1<f64>> = fragments
                .iter()
                .enumerate()
                .filter_map(|(ind_k, mol_k)| {
                    if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index {
                        let index_frag_iter: usize = indices_frags[ind_k];
                        let dgamma_ac: Array3<f64> = g1_total
                            .slice(s![
                                index_pair_iter..index_pair_iter + 3 * pair_atoms,
                                index_pair_iter..index_pair_iter + pair_atoms,
                                index_frag_iter..index_frag_iter + mol_k.n_atoms
                            ])
                            .to_owned();

                        let term_1: Array1<f64> = dgamma_ac
                            .clone()
                            .into_shape((3 * pair_atoms * pair_atoms, mol_k.n_atoms))
                            .unwrap()
                            .dot(&mol_k.final_charges)
                            .into_shape((3 * pair_atoms, pair_atoms))
                            .unwrap()
                            .dot(&ddq_arr);

                        let dw_s_a: Array1<f64> = dw_dimer
                            .clone()
                            .into_shape((3 * pair_atoms * w_dimer_dim, w_dimer_dim))
                            .unwrap()
                            .dot(&pair_smat)
                            .into_shape((3 * pair_atoms, w_dimer_dim, w_dimer_dim))
                            .unwrap()
                            .sum_axis(Axis(2))
                            .sum_axis(Axis(1));

                        let dp_grads_a: Array1<f64> = dp_dimer
                            .dot(
                                &pair_grads
                                    .clone()
                                    .into_shape((3 * pair_atoms * w_dimer_dim, w_dimer_dim))
                                    .unwrap()
                                    .t(),
                            )
                            .t()
                            .to_owned()
                            .into_shape((3 * pair_atoms, w_dimer_dim, w_dimer_dim))
                            .unwrap()
                            .sum_axis(Axis(2))
                            .sum_axis(Axis(1));

                        let term_2: Array1<f64> = (dw_s_a + dp_grads_a)
                            * gamma_tmp
                                .slice(s![
                                    index_pair_iter..index_pair_iter + pair_atoms,
                                    index_frag_iter..index_frag_iter + mol_k.n_atoms
                                ])
                                .dot(&fragments[ind_k].final_charges)
                                .sum();

                        let embedding_part_1: Array1<f64> = term_1 + term_2;

                        let dgamma_ac: Array3<f64> = g1_total
                            .slice(s![
                                index_frag_iter..index_frag_iter + 3 * mol_k.n_atoms,
                                index_pair_iter..index_pair_iter + pair_atoms,
                                index_frag_iter..index_frag_iter + mol_k.n_atoms
                            ])
                            .to_owned();

                        let term_1_k: Array1<f64> = dgamma_ac
                            .into_shape((3 * mol_k.n_atoms * pair_atoms, mol_k.n_atoms))
                            .unwrap()
                            .dot(&mol_k.final_charges)
                            .into_shape((3 * mol_k.n_atoms, pair_atoms))
                            .unwrap()
                            .dot(&ddq_arr);

                        let shape_orbs_k: usize = frag_grad_results[ind_k].grad_s.dim().1;
                        let w_mat_k: Array3<f64> = -0.5
                            * fragments[ind_k]
                                .final_p_matrix
                                .dot(
                                    &frag_grad_results[ind_k]
                                        .grad_s
                                        .clone()
                                        .into_shape((
                                            3 * fragments[ind_k].n_atoms * shape_orbs_k,
                                            shape_orbs_k,
                                        ))
                                        .unwrap()
                                        .dot(&fragments[ind_k].final_p_matrix)
                                        .t(),
                                )
                                .t()
                                .to_owned()
                                .into_shape((
                                    3 * fragments[ind_k].n_atoms,
                                    shape_orbs_k,
                                    shape_orbs_k,
                                ))
                                .unwrap();

                        let w_s_k: Array1<f64> = w_mat_k
                            .into_shape((3 * fragments[ind_k].n_atoms * shape_orbs_k, shape_orbs_k))
                            .unwrap()
                            .dot(&frag_grad_results[ind_k].s)
                            .into_shape((3 * fragments[ind_k].n_atoms, shape_orbs_k, shape_orbs_k))
                            .unwrap()
                            .sum_axis(Axis(2))
                            .sum_axis(Axis(1));

                        let p_grads_k: Array1<f64> = fragments[ind_k]
                            .final_p_matrix
                            .dot(
                                &frag_grad_results[ind_k]
                                    .grad_s
                                    .clone()
                                    .into_shape((
                                        3 * fragments[ind_k].n_atoms * shape_orbs_k,
                                        shape_orbs_k,
                                    ))
                                    .unwrap()
                                    .t(),
                            )
                            .t()
                            .to_owned()
                            .into_shape((3 * fragments[ind_k].n_atoms, shape_orbs_k, shape_orbs_k))
                            .unwrap()
                            .sum_axis(Axis(2))
                            .sum_axis(Axis(1));

                        let term_2_k: Array1<f64> = (w_s_k + p_grads_k)
                            * ddq_arr
                                .dot(&gamma_tmp.slice(s![
                                    index_pair_iter..index_pair_iter + pair_atoms,
                                    index_frag_iter..index_frag_iter + mol_k.n_atoms
                                ]))
                                .sum();

                        let embedding_part_2: Array1<f64> = term_1_k + term_2_k;

                        let mut temp_grad: Array1<f64> = Array1::zeros(grad_e0_monomers.len());

                        let index_a: usize = pair.frag_a_index;
                        let index_b: usize = pair.frag_b_index;
                        let atoms_a: usize = fragments[index_a].n_atoms;
                        let atoms_b: usize = fragments[index_b].n_atoms;
                        let index_frag_a: usize = indices_frags[index_a];
                        let index_frag_b: usize = indices_frags[index_b];

                        temp_grad
                            .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
                            .add_assign(&embedding_part_1.slice(s![0..3 * atoms_a]));
                        temp_grad
                            .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
                            .add_assign(&embedding_part_1.slice(s![3 * atoms_a..]));
                        temp_grad
                            .slice_mut(s![
                                3 * index_frag_iter..3 * index_frag_iter + 3 * mol_k.n_atoms
                            ])
                            .add_assign(&embedding_part_2);

                        Some(temp_grad)
                    } else {
                        None
                    }
                })
                .collect();
            let mut embedding_gradient: Array1<f64> = Array1::zeros(grad_e0_monomers.len());
            for grad in embedding_pot.iter() {
                embedding_gradient.add_assign(grad);
            }
            embedding_gradients.push(embedding_gradient);
            dimer_gradients.push(dimer_gradient_e0 + dimer_gradient_vrep);
        } else {
            let dimer_natoms: usize =
                fragments[pair.frag_a_index].n_atoms + fragments[pair.frag_b_index].n_atoms;
            let dimer_gradient: Array1<f64> = Array::zeros(dimer_natoms * 3);
            let shape_orbs_a: usize = frag_grad_results[pair.frag_a_index].grad_s.dim().1;
            let shape_orbs_b: usize = frag_grad_results[pair.frag_b_index].grad_s.dim().1;

            // TODO:Which part of g1 is necessary? Only off-diagonal?
            let g1_qb: Array2<f64> = pair_results[pair_index]
                .g1
                .slice(s![
                    0..3 * fragments[pair.frag_a_index].n_atoms,
                    0..fragments[pair.frag_a_index].n_atoms,
                    fragments[pair.frag_a_index].n_atoms..
                ])
                .to_owned()
                .into_shape((
                    3 * fragments[pair.frag_a_index].n_atoms * fragments[pair.frag_a_index].n_atoms,
                    fragments[pair.frag_b_index].n_atoms,
                ))
                .unwrap()
                .dot(&fragments[pair.frag_b_index].final_charges)
                .into_shape((
                    3 * fragments[pair.frag_a_index].n_atoms,
                    fragments[pair.frag_a_index].n_atoms,
                ))
                .unwrap();

            let term_1: Array1<f64> = g1_qb.dot(&fragments[pair.frag_a_index].final_charges);

            //let w_mat_a: Array3<f64> = -0.5
            //    * fragments[pair.frag_a_index]
            //        .final_p_matrix
            //        .dot(
            //            &frag_grad_results[pair.frag_a_index]
            //                .grad_s
            //                .clone()
            //                .into_shape((
            //                    3 * fragments[pair.frag_a_index].n_atoms * shape_orbs_a,
            //                    shape_orbs_a,
            //                ))
            //                .unwrap()
            //                .dot(&fragments[pair.frag_a_index].final_p_matrix)
            //                .t(),
            //        )
            //        .t()
            //        .to_owned()
            //        .into_shape((
            //            3 * fragments[pair.frag_a_index].n_atoms,
            //            shape_orbs_a,
            //            shape_orbs_a,
            //        ))
            //        .unwrap();

            let w_mat_a: Array3<f64> = -0.5*
                &frag_grad_results[pair.frag_a_index]
                    .grad_s
                    .clone()
                    .into_shape((
                        3 * fragments[pair.frag_a_index].n_atoms * shape_orbs_a,
                        shape_orbs_a,
                    ))
                    .unwrap()
                    .dot(&fragments[pair.frag_a_index].final_p_matrix)
                    .dot(&fragments[pair.frag_a_index]
                        .final_p_matrix)
                .into_shape((
                    3 * fragments[pair.frag_a_index].n_atoms,
                    shape_orbs_a,
                    shape_orbs_a,
                ))
                .unwrap();

            let w_s_a: Array1<f64> = w_mat_a
                .into_shape((
                    3 * fragments[pair.frag_a_index].n_atoms * shape_orbs_a,
                    shape_orbs_a,
                ))
                .unwrap()
                .dot(&frag_grad_results[pair.frag_a_index].s)
                .into_shape((
                    3 * fragments[pair.frag_a_index].n_atoms,
                    shape_orbs_a,
                    shape_orbs_a,
                ))
                .unwrap()
                .sum_axis(Axis(2))
                .sum_axis(Axis(1));

            let p_grads_a: Array1<f64> =
                frag_grad_results[pair.frag_a_index]
                    .grad_s
                    .clone()
                    .into_shape((
                        3 * fragments[pair.frag_a_index].n_atoms * shape_orbs_a,
                        shape_orbs_a,
                    ))
                    .unwrap()
                    .dot(&fragments[pair.frag_a_index]
                        .final_p_matrix)
                .into_shape((
                    3 * fragments[pair.frag_a_index].n_atoms,
                    shape_orbs_a,
                    shape_orbs_a,
                ))
                .unwrap()
                .sum_axis(Axis(2))
                .sum_axis(Axis(1));

            //let p_grads_a: Array1<f64> = fragments[pair.frag_a_index]
            //    .final_p_matrix
            //    .dot(
            //        &frag_grad_results[pair.frag_a_index]
            //            .grad_s
            //            .clone()
            //            .into_shape((
            //                3 * fragments[pair.frag_a_index].n_atoms * shape_orbs_a,
            //                shape_orbs_a,
            //            ))
            //            .unwrap()
            //            .t(),
            //    )
            //    .t()
            //    .to_owned()
            //    .into_shape((
            //        3 * fragments[pair.frag_a_index].n_atoms,
            //        shape_orbs_a,
            //        shape_orbs_a,
            //    ))
            //    .unwrap()
            //    .sum_axis(Axis(2))
            //    .sum_axis(Axis(1));

            // TODO: Slice g0 to get the off-diagonal of g0_ab
            let term_2: Array1<f64> = (w_s_a + p_grads_a)
                * pair_results[pair_index]
                    .g0
                    .slice(s![
                        0..fragments[pair.frag_a_index].n_atoms,
                        fragments[pair.frag_a_index].n_atoms..
                    ])
                    .dot(&fragments[pair.frag_b_index].final_charges)
                    .sum();

            let gradient_frag_a: Array1<f64> = term_1 + term_2;

            let g1_qa: Array2<f64> = pair_results[pair_index]
                .g1
                .slice(s![
                    3 * fragments[pair.frag_a_index].n_atoms..,
                    fragments[pair.frag_a_index].n_atoms..,
                    0..fragments[pair.frag_a_index].n_atoms
                ])
                .to_owned()
                .into_shape((
                    3 * fragments[pair.frag_b_index].n_atoms * fragments[pair.frag_b_index].n_atoms,
                    fragments[pair.frag_a_index].n_atoms,
                ))
                .unwrap()
                .dot(&fragments[pair.frag_a_index].final_charges)
                .into_shape((
                    3 * fragments[pair.frag_b_index].n_atoms,
                    fragments[pair.frag_b_index].n_atoms,
                ))
                .unwrap();

            let term_1: Array1<f64> = g1_qa.dot(&fragments[pair.frag_b_index].final_charges);

            //let w_mat_b: Array3<f64> = -0.5
            //    * fragments[pair.frag_b_index]
            //        .final_p_matrix
            //        .dot(
            //            &frag_grad_results[pair.frag_b_index]
            //                .grad_s
            //                .clone()
            //                .into_shape((
            //                    3 * fragments[pair.frag_b_index].n_atoms * shape_orbs_b,
            //                    shape_orbs_b,
            //                ))
            //                .unwrap()
            //                .dot(&fragments[pair.frag_b_index].final_p_matrix)
            //                .t(),
            //        )
            //        .t()
            //        .to_owned()
            //        .into_shape((
            //            3 * fragments[pair.frag_b_index].n_atoms,
            //            shape_orbs_b,
            //            shape_orbs_b,
            //        ))
            //        .unwrap();

            let w_mat_b: Array3<f64> = -0.5*
                &frag_grad_results[pair.frag_b_index]
                    .grad_s
                    .clone()
                    .into_shape((
                        3 * fragments[pair.frag_b_index].n_atoms * shape_orbs_b,
                        shape_orbs_b,
                    ))
                    .unwrap()
                    .dot(&fragments[pair.frag_b_index].final_p_matrix)
                    .dot(&fragments[pair.frag_b_index]
                        .final_p_matrix)
                .into_shape((
                    3 * fragments[pair.frag_b_index].n_atoms,
                    shape_orbs_b,
                    shape_orbs_b,
                ))
                .unwrap();

            let w_s_b: Array1<f64> = w_mat_b
                .into_shape((
                    3 * fragments[pair.frag_b_index].n_atoms * shape_orbs_b,
                    shape_orbs_b,
                ))
                .unwrap()
                .dot(&frag_grad_results[pair.frag_b_index].s)
                .into_shape((
                    3 * fragments[pair.frag_b_index].n_atoms,
                    shape_orbs_b,
                    shape_orbs_b,
                ))
                .unwrap()
                .sum_axis(Axis(2))
                .sum_axis(Axis(1));

            let p_grads_b: Array1<f64> = frag_grad_results[pair.frag_b_index]
                .grad_s
                .clone()
                .into_shape((
                    3 * fragments[pair.frag_b_index].n_atoms * shape_orbs_b,
                    shape_orbs_b,
                ))
                .unwrap()
                .dot(&fragments[pair.frag_b_index]
                    .final_p_matrix)
                .into_shape((
                    3 * fragments[pair.frag_b_index].n_atoms,
                    shape_orbs_b,
                    shape_orbs_b,
                ))
                .unwrap()
                .sum_axis(Axis(2))
                .sum_axis(Axis(1));

            //let p_grads_b: Array1<f64> = fragments[pair.frag_b_index]
            //    .final_p_matrix
            //    .dot(
            //        &frag_grad_results[pair.frag_b_index]
            //            .grad_s
            //            .clone()
            //            .into_shape((
            //                3 * fragments[pair.frag_b_index].n_atoms * shape_orbs_b,
            //                shape_orbs_b,
            //            ))
            //            .unwrap()
            //            .t(),
            //    )
            //    .t()
            //    .to_owned()
            //    .into_shape((
            //        3 * fragments[pair.frag_b_index].n_atoms,
            //        shape_orbs_b,
            //        shape_orbs_b,
            //    ))
            //    .unwrap()
            //    .sum_axis(Axis(2))
            //    .sum_axis(Axis(1));

            let term_2: Array1<f64> = (w_s_b + p_grads_b)
                * pair_results[pair_index]
                    .g0
                    .slice(s![
                        fragments[pair.frag_a_index].n_atoms..,
                        0..fragments[pair.frag_a_index].n_atoms
                    ])
                    .dot(&fragments[pair.frag_a_index].final_charges)
                    .sum();

            let gradient_frag_b: Array1<f64> = term_1 + term_2;
            let mut dimer_gradient: Vec<f64> = Vec::new();
            dimer_gradient.append(&mut gradient_frag_a.to_vec());
            dimer_gradient.append(&mut gradient_frag_b.to_vec());

            let dimer_gradient: Array1<f64> = Array::from(dimer_gradient);

            dimer_gradients.push(dimer_gradient);
        }
    }
    let grad_total_frags: Array1<f64> =
        Array::from(grad_e0_monomers) + Array::from(grad_vrep_monomers);
    let mut grad_total_dimers: Array1<f64> = Array1::zeros(grad_total_frags.raw_dim());

    for (index_pair, pair) in pair_results.iter().enumerate() {
        let index_a: usize = pair.frag_a_index;
        let index_b: usize = pair.frag_b_index;
        let atoms_a: usize = fragments[index_a].n_atoms;
        let atoms_b: usize = fragments[index_b].n_atoms;
        let index_frag_a: usize = indices_frags[index_a];
        let index_frag_b: usize = indices_frags[index_b];

        if pair.energy_pair.is_some(){
           grad_total_dimers
               .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
               .add_assign(
                   &(&dimer_gradients[index_pair].slice(s![0..3 * atoms_a])
                       - &(&frag_grad_results[index_a].grad_e0+&frag_grad_results[index_a].grad_vrep)));
           grad_total_dimers
               .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
               .add_assign(
                   &(&dimer_gradients[index_pair].slice(s![3 * atoms_a..])
                       - &(&frag_grad_results[index_b].grad_e0+&frag_grad_results[index_b].grad_vrep)));
        }

        //grad_total_dimers
        //    .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
        //    .add_assign(
        //        &(&dimer_gradients[index_pair].slice(s![0..3 * atoms_a])
        //            - &(&frag_grad_results[index_a].grad_e0+&frag_grad_results[index_a].grad_vrep)));
        //grad_total_dimers
        //    .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
        //    .add_assign(
        //        &(&dimer_gradients[index_pair].slice(s![3 * atoms_a..])
        //            - &(&frag_grad_results[index_b].grad_e0+&frag_grad_results[index_b].grad_vrep)));

        //grad_total_dimers
        //    .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
        //    .add_assign(
        //        &dimer_gradients[index_pair].slice(s![0..3 * atoms_a])
        //    );
        //grad_total_dimers
        //    .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
        //    .add_assign(
        //        &dimer_gradients[index_pair].slice(s![3 * atoms_a..]
        //        ));
    }

    //for embed in embedding_gradients.iter() {
    //    //println!("Embed {}",embed.clone());
    //    grad_total_dimers.add_assign(embed);
    //}
    let total_gradient: Array1<f64> = grad_total_dimers+ grad_total_frags;

    return total_gradient;
}

pub fn fmo_calculate_pairwise_gradients(
    fragments: &Vec<Molecule>,
    frag_grad_results: &Vec<frag_grad_result>,
    config: GeneralConfig,
    dist_mat: &Array2<f64>,
    direct_mat: &Array3<f64>,
    prox_mat: &Array2<bool>,
    indices_frags: &Vec<usize>,
    gamma_total: &Array2<f64>,
) -> (Vec<pair_grad_result>) {
    // construct a first graph in case all monomers are the same
    let mol_a = fragments[0].clone();
    let mol_b = fragments[1].clone();
    let mut atomic_numbers: Vec<u8> = Vec::new();
    atomic_numbers.append(&mut mol_a.atomic_numbers.clone());
    atomic_numbers.append(&mut mol_b.atomic_numbers.clone());
    let mut positions: Array2<f64> = Array2::zeros((mol_a.n_atoms + mol_b.n_atoms, 3));
    positions
        .slice_mut(s![0..mol_a.n_atoms, ..])
        .assign(&mol_a.positions);
    positions
        .slice_mut(s![mol_a.n_atoms.., ..])
        .assign(&mol_b.positions);
    let distance_frag: Array2<f64> = dist_mat
        .slice(s![
            0..mol_a.n_atoms + mol_b.n_atoms,
            0..mol_a.n_atoms + mol_b.n_atoms
        ])
        .to_owned();
    let dir_frag: Array3<f64> = direct_mat
        .slice(s![
            0..mol_a.n_atoms + mol_b.n_atoms,
            0..mol_a.n_atoms + mol_b.n_atoms,
            ..
        ])
        .to_owned();
    let prox_frag: Array2<bool> = prox_mat
        .slice(s![
            0..mol_a.n_atoms + mol_b.n_atoms,
            0..mol_a.n_atoms + mol_b.n_atoms
        ])
        .to_owned();
    let ga_frag: Array2<f64> = gamma_total
        .slice(s![
            0..mol_a.n_atoms + mol_b.n_atoms,
            0..mol_a.n_atoms + mol_b.n_atoms
        ])
        .to_owned();
    let connectivity_matrix: Array2<bool> =
        build_connectivity_matrix(atomic_numbers.len(), &distance_frag, &atomic_numbers);
    let (graph_new, graph_indexes, subgraph): (
        StableUnGraph<u8, f64>,
        Vec<NodeIndex>,
        Vec<StableUnGraph<u8, f64>>,
    ) = build_graph(&atomic_numbers, &connectivity_matrix, &distance_frag);
    //let (graph_new,graph_indexes, subgraph,connectivity_mat,dist_matrix, dir_matrix, prox_matrix): (StableUnGraph<u8, f64>, Vec<NodeIndex>, Vec<StableUnGraph<u8, f64>>,Array2<bool>,Array2<f64>, Array3<f64>, Array2<bool>) =
    //    create_fmo_graph(atomic_numbers.clone(), positions.clone());
    let first_graph: Graph<u8, f64, Undirected> = Graph::from(graph_new.clone());
    let first_pair: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        Some(config.mol.charge),
        Some(config.mol.multiplicity),
        Some(0.0),
        None,
        config.clone(),
        None,
        Some(connectivity_matrix),
        Some(graph_new),
        Some(graph_indexes),
        Some(subgraph),
        Some(distance_frag),
        Some(dir_frag),
        Some(prox_frag),
        Some(ga_frag),
    );
    let first_calc: DFTBCalculator = first_pair.calculator.clone();

    let mut vec_pair_result: Vec<pair_grad_result> = Vec::new();
    let mut saved_calculators: Vec<DFTBCalculator> = Vec::new();
    let mut saved_graphs: Vec<Graph<u8, f64, Undirected>> = Vec::new();

    saved_graphs.push(first_graph.clone());
    saved_calculators.push(first_calc.clone());

    for (ind1, molecule_a) in fragments.iter().enumerate() {
        for (ind2, molecule_b) in fragments.iter().enumerate() {
            //println!("Index 1 {} and Index 2 {}", ind1, ind2);
            if ind1 < ind2 {
                let mut use_saved_calc: bool = false;
                let mut saved_calc: Option<DFTBCalculator> = None;

                //let molecule_timer: Instant = Instant::now();
                let mut atomic_numbers: Vec<u8> = Vec::new();
                atomic_numbers.append(&mut molecule_a.atomic_numbers.clone());
                atomic_numbers.append(&mut molecule_b.atomic_numbers.clone());

                let mut positions: Array2<f64> =
                    Array2::zeros((molecule_a.n_atoms + molecule_b.n_atoms, 3));

                positions
                    .slice_mut(s![0..molecule_a.n_atoms, ..])
                    .assign(&molecule_a.positions);
                positions
                    .slice_mut(s![molecule_a.n_atoms.., ..])
                    .assign(&molecule_b.positions);

                let mut distance_frag: Array2<f64> = Array2::zeros((
                    molecule_a.n_atoms + molecule_b.n_atoms,
                    molecule_a.n_atoms + molecule_b.n_atoms,
                ));

                distance_frag
                    .slice_mut(s![0..molecule_a.n_atoms, 0..molecule_a.n_atoms])
                    .assign(&dist_mat.slice(s![
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms
                    ]));
                distance_frag
                    .slice_mut(s![0..molecule_a.n_atoms, molecule_a.n_atoms..])
                    .assign(&dist_mat.slice(s![
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms
                    ]));
                distance_frag
                    .slice_mut(s![molecule_a.n_atoms.., 0..molecule_a.n_atoms])
                    .assign(&dist_mat.slice(s![
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms
                    ]));
                distance_frag
                    .slice_mut(s![molecule_a.n_atoms.., molecule_a.n_atoms..])
                    .assign(&dist_mat.slice(s![
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms
                    ]));

                let mut dir_frag: Array3<f64> = Array3::zeros((
                    molecule_a.n_atoms + molecule_b.n_atoms,
                    molecule_a.n_atoms + molecule_b.n_atoms,
                    3,
                ));
                let mut prox_frag: Array2<bool> = Array::from_elem(
                    (
                        molecule_a.n_atoms + molecule_b.n_atoms,
                        molecule_a.n_atoms + molecule_b.n_atoms,
                    ),
                    false,
                );

                dir_frag
                    .slice_mut(s![0..molecule_a.n_atoms, 0..molecule_a.n_atoms, ..])
                    .assign(&direct_mat.slice(s![
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        ..
                    ]));
                dir_frag
                    .slice_mut(s![0..molecule_a.n_atoms, molecule_a.n_atoms.., ..])
                    .assign(&direct_mat.slice(s![
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        ..
                    ]));
                dir_frag
                    .slice_mut(s![molecule_a.n_atoms.., 0..molecule_a.n_atoms, ..])
                    .assign(&direct_mat.slice(s![
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        ..
                    ]));
                dir_frag
                    .slice_mut(s![molecule_a.n_atoms.., molecule_a.n_atoms.., ..])
                    .assign(&direct_mat.slice(s![
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        ..
                    ]));

                prox_frag
                    .slice_mut(s![0..molecule_a.n_atoms, 0..molecule_a.n_atoms])
                    .assign(&prox_mat.slice(s![
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms
                    ]));
                prox_frag
                    .slice_mut(s![0..molecule_a.n_atoms, molecule_a.n_atoms..])
                    .assign(&prox_mat.slice(s![
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms
                    ]));
                prox_frag
                    .slice_mut(s![molecule_a.n_atoms.., 0..molecule_a.n_atoms])
                    .assign(&prox_mat.slice(s![
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms
                    ]));
                prox_frag
                    .slice_mut(s![molecule_a.n_atoms.., molecule_a.n_atoms..])
                    .assign(&prox_mat.slice(s![
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms
                    ]));

                let connectivity_matrix: Array2<bool> = build_connectivity_matrix(
                    atomic_numbers.len(),
                    &distance_frag,
                    &atomic_numbers,
                );

                let (graph_new, graph_indexes, subgraph): (
                    StableUnGraph<u8, f64>,
                    Vec<NodeIndex>,
                    Vec<StableUnGraph<u8, f64>>,
                ) = build_graph(&atomic_numbers, &connectivity_matrix, &distance_frag);

                let graph: Graph<u8, f64, Undirected> = Graph::from(graph_new.clone());

                if saved_graphs.len() > 0 {
                    for (ind_g, saved_graph) in saved_graphs.iter().enumerate() {
                        if is_isomorphic_matching(&graph, saved_graph, |a, b| a == b, |a, b| true)
                            == true
                        {
                            use_saved_calc = true;
                            saved_calc = Some(saved_calculators[ind_g].clone());
                        }
                    }
                }
                // get shortest distance between the fragment atoms of the pair
                let distance_between_pair: Array2<f64> = distance_frag
                    .slice(s![..molecule_a.n_atoms, molecule_a.n_atoms..])
                    .to_owned();
                let min_dist: f64 = distance_between_pair
                    .iter()
                    .cloned()
                    .min_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
                    .unwrap();

                let index_min_vec: Vec<(usize, usize)> = distance_between_pair
                    .indexed_iter()
                    .filter_map(|(index, &item)| if item == min_dist { Some(index) } else { None })
                    .collect();
                let index_min = index_min_vec[0];

                let vdw_radii_sum: f64 = (constants::VDW_RADII
                    [&molecule_a.atomic_numbers[index_min.0]]
                    + constants::VDW_RADII[&molecule_b.atomic_numbers[index_min.1]])
                    / constants::BOHR_TO_ANGS;
                let mut energy_pair: Option<f64> = None;
                let mut charges_pair: Option<Array1<f64>> = None;
                let mut grad_e0_pair: Option<Array1<f64>> = None;
                let mut grad_vrep_pair: Option<Array1<f64>> = None;
                let mut pair_grad_s: Option<Array3<f64>> = None;
                let mut pair_s: Option<Array2<f64>> = None;
                let mut pair_density: Option<Array2<f64>> = None;

                let mut gamma_frag: Array2<f64> = Array2::zeros((
                    molecule_a.n_atoms + molecule_b.n_atoms,
                    molecule_a.n_atoms + molecule_b.n_atoms,
                ));
                gamma_frag
                    .slice_mut(s![0..molecule_a.n_atoms, 0..molecule_a.n_atoms])
                    .assign(&gamma_total.slice(s![
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms
                    ]));
                gamma_frag
                    .slice_mut(s![0..molecule_a.n_atoms, molecule_a.n_atoms..])
                    .assign(&gamma_total.slice(s![
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms,
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms
                    ]));
                gamma_frag
                    .slice_mut(s![molecule_a.n_atoms.., 0..molecule_a.n_atoms])
                    .assign(&gamma_total.slice(s![
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        indices_frags[ind1]..indices_frags[ind1] + molecule_a.n_atoms
                    ]));
                gamma_frag
                    .slice_mut(s![molecule_a.n_atoms.., molecule_a.n_atoms..])
                    .assign(&gamma_total.slice(s![
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms,
                        indices_frags[ind2]..indices_frags[ind2] + molecule_b.n_atoms
                    ]));

                let mut pair: Molecule = Molecule::new(
                    atomic_numbers,
                    positions,
                    Some(config.mol.charge),
                    Some(config.mol.multiplicity),
                    Some(0.0),
                    None,
                    config.clone(),
                    saved_calc,
                    Some(connectivity_matrix),
                    Some(graph_new),
                    Some(graph_indexes),
                    Some(subgraph),
                    Some(distance_frag),
                    Some(dir_frag),
                    Some(prox_frag),
                    Some(gamma_frag),
                );

                //let mut pair: Molecule = Molecule::new(
                //    atomic_numbers,
                //    positions,
                //    Some(config.mol.charge),
                //    Some(config.mol.multiplicity),
                //    Some(0.0),
                //    None,
                //    config.clone(),
                //    None,
                //    None,
                //    None,
                //    None,
                //    None,
                //    None,
                //    None,
                //    None,
                //    None,
                //);

                if use_saved_calc == false {
                    saved_calculators.push(pair.calculator.clone());
                    saved_graphs.push(graph.clone());
                }

                let (g1, g1_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
                    &pair.atomic_numbers,
                    pair.n_atoms,
                    pair.calculator.n_orbs,
                    pair.distance_matrix.view(),
                    pair.directions_matrix.view(),
                    &pair.calculator.hubbard_u,
                    &pair.calculator.valorbs,
                    Some(0.0),
                );

                // do scc routine for pair if mininmal distance is below threshold
                if (min_dist / vdw_radii_sum) < 2.0 {
                    let (energy, orbs, orbe, s, f): (
                        f64,
                        Array2<f64>,
                        Array1<f64>,
                        Array2<f64>,
                        Vec<f64>,
                    ) = scc_routine::run_scc(&mut pair);
                    energy_pair = Some(energy);
                    charges_pair = Some(pair.final_charges.clone());
                    pair_density = Some(pair.final_p_matrix.clone());

                    pair.calculator.set_active_orbitals(f);
                    let full_occ: Vec<usize> = pair.calculator.full_occ.clone().unwrap();
                    let full_virt: Vec<usize> = pair.calculator.full_virt.clone().unwrap();
                    let n_occ_full: usize = full_occ.len();

                    let orbe_occ: Array1<f64> =
                        full_occ.iter().map(|&full_occ| orbe[full_occ]).collect();
                    let orbe_virt: Array1<f64> =
                        full_virt.iter().map(|&full_virt| orbe[full_virt]).collect();
                    let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ_full));

                    for (i, index) in full_occ.iter().enumerate() {
                        orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
                    }

                    let (
                        gradE0,
                        grad_v_rep,
                        grad_s,
                        grad_h0,
                        fdmdO,
                        flrdmdO,
                        g1,
                        g1_ao,
                        g1lr,
                        g1lr_ao,
                    ): (
                        Array1<f64>,
                        Array1<f64>,
                        Array3<f64>,
                        Array3<f64>,
                        Array3<f64>,
                        Array3<f64>,
                        Array3<f64>,
                        Array3<f64>,
                        Array3<f64>,
                        Array3<f64>,
                    ) = gradient_lc_gs(&pair, &orbe_occ, &orbe_virt, &orbs_occ, &s, Some(0.0));

                    grad_e0_pair = Some(gradE0);
                    grad_vrep_pair = Some(grad_v_rep);

                    let (grad_s, grad_h0): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
                        &pair.atomic_numbers,
                        pair.positions.view(),
                        pair.calculator.n_orbs,
                        &pair.calculator.valorbs,
                        pair.proximity_matrix.view(),
                        &pair.calculator.skt,
                        &pair.calculator.orbital_energies,
                    );
                    pair_s = Some(s);
                    pair_grad_s = Some(grad_s)
                }

                let pair_res: pair_grad_result = pair_grad_result::new(
                    charges_pair,
                    energy_pair,
                    ind1,
                    ind2,
                    molecule_a.n_atoms,
                    molecule_b.n_atoms,
                    grad_e0_pair,
                    grad_vrep_pair,
                    pair.g0,
                    g1,
                    pair_s,
                    pair_grad_s,
                    pair_density,
                );

                vec_pair_result.push(pair_res);
            }
        }
    }
    return (vec_pair_result);
}

pub fn fmo_calculate_fragment_gradients(fragments: &mut Vec<Molecule>) -> (Vec<frag_grad_result>) {
    let mut results: Vec<frag_grad_result> = Vec::new();

    for frag in fragments.iter_mut() {
        let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(frag);

        frag.calculator.set_active_orbitals(f);
        let full_occ: Vec<usize> = frag.calculator.full_occ.clone().unwrap();
        let full_virt: Vec<usize> = frag.calculator.full_virt.clone().unwrap();
        let n_occ_full: usize = full_occ.len();

        let orbe_occ: Array1<f64> = full_occ.iter().map(|&full_occ| orbe[full_occ]).collect();
        let orbe_virt: Array1<f64> = full_virt.iter().map(|&full_virt| orbe[full_virt]).collect();
        let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ_full));

        for (i, index) in full_occ.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        let (gradE0, grad_v_rep, grad_s, grad_h0, fdmdO, flrdmdO, g1, g1_ao, g1lr, g1lr_ao): (
            Array1<f64>,
            Array1<f64>,
            Array3<f64>,
            Array3<f64>,
            Array3<f64>,
            Array3<f64>,
            Array3<f64>,
            Array3<f64>,
            Array3<f64>,
            Array3<f64>,
        ) = gradient_lc_gs(&frag, &orbe_occ, &orbe_virt, &orbs_occ, &s, Some(0.0));

        let frag_result: frag_grad_result =
            frag_grad_result::new(energy, gradE0, grad_v_rep, grad_s, s);

        results.push(frag_result);
    }
    return results;
}

pub fn reorder_molecule_gradients(
    fragments: &Vec<Molecule>,
    config: GeneralConfig,
    shape_positions: Ix2,
) -> (
    Vec<usize>,
    Array2<f64>,
    Array3<f64>,
    Array2<bool>,
    Array2<f64>,
    Array3<f64>,
) {
    let mut atomic_numbers: Vec<u8> = Vec::new();
    let mut positions: Array2<f64> = Array2::zeros(shape_positions);
    let mut indices_vector: Vec<usize> = Vec::new();
    let mut mut_fragments: Vec<Molecule> = fragments.clone();
    let mut prev_nat: usize = 0;

    for molecule in mut_fragments.iter_mut() {
        //for (ind, atom) in molecule.atomic_numbers.iter().enumerate() {
        //    atomic_numbers.push(*atom);
        //    positions
        //        .slice_mut(s![ind, ..])
        //        .assign(&molecule.positions.slice(s![ind, ..]));
        //}
        atomic_numbers.append(&mut molecule.atomic_numbers);
        positions
            .slice_mut(s![prev_nat..prev_nat + molecule.n_atoms, ..])
            .assign(&molecule.positions);
        indices_vector.push(prev_nat);
        prev_nat += molecule.n_atoms;
    }

    let new_mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        Some(config.mol.charge),
        Some(config.mol.multiplicity),
        Some(0.0),
        None,
        config.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let (g1, g1_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
        &new_mol.atomic_numbers,
        new_mol.n_atoms,
        new_mol.calculator.n_orbs,
        new_mol.distance_matrix.view(),
        new_mol.directions_matrix.view(),
        &new_mol.calculator.hubbard_u,
        &new_mol.calculator.valorbs,
        Some(0.0),
    );

    return (
        indices_vector,
        new_mol.g0,
        g1,
        new_mol.proximity_matrix,
        new_mol.distance_matrix,
        new_mol.directions_matrix,
    );
}

pub struct frag_grad_result {
    energy: f64,
    grad_e0: Array1<f64>,
    grad_vrep: Array1<f64>,
    grad_s: Array3<f64>,
    s: Array2<f64>,
}

impl frag_grad_result {
    pub(crate) fn new(
        energy: f64,
        grad_e0: Array1<f64>,
        grad_vrep: Array1<f64>,
        grad_s: Array3<f64>,
        s: Array2<f64>,
    ) -> (frag_grad_result) {
        let result = frag_grad_result {
            energy: energy,
            grad_e0: grad_e0,
            grad_vrep: grad_vrep,
            grad_s: grad_s,
            s: s,
        };
        return result;
    }
}

pub struct pair_grad_result {
    pair_charges: Option<Array1<f64>>,
    energy_pair: Option<f64>,
    frag_a_index: usize,
    frag_b_index: usize,
    frag_a_atoms: usize,
    frag_b_atoms: usize,
    grad_e0: Option<Array1<f64>>,
    grad_vrep: Option<Array1<f64>>,
    g0: Array2<f64>,
    g1: Array3<f64>,
    s: Option<Array2<f64>>,
    grad_s: Option<Array3<f64>>,
    p_mat: Option<Array2<f64>>,
}

impl pair_grad_result {
    pub(crate) fn new(
        pair_charges: Option<Array1<f64>>,
        energy: Option<f64>,
        frag_a_index: usize,
        frag_b_index: usize,
        frag_a_atoms: usize,
        frag_b_atoms: usize,
        grad_e0: Option<Array1<f64>>,
        grad_vrep: Option<Array1<f64>>,
        g0: Array2<f64>,
        g1: Array3<f64>,
        s: Option<Array2<f64>>,
        grad_s: Option<Array3<f64>>,
        p_mat: Option<Array2<f64>>,
    ) -> (pair_grad_result) {
        let result = pair_grad_result {
            pair_charges: pair_charges,
            energy_pair: energy,
            frag_a_index: frag_a_index,
            frag_b_index: frag_b_index,
            frag_a_atoms: frag_a_atoms,
            frag_b_atoms: frag_b_atoms,
            grad_e0: grad_e0,
            grad_vrep: grad_vrep,
            g0: g0,
            g1: g1,
            s: s,
            grad_s: grad_s,
            p_mat: p_mat,
        };
        return result;
    }
}
