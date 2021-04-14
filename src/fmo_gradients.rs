use crate::calculator::{get_gamma_gradient_matrix_atom_wise_outer_diagonal, get_gamma_matrix_atomwise_outer_diagonal};
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
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3,stack};
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
    // gamma_total: Array2<f64>,
    // prox_mat: Array2<bool>,
    dist_mat: &Array2<f64>,
    directions: &Array3<f64>,
    full_hubbard: &HashMap<u8, f64>,
) -> (Array1<f64>) {
    // sum over all monomer energies
    let mut grad_e0_monomers: Vec<f64> = Vec::new();
    let mut grad_vrep_monomers: Vec<f64> = Vec::new();

    for frag in frag_grad_results.iter() {
        grad_e0_monomers.append(&mut frag.grad_e0.clone().to_vec());
        grad_vrep_monomers.append(&mut frag.grad_vrep.clone().to_vec());
    }
    let grad_e0_monomers:Array1<f64> = Array::from(grad_e0_monomers);
    let grad_vrep_monomers:Array1<f64> = Array::from(grad_vrep_monomers);

    // get energy term for pairs
    let mut iter: usize = 0;
    let mut pair_energies: f64 = 0.0;
    let mut embedding_potential: f64 = 0.0;

    // let proximity_zeros: Array1<f64> = prox_mat
    //     .iter()
    //     .filter_map(|&item| if item == true { Some(1.0) } else { Some(0.0) })
    //     .collect();
    // let gamma_zeros: Array2<f64> = proximity_zeros.into_shape((prox_mat.raw_dim())).unwrap();
    //let gamma_tmp: Array2<f64> = gamma_zeros * gamma_total;
    // let gamma_tmp: Array2<f64> = 1.0 * gamma_total;
    let mut dimer_gradients: Vec<Array1<f64>> = Vec::new();
    let mut embedding_gradients: Vec<Array1<f64>> = Vec::new();
    let mut pair_scc_hash:HashMap<usize,usize> = HashMap::new();
    let mut pair_esdim_hash:HashMap<usize,usize> = HashMap::new();
    let mut scc_iter:usize = 0;
    let mut esdim_iter:usize = 0;

    let grad_total_frags: Array1<f64> = &grad_e0_monomers + &grad_vrep_monomers;

    let molecule_timer: Instant = Instant::now();

    for (pair_index, pair) in pair_results.iter().enumerate() {
        if pair.energy_pair.is_some() {
            let dimer_gradient_e0: Array1<f64> = pair.grad_e0.clone().unwrap();
            let dimer_gradient_vrep: Array1<f64> = pair.grad_vrep.clone().unwrap();
            pair_scc_hash.insert(pair_index,scc_iter);
            scc_iter+=1;

            let dimer_pmat: Array2<f64> = pair.p_mat.clone().unwrap();

            let pair_atoms: usize =
                fragments[pair.frag_a_index].n_atoms + fragments[pair.frag_b_index].n_atoms;
            let pair_charges = pair.pair_charges.clone().unwrap();
            let pair_smat: Array2<f64> = pair.s.clone().unwrap();
            // let pair_grads: Array3<f64> = pair.grad_s.clone().unwrap();
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

            // calculate grad s for each pair
            let mut dimer_atomic_numbers: Vec<u8> = Vec::new();
            dimer_atomic_numbers
                .append(&mut fragments[pair.frag_a_index].atomic_numbers.clone());
            dimer_atomic_numbers
                .append(&mut fragments[pair.frag_b_index].atomic_numbers.clone());

            let mut dimer_positions: Array2<f64> =
                Array2::zeros((fragments[pair.frag_a_index].n_atoms + fragments[pair.frag_b_index].n_atoms, 3));
            dimer_positions
                .slice_mut(s![0..fragments[pair.frag_a_index].n_atoms, ..])
                .assign(&fragments[pair.frag_a_index].positions);
            dimer_positions
                .slice_mut(s![fragments[pair.frag_a_index].n_atoms.., ..])
                .assign(&fragments[pair.frag_b_index].positions);

            let (pair_grads, pair_grad_h0): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
                &dimer_atomic_numbers,
                dimer_positions.view(),
                pair.pair_n_orbs.unwrap(),
                &pair.pair_valorbs.clone().unwrap(),
                pair.pair_proximity.clone().unwrap().view(),
                &pair.pair_skt.clone().unwrap(),
                &pair.pair_orbital_energies.clone().unwrap(),
            );
            drop(pair_grad_h0);

            // calculate grad s for each fragment of the pair
            let (grad_s_frag_a, grad_h0_frag_a): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
                &fragments[pair.frag_a_index].atomic_numbers,
                fragments[pair.frag_a_index].positions.view(),
                fragments[pair.frag_a_index].calculator.n_orbs,
                &fragments[pair.frag_a_index].calculator.valorbs,
                fragments[pair.frag_a_index].proximity_matrix.view(),
                &fragments[pair.frag_a_index].calculator.skt,
                &fragments[pair.frag_a_index].calculator.orbital_energies,
            );
            drop(grad_h0_frag_a);

            let (grad_s_frag_b, grad_h0_frag_b): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
                &fragments[pair.frag_b_index].atomic_numbers,
                fragments[pair.frag_b_index].positions.view(),
                fragments[pair.frag_b_index].calculator.n_orbs,
                &fragments[pair.frag_b_index].calculator.valorbs,
                fragments[pair.frag_b_index].proximity_matrix.view(),
                &fragments[pair.frag_b_index].calculator.skt,
                &fragments[pair.frag_b_index].calculator.orbital_energies,
            );
            drop(grad_h0_frag_b);

            let index_pair_iter: usize = indices_frags[pair.frag_a_index];
            let ddq_arr: Array1<f64> = Array::from(ddq_vec);
            let shape_orbs_dimer: usize = pair_grads.dim().1;
            let shape_orbs_a: usize = fragments[pair.frag_a_index].calculator.n_orbs;
            let shape_orbs_b: usize = fragments[pair.frag_b_index].calculator.n_orbs;

            let mut w_dimer: Array3<f64> =
                Array3::zeros((3 * pair_atoms, shape_orbs_dimer, shape_orbs_dimer));
            for i in (0..3 * pair_atoms).into_iter() {
                w_dimer
                    .slice_mut(s![i, .., ..])
                    .assign(&dimer_pmat.dot(&pair_grads.slice(s![i, .., ..]).dot(&dimer_pmat)));
            }
            w_dimer = -0.5 * w_dimer;

            let mut w_mat_a: Array3<f64> = Array3::zeros((
                3 * fragments[pair.frag_a_index].n_atoms,
                shape_orbs_a,
                shape_orbs_a,
            ));
            for i in (0..3 * fragments[pair.frag_a_index].n_atoms).into_iter() {
                w_mat_a.slice_mut(s![i, .., ..]).assign(
                    &fragments[pair.frag_a_index].final_p_matrix.dot(
                        &grad_s_frag_a
                            .slice(s![i, .., ..])
                            .dot(&fragments[pair.frag_a_index].final_p_matrix),
                    ),
                );
            }
            w_mat_a = -0.5 * w_mat_a;

            let mut w_mat_b: Array3<f64> = Array3::zeros((
                3 * fragments[pair.frag_b_index].n_atoms,
                shape_orbs_b,
                shape_orbs_b,
            ));
            for i in (0..3 * fragments[pair.frag_b_index].n_atoms).into_iter() {
                w_mat_b.slice_mut(s![i, .., ..]).assign(
                    &fragments[pair.frag_b_index].final_p_matrix.dot(
                        &grad_s_frag_b
                            .slice(s![i, .., ..])
                            .dot(&fragments[pair.frag_b_index].final_p_matrix),
                    ),
                );
            }
            w_mat_b = -0.5 * w_mat_b;

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

            // let molecule_timer: Instant = Instant::now();

            let embedding_pot: Vec<Array1<f64>> = fragments
                .par_iter()
                .enumerate()
                .filter_map(|(ind_k, mol_k)| {
                    if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index {
                        let index_frag_iter: usize = indices_frags[ind_k];

                        // let dgamma_ac: ArrayView3<f64> = g1_total.slice(s![
                        //     3*index_pair_iter..3*index_pair_iter + 3 * pair_atoms,
                        //     index_pair_iter..index_pair_iter + pair_atoms,
                        //     index_frag_iter..index_frag_iter + mol_k.n_atoms
                        // ]);
                        //
                        // let dgamma_ac_k: ArrayView3<f64> = g1_total.slice(s![
                        //     3*index_frag_iter..3*index_frag_iter + 3 * mol_k.n_atoms,
                        //     index_frag_iter..index_frag_iter + mol_k.n_atoms,
                        //     index_pair_iter..index_pair_iter + pair_atoms
                        // ]);

                        // calculate g1 matrix for each trimer
                        let trimer_distances: ArrayView2<f64> = dist_mat.slice(s![
                                index_pair_iter..index_pair_iter + pair_atoms,
                                index_frag_iter..index_frag_iter + mol_k.n_atoms
                            ]);

                        let trimer_directions: ArrayView3<f64> = directions.slice(s![
                                index_pair_iter..index_pair_iter + pair_atoms,
                                index_frag_iter..index_frag_iter + mol_k.n_atoms,
                                ..
                            ]);
                        let g1_trimer_ak: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
                            &dimer_atomic_numbers,
                            &mol_k.atomic_numbers,
                            pair_atoms,
                            mol_k.n_atoms,
                            trimer_distances,
                            trimer_directions,
                            full_hubbard,
                            Some(0.0),
                        );

                        let g0_trimer_ak: Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
                            &dimer_atomic_numbers,
                            &mol_k.atomic_numbers,
                            pair_atoms,
                            mol_k.n_atoms,
                            trimer_distances,
                            full_hubbard,
                            Some(0.0));

                        let trimer_distances: ArrayView2<f64> = dist_mat.slice(s![
                                index_frag_iter..index_frag_iter + mol_k.n_atoms,
                                index_pair_iter..index_pair_iter + pair_atoms
                            ]);

                        let trimer_directions: ArrayView3<f64> = directions.slice(s![
                                index_frag_iter..index_frag_iter + mol_k.n_atoms,
                                index_pair_iter..index_pair_iter + pair_atoms,
                                ..
                            ]);

                        let g1_trimer_ka: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
                            &mol_k.atomic_numbers,
                            &dimer_atomic_numbers,
                            mol_k.n_atoms,
                            pair_atoms,
                            trimer_distances,
                            trimer_directions,
                            full_hubbard,
                            Some(0.0),
                        );

                        // calculate grads for molecule k
                        let (grad_s_frag_k, grad_h0_frag_k): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
                            &mol_k.atomic_numbers,
                            mol_k.positions.view(),
                            mol_k.calculator.n_orbs,
                            &mol_k.calculator.valorbs,
                            mol_k.proximity_matrix.view(),
                            &mol_k.calculator.skt,
                            &mol_k.calculator.orbital_energies,
                        );
                        drop(grad_h0_frag_k);

                        // println!("g1 ac k {}",dgamma_ac_k.clone().slice(s![0,..,..]));
                        // println!("g1 ac k 2 {}",dgamma_ac_2.clone().slice(s![0,..,..]));
                        // println!("g1 ac k {}",dgamma_ac.clone().slice(s![1,..,..]));
                        // println!("g1 ac k 2 {}",dgamma_ac_2.clone().slice(s![1,..,..]));
                        // assert!(
                        //     dgamma_ac.abs_diff_eq(&g1_trimer_ak, 1e-12),
                        //     "Gamma matrices are NOT equal!!!!"
                        // );
                        // assert!(
                        //     dgamma_ac_k.abs_diff_eq(&g1_trimer_ka, 1e-12),
                        //     "Gamma matrices are NOT equal!!!!"
                        // );

                        // let g0_ab: ArrayView2<f64> = gamma_tmp.slice(s![
                        //     index_pair_iter..index_pair_iter + pair_atoms,
                        //     index_frag_iter..index_frag_iter + mol_k.n_atoms
                        // ]);
                        //
                        // assert!(
                        //     g0_ab.abs_diff_eq(&g0_trimer_ak, 1e-12),
                        //     "Gamma matrices are NOT equal!!!!"
                        // );

                        let mut term_1: Array1<f64> = Array1::zeros(3 * pair_atoms);
                        let mut term_2: Array1<f64> = Array1::zeros(3 * pair_atoms);
                        for dir in (0..3).into_iter() {
                            let dir_xyz: usize = dir as usize;
                            let mut diag_ind: usize = 0;
                            for a in (0..pair_atoms).into_iter() {
                                let index: usize = 3 * a + dir_xyz;
                                // term_1[index] = ddq_arr[a]
                                //     * dgamma_ac.slice(s![index, a, ..]).dot(&mol_k.final_charges);
                                term_1[index] = ddq_arr[a]
                                    * g1_trimer_ak.slice(s![index, a, ..]).dot(&mol_k.final_charges);

                                let mut atom_type: u8 = 0;
                                let mut norbs_a: usize = 0;

                                if a < pair.frag_a_atoms {
                                    atom_type = fragments[pair.frag_a_index].atomic_numbers[a];
                                    norbs_a = fragments[pair.frag_a_index].calculator.valorbs
                                        [&atom_type]
                                        .len();
                                } else {
                                    atom_type = fragments[pair.frag_b_index].atomic_numbers
                                        [a - pair.frag_a_atoms];
                                    norbs_a = fragments[pair.frag_b_index].calculator.valorbs
                                        [&atom_type]
                                        .len();
                                }

                                let tmp_1: Array2<f64> =
                                    dw_dimer.slice(s![index, .., ..]).dot(&pair_smat.t());
                                let tmp_2: Array2<f64> =
                                    dp_dimer.dot(&pair_grads.slice(s![index, .., ..]).t());
                                let sum: Array2<f64> = tmp_1 + tmp_2;

                                let diag: f64 =
                                    sum.diag().slice(s![diag_ind..diag_ind + norbs_a]).sum();
                                diag_ind += norbs_a;
                                // let tmp_1:f64 = dw_dimer.slice(s![index,..,..]).dot(&pair_smat.t()).trace().unwrap();
                                // let tmp_2:f64 = dp_dimer.dot(&pair_grads.slice(s![index,..,..]).t()).trace().unwrap();
                                // term_2[index] = (tmp_1 + tmp_2) * g0_ab.slice(s![a,..]).dot(&fragments[ind_k].final_charges);
                                // term_2[index] = diag
                                //     * g0_ab.slice(s![a, ..]).dot(&fragments[ind_k].final_charges);
                                term_2[index] = diag
                                    * g0_trimer_ak.slice(s![a, ..]).dot(&fragments[ind_k].final_charges);
                            }
                        }
                        let embedding_part_1: Array1<f64> = term_1 + term_2;

                        // let shape_orbs_k: usize = frag_grad_results[ind_k].grad_s.dim().1;
                        let shape_orbs_k: usize = grad_s_frag_k.dim().1;
                        let mut w_mat_k: Array3<f64> = Array3::zeros((
                            3 * fragments[ind_k].n_atoms,
                            shape_orbs_k,
                            shape_orbs_k,
                        ));
                        // for i in (0..3 * fragments[ind_k].n_atoms).into_iter() {
                        //     w_mat_k.slice_mut(s![i, .., ..]).assign(
                        //         &fragments[ind_k].final_p_matrix.dot(
                        //             &frag_grad_results[ind_k]
                        //                 .grad_s
                        //                 .slice(s![i, .., ..])
                        //                 .dot(&fragments[ind_k].final_p_matrix),
                        //         ),
                        //     );
                        // }
                        for i in (0..3 * fragments[ind_k].n_atoms).into_iter() {
                            w_mat_k.slice_mut(s![i, .., ..]).assign(
                                &fragments[ind_k].final_p_matrix.dot(&grad_s_frag_k
                                        .slice(s![i, .., ..])
                                        .dot(&fragments[ind_k].final_p_matrix),
                                ),
                            );
                        }
                        w_mat_k = -0.5 * w_mat_k;

                        let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[ind_k].n_atoms);
                        let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[ind_k].n_atoms);
                        for dir in (0..3).into_iter() {
                            let dir_xyz: usize = dir as usize;
                            let mut diag_ind: usize = 0;
                            for a in (0..fragments[ind_k].n_atoms).into_iter() {
                                let index: usize = 3 * a + dir_xyz;
                                // term_1[index] = fragments[ind_k].final_charges[a]
                                //     * dgamma_ac_k.slice(s![index, a, ..]).dot(&ddq_arr);
                                term_1[index] = fragments[ind_k].final_charges[a]
                                    * g1_trimer_ka.slice(s![index, a, ..]).dot(&ddq_arr);

                                let atom_type: u8 = fragments[ind_k].atomic_numbers[a];
                                let norbs_k: usize =
                                    fragments[ind_k].calculator.valorbs[&atom_type].len();

                                let tmp_1: Array2<f64> = w_mat_k
                                    .slice(s![index, .., ..])
                                    .dot(&frag_grad_results[ind_k].s.t());
                                // let tmp_2: Array2<f64> = fragments[ind_k].final_p_matrix.dot(
                                //     &frag_grad_results[ind_k].grad_s.slice(s![index, .., ..]).t(),
                                // );
                                let tmp_2: Array2<f64> = fragments[ind_k].final_p_matrix.dot(
                                    &grad_s_frag_k.slice(s![index, .., ..]).t(),
                                );

                                let sum: Array2<f64> = tmp_1 + tmp_2;

                                let diag: f64 =
                                    sum.diag().slice(s![diag_ind..diag_ind + norbs_k]).sum();
                                diag_ind += norbs_k;
                                // let tmp_1:f64 = w_mat_k.slice(s![index,..,..]).dot(&frag_grad_results[ind_k].s.t()).trace().unwrap();
                                // let tmp_2:f64 = fragments[ind_k].final_p_matrix.dot(&frag_grad_results[ind_k].grad_s.slice(s![index,..,..]).t()).trace().unwrap();
                                // term_2[index] = (tmp_1 + tmp_2) * g0_ab.slice(s![..,a]).dot(&ddq_arr);
                                // term_2[index] = diag * g0_ab.slice(s![.., a]).dot(&ddq_arr);
                                term_2[index] = diag * g0_trimer_ak.slice(s![.., a]).dot(&ddq_arr);
                            }
                        }
                        let embedding_part_2: Array1<f64> = term_1 + term_2;
                        // let term_1: Array1<f64> = dgamma_ac
                        //     .clone()
                        //     .into_shape((3 * pair_atoms * pair_atoms, mol_k.n_atoms))
                        //     .unwrap()
                        //     .dot(&mol_k.final_charges)
                        //     .into_shape((3 * pair_atoms, pair_atoms))
                        //     .unwrap()
                        //     .dot(&ddq_arr);
                        //
                        // let dw_s_a: Array1<f64> = dw_dimer
                        //     .clone()
                        //     .into_shape((3 * pair_atoms * w_dimer_dim, w_dimer_dim))
                        //     .unwrap()
                        //     .dot(&pair_smat)
                        //     .into_shape((3 * pair_atoms, w_dimer_dim, w_dimer_dim))
                        //     .unwrap()
                        //     .sum_axis(Axis(2))
                        //     .sum_axis(Axis(1));
                        //
                        // let mut dp_grads_a:Array3<f64> = Array3::zeros((3*pair_atoms,w_dimer_dim,w_dimer_dim));
                        // for i in (0..3*pair_atoms).into_iter(){
                        //     dp_grads_a.slice_mut(s![i,..,..]).assign(&dp_dimer.dot(&pair_grads.slice(s![i,..,..])));
                        // }
                        //
                        // let dp_grads_sum_a:Array1<f64> = dp_grads_a.sum_axis(Axis(2)).sum_axis(Axis(1));
                        // let term_2: Array1<f64> = (dw_s_a + dp_grads_sum_a)
                        //     * gamma_tmp
                        //         .slice(s![
                        //             index_pair_iter..index_pair_iter + pair_atoms,
                        //             index_frag_iter..index_frag_iter + mol_k.n_atoms
                        //         ])
                        //         .dot(&fragments[ind_k].final_charges)
                        //         .sum();
                        // let dgamma_ac: Array3<f64> = g1_total
                        //     .slice(s![
                        //         index_frag_iter..index_frag_iter + 3 * mol_k.n_atoms,
                        //         index_pair_iter..index_pair_iter + pair_atoms,
                        //         index_frag_iter..index_frag_iter + mol_k.n_atoms
                        //     ])
                        //     .to_owned();
                        //
                        // let term_1_k: Array1<f64> = dgamma_ac
                        //     .into_shape((3 * mol_k.n_atoms * pair_atoms, mol_k.n_atoms))
                        //     .unwrap()
                        //     .dot(&mol_k.final_charges)
                        //     .into_shape((3 * mol_k.n_atoms, pair_atoms))
                        //     .unwrap()
                        //     .dot(&ddq_arr);
                        // let w_s_k: Array1<f64> = w_mat_k
                        //     .into_shape((3 * fragments[ind_k].n_atoms * shape_orbs_k, shape_orbs_k))
                        //     .unwrap()
                        //     .dot(&frag_grad_results[ind_k].s)
                        //     .into_shape((3 * fragments[ind_k].n_atoms, shape_orbs_k, shape_orbs_k))
                        //     .unwrap()
                        //     .sum_axis(Axis(2))
                        //     .sum_axis(Axis(1));
                        //
                        // let mut p_grads_k:Array3<f64> = Array3::zeros((3 * fragments[ind_k].n_atoms,shape_orbs_k,shape_orbs_k));
                        // for i in (0..3 * fragments[ind_k].n_atoms).into_iter(){
                        //     p_grads_k.slice_mut(s![i,..,..]).assign(&fragments[ind_k].final_p_matrix.dot(&frag_grad_results[ind_k].grad_s.slice(s![i,..,..])));
                        // }
                        //
                        // let p_grads_sum_k:Array1<f64> =p_grads_k.sum_axis(Axis(2)).sum_axis(Axis(1));
                        //
                        // let term_2_k: Array1<f64> = (w_s_k + p_grads_sum_k)
                        //     * ddq_arr
                        //         .dot(&gamma_tmp.slice(s![
                        //             index_pair_iter..index_pair_iter + pair_atoms,
                        //             index_frag_iter..index_frag_iter + mol_k.n_atoms
                        //         ]))
                        //         .sum();
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
            // println!(
            //     "{:>68} {:>8.10} s",
            //     "elapsed time embedding:",
            //     molecule_timer.elapsed().as_secs_f32()
            // );
            // drop(molecule_timer);

            let mut embedding_gradient: Array1<f64> = Array1::zeros(grad_e0_monomers.len());
            for grad in embedding_pot.iter() {
                embedding_gradient.add_assign(grad);
            }
            embedding_gradients.push(embedding_gradient);
            dimer_gradients.push(dimer_gradient_e0 + dimer_gradient_vrep);
        }
        else{
            pair_esdim_hash.insert(pair_index,esdim_iter);
            esdim_iter+=1;
        }
    }
    println!(
        "{:>68} {:>8.10} s",
        "elapsed time embedding:",
        molecule_timer.elapsed().as_secs_f32()
    );
    drop(molecule_timer);
    let molecule_timer: Instant = Instant::now();

    let grad_es_dim:Vec<Array1<f64>> = pair_results.par_iter().enumerate().filter_map(|(pair_index,pair)|if pair.energy_pair.is_none(){
        let dimer_natoms: usize =
            fragments[pair.frag_a_index].n_atoms + fragments[pair.frag_b_index].n_atoms;
        let dimer_gradient: Array1<f64> = Array::zeros(dimer_natoms * 3);
        let shape_orbs_a: usize = fragments[pair.frag_a_index].calculator.n_orbs;
        let shape_orbs_b: usize = fragments[pair.frag_b_index].calculator.n_orbs;
        let index_pair_a: usize = indices_frags[pair.frag_a_index];
        let index_pair_b: usize = indices_frags[pair.frag_b_index];

        // let g1_ab:Array3<f64> = pair_results[pair_index]
        //     .g1
        //     .slice(s![
        //         0..3 * fragments[pair.frag_a_index].n_atoms,
        //         0..fragments[pair.frag_a_index].n_atoms,
        //         fragments[pair.frag_a_index].n_atoms..
        //     ])
        //     .to_owned();
        //
        // let g1_ab_2:Array3<f64> = pair_results[pair_index]
        //     .g1
        //     .slice(s![
        //         3 * fragments[pair.frag_a_index].n_atoms..,
        //         fragments[pair.frag_a_index].n_atoms..,
        //         ..fragments[pair.frag_a_index].n_atoms
        //     ])
        //     .to_owned();

        // calculate grad s for each fragment of the pair
        let (grad_s_frag_a, grad_h0_frag_a): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
            &fragments[pair.frag_a_index].atomic_numbers,
            fragments[pair.frag_a_index].positions.view(),
            fragments[pair.frag_a_index].calculator.n_orbs,
            &fragments[pair.frag_a_index].calculator.valorbs,
            fragments[pair.frag_a_index].proximity_matrix.view(),
            &fragments[pair.frag_a_index].calculator.skt,
            &fragments[pair.frag_a_index].calculator.orbital_energies,
        );
        drop(grad_h0_frag_a);

        let (grad_s_frag_b, grad_h0_frag_b): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
            &fragments[pair.frag_b_index].atomic_numbers,
            fragments[pair.frag_b_index].positions.view(),
            fragments[pair.frag_b_index].calculator.n_orbs,
            &fragments[pair.frag_b_index].calculator.valorbs,
            fragments[pair.frag_b_index].proximity_matrix.view(),
            &fragments[pair.frag_b_index].calculator.skt,
            &fragments[pair.frag_b_index].calculator.orbital_energies,
        );
        drop(grad_h0_frag_b);

        // calculate g1 matrix for each dimer
        let dimer_distances: ArrayView2<f64> = dist_mat.slice(s![
                            index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
                            index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms
                        ]);

        let dimer_directions: ArrayView3<f64> = directions.slice(s![
                            index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
                            index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms,
                            ..
                        ]);
        let g1_dimer_ab: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
            &fragments[pair.frag_a_index].atomic_numbers,
            &fragments[pair.frag_b_index].atomic_numbers,
            fragments[pair.frag_a_index].n_atoms,
            fragments[pair.frag_b_index].n_atoms,
            dimer_distances,
            dimer_directions,
            full_hubbard,
            Some(0.0),
        );

        let g0_dimer_ab:Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
            &fragments[pair.frag_a_index].atomic_numbers,
            &fragments[pair.frag_b_index].atomic_numbers,
            fragments[pair.frag_a_index].n_atoms,
            fragments[pair.frag_b_index].n_atoms,
            dimer_distances,full_hubbard,
            Some(0.0));

        let dimer_distances: ArrayView2<f64> = dist_mat.slice(s![
                            index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms,
                            index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms
                        ]);

        let dimer_directions: ArrayView3<f64> = directions.slice(s![
                            index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms,
                            index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
                            ..
                        ]);

        let g1_dimer_ba: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
            &fragments[pair.frag_b_index].atomic_numbers,
            &fragments[pair.frag_a_index].atomic_numbers,
            fragments[pair.frag_b_index].n_atoms,
            fragments[pair.frag_a_index].n_atoms,
            dimer_distances,
            dimer_directions,
            full_hubbard,
            Some(0.0),
        );

        // assert!(
        //     g1_ab.abs_diff_eq(&g1_dimer_ab, 1e-12),
        //     "Gamma matrices are NOT equal!!!!"
        // );
        // assert!(
        //     g1_ab_2.abs_diff_eq(&g1_dimer_ba, 1e-12),
        //     "Gamma matrices are NOT equal!!!!"
        // );

        // let g0_ab:Array2<f64> = pair_results[pair_index]
        //     .g0
        //     .slice(s![
        //             0..fragments[pair.frag_a_index].n_atoms,
        //             fragments[pair.frag_a_index].n_atoms..
        //         ]).to_owned();

        // let g0_ab: ArrayView2<f64> = gamma_tmp.slice(s![
        //     index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
        //     index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms
        // ]);

        // assert!(
        //     g0_ab.abs_diff_eq(&g0_dimer_ab, 1e-12),
        //     "Gamma matrices are NOT equal!!!!"
        // );

        let mut w_mat_a: Array3<f64> = Array3::zeros((
            3 * fragments[pair.frag_a_index].n_atoms,
            shape_orbs_a,
            shape_orbs_a,
        ));
        for i in (0..3 * fragments[pair.frag_a_index].n_atoms).into_iter() {
            w_mat_a.slice_mut(s![i, .., ..]).assign(
                &fragments[pair.frag_a_index].final_p_matrix.dot(
                    &grad_s_frag_a
                        .slice(s![i, .., ..])
                        .dot(&fragments[pair.frag_a_index].final_p_matrix),
                ),
            );
        }
        w_mat_a = -0.5 * w_mat_a;

        let mut w_mat_b: Array3<f64> = Array3::zeros((
            3 * fragments[pair.frag_b_index].n_atoms,
            shape_orbs_b,
            shape_orbs_b,
        ));
        for i in (0..3 * fragments[pair.frag_b_index].n_atoms).into_iter() {
            w_mat_b.slice_mut(s![i, .., ..]).assign(
                &fragments[pair.frag_b_index].final_p_matrix.dot(
                    &grad_s_frag_b
                        .slice(s![i, .., ..])
                        .dot(&fragments[pair.frag_b_index].final_p_matrix),
                ),
            );
        }
        w_mat_b = -0.5 * w_mat_b;

        let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[pair.frag_a_index].n_atoms);
        let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[pair.frag_a_index].n_atoms);
        for dir in (0..3).into_iter() {
            let dir_xyz: usize = dir as usize;
            let mut diag_ind: usize = 0;
            for a in (0..fragments[pair.frag_a_index].n_atoms).into_iter() {
                let index: usize = 3 * a + dir_xyz;
                // term_1[index] = fragments[pair.frag_a_index].final_charges[a]
                //     * g1_ab
                //         .slice(s![index, a, ..])
                //         .dot(&fragments[pair.frag_b_index].final_charges);
                term_1[index] = fragments[pair.frag_a_index].final_charges[a]
                    * g1_dimer_ab
                    .slice(s![index, a, ..])
                    .dot(&fragments[pair.frag_b_index].final_charges);

                let atom_type: u8 = fragments[pair.frag_a_index].atomic_numbers[a];
                let norbs_a: usize =
                    fragments[pair.frag_a_index].calculator.valorbs[&atom_type].len();

                let tmp_1: Array2<f64> = w_mat_a
                    .slice(s![index, .., ..])
                    .dot(&frag_grad_results[pair.frag_a_index].s.t());
                let tmp_2: Array2<f64> = fragments[pair.frag_a_index].final_p_matrix.dot(
                    &grad_s_frag_a
                        .slice(s![index, .., ..])
                        .t(),
                );
                let sum: Array2<f64> = tmp_1 + tmp_2;

                let diag: f64 = sum.diag().slice(s![diag_ind..diag_ind + norbs_a]).sum();
                diag_ind += norbs_a;
                // println!("Sliced trace {}",diag);
                // println!("Full trace {}",sum.trace().unwrap());
                // let tmp_1:f64 = w_mat_a.slice(s![index,..,..]).dot(&frag_grad_results[pair.frag_a_index].s.t()).trace().unwrap();
                // let tmp_2:f64 = fragments[pair.frag_a_index].final_p_matrix.dot(&frag_grad_results[pair.frag_a_index].grad_s.slice(s![index,..,..]).t()).trace().unwrap();
                // term_2[index] = (tmp_1 + tmp_2) * g0_ab.slice(s![a,..]).dot(&fragments[pair.frag_b_index].final_charges);
                // term_2[index] = diag
                //     * g0_ab
                //         .slice(s![a, ..])
                //         .dot(&fragments[pair.frag_b_index].final_charges);
                term_2[index] = diag
                    * g0_dimer_ab
                    .slice(s![a, ..])
                    .dot(&fragments[pair.frag_b_index].final_charges);
            }
        }
        let gradient_frag_a: Array1<f64> = term_1 + term_2;

        let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[pair.frag_b_index].n_atoms);
        let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[pair.frag_b_index].n_atoms);
        for dir in (0..3).into_iter() {
            let dir_xyz: usize = dir as usize;
            let mut diag_ind: usize = 0;
            for a in (0..fragments[pair.frag_b_index].n_atoms).into_iter() {
                let index: usize = 3 * a + dir_xyz;
                // term_1[index] = fragments[pair.frag_b_index].final_charges[a]
                //     * g1_ab_2
                //         .slice(s![index, a, ..])
                //         .dot(&fragments[pair.frag_a_index].final_charges);
                term_1[index] = fragments[pair.frag_b_index].final_charges[a]
                    * g1_dimer_ba
                    .slice(s![index, a, ..])
                    .dot(&fragments[pair.frag_a_index].final_charges);

                let atom_type: u8 = fragments[pair.frag_b_index].atomic_numbers[a];
                let norbs_b: usize =
                    fragments[pair.frag_b_index].calculator.valorbs[&atom_type].len();

                let tmp_1: Array2<f64> = w_mat_b
                    .slice(s![index, .., ..])
                    .dot(&frag_grad_results[pair.frag_b_index].s.t());
                let tmp_2: Array2<f64> = fragments[pair.frag_b_index].final_p_matrix.dot(
                    &grad_s_frag_b
                        .slice(s![index, .., ..])
                        .t(),
                );
                let sum: Array2<f64> = tmp_1 + tmp_2;

                let diag: f64 = sum.diag().slice(s![diag_ind..diag_ind + norbs_b]).sum();
                // println!("Sliced trace {}",diag);
                // println!("Full trace {}",sum.trace().unwrap());

                diag_ind += norbs_b;
                // let tmp_1:f64 = w_mat_b.slice(s![index,..,..]).dot(&frag_grad_results[pair.frag_b_index].s.t()).trace().unwrap();
                // let tmp_2:f64 = fragments[pair.frag_b_index].final_p_matrix.dot(&frag_grad_results[pair.frag_b_index].grad_s.slice(s![index,..,..]).t()).trace().unwrap();
                // term_2[index] = (tmp_1 + tmp_2) * g0_ab.slice(s![..,a]).dot(&fragments[pair.frag_a_index].final_charges);
                // term_2[index] = diag
                //     * g0_ab
                //         .slice(s![.., a])
                //         .dot(&fragments[pair.frag_a_index].final_charges);
                term_2[index] = diag
                    * g0_dimer_ab
                    .slice(s![.., a])
                    .dot(&fragments[pair.frag_a_index].final_charges);
            }
        }
        let gradient_frag_b: Array1<f64> = term_1 + term_2;
        let mut dimer_gradient: Vec<f64> = Vec::new();
        dimer_gradient.append(&mut gradient_frag_a.to_vec());
        dimer_gradient.append(&mut gradient_frag_b.to_vec());

        let dimer_gradient: Array1<f64> = Array::from(dimer_gradient);
        // dimer_gradients.push(dimer_gradient);
        Some(dimer_gradient)
    }
    else{
        None
    }).collect();

    println!(
        "{:>68} {:>8.10} s",
        "elapsed time ESDIM:",
        molecule_timer.elapsed().as_secs_f32()
    );
    drop(molecule_timer);

        // if pair.energy_pair.is_some() {
        //     let dimer_gradient_e0: Array1<f64> = pair.grad_e0.clone().unwrap();
        //     let dimer_gradient_vrep: Array1<f64> = pair.grad_vrep.clone().unwrap();
        //
        //     let dimer_pmat: Array2<f64> = pair.p_mat.clone().unwrap();
        //
        //     let pair_atoms: usize =
        //         fragments[pair.frag_a_index].n_atoms + fragments[pair.frag_b_index].n_atoms;
        //     let pair_charges = pair.pair_charges.clone().unwrap();
        //     let pair_smat: Array2<f64> = pair.s.clone().unwrap();
        //     let pair_grads: Array3<f64> = pair.grad_s.clone().unwrap();
        //     let ddq_vec: Vec<f64> = (0..pair_atoms)
        //         .into_iter()
        //         .map(|a| {
        //             let mut ddq: f64 = 0.0;
        //             if a < pair.frag_a_atoms {
        //                 ddq = pair_charges[a] - fragments[pair.frag_a_index].final_charges[a];
        //             } else {
        //                 ddq = pair_charges[a]
        //                     - fragments[pair.frag_b_index].final_charges[a - pair.frag_a_atoms];
        //             }
        //             ddq
        //         })
        //         .collect();
        //
        //     let index_pair_iter: usize = indices_frags[pair.frag_a_index];
        //     let ddq_arr: Array1<f64> = Array::from(ddq_vec);
        //     let shape_orbs_dimer: usize = pair.grad_s.clone().unwrap().dim().1;
        //     let shape_orbs_a: usize = frag_grad_results[pair.frag_a_index].grad_s.dim().1;
        //     let shape_orbs_b: usize = frag_grad_results[pair.frag_b_index].grad_s.dim().1;
        //
        //     let mut w_dimer: Array3<f64> =
        //         Array3::zeros((3 * pair_atoms, shape_orbs_dimer, shape_orbs_dimer));
        //     for i in (0..3 * pair_atoms).into_iter() {
        //         w_dimer
        //             .slice_mut(s![i, .., ..])
        //             .assign(&dimer_pmat.dot(&pair_grads.slice(s![i, .., ..]).dot(&dimer_pmat)));
        //     }
        //     w_dimer = -0.5 * w_dimer;
        //
        //     let mut w_mat_a: Array3<f64> = Array3::zeros((
        //         3 * fragments[pair.frag_a_index].n_atoms,
        //         shape_orbs_a,
        //         shape_orbs_a,
        //     ));
        //     for i in (0..3 * fragments[pair.frag_a_index].n_atoms).into_iter() {
        //         w_mat_a.slice_mut(s![i, .., ..]).assign(
        //             &fragments[pair.frag_a_index].final_p_matrix.dot(
        //                 &frag_grad_results[pair.frag_a_index]
        //                     .grad_s
        //                     .slice(s![i, .., ..])
        //                     .dot(&fragments[pair.frag_a_index].final_p_matrix),
        //             ),
        //         );
        //     }
        //     w_mat_a = -0.5 * w_mat_a;
        //
        //     let mut w_mat_b: Array3<f64> = Array3::zeros((
        //         3 * fragments[pair.frag_b_index].n_atoms,
        //         shape_orbs_b,
        //         shape_orbs_b,
        //     ));
        //     for i in (0..3 * fragments[pair.frag_b_index].n_atoms).into_iter() {
        //         w_mat_b.slice_mut(s![i, .., ..]).assign(
        //             &fragments[pair.frag_b_index].final_p_matrix.dot(
        //                 &frag_grad_results[pair.frag_b_index]
        //                     .grad_s
        //                     .slice(s![i, .., ..])
        //                     .dot(&fragments[pair.frag_b_index].final_p_matrix),
        //             ),
        //         );
        //     }
        //     w_mat_b = -0.5 * w_mat_b;
        //
        //     // Build delta p_mu,nu^I,J
        //     let mut dp_direct_sum_monomer: Array2<f64> = Array2::zeros(dimer_pmat.raw_dim());
        //     let p_dim_monomer: usize = fragments[pair.frag_a_index].final_p_matrix.dim().0;
        //     dp_direct_sum_monomer
        //         .slice_mut(s![0..p_dim_monomer, 0..p_dim_monomer])
        //         .assign(&fragments[pair.frag_a_index].final_p_matrix);
        //     dp_direct_sum_monomer
        //         .slice_mut(s![p_dim_monomer.., p_dim_monomer..])
        //         .assign(&fragments[pair.frag_b_index].final_p_matrix);
        //     let dp_dimer: Array2<f64> = &dimer_pmat - &dp_direct_sum_monomer;
        //
        //     // Build delta W_mu,nu^I,J
        //     let mut dw_dimer: Array3<f64> = Array3::zeros(w_dimer.raw_dim());
        //     let w_dimer_dim: usize = w_dimer.dim().1;
        //     let mut dw_dimer_vec: Vec<Array1<f64>> = (0..3 * pair_atoms)
        //         .into_iter()
        //         .map(|a| {
        //             let mut w_a_dimer: Array2<f64> = Array::zeros((w_dimer_dim, w_dimer_dim));
        //             if a < 3 * pair.frag_a_atoms {
        //                 w_a_dimer
        //                     .slice_mut(s![0..p_dim_monomer, 0..p_dim_monomer])
        //                     .assign(&w_mat_a.slice(s![a, .., ..]));
        //             } else {
        //                 w_a_dimer
        //                     .slice_mut(s![p_dim_monomer.., p_dim_monomer..])
        //                     .assign(&w_mat_b.slice(s![a - 3 * pair.frag_a_atoms, .., ..]));
        //             }
        //             let w_return: Array2<f64> = w_dimer.slice(s![a, .., ..]).to_owned() - w_a_dimer;
        //             let w_flat: Array1<f64> =
        //                 w_return.into_shape((w_dimer_dim * w_dimer_dim)).unwrap();
        //             w_flat
        //         })
        //         .collect();
        //
        //     // transform dW from flat array to 3d array
        //     let mut dw_dimer_vec_flat: Vec<f64> = Vec::new();
        //     for a in dw_dimer_vec.iter() {
        //         dw_dimer_vec_flat.append(&mut a.to_vec());
        //     }
        //     let dw_dimer: Array3<f64> = Array::from(dw_dimer_vec_flat)
        //         .into_shape((3 * pair_atoms, w_dimer_dim, w_dimer_dim))
        //         .unwrap();
        //
        //     // let molecule_timer: Instant = Instant::now();
        //
        //     let embedding_pot: Vec<Array1<f64>> = fragments
        //         .iter()
        //         .enumerate()
        //         .filter_map(|(ind_k, mol_k)| {
        //             if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index {
        //                 let index_frag_iter: usize = indices_frags[ind_k];
        //
        //                 // let dgamma_ac: ArrayView3<f64> = g1_total.slice(s![
        //                 //     3*index_pair_iter..3*index_pair_iter + 3 * pair_atoms,
        //                 //     index_pair_iter..index_pair_iter + pair_atoms,
        //                 //     index_frag_iter..index_frag_iter + mol_k.n_atoms
        //                 // ]);
        //                 //
        //                 // let dgamma_ac_k: ArrayView3<f64> = g1_total.slice(s![
        //                 //     3*index_frag_iter..3*index_frag_iter + 3 * mol_k.n_atoms,
        //                 //     index_frag_iter..index_frag_iter + mol_k.n_atoms,
        //                 //     index_pair_iter..index_pair_iter + pair_atoms
        //                 // ]);
        //
        //                 // calculate g1 matrix for each trimer
        //                 let mut dimer_atomic_numbers: Vec<u8> = Vec::new();
        //                 dimer_atomic_numbers
        //                     .append(&mut fragments[pair.frag_a_index].atomic_numbers.clone());
        //                 dimer_atomic_numbers
        //                     .append(&mut fragments[pair.frag_b_index].atomic_numbers.clone());
        //
        //                 let trimer_distances: ArrayView2<f64> = dist_mat.slice(s![
        //                     index_pair_iter..index_pair_iter + pair_atoms,
        //                     index_frag_iter..index_frag_iter + mol_k.n_atoms
        //                 ]);
        //
        //                 let trimer_directions: ArrayView3<f64> = directions.slice(s![
        //                     index_pair_iter..index_pair_iter + pair_atoms,
        //                     index_frag_iter..index_frag_iter + mol_k.n_atoms,
        //                     ..
        //                 ]);
        //                 let g1_trimer_ak: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
        //                     &dimer_atomic_numbers,
        //                     &mol_k.atomic_numbers,
        //                     pair_atoms,
        //                     mol_k.n_atoms,
        //                     trimer_distances,
        //                     trimer_directions,
        //                     full_hubbard,
        //                     Some(0.0),
        //                 );
        //
        //                 let g0_trimer_ak:Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
        //                     &dimer_atomic_numbers,
        //                     &mol_k.atomic_numbers,
        //                     pair_atoms,
        //                     mol_k.n_atoms,
        //                     trimer_distances,
        //                     full_hubbard,
        //                     Some(0.0));
        //
        //                 let trimer_distances: ArrayView2<f64> = dist_mat.slice(s![
        //                     index_frag_iter..index_frag_iter + mol_k.n_atoms,
        //                     index_pair_iter..index_pair_iter + pair_atoms
        //                 ]);
        //
        //                 let trimer_directions: ArrayView3<f64> = directions.slice(s![
        //                     index_frag_iter..index_frag_iter + mol_k.n_atoms,
        //                     index_pair_iter..index_pair_iter + pair_atoms,
        //                     ..
        //                 ]);
        //
        //                 let g1_trimer_ka: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
        //                     &mol_k.atomic_numbers,
        //                     &dimer_atomic_numbers,
        //                     mol_k.n_atoms,
        //                     pair_atoms,
        //                     trimer_distances,
        //                     trimer_directions,
        //                     full_hubbard,
        //                     Some(0.0),
        //                 );
        //
        //                 // println!("g1 ac k {}",dgamma_ac_k.clone().slice(s![0,..,..]));
        //                 // println!("g1 ac k 2 {}",dgamma_ac_2.clone().slice(s![0,..,..]));
        //                 // println!("g1 ac k {}",dgamma_ac.clone().slice(s![1,..,..]));
        //                 // println!("g1 ac k 2 {}",dgamma_ac_2.clone().slice(s![1,..,..]));
        //                 // assert!(
        //                 //     dgamma_ac.abs_diff_eq(&g1_trimer_ak, 1e-12),
        //                 //     "Gamma matrices are NOT equal!!!!"
        //                 // );
        //                 // assert!(
        //                 //     dgamma_ac_k.abs_diff_eq(&g1_trimer_ka, 1e-12),
        //                 //     "Gamma matrices are NOT equal!!!!"
        //                 // );
        //
        //                 // let g0_ab: ArrayView2<f64> = gamma_tmp.slice(s![
        //                 //     index_pair_iter..index_pair_iter + pair_atoms,
        //                 //     index_frag_iter..index_frag_iter + mol_k.n_atoms
        //                 // ]);
        //                 //
        //                 // assert!(
        //                 //     g0_ab.abs_diff_eq(&g0_trimer_ak, 1e-12),
        //                 //     "Gamma matrices are NOT equal!!!!"
        //                 // );
        //
        //                 let mut term_1: Array1<f64> = Array1::zeros(3 * pair_atoms);
        //                 let mut term_2: Array1<f64> = Array1::zeros(3 * pair_atoms);
        //                 for dir in (0..3).into_iter() {
        //                     let dir_xyz: usize = dir as usize;
        //                     let mut diag_ind: usize = 0;
        //                     for a in (0..pair_atoms).into_iter() {
        //                         let index: usize = 3 * a + dir_xyz;
        //                         // term_1[index] = ddq_arr[a]
        //                         //     * dgamma_ac.slice(s![index, a, ..]).dot(&mol_k.final_charges);
        //                         term_1[index] = ddq_arr[a]
        //                             * g1_trimer_ak.slice(s![index, a, ..]).dot(&mol_k.final_charges);
        //
        //                         let mut atom_type: u8 = 0;
        //                         let mut norbs_a: usize = 0;
        //
        //                         if a < pair.frag_a_atoms {
        //                             atom_type = fragments[pair.frag_a_index].atomic_numbers[a];
        //                             norbs_a = fragments[pair.frag_a_index].calculator.valorbs
        //                                 [&atom_type]
        //                                 .len();
        //                         } else {
        //                             atom_type = fragments[pair.frag_b_index].atomic_numbers
        //                                 [a - pair.frag_a_atoms];
        //                             norbs_a = fragments[pair.frag_b_index].calculator.valorbs
        //                                 [&atom_type]
        //                                 .len();
        //                         }
        //
        //                         let tmp_1: Array2<f64> =
        //                             dw_dimer.slice(s![index, .., ..]).dot(&pair_smat.t());
        //                         let tmp_2: Array2<f64> =
        //                             dp_dimer.dot(&pair_grads.slice(s![index, .., ..]).t());
        //                         let sum: Array2<f64> = tmp_1 + tmp_2;
        //
        //                         let diag: f64 =
        //                             sum.diag().slice(s![diag_ind..diag_ind + norbs_a]).sum();
        //                         diag_ind += norbs_a;
        //                         // let tmp_1:f64 = dw_dimer.slice(s![index,..,..]).dot(&pair_smat.t()).trace().unwrap();
        //                         // let tmp_2:f64 = dp_dimer.dot(&pair_grads.slice(s![index,..,..]).t()).trace().unwrap();
        //                         // term_2[index] = (tmp_1 + tmp_2) * g0_ab.slice(s![a,..]).dot(&fragments[ind_k].final_charges);
        //                         // term_2[index] = diag
        //                         //     * g0_ab.slice(s![a, ..]).dot(&fragments[ind_k].final_charges);
        //                         term_2[index] = diag
        //                             * g0_trimer_ak.slice(s![a, ..]).dot(&fragments[ind_k].final_charges);
        //                     }
        //                 }
        //                 let embedding_part_1: Array1<f64> = term_1 + term_2;
        //
        //                 let shape_orbs_k: usize = frag_grad_results[ind_k].grad_s.dim().1;
        //                 let mut w_mat_k: Array3<f64> = Array3::zeros((
        //                     3 * fragments[ind_k].n_atoms,
        //                     shape_orbs_k,
        //                     shape_orbs_k,
        //                 ));
        //                 for i in (0..3 * fragments[ind_k].n_atoms).into_iter() {
        //                     w_mat_k.slice_mut(s![i, .., ..]).assign(
        //                         &fragments[ind_k].final_p_matrix.dot(
        //                             &frag_grad_results[ind_k]
        //                                 .grad_s
        //                                 .slice(s![i, .., ..])
        //                                 .dot(&fragments[ind_k].final_p_matrix),
        //                         ),
        //                     );
        //                 }
        //                 w_mat_k = -0.5 * w_mat_k;
        //
        //                 let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[ind_k].n_atoms);
        //                 let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[ind_k].n_atoms);
        //                 for dir in (0..3).into_iter() {
        //                     let dir_xyz: usize = dir as usize;
        //                     let mut diag_ind: usize = 0;
        //                     for a in (0..fragments[ind_k].n_atoms).into_iter() {
        //                         let index: usize = 3 * a + dir_xyz;
        //                         // term_1[index] = fragments[ind_k].final_charges[a]
        //                         //     * dgamma_ac_k.slice(s![index, a, ..]).dot(&ddq_arr);
        //                         term_1[index] = fragments[ind_k].final_charges[a]
        //                             * g1_trimer_ka.slice(s![index, a, ..]).dot(&ddq_arr);
        //
        //                         let atom_type: u8 = fragments[ind_k].atomic_numbers[a];
        //                         let norbs_k: usize =
        //                             fragments[ind_k].calculator.valorbs[&atom_type].len();
        //
        //                         let tmp_1: Array2<f64> = w_mat_k
        //                             .slice(s![index, .., ..])
        //                             .dot(&frag_grad_results[ind_k].s.t());
        //                         let tmp_2: Array2<f64> = fragments[ind_k].final_p_matrix.dot(
        //                             &frag_grad_results[ind_k].grad_s.slice(s![index, .., ..]).t(),
        //                         );
        //
        //                         let sum: Array2<f64> = tmp_1 + tmp_2;
        //
        //                         let diag: f64 =
        //                             sum.diag().slice(s![diag_ind..diag_ind + norbs_k]).sum();
        //                         diag_ind += norbs_k;
        //                         // let tmp_1:f64 = w_mat_k.slice(s![index,..,..]).dot(&frag_grad_results[ind_k].s.t()).trace().unwrap();
        //                         // let tmp_2:f64 = fragments[ind_k].final_p_matrix.dot(&frag_grad_results[ind_k].grad_s.slice(s![index,..,..]).t()).trace().unwrap();
        //                         // term_2[index] = (tmp_1 + tmp_2) * g0_ab.slice(s![..,a]).dot(&ddq_arr);
        //                         // term_2[index] = diag * g0_ab.slice(s![.., a]).dot(&ddq_arr);
        //                         term_2[index] = diag * g0_trimer_ak.slice(s![.., a]).dot(&ddq_arr);
        //                     }
        //                 }
        //                 let embedding_part_2: Array1<f64> = term_1 + term_2;
        //                 // let term_1: Array1<f64> = dgamma_ac
        //                 //     .clone()
        //                 //     .into_shape((3 * pair_atoms * pair_atoms, mol_k.n_atoms))
        //                 //     .unwrap()
        //                 //     .dot(&mol_k.final_charges)
        //                 //     .into_shape((3 * pair_atoms, pair_atoms))
        //                 //     .unwrap()
        //                 //     .dot(&ddq_arr);
        //                 //
        //                 // let dw_s_a: Array1<f64> = dw_dimer
        //                 //     .clone()
        //                 //     .into_shape((3 * pair_atoms * w_dimer_dim, w_dimer_dim))
        //                 //     .unwrap()
        //                 //     .dot(&pair_smat)
        //                 //     .into_shape((3 * pair_atoms, w_dimer_dim, w_dimer_dim))
        //                 //     .unwrap()
        //                 //     .sum_axis(Axis(2))
        //                 //     .sum_axis(Axis(1));
        //                 //
        //                 // let mut dp_grads_a:Array3<f64> = Array3::zeros((3*pair_atoms,w_dimer_dim,w_dimer_dim));
        //                 // for i in (0..3*pair_atoms).into_iter(){
        //                 //     dp_grads_a.slice_mut(s![i,..,..]).assign(&dp_dimer.dot(&pair_grads.slice(s![i,..,..])));
        //                 // }
        //                 //
        //                 // let dp_grads_sum_a:Array1<f64> = dp_grads_a.sum_axis(Axis(2)).sum_axis(Axis(1));
        //                 // let term_2: Array1<f64> = (dw_s_a + dp_grads_sum_a)
        //                 //     * gamma_tmp
        //                 //         .slice(s![
        //                 //             index_pair_iter..index_pair_iter + pair_atoms,
        //                 //             index_frag_iter..index_frag_iter + mol_k.n_atoms
        //                 //         ])
        //                 //         .dot(&fragments[ind_k].final_charges)
        //                 //         .sum();
        //                 // let dgamma_ac: Array3<f64> = g1_total
        //                 //     .slice(s![
        //                 //         index_frag_iter..index_frag_iter + 3 * mol_k.n_atoms,
        //                 //         index_pair_iter..index_pair_iter + pair_atoms,
        //                 //         index_frag_iter..index_frag_iter + mol_k.n_atoms
        //                 //     ])
        //                 //     .to_owned();
        //                 //
        //                 // let term_1_k: Array1<f64> = dgamma_ac
        //                 //     .into_shape((3 * mol_k.n_atoms * pair_atoms, mol_k.n_atoms))
        //                 //     .unwrap()
        //                 //     .dot(&mol_k.final_charges)
        //                 //     .into_shape((3 * mol_k.n_atoms, pair_atoms))
        //                 //     .unwrap()
        //                 //     .dot(&ddq_arr);
        //                 // let w_s_k: Array1<f64> = w_mat_k
        //                 //     .into_shape((3 * fragments[ind_k].n_atoms * shape_orbs_k, shape_orbs_k))
        //                 //     .unwrap()
        //                 //     .dot(&frag_grad_results[ind_k].s)
        //                 //     .into_shape((3 * fragments[ind_k].n_atoms, shape_orbs_k, shape_orbs_k))
        //                 //     .unwrap()
        //                 //     .sum_axis(Axis(2))
        //                 //     .sum_axis(Axis(1));
        //                 //
        //                 // let mut p_grads_k:Array3<f64> = Array3::zeros((3 * fragments[ind_k].n_atoms,shape_orbs_k,shape_orbs_k));
        //                 // for i in (0..3 * fragments[ind_k].n_atoms).into_iter(){
        //                 //     p_grads_k.slice_mut(s![i,..,..]).assign(&fragments[ind_k].final_p_matrix.dot(&frag_grad_results[ind_k].grad_s.slice(s![i,..,..])));
        //                 // }
        //                 //
        //                 // let p_grads_sum_k:Array1<f64> =p_grads_k.sum_axis(Axis(2)).sum_axis(Axis(1));
        //                 //
        //                 // let term_2_k: Array1<f64> = (w_s_k + p_grads_sum_k)
        //                 //     * ddq_arr
        //                 //         .dot(&gamma_tmp.slice(s![
        //                 //             index_pair_iter..index_pair_iter + pair_atoms,
        //                 //             index_frag_iter..index_frag_iter + mol_k.n_atoms
        //                 //         ]))
        //                 //         .sum();
        //                 let mut temp_grad: Array1<f64> = Array1::zeros(grad_e0_monomers.len());
        //
        //                 let index_a: usize = pair.frag_a_index;
        //                 let index_b: usize = pair.frag_b_index;
        //                 let atoms_a: usize = fragments[index_a].n_atoms;
        //                 let atoms_b: usize = fragments[index_b].n_atoms;
        //                 let index_frag_a: usize = indices_frags[index_a];
        //                 let index_frag_b: usize = indices_frags[index_b];
        //
        //                 temp_grad
        //                     .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
        //                     .add_assign(&embedding_part_1.slice(s![0..3 * atoms_a]));
        //                 temp_grad
        //                     .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
        //                     .add_assign(&embedding_part_1.slice(s![3 * atoms_a..]));
        //                 temp_grad
        //                     .slice_mut(s![
        //                         3 * index_frag_iter..3 * index_frag_iter + 3 * mol_k.n_atoms
        //                     ])
        //                     .add_assign(&embedding_part_2);
        //
        //                 Some(temp_grad)
        //             } else {
        //                 None
        //             }
        //         })
        //         .collect();
        //     // println!(
        //     //     "{:>68} {:>8.10} s",
        //     //     "elapsed time embedding:",
        //     //     molecule_timer.elapsed().as_secs_f32()
        //     // );
        //     // drop(molecule_timer);
        //
        //     let mut embedding_gradient: Array1<f64> = Array1::zeros(grad_e0_monomers.len());
        //     for grad in embedding_pot.iter() {
        //         embedding_gradient.add_assign(grad);
        //     }
        //     embedding_gradients.push(embedding_gradient);
        //     dimer_gradients.push(dimer_gradient_e0 + dimer_gradient_vrep);
    //     } else {
    //         // let molecule_timer: Instant = Instant::now();
    //
    //         let dimer_natoms: usize =
    //             fragments[pair.frag_a_index].n_atoms + fragments[pair.frag_b_index].n_atoms;
    //         let dimer_gradient: Array1<f64> = Array::zeros(dimer_natoms * 3);
    //         let shape_orbs_a: usize = frag_grad_results[pair.frag_a_index].grad_s.dim().1;
    //         let shape_orbs_b: usize = frag_grad_results[pair.frag_b_index].grad_s.dim().1;
    //         let index_pair_a: usize = indices_frags[pair.frag_a_index];
    //         let index_pair_b: usize = indices_frags[pair.frag_b_index];
    //
    //         // let g1_ab:Array3<f64> = pair_results[pair_index]
    //         //     .g1
    //         //     .slice(s![
    //         //         0..3 * fragments[pair.frag_a_index].n_atoms,
    //         //         0..fragments[pair.frag_a_index].n_atoms,
    //         //         fragments[pair.frag_a_index].n_atoms..
    //         //     ])
    //         //     .to_owned();
    //         //
    //         // let g1_ab_2:Array3<f64> = pair_results[pair_index]
    //         //     .g1
    //         //     .slice(s![
    //         //         3 * fragments[pair.frag_a_index].n_atoms..,
    //         //         fragments[pair.frag_a_index].n_atoms..,
    //         //         ..fragments[pair.frag_a_index].n_atoms
    //         //     ])
    //         //     .to_owned();
    //
    //         // calculate g1 matrix for each dimer
    //         let dimer_distances: ArrayView2<f64> = dist_mat.slice(s![
    //                         index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
    //                         index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms
    //                     ]);
    //
    //         let dimer_directions: ArrayView3<f64> = directions.slice(s![
    //                         index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
    //                         index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms,
    //                         ..
    //                     ]);
    //         let g1_dimer_ab: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
    //             &fragments[pair.frag_a_index].atomic_numbers,
    //             &fragments[pair.frag_b_index].atomic_numbers,
    //             fragments[pair.frag_a_index].n_atoms,
    //             fragments[pair.frag_b_index].n_atoms,
    //             dimer_distances,
    //             dimer_directions,
    //             full_hubbard,
    //             Some(0.0),
    //         );
    //
    //         let g0_dimer_ab:Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
    //             &fragments[pair.frag_a_index].atomic_numbers,
    //             &fragments[pair.frag_b_index].atomic_numbers,
    //             fragments[pair.frag_a_index].n_atoms,
    //             fragments[pair.frag_b_index].n_atoms,
    //             dimer_distances,full_hubbard,
    //             Some(0.0));
    //
    //         let dimer_distances: ArrayView2<f64> = dist_mat.slice(s![
    //                         index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms,
    //                         index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms
    //                     ]);
    //
    //         let dimer_directions: ArrayView3<f64> = directions.slice(s![
    //                         index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms,
    //                         index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
    //                         ..
    //                     ]);
    //
    //         let g1_dimer_ba: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
    //             &fragments[pair.frag_b_index].atomic_numbers,
    //             &fragments[pair.frag_a_index].atomic_numbers,
    //             fragments[pair.frag_b_index].n_atoms,
    //             fragments[pair.frag_a_index].n_atoms,
    //             dimer_distances,
    //             dimer_directions,
    //             full_hubbard,
    //             Some(0.0),
    //         );
    //
    //         // assert!(
    //         //     g1_ab.abs_diff_eq(&g1_dimer_ab, 1e-12),
    //         //     "Gamma matrices are NOT equal!!!!"
    //         // );
    //         // assert!(
    //         //     g1_ab_2.abs_diff_eq(&g1_dimer_ba, 1e-12),
    //         //     "Gamma matrices are NOT equal!!!!"
    //         // );
    //
    //         // let g0_ab:Array2<f64> = pair_results[pair_index]
    //         //     .g0
    //         //     .slice(s![
    //         //             0..fragments[pair.frag_a_index].n_atoms,
    //         //             fragments[pair.frag_a_index].n_atoms..
    //         //         ]).to_owned();
    //
    //         // let g0_ab: ArrayView2<f64> = gamma_tmp.slice(s![
    //         //     index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
    //         //     index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms
    //         // ]);
    //
    //         // assert!(
    //         //     g0_ab.abs_diff_eq(&g0_dimer_ab, 1e-12),
    //         //     "Gamma matrices are NOT equal!!!!"
    //         // );
    //
    //         let mut w_mat_a: Array3<f64> = Array3::zeros((
    //             3 * fragments[pair.frag_a_index].n_atoms,
    //             shape_orbs_a,
    //             shape_orbs_a,
    //         ));
    //         for i in (0..3 * fragments[pair.frag_a_index].n_atoms).into_iter() {
    //             w_mat_a.slice_mut(s![i, .., ..]).assign(
    //                 &fragments[pair.frag_a_index].final_p_matrix.dot(
    //                     &frag_grad_results[pair.frag_a_index]
    //                         .grad_s
    //                         .slice(s![i, .., ..])
    //                         .dot(&fragments[pair.frag_a_index].final_p_matrix),
    //                 ),
    //             );
    //         }
    //         w_mat_a = -0.5 * w_mat_a;
    //
    //         let mut w_mat_b: Array3<f64> = Array3::zeros((
    //             3 * fragments[pair.frag_b_index].n_atoms,
    //             shape_orbs_b,
    //             shape_orbs_b,
    //         ));
    //         for i in (0..3 * fragments[pair.frag_b_index].n_atoms).into_iter() {
    //             w_mat_b.slice_mut(s![i, .., ..]).assign(
    //                 &fragments[pair.frag_b_index].final_p_matrix.dot(
    //                     &frag_grad_results[pair.frag_b_index]
    //                         .grad_s
    //                         .slice(s![i, .., ..])
    //                         .dot(&fragments[pair.frag_b_index].final_p_matrix),
    //                 ),
    //             );
    //         }
    //         w_mat_b = -0.5 * w_mat_b;
    //
    //         let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[pair.frag_a_index].n_atoms);
    //         let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[pair.frag_a_index].n_atoms);
    //         for dir in (0..3).into_iter() {
    //             let dir_xyz: usize = dir as usize;
    //             let mut diag_ind: usize = 0;
    //             for a in (0..fragments[pair.frag_a_index].n_atoms).into_iter() {
    //                 let index: usize = 3 * a + dir_xyz;
    //                 // term_1[index] = fragments[pair.frag_a_index].final_charges[a]
    //                 //     * g1_ab
    //                 //         .slice(s![index, a, ..])
    //                 //         .dot(&fragments[pair.frag_b_index].final_charges);
    //                 term_1[index] = fragments[pair.frag_a_index].final_charges[a]
    //                     * g1_dimer_ab
    //                     .slice(s![index, a, ..])
    //                     .dot(&fragments[pair.frag_b_index].final_charges);
    //
    //                 let atom_type: u8 = fragments[pair.frag_a_index].atomic_numbers[a];
    //                 let norbs_a: usize =
    //                     fragments[pair.frag_a_index].calculator.valorbs[&atom_type].len();
    //
    //                 let tmp_1: Array2<f64> = w_mat_a
    //                     .slice(s![index, .., ..])
    //                     .dot(&frag_grad_results[pair.frag_a_index].s.t());
    //                 let tmp_2: Array2<f64> = fragments[pair.frag_a_index].final_p_matrix.dot(
    //                     &frag_grad_results[pair.frag_a_index]
    //                         .grad_s
    //                         .slice(s![index, .., ..])
    //                         .t(),
    //                 );
    //                 let sum: Array2<f64> = tmp_1 + tmp_2;
    //
    //                 let diag: f64 = sum.diag().slice(s![diag_ind..diag_ind + norbs_a]).sum();
    //                 diag_ind += norbs_a;
    //                 // println!("Sliced trace {}",diag);
    //                 // println!("Full trace {}",sum.trace().unwrap());
    //                 // let tmp_1:f64 = w_mat_a.slice(s![index,..,..]).dot(&frag_grad_results[pair.frag_a_index].s.t()).trace().unwrap();
    //                 // let tmp_2:f64 = fragments[pair.frag_a_index].final_p_matrix.dot(&frag_grad_results[pair.frag_a_index].grad_s.slice(s![index,..,..]).t()).trace().unwrap();
    //                 // term_2[index] = (tmp_1 + tmp_2) * g0_ab.slice(s![a,..]).dot(&fragments[pair.frag_b_index].final_charges);
    //                 // term_2[index] = diag
    //                 //     * g0_ab
    //                 //         .slice(s![a, ..])
    //                 //         .dot(&fragments[pair.frag_b_index].final_charges);
    //                 term_2[index] = diag
    //                     * g0_dimer_ab
    //                     .slice(s![a, ..])
    //                     .dot(&fragments[pair.frag_b_index].final_charges);
    //             }
    //         }
    //         let gradient_frag_a: Array1<f64> = term_1 + term_2;
    //
    //         let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[pair.frag_b_index].n_atoms);
    //         let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[pair.frag_b_index].n_atoms);
    //         for dir in (0..3).into_iter() {
    //             let dir_xyz: usize = dir as usize;
    //             let mut diag_ind: usize = 0;
    //             for a in (0..fragments[pair.frag_b_index].n_atoms).into_iter() {
    //                 let index: usize = 3 * a + dir_xyz;
    //                 // term_1[index] = fragments[pair.frag_b_index].final_charges[a]
    //                 //     * g1_ab_2
    //                 //         .slice(s![index, a, ..])
    //                 //         .dot(&fragments[pair.frag_a_index].final_charges);
    //                 term_1[index] = fragments[pair.frag_b_index].final_charges[a]
    //                     * g1_dimer_ba
    //                     .slice(s![index, a, ..])
    //                     .dot(&fragments[pair.frag_a_index].final_charges);
    //
    //                 let atom_type: u8 = fragments[pair.frag_b_index].atomic_numbers[a];
    //                 let norbs_b: usize =
    //                     fragments[pair.frag_b_index].calculator.valorbs[&atom_type].len();
    //
    //                 let tmp_1: Array2<f64> = w_mat_b
    //                     .slice(s![index, .., ..])
    //                     .dot(&frag_grad_results[pair.frag_b_index].s.t());
    //                 let tmp_2: Array2<f64> = fragments[pair.frag_b_index].final_p_matrix.dot(
    //                     &frag_grad_results[pair.frag_b_index]
    //                         .grad_s
    //                         .slice(s![index, .., ..])
    //                         .t(),
    //                 );
    //                 let sum: Array2<f64> = tmp_1 + tmp_2;
    //
    //                 let diag: f64 = sum.diag().slice(s![diag_ind..diag_ind + norbs_b]).sum();
    //                 // println!("Sliced trace {}",diag);
    //                 // println!("Full trace {}",sum.trace().unwrap());
    //
    //                 diag_ind += norbs_b;
    //                 // let tmp_1:f64 = w_mat_b.slice(s![index,..,..]).dot(&frag_grad_results[pair.frag_b_index].s.t()).trace().unwrap();
    //                 // let tmp_2:f64 = fragments[pair.frag_b_index].final_p_matrix.dot(&frag_grad_results[pair.frag_b_index].grad_s.slice(s![index,..,..]).t()).trace().unwrap();
    //                 // term_2[index] = (tmp_1 + tmp_2) * g0_ab.slice(s![..,a]).dot(&fragments[pair.frag_a_index].final_charges);
    //                 // term_2[index] = diag
    //                 //     * g0_ab
    //                 //         .slice(s![.., a])
    //                 //         .dot(&fragments[pair.frag_a_index].final_charges);
    //                 term_2[index] = diag
    //                     * g0_dimer_ab
    //                     .slice(s![.., a])
    //                     .dot(&fragments[pair.frag_a_index].final_charges);
    //             }
    //         }
    //         let gradient_frag_b: Array1<f64> = term_1 + term_2;
    //
    //         let mut dimer_gradient: Vec<f64> = Vec::new();
    //         dimer_gradient.append(&mut gradient_frag_a.to_vec());
    //         dimer_gradient.append(&mut gradient_frag_b.to_vec());
    //
    //         let dimer_gradient: Array1<f64> = Array::from(dimer_gradient);
    //         // dimer_gradients.push(dimer_gradient);
    //         dimer_gradient
    //
    //         // println!(
    //         //     "{:>68} {:>8.10} s",
    //         //     "elapsed time ES-DIM:",
    //         //     molecule_timer.elapsed().as_secs_f32()
    //         // );
    //         // drop(molecule_timer);
    //     }
    // }
    // let grad_total_frags: Array1<f64> =
    //     Array::from(grad_e0_monomers) + Array::from(grad_vrep_monomers);
    let mut grad_total_dimers: Array1<f64> = Array1::zeros(grad_total_frags.raw_dim());

    for (index, pair) in pair_results.iter().enumerate() {
        let index_a: usize = pair.frag_a_index;
        let index_b: usize = pair.frag_b_index;
        let atoms_a: usize = fragments[index_a].n_atoms;
        let atoms_b: usize = fragments[index_b].n_atoms;
        let index_frag_a: usize = indices_frags[index_a];
        let index_frag_b: usize = indices_frags[index_b];

        if pair.energy_pair.is_some() {
            let index_pair = pair_scc_hash[&index];
            grad_total_dimers
                .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
                .add_assign(
                    &(&dimer_gradients[index_pair].slice(s![0..3 * atoms_a])
                        - &(&frag_grad_results[index_a].grad_e0
                            + &frag_grad_results[index_a].grad_vrep)),
                );
            grad_total_dimers
                .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
                .add_assign(
                    &(&dimer_gradients[index_pair].slice(s![3 * atoms_a..])
                        - &(&frag_grad_results[index_b].grad_e0
                            + &frag_grad_results[index_b].grad_vrep)),
                );
        } else {
            let index_pair = pair_esdim_hash[&index];
            grad_total_dimers
                .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
                .add_assign(&grad_es_dim[index_pair].slice(s![0..3 * atoms_a]));
            grad_total_dimers
                .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
                .add_assign(&grad_es_dim[index_pair].slice(s![3 * atoms_a..]));
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
        //   .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
        //   .add_assign(
        //       &dimer_gradients[index_pair].slice(s![0..3 * atoms_a])
        //   );
        //grad_total_dimers
        //   .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
        //   .add_assign(
        //       &dimer_gradients[index_pair].slice(s![3 * atoms_a..]
        //       ));
    }

    for embed in embedding_gradients.iter() {
        //println!("Embed {}",embed.clone());
        grad_total_dimers.add_assign(embed);
    }
    let total_gradient: Array1<f64> = grad_total_frags + grad_total_dimers;

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
                // let mut pair_grad_s: Option<Array3<f64>> = None;
                let mut pair_s: Option<Array2<f64>> = None;
                let mut pair_density: Option<Array2<f64>> = None;
                let mut pair_n_orbs:Option<usize> = None;
                let mut pair_valorbs:Option<HashMap<u8, Vec<(i8, i8, i8)>>> = None;
                let mut pair_skt:Option<HashMap<(u8, u8), SlaterKosterTable>> = None;
                let mut pair_orbital_energies:Option<HashMap<u8, HashMap<(i8, i8), f64>>> = None;
                let mut pair_proximity:Option<Array2<bool>> = None;

                // let mut pair: Molecule = Molecule::new(
                //     atomic_numbers,
                //     positions,
                //     Some(config.mol.charge),
                //     Some(config.mol.multiplicity),
                //     Some(0.0),
                //     None,
                //     config.clone(),
                //     saved_calc,
                //     Some(connectivity_matrix),
                //     Some(graph_new),
                //     Some(graph_indexes),
                //     Some(subgraph),
                //     Some(distance_frag),
                //     Some(dir_frag),
                //     Some(prox_frag),
                //     None,
                // );
                //
                // if use_saved_calc == false {
                //     saved_calculators.push(pair.calculator.clone());
                //     saved_graphs.push(graph.clone());
                // }

                // let (g1, g1_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
                //     &pair.atomic_numbers,
                //     pair.n_atoms,
                //     pair.calculator.n_orbs,
                //     pair.distance_matrix.view(),
                //     pair.directions_matrix.view(),
                //     &pair.calculator.hubbard_u,
                //     &pair.calculator.valorbs,
                //     Some(0.0),
                // );
                // pair.set_g1_gradients(&g1, &g1_ao);

                // do scc routine for pair if mininmal distance is below threshold
                if (min_dist / vdw_radii_sum) < 2.0 {
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
                        None,
                    );

                    if use_saved_calc == false {
                        saved_calculators.push(pair.calculator.clone());
                        saved_graphs.push(graph.clone());
                    }
                    let (energy, orbs, orbe, s, f): (
                        f64,
                        Array2<f64>,
                        Array1<f64>,
                        Array2<f64>,
                        Vec<f64>,
                    ) = scc_routine::run_scc(&mut pair);
                    // println!("FMO pair energy {}",energy.clone());
                    energy_pair = Some(energy);
                    charges_pair = Some(pair.final_charges.clone());
                    pair_density = Some(pair.final_p_matrix.clone());

                    pair.calculator.set_active_orbitals(f);
                    // let full_occ: Vec<usize> = pair.calculator.full_occ.clone().unwrap();
                    // let full_virt: Vec<usize> = pair.calculator.full_virt.clone().unwrap();
                    // let n_occ_full: usize = full_occ.len();
                    //
                    // let orbe_occ: Array1<f64> =
                    //     full_occ.iter().map(|&full_occ| orbe[full_occ]).collect();
                    // let orbe_virt: Array1<f64> =
                    //     full_virt.iter().map(|&full_virt| orbe[full_virt]).collect();
                    // let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ_full));
                    //
                    // for (i, index) in full_occ.iter().enumerate() {
                    //     orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
                    // }
                    //
                    // let (
                    //     gradE0,
                    //     grad_v_rep,
                    //     grad_s,
                    //     grad_h0,
                    //     fdmdO,
                    //     flrdmdO,
                    //     g1,
                    //     g1_ao,
                    //     g1lr,
                    //     g1lr_ao,
                    // ): (
                    //     Array1<f64>,
                    //     Array1<f64>,
                    //     Array3<f64>,
                    //     Array3<f64>,
                    //     Array3<f64>,
                    //     Array3<f64>,
                    //     Array3<f64>,
                    //     Array3<f64>,
                    //     Array3<f64>,
                    //     Array3<f64>,
                    // ) = gradient_lc_gs(&pair, &orbe_occ, &orbe_virt, &orbs_occ, &s, Some(0.0));

                    let (gradE0, grad_v_rep, grad_exc, empty_z_vec): (
                        Array1<f64>,
                        Array1<f64>,
                        Array1<f64>,
                        Array3<f64>,
                    ) = get_gradients(&orbe, &orbs, &s, &mut pair, &None, &None, None, &None, None);

                    grad_e0_pair = Some(gradE0);
                    grad_vrep_pair = Some(grad_v_rep);

                    // let (grad_s, grad_h0): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
                    //     &pair.atomic_numbers,
                    //     pair.positions.view(),
                    //     pair.calculator.n_orbs,
                    //     &pair.calculator.valorbs,
                    //     pair.proximity_matrix.view(),
                    //     &pair.calculator.skt,
                    //     &pair.calculator.orbital_energies,
                    // );
                    pair_s = Some(s);
                    // pair_grad_s = pair.s_grad;
                    pair_n_orbs = Some(pair.calculator.n_orbs);
                    pair_valorbs = Some(pair.calculator.valorbs);
                    pair_skt = Some(pair.calculator.skt);
                    pair_orbital_energies = Some(pair.calculator.orbital_energies);
                    pair_proximity = Some(pair.proximity_matrix);
                }

                // let pair_res: pair_grad_result = pair_grad_result::new(
                //     charges_pair,
                //     energy_pair,
                //     ind1,
                //     ind2,
                //     molecule_a.n_atoms,
                //     molecule_b.n_atoms,
                //     grad_e0_pair,
                //     grad_vrep_pair,
                //     pair.g0,
                //     g1,
                //     pair_s,
                //     pair_grad_s,
                //     pair_density,
                // );

                // removed g0 and g1
                let pair_res: pair_grad_result = pair_grad_result::new(
                   charges_pair,
                   energy_pair,
                   ind1,
                   ind2,
                   molecule_a.n_atoms,
                   molecule_b.n_atoms,
                   grad_e0_pair,
                   grad_vrep_pair,
                   pair_s,
                   // pair_grad_s,
                   pair_density,
                   pair_n_orbs,
                   pair_valorbs,
                   pair_skt,
                   pair_orbital_energies,
                   pair_proximity,
                    None,
                    None
                );

                vec_pair_result.push(pair_res);
            }
        }
    }
    return (vec_pair_result);
}

pub fn fmo_calculate_pairwise_gradients_par(
    fragments: &Vec<Molecule>,
    frag_grad_results: &Vec<frag_grad_result>,
    config: GeneralConfig,
    dist_mat: &Array2<f64>,
    direct_mat: &Array3<f64>,
    prox_mat: &Array2<bool>,
    indices_frags: &Vec<usize>,
    // gamma_total: &Array2<f64>,
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
        None,
    );
    let first_calc: DFTBCalculator = first_pair.calculator.clone();

    let mut result: Vec<Vec<pair_grad_result>> = fragments
        .par_iter()
        .enumerate()
        .map(|(ind1, molecule_a)| {
            let mut vec_pair_result: Vec<pair_grad_result> = Vec::new();
            let mut saved_calculators: Vec<DFTBCalculator> = Vec::new();
            let mut saved_graphs: Vec<Graph<u8, f64, Undirected>> = Vec::new();

            saved_graphs.push(first_graph.clone());
            saved_calculators.push(first_calc.clone());

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
                            if is_isomorphic_matching(
                                &graph,
                                saved_graph,
                                |a, b| a == b,
                                |a, b| true,
                            ) == true
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
                        .filter_map(
                            |(index, &item)| if item == min_dist { Some(index) } else { None },
                        )
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
                    // let mut pair_grad_s: Option<Array3<f64>> = None;
                    let mut pair_s: Option<Array2<f64>> = None;
                    let mut pair_density: Option<Array2<f64>> = None;
                    let mut pair_n_orbs:Option<usize> = None;
                    let mut pair_valorbs:Option<HashMap<u8, Vec<(i8, i8, i8)>>> = None;
                    let mut pair_skt:Option<HashMap<(u8, u8), SlaterKosterTable>> = None;
                    let mut pair_orbital_energies:Option<HashMap<u8, HashMap<(i8, i8), f64>>> = None;
                    let mut pair_proximity:Option<Array2<bool>> = None;

                    // let mut pair: Molecule = Molecule::new(
                    //     atomic_numbers,
                    //     positions,
                    //     Some(config.mol.charge),
                    //     Some(config.mol.multiplicity),
                    //     Some(0.0),
                    //     None,
                    //     config.clone(),
                    //     saved_calc,
                    //     Some(connectivity_matrix),
                    //     Some(graph_new),
                    //     Some(graph_indexes),
                    //     Some(subgraph),
                    //     Some(distance_frag),
                    //     Some(dir_frag),
                    //     Some(prox_frag),
                    //     None,
                    // );
                    // if use_saved_calc == false {
                    //     saved_calculators.push(pair.calculator.clone());
                    //     saved_graphs.push(graph.clone());
                    // }

                    // let (g1, g1_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
                    //     &pair.atomic_numbers,
                    //     pair.n_atoms,
                    //     pair.calculator.n_orbs,
                    //     pair.distance_matrix.view(),
                    //     pair.directions_matrix.view(),
                    //     &pair.calculator.hubbard_u,
                    //     &pair.calculator.valorbs,
                    //     Some(0.0),
                    // );
                    // pair.set_g1_gradients(&g1, &g1_ao);

                    // do scc routine for pair if mininmal distance is below threshold
                    if (min_dist / vdw_radii_sum) < 2.0 {
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
                            None,
                        );
                        if use_saved_calc == false {
                            saved_calculators.push(pair.calculator.clone());
                            saved_graphs.push(graph.clone());
                        }
                        let (energy, orbs, orbe, s, f): (
                            f64,
                            Array2<f64>,
                            Array1<f64>,
                            Array2<f64>,
                            Vec<f64>,
                        ) = scc_routine::run_scc(&mut pair);
                        // println!("FMO pair energy {}",energy.clone());
                        energy_pair = Some(energy);
                        charges_pair = Some(pair.final_charges.clone());
                        pair_density = Some(pair.final_p_matrix.clone());

                        pair.calculator.set_active_orbitals(f);

                        let (gradE0, grad_v_rep, grad_exc, empty_z_vec): (
                            Array1<f64>,
                            Array1<f64>,
                            Array1<f64>,
                            Array3<f64>,
                        ) = get_gradients(
                            &orbe, &orbs, &s, &mut pair, &None, &None, None, &None, None,
                        );

                        grad_e0_pair = Some(gradE0);
                        grad_vrep_pair = Some(grad_v_rep);

                        // let (grad_s, grad_h0): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
                        //     &pair.atomic_numbers,
                        //     pair.positions.view(),
                        //     pair.calculator.n_orbs,
                        //     &pair.calculator.valorbs,
                        //     pair.proximity_matrix.view(),
                        //     &pair.calculator.skt,
                        //     &pair.calculator.orbital_energies,
                        // );
                        pair_s = Some(s);
                        // pair_grad_s = pair.s_grad;
                        pair_n_orbs = Some(pair.calculator.n_orbs);
                        pair_valorbs = Some(pair.calculator.valorbs);
                        pair_skt = Some(pair.calculator.skt);
                        pair_orbital_energies = Some(pair.calculator.orbital_energies);
                        pair_proximity = Some(pair.proximity_matrix);
                    }

                    // let pair_res: pair_grad_result = pair_grad_result::new(
                    //     charges_pair,
                    //     energy_pair,
                    //     ind1,
                    //     ind2,
                    //     molecule_a.n_atoms,
                    //     molecule_b.n_atoms,
                    //     grad_e0_pair,
                    //     grad_vrep_pair,
                    //     pair.g0,
                    //     g1,
                    //     pair_s,
                    //     pair_grad_s,
                    //     pair_density,
                    // );

                    let pair_res: pair_grad_result = pair_grad_result::new(
                        charges_pair,
                        energy_pair,
                        ind1,
                        ind2,
                        molecule_a.n_atoms,
                        molecule_b.n_atoms,
                        grad_e0_pair,
                        grad_vrep_pair,
                        pair_s,
                        // pair_grad_s,
                        pair_density,
                        pair_n_orbs,
                        pair_valorbs,
                        pair_skt,
                        pair_orbital_energies,
                        pair_proximity,
                        None,
                        None
                    );

                    vec_pair_result.push(pair_res);
                }
            }
            vec_pair_result
        })
        .collect();

    let mut pair_result: Vec<pair_grad_result> = Vec::new();
    for pair in result.iter_mut() {
        pair_result.append(pair);
    }
    return (pair_result);
}

pub fn fmo_calculate_pairs_embedding_esdim(
    fragments: &Vec<Molecule>,
    frag_grad_results: &Vec<frag_grad_result>,
    config: GeneralConfig,
    dist_mat: &Array2<f64>,
    direct_mat: &Array3<f64>,
    prox_mat: &Array2<bool>,
    indices_frags: &Vec<usize>,
    full_hubbard: &HashMap<u8, f64>,
) -> (Array1<f64>) {
    // calculate gradient for monomers
    let mut grad_e0_monomers: Vec<f64> = Vec::new();
    let mut grad_vrep_monomers: Vec<f64> = Vec::new();

    for frag in frag_grad_results.iter() {
        grad_e0_monomers.append(&mut frag.grad_e0.clone().to_vec());
        grad_vrep_monomers.append(&mut frag.grad_vrep.clone().to_vec());
    }
    let grad_e0_monomers:Array1<f64> = Array::from(grad_e0_monomers);
    let grad_vrep_monomers:Array1<f64> = Array::from(grad_vrep_monomers);
    let grad_total_frags: Array1<f64> = grad_e0_monomers + grad_vrep_monomers;

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
        None,
    );
    let first_calc: DFTBCalculator = first_pair.calculator.clone();
    drop(first_pair);

    let mut result: Vec<Vec<pair_gradients_result>> = fragments
        .par_iter()
        .enumerate()
        .map(|(ind1, molecule_a)| {
            let mut vec_pair_result: Vec<pair_gradients_result> = Vec::new();
            let mut saved_calculators: Vec<DFTBCalculator> = Vec::new();
            let mut saved_graphs: Vec<Graph<u8, f64, Undirected>> = Vec::new();

            saved_graphs.push(first_graph.clone());
            saved_calculators.push(first_calc.clone());

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
                            if is_isomorphic_matching(
                                &graph,
                                saved_graph,
                                |a, b| a == b,
                                |a, b| true,
                            ) == true
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
                        .filter_map(
                            |(index, &item)| if item == min_dist { Some(index) } else { None },
                        )
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
                    // let mut pair_grad_s: Option<Array3<f64>> = None;
                    let mut pair_s: Option<Array2<f64>> = None;
                    let mut pair_density: Option<Array2<f64>> = None;
                    let mut pair_n_orbs:Option<usize> = None;
                    let mut pair_valorbs:Option<HashMap<u8, Vec<(i8, i8, i8)>>> = None;
                    let mut pair_skt:Option<HashMap<(u8, u8), SlaterKosterTable>> = None;
                    let mut pair_orbital_energies:Option<HashMap<u8, HashMap<(i8, i8), f64>>> = None;
                    let mut pair_proximity:Option<Array2<bool>> = None;
                    let mut pair_gradient:Option<Array1<f64>> = None;
                    let mut pair_embedding_gradient:Option<Array1<f64>> = None;

                    // do scc routine for pair if mininmal distance is below threshold
                    if (min_dist / vdw_radii_sum) < 2.0 {
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
                            None,
                        );
                        if use_saved_calc == false {
                            saved_calculators.push(pair.calculator.clone());
                            saved_graphs.push(graph.clone());
                        }
                        let (energy, orbs, orbe, s, f): (
                            f64,
                            Array2<f64>,
                            Array1<f64>,
                            Array2<f64>,
                            Vec<f64>,
                        ) = scc_routine::run_scc(&mut pair);
                        // println!("FMO pair energy {}",energy.clone());
                        energy_pair = Some(energy);
                        charges_pair = Some(pair.final_charges.clone());
                        pair_density = Some(pair.final_p_matrix.clone());

                        pair.calculator.set_active_orbitals(f);

                        let full_occ: Vec<usize> = pair.calculator.full_occ.clone().unwrap();
                        let full_virt: Vec<usize> = pair.calculator.full_virt.clone().unwrap();
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
                        ) = gradient_lc_gs(&mut pair, &orbe_occ, &orbe_virt, &orbs_occ, &s, Some(0.0));
                        drop(grad_h0);
                        drop(fdmdO);
                        drop(flrdmdO);
                        drop(g1);
                        drop(g1_ao);
                        drop(g1lr);
                        drop(g1lr_ao);

                        // let (gradE0, grad_v_rep, grad_exc, empty_z_vec): (
                        //     Array1<f64>,
                        //     Array1<f64>,
                        //     Array1<f64>,
                        //     Array3<f64>,
                        // ) = get_gradients(
                        //     &orbe, &orbs, &s, &mut pair, &None, &None, None, &None, None,
                        // );

                        // grad_e0_pair = Some(gradE0);
                        // grad_vrep_pair = Some(grad_v_rep);

                        // let (grad_s, grad_h0): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
                        //     &pair.atomic_numbers,
                        //     pair.positions.view(),
                        //     pair.calculator.n_orbs,
                        //     &pair.calculator.valorbs,
                        //     pair.proximity_matrix.view(),
                        //     &pair.calculator.skt,
                        //     &pair.calculator.orbital_energies,
                        // );
                        // pair_s = Some(s);
                        // pair_grad_s = pair.s_grad;
                        // pair_n_orbs = Some(pair.calculator.n_orbs);
                        // pair_valorbs = Some(pair.calculator.valorbs);
                        // pair_skt = Some(pair.calculator.skt);
                        // pair_orbital_energies = Some(pair.calculator.orbital_energies);
                        // pair_proximity = Some(pair.proximity_matrix);

                        let pair_atoms: usize = pair.n_atoms;
                        let pair_charges:Array1<f64> = pair.final_charges.clone();
                        let dimer_pmat:Array2<f64> = pair.final_p_matrix.clone();
                        let pair_smat: Array2<f64> = s;
                        let frag_a_atoms:usize = fragments[ind1].n_atoms;
                        let frag_b_atoms:usize = fragments[ind2].n_atoms;

                        let ddq_vec: Vec<f64> = (0..pair_atoms)
                            .into_iter()
                            .map(|a| {
                                let mut ddq: f64 = 0.0;
                                if a < frag_a_atoms {
                                    ddq = pair_charges[a] - fragments[ind1].final_charges[a];
                                } else {
                                    ddq = pair_charges[a]
                                        - fragments[ind2].final_charges[a - frag_a_atoms];
                                }
                                ddq
                            })
                            .collect();

                        let index_pair_a: usize = indices_frags[ind1];
                        let index_pair_b: usize = indices_frags[ind2];
                        let ddq_arr: Array1<f64> = Array::from(ddq_vec);
                        let shape_orbs_dimer: usize = pair.calculator.n_orbs;
                        let dimer_atomic_numbers:Vec<u8> = pair.atomic_numbers.clone();
                        drop(pair);
                        let shape_orbs_a: usize = fragments[ind1].calculator.n_orbs;
                        let shape_orbs_b: usize = fragments[ind2].calculator.n_orbs;

                        let mut w_dimer: Array3<f64> =
                            Array3::zeros((3 * pair_atoms, shape_orbs_dimer, shape_orbs_dimer));
                        for i in (0..3 * pair_atoms).into_iter() {
                            w_dimer
                                .slice_mut(s![i, .., ..])
                                .assign(&dimer_pmat.dot(&grad_s.slice(s![i, .., ..]).dot(&dimer_pmat)));
                        }
                        w_dimer = -0.5 * w_dimer;

                        let mut w_mat_a: Array3<f64> = Array3::zeros((
                            3 * fragments[ind1].n_atoms,
                            shape_orbs_a,
                            shape_orbs_a,
                        ));

                        for i in (0..3 * fragments[ind1].n_atoms).into_iter() {
                            w_mat_a.slice_mut(s![i, .., ..]).assign(
                                &fragments[ind1].final_p_matrix.dot(
                                    &frag_grad_results[ind1].grad_s
                                        .slice(s![i, .., ..])
                                        .dot(&fragments[ind1].final_p_matrix),
                                ),
                            );
                        }
                        w_mat_a = -0.5 * w_mat_a;

                        let mut w_mat_b: Array3<f64> = Array3::zeros((
                            3 * fragments[ind2].n_atoms,
                            shape_orbs_b,
                            shape_orbs_b,
                        ));
                        for i in (0..3 * fragments[ind2].n_atoms).into_iter() {
                            w_mat_b.slice_mut(s![i, .., ..]).assign(
                                &fragments[ind2].final_p_matrix.dot(
                                    &frag_grad_results[ind2].grad_s
                                        .slice(s![i, .., ..])
                                        .dot(&fragments[ind2].final_p_matrix),
                                ),
                            );
                        }
                        w_mat_b = -0.5 * w_mat_b;

                        // Build delta p_mu,nu^I,J
                        let mut dp_direct_sum_monomer: Array2<f64> = Array2::zeros(dimer_pmat.raw_dim());
                        let p_dim_monomer: usize = fragments[ind1].final_p_matrix.dim().0;
                        dp_direct_sum_monomer
                            .slice_mut(s![0..p_dim_monomer, 0..p_dim_monomer])
                            .assign(&fragments[ind1].final_p_matrix);
                        dp_direct_sum_monomer
                            .slice_mut(s![p_dim_monomer.., p_dim_monomer..])
                            .assign(&fragments[ind2].final_p_matrix);
                        let dp_dimer: Array2<f64> = &dimer_pmat - &dp_direct_sum_monomer;

                        // Build delta W_mu,nu^I,J
                        let mut dw_dimer: Array3<f64> = Array3::zeros(w_dimer.raw_dim());
                        let w_dimer_dim: usize = w_dimer.dim().1;
                        let mut dw_dimer_vec: Vec<Array1<f64>> = (0..3 * pair_atoms)
                            .into_iter()
                            .map(|a| {
                                let mut w_a_dimer: Array2<f64> = Array::zeros((w_dimer_dim, w_dimer_dim));
                                if a < 3 * frag_a_atoms {
                                    w_a_dimer
                                        .slice_mut(s![0..p_dim_monomer, 0..p_dim_monomer])
                                        .assign(&w_mat_a.slice(s![a, .., ..]));
                                } else {
                                    w_a_dimer
                                        .slice_mut(s![p_dim_monomer.., p_dim_monomer..])
                                        .assign(&w_mat_b.slice(s![a - 3 * frag_a_atoms, .., ..]));
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
                            .par_iter()
                            .enumerate()
                            .filter_map(|(ind_k, mol_k)| {
                                if ind_k != ind1 && ind_k != ind2 {
                                    let index_frag_iter: usize = indices_frags[ind_k];

                                    let trimer_distances_a: ArrayView2<f64> = dist_mat.slice(s![
                                        index_pair_a..index_pair_a + frag_a_atoms,
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms
                                    ]);
                                    let trimer_distances_b: ArrayView2<f64> = dist_mat.slice(s![
                                        index_pair_b..index_pair_b + frag_b_atoms,
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms
                                    ]);
                                    let trimer_distances:Array2<f64> = stack(Axis(0),&[trimer_distances_a,trimer_distances_b]).unwrap();

                                    let trimer_directions_a: ArrayView3<f64> = direct_mat.slice(s![
                                        index_pair_a..index_pair_a + frag_a_atoms,
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms, ..
                                    ]);
                                    let trimer_directions_b: ArrayView3<f64> = direct_mat.slice(s![
                                        index_pair_b..index_pair_b + frag_b_atoms,
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms, ..
                                    ]);
                                    let trimer_directions:Array3<f64> = stack(Axis(0),&[trimer_directions_a,trimer_directions_b]).unwrap();

                                    let g1_trimer_ak: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
                                        &dimer_atomic_numbers,
                                        &mol_k.atomic_numbers,
                                        pair_atoms,
                                        mol_k.n_atoms,
                                        trimer_distances.view(),
                                        trimer_directions.view(),
                                        full_hubbard,
                                        Some(0.0),
                                    );

                                    let g0_trimer_ak: Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
                                        &dimer_atomic_numbers,
                                        &mol_k.atomic_numbers,
                                        pair_atoms,
                                        mol_k.n_atoms,
                                        trimer_distances.view(),
                                        full_hubbard,
                                        Some(0.0));

                                    let trimer_distances_a: ArrayView2<f64> = dist_mat.slice(s![
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms,
                                        index_pair_a..index_pair_a + frag_a_atoms,
                                    ]);
                                    let trimer_distances_b: ArrayView2<f64> = dist_mat.slice(s![
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms,
                                        index_pair_b..index_pair_b + frag_b_atoms,
                                    ]);
                                    let trimer_distances:Array2<f64> = stack(Axis(1),&[trimer_distances_a,trimer_distances_b]).unwrap();

                                    let trimer_directions_a: ArrayView3<f64> = direct_mat.slice(s![
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms,
                                        index_pair_a..index_pair_a + frag_a_atoms, ..
                                    ]);
                                    let trimer_directions_b: ArrayView3<f64> = direct_mat.slice(s![
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms,
                                        index_pair_b..index_pair_b + frag_b_atoms,..
                                    ]);
                                    let trimer_directions:Array3<f64> = stack(Axis(1),&[trimer_directions_a,trimer_directions_b]).unwrap();

                                    let g1_trimer_ka: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
                                        &mol_k.atomic_numbers,
                                        &dimer_atomic_numbers,
                                        mol_k.n_atoms,
                                        pair_atoms,
                                        trimer_distances.view(),
                                        trimer_directions.view(),
                                        full_hubbard,
                                        Some(0.0),
                                    );

                                    // calculate grads for molecule k
                                    let mut term_1: Array1<f64> = Array1::zeros(3 * pair_atoms);
                                    let mut term_2: Array1<f64> = Array1::zeros(3 * pair_atoms);
                                    for dir in (0..3).into_iter() {
                                        let dir_xyz: usize = dir as usize;
                                        let mut diag_ind: usize = 0;
                                        for a in (0..pair_atoms).into_iter() {
                                            let index: usize = 3 * a + dir_xyz;

                                            term_1[index] = ddq_arr[a]
                                                * g1_trimer_ak.slice(s![index, a, ..]).dot(&mol_k.final_charges);

                                            let mut atom_type: u8 = 0;
                                            let mut norbs_a: usize = 0;

                                            if a < frag_a_atoms {
                                                atom_type = fragments[ind1].atomic_numbers[a];
                                                norbs_a = fragments[ind1].calculator.valorbs
                                                    [&atom_type]
                                                    .len();
                                            } else {
                                                atom_type = fragments[ind2].atomic_numbers
                                                    [a - frag_a_atoms];
                                                norbs_a = fragments[ind2].calculator.valorbs
                                                    [&atom_type]
                                                    .len();
                                            }

                                            let tmp_1: Array2<f64> =
                                                dw_dimer.slice(s![index, .., ..]).dot(&pair_smat.t());
                                            let tmp_2: Array2<f64> =
                                                dp_dimer.dot(&grad_s.slice(s![index, .., ..]).t());
                                            let sum: Array2<f64> = tmp_1 + tmp_2;

                                            let diag: f64 =
                                                sum.diag().slice(s![diag_ind..diag_ind + norbs_a]).sum();
                                            diag_ind += norbs_a;

                                            term_2[index] = diag
                                                * g0_trimer_ak.slice(s![a, ..]).dot(&fragments[ind_k].final_charges);
                                        }
                                    }
                                    let embedding_part_1: Array1<f64> = term_1 + term_2;

                                    // let shape_orbs_k: usize = frag_grad_results[ind_k].grad_s.dim().1;
                                    let shape_orbs_k: usize = mol_k.calculator.n_orbs;
                                    let mut w_mat_k: Array3<f64> = Array3::zeros((
                                        3 * fragments[ind_k].n_atoms,
                                        shape_orbs_k,
                                        shape_orbs_k,
                                    ));

                                    for i in (0..3 * fragments[ind_k].n_atoms).into_iter() {
                                        w_mat_k.slice_mut(s![i, .., ..]).assign(
                                            &fragments[ind_k].final_p_matrix.dot(&frag_grad_results[ind_k].grad_s
                                                .slice(s![i, .., ..])
                                                .dot(&fragments[ind_k].final_p_matrix),
                                            ),
                                        );
                                    }
                                    w_mat_k = -0.5 * w_mat_k;

                                    let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[ind_k].n_atoms);
                                    let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[ind_k].n_atoms);
                                    for dir in (0..3).into_iter() {
                                        let dir_xyz: usize = dir as usize;
                                        let mut diag_ind: usize = 0;
                                        for a in (0..fragments[ind_k].n_atoms).into_iter() {
                                            let index: usize = 3 * a + dir_xyz;

                                            term_1[index] = fragments[ind_k].final_charges[a]
                                                * g1_trimer_ka.slice(s![index, a, ..]).dot(&ddq_arr);

                                            let atom_type: u8 = fragments[ind_k].atomic_numbers[a];
                                            let norbs_k: usize =
                                                fragments[ind_k].calculator.valorbs[&atom_type].len();

                                            let tmp_1: Array2<f64> = w_mat_k
                                                .slice(s![index, .., ..])
                                                .dot(&frag_grad_results[ind_k].s.t());

                                            let tmp_2: Array2<f64> = fragments[ind_k].final_p_matrix.dot(
                                                &frag_grad_results[ind_k].grad_s.slice(s![index, .., ..]).t(),
                                            );

                                            let sum: Array2<f64> = tmp_1 + tmp_2;

                                            let diag: f64 =
                                                sum.diag().slice(s![diag_ind..diag_ind + norbs_k]).sum();
                                            diag_ind += norbs_k;

                                            term_2[index] = diag * g0_trimer_ak.slice(s![.., a]).dot(&ddq_arr);
                                        }
                                    }
                                    let embedding_part_2: Array1<f64> = term_1 + term_2;

                                    let mut temp_grad: Array1<f64> = Array1::zeros(grad_total_frags.len());

                                    let index_a: usize = ind1;
                                    let index_b: usize = ind2;
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

                        let mut embedding_gradient: Array1<f64> = Array1::zeros(grad_total_frags.len());
                        for grad in embedding_pot.iter() {
                            embedding_gradient.add_assign(grad);
                        }
                        pair_embedding_gradient = Some(embedding_gradient);
                        pair_gradient = Some(gradE0 + grad_v_rep);
                    }
                    else{
                        drop(distance_frag);
                        drop(prox_frag);
                        drop(dir_frag);

                        let dimer_natoms: usize =
                            fragments[ind1].n_atoms + fragments[ind2].n_atoms;
                        let dimer_gradient: Array1<f64> = Array::zeros(dimer_natoms * 3);
                        let shape_orbs_a: usize = fragments[ind1].calculator.n_orbs;
                        let shape_orbs_b: usize = fragments[ind2].calculator.n_orbs;
                        let index_pair_a: usize = indices_frags[ind1];
                        let index_pair_b: usize = indices_frags[ind2];

                        // calculate g1 matrix for each dimer
                        let dimer_distances: ArrayView2<f64> = dist_mat.slice(s![
                            index_pair_a..index_pair_a + fragments[ind1].n_atoms,
                            index_pair_b..index_pair_b + fragments[ind2].n_atoms
                        ]);

                        let dimer_directions: ArrayView3<f64> = direct_mat.slice(s![
                            index_pair_a..index_pair_a + fragments[ind1].n_atoms,
                            index_pair_b..index_pair_b + fragments[ind2].n_atoms,
                            ..
                        ]);
                        let g1_dimer_ab: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
                            &fragments[ind1].atomic_numbers,
                            &fragments[ind2].atomic_numbers,
                            fragments[ind1].n_atoms,
                            fragments[ind2].n_atoms,
                            dimer_distances,
                            dimer_directions,
                            full_hubbard,
                            Some(0.0),
                        );

                        let g0_dimer_ab:Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
                            &fragments[ind1].atomic_numbers,
                            &fragments[ind2].atomic_numbers,
                            fragments[ind1].n_atoms,
                            fragments[ind2].n_atoms,
                            dimer_distances,full_hubbard,
                            Some(0.0));

                        let dimer_distances: ArrayView2<f64> = dist_mat.slice(s![
                            index_pair_b..index_pair_b + fragments[ind2].n_atoms,
                            index_pair_a..index_pair_a + fragments[ind1].n_atoms
                        ]);

                        let dimer_directions: ArrayView3<f64> = direct_mat.slice(s![
                            index_pair_b..index_pair_b + fragments[ind2].n_atoms,
                            index_pair_a..index_pair_a + fragments[ind1].n_atoms,
                            ..
                        ]);

                        let g1_dimer_ba: Array3<f64> = get_gamma_gradient_matrix_atom_wise_outer_diagonal(
                            &fragments[ind2].atomic_numbers,
                            &fragments[ind1].atomic_numbers,
                            fragments[ind2].n_atoms,
                            fragments[ind1].n_atoms,
                            dimer_distances,
                            dimer_directions,
                            full_hubbard,
                            Some(0.0),
                        );

                        let mut w_mat_a: Array3<f64> = Array3::zeros((
                            3 * fragments[ind1].n_atoms,
                            shape_orbs_a,
                            shape_orbs_a,
                        ));

                        for i in (0..3 * fragments[ind1].n_atoms).into_iter() {
                            w_mat_a.slice_mut(s![i, .., ..]).assign(
                                &fragments[ind1].final_p_matrix.dot(
                                    &frag_grad_results[ind1].grad_s
                                        .slice(s![i, .., ..])
                                        .dot(&fragments[ind1].final_p_matrix),
                                ),
                            );
                        }
                        w_mat_a = -0.5 * w_mat_a;

                        let mut w_mat_b: Array3<f64> = Array3::zeros((
                            3 * fragments[ind2].n_atoms,
                            shape_orbs_b,
                            shape_orbs_b,
                        ));
                        for i in (0..3 * fragments[ind2].n_atoms).into_iter() {
                            w_mat_b.slice_mut(s![i, .., ..]).assign(
                                &fragments[ind2].final_p_matrix.dot(
                                    &frag_grad_results[ind2].grad_s
                                        .slice(s![i, .., ..])
                                        .dot(&fragments[ind2].final_p_matrix),
                                ),
                            );
                        }
                        w_mat_b = -0.5 * w_mat_b;

                        let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[ind1].n_atoms);
                        let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[ind1].n_atoms);
                        for dir in (0..3).into_iter() {
                            let dir_xyz: usize = dir as usize;
                            let mut diag_ind: usize = 0;
                            for a in (0..fragments[ind1].n_atoms).into_iter() {
                                let index: usize = 3 * a + dir_xyz;
                                term_1[index] = fragments[ind1].final_charges[a]
                                    * g1_dimer_ab
                                    .slice(s![index, a, ..])
                                    .dot(&fragments[ind2].final_charges);

                                let atom_type: u8 = fragments[ind1].atomic_numbers[a];
                                let norbs_a: usize =
                                    fragments[ind1].calculator.valorbs[&atom_type].len();

                                let tmp_1: Array2<f64> = w_mat_a
                                    .slice(s![index, .., ..])
                                    .dot(&frag_grad_results[ind1].s.t());
                                let tmp_2: Array2<f64> = fragments[ind1].final_p_matrix.dot(
                                    &frag_grad_results[ind1].grad_s
                                        .slice(s![index, .., ..])
                                        .t(),
                                );
                                let sum: Array2<f64> = tmp_1 + tmp_2;

                                let diag: f64 = sum.diag().slice(s![diag_ind..diag_ind + norbs_a]).sum();
                                diag_ind += norbs_a;
                                term_2[index] = diag
                                    * g0_dimer_ab
                                    .slice(s![a, ..])
                                    .dot(&fragments[ind2].final_charges);
                            }
                        }
                        let gradient_frag_a: Array1<f64> = term_1 + term_2;

                        let mut term_1: Array1<f64> = Array1::zeros(3 * fragments[ind2].n_atoms);
                        let mut term_2: Array1<f64> = Array1::zeros(3 * fragments[ind2].n_atoms);
                        for dir in (0..3).into_iter() {
                            let dir_xyz: usize = dir as usize;
                            let mut diag_ind: usize = 0;
                            for a in (0..fragments[ind2].n_atoms).into_iter() {
                                let index: usize = 3 * a + dir_xyz;
                                // term_1[index] = fragments[ind2].final_charges[a]
                                //     * g1_ab_2
                                //         .slice(s![index, a, ..])
                                //         .dot(&fragments[ind1].final_charges);
                                term_1[index] = fragments[ind2].final_charges[a]
                                    * g1_dimer_ba
                                    .slice(s![index, a, ..])
                                    .dot(&fragments[ind1].final_charges);

                                let atom_type: u8 = fragments[ind2].atomic_numbers[a];
                                let norbs_b: usize =
                                    fragments[ind2].calculator.valorbs[&atom_type].len();

                                let tmp_1: Array2<f64> = w_mat_b
                                    .slice(s![index, .., ..])
                                    .dot(&frag_grad_results[ind2].s.t());
                                let tmp_2: Array2<f64> = fragments[ind2].final_p_matrix.dot(
                                    &frag_grad_results[ind2].grad_s
                                        .slice(s![index, .., ..])
                                        .t(),
                                );
                                let sum: Array2<f64> = tmp_1 + tmp_2;

                                let diag: f64 = sum.diag().slice(s![diag_ind..diag_ind + norbs_b]).sum();

                                diag_ind += norbs_b;
                                term_2[index] = diag
                                    * g0_dimer_ab
                                    .slice(s![.., a])
                                    .dot(&fragments[ind1].final_charges);
                            }
                        }
                        let gradient_frag_b: Array1<f64> = term_1 + term_2;
                        let mut dimer_gradient: Vec<f64> = Vec::new();
                        dimer_gradient.append(&mut gradient_frag_a.to_vec());
                        dimer_gradient.append(&mut gradient_frag_b.to_vec());

                        let dimer_gradient: Array1<f64> = Array::from(dimer_gradient);
                        pair_gradient = Some(dimer_gradient);
                    }

                    let pair_res: pair_gradients_result = pair_gradients_result::new(
                        energy_pair,
                        ind1,
                        ind2,
                        molecule_a.n_atoms,
                        molecule_b.n_atoms,
                        pair_embedding_gradient,
                        pair_gradient,
                    );

                    vec_pair_result.push(pair_res);
                }
            }
            vec_pair_result
        })
        .collect();

    let mut pair_result: Vec<pair_gradients_result> = Vec::new();
    for pair in result.iter_mut() {
        pair_result.append(pair);
    }
    drop(result);
    let mut grad_total_dimers: Array1<f64> = Array1::zeros(grad_total_frags.raw_dim());

    for (index, pair) in pair_result.iter().enumerate() {
        let index_a: usize = pair.frag_a_index;
        let index_b: usize = pair.frag_b_index;
        let atoms_a: usize = fragments[index_a].n_atoms;
        let atoms_b: usize = fragments[index_b].n_atoms;
        let index_frag_a: usize = indices_frags[index_a];
        let index_frag_b: usize = indices_frags[index_b];
        let dimer_gradient:Array1<f64> = pair.dimer_gradient.clone().unwrap();

        if pair.energy_pair.is_some() {
            grad_total_dimers
                .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
                .add_assign(
                    &(&dimer_gradient.slice(s![0..3 * atoms_a])
                        - &(&frag_grad_results[index_a].grad_e0
                        + &frag_grad_results[index_a].grad_vrep)),
                );
            grad_total_dimers
                .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
                .add_assign(
                    &(&dimer_gradient.slice(s![3 * atoms_a..])
                        - &(&frag_grad_results[index_b].grad_e0
                        + &frag_grad_results[index_b].grad_vrep)),
                );
            let embedding_gradient:Array1<f64> = pair.embedding_gradient.clone().unwrap();
            grad_total_dimers.add_assign(&embedding_gradient);
        } else {
            grad_total_dimers
                .slice_mut(s![3 * index_frag_a..3 * index_frag_a + 3 * atoms_a])
                .add_assign(&dimer_gradient.slice(s![0..3 * atoms_a]));
            grad_total_dimers
                .slice_mut(s![3 * index_frag_b..3 * index_frag_b + 3 * atoms_b])
                .add_assign(&dimer_gradient.slice(s![3 * atoms_a..]));
        }
    }
    let total_gradient: Array1<f64> = grad_total_frags + grad_total_dimers;
    return total_gradient;
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
        ) = gradient_lc_gs(frag, &orbe_occ, &orbe_virt, &orbs_occ, &s, Some(0.0));

        let frag_result: frag_grad_result =
            frag_grad_result::new(energy, gradE0, grad_v_rep, grad_s,s);

        results.push(frag_result);
    }
    return results;
}

pub fn fmo_calculate_fragment_gradients_par(
    fragments: &mut Vec<Molecule>,
) -> (Vec<frag_grad_result>) {
    let results: Vec<frag_grad_result> = fragments
        .par_iter_mut()
        .map(|frag| {
            let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
                scc_routine::run_scc(frag);

            frag.calculator.set_active_orbitals(f);
            let full_occ: Vec<usize> = frag.calculator.full_occ.clone().unwrap();
            let full_virt: Vec<usize> = frag.calculator.full_virt.clone().unwrap();
            let n_occ_full: usize = full_occ.len();

            let orbe_occ: Array1<f64> = full_occ.iter().map(|&full_occ| orbe[full_occ]).collect();
            let orbe_virt: Array1<f64> =
                full_virt.iter().map(|&full_virt| orbe[full_virt]).collect();
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
            ) = gradient_lc_gs(frag, &orbe_occ, &orbe_virt, &orbs_occ, &s, Some(0.0));

            let frag_result: frag_grad_result =
                frag_grad_result::new(energy, gradE0, grad_v_rep, grad_s,s);
            frag_result
        })
        .collect();

    return results;
}

pub fn reorder_molecule_gradients(
    fragments: &Vec<Molecule>,
    config: GeneralConfig,
    shape_positions: Ix2,
) -> (
    Vec<usize>,
    // Array2<f64>,
    Array2<bool>,
    Array2<f64>,
    Array3<f64>,
    HashMap<u8, f64>,
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
    let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
        distance_matrix(positions.view(), None);
    let (atomtypes, unique_numbers): (HashMap<u8, String>, Vec<u8>) =
        get_atomtypes(atomic_numbers.clone());

    let calculator: DFTBCalculator = DFTBCalculator::new(
        &atomic_numbers,
        &atomtypes,
        None,
        &dist_matrix,
        Some(0.0),
    );

    // let new_mol: Molecule = Molecule::new(
    //     atomic_numbers,
    //     positions,
    //     Some(config.mol.charge),
    //     Some(config.mol.multiplicity),
    //     Some(0.0),
    //     None,
    //     config.clone(),
    //     None,
    //     None,
    //     None,
    //     None,
    //     None,
    //     None,
    //     None,
    //     None,
    //     None,
    // );
    // let (g1, g1_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
    //     &new_mol.atomic_numbers,
    //     new_mol.n_atoms,
    //     new_mol.calculator.n_orbs,
    //     new_mol.distance_matrix.view(),
    //     new_mol.directions_matrix.view(),
    //     &new_mol.calculator.hubbard_u,
    //     &new_mol.calculator.valorbs,
    //     Some(0.0),
    // );

    // return (
    //     indices_vector,
    //     // new_mol.g0,
    //     new_mol.proximity_matrix,
    //     new_mol.distance_matrix,
    //     new_mol.directions_matrix,
    //     new_mol.calculator.hubbard_u,
    // );
    return (
        indices_vector,
        prox_matrix,
        dist_matrix,
        dir_matrix,
        calculator.hubbard_u
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
    // g0: Array2<f64>,
    // g1: Array3<f64>,
    s: Option<Array2<f64>>,
    // grad_s: Option<Array3<f64>>,
    p_mat: Option<Array2<f64>>,
    pair_n_orbs:Option<usize>,
    pair_valorbs:Option<HashMap<u8, Vec<(i8, i8, i8)>>> ,
    pair_skt:Option<HashMap<(u8, u8), SlaterKosterTable>>,
    pair_orbital_energies:Option<HashMap<u8, HashMap<(i8, i8), f64>>>,
    pair_proximity:Option<Array2<bool>>,
    embedding_gradient:Option<Array1<f64>>,
    dimer_gradient:Option<Array1<f64>>
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
        // g0: Array2<f64>,
        // g1: Array3<f64>,
        s: Option<Array2<f64>>,
        // grad_s: Option<Array3<f64>>,
        p_mat: Option<Array2<f64>>,
        pair_n_orbs:Option<usize>,
        pair_valorbs:Option<HashMap<u8, Vec<(i8, i8, i8)>>> ,
        pair_skt:Option<HashMap<(u8, u8), SlaterKosterTable>>,
        pair_orbital_energies:Option<HashMap<u8, HashMap<(i8, i8), f64>>>,
        pair_proximity:Option<Array2<bool>>,
        embedding_gradient:Option<Array1<f64>>,
        dimer_gradient:Option<Array1<f64>>

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
            // g0: g0,
            // g1: g1,
            s: s,
            // grad_s: grad_s,
            p_mat: p_mat,
            pair_n_orbs:pair_n_orbs,
            pair_valorbs:pair_valorbs,
            pair_skt:pair_skt,
            pair_orbital_energies:pair_orbital_energies,
            pair_proximity:pair_proximity,
            embedding_gradient:embedding_gradient,
            dimer_gradient:dimer_gradient,
        };
        return result;
    }
}

pub struct pair_gradients_result {
    energy_pair: Option<f64>,
    frag_a_index: usize,
    frag_b_index: usize,
    frag_a_atoms: usize,
    frag_b_atoms: usize,
    embedding_gradient:Option<Array1<f64>>,
    dimer_gradient:Option<Array1<f64>>
}

impl pair_gradients_result {
    pub(crate) fn new(
        energy: Option<f64>,
        frag_a_index: usize,
        frag_b_index: usize,
        frag_a_atoms: usize,
        frag_b_atoms: usize,
        embedding_gradient:Option<Array1<f64>>,
        dimer_gradient:Option<Array1<f64>>

    ) -> (pair_gradients_result) {
        let result = pair_gradients_result {
            energy_pair: energy,
            frag_a_index: frag_a_index,
            frag_b_index: frag_b_index,
            frag_a_atoms: frag_a_atoms,
            frag_b_atoms: frag_b_atoms,
            embedding_gradient:embedding_gradient,
            dimer_gradient:dimer_gradient,
        };
        return result;
    }
}
