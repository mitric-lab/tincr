use crate::calculator::{
    get_gamma_gradient_matrix_atom_wise_outer_diagonal, get_gamma_matrix_atomwise_outer_diagonal,
};
use crate::calculator::{
    get_gamma_matrix, get_only_gamma_matrix_atomwise, import_pseudo_atom, Calculator,
    DFTBCalculator,
};
use crate::constants;
use crate::constants::VDW_RADII;
use crate::defaults;
use crate::fmo_ncc_routine::*;
use crate::gradients::{get_gradients, ToOwnedF};
use crate::graph::*;
use crate::h0_and_s::h0_and_s;
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
use ndarray::{stack, Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use peroxide::prelude::*;
use petgraph::algo::{is_isomorphic, is_isomorphic_matching};
use petgraph::dot::{Config, Dot};
use petgraph::stable_graph::*;
use petgraph::{Graph, Undirected};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use crate::fmo_gradients::fmo_fragment_gradients;

pub fn fmo_numerical_gradient(
    atomic_numbers: &Vec<u8>,
    positions: &Array1<f64>,
    config: GeneralConfig,
) -> Array1<f64> {
    let mut gradient: Array1<f64> = Array1::zeros(positions.raw_dim());
    let positions_len: usize = positions.len() / 3;
    let coordinates: Array2<f64> = positions.clone().into_shape((positions_len, 3)).unwrap();
    let subgraph: Vec<StableUnGraph<u8, f64>> =
        create_fmo_graph(atomic_numbers.clone(), coordinates.clone());

    let energy: f64 = calculate_energy_for_coordinates(
        atomic_numbers,
        &coordinates,
        config.clone(),
        subgraph.clone(),
    );
    println!("FMO Energy num gradient {}", energy);
    println!("");

    for ind in (0..positions.len()).into_iter() {
        let energy_1: f64 = numerical_gradient_routine(
            &atomic_numbers,
            positions,
            config.clone(),
            subgraph.clone(),
            1e-5,
            ind,
        );
        let energy_2: f64 = numerical_gradient_routine(
            &atomic_numbers,
            positions,
            config.clone(),
            subgraph.clone(),
            -1e-5,
            ind,
        );

        let grad_temp: f64 = (energy_1 - energy_2) / (2.0 * 1e-5);
        gradient[ind] = grad_temp;
    }
    return gradient;
}

pub fn fmo_numerical_gradient_new(
    atomic_numbers: &Vec<u8>,
    positions: &Array1<f64>,
    config: GeneralConfig,
) -> Array1<f64> {
    // numerical gradient with higher accuracy
    // employing Ridders' method of polynomial extrapolation to calculate the derivative
    // of the fmo energy

    let mut gradient: Array1<f64> = Array1::zeros(positions.raw_dim());
    let positions_len: usize = positions.len() / 3;
    let coordinates: Array2<f64> = positions.clone().into_shape((positions_len, 3)).unwrap();
    let subgraph: Vec<StableUnGraph<u8, f64>> =
        create_fmo_graph(atomic_numbers.clone(), coordinates.clone());

    let energy: f64 = calculate_energy_for_coordinates(
        atomic_numbers,
        &coordinates,
        config.clone(),
        subgraph.clone(),
    );
    println!("FMO Energy num gradient ridders {}", energy);
    println!("");

    // parameters for ridders' method
    let con: f64 = 1.4;
    let con_2: f64 = con.powi(2);
    let ntab: usize = 20;
    let safe: f64 = 2.0;
    let h: f64 = 5e-3;
    let big: f64 = 1e30;

    for ind in (0..positions.len()).into_iter() {
        let mut a_mat: Array2<f64> = Array2::zeros((ntab, ntab));
        let energy_1: f64 = numerical_gradient_routine(
            &atomic_numbers,
            positions,
            config.clone(),
            subgraph.clone(),
            h,
            ind,
        );
        let energy_2: f64 = numerical_gradient_routine(
            &atomic_numbers,
            positions,
            config.clone(),
            subgraph.clone(),
            -h,
            ind,
        );

        a_mat[[0, 0]] = (energy_1 - energy_2) / (2.0 * h);
        let mut hh: f64 = h;
        let mut err: f64 = big;
        let mut ans: f64 = 0.0;

        for i in (1..ntab + 1).into_iter() {
            hh = hh / con;
            let energy_1: f64 = numerical_gradient_routine(
                &atomic_numbers,
                positions,
                config.clone(),
                subgraph.clone(),
                hh,
                ind,
            );
            let energy_2: f64 = numerical_gradient_routine(
                &atomic_numbers,
                positions,
                config.clone(),
                subgraph.clone(),
                -hh,
                ind,
            );
            a_mat[[0, i]] = (energy_1 - energy_2) / (2.0 * hh);
            let mut fac: f64 = con_2;

            for j in (1..i + 1).into_iter() {
                a_mat[[j, i]] = (a_mat[[j - 1, i]] * fac - a_mat[[j - 1, i - 1]]) / (fac - 1.0);
                fac = con_2 * fac;
                let errt: f64 = (a_mat[[j, i]] - a_mat[[j - 1, i]])
                    .abs()
                    .max((a_mat[[j, i]] - a_mat[[j - 1, i - 1]]).abs());
                // println!("Errt {}",errt);
                // println!("Err {}",err);

                if errt <= err {
                    err = errt;
                    ans = a_mat[[j, i]];
                    // println!("ans {}",ans);
                }
            }
            if (a_mat[[i, i]] - a_mat[[i - 1, i - 1]]).abs() >= safe * err {
                // println!("Break at i = {}",i);
                break;
            }
        }
        gradient[ind] = ans;
    }
    return gradient;
}

pub fn numerical_gradient_routine(
    atomic_numbers: &Vec<u8>,
    positions: &Array1<f64>,
    config: GeneralConfig,
    subgraph: Vec<StableUnGraph<u8, f64>>,
    h: f64,
    ind: usize,
) -> (f64) {
    let positions_len: usize = positions.len() / 3;
    let mut ei: Array1<f64> = Array1::zeros(positions.raw_dim());
    ei[ind] = 1.0;

    let positions: Array1<f64> = positions + &(h * ei.clone());
    let coordinates_1: Array2<f64> = positions.into_shape((positions_len, 3)).unwrap();
    let energy: f64 =
        calculate_energy_for_coordinates(atomic_numbers, &coordinates_1, config.clone(), subgraph);
    return energy;
}

pub fn calculate_energy_for_coordinates(
    atomic_numbers: &Vec<u8>,
    positions: &Array2<f64>,
    config: GeneralConfig,
    subgraph: Vec<StableUnGraph<u8, f64>>,
) -> (f64) {
    let mut fragments: Vec<Molecule> = create_fragment_molecules(
        subgraph.clone(),
        config.clone(),
        atomic_numbers.clone(),
        positions.clone(),
    );
    let (indices_frags, gamma_total, prox_mat, dist_mat, direct_mat): (
        Vec<usize>,
        Array2<f64>,
        Array2<bool>,
        Array2<f64>,
        Array3<f64>,
    ) = reorder_molecule(&fragments, config.clone(), positions.raw_dim());

    let fragments_data: cluster_frag_result = fmo_calculate_fragments(&mut fragments);

    let (h0, pairs_data): (Array2<f64>, Vec<pair_result>) = fmo_calculate_pairwise_single(
        &fragments,
        &fragments_data,
        config.clone(),
        &dist_mat,
        &direct_mat,
        &prox_mat,
        &indices_frags,
    );

    let energy: f64 = fmo_gs_energy(
        &fragments,
        &fragments_data,
        &pairs_data,
        &indices_frags,
        gamma_total,
        prox_mat,
    );
    return energy;
}

pub fn fmo_gs_energy(
    fragments: &Vec<Molecule>,
    cluster_results: &cluster_frag_result,
    pair_results: &Vec<pair_result>,
    indices_frags: &Vec<usize>,
    gamma_total: Array2<f64>,
    prox_mat: Array2<bool>,
) -> (f64) {
    // sum over all monomer energies
    let energy_monomers: f64 = cluster_results.energy.sum();
    println!("Monomer energies old {}", energy_monomers);

    // get energy term for pairs
    let mut iter: usize = 0;
    let mut pair_energies: f64 = 0.0;
    let mut embedding_potential: f64 = 0.0;

    let proximity_zeros: Array1<f64> = prox_mat
        .iter()
        .filter_map(|&item| if item == true { Some(1.0) } else { Some(0.0) })
        .collect();
    let gamma_zeros: Array2<f64> = proximity_zeros.into_shape((prox_mat.raw_dim())).unwrap();
    let gamma_tmp: Array2<f64> = 1.0 * gamma_total;

    for pair in pair_results.iter() {
        let mut pair_energy: f64 = 0.0;
        if pair.energy_pair.is_some() {
            let pair_charges: Array1<f64> = pair.pair_charges.clone().unwrap();
            // E_ij - E_i - E_j
            pair_energy = pair.energy_pair.unwrap()
                - cluster_results.energy[pair.frag_a_index]
                - cluster_results.energy[pair.frag_b_index];

            // get embedding potential of pairs
            // only relevant if the scc energy of the pair was calculated
            // TODO: Change loop over a to matrix multiplications
            let pair_atoms: usize =
                fragments[pair.frag_a_index].n_atoms + fragments[pair.frag_b_index].n_atoms;
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
            let frag_a_atoms: usize = fragments[pair.frag_a_index].n_atoms;
            let frag_b_atoms: usize = fragments[pair.frag_b_index].n_atoms;
            let index_pair_a: usize = indices_frags[pair.frag_a_index];
            let index_pair_b: usize = indices_frags[pair.frag_b_index];

            // TODO:deactivate par_iter if threads = 1
            let embedding_pot: Vec<f64> = fragments
                .iter()
                .enumerate()
                .filter_map(|(ind_k, mol_k)| {
                    if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index {
                        let index_frag_iter: usize = indices_frags[ind_k];
                        let g0_trimer_a: ArrayView2<f64> = gamma_tmp.slice(s![
                            index_pair_a..index_pair_a + frag_a_atoms,
                            index_frag_iter..index_frag_iter + mol_k.n_atoms
                        ]);
                        let g0_trimer_b: ArrayView2<f64> = gamma_tmp.slice(s![
                            index_pair_b..index_pair_b + frag_b_atoms,
                            index_frag_iter..index_frag_iter + mol_k.n_atoms
                        ]);
                        let g0_trimer_ak: Array2<f64> =
                            stack(Axis(0), &[g0_trimer_a, g0_trimer_b]).unwrap();
                        let embedding: f64 = ddq_arr.dot(&g0_trimer_ak.dot(&mol_k.final_charges));

                        Some(embedding)
                    } else {
                        None
                    }
                })
                .collect();

            let embedding_pot_sum: f64 = embedding_pot.sum();
            embedding_potential += embedding_pot_sum;

        //for a in (0..pair_atoms).into_iter(){
        //    println!("Atom {} in pair",a);
        //    let mut ddq:f64 = 0.0;
        //    // check if atom a sits on fragment a or b of the pair
        //    let mut index_a:usize = 0;
        //    if a < pair.frag_a_atoms{
        //        ddq = pair_charges[a] - fragments[pair.frag_a_index].final_charges[a];
        //        index_a = indices_frags[pair.frag_a_index] + a;
        //    }
        //    else{
        //        ddq = pair_charges[a] - fragments[pair.frag_b_index].final_charges[a-pair.frag_a_atoms];
        //        index_a = indices_frags[pair.frag_b_index] + (a -pair.frag_a_atoms);
        //    }
        //    //for (ind_k, mol_k) in fragments.iter().enumerate(){
        //    let embedding_pot:Vec<f64> = fragments.par_iter().enumerate().filter_map(|(ind_k,mol_k)| if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index{
        //        let mut embedding:f64 = 0.0;
        //        if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index{
        //            println!("Fragment Index {}",ind_k);
        //
        //            for c in (0..mol_k.n_atoms).into_iter(){
        //                let c_index:usize = indices_frags[ind_k] + c;
        //                println!("Atom {} in Fragment",c);
        //                // embedding_potential = gamma_ac ddq_a^ij dq_c^k
        //                embedding += gamma_tmp[[index_a,c_index]] *ddq * mol_k.final_charges[c];
        //            }
        //        }
        //        Some(embedding)
        //    }
        //    else{
        //        None
        //    }).collect();
        //    let embedding_pot_sum:f64 = embedding_pot.sum();
        //    embedding_potential += embedding_pot_sum;
        //}
        } else {
            // E_ij = E_i + E_j + sum_(a in I) sum_(B in j) gamma_ab dq_a^i dq_b^j
            let index_pair_a: usize = indices_frags[pair.frag_a_index];
            let index_pair_b: usize = indices_frags[pair.frag_b_index];
            let gamma_ab: ArrayView2<f64> = gamma_tmp.slice(s![
                index_pair_a..index_pair_a + fragments[pair.frag_a_index].n_atoms,
                index_pair_b..index_pair_b + fragments[pair.frag_b_index].n_atoms
            ]);
            //let gamma_ab: ArrayView2<f64> = pair_results[iter].pair_gamma.slice(s![
            //                0..fragments[pair.frag_a_index].n_atoms,
            //                fragments[pair.frag_a_index].n_atoms..
            //            ]);
            pair_energy += fragments[pair.frag_a_index]
                .final_charges
                .dot(&gamma_ab.dot(&fragments[pair.frag_b_index].final_charges));
            // loop version
            //for a in (0..fragments[pair.frag_a_index].n_atoms).into_iter() {
            //    for b in (0..fragments[pair.frag_b_index].n_atoms).into_iter() {
            //        pair_energy += pair_results[iter].pair_gamma[[a, b]]
            //            * fragments[pair.frag_a_index].final_charges[a]
            //            * fragments[pair.frag_b_index].final_charges[b];
            //    }
            //}
            //let pair_sum:Vec<f64> = (0..fragments[pair.frag_a_index].n_atoms).into_par_iter().map(|a|{
            //    let mut pair_energy_loop:f64 = 0.0;
            //    for b in (0..fragments[pair.frag_b_index].n_atoms).into_iter() {
            //        pair_energy_loop += pair_results[iter].pair_gamma[[a, b]]
            //            * fragments[pair.frag_a_index].final_charges[a]
            //            * fragments[pair.frag_b_index].final_charges[b];
            //    }
            //    pair_energy_loop
            //}).collect();
            //pair_energy += pair_sum.sum();
        }
        iter += 1;
        pair_energies += pair_energy;
    }
    let e_tot: f64 = energy_monomers + pair_energies + embedding_potential;
    return e_tot;
}

pub fn fmo_construct_mos(
    n_mo: Vec<usize>,
    h_0_complete: &Array2<f64>,
    n_frags: usize,
    homo_orbs: Vec<Array1<f64>>,
    lumo_orbs: Vec<Array1<f64>>,
) -> Array2<f64> {
    let (e_vals, e_vecs): (Array1<f64>, Array2<f64>) = h_0_complete.eigh(UPLO::Upper).unwrap();
    let n_aos: usize = n_mo.iter().cloned().max().unwrap();
    let mut orbs: Array2<f64> = Array2::zeros((n_frags * 2, n_frags * n_aos));

    for idx in (0..n_frags).into_iter() {
        let i: usize = 2 * idx;
        let j_1: usize = n_aos * idx;
        let j_2: usize = n_aos * (idx + 1);

        orbs.slice_mut(s![i, j_1..j_2]).assign(&homo_orbs[idx]);
        orbs.slice_mut(s![i + 1, j_1..j_2]).assign(&lumo_orbs[idx]);
    }
    let orbs_final: Array2<f64> = e_vecs.t().to_owned().dot(&orbs).t().to_owned();

    return (orbs_final);
}

pub fn fmo_calculate_pairwise(
    mol: &Molecule,
    fragments: &Vec<Molecule>,
    n_mo: Vec<usize>,
    h_0_complete: &Array2<f64>,
    homo_orbs: Vec<Array1<f64>>,
    lumo_orbs: Vec<Array1<f64>>,
    config: GeneralConfig,
) -> (Array2<f64>, Vec<Molecule>) {
    let mut s_complete: Array2<f64> = Array2::zeros(h_0_complete.raw_dim());
    let mut h_0_complete_mut: Array2<f64> = h_0_complete.clone();
    let mut pair_vec: Vec<Molecule> = Vec::new();

    for (ind1, molecule_a) in fragments.iter().enumerate() {
        for (ind2, molecule_b) in fragments.iter().enumerate() {
            if ind1 == ind2 || ind2 < ind1 {
                continue;
            }
            // construct the pair of monomers
            let mut atomic_numbers: Vec<u8> = Vec::new();
            let mut positions: Array2<f64> =
                Array2::zeros((molecule_a.n_atoms + molecule_b.n_atoms, 3));

            for i in 0..molecule_a.n_atoms {
                atomic_numbers.push(molecule_a.atomic_numbers[i]);
                positions
                    .slice_mut(s![i, ..])
                    .assign(&molecule_a.positions.slice(s![i, ..]));
            }
            for i in 0..molecule_b.n_atoms {
                atomic_numbers.push(molecule_b.atomic_numbers[i]);
                positions
                    .slice_mut(s![molecule_a.n_atoms + i, ..])
                    .assign(&molecule_b.positions.slice(s![i, ..]));
            }
            let pair: Molecule = Molecule::new(
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
            // compute Slater-Koster matrix elements for overlap (S) and 0-th order Hamiltonian (H0)
            let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(
                &pair.atomic_numbers,
                pair.positions.view(),
                pair.calculator.n_orbs,
                &pair.calculator.valorbs,
                pair.proximity_matrix.view(),
                &pair.calculator.skt,
                &pair.calculator.orbital_energies,
            );
            pair_vec.push(pair);
            // Now select off-diagonal couplings. The block `H0_AB` contains matrix elements
            // between atomic orbitals on fragments A and B:
            //
            //      ( H0_AA  H0_AB )
            // H0 = (              )
            //      ( H0_BA  H0_BB )
            let h0_ab: Array2<f64> = h0.slice(s![0..n_mo[ind1], n_mo[ind2]..]).to_owned();
            let s_ab: Array2<f64> = s.slice(s![0..n_mo[ind1], n_mo[ind2]..]).to_owned();

            // contract Hamiltonian with coefficients of HOMOs on fragments A and B
            let i: usize = ind1 * 2;
            let j: usize = ind2 * 2;

            h_0_complete_mut[[i, j]] = homo_orbs[ind1].dot(&h0_ab.dot(&homo_orbs[ind2]));
            s_complete[[i, j]] = homo_orbs[ind1].dot(&h0_ab.dot(&homo_orbs[ind2]));

            let i: usize = ind1 * 2 + 1;
            let j: usize = ind2 * 2 + 1;

            h_0_complete_mut[[i, j]] = lumo_orbs[ind1].dot(&s.dot(&lumo_orbs[ind2]));
            s_complete[[i, j]] = homo_orbs[ind1].dot(&s.dot(&homo_orbs[ind2]));
        }
    }
    h_0_complete_mut = h_0_complete_mut.clone()
        + (h_0_complete_mut.clone() - Array::from_diag(&h_0_complete_mut.diag())).reversed_axes();

    return (h_0_complete_mut, pair_vec);
}

pub fn fmo_calculate_pairwise_single(
    fragments: &Vec<Molecule>,
    cluster_results: &cluster_frag_result,
    config: GeneralConfig,
    dist_mat: &Array2<f64>,
    direct_mat: &Array3<f64>,
    prox_mat: &Array2<bool>,
    indices_frags: &Vec<usize>,
) -> (Array2<f64>, Vec<pair_result>) {
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

    let mut vec_pair_result: Vec<pair_result> = Vec::new();
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

                // do scc routine for pair if mininmal distance is below threshold
                if (min_dist / vdw_radii_sum) < 2.0 {
                    // println!("New gamma {}",gamma_frag);

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
                    // println!("old gamma {}",pair.g0);
                    // assert!(gamma_frag.abs_diff_eq(&pair.g0,1e-10),"Gamma is wroooooooooooooooooooooooooooooooooooooooooooooooooooooooooong!!!!!");

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
                    energy_pair = Some(energy);
                    charges_pair = Some(pair.final_charges);
                }

                let mut indices_vec: Vec<(usize, usize)> = Vec::new();
                let mut h0_vals: Vec<f64> = Vec::new();

                let pair_res: pair_result = pair_result::new(
                    charges_pair,
                    h0_vals,
                    indices_vec,
                    energy_pair,
                    ind1,
                    ind2,
                    molecule_a.n_atoms,
                    molecule_b.n_atoms,
                );

                vec_pair_result.push(pair_res);
            }
        }
    }

    let mut h_0_complete_mut: Array2<f64> = cluster_results.h_0.clone();
    // for pair in pair_result.iter() {
    //     h_0_complete_mut[[pair.h0_indices[0].0, pair.h0_indices[0].1]] = pair.h0_vals[0];
    //     h_0_complete_mut[[pair.h0_indices[1].0, pair.h0_indices[1].1]] = pair.h0_vals[1];
    // }
    //
    // h_0_complete_mut = h_0_complete_mut.clone()
    //     + (h_0_complete_mut.clone() - Array::from_diag(&h_0_complete_mut.diag())).reversed_axes();

    return (h_0_complete_mut, vec_pair_result);
}

pub fn fmo_calculate_pairwise_par(
    fragments: &Vec<Molecule>,
    cluster_results: &cluster_frag_result,
    config: GeneralConfig,
    dist_mat: &Array2<f64>,
    direct_mat: &Array3<f64>,
    prox_mat: &Array2<bool>,
    indices_frags: &Vec<usize>,
) -> (Array2<f64>, Vec<pair_result>) {
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

    let mut result: Vec<Vec<pair_result>> = fragments
        .par_iter()
        .enumerate()
        .map(|(ind1, molecule_a)| {
            let mut vec_pair_result: Vec<pair_result> = Vec::new();
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
                    // get indices of the atoms
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
                        energy_pair = Some(energy);
                        charges_pair = Some(pair.final_charges);
                    }

                    let mut indices_vec: Vec<(usize, usize)> = Vec::new();
                    let mut h0_vals: Vec<f64> = Vec::new();

                    let pair_res: pair_result = pair_result::new(
                        charges_pair,
                        h0_vals,
                        indices_vec,
                        energy_pair,
                        ind1,
                        ind2,
                        molecule_a.n_atoms,
                        molecule_b.n_atoms,
                    );

                    vec_pair_result.push(pair_res);
                }
            }
            vec_pair_result
        })
        .collect();

    // transform Vec<Vec> back to Vec<>
    let mut pair_result: Vec<pair_result> = Vec::new();
    for pair in result.iter_mut() {
        pair_result.append(pair);
    }
    let mut h_0_complete_mut: Array2<f64> = cluster_results.h_0.clone();

    return (h_0_complete_mut, pair_result);
}

pub fn fmo_calculate_pairs_esdim_embedding(
    fragments: &Vec<Molecule>,
    cluster_results: &cluster_frag_result,
    config: GeneralConfig,
    dist_mat: &Array2<f64>,
    direct_mat: &Array3<f64>,
    prox_mat: &Array2<bool>,
    indices_frags: &Vec<usize>,
    full_hubbard: &HashMap<u8, f64>,
) -> (f64) {
    // construct a first graph in case all monomers are the same
    let mol_a = fragments[0].clone();
    let mol_b = fragments[1].clone();
    let mut atomic_numbers: Vec<u8> = Vec::new();
    atomic_numbers.append(&mut mol_a.atomic_numbers.clone());
    atomic_numbers.append(&mut mol_b.atomic_numbers.clone());

    let distance_frag: Array2<f64> = dist_mat
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
    let (atomtypes, unique_numbers): (HashMap<u8, String>, Vec<u8>) =
        get_atomtypes(atomic_numbers.clone());
    let first_calc: DFTBCalculator =
        DFTBCalculator::new(&atomic_numbers, &atomtypes, None, &distance_frag, Some(0.0));

    let mut result: Vec<Vec<f64>> = fragments
        .par_iter()
        .enumerate()
        .map(|(ind1, molecule_a)| {
            let mut vec_pair_result: Vec<f64> = Vec::new();
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
                    let mut embedding_energy: Option<f64> = None;
                    let mut charges_pair: Option<Array1<f64>> = None;

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
                        // energy_pair = Some(energy);
                        // charges_pair = Some(pair.final_charges);

                        let pair_atoms: usize = pair.n_atoms;
                        let pair_charges: Array1<f64> = pair.final_charges.clone();
                        let dimer_pmat: Array2<f64> = pair.final_p_matrix.clone();
                        let pair_smat: Array2<f64> = s;
                        let frag_a_atoms: usize = fragments[ind1].n_atoms;
                        let frag_b_atoms: usize = fragments[ind2].n_atoms;
                        let index_pair_a: usize = indices_frags[ind1];
                        let index_pair_b: usize = indices_frags[ind2];
                        let dimer_atomic_numbers: Vec<u8> = pair.atomic_numbers.clone();
                        let pair_energy =
                            energy - cluster_results.energy[ind1] - cluster_results.energy[ind2];

                        // get embedding potential of pairs
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
                        let ddq_arr: Array1<f64> = Array::from(ddq_vec);

                        let embedding_pot: Vec<f64> = fragments
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
                                    let trimer_distances: Array2<f64> =
                                        stack(Axis(0), &[trimer_distances_a, trimer_distances_b])
                                            .unwrap();

                                    let g0_trimer_ak: Array2<f64> =
                                        get_gamma_matrix_atomwise_outer_diagonal(
                                            &dimer_atomic_numbers,
                                            &mol_k.atomic_numbers,
                                            pair_atoms,
                                            mol_k.n_atoms,
                                            trimer_distances.view(),
                                            full_hubbard,
                                            Some(0.0),
                                        );

                                    let embedding: f64 =
                                        ddq_arr.dot(&g0_trimer_ak.dot(&mol_k.final_charges));
                                    Some(embedding)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let embedding_pot_sum: f64 = embedding_pot.sum();
                        // energy_pair = Some(pair_energy + embedding_pot_sum);
                        energy_pair = Some(pair_energy); //+ embedding_pot_sum);
                    } else {
                        let index_pair_a: usize = indices_frags[ind1];
                        let index_pair_b: usize = indices_frags[ind2];
                        let dimer_distances: ArrayView2<f64> = dist_mat.slice(s![
                            index_pair_a..index_pair_a + fragments[ind1].n_atoms,
                            index_pair_b..index_pair_b + fragments[ind2].n_atoms
                        ]);

                        let g0_dimer_ab: Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
                            &fragments[ind1].atomic_numbers,
                            &fragments[ind2].atomic_numbers,
                            fragments[ind1].n_atoms,
                            fragments[ind2].n_atoms,
                            dimer_distances,
                            full_hubbard,
                            Some(0.0),
                        );

                        let pair_energy = fragments[ind1]
                            .final_charges
                            .dot(&g0_dimer_ab.dot(&fragments[ind2].final_charges));

                        energy_pair = Some(pair_energy);
                    }

                    // let pair_res: pair_energy_result = pair_energy_result::new(
                    //     energy_pair,
                    //     ind1,
                    //     ind2,
                    //     molecule_a.n_atoms,
                    //     molecule_b.n_atoms,
                    // );

                    vec_pair_result.push(energy_pair.unwrap());
                }
            }
            vec_pair_result
        })
        .collect();

    // transform Vec<Vec> back to Vec<>
    let mut pair_result: Vec<f64> = Vec::new();
    for pair in result.iter_mut() {
        pair_result.append(pair);
    }
    let energy_monomers: f64 = cluster_results.energy.sum();
    let mut total_energy: f64 = pair_result.sum() + energy_monomers;

    return (total_energy);
}

pub struct pair_result {
    pair_charges: Option<Array1<f64>>,
    h0_vals: Vec<f64>,
    h0_indices: Vec<(usize, usize)>,
    energy_pair: Option<f64>,
    frag_a_index: usize,
    frag_b_index: usize,
    frag_a_atoms: usize,
    frag_b_atoms: usize,
}

impl pair_result {
    pub(crate) fn new(
        pair_charges: Option<Array1<f64>>,
        h0_vals: Vec<f64>,
        h0_indices: Vec<(usize, usize)>,
        energy: Option<f64>,
        frag_a_index: usize,
        frag_b_index: usize,
        frag_a_atoms: usize,
        frag_b_atoms: usize,
    ) -> (pair_result) {
        let result = pair_result {
            pair_charges: pair_charges,
            h0_vals: h0_vals,
            h0_indices: h0_indices,
            energy_pair: energy,
            frag_a_index: frag_a_index,
            frag_b_index: frag_b_index,
            frag_a_atoms: frag_a_atoms,
            frag_b_atoms: frag_b_atoms,
        };
        return result;
    }
}

pub struct fragment_result {
    n_mo: usize,
    h_diag: Vec<f64>,
    homo_orbs: Array1<f64>,
    lumo_orbs: Array1<f64>,
    ind_homo: usize,
    ind_lumo: usize,
    energy: f64,
}

impl fragment_result {
    pub(crate) fn new(
        n_mo: usize,
        h_diag: Vec<f64>,
        homo_orbs: Array1<f64>,
        lumo_orbs: Array1<f64>,
        ind_homo: usize,
        ind_lumo: usize,
        energy: f64,
    ) -> (fragment_result) {
        let result = fragment_result {
            n_mo: n_mo,
            h_diag: h_diag,
            homo_orbs: homo_orbs,
            lumo_orbs: lumo_orbs,
            ind_homo: ind_homo,
            ind_lumo: ind_lumo,
            energy: energy,
        };
        return result;
    }
}

pub struct cluster_frag_result {
    n_mo: Vec<usize>,
    h_0: Array2<f64>,
    homo_orbs: Vec<Array1<f64>>,
    lumo_orbs: Vec<Array1<f64>>,
    ind_homo: Vec<usize>,
    ind_lumo: Vec<usize>,
    energy: Vec<f64>,
}

impl cluster_frag_result {
    pub(crate) fn new(
        n_mo: Vec<usize>,
        h_0: Array2<f64>,
        homo_orbs: Vec<Array1<f64>>,
        lumo_orbs: Vec<Array1<f64>>,
        ind_homo: Vec<usize>,
        ind_lumo: Vec<usize>,
        energy: Vec<f64>,
    ) -> (cluster_frag_result) {
        let result = cluster_frag_result {
            n_mo: n_mo,
            h_0: h_0,
            homo_orbs: homo_orbs,
            lumo_orbs: lumo_orbs,
            ind_homo: ind_homo,
            ind_lumo: ind_lumo,
            energy: energy,
        };
        return result;
    }
}

pub struct pair_energy_result {
    energy_pair: Option<f64>,
    frag_a_index: usize,
    frag_b_index: usize,
    frag_a_atoms: usize,
    frag_b_atoms: usize,
}

impl pair_energy_result {
    pub(crate) fn new(
        energy: Option<f64>,
        frag_a_index: usize,
        frag_b_index: usize,
        frag_a_atoms: usize,
        frag_b_atoms: usize,
    ) -> (pair_energy_result) {
        let result = pair_energy_result {
            energy_pair: energy,
            frag_a_index: frag_a_index,
            frag_b_index: frag_b_index,
            frag_a_atoms: frag_a_atoms,
            frag_b_atoms: frag_b_atoms,
        };
        return result;
    }
}

pub fn fmo_calculate_fragments(fragments: &mut Vec<Molecule>) -> (cluster_frag_result) {
    //let norb_frag: usize = 2;
    //let size: usize = norb_frag * fragments.len();

    let mut results: Vec<fragment_result> = fragments
        .par_iter_mut()
        .map(|frag| {
            let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
                scc_routine::run_scc(frag);
            let f_temp: Array1<f64> = Array::from(f);
            let occ_indices: Array1<usize> = f_temp
                .indexed_iter()
                .filter_map(|(index, &item)| if item > 0.1 { Some(index) } else { None })
                .collect();
            let virt_indices: Array1<usize> = f_temp
                .indexed_iter()
                .filter_map(|(index, &item)| if item <= 0.1 { Some(index) } else { None })
                .collect();
            let homo_ind: usize = occ_indices[occ_indices.len() - 1];
            let lumo_ind: usize = virt_indices[0];
            let e_homo: f64 = orbe[homo_ind];
            let e_lumo: f64 = orbe[lumo_ind];
            let frag_homo_orbs: Array1<f64> = orbs.slice(s![.., homo_ind]).to_owned();
            let frag_lumo_orbs: Array1<f64> = orbs.slice(s![.., lumo_ind]).to_owned();
            let mut h_diag: Vec<f64> = Vec::new();
            h_diag.push(e_homo);
            h_diag.push(e_lumo);

            let frag_result: fragment_result = fragment_result::new(
                orbs.dim().0,
                h_diag,
                frag_homo_orbs,
                frag_lumo_orbs,
                homo_ind,
                lumo_ind,
                energy,
            );
            frag_result
        })
        .collect();

    let mut n_mo: Vec<usize> = Vec::new();
    let mut h_diag: Vec<f64> = Vec::new();
    let mut homo_orbs: Vec<Array1<f64>> = Vec::new();
    let mut lumo_orbs: Vec<Array1<f64>> = Vec::new();
    let mut ind_homo: Vec<usize> = Vec::new();
    let mut ind_lumo: Vec<usize> = Vec::new();
    let mut energies: Vec<f64> = Vec::new();

    for frag_res in results.iter_mut() {
        n_mo.push(frag_res.n_mo);
        h_diag.append(&mut frag_res.h_diag);
        homo_orbs.push(frag_res.homo_orbs.clone());
        lumo_orbs.push(frag_res.lumo_orbs.clone());
        ind_homo.push(frag_res.ind_homo);
        ind_lumo.push(frag_res.ind_lumo);
        energies.push(frag_res.energy);
    }
    let h_0: Array2<f64> = Array::from_diag(&Array::from(h_diag));

    let cluster_result: cluster_frag_result = cluster_frag_result::new(
        n_mo, h_0, homo_orbs, lumo_orbs, ind_homo, ind_lumo, energies,
    );

    //let h_0: Array2<f64> = Array::from_diag(&Array::from(h_diag));
    //for (ind, frag) in fragments.iter().enumerate() {
    //    let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
    //        scc_routine::run_scc(frag);
    //    let f_temp: Array1<f64> = Array::from(f);
    //    let occ_indices: Array1<usize> = f_temp
    //        .indexed_iter()
    //        .filter_map(|(index, &item)| if item > 0.1 { Some(index) } else { None })
    //        .collect();
    //    let virt_indices: Array1<usize> = f_temp
    //        .indexed_iter()
    //        .filter_map(|(index, &item)| if item <= 0.1 { Some(index) } else { None })
    //        .collect();
    //    let homo_ind: usize = occ_indices[occ_indices.len() - 1];
    //    let lumo_ind: usize = virt_indices[0];
    //    ind_homo.push(homo_ind);
    //    ind_lumo.push(lumo_ind);
    //    let e_homo: f64 = orbe[homo_ind];
    //    let e_lumo: f64 = orbe[lumo_ind];
    //    let frag_homo_orbs: Array1<f64> = orbs.slice(s![.., homo_ind]).to_owned();
    //    let frag_lumo_orbs: Array1<f64> = orbs.slice(s![.., lumo_ind]).to_owned();
    //
    //    h_diag.push(e_homo);
    //    h_diag.push(e_lumo);
    //    n_mo.push(orbs.dim().0);
    //    homo_orbs.push(frag_homo_orbs);
    //    lumo_orbs.push(frag_lumo_orbs);
    //}

    return cluster_result;
}

pub fn create_fragment_molecules(
    subgraphs: Vec<StableUnGraph<u8, f64>>,
    config: GeneralConfig,
    cluster_atomic_numbers: Vec<u8>,
    cluster_positions: Array2<f64>,
) -> Vec<Molecule> {
    //let mut fragments: Vec<Molecule> = Vec::new();
    //for frag in subgraphs.iter() {
    //    let molecule_timer: Instant = Instant::now();
    //
    //    let mut atomic_numbers: Vec<u8> = Vec::new();
    //    let mut positions: Array2<f64> = Array2::zeros((frag.node_count(), 3));
    //
    //    for (ind, val) in frag.node_indices().enumerate() {
    //        atomic_numbers.push(cluster_atomic_numbers[val.index()]);
    //        positions
    //            .slice_mut(s![ind, ..])
    //            .assign(&cluster_positions.slice(s![val.index(), ..]));
    //    }
    //    info!("{:>68} {:>8.2} s", "elapsed time slices:", molecule_timer.elapsed().as_secs_f32());
    //    drop(molecule_timer);
    //    let molecule_timer: Instant = Instant::now();
    //    let frag_mol: Molecule = Molecule::new(
    //        atomic_numbers,
    //        positions,
    //        Some(config.mol.charge),
    //        Some(config.mol.multiplicity),
    //        Some(0.0),
    //        None,
    //        config.clone(),
    //    );
    //    info!("{:>68} {:>8.2} s", "elapsed time create mols:", molecule_timer.elapsed().as_secs_f32());
    //    drop(molecule_timer);
    //    fragments.push(frag_mol);
    //}
    let graphs: Vec<Graph<u8, f64, Undirected>> = subgraphs
        .clone()
        .into_par_iter()
        .map(|graph| {
            let graph: Graph<u8, f64, Undirected> = Graph::from(graph);
            graph
        })
        .collect();
    let mut fragments: Vec<Molecule> = Vec::new();
    let mut saved_calculators: Vec<DFTBCalculator> = Vec::new();
    let mut saved_graphs: Vec<Graph<u8, f64, Undirected>> = Vec::new();

    for (ind_graph, frag) in subgraphs.iter().enumerate() {
        let mut use_saved_calc: bool = false;
        let mut saved_calc: Option<DFTBCalculator> = None;
        if saved_graphs.len() > 0 {
            for (ind_g, saved_graph) in saved_graphs.iter().enumerate() {
                //let test_nodes = |a: &u8, b: &u8| a == b;
                //let test_edges = |_: &f64, _: &f64| false;
                if is_isomorphic_matching(
                    &graphs[ind_graph],
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

        let mut atomic_numbers: Vec<u8> = Vec::new();
        let mut positions: Array2<f64> = Array2::zeros((frag.node_count(), 3));
        for (ind, val) in frag.node_indices().enumerate() {
            atomic_numbers.push(cluster_atomic_numbers[val.index()]);
            positions
                .slice_mut(s![ind, ..])
                .assign(&cluster_positions.slice(s![val.index(), ..]));
        }
        let frag_mol: Molecule = Molecule::new(
            atomic_numbers,
            positions,
            Some(config.mol.charge),
            Some(config.mol.multiplicity),
            Some(0.0),
            None,
            config.clone(),
            saved_calc,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        if use_saved_calc == false {
            saved_calculators.push(frag_mol.calculator.clone());
            saved_graphs.push(graphs[ind_graph].clone());
        }
        fragments.push(frag_mol);
    }

    //let fragments: Vec<Molecule> = subgraphs
    //    .par_iter()
    //    .map(|frag| {
    //        let mut atomic_numbers: Vec<u8> = Vec::new();
    //        let mut positions: Array2<f64> = Array2::zeros((frag.node_count(), 3));
    //        for (ind, val) in frag.node_indices().enumerate() {
    //            atomic_numbers.push(cluster_atomic_numbers[val.index()]);
    //            positions
    //                .slice_mut(s![ind, ..])
    //                .assign(&cluster_positions.slice(s![val.index(), ..]));
    //        }
    //        let frag_mol: Molecule = Molecule::new(
    //            atomic_numbers,
    //            positions,
    //            Some(config.mol.charge),
    //            Some(config.mol.multiplicity),
    //            Some(0.0),
    //            None,
    //            config.clone(),
    //        );
    //        frag_mol
    //    })
    //    .collect();

    return fragments;
}
// TODO: creating the complete cluster as a molecule is problematic
// The creations of a new molecule includes the calculation of the gamma matrix,
// which is too costly for huge clusters
pub fn reorder_molecule(
    fragments: &Vec<Molecule>,
    config: GeneralConfig,
    shape_positions: Ix2,
) -> (
    Vec<usize>,
    Array2<f64>,
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
    return (
        indices_vector,
        new_mol.g0,
        new_mol.proximity_matrix,
        new_mol.distance_matrix,
        new_mol.directions_matrix,
    );
}

pub fn reorder_molecule_v2(
    fragments: &Vec<Molecule>,
    config: GeneralConfig,
    shape_positions: Ix2,
) -> (
    Vec<usize>,
    Array2<f64>,
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

    let calculator: DFTBCalculator =
        DFTBCalculator::new(&atomic_numbers, &atomtypes, None, &dist_matrix, Some(0.0));

    let g0_total: Array2<f64> = get_only_gamma_matrix_atomwise(
        &atomic_numbers,
        atomic_numbers.len(),
        dist_matrix.view(),
        &calculator.hubbard_u,
        Some(0.0),
    );

    return (
        indices_vector,
        g0_total,
        prox_matrix,
        dist_matrix,
        dir_matrix,
        calculator.hubbard_u,
    );
}

pub fn fmo_calculate_fragments_ncc(
    fragments: &mut Vec<Molecule>,
    g0_total: ArrayView2<f64>,
    frag_indices: &Vec<usize>,
) -> (Array1<f64>, Vec<Array2<f64>>, Vec<Array1<f64>>,Vec<Array1<f64>>) {
    let mut converged: bool = false;
    let max_iter: usize = 40;
    let mut energy_old: Array1<f64> = Array1::zeros(fragments.len());
    let mut dq_old: Vec<Array1<f64>> = Vec::new();
    let mut pmat_old: Vec<Array2<f64>> = Vec::new();
    let mut dq_rmsd: Array1<f64> = Array1::zeros(fragments.len());
    let mut p_rmsd: Array1<f64> = Array1::zeros(fragments.len());
    let mut s_matrices: Vec<Array2<f64>> = Vec::new();
    let mut om_monomer_matrices: Vec<Array1<f64>> = Vec::new();
    let conv: f64 = 1e-6;
    let length: usize = fragments.len();
    let ncc_mats: Vec<ncc_matrices> = generate_initial_fmo_monomer_guess(fragments);

    'ncc_loop: for i in 0..max_iter {
        let energies_vec: Vec<f64> = fragments
            .iter_mut()
            .enumerate()
            .map(|(index, frag)| {
                let mut energy: f64 = 0.0;
                let mut s: Array2<f64> =
                    Array2::zeros((frag.calculator.n_orbs, frag.calculator.n_orbs));
                let mut om_monomer: Array1<f64> = Array1::zeros(frag.n_atoms);
                if i == 0 {
                    let (energy_temp, s_temp, x_opt, h0, h0_coul, om_monomer): (
                        f64,
                        Array2<f64>,
                        Option<Array2<f64>>,
                        Option<Array2<f64>>,
                        Option<Array2<f64>>,
                        Option<Array1<f64>>,
                    ) = fmo_ncc(
                        frag,
                        ncc_mats[index].x.clone(),
                        Some(ncc_mats[index].s.clone()),
                        ncc_mats[index].h0.clone(),
                        None,
                        None,
                        None,
                        None,
                        false,
                    );
                    energy = energy_temp;
                    s = s_temp;
                } else {
                    let (energy_temp, s_temp, x_opt, h0, h0_coul, om_temp): (
                        f64,
                        Array2<f64>,
                        Option<Array2<f64>>,
                        Option<Array2<f64>>,
                        Option<Array2<f64>>,
                        Option<Array1<f64>>,
                    ) = fmo_ncc(
                        frag,
                        ncc_mats[index].x.clone(),
                        Some(ncc_mats[index].s.clone()),
                        ncc_mats[index].h0.clone(),
                        Some(dq_old.clone()),
                        Some(g0_total),
                        Some(index),
                        Some(frag_indices.clone()),
                        true,
                    );
                    energy = energy_temp;
                    om_monomer = om_temp.unwrap();
                    if i == 6{
                        let gradient:Array1<f64> = fmo_fragment_gradients(&frag,h0_coul.unwrap().view(),frag_indices,index,s_temp);
                        println!("Gradient of monomer: {}",gradient);
                    }
                }

                // calculate dq diff and pmat diff
                if i == 0 {
                    let dq_diff: Array1<f64> = frag.final_charges.clone();
                    let p_diff: Array2<f64> = frag.final_p_matrix.clone();
                    // dq_rmsd[index] = dq_diff.map(|val| val*val).mean().unwrap().sqrt();
                    dq_rmsd[index] = dq_diff.map(|x| x.abs()).max().unwrap().to_owned();
                    p_rmsd[index] = p_diff.map(|val| val * val).mean().unwrap().sqrt();

                    dq_old.push(frag.final_charges.clone());
                    pmat_old.push(frag.final_p_matrix.clone());
                    s_matrices.push(s);
                    om_monomer_matrices.push(Array1::zeros(length));
                } else {
                    let dq_diff: Array1<f64> = &frag.final_charges - &dq_old[index];
                    let p_diff: Array2<f64> = &frag.final_p_matrix - &pmat_old[index];
                    // dq_rmsd[index] = dq_diff.map(|val| val*val).mean().unwrap().sqrt();
                    dq_rmsd[index] = dq_diff.map(|x| x.abs()).max().unwrap().to_owned();
                    p_rmsd[index] = p_diff.map(|val| val * val).mean().unwrap().sqrt();

                    dq_old[index] = frag.final_charges.clone();
                    pmat_old[index] = frag.final_p_matrix.clone();

                    om_monomer_matrices[index] = om_monomer;
                }
                energy
            })
            .collect();

        // calculate embedding energy
        // let embedding_vec: Vec<f64> = fragments
        //     .iter()
        //     .enumerate()
        //     .map(|(ind_a, frag)| {
        //         let mut embedding: f64 = 0.0;
        //         for (ind_k, mol_k) in fragments.iter().enumerate() {
        //             if ind_k != ind_a {
        //                 // calculate g0 of the pair
        //                 // let g0_dimer_ab:Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
        //                 //     &fragments[ind1].atomic_numbers,
        //                 //     &fragments[ind2].atomic_numbers,
        //                 //     fragments[ind1].n_atoms,
        //                 //     fragments[ind2].n_atoms,
        //                 //     dimer_distances,full_hubbard,
        //                 //     Some(0.0));
        //                 let g0_ab: ArrayView2<f64> = g0_total.slice(s![
        //                     ind_a..ind_a + frag.n_atoms,
        //                     ind_k..ind_k + mol_k.n_atoms
        //                 ]);
        //                 embedding += frag.final_charges.dot(&g0_ab.dot(&mol_k.final_charges));
        //             }
        //         }
        //         embedding
        //     })
        //     .collect();
        // let embedding_arr: Array1<f64> = Array::from(embedding_vec);
        // println!("Embedding energy per monomer {}",embedding_arr);
        let energies_arr: Array1<f64> = Array::from(energies_vec); // + embedding_arr;
        let energy_diff: Array1<f64> = (&energies_arr - &energy_old).mapv(|val| val.abs());
        energy_old = energies_arr;
        // println!("energies of the monomers {}",energy_old);
        // check convergence
        let converged_energies: Array1<usize> = energy_diff
            .iter()
            .filter_map(|&item| if item < conv { Some(1) } else { None })
            .collect();
        let converged_dq: Array1<usize> = dq_rmsd
            .iter()
            .filter_map(|&item| if item < conv { Some(1) } else { None })
            .collect();
        let converged_p: Array1<usize> = p_rmsd
            .iter()
            .filter_map(|&item| if item < conv { Some(1) } else { None })
            .collect();
        // println!("dq diff {}",dq_rmsd);
        if converged_energies.len() == length
            && converged_dq.len() == length
            && converged_p.len() == length
        {
            println!("Iteration {}", i);
            println!(
                "Number of converged fragment energies {}, charges {} and pmatrices {}",
                converged_energies.len(),
                converged_dq.len(),
                converged_p.len()
            );
            break 'ncc_loop;
        } else {
            println!("Iteration {}", i);
            println!(
                "Number of converged fragment energies {}, charges {} and pmatrices {}",
                converged_energies.len(),
                converged_dq.len(),
                converged_p.len()
            );
        }
    }

    return (energy_old, s_matrices, om_monomer_matrices,dq_old);
}

pub fn fmo_ncc_pairs_esdim_embedding(
    fragments: &Vec<Molecule>,
    monomer_energies: ArrayView1<f64>,
    config: GeneralConfig,
    dist_mat: &Array2<f64>,
    direct_mat: &Array3<f64>,
    prox_mat: &Array2<bool>,
    indices_frags: &Vec<usize>,
    full_hubbard: &HashMap<u8, f64>,
    g0_total: ArrayView2<f64>,
    om_monomers: &Vec<Array1<f64>>,
    dq_vec:&Vec<Array1<f64>>,
) -> (f64) {
    // construct a first graph in case all monomers are the same
    let mol_a = fragments[0].clone();
    let mol_b = fragments[1].clone();
    let mut atomic_numbers: Vec<u8> = Vec::new();
    atomic_numbers.append(&mut mol_a.atomic_numbers.clone());
    atomic_numbers.append(&mut mol_b.atomic_numbers.clone());

    let distance_frag: Array2<f64> = dist_mat
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
    let (atomtypes, unique_numbers): (HashMap<u8, String>, Vec<u8>) =
        get_atomtypes(atomic_numbers.clone());
    let first_calc: DFTBCalculator =
        DFTBCalculator::new(&atomic_numbers, &atomtypes, None, &distance_frag, Some(0.0));

    let mut result: Vec<Vec<f64>> = fragments
        .iter()
        .enumerate()
        .map(|(ind1, molecule_a)| {
            let mut vec_pair_result: Vec<f64> = Vec::new();
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

                    // let molecule_timer: Instant = Instant::now();
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
                    // println!(
                    //     "{:>20} {:>8.8} s",
                    //     "elapsed time connectivity,graph,isomorphic matching",
                    //     molecule_timer.elapsed().as_secs_f32()
                    // );
                    // let molecule_timer: Instant = Instant::now();
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
                    // println!(
                    //     "{:>20} {:>8.8} s",
                    //     "elapsed time check for esdim",
                    //     molecule_timer.elapsed().as_secs_f32()
                    // );

                    let mut energy_pair: Option<f64> = None;
                    let mut embedding_energy: Option<f64> = None;
                    let mut charges_pair: Option<Array1<f64>> = None;

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
                        // build initial guess for pair energy
                        let mut dq: Array1<f64> = Array1::zeros(pair.n_atoms);
                        dq.slice_mut(s![0..mol_a.n_atoms])
                            .assign(&mol_a.final_charges);
                        dq.slice_mut(s![mol_a.n_atoms..])
                            .assign(&mol_b.final_charges);
                        pair.set_final_charges(dq);

                        let (energy, s, h0_coul): (f64, Array2<f64>, Option<Array2<f64>>) =
                            fmo_pair_ncc(
                                &mut pair,
                                mol_a.n_atoms,
                                mol_b.n_atoms,
                                ind1,
                                ind2,
                                false,
                                om_monomers,
                                dq_vec,
                                g0_total,
                                indices_frags
                            );
                        // energy_pair = Some(energy);
                        // charges_pair = Some(pair.final_charges);

                        let pair_atoms: usize = pair.n_atoms;
                        let pair_charges: Array1<f64> = pair.final_charges.clone();
                        let dimer_pmat: Array2<f64> = pair.final_p_matrix.clone();
                        let pair_smat: Array2<f64> = s;
                        let frag_a_atoms: usize = fragments[ind1].n_atoms;
                        let frag_b_atoms: usize = fragments[ind2].n_atoms;
                        let index_pair_a: usize = indices_frags[ind1];
                        let index_pair_b: usize = indices_frags[ind2];
                        let dimer_atomic_numbers: Vec<u8> = pair.atomic_numbers.clone();
                        let pair_energy = energy - monomer_energies[ind1] - monomer_energies[ind2];

                        // get embedding potential of pairs
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
                        let ddq_arr: Array1<f64> = Array::from(ddq_vec);

                        let embedding_pot: Vec<f64> = fragments
                            .iter()
                            .enumerate()
                            .filter_map(|(ind_k, mol_k)| {
                                if ind_k != ind1 && ind_k != ind2 {
                                    let index_frag_iter: usize = indices_frags[ind_k];
                                    // let trimer_distances_a: ArrayView2<f64> = dist_mat.slice(s![
                                    //     index_pair_a..index_pair_a + frag_a_atoms,
                                    //     index_frag_iter..index_frag_iter + mol_k.n_atoms
                                    // ]);
                                    // let trimer_distances_b: ArrayView2<f64> = dist_mat.slice(s![
                                    //     index_pair_b..index_pair_b + frag_b_atoms,
                                    //     index_frag_iter..index_frag_iter + mol_k.n_atoms
                                    // ]);
                                    // let trimer_distances:Array2<f64> = stack(Axis(0),&[trimer_distances_a,trimer_distances_b]).unwrap();
                                    //
                                    // let g0_trimer_ak: Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
                                    //     &dimer_atomic_numbers,
                                    //     &mol_k.atomic_numbers,
                                    //     pair_atoms,
                                    //     mol_k.n_atoms,
                                    //     trimer_distances.view(),
                                    //     full_hubbard,
                                    //     Some(0.0));
                                    let g0_trimer_a: ArrayView2<f64> = g0_total.slice(s![
                                        index_pair_a..index_pair_a + frag_a_atoms,
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms
                                    ]);
                                    let g0_trimer_b: ArrayView2<f64> = g0_total.slice(s![
                                        index_pair_b..index_pair_b + frag_b_atoms,
                                        index_frag_iter..index_frag_iter + mol_k.n_atoms
                                    ]);
                                    let g0_trimer_ak: Array2<f64> =
                                        stack(Axis(0), &[g0_trimer_a, g0_trimer_b]).unwrap();

                                    let embedding: f64 =
                                        ddq_arr.dot(&g0_trimer_ak.dot(&mol_k.final_charges));
                                    Some(embedding)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let embedding_pot_sum: f64 = embedding_pot.sum();
                        energy_pair = Some(pair_energy + embedding_pot_sum);
                    } else {
                        // TODO: calculate g0 on the fly if total g0.is_none()
                        let index_pair_a: usize = indices_frags[ind1];
                        let index_pair_b: usize = indices_frags[ind2];
                        // let dimer_distances: ArrayView2<f64> = dist_mat.slice(s![
                        //     index_pair_a..index_pair_a + fragments[ind1].n_atoms,
                        //     index_pair_b..index_pair_b + fragments[ind2].n_atoms
                        // ]);
                        //
                        // let g0_dimer_ab:Array2<f64> = get_gamma_matrix_atomwise_outer_diagonal(
                        //     &fragments[ind1].atomic_numbers,
                        //     &fragments[ind2].atomic_numbers,
                        //     fragments[ind1].n_atoms,
                        //     fragments[ind2].n_atoms,
                        //     dimer_distances,full_hubbard,
                        //     Some(0.0));
                        let g0_dimer_ab: ArrayView2<f64> = g0_total.slice(s![
                            index_pair_a..index_pair_a + fragments[ind1].n_atoms,
                            index_pair_b..index_pair_b + fragments[ind2].n_atoms
                        ]);

                        let pair_energy = fragments[ind1]
                            .final_charges
                            .dot(&g0_dimer_ab.dot(&fragments[ind2].final_charges));

                        energy_pair = Some(pair_energy);
                    }

                    // let pair_res: pair_energy_result = pair_energy_result::new(
                    //     energy_pair,
                    //     ind1,
                    //     ind2,
                    //     molecule_a.n_atoms,
                    //     molecule_b.n_atoms,
                    // );
                    vec_pair_result.push(energy_pair.unwrap());
                }
            }
            vec_pair_result
        })
        .collect();

    // transform Vec<Vec> back to Vec<>
    let mut pair_result: Vec<f64> = Vec::new();
    for pair in result.iter_mut() {
        pair_result.append(pair);
    }
    let energy_monomers: f64 = monomer_energies.sum();
    let mut total_energy: f64 = pair_result.sum() + energy_monomers;

    return (total_energy);
}

pub fn create_fmo_graph(
    atomic_numbers: Vec<u8>,
    positions: Array2<f64>,
) -> (Vec<StableUnGraph<u8, f64>>) {
    let n_atoms: usize = atomic_numbers.len();
    let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
        distance_matrix(positions.view(), None);

    let connectivity_matrix: Array2<bool> =
        build_connectivity_matrix(n_atoms, &dist_matrix, &atomic_numbers);

    let (graph, graph_indexes, subgraphs): (
        StableUnGraph<u8, f64>,
        Vec<NodeIndex>,
        Vec<StableUnGraph<u8, f64>>,
    ) = build_graph(&atomic_numbers, &connectivity_matrix, &dist_matrix);

    return (subgraphs);
}
