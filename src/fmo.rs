use crate::calculator::{
    get_gamma_matrix, get_only_gamma_matrix_atomwise, import_pseudo_atom, Calculator,
    DFTBCalculator,
};
use crate::constants;
use crate::constants::VDW_RADII;
use crate::defaults;
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
use std::time::Instant;

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

    // get energy term for pairs
    let mut iter: usize = 0;
    let mut pair_energies: f64 = 0.0;
    let mut embedding_potential: f64 = 0.0;

    let proximity_zeros: Array1<f64> = prox_mat
        .iter()
        .filter_map(|&item| if item == true { Some(1.0) } else { Some(0.0) })
        .collect();
    let gamma_zeros: Array2<f64> = proximity_zeros.into_shape((prox_mat.raw_dim())).unwrap();
    let gamma_tmp: Array2<f64> = gamma_zeros * gamma_total;

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
                .into_par_iter()
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

            let embedding_pot: Vec<f64> = fragments
                .par_iter()
                .enumerate()
                .filter_map(|(ind_k, mol_k)| {
                    if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index {
                        let index_frag_iter: usize = indices_frags[ind_k];
                        let gamma_ac: ArrayView2<f64> = gamma_tmp.slice(s![
                            index_pair_iter..index_pair_iter + pair_atoms,
                            index_frag_iter..index_frag_iter + mol_k.n_atoms
                        ]);
                        let embedding: f64 = ddq_arr.dot(&gamma_ac.dot(&mol_k.final_charges));

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
            // loop version
            for a in (0..fragments[pair.frag_a_index].n_atoms).into_iter() {
                for b in (0..fragments[pair.frag_b_index].n_atoms).into_iter() {
                    pair_energy += pair_results[iter].pair_gamma[[a, b]]
                        * fragments[pair.frag_a_index].final_charges[a]
                        * fragments[pair.frag_b_index].final_charges[b];
                }
            }
        }
        iter += 1;
        pair_energies += pair_energy;
    }

    //for pair in pair_results.iter() {
    //    let mut pair_energy: f64 = 0.0;
    //    if pair.energy_pair.is_some() {
    //        println!("Pair is some");
    //        // E_ij - E_i - E_j
    //        pair_energy = pair.energy_pair.unwrap()
    //            - cluster_results.energy[pair.frag_a_index]
    //            - cluster_results.energy[pair.frag_b_index];
    //
    //        // get embedding potential of pairs
    //        // only relevant if the scc energy of the pair was calculated
    //        // TODO: Change loop over a to matrix multiplications
    //        for a in (0..pair.pair.n_atoms).into_iter(){
    //            println!("Atom {} in pair",a);
    //            let mut ddq:f64 = 0.0;
    //            // check if atom a sits on fragment a or b of the pair
    //            if a < pair.frag_a_atoms{
    //                ddq = pair.pair.final_charges[a] - fragments[pair.frag_a_index].final_charges[a];
    //            }
    //            else{
    //                ddq = pair.pair.final_charges[a] - fragments[pair.frag_b_index].final_charges[a-pair.frag_a_atoms];
    //            }
    //            //for (ind_k, mol_k) in fragments.iter().enumerate(){
    //            let embedding_pot:Vec<f64> = fragments.par_iter().enumerate().filter_map(|(ind_k,mol_k)| if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index{
    //                let mut embedding:f64 = 0.0;
    //                if ind_k != pair.frag_a_index && ind_k != pair.frag_b_index{
    //                    println!("Fragment Index {}",ind_k);
    //                    // embedding_potential = gamma_ac ddq_a^ij dq_c^k
    //                    // calculate distance matrix for gamma_ac
    //                    let mut new_positions:Array2<f64> = Array::zeros((pair.pair.n_atoms + mol_k.n_atoms,3));
    //                    new_positions.slice_mut(s![..pair.pair.n_atoms,..]).assign(&pair.pair.positions);
    //                    new_positions.slice_mut(s![pair.pair.n_atoms..,..]).assign(&mol_k.positions);
    //                    let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
    //                        distance_matrix(new_positions.view(), Some(10.0));
    //
    //                    // check proximity matrix for distances
    //                    let proximity_indices:Array1<(usize,usize)> = prox_matrix.indexed_iter()
    //                        .filter_map(
    //                            |(index, &item)| if item == true { Some(index) } else { None },
    //                        )
    //                        .collect();
    //
    //                    //let proximity_zeros:Array1<f64> = prox_matrix.iter()
    //                    //    .filter_map(
    //                    //        | &item| if item == true { Some(1.0) } else { Some(0.0)},
    //                    //    )
    //                    //    .collect();
    //
    //                    //let gamma_zeros:Array2<f64> = proximity_zeros.into_shape((pair.pair.n_atoms + mol_k.n_atoms,pair.pair.n_atoms + mol_k.n_atoms)).unwrap();
    //
    //                    // if the fragment is near the pair, calculate gamma matrix
    //                    if proximity_indices.len() > 0{
    //                        let mut atomic_numbers_pair:Vec<u8> = pair.pair.atomic_numbers.clone();
    //                        let mut atomic_numbers_mol:Vec<u8> = mol_k.atomic_numbers.clone();
    //                        let mut atomic_numbers:Vec<u8> = Vec::new();
    //                        atomic_numbers.append(&mut atomic_numbers_pair);
    //                        atomic_numbers.append(&mut atomic_numbers_mol);
    //
    //                        let (atomtypes, unique_numbers): (HashMap<u8, String>, Vec<u8>) =
    //                            get_atomtypes(atomic_numbers.clone());
    //
    //                        let mut hubbard_u: HashMap<u8, f64> = HashMap::new();
    //                        for (zi, symbol) in atomtypes.iter() {
    //                            let (atom, free_atom): (PseudoAtom, PseudoAtom) = import_pseudo_atom(zi);
    //                            hubbard_u.insert(*zi, atom.hubbard_u);
    //                        }
    //
    //                        let g0:Array2<f64>= get_only_gamma_matrix_atomwise(
    //                            &atomic_numbers,
    //                            atomic_numbers.len(),
    //                            dist_matrix.view(),
    //                            &hubbard_u,
    //                            Some(0.0),
    //                        );
    //
    //                        for c in (0..mol_k.n_atoms).into_iter(){
    //                            let c_index:usize = pair.pair.n_atoms + c;
    //                            if prox_matrix[[a,c_index]] == true{
    //                                println!("Atom {} in Fragment",c);
    //                                // embedding_potential = gamma_ac ddq_a^ij dq_c^k
    //                                embedding += g0[[a,c_index]] *ddq * mol_k.final_charges[c];
    //                            }
    //                        }
    //                    }
    //                }
    //            Some(embedding)
    //            }
    //            else{
    //                None
    //            }).collect();
    //            let embedding_pot_sum:f64 = embedding_pot.sum();
    //            embedding_potential += embedding_pot_sum;
    //        }
    //
    //    } else {
    //        // E_ij = E_i + E_j + sum_(a in I) sum_(B in j) gamma_ab dq_a^i dq_b^j
    //        // loop version
    //        for a in (0..fragments[pair.frag_a_index].n_atoms).into_iter() {
    //            for b in (0..fragments[pair.frag_b_index].n_atoms).into_iter() {
    //                pair_energy += pair_results[iter].pair.calculator.g0[[a, b]]
    //                    * fragments[pair.frag_a_index].final_charges[a]
    //                    * fragments[pair.frag_b_index].final_charges[b];
    //            }
    //        }
    //    }
    //    iter += 1;
    //    pair_energies += pair_energy;
    //}
    // calculate total energy
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

pub fn fmo_calculate_pairwise_par(
    fragments: &Vec<Molecule>,
    cluster_results: &cluster_frag_result,
    config: GeneralConfig,
) -> (Array2<f64>, Vec<pair_result>) {
    //let result: Vec<pair_result> = fragments
    //    .into_par_iter()
    //    .enumerate()
    //    .zip(fragments.into_par_iter().enumerate())
    //    .filter_map(|((ind1, molecule_a), (ind2, molecule_b))| {
    //        if ind1 < ind2 {
    //            let mut atomic_numbers: Vec<u8> = Vec::new();
    //            let mut positions: Array2<f64> =
    //                Array2::zeros((molecule_a.n_atoms + molecule_b.n_atoms, 3));
    //
    //            for i in 0..molecule_a.n_atoms {
    //                atomic_numbers.push(molecule_a.atomic_numbers[i]);
    //                positions
    //                    .slice_mut(s![i, ..])
    //                    .assign(&molecule_a.positions.slice(s![i, ..]));
    //            }
    //            for i in 0..molecule_b.n_atoms {
    //                atomic_numbers.push(molecule_b.atomic_numbers[i]);
    //                positions
    //                    .slice_mut(s![molecule_a.n_atoms + i, ..])
    //                    .assign(&molecule_b.positions.slice(s![i, ..]));
    //            }
    //            let mut pair: Molecule = Molecule::new(
    //                atomic_numbers,
    //                positions,
    //                Some(config.mol.charge),
    //                Some(config.mol.multiplicity),
    //                Some(0.0),
    //                None,
    //                config.clone(),
    //            );
    //            // get shortest distance between the fragment atoms of the pair
    //            let distance_between_pair: Array2<f64> = pair
    //                .distance_matrix
    //                .slice(s![..molecule_a.n_atoms, molecule_a.n_atoms..])
    //                .to_owned();
    //            let min_dist: f64 = distance_between_pair
    //                .iter()
    //                .cloned()
    //                .min_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
    //                .unwrap();
    //
    //            // get indices of the atoms
    //            let mut index_min: (usize, usize) = (0, 0);
    //            for (ind_1, val_1) in distance_between_pair.outer_iter().enumerate() {
    //                for (ind_2, val_2) in val_1.iter().enumerate() {
    //                    if *val_2 == min_dist {
    //                        index_min = (ind_1, ind_2);
    //                    }
    //                }
    //            }
    //            //let index_min_vec:Vec<(usize,usize)> = distance_between_pair.indexed_iter()
    //            //    .filter_map(
    //            //        |(index, &item)| if item == min_dist{ Some(index) } else { None },
    //            //    )
    //            //    .collect();
    //            //let index_min = index_min_vec[0];
    //
    //            let vdw_radii_sum: f64 = (constants::VDW_RADII
    //                [&molecule_a.atomic_numbers[index_min.0]]
    //                + constants::VDW_RADII[&molecule_b.atomic_numbers[index_min.1]])
    //                / constants::BOHR_TO_ANGS;
    //            let mut energy_pair: Option<f64> = None;
    //
    //            // do scc routine for pair if mininmal distance is below threshold
    //            if (min_dist / vdw_radii_sum) < 2.0 {
    //                let (energy, orbs, orbe, s, f): (
    //                    f64,
    //                    Array2<f64>,
    //                    Array1<f64>,
    //                    Array2<f64>,
    //                    Vec<f64>,
    //                ) = scc_routine::run_scc(&mut pair);
    //                energy_pair = Some(energy);
    //            }
    //
    //            // compute Slater-Koster matrix elements for overlap (S) and 0-th order Hamiltonian (H0)
    //            let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(
    //                &pair.atomic_numbers,
    //                pair.positions.view(),
    //                pair.calculator.n_orbs,
    //                &pair.calculator.valorbs,
    //                pair.proximity_matrix.view(),
    //                &pair.calculator.skt,
    //                &pair.calculator.orbital_energies,
    //            );
    //            // Now select off-diagonal couplings. The block `H0_AB` contains matrix elements
    //            // between atomic orbitals on fragments A and B:
    //            //
    //            //      ( H0_AA  H0_AB )
    //            // H0 = (              )
    //            //      ( H0_BA  H0_BB )
    //            let mut indices_vec: Vec<(usize, usize)> = Vec::new();
    //            let mut h0_vals: Vec<f64> = Vec::new();
    //
    //            let h0_ab: Array2<f64> = h0
    //                .slice(s![
    //                    0..cluster_results.n_mo[ind1],
    //                    cluster_results.n_mo[ind2]..
    //                ])
    //                .to_owned();
    //            // contract Hamiltonian with coefficients of HOMOs on fragments A and B
    //            let i: usize = ind1 * 2;
    //            let j: usize = ind2 * 2;
    //            indices_vec.push((i, j));
    //            let h0_val: f64 = cluster_results.homo_orbs[ind1]
    //                .dot(&h0_ab.dot(&cluster_results.homo_orbs[ind2]));
    //            h0_vals.push(h0_val);
    //
    //            let i: usize = ind1 * 2 + 1;
    //            let j: usize = ind2 * 2 + 1;
    //            indices_vec.push((i, j));
    //            let h0_val: f64 =
    //                cluster_results.lumo_orbs[ind1].dot(&s.dot(&cluster_results.lumo_orbs[ind2]));
    //            h0_vals.push(h0_val);
    //
    //            let pair_res: pair_result =
    //                pair_result::new(pair, h0_vals, indices_vec, energy_pair, ind1, ind2, molecule_a.n_atoms,molecule_b.n_atoms);
    //
    //            Some(pair_res)
    //        } else {
    //            None
    //        }
    //    })
    //    .collect();

    // construct a first graph in case all monomers are the same
    let molecule_a = fragments[0].clone();
    let molecule_b = fragments[1].clone();
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

    let (graph, subgraph): (StableUnGraph<u8, f64>, Vec<StableUnGraph<u8, f64>>) =
        create_fmo_graph(atomic_numbers.clone(), positions.clone());

    let first_graph: Graph<u8, f64, Undirected> = Graph::from(graph);
    let first_pair: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        Some(config.mol.charge),
        Some(config.mol.multiplicity),
        Some(0.0),
        None,
        config.clone(),
        None,
    );
    let first_calc:DFTBCalculator = first_pair.calculator.clone();


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
                println!("Index 1 {} and Index 2 {}", ind1, ind2);
                if ind1 < ind2 {
                    let mut use_saved_calc: bool = false;
                    let mut saved_calc: Option<DFTBCalculator> = None;

                    let molecule_timer: Instant = Instant::now();
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
                    println!(
                        "{:>68} {:>8.6} s",
                        "elapsed time slices:",
                        molecule_timer.elapsed().as_secs_f32()
                    );
                    drop(molecule_timer);
                    let molecule_timer: Instant = Instant::now();

                    let (graph, subgraph): (StableUnGraph<u8, f64>, Vec<StableUnGraph<u8, f64>>) =
                        create_fmo_graph(atomic_numbers.clone(), positions.clone());

                    let graph: Graph<u8, f64, Undirected> = Graph::from(graph);

                    if saved_graphs.len() > 0{
                        for (ind_g, saved_graph) in saved_graphs.iter().enumerate(){
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
                    println!(
                        "{:>68} {:>8.6} s",
                        "elapsed time create pair graph and check isomorphic:",
                        molecule_timer.elapsed().as_secs_f32()
                    );
                    drop(molecule_timer);

                    let molecule_timer: Instant = Instant::now();
                    let mut pair: Molecule = Molecule::new(
                        atomic_numbers,
                        positions,
                        Some(config.mol.charge),
                        Some(config.mol.multiplicity),
                        Some(0.0),
                        None,
                        config.clone(),
                        saved_calc,
                    );
                    println!(
                        "{:>68} {:>8.6} s",
                        "elapsed time molecule:",
                        molecule_timer.elapsed().as_secs_f32()
                    );
                    drop(molecule_timer);
                    let molecule_timer: Instant = Instant::now();

                    if use_saved_calc == false{
                        saved_calculators.push(pair.calculator.clone());
                        saved_graphs.push(graph.clone());
                    }
                    // get shortest distance between the fragment atoms of the pair
                    let distance_between_pair: Array2<f64> = pair
                        .distance_matrix
                        .slice(s![..molecule_a.n_atoms, molecule_a.n_atoms..])
                        .to_owned();
                    let min_dist: f64 = distance_between_pair
                        .iter()
                        .cloned()
                        .min_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
                        .unwrap();
                    // get indices of the atoms
                    //// TODO: use a more efficient method to determine the indices
                    //let mut index_min: (usize, usize) = (0, 0);
                    //for (index_1, val_1) in distance_between_pair.outer_iter().enumerate() {
                    //    for (index_2, val_2) in val_1.iter().enumerate() {
                    //        if *val_2 == min_dist {
                    //            index_min = (index_1, index_2);
                    //        }
                    //    }
                    //}
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

                    println!(
                        "{:>68} {:>8.6} s",
                        "elapsed time distances:",
                        molecule_timer.elapsed().as_secs_f32()
                    );
                    drop(molecule_timer);
                    let molecule_timer: Instant = Instant::now();

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
                        charges_pair = Some(pair.final_charges);
                    }
                    println!(
                        "{:>68} {:>8.6} s",
                        "elapsed time scc:",
                        molecule_timer.elapsed().as_secs_f32()
                    );
                    drop(molecule_timer);
                    let molecule_timer: Instant = Instant::now();
                    // compute Slater-Koster matrix elements for overlap (S) and 0-th order Hamiltonian (H0)
                    // let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(
                    //     &pair.atomic_numbers,
                    //     pair.positions.view(),
                    //     pair.calculator.n_orbs,
                    //     &pair.calculator.valorbs,
                    //     pair.proximity_matrix.view(),
                    //     &pair.calculator.skt,
                    //     &pair.calculator.orbital_energies,
                    // );
                    // Now select off-diagonal couplings. The block `H0_AB` contains matrix elements
                    // between atomic orbitals on fragments A and B:
                    //
                    //      ( H0_AA  H0_AB )
                    // H0 = (              )
                    //      ( H0_BA  H0_BB )
                    let mut indices_vec: Vec<(usize, usize)> = Vec::new();
                    let mut h0_vals: Vec<f64> = Vec::new();

                    // let h0_ab: Array2<f64> = h0
                    //     .slice(s![
                    //         0..cluster_results.n_mo[ind1],
                    //         cluster_results.n_mo[ind2]..
                    //     ])
                    //     .to_owned();
                    // let s_ab: Array2<f64> = s
                    //     .slice(s![
                    //         0..cluster_results.n_mo[ind1],
                    //         cluster_results.n_mo[ind2]..
                    //     ])
                    //     .to_owned();
                    // // contract Hamiltonian with coefficients of HOMOs on fragments A and B
                    // let i: usize = ind1 * 2;
                    // let j: usize = ind2 * 2;
                    // indices_vec.push((i, j));
                    // let h0_val: f64 = cluster_results.homo_orbs[ind1]
                    //     .dot(&h0_ab.dot(&cluster_results.homo_orbs[ind2]));
                    // h0_vals.push(h0_val);
                    //
                    // let i: usize = ind1 * 2 + 1;
                    // let j: usize = ind2 * 2 + 1;
                    // indices_vec.push((i, j));
                    // println!("Test12345");
                    // let h0_val: f64 = cluster_results.lumo_orbs[ind1]
                    //     .t()
                    //     .dot(&s_ab.dot(&cluster_results.lumo_orbs[ind2]));
                    // h0_vals.push(h0_val);

                    let pair_res: pair_result = pair_result::new(
                        pair.g0,
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
                    println!(
                        "{:>68} {:>8.6} s",
                        "elapsed time excited state preps:",
                        molecule_timer.elapsed().as_secs_f32()
                    );
                    drop(molecule_timer);
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
    // for pair in pair_result.iter() {
    //     h_0_complete_mut[[pair.h0_indices[0].0, pair.h0_indices[0].1]] = pair.h0_vals[0];
    //     h_0_complete_mut[[pair.h0_indices[1].0, pair.h0_indices[1].1]] = pair.h0_vals[1];
    // }
    //
    // h_0_complete_mut = h_0_complete_mut.clone()
    //     + (h_0_complete_mut.clone() - Array::from_diag(&h_0_complete_mut.diag())).reversed_axes();

    return (h_0_complete_mut, pair_result);
}

pub struct pair_result {
    pair_gamma: Array2<f64>,
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
        pair_gamma: Array2<f64>,
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
            pair_gamma: pair_gamma,
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
) -> (Vec<usize>, Array2<f64>, Array2<bool>) {
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
    );
    return (indices_vector, new_mol.g0, new_mol.proximity_matrix);
}

pub fn create_fmo_graph(
    atomic_numbers: Vec<u8>,
    positions: Array2<f64>,
) -> (StableUnGraph<u8, f64>, Vec<StableUnGraph<u8, f64>>) {
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

    return (graph, subgraphs);
}
