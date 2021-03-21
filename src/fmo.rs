use crate::defaults;
use crate::gradients::{get_gradients, ToOwnedF};
use crate::graph::*;
use crate::h0_and_s::h0_and_s;
use crate::internal_coordinates::*;
use crate::io::GeneralConfig;
use crate::molecule::distance_matrix;
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
use petgraph::stable_graph::*;
use rayon::prelude::*;
use std::time::Instant;

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
    let result: Vec<pair_result> = fragments
        .into_par_iter()
        .enumerate()
        .zip(fragments.into_par_iter().enumerate())
        .filter_map(|((ind1, molecule_a), (ind2, molecule_b))| {
            if ind1 < ind2 {
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
                // Now select off-diagonal couplings. The block `H0_AB` contains matrix elements
                // between atomic orbitals on fragments A and B:
                //
                //      ( H0_AA  H0_AB )
                // H0 = (              )
                //      ( H0_BA  H0_BB )
                let mut indices_vec: Vec<(usize, usize)> = Vec::new();
                let mut h0_vals: Vec<f64> = Vec::new();

                let h0_ab: Array2<f64> = h0
                    .slice(s![
                        0..cluster_results.n_mo[ind1],
                        cluster_results.n_mo[ind2]..
                    ])
                    .to_owned();
                // contract Hamiltonian with coefficients of HOMOs on fragments A and B
                let i: usize = ind1 * 2;
                let j: usize = ind2 * 2;
                indices_vec.push((i, j));
                let h0_val: f64 = cluster_results.homo_orbs[ind1]
                    .dot(&h0_ab.dot(&cluster_results.homo_orbs[ind2]));
                h0_vals.push(h0_val);

                let i: usize = ind1 * 2 + 1;
                let j: usize = ind2 * 2 + 1;
                indices_vec.push((i, j));
                let h0_val: f64 =
                    cluster_results.lumo_orbs[ind1].dot(&s.dot(&cluster_results.lumo_orbs[ind2]));
                h0_vals.push(h0_val);

                let pair_res: pair_result = pair_result::new(pair, h0_vals, indices_vec);

                Some(pair_res)
            } else {
                None
            }
        })
        .collect();

    let mut h_0_complete_mut: Array2<f64> = cluster_results.h_0.clone();
    for pair in result.iter() {
        h_0_complete_mut[[pair.h0_indices[0].0, pair.h0_indices[0].1]] = pair.h0_vals[0];
        h_0_complete_mut[[pair.h0_indices[1].0, pair.h0_indices[1].1]] = pair.h0_vals[1];
    }

    h_0_complete_mut = h_0_complete_mut.clone()
        + (h_0_complete_mut.clone() - Array::from_diag(&h_0_complete_mut.diag())).reversed_axes();

    return (h_0_complete_mut, result);
}

pub struct pair_result {
    pair: Molecule,
    h0_vals: Vec<f64>,
    h0_indices: Vec<(usize, usize)>,
}

impl pair_result {
    pub(crate) fn new(
        pair: Molecule,
        h0_vals: Vec<f64>,
        h0_indices: Vec<(usize, usize)>,
    ) -> (pair_result) {
        let result = pair_result {
            pair: pair,
            h0_vals: h0_vals,
            h0_indices: h0_indices,
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

pub fn fmo_calculate_fragments(fragments: &Vec<Molecule>) -> (cluster_frag_result) {
    let norb_frag: usize = 2;
    let size: usize = norb_frag * fragments.len();

    let mut results: Vec<fragment_result> = fragments
        .into_par_iter()
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
    let mut fragments: Vec<Molecule> = Vec::new();

    for frag in subgraphs.iter() {
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
        );
        fragments.push(frag_mol);
    }

    return fragments;
}
// TODO: creating the complete cluster as a molecule is problematic
// The creations of a new molecule includes the calculation of the gamma matrix,
// which is too costly for huge clusters
//pub fn reorder_molecule(
//    mol: &Molecule,
//    fragments: &Vec<Molecule>,
//    config: GeneralConfig,
//) -> Molecule {
//    let mut atomic_numbers: Vec<u8> = Vec::new();
//    let mut positions: Array2<f64> = Array2::zeros(mol.positions.raw_dim());
//
//    for molecule in fragments.iter() {
//        for (ind, atom) in molecule.atomic_numbers.iter().enumerate() {
//            atomic_numbers.push(*atom);
//            positions
//                .slice_mut(s![ind, ..])
//                .assign(&molecule.positions.slice(s![ind, ..]));
//        }
//    }
//    let new_mol: Molecule = Molecule::new(
//        atomic_numbers,
//        positions,
//        Some(mol.charge),
//        Some(mol.multiplicity),
//        mol.calculator.r_lr,
//        mol.calculator.active_orbitals,
//        config.clone(),
//    );
//    return new_mol;
//}

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
