use crate::defaults;
use crate::gradients::{get_gradients, ToOwnedF};
use crate::h0_and_s::h0_and_s;
use crate::internal_coordinates::*;
use crate::io::GeneralConfig;
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
use std::time::Instant;

pub fn fmo_construct_mos(
    n_mo: Vec<usize>,
    h_0_complete: &Array2<f64>,
    n_frags:usize,
    homo_orbs: Vec<Array1<f64>>,
    lumo_orbs: Vec<Array1<f64>>,
)->Array2<f64>{
    let (e_vals,e_vecs):(Array1<f64>,Array2<f64>) = h_0_complete.eigh(UPLO::Upper).unwrap();
    let n_aos:usize = n_mo.iter().cloned().max().unwrap();
    let mut orbs:Array2<f64> = Array2::zeros((n_frags*2,n_frags*n_aos));

    for idx in (0..n_frags).into_iter(){
        let i:usize = 2 * idx;
        let j_1:usize = n_aos * idx;
        let j_2:usize = n_aos * (idx + 1);

        orbs.slice_mut(s![i,j_1..j_2]).assign(&homo_orbs[idx]);
        orbs.slice_mut(s![i+1,j_1..j_2]).assing(&lumo_orbs[idx]);
    }
    let orbs_final:Array2<f64> = e_vecs.t().to_owned().dot(&orbs).t().to_owned();

    return (orbs_final);
}

pub fn fmo_calculate_pairwise(
    mol: &Molecule,
    fragments: &Vec<Molecule>,
    n_mo: Vec<usize>,
    h_0_complete: &Array2<f64>,
    homo_orbs: Vec<Array1<f64>>,
    lumo_orbs: Vec<Array1<f64>>,
) -> Array2<f64> {
    let mut s_complete: Array2<f64> = Array2::zeros(h_0_complete.raw_dim());
    let mut h_0_complete_mut: Array2<f64> = h_0_complete.clone();

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
                Some(mol.charge),
                Some(mol.multiplicity),
                mol.calculator.r_lr,
                mol.calculator.active_orbitals,
                mol.config.clone(),
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

    return h_0_complete_mut;
}

pub fn fmo_calculate_fragments(
    mol: &Molecule,
    fragments: &Vec<Molecule>,
) -> (Array2<f64>, Vec<usize>, Vec<Array1<f64>>, Vec<Array1<f64>>,Vec<usize>,Vec<usize>) {
    let norb_frag: usize = 2;
    let size: usize = norb_frag * fragments.len();

    let mut n_mo: Vec<usize> = Vec::new();
    let mut h_diag: Vec<f64> = Vec::new();
    let mut homo_orbs: Vec<Array1<f64>> = Vec::new();
    let mut lumo_orbs: Vec<Array1<f64>> = Vec::new();
    let mut ind_homo:Vec<usize>= Vec::new();
    let mut ind_lumo:Vec<usize> = Vec::new();

    for (ind, frag) in fragments.iter().enumerate() {
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
        ind_homo.push(homo_ind);
        ind_lumo.push(lumo_ind);
        let e_homo: f64 = orbe[homo_ind];
        let e_lumo: f64 = orbe[lumo_ind];
        let frag_homo_orbs: Array1<f64> = orbs.slice(s![.., homo_ind]).to_owned();
        let frag_lumo_orbs: Array1<f64> = orbs.slice(s![.., lumo_ind]).to_owned();

        h_diag.push(e_homo);
        h_diag.push(e_lumo);
        n_mo.push(orbs.dim().0);
        homo_orbs.push(frag_homo_orbs);
        lumo_orbs.push(frag_lumo_orbs);
    }
    let h_0: Array2<f64> = Array::from_diag(&Array::from(h_diag));

    return (h_0, n_mo, homo_orbs, lumo_orbs,ind_homo,ind_homo);
}

pub fn create_fragment_molecules(mol: &Molecule, config: GeneralConfig) -> Vec<Molecule> {
    let mut fragments: Vec<Molecule> = Vec::new();

    for frag in mol.sub_graphs.iter() {
        let mut atomic_numbers: Vec<u8> = Vec::new();
        let mut positions: Array2<f64> = Array2::zeros((frag.node_count(), 3));

        for (ind, val) in frag.node_indices().enumerate() {
            atomic_numbers.push(mol.atomic_numbers[val.index()]);
            positions
                .slice_mut(s![ind, ..])
                .assign(&mol.positions.slice(s![val.index(), ..]));
        }
        let frag_mol: Molecule = Molecule::new(
            atomic_numbers,
            positions,
            Some(mol.charge),
            Some(mol.multiplicity),
            mol.calculator.r_lr,
            mol.calculator.active_orbitals,
            config.clone(),
        );
        fragments.push(frag_mol);
    }

    return fragments;
}

pub fn reorder_molecule(
    mol: &Molecule,
    fragments: &Vec<Molecule>,
    config: GeneralConfig,
) -> Molecule {
    let mut atomic_numbers: Vec<u8> = Vec::new();
    let mut positions: Array2<f64> = Array2::zeros(mol.positions.raw_dim());

    for molecule in fragments.iter() {
        for (ind, atom) in molecule.atomic_numbers.iter().enumerate() {
            atomic_numbers.push(*atom);
            positions
                .slice_mut(s![ind, ..])
                .assign(&molecule.positions.slice(s![ind, ..]));
        }
    }
    let new_mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        Some(mol.charge),
        Some(mol.multiplicity),
        mol.calculator.r_lr,
        mol.calculator.active_orbitals,
        config.clone(),
    );
    return new_mol;
}
