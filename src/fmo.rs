use crate::defaults;
use crate::gradients::{get_gradients, ToOwnedF};
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

pub fn fmo_calculate_fragments(mol:&Molecule, fragments:&Vec<Molecule>){
    let norb_frag:usize = 2;
    let size:usize = norb_frag * fragments.len();

    let mut n_mo:Vec<usize> = Vec::new();
    let mut h_diag:Vec<f64> = Vec::new();
    let mut homo_orbs:Vec<Array1<f64>> = Vec::new();
    let mut lumo_orbs:Vec<Array1<f64>> = Vec::new();

    for (ind, frag) in fragments.iter().enumerate(){
        let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(frag);
        let f_temp:Array1<f64> = Array::from(f);
        let occ_indices: Array1<usize> = f_temp
            .indexed_iter()
            .filter_map(|(index, &item)| if item > 0.1 { Some(index) } else { None })
            .collect();
        let virt_indices: Array1<usize> = f_temp
            .indexed_iter()
            .filter_map(|(index, &item)| if item <= 0.1 { Some(index) } else { None })
            .collect();
        let homo_ind:usize = occ_indices[occ_indices.len()-1];
        let lumo_ind:usize = virt_indices[0];
        let e_homo:f64 = orbe[homo_ind];
        let e_lumo:f64 = orbe[lumo_ind];
        let frag_homo_orbs:Array1<f64> = orbs.slice(s![..,homo_ind]).to_owned();
        let frag_lumo_orbs:Array1<f64> = orbs.slice(s![..,lumo_ind]).to_owned();

        h_diag.push(e_homo);
        h_diag.push(e_lumo);
        n_mo.push(orbs.dim().0);
        homo_orbs.push(frag_homo_orbs);
        lumo_orbs.push(frag_lumo_orbs);
    }
    let h_0:Array2<f64> = Array::from_diag(&Array::from(h_diag));
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
