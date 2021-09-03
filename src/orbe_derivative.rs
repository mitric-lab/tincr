use ndarray::prelude::*;
use crate::molecule::Molecule;
use crate::scc_routine;

pub fn numerical_orbe_gradients(molecule:&mut Molecule)->Array1<f64>{
    let positions:Array2<f64> = molecule.positions.clone();
    let mut gradient:Array1<f64> = Array1::zeros(positions.dim().0*3);
    let h:f64 = 1.0e-4;

    for (ind,coord) in positions.iter().enumerate(){
        let mut ei:Array1<f64> = Array1::zeros(positions.dim().0*3);
        ei[ind] = 1.0;
        let ei:Array2<f64> = ei.into_shape(positions.raw_dim()).unwrap();
        let positions_1:Array2<f64> = &positions + &(h *&ei);
        let positions_2:Array2<f64> = &positions + &(-h *&ei);

        molecule.update_geometry(positions_1);
        let (e_gs, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(molecule);
        molecule.calculator.set_active_orbitals(f.to_vec());
        let full_occ: Vec<usize> = molecule.calculator.full_occ.clone().unwrap();
        let homo_ind:usize = full_occ[full_occ.len()-1];
        let energy_1:f64 = orbe[homo_ind];

        molecule.update_geometry(positions_2);
        let (e_gs_2, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
            scc_routine::run_scc(molecule);
        molecule.calculator.set_active_orbitals(f.to_vec());
        let full_occ: Vec<usize> = molecule.calculator.full_occ.clone().unwrap();
        let homo_ind:usize = full_occ[full_occ.len()-1];
        let energy_2:f64 = orbe[homo_ind];

        let grad_temp:f64 = (energy_1 - energy_2)/(2.0*h);
        gradient[ind] = grad_temp;
    }
    return gradient;
}