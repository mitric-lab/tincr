
use log::info;
use ndarray::prelude::*;
use crate::fmo::{Monomer, BasisState};
use crate::constants::HARTREE_TO_EV;
use nalgebra::Vector3;
use num_traits::identities::Zero;
use std::ops::Mul;

pub fn print_lcmo_states(energies: ArrayView1<f64>, eigvecs: ArrayView2<f64>, basis: &[BasisState], e_tot: f64) {
    let threshold: f64 = 0.1;
    info!("{:^80}", "");
    info!("{: ^80}", "FMO-LCMO Excitation Energies");
    info!("{:-^75}", "");
    for (n, (e, v)) in energies.iter().zip(eigvecs.axis_iter(Axis(1))).enumerate() {
        let energy: f64 = e_tot + e;
        print_lcmo_state(n + 1, *e, energy, v, &basis);
        print_states(v.view(), &basis, threshold);
        if n < energies.len() - 1 {
            info!("{: ^80}", "");
        }
    }
    info!("All transition with amplitudes > {:10.8} were printed.", threshold);
    info!("{:-^75} ", "");
}

fn print_lcmo_state(index: usize, exc_energy: f64, energy: f64, eigvec: ArrayView1<f64>, basis: &[BasisState]) {
    let mut tr_dip: Vector3<f64> = Vector3::zero();
    for (i, v) in eigvec.iter().enumerate() {
        match basis.get(i).unwrap() {
            BasisState::LE(state) => {tr_dip += state.tr_dipole.scale(*v);}
            BasisState::CT(_) => {}
        }
    }
    let f: f64 = 2.0 / 3.0 * exc_energy * tr_dip.dot(&tr_dip);
    let exc_energy: f64 = exc_energy * HARTREE_TO_EV;
    info!("Excited state {: >5}: Excitation energy = {:>8.6} eV", index, exc_energy);
    info!("Total energy for state {: >5}: {:22.12} Hartree", index, energy);
    info!("  Multiplicity: Singlet");
    info!("  Trans. Mom. (a.u.): {:10.6} X  {:10.6} Y  {:10.6} Z", tr_dip.x, tr_dip.y, tr_dip.z);
    info!("  Oscillator Strength:  {:12.8}", f);
}

fn print_states(eigvec: ArrayView1<f64>, basis: &[BasisState], threshold: f64) {
    for (i, v) in eigvec.iter().enumerate() {
        if v.abs() > threshold {
            info!("  {}  Amplitude: {:6.4} => {:>4.1} %", basis.get(i).unwrap(), v.abs(), v.powi(2) * 1e2);
        }
    }
}