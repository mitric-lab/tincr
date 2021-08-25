
use log::info;
use ndarray::prelude::*;
use crate::fmo::Monomer;
use crate::constants::HARTREE_TO_EV;
use nalgebra::Vector3;


pub fn print_states(fragment: &Monomer, n_roots: usize) {
    let threshold: f64 = 0.1;
    let total_energy: f64 = fragment.properties.last_energy().unwrap();
    let f: ArrayView1<f64> = fragment.properties.oscillator_strengths().unwrap();
    info!("{:^80}", "");
    info!("{: ^80}", "TDA Excitation Energies");
    info!("{:-^75}", "");
    for n in 0..n_roots {
        let exc_energy: f64 = fragment.properties.ci_eigenvalue(n).unwrap();
        let energy: f64 = total_energy + exc_energy;
        let tdm: ArrayView2<f64> = fragment.properties.tdm(n).unwrap();
        let tr_dipole: Vector3<f64> = fragment.properties.tr_dipole(n).unwrap();
        print_excited_state(n + 1, exc_energy, energy, tr_dipole, f[n]);
        print_eigenvalues(tdm, threshold);
        if n < n_roots - 1 {
            info!("{: ^80}", "");
        }
    }
    info!("All transition with amplitudes > {:10.8} were printed.", threshold);
    info!("{:-^75} ", "");
}

fn print_excited_state(index: usize, exc_energy: f64, energy: f64, tr_dip: Vector3<f64>, f: f64) {
    let exc_energy: f64 = exc_energy * HARTREE_TO_EV;
    info!("Excited state {: >5}: Excitation energy = {:>8.6} eV", index, exc_energy);
    info!("Total energy for state {: >5}: {:22.12} Hartree", index, energy);
    info!("  Multiplicity: Singlet");
    info!("  Trans. Mom. (a.u.): {:10.6} X  {:10.6} Y  {:10.6} Z", tr_dip.x, tr_dip.y, tr_dip.z);
    info!("  Oscillator Strength:  {:12.8}", f);
}

fn print_eigenvalues(tdm: ArrayView2<f64>, threshold: f64) {
    for (h, row) in tdm.axis_iter(Axis(0)).rev().enumerate() {
        let occ_label: String = if h == 0 {
            format!("H")
        } else {
            format!("H-{}", h)
        };
        for (l, value) in row.iter().enumerate() {
            let virt_label: String = if l == 0 {
                format!("L")
            } else {
                format!("L+{}", l)
            };
            if value.abs() > threshold {
                info!("  {: <4} --> {: <4}  Amplitude: {:6.4} => {:>4.1} %", occ_label, virt_label, value.abs(), value.powi(2) * 1e2);
            }
        }
    }
}