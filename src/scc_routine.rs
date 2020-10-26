use crate::constants::*;
use crate::molecule::*;
use ndarray::*;

pub fn run_scc(
    molecule: &Molecule,
    max_iter: Option<usize>,
    scf_conv: Option<f64>,
    temperature: Option<f64>,
) -> f64 {
    let max_iter: usize = max_iter.unwrap_or(DEFAULT_MAX_ITER);
    let scf_conv: f64 = scf_conv.unwrap_or(DEFAULT_SCF_CONV);
    let temperature: f64 = temperature.unwrap_or(DEFAULT_TEMPERATURE);

    // charge guess
    let dq: Array1<f64> = Array1::zeros([molecule.n_atoms]);
    let ddip: Array2<f64> = Array2::zeros([molecule.n_atoms, 3]);
    let converged: bool = false;
    let shift_flag: bool = false;
    let mixing_flag: bool = false;
    for i in 0..max_iter {
        let h_coul = 0;
    }
    return 1.0;
}

fn construct_h1(molecule: &Molecule, gamma: ArrayView2<f64>, dq: ArrayView1<f64>) -> Array2<f64> {
    let e_stat_pot: Array2<f64> = gamma.dot(&dq);
    let mut h1: Array2<f64> = Array2::zeros([molecule.n_orbs, molecule.n_orbs]);

    let mu: usize = 0;
    let mu: usize = 0;
    for (i, (z_i, pos_i)) in mol.iter_atomlist().enumerate() {
        for (n_i, l_i, m_i) in &mol.valorbs[z_i] {
            let nu: usize = 0;
            for (j, (z_j, pos_j)) in mol.iter_atomlist().enumerate() {
                for (n_j, l_j, m_j) in &mol.valorbs[z_j] {
                    h1[[mu, nu]] = 0.5 * (e_stat_pot[i] + e_stat_pot[j]);
                    let nu = nu + 1;
                }
            }
            let mu = mu + 1;
        }
    }
    return h1;
    //h_coul =
}
