use crate::constants::*;
use crate::molecule::*;
use ndarray::*;
use crate::defaults;


// self.temperature = temperature
// self._distance_matrix()
// self._proximity_matrix()
// self.S, self.H0 = self._constructH0andS()
// if len(self.point_charges) > 0:
// hpc = self._construct_h_point_charges()
// H = self.H0 + hpc*self.S
// else:
// H = self.H0
// self.orbe, self.orbs = eigh(H,self.S)
// self._constructDensityMatrix()
// self.HLgap = self.getHOMO_LUMO_gap()
// # dq should be zero anyway
// self.gamma = self.gm.gamma_atomwise(self.atomlist, self.distances)[0]
//
// self.q = self.q0
// self.dq = zeros(len(self.atomlist))
// self.getEnergies()
// # write_iteration expects these member variables to be set
// self.i = 0
// self.relative_change = 0.0
// self.writeIteration()

// pub fn run_nonscc(molecule: &Molecule) -> f64 {
//     let (s, h0) = get
//
//     return 1.0;
// }

// INCOMPLETE
// pub fn run_scc(
//     molecule: &Molecule,
//     max_iter: Option<usize>,
//     scf_conv: Option<f64>,
//     temperature: Option<f64>,
// ) -> f64 {
//     let max_iter: usize = max_iter.unwrap_or(defaults::MAX_ITER);
//     let scf_conv: f64 = scf_conv.unwrap_or(defaults::SCF_CONV);
//     let temperature: f64 = temperature.unwrap_or(defaults::TEMPERATURE);
//
//     // charge guess
//     let dq: Array1<f64> = Array1::zeros([molecule.n_atoms]);
//     let ddip: Array2<f64> = Array2::zeros([molecule.n_atoms, 3]);
//     let converged: bool = false;
//     let shift_flag: bool = false;
//     let mixing_flag: bool = false;
//     for i in 0..max_iter {
//         let h_coul = 0;
//     }
//     return 1.0;
// }

fn construct_h1(mol : &Molecule, gamma: ArrayView2<f64>, dq: Array1<f64>) -> Array2<f64> {
    let e_stat_pot: Array1<f64> = gamma.dot(&dq);
    let mut h1: Array2<f64> = Array2::zeros([mol.n_orbs, mol.n_orbs]);

    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, (z_i, pos_i)) in mol.iter_atomlist().enumerate() {
        for (n_i, l_i, m_i) in &mol.valorbs[z_i] {
            nu = 0;
            for (j, (z_j, pos_j)) in mol.iter_atomlist().enumerate() {
                for (n_j, l_j, m_j) in &mol.valorbs[z_j] {
                    h1[[mu, nu]] = 0.5 * (e_stat_pot[i] + e_stat_pot[j]);
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h1;
}



#[test]
fn h1_construction() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers, positions, charge, multiplicity);
    let (gm, gm_a0) = get_gamma_matrix(&mol, Some(0.0));
    let dq: Array1<f64> = array![0.4900936727759634, -0.2450466365939161, -0.2450470361820512];
    let h1: Array2<f64> = construct_h1(&mol, gm.view(), dq);
    let h1_ref: Array2<f64> = array![
        [ 0.0296041126328175,  0.0296041126328175,  0.0296041126328175,
          0.0296041126328175,  0.0138472664342115,  0.0138473229910027],
        [ 0.0296041126328175,  0.0296041126328175,  0.0296041126328175,
          0.0296041126328175,  0.0138472664342115,  0.0138473229910027],
        [ 0.0296041126328175,  0.0296041126328175,  0.0296041126328175,
          0.0296041126328175,  0.0138472664342115,  0.0138473229910027],
        [ 0.0296041126328175,  0.0296041126328175,  0.0296041126328175,
          0.0296041126328175,  0.0138472664342115,  0.0138473229910027],
        [ 0.0138472664342115,  0.0138472664342115,  0.0138472664342115,
          0.0138472664342115, -0.0019095797643945, -0.0019095232076034],
        [ 0.0138473229910027,  0.0138473229910027,  0.0138473229910027,
          0.0138473229910027, -0.0019095232076034, -0.0019094666508122]];
    assert!(h1.all_close(&h1_ref, 1e-06));
}