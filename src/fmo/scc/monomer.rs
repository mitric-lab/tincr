use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray::stack;
use ndarray_stats::{QuantileExt, DeviationExt};
use crate::fmo::scc::helpers::*;
use crate::fmo::{Pair, Monomer, Fragment};
use crate::scc::h0_and_s::*;
use crate::scc::gamma_approximation::*;
use crate::scc::{density_matrix_ref, lc_exact_exchange, density_matrix, get_repulsive_energy, get_electronic_energy};
use crate::scc::mixer::{BroydenMixer, Mixer};
use crate::scc::mulliken::mulliken;
use crate::initialization::Atom;
use crate::io::SccConfig;

impl Fragment for Monomer {}

impl Monomer {
    pub fn prepare_scc(&mut self, atoms: &[Atom]) {
        // get H0 and S
        let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(self.n_orbs, &atoms, &self.slako);
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // and save it in the self properties
        self.data.set_h0(h0);
        self.data.set_s(s);
        self.data.set_x(x);
        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = atoms.iter().map(|atom| atom.number).collect();
        self.data.set_atomic_numbers(atomic_numbers);
        // get the gamma matrix

        let gamma: Array2<f64> = gamma_atomwise(&self.gammafunction, &atoms, self.n_atoms);
        // and save it as a `Property`
        self.data.set_gamma(gamma);

        // calculate the number of electrons
        let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);

        // Set the indices of the occupied and virtual orbitals based on the number of electrons.
        self.set_mo_indices(n_elec);

        // occupation is determined by Aufbau principle and no electronic temperature is considered
        let f: Vec<f64> = (0..self.n_orbs)
            .map(|idx| if idx < n_elec / 2 { 2.0 } else { 0.0 })
            .collect();
        self.data.set_occupation(f);

        // if the system contains a long-range corrected Gammafunction the gamma matrix will be computed
        if self.gammafunction_lc.is_some() {
            let (gamma_lr, gamma_lr_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.data.set_gamma_lr(gamma_lr);
            self.data.set_gamma_lr_ao(gamma_lr_ao);
        }

        self.data.set_mixer(BroydenMixer::new(self.n_orbs));

        // if this is the first SCC calculation the charge differences will be initialized to zeros
        if !self.properties.contains_key("dq") {
            self.data.set_dq(Array1::zeros(self.n_atoms));
            self.data.set_q_ao(Array1::zeros(self.n_orbs));
        }

        // this is also only needed in the first SCC calculation
        if !self.properties.contains_key("ref_density_matrix") {
            self.data
                .set_p_ref(density_matrix_ref(self.n_orbs, &atoms));
        }

        // in the first SCC calculation the density matrix is set to the reference density matrix
        if !self.properties.contains_key("P") {
            self.data
                .set_p(self.data.p_ref().to_owned());
        }
    }

    pub fn scc_step(&mut self, atoms: &[Atom], v_esp: Array2<f64>, config: SccConfig) -> bool {
        let scf_charge_conv: f64 = config.scf_charge_conv;
        let scf_energy_conv: f64 = config.scf_energy_conv;
        let mut dq: Array1<f64> = self.data.take_dq();
        let mut q_ao: Array1<f64> = self.data.take_q_ao();
        let mut mixer: BroydenMixer = self.data.take_mixer();
        let mut p: Array2<f64> = self.data.take_p();
        let x: ArrayView2<f64> = self.data.x();
        let s: ArrayView2<f64> = self.data.s();
        let gamma: ArrayView2<f64> = self.data.gamma();
        let h0: ArrayView2<f64> = self.data.h0();
        let p0: ArrayView2<f64> = self.data.p_ref();
        let last_energy: f64 = self.data.last_energy();
        let f: &[f64] = self.data.occupation();
        // electrostatic interaction between the atoms of the same monomer and all the other atoms
        // the coulomb term and the electrostatic potential term are combined into one:
        // H_mu_nu = H0_mu_nu + HCoul_mu_nu + HESP_mu_nu
        // H_mu_nu = H0_mu_nu + 1/2 S_mu_nu sum_k sum_c_on_k (gamma_ac + gamma_bc) dq_c
        let h_coul: Array2<f64> = v_esp * &s * 0.5;
        let mut h: Array2<f64> = h_coul + h0;
        if self.gammafunction_lc.is_some() {
            let dp: Array2<f64> = &p - &p0;
            let h_x: Array2<f64> =
                lc_exact_exchange(s, self.data.gamma_lr_ao(), dp.view());
            h = h + h_x;
        }
        let mut h_save:Array2<f64> = h.clone();

        // H' = X^t.H.X
        h = x.t().dot(&h).dot(&x);
        let tmp: (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Upper).unwrap();
        let orbe: Array1<f64> = tmp.0;
        // C = X.C'
        let orbs: Array2<f64> = x.dot(&tmp.1);

        // calculate the density matrix
        p = density_matrix(orbs.view(), &f[..]);

        // New Mulliken charges for each atomic orbital.
        let q_ao_n: Array1<f64> = s.dot(&p).diag().to_owned();

        // Charge difference to previous iteration
        let delta_dq: Array1<f64> = &q_ao_n - &q_ao;

        let diff_dq_max: f64 = q_ao.root_mean_sq_err(&q_ao_n).unwrap();

        // Broyden mixing of Mulliken charges.
        q_ao = mixer.next(q_ao, delta_dq);

        // The density matrix is updated in accordance with the Mulliken charges.
        p = p * &(&q_ao / &q_ao_n);
        let dp: Array2<f64> = &p - &p0;
        dq = mulliken(dp.view(), s.view(), &atoms);

        // compute electronic energy
        let scf_energy = get_electronic_energy(
            p.view(),
            p0.view(),
            s.view(),
            h0.view(),
            dq.view(),
            gamma.view(),
            self.data.gamma_lr_ao(),
        );

        // check if charge difference to the previous iteration is lower than threshold
        let conv_charge: bool = diff_dq_max < scf_charge_conv;
        // same check for the electronic energy
        let conv_energy: bool = (last_energy - scf_energy).abs() < scf_energy_conv;

        self.data.set_orbs(orbs);
        self.data.set_orbe(orbe);
        self.data.set_p(p);
        self.data.set_dq(dq);
        self.data.set_mixer(mixer);
        self.data.set_last_energy(scf_energy);
        self.data.set_q_ao(q_ao);
        self.data.set_h_coul_x(h_save);

        // scc (for one fragment) is converged if both criteria are passed
        conv_charge && conv_energy
    }
}
