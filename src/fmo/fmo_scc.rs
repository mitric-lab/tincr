use crate::fmo::{Monomer, SuperSystem};
use crate::scc::gamma_approximation::{gamma_ao_wise, gamma_atomwise};
use crate::scc::h0_and_s::h0_and_s;
use crate::scc::mixer::{BroydenMixer, Mixer};
use crate::scc::scc_routine::{RestrictedSCC, SCCError};
use crate::scc::{construct_h1, density_matrix, density_matrix_ref, get_repulsive_energy, lc_exact_exchange, get_electronic_energy};
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use ndarray_stats::QuantileExt;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use crate::scc::mulliken::mulliken;


impl RestrictedSCC for SuperSystem {
    ///  To run the SCC calculation of the FMO [SuperSystem] the following properties need to be set:
    /// For each [Monomer]
    /// - H0
    /// - S: overlap matrix in AO basis
    /// - Gamma matrix (and long-range corrected Gamma matrix if we use LRC)
    /// - If there are no charge differences, `dq`, from a previous calculation
    ///  they are initialized to zeros
    /// - the density matrix and reference density matrix
    fn prepare_scc(&mut self) {
        // prepare all individual monomers
        self.monomers
            .par_iter_mut()
            .for_each(|molecule: &mut Monomer| {
                molecule.prepare_scc();
            });
        println!("FINISHED");
    }

    fn run_scc(&mut self) -> Result<f64, SCCError> {
        // SCC settings from the user input
        let temperature: f64 = self.config.scf.electronic_temperature;
        let max_iter: usize = self.config.scf.scf_max_cycles;
        let mut converged: Vec<bool> = vec![false; self.n_mol];
        'scf_loop: for iter in 0..max_iter {
            for (i, mol) in self.monomers.iter_mut().enumerate() {
                let mut tmp: bool = false;
                if !converged[i] {
                    //tmp = mol.scc_step();
                } else {
                    tmp = true;
                }
                converged[i] = tmp;
            }
            if !converged.contains(&false) {
                break 'scf_loop;
            }
            println!("iter {}", iter + 1);
        }
        println!("SCC");
        Ok(0.0)
    }
}

impl Monomer {
    pub fn prepare_scc(&mut self) {
        // get H0 and S
        let (s, h0): (Array2<f64>, Array2<f64>) =
            h0_and_s(self.n_orbs, &self.atoms, &self.geometry, &self.slako);
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // and save it in the self properties
        self.properties.set_h0(h0);
        self.properties.set_s(s);
        self.properties.set_x(x);
        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = self.atoms.iter().map(|atom| atom.number).collect();
        self.properties.set_atomic_numbers(atomic_numbers);
        // get the gamma matrix
        let gamma: Array2<f64> = gamma_atomwise(&self.gammafunction, &self.atoms, self.n_atoms);
        // and save it as a `Property`
        self.properties.set_gamma(gamma);
        // occupation is determined by Aufbau principle and no electronic temperature is considered
        let f: Vec<f64> = (0..self.n_orbs).map(|idx| if idx < self.n_elec/2 {2.0} else {0.0}).collect();
        self.properties.set_occupation(f);

        // if the system contains a long-range corrected Gammafunction the gamma matrix will be computed
        if self.gammafunction_lc.is_some() {
            let (gamma_lr, gamma_lr_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &self.atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.properties.set_gamma_lr(gamma_lr);
            self.properties.set_gamma_lr_ao(gamma_lr_ao);
        }

        // if this is the first SCC calculation the charge differences will be initialized to zeros
        if !self.properties.contains_key("dq") {
            self.properties.set_dq(Array1::zeros(self.n_atoms));
        }

        self.properties.set_mixer(BroydenMixer::new(self.n_atoms));

        // this is also only needed in the first SCC calculation
        if !self.properties.contains_key("ref_density_matrix") {
            self.properties
                .set_p_ref(density_matrix_ref(self.n_orbs, &self.atoms));
        }

        // in the first SCC calculation the density matrix is set to the reference density matrix
        if !self.properties.contains_key("P") {
            self.properties
                .set_p(self.properties.p_ref().unwrap().to_owned());
        }
    }

    //pub fn scc_step(&mut self) -> bool {
    pub fn scc_step(&mut self, gamma_x: ArrayView2<f64>, gamma_xk: ArrayView2<f64>, super_dq: &[ArrayView1<f64>]) -> bool {
        let scf_charge_conv: f64 = self.config.scf.scf_charge_conv;
        let scf_energy_conv: f64 = self.config.scf.scf_energy_conv;
        let mut dq: Array1<f64> = self.properties.take_dq().unwrap();
        let mut mixer: BroydenMixer = self.properties.take_mixer().unwrap();
        let x: ArrayView2<f64> = self.properties.x().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let p0: ArrayView2<f64> = self.properties.p_ref().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let mut last_energy: f64 = self.properties.last_energy().unwrap();
        let f: &[f64] = self.properties.occupation().unwrap();

        let h_coul: Array2<f64> = construct_h1(self.n_orbs, &self.atoms, gamma, dq.view()) * &s;
        let mut h: Array2<f64> = h_coul + h0;
        if self.gammafunction_lc.is_some() {
            let h_x: Array2<f64> =
                lc_exact_exchange(s, self.properties.gamma_lr_ao().unwrap(), p0, p);
            h = h + h_x;
        }

        // H' = X^t.H.X
        h = x.t().dot(&h).dot(&x);
        let tmp: (Array1<f64>, Array2<f64>) = h.eigh(UPLO::Upper).unwrap();
        let orbe: Array1<f64> = tmp.0;
        // C = X.C'
        let orbs: Array2<f64> = x.dot(&tmp.1);

        // calculate the density matrix
        let p: Array2<f64> = density_matrix(orbs.view(), &f[..]);

        // update partial charges using Mulliken analysis
        let (new_q, new_dq): (Array1<f64>, Array1<f64>) =
            mulliken(p.view(), p0.view(), s.view(), &self.atoms, self.n_atoms);

        // charge difference to previous iteration
        let delta_dq: Array1<f64> = &new_dq - &dq;

        let delta_dq_max: f64 = *delta_dq.map(|x| x.abs()).max().unwrap();

        // Broyden mixing of partial charges # changed new_dq to dq
        dq = mixer.next(dq, delta_dq);
        let q: Array1<f64> = new_q;

        // compute electronic energy
        let scf_energy = get_electronic_energy(
            p.view(),
            p0.view(),
            s.view(),
            h0.view(),
            dq.view(),
            gamma.view(),
            self.properties.gamma_lr_ao(),
        );

        // check if charge difference to the previous iteration is lower than 1e-5
        let converged: bool = if (delta_dq_max < scf_charge_conv)
            && (last_energy - scf_energy).abs() < scf_energy_conv
        {
            true
        } else {
            false
        };

        self.properties.set_dq(dq);
        self.properties.set_mixer(mixer);
        self.properties.set_last_energy(scf_energy);
        return converged;
    }
}
