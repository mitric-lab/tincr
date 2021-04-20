use crate::fmo::{Monomer, SuperSystem, Pair};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_ao_wise, gamma_atomwise, gamma_atomwise_ab};
use crate::scc::h0_and_s::{h0_and_s, h0_and_s_ab};
use crate::scc::mixer::{BroydenMixer, Mixer};
use crate::scc::mulliken::mulliken;
use crate::scc::scc_routine::{RestrictedSCC, SCCError};
use crate::scc::{
    construct_h1, density_matrix, density_matrix_ref, get_electronic_energy, get_repulsive_energy,
    lc_exact_exchange,
};
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::prelude::*;
use ndarray::stack;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use ndarray_stats::QuantileExt;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;

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
        // Vector that holds the information if the scc calculation of the individual monomer
        // is converged or not
        let mut converged: Vec<bool> = vec![false; self.n_mol];
        // charge differences of all atoms. these are needed to compute the electrostatic potential
        // that acts on the monomers.
        let mut dq: Array1<f64> = Array1::zeros([self.atoms.len()]);
        // charge consistent loop for the monomers
        'scf_loop: for iter in 0..max_iter {
            // the matrix vector product of the gamma matrix for all atoms and the charge diffrences
            // yields the electrostatic potential for all atoms. this is then converted into ao basis
            // and given to each monomer scc step
            let esp_at: Array1<f64> = self.properties.gamma().unwrap().dot(&dq);
            for (i, mol) in self.monomers.iter_mut().enumerate() {
                let v_esp: Array2<f64> =
                    atomvec_to_aomat(esp_at.slice(s![mol.atom_slice]), mol.n_orbs, &mol.atoms);
                converged[i] = mol.scc_step(v_esp);
                // save the dq's from the monomer calculation
                dq.slice_mut(s![mol.atom_slice])
                    .assign(&mol.properties.dq().unwrap());
            }
            // the loop ends if all monomers are converged
            if !converged.contains(&false) {
                break 'scf_loop;
            }
            println!("Iteration");
        }
        // this is the electrostatic potential that acts on the pairs
        // PARALLEL: The dot product could be parallelized and then it is not necessary to convert
        // the ArrayView into an owned ArrayBase
        let esp_at: Array1<f64> = self.properties.gamma().unwrap().dot(&dq);
        for mol in self.monomers.iter_mut() {
            mol.properties.set_esp_q(esp_at.slice(s![mol.atom_slice]).to_owned());
        }
        for pair in self.pairs.iter_mut() {
            pair.prepare_scc(&self.monomers[pair.i], &self.monomers[pair.j]);
            pair.run_scc();

        }

        // scc loop for each dimer that is not labeled as ESD pair
        //for pair in self.pairs
        println!("SCC");
        Ok(0.0)
    }
}

impl Monomer {
    pub fn prepare_scc(&mut self) {
        // get H0 and S
        let (s, h0): (Array2<f64>, Array2<f64>) =
            h0_and_s(self.n_orbs, &self.atoms,&self.slako);
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
        let f: Vec<f64> = (0..self.n_orbs)
            .map(|idx| if idx < self.n_elec / 2 { 2.0 } else { 0.0 })
            .collect();
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
    pub fn scc_step(&mut self, v_esp: Array2<f64>) -> bool {
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

        // electrostatic interaction between the atoms of the same monomer and all the other atoms
        // the coulomb term and the electrostatic potential term are combined into one:
        // H_mu_nu = H0_mu_nu + HCoul_mu_nu + HESP_mu_nu
        // H_mu_nu = H0_mu_nu + 1/2 S_mu_nu sum_k sum_c_on_k (gamma_ac + gamma_bc) dq_c
        let h_coul: Array2<f64> = v_esp * &s * 0.5;
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

impl Pair {
    fn prepare_scc(&mut self, m1: &Monomer, m2: &Monomer) {
        // get H0 and S outer diagonal block
        let (s_ab, h0_ab): (Array2<f64>, Array2<f64>) =
            h0_and_s_ab(m1.n_orbs, m2.n_orbs, &m1.atoms, &m2.atoms, &m1.slako);
        let mut s: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);
        let mut h0: Array2<f64> = s.clone();

        let mu: usize = m1.n_orbs;
        let a: usize = m1.n_atoms;

        s.slice_mut(s![0..mu, 0..mu]).assign(&m1.properties.s().unwrap());
        s.slice_mut(s![mu.., mu..]).assign(&m2.properties.s().unwrap());
        s.slice_mut(s![0..mu, mu..]).assign(&s_ab);
        s.slice_mut(s![mu.., 0..mu]).assign(&s_ab.t());

        h0.slice_mut(s![0..mu, 0..mu]).assign(&m1.properties.h0().unwrap());
        h0.slice_mut(s![mu.., mu..]).assign(&m2.properties.h0().unwrap());
        h0.slice_mut(s![0..mu, mu..]).assign(&h0_ab);
        h0.slice_mut(s![mu.., 0..mu]).assign(&h0_ab.t());

        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // get the gamma matrix
        let gamma_ab: Array2<f64> = gamma_atomwise_ab(&self.gammafunction, &m1.atoms, &m2.atoms, m1.n_atoms, m2.n_atoms);
        let mut gamma: Array2<f64> = Array2::zeros([self.n_atoms, self.n_atoms]);

        gamma.slice_mut(s![0..a, 0..a]).assign(&m1.properties.gamma().unwrap());
        gamma.slice_mut(s![a.., a..]).assign(&m2.properties.gamma().unwrap());
        gamma.slice_mut(s![0..a, a..]).assign(&gamma_ab);
        gamma.slice_mut(s![a.., 0..a]).assign(&gamma_ab.t());

        // get electrostatic potential that acts on the pair. This is based on Eq. 44 from the book
        // chapter "The FMO-DFTB Method" by Yoshio Nishimoto and Stephan Irle on page 474 in
        // Recent Advances of the Fragment Molecular Orbital Method
        // See: https://www.springer.com/gp/book/9789811592348
        let mut esp: Array1<f64> = Array1::zeros([self.n_atoms]);
        esp.slice_mut(s![0..a]).assign(&(&m1.properties.esp_q().unwrap() - &(gamma_ab.dot(&m2.properties.dq().unwrap()))));
        esp.slice_mut(s![a..]).assign(&(&m2.properties.esp_q().unwrap() - &(gamma_ab.t().dot(&m1.properties.dq().unwrap()))));
        // and convert it into a matrix in AO basis
        let omega: Array2<f64> = atomvec_to_aomat(esp.view(), self.n_orbs, &self.atoms);
        self.properties.set_v(omega*&s*0.5);

        // and save it in the self properties
        self.properties.set_h0(h0);
        self.properties.set_s(s);
        self.properties.set_x(x);
        self.properties.set_gamma(gamma);

        // save the atomic numbers since we need them multiple times
        let atomic_numbers: Vec<u8> = self.atoms.iter().map(|atom| atom.number).collect();
        self.properties.set_atomic_numbers(atomic_numbers);

        // occupation is determined by Aufbau principle and no electronic temperature is considered
        let f: Vec<f64> = (0..self.n_orbs)
            .map(|idx| if idx < self.n_elec / 2 { 2.0 } else { 0.0 })
            .collect();
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
            self.properties.set_dq(stack![Axis(0), m1.properties.dq().unwrap(), m2.properties.dq().unwrap()]);
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

    fn run_scc(&mut self) {
        let scf_charge_conv: f64 = self.config.scf.scf_charge_conv;
        let scf_energy_conv: f64 = self.config.scf.scf_energy_conv;
        let max_iter: usize = 10;//self.config.scf.scf_max_cycles;
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
        let v: ArrayView2<f64> = self.properties.v().unwrap();

        let h_esp: Array2<f64> = &h0 + &v;

        'scf_loop: for iter in 0..max_iter {
            let h_coul: Array2<f64> = atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &self.atoms) * &s * 0.5;
            let mut h: Array2<f64> = h_coul + &h_esp;
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

            if converged {
                self.properties.set_last_energy(scf_energy);
                break 'scf_loop;
            }
        }
        self.properties.reset();
        self.properties.set_dq(dq);
        //return converged;
    }
}

fn atomvec_to_aomat(esp_atomwise: ArrayView1<f64>, n_orbs: usize, atoms: &[Atom]) -> Array2<f64> {
    let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
    let mut mu: usize = 0;
    for (atom, esp_at) in atoms.iter().zip(esp_atomwise.iter()) {
        for _ in 0..atom.n_orbs {
            esp_ao_row[mu] = *esp_at;
            mu = mu + 1;
        }
    }
    let esp_ao_column: Array2<f64> = esp_ao_row.clone().insert_axis(Axis(1));
    let esp_ao: Array2<f64> = &esp_ao_column.broadcast((n_orbs, n_orbs)).unwrap() + &esp_ao_row;
    return esp_ao;
}
