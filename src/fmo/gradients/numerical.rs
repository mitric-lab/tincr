//! These function do not provide numerical gradients. The intention for this file is to create
//! wrapper functions of the kind: `Fn(Array1<f64>) -> f64`, that take the coordinates of a molecule
//! and returns a part of the FMO energy (e.g. monomer energy, pair energy, embedding energy...). This
//! should allow the use of functions for the generation of numerical gradients (using the
//! Ridder's method as implemented in [ridders_method](crate::gradients::numerical::ridders_method)).
//! In this way the analytic gradients can be tested.

use crate::fmo::SuperSystem;
use ndarray::prelude::*;
use crate::scc::scc_routine::RestrictedSCC;
use crate::gradients::assert_deriv;
use crate::initialization::Atom;


impl SuperSystem {
    pub fn monomer_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (energy, _dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        energy
    }

    pub fn pair_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let energy: f64 = self.pair_scc(dq.view());
        energy
    }

    pub fn embedding_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let _energy: f64 = self.pair_scc(dq.view());
        self.properties.set_dq(dq);
        self.embedding_energy()
    }

    pub fn esd_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        self.monomer_scc(20);
        //println!("{}", self.properties.gamma().unwrap());
        self.esd_pair_energy()
    }

    pub fn total_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (monomer_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let pair_energy: f64 = self.pair_scc(dq.view());
        self.properties.set_dq(dq);
        let emb_energy:f64 = self.embedding_energy();
        let esd_energy: f64 = self.esd_pair_energy();
        monomer_energy + pair_energy + esd_energy +emb_energy
    }

    pub fn test_monomer_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let _energy: f64 = self.pair_scc(dq.view());

        self.properties.set_dq(dq);

        assert_deriv(self, SuperSystem::monomer_energy_wrapper, SuperSystem::monomer_gradients, self.get_xyz(), 0.001, 1e-6);
    }

    pub fn test_pair_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let _energy: f64 = self.pair_scc(dq.view());

        self.properties.set_dq(dq);

        assert_deriv(self, SuperSystem::pair_energy_wrapper, SuperSystem::pair_gradients_for_testing, self.get_xyz(), 0.001, 1e-6);
    }

    pub fn test_embedding_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let _energy: f64 = self.pair_scc(dq.view());

        self.properties.set_dq(dq);
        let m_gradients: Array1<f64> = self.monomer_gradients();
        self.pair_gradients(m_gradients.view());

        assert_deriv(self, SuperSystem::embedding_energy_wrapper, SuperSystem::embedding_gradient, self.get_xyz(), 0.01, 1e-6);
    }

    pub  fn test_esd_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        self.pair_scc(dq.view());
        let atoms: &[Atom] = &self.atoms[..];
        for mol in self.monomers.iter_mut() {
            let q_vo: Array2<f64> = mol.compute_q_vo(&atoms[mol.slice.atom_as_range()], None);
            mol.properties.set_q_vo(q_vo);
        }
        println!("ESD ENERGY {}", self.esd_pair_energy());
        self.properties.set_dq(dq);
        self.self_consistent_z_vector(1e-10);
        let m_gradients: Array1<f64> = self.monomer_gradients();

        assert_deriv(self, SuperSystem::esd_energy_wrapper, SuperSystem::es_dimer_gradient, self.get_xyz(), 0.01, 1e-6);
    }

    pub  fn test_total_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (monomer_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let pair_energy: f64 = self.pair_scc(dq.view());
        self.properties.set_dq(dq);
        let emb_energy:f64 = self.embedding_energy();
        let esd_energy: f64 = self.esd_pair_energy();
        println!("FMO MONOMER {}", monomer_energy);
        println!("FMO PAIR {}", pair_energy);
        println!("FMO ESD {}", esd_energy);
        println!("FMO EMB {}", emb_energy);
        println!("FMO ENERGY WITHOUT EMBEDDING {}", monomer_energy + pair_energy + esd_energy);
        println!("FMO ENERGY {}", monomer_energy + pair_energy + emb_energy + esd_energy);
        assert_deriv(self, SuperSystem::total_energy_wrapper, SuperSystem::ground_state_gradient, self.get_xyz(), 0.01, 1e-6);
    }

}

