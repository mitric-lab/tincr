//! These function do not provide numerical gradients. The intention for this file is to create
//! wrapper functions of the kind: `Fn(Array1<f64>) -> f64`, that take the coordinates of a molecule
//! and returns a part of the FMO energy (e.g. monomer energy, pair energy, embedding energy...). This
//! should allow the use of functions for the generation of numerical gradients (using the
//! Ridder's method as implemented in [ridders_method](crate::gradients::numerical::ridders_method)).
//! In this way the analytic gradients can be tested.

use crate::fmo::{Monomer, SuperSystem, ExcitedStateMonomerGradient, GroundStateGradient, PairChargeTransfer, PairType, ChargeTransferPair};
use ndarray::prelude::*;
use crate::scc::scc_routine::RestrictedSCC;
use crate::gradients::assert_deriv;
use crate::initialization::Atom;
use crate::constants::BOHR_TO_ANGS;
use std::net::UdpSocket;
use std::time::Instant;
use crate::properties::Properties;

impl SuperSystem {
    pub fn monomer_orbital_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64{
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

        // get homo orbital energy of monomer 0
        let mol = &self.monomers[0];
        let orbe = mol.properties.orbe().unwrap();
        let virtual_indices = mol.properties.virt_indices().unwrap();
        let orbe_homo:f64 = orbe[virtual_indices[0]-1];
        return orbe_homo;
    }

    pub fn test_monomer_orbital_energy_gradient(&mut self)->Array1<f64>{
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

        let mol = &mut self.monomers[0];
        let atoms = &self.atoms[mol.slice.atom_as_range()];
        mol.prepare_excited_gradient(atoms);
        let grad = mol.calculate_ct_fock_gradient(atoms,0,true);
        mol.properties.reset();

        return grad.0;
    }

    pub fn test_orbital_energy_derivative(&mut self){
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

        assert_deriv(self, SuperSystem::monomer_orbital_energy_wrapper, SuperSystem::test_monomer_orbital_energy_gradient, self.get_xyz(), 0.0001, 1e-6);
    }

    pub fn fmo_ct_energy_wrapper(&mut self,geometry: Array1<f64>)->f64{
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        self.run_scc();

        let monomer_index_i:usize = 0;
        let monomer_index_j:usize = 1;
        let val:f64 = self.exciton_ct_energy(monomer_index_i,monomer_index_j,0,0,true);
        // let val = self.exciton_hamiltonian_ct_test();
        return val;
    }

    pub fn fmo_ct_gradient_wrapper(&mut self)->Array1<f64>{
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();
        for mol in self.monomers.iter_mut() {
            mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
        }
        let monomer_index_i:usize = 0;
        let monomer_index_j:usize = 1;
        let timer: Instant = Instant::now();
        // calculate the gradient of the charge-transfer energy
        let ct_energy = self.exciton_ct_energy(monomer_index_i,monomer_index_j,0,0,true);
        // let ct_energy = self.exciton_hamiltonian_ct_test();
        let mut grad:Array1<f64> = self.ct_gradient_new(monomer_index_i,monomer_index_j,0,0,ct_energy,true);
        // let grad = self.ct_gradient(monomer_index_i,monomer_index_j,0,0);
        println!("Elapsed time ct energy gradient: {:>8.6}",timer.elapsed().as_secs_f64());
        drop(timer);
        println!("Start CPHF");
        let timer: Instant = Instant::now();
        grad = grad + self.calculate_cphf_correction(monomer_index_i,monomer_index_j,0,0,true);
        println!("Elapsed time CPHF correction: {:>8.6}",timer.elapsed().as_secs_f64());
        drop(timer);
        // grad = grad + self.ct_gradient(monomer_index_i,monomer_index_j,0,0);
        let mol_i = &self.monomers[monomer_index_i];
        let mol_j = &self.monomers[monomer_index_j];
        let mut full_gradient:Array1<f64> = Array1::zeros(self.atoms.len()*3);
        full_gradient.slice_mut(s![mol_i.slice.grad]).assign(&grad.slice(s![..mol_i.n_atoms*3]));
        full_gradient.slice_mut(s![mol_j.slice.grad]).assign(&grad.slice(s![mol_i.n_atoms*3..]));
        // let grad:Array1<f64> = self.ct_gradient(0,1,0,0);
        return  full_gradient;
    }

    pub fn test_ct_gradient(&mut self){
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        assert_deriv(self, SuperSystem::fmo_ct_energy_wrapper, SuperSystem::fmo_ct_gradient_wrapper, self.get_xyz(), 0.01, 1e-6);
    }

    pub fn new_fmo_ct_energy_wrapper(&mut self,geometry: Array1<f64>)->f64{
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for pair in self.esd_pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        self.run_scc();
        for mol in self.monomers.iter_mut() {
            mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
        }
        let monomer_index_i:usize = 0;
        let monomer_index_j:usize = 1;
        let m_h:&Monomer = &self.monomers[monomer_index_i];
        let m_l:&Monomer = &self.monomers[monomer_index_j];
        let type_ij: PairType = self.properties.type_of_pair(monomer_index_i, monomer_index_j);

        // create CT states
        let mut state_1 = PairChargeTransfer{
            m_h:m_h,
            m_l:m_l,
            pair_type:type_ij,
            properties:Properties::new(),
        };
        // prepare the TDA calculation of both states
        state_1.prepare_ct_tda(
            self.properties.gamma().unwrap(),
            self.properties.gamma_lr().unwrap(),
            self.properties.s().unwrap(),
            &self.atoms
        );
        state_1.run_ct_tda(&self.atoms,10,150,1.0e-4);

        let val:f64 = state_1.properties.ci_eigenvalue(0).unwrap();
        // let val = self.exciton_hamiltonian_ct_test();
        return val;
    }

    pub fn new_ct_gradient_wrapper(&mut self)->Array1<f64>{
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for pair in self.esd_pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();
        for mol in self.monomers.iter_mut() {
            mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
        }
        let monomer_index_i:usize = 0;
        let monomer_index_j:usize = 1;

        let m_h:&Monomer = &self.monomers[monomer_index_i];
        let m_l:&Monomer = &self.monomers[monomer_index_j];
        let type_ij: PairType = self.properties.type_of_pair(monomer_index_i, monomer_index_j);

        // create CT states
        let mut state_1 = PairChargeTransfer{
            m_h:m_h,
            m_l:m_l,
            pair_type:type_ij,
            properties:Properties::new(),
        };
        // prepare the TDA calculation of both states
        state_1.prepare_ct_tda(
            self.properties.gamma().unwrap(),
            self.properties.gamma_lr().unwrap(),
            self.properties.s().unwrap(),
            &self.atoms
        );
        state_1.run_ct_tda(&self.atoms,10,150,1.0e-4);
        let q_ov_1:ArrayView2<f64> = state_1.properties.q_ov().unwrap();

        let tdm_1:ArrayView1<f64> = state_1.properties.ci_coefficient(0).unwrap();
        let ct_1 = ChargeTransferPair{
            m_h:m_h.index,
            m_l:m_l.index,
            state_index:0,
            state_energy:state_1.properties.ci_eigenvalue(0).unwrap(),
            eigenvectors: state_1.properties.tdm(0).unwrap().to_owned(),
            q_tr: q_ov_1.dot(&tdm_1),
            tr_dipole: state_1.properties.tr_dipole(0).unwrap(),
        };
        drop(m_h);
        drop(m_l);
        let grad = self.new_charge_transfer_pair_gradient(&ct_1);
        let m_h:&Monomer = &self.monomers[monomer_index_i];
        let m_l:&Monomer = &self.monomers[monomer_index_j];

        let mut full_gradient:Array1<f64> = Array1::zeros(self.atoms.len()*3);
        full_gradient.slice_mut(s![m_h.slice.grad]).assign(&grad.slice(s![..m_h.n_atoms*3]));
        full_gradient.slice_mut(s![m_l.slice.grad]).assign(&grad.slice(s![m_h.n_atoms*3..]));

        return  full_gradient;
    }

    pub fn test_new_charge_transfer_gradient(&mut self){
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        assert_deriv(self, SuperSystem::new_fmo_ct_energy_wrapper, SuperSystem::new_ct_gradient_wrapper, self.get_xyz(), 0.01, 1e-6);
    }

    pub fn fmo_le_energy_wrapper(&mut self,geometry: Array1<f64>)->f64{
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry);
        self.prepare_scc();
        self.run_scc();

        let monomer_index:usize = 0;
        let le_state:usize = 0;
        let val:f64 = self.exciton_le_energy(monomer_index,le_state);

        return val;
    }

    pub fn fmo_le_gradient_wrapper(&mut self)->Array1<f64>{
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        let monomer_index:usize = 0;
        let le_state:usize = 0;
        // calculate the gradient of the le_energy
        let grad:Array1<f64> = self.exciton_le_gradient(monomer_index,le_state);

        let mut full_gradient:Array1<f64> = Array1::zeros(self.atoms.len()*3);
        let mol = &self.monomers[monomer_index];
        full_gradient.slice_mut(s![mol.slice.grad]).assign(&grad);

        return full_gradient;
    }

    pub fn test_le_gradient(&mut self){
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        self.run_scc();

        assert_deriv(self, SuperSystem::fmo_le_energy_wrapper, SuperSystem::fmo_le_gradient_wrapper, self.get_xyz(), 0.01, 1e-6);
    }


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
        self.update_xyz(geometry.clone());
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (monomer_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let pair_energy: f64 = self.pair_scc(dq.view());
        self.properties.set_dq(dq);
        let emb_energy:f64 = self.embedding_energy();
        let esd_energy: f64 = self.esd_pair_energy();
        // println!("LEN {}", geometry.len());
        // for (atom, coord) in self.atoms.iter().zip(geometry.into_shape([self.atoms.len(), 3]).unwrap().axis_iter(Axis(0))) {
        //     println!("{:<6} {:>12.6} {:>12.6} {:>12.6}", atom.name, coord[0] * BOHR_TO_ANGS, coord[1] * BOHR_TO_ANGS, coord[2]*BOHR_TO_ANGS);
        // }
        // println!("{} {} {} {}", monomer_energy, pair_energy, esd_energy, emb_energy);
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

