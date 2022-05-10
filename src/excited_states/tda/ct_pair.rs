use crate::excited_states::solvers::DavidsonEngine;
use crate::excited_states::{orbe_differences, trans_charges, ProductCache};
use crate::fmo::{Monomer, PairChargeTransfer, PairType};
use crate::initialization::Atom;
use ndarray::prelude::*;
use crate::{Davidson, initial_subspace};
use crate::excited_states::tda::moments::{mulliken_dipoles, oscillator_strength};
use crate::excited_states::tda::new_mod::TdaStates;
use crate::fmo::helpers::get_pair_slice;

impl PairChargeTransfer<'_> {
    pub fn prepare_ct_tda(&mut self,g0:ArrayView2<f64>,g0_lr:ArrayView2<f64>,s_full:ArrayView2<f64>,atoms: &[Atom]) {
        let occ_indices: &[usize] = self.m_h.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.m_l.properties.virt_indices().unwrap();

        let natoms_h:usize = self.m_h.n_atoms;
        let natoms_l:usize = self.m_l.n_atoms;
        let n_atoms:usize =  natoms_h + natoms_l;

        // set the gamma matrix
        let mut gamma:Array2<f64> = Array2::zeros([n_atoms,n_atoms]);
        gamma.slice_mut(s![..natoms_h,..natoms_h]).assign(&self.m_h.properties.gamma().unwrap());
        gamma.slice_mut(s![natoms_h..,natoms_h..]).assign(&self.m_l.properties.gamma().unwrap());
        let gamma_ab:ArrayView2<f64> = g0.slice(s![self.m_h.slice.atom,self.m_l.slice.atom]);
        gamma.slice_mut(s![..natoms_h,natoms_h..]).assign(&gamma_ab);
        gamma.slice_mut(s![natoms_h..,..natoms_h]).assign(&gamma_ab.t());

        // get the overlap matrix
        let s:ArrayView2<f64> = s_full.slice(s![self.m_h.slice.orb,self.m_l.slice.orb]);
        self.properties.set_gamma(gamma);

        // set the gamma lr matrix
        if self.pair_type == PairType::Pair{
            let gamma_lr:ArrayView2<f64> = g0_lr.slice(s![self.m_h.slice.atom,self.m_l.slice.atom]);
            self.properties.set_gamma_lr(gamma_lr.to_owned());
        }

        // The index of the HOMO (zero based).
        let homo: usize = occ_indices[occ_indices.len() - 1];

        // The index of the LUMO (zero based).
        let lumo: usize = virt_indices[0];

        // Energies of the occupied orbitals.
        let orbe_h:ArrayView1<f64> = self.m_h.properties.orbe().unwrap();
        let orbe_occ: ArrayView1<f64> = orbe_h.slice(s![0..homo + 1]);

        // Energies of the virtual orbitals.
        let orbe_l:ArrayView1<f64> = self.m_l.properties.orbe().unwrap();
        let orbe_virt: ArrayView1<f64> = orbe_l.slice(s![lumo..]);

        // Energy differences between virtual and occupied orbitals.
        let omega: Array1<f64> = orbe_differences(orbe_occ, orbe_virt);

        // Energy differences are stored in the molecule.
        self.properties.set_omega(omega);
        self.properties.set_homo(homo);
        self.properties.set_lumo(lumo);

        // get the atoms of the fragments
        let atoms_h:&[Atom] = &atoms[self.m_h.slice.atom_as_range()];
        let atoms_l:&[Atom] = &atoms[self.m_l.slice.atom_as_range()];

        // calculate q_ov
        let q_ov:Array2<f64> = self.calculate_q_ov(s,atoms_h,atoms_l);
        // store the transition charges
        self.properties.set_q_oo(self.m_h.properties.q_oo().unwrap().to_owned());
        self.properties.set_q_ov(q_ov);
        self.properties.set_q_vv(self.m_l.properties.q_vv().unwrap().to_owned());
        self.properties.set_occ_indices(occ_indices.to_vec());
        self.properties.set_virt_indices(virt_indices.to_vec());
    }

    pub fn calculate_q_ov(&self, s:ArrayView2<f64>,atoms_h:&[Atom],atoms_l:&[Atom])->Array2<f64>{
        let homo = self.properties.homo().unwrap();
        let occs = self.m_h.properties.orbs_slice(0, Some(homo + 1)).unwrap();
        let lumo = self.properties.lumo().unwrap();
        let virts = self.m_l.properties.orbs_slice(lumo, None).unwrap();

        // Matrix product of overlap matrix with the orbitals on L.
        let s_c_l:Array2<f64> = s.dot(&virts);
        // Matrix product of overlap matrix with the orbitals on H.
        let s_c_h:Array2<f64> = s.t().dot(&occs);
        // Number of molecular orbitals on monomer I.
        let dim_h: usize = occs.ncols();
        // Number of molecular orbitals on monomer J.
        let dim_l: usize = virts.ncols();
        // get the number of atoms
        let natoms_h:usize = atoms_h.len();
        let natoms_l:usize = atoms_l.len();
        let n_atoms:usize = natoms_h + natoms_l;
        // The transition charges between the two sets of MOs  are initialized.
        let mut q_trans: Array3<f64> = Array3::zeros([n_atoms, dim_h, dim_l]);

        let mut mu: usize = 0;
        for (atom_h,mut q_n) in atoms_h.iter().zip(q_trans.slice_mut(s![0..natoms_h, .., ..]).axis_iter_mut(Axis(0))){
            for _ in 0..atom_h.n_orbs {
                for (orb_h, mut q_h) in occs.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (sc, mut q) in s_c_l.row(mu).iter().zip(q_h.iter_mut()) {
                        *q += orb_h * sc;
                    }
                }
                mu += 1;
            }
        }
        mu = 0;
        for (atom_l, mut q_n) in atoms_l.iter().zip(q_trans.slice_mut(s![natoms_h.., .., ..]).axis_iter_mut(Axis(0))){
            for _ in 0..atom_l.n_orbs{
                for (sc, mut q_l) in s_c_h.row(mu).iter().zip(q_n.axis_iter_mut(Axis(0))) {
                    for (orb_l, mut q) in virts.row(mu).iter().zip(q_l.iter_mut()) {
                        *q += orb_l * sc;
                    }
                }
                mu += 1;
            }
        }
        q_trans = 0.5 * q_trans;
        q_trans.into_shape([n_atoms,dim_h*dim_l]).unwrap()
    }

    pub fn run_ct_tda(&mut self, atoms: &[Atom], n_roots: usize, max_iter: usize, tolerance: f64) {
        // Set an empty product cache.
        self.properties.set_cache(ProductCache::new());

        // Reference to the energy differences between virtuals and occupied orbitals.
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();

        // The initial guess for the subspace is created.
        let guess: Array2<f64> = initial_subspace(omega.view(), n_roots);

        // Davidson iteration.
        let davidson: Davidson = Davidson::new(self, guess, n_roots, tolerance, max_iter).unwrap();

        // Reference to the o-v transition charges.
        let q_ov: ArrayView2<f64> = self.properties.q_ov().unwrap();

        // The transition charges for all excited states are computed.
        let q_trans: Array2<f64> = q_ov.dot(&davidson.eigenvectors);

        let pair_atoms: Vec<Atom> = get_pair_slice(
            atoms,
            self.m_h.slice.atom_as_range(),
            self.m_l.slice.atom_as_range(),
        );

        // The Mulliken transition dipole moments are computed.
        let tr_dipoles: Array2<f64> = mulliken_dipoles(q_trans.view(), &pair_atoms);

        // The oscillator strengths are computed.
        let f: Array1<f64> = oscillator_strength(davidson.eigenvalues.view(), tr_dipoles.view());

        // let n_occ: usize = self.m_h.properties.occ_indices().unwrap().len();
        // let n_virt: usize = self.m_l.properties.virt_indices().unwrap().len();
        // let tdm: Array3<f64> = davidson
        //     .eigenvectors
        //     .clone()
        //     .into_shape([n_occ, n_virt, f.len()])
        //     .unwrap();
        // let states: TdaStates = TdaStates {
        //     total_energy: self.properties.last_energy().unwrap(),
        //     energies: davidson.eigenvalues.clone(),
        //     tdm: tdm,
        //     f: f.clone(),
        //     tr_dip: tr_dipoles.clone(),
        //     orbs: self.properties.orbs().unwrap().to_owned(),
        // };

        // The eigenvalues are the excitation energies and the eigenvectors are the CI coefficients.
        self.properties.set_ci_eigenvalues(davidson.eigenvalues);
        self.properties.set_ci_coefficients(davidson.eigenvectors);
        self.properties.set_q_trans(q_trans);
        self.properties.set_tr_dipoles(tr_dipoles);
        self.properties.set_oscillator_strengths(f);

        // println!("{}", states);
    }
}
