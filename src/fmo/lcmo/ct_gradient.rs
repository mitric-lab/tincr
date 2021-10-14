use crate::fmo::helpers::get_pair_slice;
use crate::fmo::{Monomer, Pair, SuperSystem, GroundStateGradient, PairType, ESDPair};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_atomwise_ab};
use ndarray::prelude::*;
use std::ops::AddAssign;
use crate::scc::h0_and_s::h0_and_s_ab;

impl SuperSystem {
    pub fn ct_gradient_new(
        &mut self,
        index_i: usize,
        index_j: usize,
        ct_ind_i: usize,
        ct_ind_j: usize,
        ct_energy:f64,
        hole_i:bool,
    )->Array1<f64>{
        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        // get pair type
        let pair_type:PairType = self.properties.type_of_pair(index_i, index_j);
        let mut ct_gradient:Array1<f64> = Array1::zeros([3*(m_i.n_atoms+m_j.n_atoms)]);

        if pair_type == PairType::Pair{
            // get pair index
            let pair_index:usize = self.properties.index_of_pair(index_i,index_j);
            // get correct pair from pairs vector
            let pair_ij: &mut Pair = &mut self.pairs[pair_index];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            // calculate the overlap matrix
            if pair_ij.properties.s().is_none() {
                let mut s: Array2<f64> = Array2::zeros([pair_ij.n_orbs, pair_ij.n_orbs]);
                let (s_ab, h0_ab): (Array2<f64>, Array2<f64>) = h0_and_s_ab(
                    m_i.n_orbs,
                    m_j.n_orbs,
                    &pair_atoms[0..m_i.n_atoms],
                    &pair_atoms[m_i.n_atoms..],
                    &m_i.slako,
                );
                let mu: usize = m_i.n_orbs;
                s.slice_mut(s![0..mu, 0..mu])
                    .assign(&m_i.properties.s().unwrap());
                s.slice_mut(s![mu.., mu..])
                    .assign(&m_j.properties.s().unwrap());
                s.slice_mut(s![0..mu, mu..]).assign(&s_ab);
                s.slice_mut(s![mu.., 0..mu]).assign(&s_ab.t());

                pair_ij.properties.set_s(s);
            }
            // get the gamma matrix
            if pair_ij.properties.gamma().is_none() {
                let a: usize = m_i.n_atoms;
                let mut gamma_pair: Array2<f64> = Array2::zeros([pair_ij.n_atoms, pair_ij.n_atoms]);
                let gamma_ab: Array2<f64> = gamma_atomwise_ab(
                    &pair_ij.gammafunction,
                    &pair_atoms[0..m_i.n_atoms],
                    &pair_atoms[m_j.n_atoms..],
                    m_i.n_atoms,
                    m_j.n_atoms,
                );
                gamma_pair
                    .slice_mut(s![0..a, 0..a])
                    .assign(&m_i.properties.gamma().unwrap());
                gamma_pair
                    .slice_mut(s![a.., a..])
                    .assign(&m_j.properties.gamma().unwrap());
                gamma_pair.slice_mut(s![0..a, a..]).assign(&gamma_ab);
                gamma_pair.slice_mut(s![a.., 0..a]).assign(&gamma_ab.t());

                pair_ij.properties.set_gamma(gamma_pair);
            }
            pair_ij.prepare_ct_state(&pair_atoms,m_i,m_j,ct_ind_i,ct_ind_j,ct_energy,hole_i);
            ct_gradient = pair_ij.tda_gradient_lc(0);
        }
        else{
            // Do something for ESD pairs
            // get pair index
            let pair_index:usize = self.properties.index_of_esd_pair(index_i,index_j);
            // get correct pair from pairs vector
            let pair_ij: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            // do a scc calculation of the ESD pair
            pair_ij.prepare_scc(&pair_atoms,m_i,m_j);
            pair_ij.run_scc(&pair_atoms,self.config.scf);

            pair_ij.prepare_ct_state(&pair_atoms,m_i,m_j,ct_ind_i,ct_ind_j,ct_energy,hole_i);
            ct_gradient = pair_ij.tda_gradient_nolc(0);
            pair_ij.properties.reset();
        }

        return ct_gradient;
    }
}