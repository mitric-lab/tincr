use crate::data::{Parametrization};
use ndarray::prelude::*;
use crate::scc::gamma_approximation::gamma_atomwise;
use core::::Atom;
use crate::fmo::MolecularSlice;

impl<'a> Parametrization<'a> {
    /// The Gamma matrix as well as the Overlap matrix and the core Hamiltonian is computed.
    pub fn new(atoms: &[Atom], params: &Parametrization) -> Self {
        todo!()
        // let gamma: Array2<f64> = gamma_atomwise(&gf, &atoms);
        // let gamma_lrc: Option<Array2<f64>> = if config.lc.long_range_correction {
        //     Some(gamma_atomwise(&gf_lrc, &atoms))
        // } else {
        //     None
        // };
        //
        // // The overlap matrix, S, and the one-electron integral Fock matrix is computed from the
        // // precalculated splines that are tabulated in the parameter files.
        // let (s, h0): (Array2<f64>, Array2<f64>) = slako.h0_and_s(n_orbs, &atoms);
        //
        // Self{ h0, s, gamma, gamma_lrc }
    }
    
    pub fn slice(&self, slice: MolecularSlice) -> Self {

        let gamma_lrc: Option<ArrayView2<f64>> = match self.gamma_lrc {
            Some(gamma) => Some(gamma.slice(s![slice.atom, slice.atom])),
            None => None,
        };

        Self{
            h0: self.h0.slice(s![slice.orb, slice.orb]),
            s: self.s.slice(s![slice.orb, slice.orb]),
            gamma: self.gamma.slice(s![slice.atom, slice.atom]),
            gamma_lrc,
        }
    }
    
}