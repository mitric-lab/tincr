use crate::defaults;
use ndarray::*;
use ndarray_linalg::{Norm, Solve};
use ndarray_stats::QuantileExt;
use std::cmp::min;
use std::iter::FromIterator;

pub struct BroydenMixer {
    trial_vectors: Vec<Array2<f64>>,
    residual_vectors: Vec<Array1<f64>>,
    memory: usize,
    iter: usize,
    start: bool,
}

impl BroydenMixer {
    pub fn new() -> BroydenMixer {
        let t_v: Vec<Array2<f64>> = Vec::new();
        let r_v: Vec<Array1<f64>> = Vec::new();
        return BroydenMixer {
            trial_vectors: t_v,
            residual_vectors: r_v,
            memory: defaults::DIIS_LIMIT,
            iter: 0,
            start: false,
        };
    }

    pub fn reset(&mut self) {
        self.residual_vectors = Vec::new();
        self.trial_vectors = Vec::new();
        self.iter = 0;
        self.start = false;
    }



    fn get_approximation(&mut self) -> Array2<f64> {
        // determine the best linear combination of the previous
        // solution vectors.
        let diis_count: usize = self.residual_vectors.len();
        assert!(
            diis_count > 1,
            "There should be at least 2 residual vectors"
        );
        // build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        let mut b: Array2<f64> = Array2::zeros((diis_count + 1, diis_count + 1));
        for (idx1, e1) in self.residual_vectors.iter().enumerate() {
            for (idx2, e2) in self.residual_vectors.iter().enumerate() {
                if idx2 <= idx1 {
                    let val: f64 = e1.dot(e2);
                    b[[idx1, idx2]] = val;
                    b[[idx2, idx1]] = val;
                }
            }
        }
        b.slice_mut(s![diis_count, ..]).fill(-1.0);
        b.slice_mut(s![.., diis_count]).fill(-1.0);
        b[[diis_count, diis_count]] = 0.0;

        // normalize
        // calculate the maximal element of the array slice
        let max: f64 = *b
            .slice(s![0..diis_count, 0..diis_count])
            .map(|x| x.abs())
            .max()
            .unwrap();
        b.slice_mut(s![0..diis_count, 0..diis_count])
            .map(|x| x / max);

        // build residual vector, [Pulay:1980:393], Eqn. 6, RHS
        let mut resid: Array1<f64> = Array1::zeros((diis_count + 1));
        resid[diis_count] = -1.0;

        // Solve Pulay equations, [Pulay:1980:393], Eqn. 6
        let ci: Array1<f64> = b.solve_into(resid).unwrap();

        // calculate new density matrix as linear combination of previous density matrices
        let mut p_next: Array2<f64> = Array2::zeros(self.trial_vectors[0].raw_dim());
        for (idx, coeff) in ci.slice(s![0..diis_count]).iter().enumerate() {
            p_next += &self.trial_vectors[idx].map(|x| x * *coeff);
        }
        let t_len: usize = self.trial_vectors.len();
        self.trial_vectors[t_len - 1] = p_next.clone();
        self.residual_vectors[diis_count - 1] = Array1::from_iter(
            (&self.trial_vectors[t_len - 1] - &self.trial_vectors[t_len - 2])
                .iter()
                .cloned(),
        );
        return p_next;
    }



}