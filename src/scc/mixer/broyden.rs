use ndarray::*;
use ndarray_linalg::krylov::qr;
use ndarray_linalg::{Inverse, Norm, Solve};
use ndarray_stats::QuantileExt;
use peroxide::special::function::beta;
use std::cmp::{max, min};
use std::iter::FromIterator;
use crate::defaults;
use crate::scc::mixer::Mixer;

/// Modified Broyden mixer
///
/// The algorithm is based on the implementation in the DFTB+ Code
/// see https://github.com/dftbplus/dftbplus/blob/master/prog/dftb%2B/lib_mixer/broydenmixer.F90
/// and J. Chem. Phys. 152, 124101 (2020); https://doi.org/10.1063/1.5143190
pub struct BroydenMixer {
    // current iteration
    iter: usize,
    miter: usize,
    omega0: f64,
    // mixing parameter
    alpha: f64,
    // minimal weight allowed
    min_weight: f64,
    // maximal weight allowed
    max_weight: f64,
    // numerator of the weight
    weight_factor: f64,
    ww: Array1<f64>,
    // charge difference in last iteration
    q_diff_last: Array1<f64>,
    // input charges in last iteration
    q_inp_last: Array1<f64>,
    // storage for A matrix
    a_mat: Array2<f64>,
    // df vectors
    df: Array2<f64>,
    // uu vectors
    uu: Array2<f64>,
}

impl Mixer for BroydenMixer {
    fn new(n_atoms: usize) -> BroydenMixer {
        BroydenMixer {
            iter: 0,
            miter: defaults::MAX_ITER,
            omega0: defaults::BROYDEN_OMEGA0,
            alpha: defaults::BROYDEN_MIXING_PARAMETER,
            min_weight: defaults::BROYDEN_MIN_WEIGHT,
            max_weight: defaults::BROYDEN_MAX_WEIGHT,
            weight_factor: defaults::BROYDEN_WEIGHT_FACTOR,
            ww: Array1::zeros([defaults::MAX_ITER - 1]),
            q_diff_last: Array1::zeros([n_atoms]),
            q_inp_last: Array1::zeros([n_atoms]),
            a_mat: Array2::zeros([defaults::MAX_ITER - 1, defaults::MAX_ITER - 1]),
            df: Array2::zeros([n_atoms, defaults::MAX_ITER - 1]),
            uu: Array2::zeros([n_atoms, defaults::MAX_ITER - 1]),
        }
    }

    fn next(&mut self, q_inp_result: Array1<f64>, q_diff: Array1<f64>) -> Array1<f64> {
        assert!(
            self.iter < self.miter,
            "Broyden Mixer: Maximal nr. of steps exceeded"
        );

        let q_result: Array1<f64> = self.mix(q_inp_result, q_diff);
        self.iter += 1;
        return q_result;
    }

    fn reset(&mut self, n_atoms: usize) {
        self.iter = 0;
        self.ww = Array1::zeros([self.miter - 1]);
        self.a_mat = Array2::zeros([self.miter - 1, self.miter - 1]);
        self.q_diff_last = Array1::zeros([n_atoms]);
        self.q_inp_last = Array1::zeros([n_atoms]);
        self.a_mat = Array2::zeros([defaults::MAX_ITER - 1, defaults::MAX_ITER - 1]);
        self.df = Array2::zeros([n_atoms, defaults::MAX_ITER - 1]);
        self.uu = Array2::zeros([n_atoms, defaults::MAX_ITER - 1]);
    }

    /// Mixes dq from current diagonalization and the difference to the last iteration
    fn mix(
        &mut self,
        q_inp_result: Array1<f64>,
        q_diff: Array1<f64>,
    ) -> (Array1<f64>) {
        // In the first iteration the counter `self.iter` is 1
        // Therefore the that corresponds to the iterations should be lower by 1
        let mut q_inp_result: Array1<f64> = q_inp_result;
        // First iteration: simple mix and storage of qInp and qDiff
        if self.iter == 0 {
            self.q_inp_last = q_inp_result.clone();
            self.q_diff_last = q_diff.clone();
            q_inp_result = q_inp_result + q_diff.mapv(|x| x * self.alpha);
        } else {
            let nn_1: usize = self.iter - 1;
            // Create weight factor
            let mut ww_at_n1: f64 = q_diff.dot(&q_diff).sqrt();
            if ww_at_n1 > self.weight_factor / self.max_weight {
                ww_at_n1 = self.weight_factor / ww_at_n1;
            } else {
                ww_at_n1 = self.max_weight;
            }
            if ww_at_n1 < self.min_weight {
                ww_at_n1 = self.min_weight;
            }
            self.ww[nn_1] = ww_at_n1;

            // Build |DF(m-1)> and  (m is the current iteration number)
            let mut df_uu: Array1<f64> = &q_diff - &self.q_diff_last;
            let mut inv_norm: f64 = df_uu.dot(&df_uu).sqrt();
            inv_norm = if inv_norm > 1e-12 { inv_norm } else { 1e-12 };
            inv_norm = 1.0 / inv_norm;
            df_uu = df_uu.mapv(|x| x * inv_norm);

            let mut cc: Array2<f64> = Array2::zeros([1, self.iter]);
            // Build a, beta, c, and gamma
            for i in 0..nn_1 {
                self.a_mat[[i, nn_1]] = self.df.slice(s![.., i]).dot(&df_uu);
                self.a_mat[[nn_1, i]] = self.a_mat[[i, nn_1]];
                cc[[0, i]] = self.df.slice(s![.., i]).dot(&q_diff) * self.ww[i];
            }
            self.a_mat[[nn_1, nn_1]] = 1.0;
            cc[[0, nn_1]] = self.ww[nn_1] * df_uu.dot(&q_diff);

            let mut beta: Array2<f64> = Array2::zeros([self.iter, self.iter]);
            for i in 0..self.iter {
                beta.slice_mut(s![..self.iter, i]).assign(
                    &(&self.ww.slice(s![..self.iter]).mapv(|x| x * self.ww[i])
                        * &self.a_mat.slice(s![..self.iter, i])),
                );
                beta[[i, i]] = beta[[i, i]] + self.omega0.powi(2) + 1.0;
            }

            beta = beta.inv().unwrap();
            let gamma: Array2<f64> = cc.dot(&beta);
            // Store |dF(m-1)>
            self.df.slice_mut(s![.., nn_1]).assign(&df_uu);

            // Create |u(m-1)>
            df_uu = df_uu.mapv(|x| x * self.alpha)
                + (&q_inp_result - &self.q_inp_last).mapv(|x| x * inv_norm);

            // Save charge vectors before overwriting
            self.q_inp_last = q_inp_result.clone();
            self.q_diff_last = q_diff.clone();

            // Build new vector
            q_inp_result = q_diff.mapv(|x| x * self.alpha) + &q_inp_result;
            for i in 0..nn_1 {
                q_inp_result = q_inp_result
                    - self
                    .uu
                    .slice(s![.., i])
                    .mapv(|x| x * self.ww[i] * gamma[[0, i]]);
            }

            q_inp_result = q_inp_result - &df_uu.mapv(|x| x * self.ww[nn_1] * gamma[[0, nn_1]]);

            // Save |u(m-1)>
            self.uu.slice_mut(s![.., nn_1]).assign(&df_uu);
        }

        return q_inp_result;
    }

}

