use crate::defaults;
use ndarray::*;
use ndarray_linalg::{Inverse, Norm, Solve};
use ndarray_stats::QuantileExt;
use peroxide::special::function::beta;
use std::cmp::{max, min};
use std::iter::FromIterator;
use ndarray_linalg::krylov::qr;

/// Modified Broyden mixer
///
///
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

impl BroydenMixer {
    pub fn new(n_atoms: usize) -> BroydenMixer {
        assert!(alpha > 0.0);
        assert!(omega0 > 0.0);
        BroydenMixer{
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
            uu: Array2::zeros([n_atoms, defaults::MAX_ITER - 1])
        }
    }

    pub fn reset(&mut self, n_atoms: usize) {
        self.iter = 0;
        self.ww = Array1::zeros([self.miter-1]);
        self.a_mat = Array2::zeros([self.miter-1, self.miter-1]);
        self.q_diff_last = Array1::zeros([n_atoms]);
        self.q_inp_last = Array1::zeros([n_atoms]);
        self.a_mat = Array2::zeros([defaults::MAX_ITER - 1, defaults::MAX_ITER - 1]);
        self.df = Array2::zeros([n_atoms, defaults::MAX_ITER - 1]);
        self.uu = Array2::zeros([n_atoms, defaults::MAX_ITER - 1]);
    }

    pub fn next(&mut self, q_inp_result: Array1<f64>, q_diff: Array1<f64>) -> Array1<f64> {
        self.iter += 1;
        assert!(self.iter < self.miter, "Broyden Mixer: Maximal nr. of steps exceeded");

        let q_result: Array1<f64> = self.get_approximation(q_inp_result, q_diff);
        return q_result;
    }

    // Does the real work for the Broyden mixer
    fn get_approximation(&mut self, q_inp_result: Array1<f64>, q_diff: Array1<f64>) -> (Array1<f64>) {
        let nn_1: usize = self.iter - 1;
        let mut q_inp_result: Array1<f64> = q_inp_result;
        // First iteration: simple mix and storage of qInp and qDiff
        if self.iter == 0 {
            self.q_inp_last = q_inp_result.cloned();
            self.q_diff_last = q_diff;
            q_inp_result = q_inp_result + q_diff.mapv(|x| x * self.alpha);
        } else {
            // Create weight factor
            let mut ww_at_n1: f64 = q_diff.dot(&qdiff).sqrt();
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
            let mut df_uu: Array1<f64> = &q_diff - &q_diff_last;
            let mut inv_norm: f64 = self.df.dot(&self.df).sqrt();
            inv_norm = max(inv_norm, 1e-12);
            inv_norm = 1.0 / inv_norm;
            df_uu = df_uu.mapv(|x| x * inv_norm);

            let mut cc: Array2<f64> = Array2::zeros([1, self.iter]);
            // Build a, beta, c, and gamma
            for ii in 0..self.iter - 2 {
                self.aa[[ii, nn_1]] = self.df.slice(s![.., ii]).dot(&df_uu);
                self.aa[[nn_1, ii]] = a[[ii, nn_1]];
                cc[[0, ii]] = self
                    .df
                    .slice(s![.., ii])
                    .dot(&self.q_diff_last)
                    .mapv(|x| x * self.ww[ii]);
            }
            self.aa[[nn_1, nn_1]] = 1.0;
            cc[[0, nn_1]] = self.ww[nn_1] * df_uu.dot(&self.q_diff_last);

            let mut beta: Array2<f64> = Array2::zeros([]);
            for ii in 0..nn_1 {
                beta[[ii, ii]] = beta[[ii, ii]] * omega0 * *2;
            }
            beta = beta.inv().unwrap();

            let gamma: Array2<f64> = cc.dot(&beta);
            // Store |dF(m-1)>
            self.df.slice_mut(s![.., nn_1]) = df_uu.view_mut();

            // Create |u(m-1)>
            df_uu = df_uu.mapv(|x| x * self.alpha)
                + (&q_inp_result - &q_inp_last).mapv(|x| x * inv_norm);

            // Save charge vectors before overwriting
            q_inp_last = q_inp_result.clone();
            q_diff_last = q_diff.clone();

            // Build new vector
            q_inp_result = &q_inp_result + q_diff.mapv(|x| x * self.alpha);
            for ii in 0..nn - 2 {
                q_inp_result = q_inp_result - self.ww[ii] * gamma[[0, ii]] * uu[[.., ii]];
            }
            q_inp_result = q_inp_result - ww[nn_1] * gamma[[0, nn_1]] * df_uu;

            // Save |u(m-1)>
            uu.slice_mut(s![.., nn_1]) = df_uu.view_mut();
        }

        return q_inp_result;
    }


}
