use crate::defaults;
use ndarray::*;
use ndarray_linalg::{Norm, Solve, Inverse};
use ndarray_stats::QuantileExt;
use std::cmp::{min, max};
use std::iter::FromIterator;
use peroxide::special::function::beta;

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
    pub fn new(alpha: f64, omega0:f64, min_weight: f64, max_weight: f64, weight_factor: f64)  {
        assert!(alpha > 0.0);
        assert!(omega0 > 0.0);

    }

    pub fn reset(&mut self) {
        self.residual_vectors = Vec::new();
        self.trial_vectors = Vec::new();
        self.iter = 0;
        self.start = false;
    }


    fn mix(&mut self, nn: usize) {
        let nn_1: usize = self.iter - 1;
        // First iteration: simple mix and storage of qInp and qDiff
        if self.iter == 0 {
            self.q_inp_last = q_inp_result;
            self.q_diff_last = q_diff;
            q_inp_result = q_inp_result + alpha * q_diff;
        } else {

        }

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
        for ii in 0..self.iter-2 {
            self.aa[[ii, nn_1]] = self.df.slice(s![.., ii]).dot(&df_uu);
            self.aa[[nn_1, ii]] = a[[ii, nn_1]];
            cc[[0, ii]] = self.df.slice(s![.., ii]).dot(&self.q_diff_last).mapv(|x| x * self.ww[ii]);
        }
        self.aa[[nn_1, nn_1]] = 1.0;
        cc[[0, nn_1]] = self.ww[nn_1] * df_uu.dot(&self.q_diff_last);

        let mut beta: Array2<f64> = Array2::zeros([]);
        for ii in 0..nn_1 {

            beta[[ii, ii]] = beta[[ii, ii]] * omega0 ** 2;
        }
        beta = beta.inv().unwrap();

        let gamma: Array2<f64> = cc.dot(&beta);

        // Store |dF(m-1)>
        self.df.slice_mut(s![.., nn_1]) = df_uu.view_mut();

        // Create |u(m-1)>
        df_uu = df_uu.mapv(|x| x * self.alpha) + (&q_inp_result - &q_inp_last).mapv(|x| x * inv_norm);

        // Save charge vectors before overwriting
        q_inp_last = q_inp_result;
        q_diff_last = q_diff;

        // Build new vector
        q_inp_result = q_inp_result + self.alpha * q_diff;
        for ii in 0..nn-2 {
            q_inp_result = q_inp_result - self.ww[ii] * gamma[[0,ii]] * uu[[..,ii]];
        }
        q_inp_result = q_inp_result - ww[nn_1] * gamma[[0, nn_1]] * df_uu;

        // Save |u(m-1)>
        uu.slice_mut(s![.., nn_1])= df_uu.view_mut();

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