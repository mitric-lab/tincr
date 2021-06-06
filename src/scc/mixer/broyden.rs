use ndarray::*;
use ndarray_linalg::{Inverse};
use crate::defaults;
use crate::scc::mixer::Mixer;

/// Modified Broyden mixer
///
/// The algorithm is based on the implementation in the DFTB+ Code
/// see https://github.com/dftbplus/dftbplus/blob/master/prog/dftb%2B/lib_mixer/broydenmixer.F90
/// and J. Chem. Phys. 152, 124101 (2020); https://doi.org/10.1063/1.5143190
#[derive(Debug, Clone)]
pub struct BroydenMixer {
    // current iteration
    iter: usize,
    maxiter: usize,
    omega0: f64,
    // mixing parameter
    alpha: f64,
    // minimal weight allowed
    min_weight: f64,
    // maximal weight allowed
    max_weight: f64,
    // numerator of the weight
    weight_factor: f64,
    weights: Array1<f64>,
    // charge difference in last iteration
    delta_q_old: Array1<f64>,
    // input charges in last iteration
    pub q_old: Array1<f64>,
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
            maxiter: defaults::MAX_ITER,
            omega0: defaults::BROYDEN_OMEGA0,
            alpha: defaults::BROYDEN_MIXING_PARAMETER,
            min_weight: defaults::BROYDEN_MIN_WEIGHT,
            max_weight: defaults::BROYDEN_MAX_WEIGHT,
            weight_factor: defaults::BROYDEN_WEIGHT_FACTOR,
            weights: Array1::zeros([defaults::MAX_ITER - 1]),
            delta_q_old: Array1::zeros([n_atoms]),
            q_old: Array1::zeros([n_atoms]),
            a_mat: Array2::zeros([defaults::MAX_ITER - 1, defaults::MAX_ITER - 1]),
            df: Array2::zeros([n_atoms, defaults::MAX_ITER - 1]),
            uu: Array2::zeros([n_atoms, defaults::MAX_ITER - 1]),
        }
    }

    fn next(&mut self, q: Array1<f64>, delta_q: Array1<f64>) -> Array1<f64> {
        self.iter += 1;
        self.mix(q, delta_q)
    }

    fn reset(&mut self, n_atoms: usize) {
        self.iter = 0;
        self.weights = Array1::zeros([self.maxiter - 1]);
        self.a_mat = Array2::zeros([self.maxiter - 1, self.maxiter - 1]);
        self.delta_q_old = Array1::zeros([n_atoms]);
        self.q_old = Array1::zeros([n_atoms]);
        self.a_mat = Array2::zeros([defaults::MAX_ITER - 1, defaults::MAX_ITER - 1]);
        self.df = Array2::zeros([n_atoms, defaults::MAX_ITER - 1]);
        self.uu = Array2::zeros([n_atoms, defaults::MAX_ITER - 1]);
    }

    /// Mixes dq from current diagonalization and the difference to the last iteration
    fn mix(
        &mut self,
        q: Array1<f64>,
        delta_q: Array1<f64>,
    ) -> Array1<f64> {
        let mut q: Array1<f64> = q;
        if self.iter > 14 {
            println!("iter {} QINP {} QDIFF {} QINPLAST {} ", self.iter, q, delta_q, self.q_old);
        }

        let q_out: Result<Array1<f64>, _> = match self.iter {
            // In the first iteration a linear damping scheme is used.
            // q = q + alpha * Delta q, where alpha is the Broyden mixing parameter
            0 => {
                // The current q is stored for the next iteration.
                self.q_old = q.clone();
                // The same is done for the difference.
                self.delta_q_old = delta_q.clone();
                // Linear interpolation/damping.
                Ok(&q + &(&delta_q * self.alpha))
            },
            // For all other iterations the Broyden mixing is used.
            _ if self.iter < self.maxiter - 1  => {
                // Index variable to acess the matrix/vector element of the current iteration.
                let idx: usize = self.iter - 1;
                
                // Create the weight factor of the current iteration.
                let mut weight: f64 = delta_q.dot(&delta_q).sqrt();
                if weight > self.weight_factor / self.max_weight {
                    weight = self.weight_factor / weight;
                } else {
                    weight = self.max_weight;
                }
                if weight < self.min_weight {
                    weight = self.min_weight;
                }
                // Store the current weight in the Struct.
                self.weights[idx] = weight;

                // Build |DF(idx)>.
                let mut df_idx: Array1<f64> = &delta_q - &self.delta_q_old;
                // Normalize it.
                let mut inv_norm: f64 = df_idx.dot(&df_idx).sqrt();
                // Prevent division by zero.
                inv_norm = if inv_norm > 1e-14 { inv_norm } else { 1e-14 };
                // Take the inverse of the vector norm, since it is used later again.
                inv_norm = 1.0 / inv_norm;
                df_idx = &df_idx * inv_norm;

                let mut c: Array1<f64> = Array1::zeros([self.iter]);
                // Build a, beta, c, and gamma
                for i in 0..idx {
                    self.a_mat[[i, idx]] = self.df.slice(s![.., i]).dot(&df_idx);
                    self.a_mat[[idx, i]] = self.a_mat[[i, idx]];
                    c[i] = self.weights[i] * self.df.slice(s![.., i]).dot(&delta_q);
                    //println!("CC {} DF dot qdiff {} weights {} len df {}", c[i], self.df.slice(s![.., i]).dot(&delta_q), self.weights[i], delta_q.dot(&delta_q));
                }
                self.a_mat[[idx, idx]] = 1.0;
                c[idx] = self.weights[idx] * df_idx.dot(&delta_q);

                let mut beta: Array2<f64> = Array2::zeros([self.iter, self.iter]);
                for i in 0..self.iter {
                    beta.slice_mut(s![0.., i]).assign(
                        &(self.weights[i] * &(&self.weights.slice(s![0..self.iter]) * &self.a_mat.slice(s![0..self.iter, i])))
                    );
                    beta[[i, i]] = beta[[i, i]] + self.omega0.powi(2);
                }
                // The inverse of the matrix is computed.
                beta = beta.inv().unwrap();
                let gamma: Array1<f64> = c.dot(&beta);
                // Store |dF(m-1)>
                self.df.slice_mut(s![.., idx]).assign(&df_idx);

                // Create |u(m-1)>
                self.uu.slice_mut(s![.., idx]).assign(&(&(&df_idx * self.alpha) + &((&q - &self.q_old) * inv_norm)));
                // Save charge vectors before overwriting
                self.q_old = q.clone();
                self.delta_q_old = delta_q.clone();

                // Build new vector
                q = &q + &(self.alpha * &delta_q);
                for i in 0..self.iter {
                    //println!("UU {} WW {} CC {} GAMMA {} Q {} MINUS {}", self.uu.slice(s![.., i]), self.weights[i], c[i],gamma[i], q, &(&self.uu.slice(s![.., i]) * self.weights[i] * gamma[i]));
                    q = q - &(&self.uu.slice(s![.., i]) * self.weights[i] * gamma[i]);
                }
                Ok(q)
            },
            _ => Err("SCC did not converge"),
        };

        return q_out.unwrap();
    }

}

