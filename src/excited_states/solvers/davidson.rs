/*!

# Davidson Diagonalization

The Davidson method is suitable for diagonal-dominant symmetric matrices,
that are quite common in certain scientific problems like [electronic
structure](https://en.wikipedia.org/wiki/Electronic_structure). The Davidson
method could be not practical for other kind of symmetric matrices.

The current implementation uses a general davidson algorithm, meaning
that it compute all the requested eigenvalues simultaneusly using a variable
size block approach. The family of Davidson algorithm only differ in the way
that the correction vector is computed.

*/

use crate::excited_states::solvers::utils;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_stats::QuantileExt;
use std::error;
use std::fmt;
use std::time::Instant;

#[derive(Debug, PartialEq)]
pub struct DavidsonError;

impl fmt::Display for DavidsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Davidson Algorithm did not converge!")
    }
}

impl error::Error for DavidsonError {}

use ndarray::prelude::*;
use ndarray::Data;

/// Abstract Trait defining the API required by solver engines.
///
/// Engines implement the correct product functions for iterative solvers that
/// do not require the target matrix be stored directly.
/// Classes intended to be used as an `engine` for `Davidson` or
/// `Hamiltonian` should implement this Trait to ensure
/// that the required methods are defined.
pub trait DavidsonEngine {
    /// Compute a Matrix * trial vector products
    /// Expected output:
    ///  The product`A x X_{i}` for each `X_{i}` in `X`, in that order.
    ///   Where `A` is the hermitian matrix to be diagonalized.
    fn compute_products(&mut self, x: ArrayView2<f64>) -> Array2<f64>;

    /// Apply the preconditioner to a Residual vector.
    /// The preconditioner is usually defined as :math:`(w_k - D_{i})^-1` where
    /// `D` is an approximation of the diagonal of the matrix that is being diagonalized.
    fn precondition(&self, r_k: ArrayView1<f64>, w_k:f64) -> Array1<f64>;

    /// Return the size of the matrix problem.
    fn get_size(&self) -> usize;
}

impl<S> DavidsonEngine for ArrayBase<S, Ix2>
    where
        S: Data<Elem = f64>,
{
    fn compute_products(&mut self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        self.dot(&x)
    }

    fn precondition(&self, r_k: ArrayView1<'_, f64>, w_k: f64) -> Array1<f64> {
        &r_k / &(Array1::from_elem(self.nrows(), w_k) - self.diag())
    }

    fn get_size(&self) -> usize {
        self.nrows()
    }
}


/// Structure with the configuration data
pub struct Davidson {
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
}

impl Davidson {
    /// Compute the lowest eigenvalues of a symmetric, diagonal dominant matrix.
    /// * `engine` an object that implements the `DavidsonEngine` trait.
    /// * `guess` the initial guess for the eigenvectors.
    /// * `nvalues` - the number of eigenvalues/eigenvectors pair to compute.
    /// * `n_roots` the number of (lowest) eigenvalues/eigenvectors to compute.
    /// * `tolerance` numerical tolerance for convergence.
    /// * `max_iter` the maximal number of iterations.
    pub fn new<D: DavidsonEngine>(
        engine: &mut D,
        guess: Array2<f64>,
        n_roots: usize,
        tolerance: f64,
        max_iter: usize,
    ) -> Result<Self, DavidsonError> {
        // Timer to measure the time within the Davidson routine.
        let timer: Instant = Instant::now();

        // Dimension of the original matrix problem.
        let dim: usize = engine.get_size();

        // The initial guess needs to be mutable.
        let mut guess: Array2<f64> = guess;

        // Dimension of the subspace.
        let dim_sub_origin: usize = guess.ncols();
        let mut dim_sub: usize = dim_sub_origin;

        // The maximal possible subspace, before it will be collapsed.
        let max_space: usize = 50;

        // The initial information of the Davidson routine are printed.
        utils::print_davidson_init(max_iter, n_roots, tolerance);

        // Initialization of the result.
        let mut result = Err(DavidsonError);

        // Outer loop block Davidson schema.
        for i in 0..max_iter {
            // 1. The initial subspace is formed by projecting into the new guess vectors.
            // Matrix-vector product of A with the trial vectors.
            let ax: Array2<f64> = engine.compute_products(guess.view());

            // 1.1 Initialization of the subspace Hamiltonian.
            let a_proj: Array2<f64> = guess.t().dot(&ax);

            // 2. Solve the eigenvalue problem for the subspace Hamiltonian.
            // The eigenvalues (u) and eigenvectors (v) are already sorted in ascending order.
            let (u, v): (Array1<f64>, Array2<f64>) = a_proj.eigh(UPLO::Upper).unwrap();

            // 3. Convergence checks are made.
            // 3.1 Compute the Ritz vectors.
            let ritz: Array2<f64> = guess.dot(&v);

            // 3.2 Compute the residue vectors.
            let rk: Array2<f64> = ax.dot(&v) - ritz.dot(&Array::from_diag(&u));

            // 3.3 Convergence check for each pair of eigenvalue and eigenvector.
            let errors: Array1<f64> = rk.slice(s![.., 0..n_roots])
                .axis_iter(Axis(1))
                .map(|col| col.norm()).collect();

            // The sum of all errors.
            let error: f64 = errors.sum();
            // The maximum value of the errors.
            let max_error: f64 = *errors.max().unwrap();

            // 4.3 Check how many eigenvalues are converged.
            let roots_cvd: usize = errors
                .iter()
                .fold(0, |n, &x| if x < tolerance { n + 1 } else { n });
            let roots_lft: usize = n_roots - roots_cvd;

            // If all eigenvalues are converged, the Davidson routine finished successfully.
            if roots_lft == 0 && i > 0{
                result = Ok(Self::create_results(
                    u.view(),
                    ritz.view(),
                    n_roots,
                ));
                utils::print_davidson_iteration(i, roots_cvd, n_roots - roots_cvd, error, max_error);
                break;
            }
            // The information of the current iteration is printed to the console.
            utils::print_davidson_iteration(i, roots_cvd, n_roots - roots_cvd, error, max_error);

            // 5.  If the eigenvalues are not yet converged, the subspace basis is updated.
            // 5.1 Correction vectors are added to the current subspace basis, if the new
            //     dimension is lower than the maximal subspace size.
            if dim_sub + roots_lft <= max_space {
                // For each (not converged) eigenvalue a new preconditioned subspace vector is
                // added.
                let mut add_space: Array2<f64> = Array::zeros([dim, roots_lft]);
                for ((idx, _), mut space_k) in errors
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x > tolerance)
                    .zip(add_space.axis_iter_mut(Axis(1)))
                {
                    space_k.assign(&engine.precondition(rk.column(idx), u[idx]));
                }
                // The dimension of the subspace is updated.
                dim_sub += roots_lft;

                // The new subspace vectors are orthonormalized and added to the existing basis.
                guess.append(Axis(1), add_space.view()).unwrap();
                guess = guess.qr().unwrap().0;

            }
            // 5.1 If the dimension is larger than the maximal subspace size, the subspace is
            //     collapsed.
            else {
                // The dimension of the subspace is reset to the initial value.
                dim_sub = dim_sub_origin;
                guess = ritz.slice(s![.., 0..dim_sub]).to_owned();
            }
        }
        // The end of the Davidson routine is noted in the console together with information
        // about the used wall time.
        utils::print_davidson_end(result.is_ok(), timer);

        // The returned result contains either an Err if the iteration is not converged or
        // an instance of Davidson that contains the eigenvectors and eigenvalues.
        result
    }

    /// Extract the requested eigenvalues/eigenvectors pairs
    fn create_results(
        subspace_eigenvalues: ArrayView1<f64>,
        ritz_vectors: ArrayView2<f64>,
        nvalues: usize,
    ) -> Davidson {
        Davidson {
            eigenvalues: subspace_eigenvalues.slice(s![0..nvalues]).to_owned(),
            eigenvectors: ritz_vectors.slice(s![.., 0..nvalues]).to_owned(),
        }
    }
}
