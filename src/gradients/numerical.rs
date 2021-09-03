use ndarray::prelude::*;
use ndarray_stats::{QuantileExt, DeviationExt};
use std::cmp::max;
use crate::scc::scc_routine::RestrictedSCC;
use crate::fmo::GroundStateGradient;
use crate::initialization::System;
use crate::excited_states::tda;

impl System{
    pub fn tda_gradient_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        self.update_xyz(geometry);
        self.prepare_scc();
        self.prepare_tda();
        self.run_tda(self.config.excited.nstates, 50, 1e-4);

        return self.properties.ci_eigenvalue(0).unwrap();
    }

    pub fn tda_nolc_gradients_for_testing(&mut self)->Array1<f64>{
        self.properties.reset();
        self.prepare_scc();
        self.run_scc();
        self.prepare_tda();
        self.run_tda(self.config.excited.nstates, 50, 1e-4);

        self.ground_state_gradient(true);
        self.prepare_excited_grad();
        let grad_exc:Array1<f64> = self.tda_gradient_nolc(0);

        return grad_exc;
    }

    pub fn tda_lc_gradients_for_testing(&mut self)->Array1<f64>{
        self.properties.reset();
        self.prepare_scc();
        self.run_scc();
        self.prepare_tda();
        self.run_tda(self.config.excited.nstates, 50, 1e-4);

        self.ground_state_gradient(true);
        self.prepare_excited_grad();
        let grad_exc:Array1<f64> = self.tda_gradient_lc(0);

        return grad_exc;
    }

    pub fn gs_grad(&mut self)->Array1<f64> {
        self.properties.reset();
        self.prepare_scc();
        self.run_scc();

        let grad = self.ground_state_gradient(false);
        return grad;
    }

    pub fn gs_gradient_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        self.update_xyz(geometry);
        self.prepare_scc();
        let en = self.run_scc().unwrap();

        return en
    }

    pub fn test_tda_nolc_gradient(&mut self){
        assert_deriv(self,System::tda_gradient_wrapper,System::tda_nolc_gradients_for_testing,self.get_xyz(), 0.0001, 1e-6);
    }

    pub fn test_tda_lc_gradient(&mut self){
        assert_deriv(self,System::tda_gradient_wrapper,System::tda_lc_gradients_for_testing,self.get_xyz(), 0.0001, 1e-6);
    }

    pub fn test_gs_gradient(&mut self){
        assert_deriv(self,System::gs_gradient_wrapper,System::gs_grad,self.get_xyz(), 0.001, 1e-6);
    }
}

/// Returns the derivative of a function `function` at an Array of points `origin` by Ridder's method.
/// The value `stepsize` is an initial stepsize, it need to be small, but should be an increment
/// over which the `function` changes substantially. An estimate of the error in the derivative is
/// returned. The method was developed by C.J.F Ridders in 1982 (see the original article
/// ["Accurate computation of F′(x) and F′(x) F″(x)"](https://doi.org/10.1016/S0141-1195(82)80057-0))
/// The implementation is based on the one described in the Book Numerical Recipes by
/// W. H. Press and S. A. Teukolsky, the section is available as an article in
/// [Computers in Physics](https://aip.scitation.org/doi/pdf/10.1063/1.4822971). Also the Python
/// implementation derivcheckby T. Verstraelen
/// influenced the implementation and the idea to create an `assert_deriv` function was adopted.
fn ridders_method<S, F, D>(
    system: &mut S,
    function: F,               // Function which should be differentiated
    origin: ArrayBase<D, Ix1>, // Origin of coordinates, that are used for the function
    index: usize,              // Index for which the derivative is computed
    stepsize: f64,             // Initial step size
    con: f64,                  // Rate at which the step size is contracted
    safe: f64,                 // Safety check to terminate the algorithm
    maxiter: usize, // Maximum number of iterations/function calls/order in Neville method
) -> (f64, f64)
where
    S: RestrictedSCC,
    F: Fn(&mut S, Array1<f64>) -> f64,
    D: ndarray::Data<Elem = f64>,
{
    // make the stepsize mutable
    let mut stepsize: f64 = stepsize;
    let mut step: Array1<f64> = Array1::zeros([origin.len()]);
    step[index] = 1.0;
    // compute the square of the contraction rate
    let con2: f64 = con.powi(2);
    // initialize the error
    let mut error: f64 = 0.0;

    let mut table: Vec<Vec<f64>> = vec![vec![
        (function(system, &origin + &(&step * stepsize)) - function(system, &origin - &(&step * stepsize))) / (2.0 * stepsize),
    ]];

    let mut estimate: f64 = 0.0;

    // Successive columns in the Neville tableau will go to smaller stepsizes and higher orders of
    // extrapolations.
    'main: for i in 1..maxiter {
        // Try new, smaller stepsize.
        stepsize /= con;
        // first-order approximation at current step
        table.push(vec![
            (function(system, &origin + &(&step * stepsize)) - function(system, &origin - &(&step * stepsize))) / (2.0 * stepsize),
        ]);

        // compute higher orders
        let mut fac = con2;
        for j in 1..(i + 1) {
            // Recursion relation based on Neville's method. It computes the extrapolations
            // of various orders, but requires no new function evaluation
            let tmp: f64 = (table[i][j - 1] * fac - table[i - 1][j - 1]) / (fac - 1.0);
            table[i].push(tmp);
            fac *= con2;

            // The error strategy is compare each new extrapolation to one order lower,
            // both at the present stepsize and the previous one.
            let current: f64 = (table[i][j] - table[i][j - 1]).abs();
            let last: f64 = (table[i][j] - table[i - 1][j - 1]).abs();
            let current_error: f64 = current.max(last);

            // If error is decreased, save the improved answer.
            if j == 1 || current_error <= error {
                error = current_error;
                estimate = table[i][j];
            }
        }
        // If higher order is worse by a significant factor `safe`, then quit early.
        if (&table[i][i] - &table[i - 1][i - 1]).abs() >= safe * error {
            break 'main;
        }
    }
    return (estimate, error);
}


/// Test the gradient of a function.
/// * function: The function whose derivatives must be tested, takes one argument
/// * gradient: Computes the gradient of the function, to be tested.
/// * origin: The point at which the derivatives are computed.
/// * stepsize: The initial (maximal) step size for the finite difference method.
/// * tol: The allowed relative error on the derivative.
/// The idea of this function comes from the derivcheck
/// Python package by T. Verstraelen.
pub fn assert_deriv<S, F, G>(
    system: &mut S,
    function: F,
    gradient: G,
    origin: Array1<f64>,
    stepsize: f64,
    tol: f64,
) where
    S: RestrictedSCC,
    F: Fn(&mut S, Array1<f64>) -> f64,
    G: Fn(&mut S) -> Array1<f64>,
{
    // Parameters for Ridder's method,
    let con: f64 = 1.4;
    let safe: f64 = 2.0;
    let maxiter: usize = 15;

    // compute the analytic gradient
    let analytic_grad: Array1<f64> = gradient(system);
    // initialize numerical grad
    let mut errors: Vec<bool> = Vec::with_capacity(origin.len());

    assert!(stepsize > 0.0, "The stepsize has to be > 0.0, but it is {}", stepsize);

    // The differences are stored in an Array
    let mut error_values: Array1<f64> = Array1::zeros([origin.len()]);

    println!(
        "{: <5} {: >18} {: >18} {: >18} {: >18} {: <8}",
        "Index", "Analytic", "Numerical", "Error", "Acc. Num.", "Correct?");
    // PARALLEL
    for i in 0..origin.len() {
        // get the corresponding analytic derivative
        let analytic_deriv: f64 = analytic_grad[i];
        // compute the numerical derivative of this function and an error estimate using
        // Ridder's method
        let (numerical_deriv, deriv_error): (f64, f64) = ridders_method(system, &function, origin.clone(), i, stepsize, con, safe, maxiter);
        let diff: f64 = (numerical_deriv - analytic_deriv).abs();
        let correct: bool = if diff >= deriv_error && diff > 1e-8 {false} else {true};
        errors.push(correct);
        error_values[i] = diff;

        println!(
            "{: >5} {:>18.14} {:>18.14} {:>18.14} {:>18.14} {: >5}",
            i, analytic_deriv, numerical_deriv, diff, deriv_error, correct);
    }
    let rmsd: f64 = (&error_values * &error_values).mean().unwrap().sqrt();
    let max: f64 = *error_values.max().unwrap();

    println!("{: <30} {:>18.4e}", "RMSD of Gradient", rmsd);
    println!("{: <30} {:18.4e}", "Max deviation of Gradient", max);

    assert!(!errors.contains(&false), "Gradient test failed")
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    // returns the sum of the square of all elements: y = x * x
    fn simple_function(values: Array1<f64>) -> f64 {
        values.iter().fold(0.0, |n, i| n+(i.powi(2)))
    }
    // returns the gradient of the function above: y' = 2 * x
    fn simple_gradient(values: Array1<f64>) -> Array1<f64> {
        2.0 * values
    }

    #[test]
    fn assert_deriv_simple_function() {
        let data: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
        panic!("SOMETHING IS WRONG HERE");
        //assert_deriv(simple_function, simple_gradient, data, 0.01, 1e-10);
    }
}