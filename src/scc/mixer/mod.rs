use ndarray::Array1;

pub mod broyden;
pub mod anderson;

pub use broyden::BroydenMixer;
pub use anderson::*;

/// Trait that allows mixing of partial charge differences for the acceleration
/// of the SCC routine
pub trait Mixer {
    fn new(n_atoms: usize) -> Self;
    fn mix(&mut self, q_inp: Array1<f64>, q_diff: Array1<f64>) -> Array1<f64>;
    fn next(&mut self, q_inp: Array1<f64>, q_diff: Array1<f64>) -> Array1<f64>;
    fn reset(&mut self, n_atoms: usize);
}


