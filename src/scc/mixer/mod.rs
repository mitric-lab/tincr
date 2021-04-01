use ndarray::Array1;

pub mod broyden;
pub use broyden::BroydenMixer;

/// Trait that allows mixing of partial charge differences for the acceleration
/// of the SCC routine
pub trait Mixer {
    fn mix(&mut self, q_inp: Array1<f64>, q_diff: Array1<f64>) -> Array1<f64>;
}


