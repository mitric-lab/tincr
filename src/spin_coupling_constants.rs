use crate::molecule::{distance_matrix, Molecule};
use approx::AbsDiffEq;
use ndarray::{array, Array1, Array2, Array3, ArrayView2};
use std::collections::HashMap;
use std::f64::consts::PI;

/// NOTE: The spin-coupling constants were taken from
/// https://dftb.org/fileadmin/DFTB/public/slako/mio/mio-1-1.spinw.txt
/// However, the coupling constants are only tabulated there on an angular momentum level
/// and we use only one spin-coupling constant per element type. Therefore, the average was
/// used in the confined_pseudo_atom parameter files
pub fn spin_coupling_matrix(atomic_numbers: &[u8], n_atoms: usize) -> Array1<f64> {



}