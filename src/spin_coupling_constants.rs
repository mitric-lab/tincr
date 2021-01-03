use crate::molecule::{distance_matrix, Molecule};
use approx::AbsDiffEq;
use ndarray::{array, Array, Array1, Array2, ArrayView2};
use serde_json::value::Value::Array;
use std::collections::HashMap;
use std::f64::consts::PI;

/// NOTE: The spin-coupling constants were taken from
/// https://dftb.org/fileadmin/DFTB/public/slako/mio/mio-1-1.spinw.txt
/// However, the coupling constants are only tabulated there on an angular momentum level
/// and we use only one spin-coupling constant per element type. Therefore, the average was
/// used in the confined_pseudo_atom parameter files
pub fn get_spin_coupling(
    atomic_numbers: &[u8],
    n_atoms: usize,
    spin_couplings: HashMap<u8, f64>,
) -> Array1<f64> {
    let spin_coupling_vec: Array1<f64> =
        Array::from(atomic_numbers.iter().map(|x| spin_couplings[x]).collect());

}
