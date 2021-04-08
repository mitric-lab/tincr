use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2};

//////////////////////////////////////////////////
// Fortran routines
//////////////////////////////////////////////////

/// Returns tsource if mask is true, elsewise fsource.
/// Both sources should be isize. (MERGE)
pub fn mergei(tsource: isize, fsource: isize, mask: bool) -> isize {
    if mask {
        tsource
    } else {
        fsource
    }
}

/// Returns the index of minimum value and its value
/// of an 1D-Array. (MINLOC)
pub fn argminf(a: ArrayView1<f64>) -> (Option<usize>, Option<f64>) {
    if a.dim() == 0 {
        return (None, None);
    } else {
        let (min_idx, min_val) = a.iter().enumerate()
            .fold((0, a[0]), |(idx_min, val_min), (idx, val)| {
                if &val_min < val {
                    (idx_min, val_min)
                } else {
                    (idx, *val)
                }
            });

        return (Some(min_idx), Some(min_val));
    }
}

/// Replicates a source array copies times along a specified dimension axis. (SPREAD)
pub fn spread(source: &Array1<f64>, axis: u8, copies: usize) -> Array2<f64> {
    let mut a: Array2<f64> = Array::zeros((copies, source.len()));
    for i in 0..copies {
        a.slice_mut(s![i, ..]).assign(&source);
    }
    if axis == 0 {
        a.reversed_axes()
    } else {
        a
    }
}

//////////////////////////////////////////////////
// Linear algebra routines
//////////////////////////////////////////////////

/// Perform cross product of two 3D vectors.
pub fn crossproduct(
    a: &[f64; 3],
    b: &[f64; 3]
) -> [f64; 3] {
    let mut c: [f64; 3] = [0.0; 3];
    c[0] = a[1]*b[2] - b[1]*a[2];
    c[1] = a[2]*b[0] - b[2]*a[0];
    c[2] = a[0]*b[1] - b[0]*a[1];
    return c;
}

/// Return the dot product of two rust arrays.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

/// Return the 2 norm of a rust array.
pub fn norm2(a: &[f64]) -> f64 {
    a.iter().map(|a| a.powi(2)).sum::<f64>().sqrt()
}




//////////////////////////////////////////////////
// Rust specific routines
//////////////////////////////////////////////////
/// Convert a lattice as a list of lists into Array2
pub fn lat_to_array(
    lat: &[[f64; 3]; 3],
) -> Array2<f64> {
    let mut lat_a: Array2<f64> = Array::zeros((3, 3));
    for i in 0..3 {
        lat_a.slice_mut(s![i, ..]).assign(&Array::from((&lat[i]).to_vec()));
    }

    return lat_a;
}