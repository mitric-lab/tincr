use ndarray::prelude::*;
use ndarray::Order;

/// The differences between the virtual and occupied orbitals are computed. The quantity to be
/// computed can be either the energies of the orbitals sets or e.g. the occupation. The length
/// of the output Array will be the `len(occ_quant) x len(virt_quant)`.
pub fn orbe_differences(occ_quant: ArrayView1<f64>, virt_quant: ArrayView1<f64>) -> Array1<f64> {
    // Number of occupied orbitals.
    let n_occ: usize = occ_quant.len();
    // Number of virtual orbitals.
    let n_virt: usize = virt_quant.len();
    // Compute the distance matrix by broadcasting the energies of the occupied orbitals to
    // 2D array of the shape n_occ x n_virt and subtract the virtual energies.
    (&virt_quant
        - &occ_quant
            .to_owned()
            .insert_axis(Axis(1))
            .broadcast((n_occ, n_virt))
            .unwrap())
    .to_shape(((n_occ * n_virt), Order::RowMajor))
    .unwrap()
        .to_owned()
}

