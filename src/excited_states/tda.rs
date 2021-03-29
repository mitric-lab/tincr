use ndarray::{ArrayView2, ArrayView3, ArrayView1, Array1, Array3, Array2};
use crate::excited_states::a_and_b::build_a_matrix;
use ndarray_linalg::{Eigh, UPLO};

pub fn tda(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
    n_occ: usize,
    n_virt: usize,
    spin_couplings: ArrayView1<f64>,
) -> (Array1<f64>, Array3<f64>) {
    let h_tda: Array2<f64> = build_a_matrix(
        gamma,
        gamma_lr,
        q_trans_ov,
        q_trans_oo,
        q_trans_vv,
        omega,
        df,
        multiplicity,
        spin_couplings,
    );
    // diagonalize TDA Hamiltonian
    let (omega, x): (Array1<f64>, Array2<f64>) = h_tda.eigh(UPLO::Upper).unwrap();
    let c_ij: Array3<f64> = x
        .reversed_axes()
        .into_shape((n_occ * n_virt, n_occ, n_virt))
        .unwrap();
    // println!("{}", c_ij);
    return (omega, c_ij);
}