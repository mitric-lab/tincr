use ndarray::{ArrayView3, ArrayView2, ArrayView1, Array3, Array1, Array2};
use crate::excited_states::a_and_b::{build_a_matrix, build_b_matrix};
use ndarray_linalg::{SymmetricSqrt, Eigh, Solve, UPLO, Norm};
use crate::utils::ToOwnedF;

pub fn casida(
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
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    let A: Array2<f64> = build_a_matrix(
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
    let B: Array2<f64> = build_b_matrix(
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
    //check whether A - B is diagonal
    let AmB: Array2<f64> = &A - &B;
    let ApB: Array2<f64> = &A + &B;
    //let n_occ: usize = q_trans_oo.dim().1;
    //let n_virt: usize = q_trans_vv.dim().1;
    let mut sqAmB: Array2<f64> = Array2::zeros((n_occ * n_virt, n_occ * n_virt));
    let offdiag: f64 = (Array2::from_diag(&AmB.diag()) - &AmB).norm();
    if offdiag < 1.0e-10 {
        // calculate the sqareroot of the diagonal and transform to 2d matrix
        sqAmB = Array2::from_diag(&AmB.diag().mapv(f64::sqrt));
    } else {
        // calculate matrix squareroot
        sqAmB = AmB.ssqrt(UPLO::Upper).unwrap();
    }

    // construct hermitian eigenvalue problem
    // (A-B)^(1/2) (A+B) (A-B)^(1/2) F = Omega^2 F
    let R: Array2<f64> = sqAmB.dot(&ApB.dot(&sqAmB));
    let (omega2, F): (Array1<f64>, Array2<f64>) = R.eigh(UPLO::Lower).unwrap();

    let omega: Array1<f64> = omega2.mapv(f64::sqrt);
    //let omega: Array1<f64> = omega2.map(|omega2| ndarray_linalg::Scalar::sqrt(omega2));

    // compute X-Y and X+Y
    // X+Y = 1/sqrt(Omega) * (A-B)^(1/2).F
    // X-Y = 1/Omega * (A+B).(X+Y)

    let XpY: Array2<f64> = &sqAmB.dot(&F) / &omega.mapv(f64::sqrt);
    let XmY: Array2<f64> = &ApB.dot(&XpY) / &omega;

    //assert!((XpY.slice(s![..,0]).to_owned()*XmY.slice(s![..,0]).to_owned()).sum().abs()<1.0e-10);
    //assert!((ApB.dot(&XpY)-omega*XmY).abs().sum() < 1.0e-5);

    //C = (A-B)^(-1/2).((X+Y) * sqrt(Omega))
    // so that C^T.C = (X+Y)^T.(A-B)^(-1).(X+Y) * Omega
    //               = (X+Y)^T.(X-Y)
    // since (A-B).(X-Y) = Omega * (X+Y)
    let temp = &XpY * &omega.mapv(f64::sqrt);
    let mut c_matrix: Array2<f64> = Array2::zeros((omega.len(), omega.len()));
    for i in 0..(omega.len()) {
        c_matrix
            .slice_mut(s![i, ..])
            .assign((&sqAmB.solve(&temp.slice(s![.., i])).unwrap()));
    }
    c_matrix = c_matrix.reversed_axes();
    assert!(
        (((&c_matrix.slice(s![.., 0]).to_owned() * &c_matrix.slice(s![.., 0])).to_owned())
            .sum()
            .abs()
            - 1.0)
            < 1.0e-10
    );

    let XmY_final: Array3<f64> = XmY
        .to_owned_f()
        .t()
        .into_shape((n_occ * n_virt, n_occ, n_virt))
        .unwrap()
        .to_owned();
    let XpY_final: Array3<f64> = XpY
        .to_owned_f()
        .t()
        .into_shape((n_occ * n_virt, n_occ, n_virt))
        .unwrap()
        .to_owned();

    let c_matrix_transformed: Array3<f64> = c_matrix
        .reversed_axes()
        .into_shape((n_occ * n_virt, n_occ, n_virt))
        .unwrap();

    return (omega, c_matrix_transformed, XmY_final, XpY_final);
}