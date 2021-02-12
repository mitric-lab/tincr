use ndarray::prelude::*;
use ndarray::{Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use std::ops::AddAssign;
use approx::AbsDiffEq;

pub fn build_a_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
) -> (Array2<f64>) {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;
    let mut k_lr_a: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
    let mut k_a: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
    let mut k_singlet: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
    // K_lr_A = np.tensordot(qtrans_oo, np.tensordot(gamma_lr, qtrans_vv, axes=(1,0)),axes=(0,0))
    k_lr_a = tensordot(
        &q_trans_oo,
        &tensordot(&gamma_lr, &q_trans_vv, &[Axis(1)], &[Axis(0)]),
        &[Axis(0)],
        &[Axis(0)],
    )
        .into_dimensionality::<Ix4>()
        .unwrap();
    // K_lr_A = np.swapaxes(K_lr_A, 1, 2)
    // swap axes still missing
    k_lr_a.swap_axes(1, 2);
    k_a.assign(&-k_lr_a);

    if multiplicity == 1 {
        //K_singlet = 2.0*np.tensordot(qtrans_ov, np.tensordot(gamma, qtrans_ov, axes=(1,0)),axes=(0,0))
        //K_A += K_singlet
        k_singlet = 2.0
            * tensordot(
            &q_trans_ov,
            &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]),
            &[Axis(0)],
            &[Axis(0)],
        )
            .into_dimensionality::<Ix4>()
            .unwrap();
        k_a = k_a + k_singlet;
    }
    let mut k_coupling: Array2<f64> = k_a.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(&omega) + &df_half.dot(&k_coupling.dot(&df_half));
}

pub fn build_b_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
) -> (Array2<f64>) {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;
    let mut k_lr_b: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
    let mut k_b: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
    let mut k_singlet: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
    //K_lr_B = np.tensordot(qtrans_ov, np.tensordot(gamma_lr, qtrans_ov, axes=(1,0)),axes=(0,0))
    k_lr_b = tensordot(
        &q_trans_ov,
        &tensordot(&gamma_lr, &q_trans_ov, &[Axis(1)], &[Axis(0)]),
        &[Axis(0)],
        &[Axis(0)],
    )
        .into_dimensionality::<Ix4>()
        .unwrap();
    //# got K_ia_jb but we need K_ib_ja
    //K_lr_B = np.swapaxes(K_lr_B, 1, 3)
    k_lr_b.swap_axes(1, 3);
    k_b.assign(&-k_lr_b);

    if multiplicity == 1 {
        //K_singlet = 2.0*np.tensordot(qtrans_ov, np.tensordot(gamma, qtrans_ov, axes=(1,0)),axes=(0,0))
        //K_A += K_singlet
        k_singlet = 2.0
            * tensordot(
            &q_trans_ov,
            &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]),
            &[Axis(0)],
            &[Axis(0)],
        )
            .into_dimensionality::<Ix4>()
            .unwrap();
        k_b = k_b + k_singlet;
    }
    let mut k_coupling: Array2<f64> = k_b.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(&k_coupling.dot(&df_half));
}

fn get_orbital_en_diff(
    orbe: ArrayView1<f64>,
    n_occ: usize,
    n_virt: usize,
    active_occupied_orbs: &[usize],
    active_virtual_orbs: &[usize],
) -> Array2<f64> {
    // energy difference between occupied and virtual Kohn-Sham orbitals
    // omega_ia = omega_a - omega_i
    let mut omega: Array2<f64> = Array2::zeros([n_occ, n_virt]);
    for (i, occ_i) in active_occupied_orbs.iter().enumerate() {
        for (a, virt_a) in active_virtual_orbs.iter().enumerate() {
            omega[[i, a]] = orbe[*virt_a] - orbe[*occ_i];
        }
    }
    return omega;
}

fn get_orbital_occ_diff(
    f: ArrayView1<f64>,
    n_occ: usize,
    n_virt: usize,
    active_occupied_orbs: &[usize],
    active_virtual_orbs: &[usize],
) -> Array2<f64> {
    // occupation difference between occupied and virtual Kohn-Sham orbitals
    // f_ia = f_a - f_i
    let mut df: Array2<f64> = Array2::zeros([n_occ, n_virt]);
    for (i, occ_i) in active_occupied_orbs.iter().enumerate() {
        for (a, virt_a) in active_virtual_orbs.iter().enumerate() {
            df[[i, a]] = f[*virt_a] - f[*occ_i];
        }
    }
    return df;
}

fn tda(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
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
    );
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;
    // diagonalize TDA Hamiltonian
    let (omega, x): (Array1<f64>, Array2<f64>) = h_tda.eigh(UPLO::Upper).unwrap();
    let c_ij: Array3<f64> = x
        .reversed_axes()
        .into_shape((n_occ * n_virt, n_occ, n_virt))
        .unwrap();
    return (omega, c_ij);
}

fn casida(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
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
    );
    //check whether A - B is diagonal
    let AmB: Array2<f64> = &A - &B;
    let ApB: Array2<f64> = &A + &B;
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;
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
    let (omega2, F): (Array1<f64>, Array2<f64>) = R.eigh(UPLO::Upper).unwrap();
    let omega: Array1<f64> = omega2.mapv(f64::sqrt);

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
            .slice_mut(s![.., i])
            .assign((&sqAmB.solve(&temp.slice(s![.., i])).unwrap()));
    }
    assert!(
        ((&c_matrix.slice(s![.., 0]).to_owned() * &c_matrix.slice(s![.., 0])).to_owned())
            .sum()
            .abs()
            < 1.0e-10
    );

    let XmY_transformed: Array3<f64> = XmY.into_shape((n_occ * n_virt, n_occ, n_virt)).unwrap();
    let XpY_transformed: Array3<f64> = XpY.into_shape((n_occ * n_virt, n_occ, n_virt)).unwrap();

    let c_matrix_transformed: Array3<f64> = c_matrix
        .reversed_axes()
        .into_shape((n_occ * n_virt, n_occ, n_virt))
        .unwrap();

    return (
        omega,
        c_matrix_transformed,
        XmY_transformed,
        XpY_transformed,
    );
}

fn hermitian_davidson(
    gamma: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    omega_shift: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
    XmYguess: Option<ArrayView2<f64>>,
    XpYguess: Option<ArrayView2<f64>>,
    Oia: ArrayView2<f64>,
    multiplicity: usize,
) -> () {
    // f A-B is diagonal the TD-DFT equations can be made hermitian
    //       (A-B)^(1/2).(A+B).(A-B)^(1/2).T = Omega^2 T
    //                    R               .T = Omega^2 T

    let nstates: usize = 4;
    let ifact: usize = 1;
    let maxiter: u8 = 10;
    let conv: f64 = 1.0e-14;
    let l2_treshold: f64 = 0.5;

    let omega2: Array2<f64> = omega.map(|omega| ndarray_linalg::Scalar::powi(omega, 2));
    let omega_sq: Array2<f64> = omega.map(|omega| ndarray_linalg::Scalar::sqrt(omega));
    let omega_sq_inv: Array2<f64> = 1.0 / &omega_sq;
    let wq_ov = &qtrans_ov * &omega_sq;
    //# diagonal elements of R
    let om: Array2<f64> = omega2 + &omega * &omega_shift * 2.0;

    // initial number of expansion vectors
    // at most there are nocc*nvirt excited states
    let kmax = &n_occ * &n_virt;
    let lmax = (&ifact * &nstates).min(kmax);

    let mut bs:Array3<f64> = Array::zeros((n_occ,n_virt,lmax));
    if XpYguess.contains() == false{
        let omega_guess:Array2<f64> = om.map(|om| ndarray_linalg::Scalar::sqrt(om));
        // new function to calculate bs
    }
}

#[test]
fn tda_routine() {
    let orbe: Array1<f64> = array![
        -0.8688870777877312,
        -0.4499943390169377,
        -0.3563252311271602,
        -0.2832985695381462,
        0.3766573907852607,
        0.4290409093390336
    ];
    let active_occupied_orbs: Vec<usize> = vec![2, 3];
    let active_virtual_orbs: Vec<usize> = vec![4, 5];
    let gamma: Array2<f64> = array![
        [0.4467609798860577, 0.3863557889890281, 0.3863561531176491],
        [0.3863557889890281, 0.4720158398964135, 0.3084885848056254],
        [0.3863561531176491, 0.3084885848056254, 0.4720158398964135]
    ];
    let gamma_lr: Array2<f64> = array![
        [0.2860554418243039, 0.2692279296946004, 0.2692280400920803],
        [0.2692279296946004, 0.2923649998054588, 0.2429686492032624],
        [0.2692280400920803, 0.2429686492032624, 0.2923649998054588]
    ];
    let q_trans_ov: Array3<f64> = array![
        [
            [2.6230764031964782e-05, 3.7065733463488038e-01],
            [-4.9209998651226938e-17, 2.3971084358783751e-16]
        ],
        [
            [-1.7348142939318700e-01, -1.8531691862558541e-01],
            [-7.2728474862656226e-17, -7.7779165808212125e-17]
        ],
        [
            [1.7345519862915512e-01, -1.8534041600929513e-01],
            [1.5456547682172723e-16, -1.6527399530138889e-16]
        ]
    ];
    let q_trans_oo: Array3<f64> = array![
        [
            [8.3509500972984507e-01, -3.0814858028948981e-16],
            [-3.0814858028948981e-16, 9.9999999999999978e-01]
        ],
        [
            [8.2452864978581231e-02, 3.8129127163009314e-17],
            [3.8129127163009314e-17, 1.6846288898245608e-32]
        ],
        [
            [8.2452125291573627e-02, 7.8185267908421217e-17],
            [7.8185267908421217e-17, 7.2763969108729995e-32]
        ]
    ];
    let q_trans_vv: Array3<f64> = array![
        [
            [4.1303771372197096e-01, -5.9782394554452889e-06],
            [-5.9782394554452889e-06, 3.2642696006563388e-01]
        ],
        [
            [2.9352476622180407e-01, 3.1439790351905961e-01],
            [3.1439790351905961e-01, 3.3674286510673440e-01]
        ],
        [
            [2.9343752005622487e-01, -3.1439192527960413e-01],
            [-3.1439192527960413e-01, 3.3683017482763289e-01]
        ]
    ];
    let omega_0: Array2<f64> = array![
        [0.7329826219124209, 0.7853661404661938],
        [0.6599559603234070, 0.7123394788771799]
    ];
    let df: Array2<f64> = array![[2., 2.], [2., 2.]];
    let omega_ref: Array1<f64> = array![0.3837776010960228, 0.4376185583677501, 0.4777844855653459,
 0.529392732956824];
    let c_ij_ref: Array3<f64> = array![
    [[ 7.1048609280539423e-16, -1.2491679807793276e-17],
     [-9.9999999999684230e-01,  2.5130863202938214e-06]],
    [[ 1.0807435052922551e-16, -7.0715300399956809e-16],
     [ 2.5130863202798792e-06,  9.9999999999684219e-01]],
    [[-9.9999999999915401e-01,  1.3008338757459926e-06],
     [-7.1048583744966613e-16,  1.0807705593029713e-16]],
    [[-1.3008338757459926e-06, -9.9999999999915401e-01],
     [ 1.2488978235912678e-17, -7.0715289477295315e-16]]];

    let (omega, c_ij): (Array1<f64>, Array3<f64>) = tda(
        gamma.view(),
        gamma_lr.view(),
        q_trans_ov.view(),
        q_trans_oo.view(),
        q_trans_vv.view(),
        omega_0.view(),
        df.view(),
        1,
    );
    println!("omega {}", omega);
    println!("omega_ref {}", omega_ref);
    assert!(omega.abs_diff_eq(&omega_ref, 1e-14));
    assert!(c_ij.abs_diff_eq(&c_ij_ref, 1e-14));

}
