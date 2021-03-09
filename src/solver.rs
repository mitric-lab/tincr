use crate::calculator::{get_gamma_gradient_matrix, lambda2_calc_oia};
use crate::gamma_approximation::m_atomwise;
use crate::defaults;
use crate::gradients;
use crate::transition_charges::trans_charges;
use crate::Molecule;
use approx::AbsDiffEq;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{Array2, Array4, Array6, ArrayView1, ArrayView2, ArrayView3, stack};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use std::cmp::Ordering;
use std::ops::AddAssign;
use crate::scc_routine::*;
use std::time::Instant;

pub trait ToOwnedF<A, D> {
    fn to_owned_f(&self) -> Array<A, D>;
}
impl<A, S, D> ToOwnedF<A, D> for ArrayBase<S, D>
where
    A: Copy + Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn to_owned_f(&self) -> Array<A, D> {
        let mut tmp = unsafe { Array::uninitialized(self.dim().f()) };
        tmp.assign(self);
        tmp
    }
}

pub fn get_exc_energies(
    f_occ: &Vec<f64>,
    molecule: &Molecule,
    nstates: Option<usize>,
    s: &Array2<f64>,
    orbe: &Array1<f64>,
    orbs: &Array2<f64>,
    magnetic_correction: bool,
    response_method: Option<String>,
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    // set active orbitals first
    let active_occ: Vec<usize> = molecule.calculator.active_occ.clone().unwrap();
    let active_virt: Vec<usize> = molecule.calculator.active_virt.clone().unwrap();
    let n_occ: usize = active_occ.len();
    let n_virt: usize = active_virt.len();

    let n_at: usize = molecule.n_atoms;

    // set transtition charges of the active space
    let (qtrans_ov, qtrans_oo, qtrans_vv): (Array3<f64>, Array3<f64>, Array3<f64>) = trans_charges(
        &molecule.atomic_numbers,
        &molecule.calculator.valorbs,
        orbs.view(),
        s.view(),
        &active_occ[..],
        &active_virt[..],
    );

    // check for selected number of excited states
    let mut complete_states: bool = false;
    let mut nstates: usize = nstates.unwrap_or(defaults::EXCITED_STATES);
    // if number is bigger than theoretically possible, set the number to the
    // complete range of states. Thus, casida is required.
    if nstates > (n_occ * n_virt) {
        nstates = n_occ * n_virt;
        complete_states = true;
    }

    // get gamma matrices
    let mut gamma0_lr: Array2<f64> = Array::zeros((molecule.calculator.g0.shape()))
        .into_dimensionality::<Ix2>()
        .unwrap();

    let r_lr = molecule
        .calculator
        .r_lr
        .unwrap_or(defaults::LONG_RANGE_RADIUS);

    if r_lr == 0.0 {
        gamma0_lr = &molecule.calculator.g0 * 0.0;
    } else {
        gamma0_lr = molecule.calculator.g0_lr.clone();
    }
    // get omega
    // omega_ia = en_a - en_i
    // energy differences between occupied and virtual Kohn-Sham orbitals
    let omega: Array2<f64> = get_orbital_en_diff(
        orbe.view(),
        n_occ,
        n_virt,
        &active_occ[..],
        &active_virt[..],
    );
    // get df
    // occupation differences between occupied and virtual Kohn-Sham orbitals
    let df: Array2<f64> = get_orbital_occ_diff(
        Array::from_vec(f_occ.clone()).view(),
        n_occ,
        n_virt,
        &active_occ[..],
        &active_virt[..],
    );

    // Get M matrix if required
    let m: Array4<f64> = if magnetic_correction {
        m_atomwise(&molecule.atomic_numbers[..], molecule.n_atoms,
                   molecule.calculator.spin_couplings.view())
    } else {
        Array::zeros((molecule.n_atoms, molecule.n_atoms, 2, 2))
    };


    let mut omega_out: Array1<f64> = Array::zeros((n_occ * n_virt));
    let mut c_ij: Array3<f64> = Array::zeros((n_occ * n_virt, n_occ, n_virt));
    let mut XpY: Array3<f64> = Array::zeros((n_occ * n_virt, n_occ, n_virt));
    let mut XmY: Array3<f64> = Array::zeros((n_occ * n_virt, n_occ, n_virt));

    // check if complete nstates is used
    if complete_states {
        // check if Tamm-Dancoff is demanded
        if response_method.is_some() && response_method.unwrap() == "TDA" {
            let tmp: (Array1<f64>, Array3<f64>) = tda(
                (&molecule.calculator.g0).view(),
                gamma0_lr.view(),
                qtrans_ov.view(),
                qtrans_oo.view(),
                qtrans_vv.view(),
                omega.view(),
                df.view(),
                molecule.multiplicity.clone(),
                n_occ,
                n_virt,
                Some(m.view()),
            );
            omega_out = tmp.0;
            c_ij = tmp.1;
        } else {
            let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = casida(
                (&molecule.calculator.g0).view(),
                gamma0_lr.view(),
                qtrans_ov.view(),
                qtrans_oo.view(),
                qtrans_vv.view(),
                omega.view(),
                df.view(),
                molecule.multiplicity.clone(),
                n_occ,
                n_virt,
                Some(m.view()),
            );
            omega_out = tmp.0;
            c_ij = tmp.1;
            XmY = tmp.2;
            XpY = tmp.3;
        }
    } else {
        // solve the eigenvalue problem
        // for nstates using an iterative
        // approach

        //check for lr_correction
        if r_lr == 0.0 {
            println!("Hermitian Davidson");
            // calculate o_ia
            let o_ia: Array2<f64> =
                lambda2_calc_oia(molecule, &active_occ, &active_virt, &qtrans_oo, &qtrans_vv);
            // use hermitian davidson routine, only possible with lc off
            let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = hermitian_davidson(
                (&molecule.calculator.g0).view(),
                qtrans_ov.view(),
                omega.view(),
                (0.0 * &omega).view(),
                n_occ,
                n_virt,
                None,
                None,
                o_ia.view(),
                molecule.multiplicity.clone() as usize,
                Some(nstates),
                None,
                None,
                None,
                None,
            );
            omega_out = tmp.0;
            c_ij = tmp.1;
            XmY = tmp.2;
            XpY = tmp.3;
        } else {
            let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = non_hermitian_davidson(
                (&molecule.calculator.g0).view(),
                (&molecule.calculator.g0_lr).view(),
                qtrans_oo.view(),
                qtrans_vv.view(),
                qtrans_ov.view(),
                omega.view(),
                n_occ,
                n_virt,
                None,
                None,
                None,
                molecule.multiplicity.clone() as usize,
                Some(nstates),
                None,
                None,
                None,
                None,
                Some(1),
            );
            omega_out = tmp.0;
            c_ij = tmp.1;
            XmY = tmp.2;
            XpY = tmp.3;
        }
    }
    return (omega_out, c_ij, XmY, XpY);
}

pub fn argsort(v: ArrayView1<f64>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(Ordering::Equal));
    idx
}

pub fn build_a_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
    m: Option<ArrayView4<f64>>,
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
        // println!("k_singlet_NO_M {}", k_singlet);

        k_a = k_a + k_singlet;
    }
    let mut k_coupling: Array2<f64> = k_a.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();

    // println!("k_coupling_no_M {}", k_coupling);

    let k_m_coupling_red: Array2<f64> = if m.is_some() {
        // Manipulate q_trans_ov
        let mut q_trans_ovs: Array4<f64> = q_trans_ov.to_owned().insert_axis(Axis(3));
        q_trans_ovs = stack![Axis(3), q_trans_ovs, q_trans_ovs];

        // Get the correction matrix K_m
        let delta_st: Array2<f64> = Array::eye(2);
        let k_m: Array6<f64> = 2.0 * einsum("aijs,bklt,abst,st->sijtkl", &[&q_trans_ovs, &q_trans_ovs, &m.unwrap(), &delta_st])
            .unwrap().into_dimensionality::<Ix6>().unwrap().as_standard_layout().to_owned();

        let k_m_coupling: Array2<f64> = k_m.into_shape((2 * n_occ * n_virt, 2 * n_occ * n_virt)).unwrap();
        // println!("K_m_coupling {}", k_m_coupling);
        // assert!(k_m_coupling.abs_diff_eq(&k_m_coupling_2, 1e-16));

        k_m_coupling.slice(s![0..n_occ*n_virt, 0..n_occ*n_virt]).to_owned()
    } else {
        Array::zeros(k_coupling.raw_dim())
    };

    k_coupling = k_coupling + k_m_coupling_red;

    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(&omega) + &df_half.dot(&k_coupling.dot(&df_half));
}

// pub fn build_a_matrix_spin(
//     gamma: ArrayView2<f64>,
//     gamma_lr: ArrayView2<f64>,
//     m: ArrayView4<f64>,
//     q_trans_ov: ArrayView3<f64>,
//     q_trans_oo: ArrayView3<f64>,
//     q_trans_vv: ArrayView3<f64>,
//     omega: ArrayView2<f64>,
//     df: ArrayView2<f64>,
//     multiplicity: u8,
// ) -> (Array2<f64>) {
//     let n_occ: usize = q_trans_oo.dim().1;
//     let n_virt: usize = q_trans_vv.dim().1;
//     let n_atom: usize = q_trans_oo.dim().0;
//
//     let mut k_lr_a: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     let mut k_a: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     let mut k_singlet: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     // K_lr_A = np.tensordot(qtrans_oo, np.tensordot(gamma_lr, qtrans_vv, axes=(1,0)),axes=(0,0))
//     k_lr_a = tensordot(
//         &q_trans_oo,
//         &tensordot(&gamma_lr, &q_trans_vv, &[Axis(1)], &[Axis(0)]),
//         &[Axis(0)],
//         &[Axis(0)],
//     )
//         .into_dimensionality::<Ix4>()
//         .unwrap();
//
//     // K_lr_A = np.swapaxes(K_lr_A, 1, 2)
//     k_lr_a.swap_axes(1, 2);
//     k_a.assign(&-k_lr_a);
//
//     if multiplicity == 1 {
//         //K_singlet = 2.0*np.tensordot(qtrans_ov, np.tensordot(gamma, qtrans_ov, axes=(1,0)),axes=(0,0))
//         //K_A += K_singlet
//         k_singlet = 2.0
//             * tensordot(
//             &q_trans_ov,
//             &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]),
//             &[Axis(0)],
//             &[Axis(0)],
//         )
//             .into_dimensionality::<Ix4>()
//             .unwrap();
//
//         k_a = k_a + k_singlet;
//     }
//     let mut k_coupling: Array2<f64> = k_a.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
//
//     // Manipulate q_trans_ov
//     let mut q_trans_ovs: Array4<f64> = q_trans_ov.to_owned().insert_axis(Axis(3));
//     q_trans_ovs = stack![Axis(3), q_trans_ovs, q_trans_ovs];
//
//     // Get the correction matrix K_m
//     let delta_st: Array2<f64> = Array::eye(2);
//     let k_m: Array6<f64> = 2.0 * einsum("aijs,bklt,abst,st->sijtkl", &[&q_trans_ovs, &q_trans_ovs, &m, &delta_st])
//         .unwrap().into_dimensionality::<Ix6>().unwrap().as_standard_layout().to_owned();
//
//     let k_m_coupling: Array2<f64> = k_m.into_shape((2 * n_occ * n_virt, 2 * n_occ * n_virt)).unwrap();
//     // println!("K_m_coupling {}", k_m_coupling);
//     // assert!(k_m_coupling.abs_diff_eq(&k_m_coupling_2, 1e-16));
//
//     let k_m_coupling_red: Array2<f64> = k_m_coupling.slice(s![0..n_occ*n_virt, 0..n_occ*n_virt]).to_owned();
//
//     // // Spin block k_coupling to match the shape of k_m
//     // let null_mat: Array2<f64> = Array::zeros(k_coupling.raw_dim());
//     // k_coupling = stack![Axis(0), stack![Axis(1), k_coupling, null_mat], stack![Axis(1), null_mat, k_coupling]];
//     // // println!("K_spin {}", k_coupling);
//
//     k_coupling = k_coupling + k_m_coupling_red;
//
//     // // spin block df and omega
//     // let mut df_block: Array1<f64> = df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap();
//     // df_block = stack![Axis(0), df_block, df_block];
//     // let df_half: Array2<f64> = Array2::from_diag(&df_block);
//     // let mut omega_block: Array1<f64> = omega.into_shape((n_occ * n_virt)).unwrap().to_owned();
//     // omega_block = stack![Axis(0), omega_block, omega_block];
//     // let omega: Array2<f64> = Array2::from_diag(&omega_block);
//
//     let mut df_half: Array2<f64> =
//         Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
//     let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
//     return df_half.dot(&omega) + &df_half.dot(&k_coupling.dot(&df_half));
//
//     return &df_half.dot(&omega) + &df_half.dot(&k_coupling.dot(&df_half));
// }


pub fn build_b_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
    m: Option<ArrayView4<f64>>,
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
    k_b.assign(&(-1.0 * k_lr_b));

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

    // Build m matrix if neccesary
    let k_m_coupling_red: Array2<f64> = if m.is_some() {
        // Manipulate q_trans_ov
        let mut q_trans_ovs: Array4<f64> = q_trans_ov.to_owned().insert_axis(Axis(3));
        q_trans_ovs = stack![Axis(3), q_trans_ovs, q_trans_ovs];

        // Get the correction matrix K_m
        let delta_st: Array2<f64> = Array::eye(2);
        let k_m: Array6<f64> = 2.0 * einsum("aijs,bklt,abst,st->sijtlk", &[&q_trans_ovs, &q_trans_ovs, &m.unwrap(), &delta_st])
            .unwrap().into_dimensionality::<Ix6>().unwrap().as_standard_layout().to_owned();

        let k_m_coupling: Array2<f64> = k_m.into_shape((2 * n_occ * n_virt, 2 * n_occ * n_virt)).unwrap();
        // println!("K_m_coupling {}", k_m_coupling);
        // assert!(k_m_coupling.abs_diff_eq(&k_m_coupling_2, 1e-16));

        k_m_coupling.slice(s![0..n_occ*n_virt, 0..n_occ*n_virt]).to_owned()
    } else {
        Array::zeros(k_coupling.raw_dim())
    };

    k_coupling = k_coupling + k_m_coupling_red;

    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(&k_coupling.dot(&df_half));
}

pub fn get_orbital_en_diff(
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

pub fn get_orbital_occ_diff(
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
            df[[i, a]] = f[*occ_i] - f[*virt_a];
        }
    }
    return df;
}

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
    m: Option<ArrayView4<f64>>,
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
        m,
    );
    // diagonalize TDA Hamiltonian
    let (omega, x): (Array1<f64>, Array2<f64>) = h_tda.eigh(UPLO::Upper).unwrap();
    let c_ij: Array3<f64> = x
        .reversed_axes()
        .into_shape((n_occ * n_virt, n_occ, n_virt))
        .unwrap();
    println!("{}", c_ij);
    return (omega, c_ij);
}

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
    m: Option<ArrayView4<f64>>,
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
        m,
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
        m,
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

pub fn hermitian_davidson(
    gamma: ArrayView2<f64>,
    qtrans_ov: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    omega_shift: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
    XmYguess: Option<ArrayView3<f64>>,
    XpYguess: Option<ArrayView3<f64>>,
    Oia: ArrayView2<f64>,
    multiplicity: usize,
    nstates: Option<usize>,
    ifact: Option<usize>,
    maxiter: Option<usize>,
    conv: Option<f64>,
    l2_treshold: Option<f64>,
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    // f A-B is diagonal the TD-DFT equations can be made hermitian
    //       (A-B)^(1/2).(A+B).(A-B)^(1/2).T = Omega^2 T
    //                    R               .T = Omega^2 T

    let nstates: usize = nstates.unwrap_or(4);
    let ifact: usize = ifact.unwrap_or(1);
    let maxiter: usize = maxiter.unwrap_or(10);
    let conv: f64 = conv.unwrap_or(1.0e-14);
    let l2_treshold: f64 = l2_treshold.unwrap_or(0.5);

    let omega2: Array2<f64> = omega.map(|omega| ndarray_linalg::Scalar::powi(omega, 2));
    let omega_sq: Array2<f64> = omega.map(|omega| ndarray_linalg::Scalar::sqrt(omega));
    let omega_sq_inv: Array2<f64> = 1.0 / &omega_sq;
    let wq_ov: Array3<f64> = &qtrans_ov * &omega_sq;
    //# diagonal elements of R
    let om: Array2<f64> = omega2 + &omega * &omega_shift * 2.0;
    // initial number of expansion vectors
    // at most there are nocc*nvirt excited states
    let kmax = &n_occ * &n_virt;
    let lmax = (&ifact * &nstates).min(kmax);

    let mut bs: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    let mut bs_first: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    if XpYguess.is_none() {
        let omega_guess: Array2<f64> = om.map(|om| ndarray_linalg::Scalar::sqrt(om));
        bs = initial_expansion_vectors(omega_guess.clone(), lmax);
    } else {
        for i in 0..lmax {
            //bs.slice_mut(s![.., .., i]).assign(&(&omega_sq_inv * &XpYguess.unwrap().slice(s![i, .., ..])));
            let tmp_array: Array2<f64> = (&omega_sq_inv * &XpYguess.unwrap().slice(s![i, .., ..]));
            let norm_temp: f64 = norm_special(&tmp_array);
            //tmp_array = tmp_array / norm_special(&tmp_array);
            bs.slice_mut(s![.., .., i])
                .assign(&(&tmp_array / norm_temp));
        }
    }
    let mut l: usize = lmax;
    let k: usize = nstates;
    let mut w: Array1<f64> = Array::zeros(lmax);
    let mut T_new: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    let mut r_bs_first: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));

    for it in 0..maxiter {
        let lmax: usize = bs.dim().2;
        let r_bs: Array3<f64> = matrix_v_product(&bs, lmax, n_occ, n_virt, &om, &wq_ov, &gamma);
        r_bs_first = r_bs.clone();
        // shape of Hb: (lmax, lmax)

        let Hb: Array2<f64> = tensordot(&bs, &r_bs, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();
        let (w2, Tb): (Array1<f64>, Array2<f64>) = Hb.eigh(UPLO::Upper).unwrap();
        // shape of T : (n_occ,n_virt,lmax)
        let T: Array3<f64> = tensordot(&bs, &Tb, &[Axis(2)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();

        // In DFTBaby a selector of symmetry could be used here
        let temp: (Array1<f64>, Array3<f64>);

        if l2_treshold > 0.0 {
            temp = reorder_vectors_lambda2(&Oia, &w2, &T, l2_treshold);
        } else {
            temp = (Array::zeros(lmax), Array::zeros((n_occ, n_virt, lmax)));
        }
        let (w2_new, T_temp): (Array1<f64>, Array3<f64>) = temp;
        T_new = T_temp;

        w = w2_new.mapv(f64::sqrt);
        //residual vectors
        //let W_res: Array3<f64> = matrix_v_product(&T, lmax, &om, &wq_ov, &gamma) -  &T*&w2_new;
        let W_res: Array3<f64> = matrix_v_product(&T, lmax, n_occ, n_virt, &om, &wq_ov, &gamma)
            - einsum("k,ijk->ijk", &[&w2_new, &T])
                .unwrap()
                .into_dimensionality::<Ix3>()
                .unwrap();

        let mut norms_res: Array1<f64> = Array::zeros(k);
        for i in 0..k {
            norms_res[i] = norm_special(&W_res.slice(s![.., .., i]).to_owned());
        }
        // check if all norms are below the convergence criteria
        // maybe there is a faster method
        let indices_norms: Array1<usize> = norms_res
            .indexed_iter()
            .filter_map(|(index, &item)| if item < conv { Some(index) } else { None })
            .collect();
        if indices_norms.len() == norms_res.len() {
            break;
        }

        // # enlarge dimension of subspace by dk vectors
        // # At most k new expansion vectors are added
        let dkmax = (kmax - l).min(k);
        // # count number of non-converged vectors
        // # residual vectors that are zero cannot be used as new expansion vectors
        //1.0e-16
        let eps = 0.01 * conv;
        // version for nc = np.sum(norms > eps)

        let indices_norm_over_eps: Array1<usize> = norms_res
            .indexed_iter()
            .filter_map(|(index, &item)| if item > eps { Some(index) } else { None })
            .collect();

        let nc: usize = indices_norm_over_eps.len();
        let dk: usize = dkmax.min(nc as usize);

        let mut Qs: Array3<f64> = Array::zeros((n_occ, n_virt, dk));
        let mut nb: i32 = 0;

        // # select new expansion vectors among the residual vectors
        for i in 0..dkmax {
            let wD: Array2<f64> = w[i] - &omega.to_owned();
            // quite the ugly method in order to reproduce
            // indx = abs(wD) < 1.0e-6
            // wD[indx] = 1.0e-6 * omega[indx]
            // from numpy

            let temp: Array2<f64> = wD.map(|wD| if wD < &1.0e-6 { 1.0e-6 } else { 0.0 });
            let temp_2: Array2<f64> = wD.map(|&wD| if wD < 1.0e-6 { 0.0 } else { wD });
            let mut wD_new: Array2<f64> = &temp * &omega.to_owned();
            wD_new = wD_new + temp_2;

            if norms_res[i] > eps {
                Qs.slice_mut(s![.., .., nb])
                    .assign(&((1.0 / &wD_new) * W_res.slice(s![.., .., i])));
                nb += 1;
            }
        }

        // new expansion vectors are bs + Qs
        let mut bs_new: Array3<f64> = Array::zeros((n_occ, n_virt, l + dk));
        bs_new.slice_mut(s![.., .., ..l]).assign(&bs);
        bs_new.slice_mut(s![.., .., l..]).assign(&Qs);

        //QR decomposition
        let nvec: usize = l + dk;

        let bs_flat: Array2<f64> = bs_new.into_shape((n_occ * n_virt, nvec)).unwrap();
        let (Q, R): (Array2<f64>, Array2<f64>) = bs_flat.qr().unwrap();
        bs = Q.into_shape((n_occ, n_virt, nvec)).unwrap();
        l = bs.dim().2;
    }
    let Omega: Array1<f64> = w.slice(s![..k]).to_owned();
    let mut XpY: Array3<f64> = Array::zeros((n_occ, n_virt, k));
    let mut XmY: Array3<f64> = Array::zeros((n_occ, n_virt, k));
    let mut c_matrix: Array3<f64> = Array::zeros((n_occ, n_virt, k));

    for i in 0..k {
        let temp_T: Array2<f64> = T_new.slice(s![.., .., i]).to_owned();
        // # X+Y = 1/sqrt(Omega)*(A-B)^(1/2).T
        XpY.slice_mut(s![.., .., i])
            .assign(&(&(&omega_sq / Omega[i].sqrt()) * &temp_T));
        // # X-Y = sqrt(Omega)*(A-B)^(-1).(X+Y)
        XmY.slice_mut(s![.., .., i])
            .assign(&(Omega[i].sqrt() * &omega_sq_inv * &temp_T));
        // # C = (A-B)^(-1/2).(X+Y) * sqrt(Omega)
        c_matrix.slice_mut(s![.., .., i]).assign(&temp_T);
    }
    // # XmY, XpY and C have shape (nocc,nvirt, nstates)
    // # bring the last axis to the front
    XpY.swap_axes(1, 2);
    XpY.swap_axes(0, 1);
    XmY.swap_axes(1, 2);
    XmY.swap_axes(0, 1);
    c_matrix.swap_axes(1, 2);
    c_matrix.swap_axes(0, 1);

    return (Omega, c_matrix, XmY, XpY);
}

pub fn non_hermitian_davidson(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    qtrans_oo: ArrayView3<f64>,
    qtrans_vv: ArrayView3<f64>,
    qtrans_ov: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    n_occ: usize,
    n_virt: usize,
    XmYguess: Option<ArrayView3<f64>>,
    XpYguess: Option<ArrayView3<f64>>,
    w_guess: Option<ArrayView1<f64>>,
    multiplicity: usize,
    nstates: Option<usize>,
    ifact: Option<usize>,
    maxiter: Option<usize>,
    conv: Option<f64>,
    l2_treshold: Option<f64>,
    lc: Option<usize>,
) -> (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) {
    // set values or defaults
    let nstates: usize = nstates.unwrap_or(4);
    let ifact: usize = ifact.unwrap_or(1);
    let maxiter: usize = maxiter.unwrap_or(10);
    let conv: f64 = conv.unwrap_or(1.0e-14);
    let l2_treshold: f64 = l2_treshold.unwrap_or(0.5);
    let lc: usize = lc.unwrap_or(1);

    // at most there are nocc*nvirt excited states
    let kmax = n_occ * n_virt;
    // To achieve fast convergence the solution vectors from a nearby geometry
    // should be used as initial expansion vectors

    let lmax: usize = (ifact * nstates).min(kmax);
    let mut bs: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));

    if XpYguess.is_none() {
        bs = initial_expansion_vectors(omega.to_owned(), lmax);
    } else {
        // For the expansion vectors we use X+Y
        let lmaxp: usize = nstates;
        // and X-Y if the vectors space has not been exhausted yet
        let lmaxm: usize = nstates.min(kmax - nstates);
        let lmax: usize = lmaxp + lmaxm;

        bs = Array::zeros((n_occ, n_virt, lmax));
        // bring axis with state indeces to the back
        let mut XmY_temp: Array3<f64> = XmYguess.unwrap().to_owned();
        XmY_temp.swap_axes(0, 1);
        XmY_temp.swap_axes(1, 2);
        let mut XpY_temp: Array3<f64> = XpYguess.unwrap().to_owned();
        XpY_temp.swap_axes(0, 1);
        XpY_temp.swap_axes(1, 2);

        bs.slice_mut(s![.., .., ..lmaxp]).assign(&XmY_temp);
        bs.slice_mut(s![.., .., lmaxp..])
            .assign(&XpY_temp.slice(s![.., .., ..lmaxm]));

        for i in 0..lmax {
            let temp: f64 = norm_special(&bs.slice(s![.., .., i]).to_owned());
            let tmp_array: Array2<f64> = bs.slice(s![.., .., i]).to_owned();
            bs.slice_mut(s![.., .., i]).assign(&(tmp_array / temp));
        }
    }
    let mut l: usize = lmax;
    let k: usize = nstates;

    let mut w: Array1<f64> = Array::zeros(lmax);
    let mut Tb: Array2<f64> = Array::zeros((lmax, lmax));

    let mut l_canon: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    let mut r_canon: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));

    for it in 0..maxiter {
        let lmax: usize = bs.dim().2;

        if XpYguess.is_none() || it > 0 {
            // # evaluate (A+B).b and (A-B).b
            let bp: Array3<f64> = get_apbv(
                &gamma,
                &Some(gamma_lr),
                &Some(qtrans_oo),
                &Some(qtrans_vv),
                &qtrans_ov,
                &omega,
                &bs,
                lc,
            );
            let bm: Array3<f64> = get_ambv(
                &gamma, &gamma_lr, &qtrans_oo, &qtrans_vv, &qtrans_ov, &omega, &bs, lc,
            );

            // # M^+ = (b_i, (A+B).b_j)
            let mp: Array2<f64> = tensordot(&bs, &bp, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
                .into_dimensionality::<Ix2>()
                .unwrap();
            // # M^- = (b_i, (A-B).b_j)
            let mm: Array2<f64> = tensordot(&bs, &bm, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
                .into_dimensionality::<Ix2>()
                .unwrap();
            let mmsq: Array2<f64> = mm.ssqrt(UPLO::Upper).unwrap();

            // # Mh is the analog of (A-B)^(1/2).(A+B).(A-B)^(1/2)
            // # in the reduced subspace spanned by the expansion vectors bs
            let mh: Array2<f64> = mmsq.dot(&mp.dot(&mmsq));
            let mh_reversed: Array2<f64> = mh.clone().reversed_axes();
            // check that Mh is hermitian
            let mut subst: Array2<f64> = mh.clone() - mh_reversed;
            subst = subst.map(|subst| subst.abs());
            let err: f64 = subst.sum();

            if err > 1.0e-10 {
                // could raise error here
                println!("Mh is not hermitian");
            }
            let tmp: (Array1<f64>, Array2<f64>) = mh.eigh(UPLO::Upper).unwrap();
            let w2: Array1<f64> = tmp.0;
            Tb = tmp.1;
            //In DFTBaby check for selector(symmetry checker)
            w = w2.mapv(f64::sqrt);
            let wsq: Array1<f64> = w.mapv(f64::sqrt);

            // approximate right R = (X+Y) and left L = (X-Y) eigenvectors
            // in the basis bs
            // (X+Y) = (A-B)^(1/2).T / sqrt(w)
            let rb: Array2<f64> = mmsq.dot(&Tb) / wsq;
            // L = (X-Y) = 1/w * (A+B).(X+Y)
            let lb: Array2<f64> = mp.dot(&rb) / &w;
            // check that (Lb^T, Rb) = 1 is fulfilled
            let temp_eye: Array2<f64> = Array::eye(lmax);
            let temp: Array2<f64> = lb.clone().reversed_axes().dot(&rb) - temp_eye;
            let err: f64 = temp.sum();
            if err > 1.0e-3 {
                // could raise error here
                println!("(X+Y) and (X-Y) vectors not orthonormal!");
            }
            // transform to the canonical basis Lb -> L, Rb -> R
            l_canon = tensordot(&bs, &lb, &[Axis(2)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            r_canon = tensordot(&bs, &rb, &[Axis(2)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
        } else {
            // bring axis with state indeces to the back
            let mut XmY_temp: Array3<f64> = XmYguess.unwrap().to_owned();
            XmY_temp.swap_axes(0, 1);
            XmY_temp.swap_axes(1, 2);
            let mut XpY_temp: Array3<f64> = XpYguess.unwrap().to_owned();
            XpY_temp.swap_axes(0, 1);
            XpY_temp.swap_axes(1, 2);

            r_canon = XpY_temp;
            l_canon = XmY_temp;
            w = w_guess.unwrap().to_owned();
        }
        // residual vectors
        let wl = get_apbv(
            &gamma,
            &Some(gamma_lr),
            &Some(qtrans_oo),
            &Some(qtrans_vv),
            &qtrans_ov,
            &omega,
            &r_canon,
            lc,
        ) - &l_canon * &w;
        let wr = get_ambv(
            &gamma, &gamma_lr, &qtrans_oo, &qtrans_vv, &qtrans_ov, &omega, &l_canon, lc,
        ) - &r_canon * &w;
        //norms
        let mut norms: Array1<f64> = Array::zeros(k);
        let mut norms_l: Array1<f64> = Array::zeros(k);
        let mut norms_r: Array1<f64> = Array::zeros(k);
        for i in 0..k {
            norms_l[i] = norm_special(&wl.slice(s![.., .., i]).to_owned());
            norms_r[i] = norm_special(&wr.slice(s![.., .., i]).to_owned());
            norms[i] = norms_l[i] + norms_r[i];
        }
        // check for convergence
        let indices_norms: Array1<usize> = norms
            .indexed_iter()
            .filter_map(|(index, &item)| if item < conv { Some(index) } else { None })
            .collect();
        if indices_norms.len() == norms.len() && it > 0 {
            break;
        }
        //  enlarge dimension of subspace by dk vectors
        //  At most 2*k new expansion vectors are added
        let dkmax = (kmax - l).min(2 * k);
        // # count number of non-converged vectors
        // # residual vectors that are zero cannot be used as new expansion vectors
        //1.0e-16
        let eps = 0.01 * conv;

        let indices_norm_r_over_eps: Array1<usize> = norms_r
            .indexed_iter()
            .filter_map(|(index, &item)| if item > eps { Some(index) } else { None })
            .collect();
        let indices_norm_l_over_eps: Array1<usize> = norms_l
            .indexed_iter()
            .filter_map(|(index, &item)| if item > eps { Some(index) } else { None })
            .collect();

        let nc_l: usize = indices_norm_r_over_eps.len();
        let nc_r: usize = indices_norm_l_over_eps.len();
        // Half the new expansion vectors should come from the left residual vectors
        // the other half from the right residual vectors.
        let dk_r: usize = ((dkmax as f64 / 2.0) as usize).min(nc_l);
        let dk_l: usize = (dkmax - dk_r).min(nc_r);
        let dk: usize = dk_r + dk_l;

        let mut Qs: Array3<f64> = Array::zeros((n_occ, n_virt, dk));
        let mut nb: i32 = 0;
        // select new expansion vectors among the non-converged left residual vectors
        for i in 0..k {
            if nb as usize == dk {
                //got enough new expansion vectors
                break;
            }
            let wD: Array2<f64> = w[i] - &omega.to_owned();
            // quite the ugly method in order to reproduce
            // indx = abs(wD) < 1.0e-6
            // wD[indx] = 1.0e-6 * omega[indx]
            // from numpy
            let temp: Array2<f64> = wD.map(|wD| if wD < &1.0e-6 { 1.0e-6 } else { 0.0 });
            let temp_2: Array2<f64> = wD.map(|&wD| if wD < 1.0e-6 { 0.0 } else { wD });
            let mut wD_new: Array2<f64> = &temp * &omega.to_owned();
            wD_new = wD_new + temp_2;
            if norms_l[i] > eps {
                Qs.slice_mut(s![.., .., nb])
                    .assign(&((1.0 / &wD_new) * wl.slice(s![.., .., i])));
                nb += 1;
            }
        }
        for i in 0..k {
            if nb as usize == dk {
                //got enough new expansion vectors
                break;
            }
            let wD: Array2<f64> = w[i] - &omega.to_owned();
            // quite the ugly method in order to reproduce
            // indx = abs(wD) < 1.0e-6
            // wD[indx] = 1.0e-6 * omega[indx]
            // from numpy
            let temp: Array2<f64> = wD.map(|wD| if wD < &1.0e-6 { 1.0e-6 } else { 0.0 });
            let temp_2: Array2<f64> = wD.map(|&wD| if wD < 1.0e-6 { 0.0 } else { wD });
            let mut wD_new: Array2<f64> = &temp * &omega.to_owned();
            wD_new = wD_new + temp_2;
            if norms_r[i] > eps {
                Qs.slice_mut(s![.., .., nb])
                    .assign(&((1.0 / &wD_new) * wr.slice(s![.., .., i])));
                nb += 1;
            }
        }
        // new expansion vectors are bs + Qs
        let mut bs_new: Array3<f64> = Array::zeros((n_occ, n_virt, l + dk));
        bs_new.slice_mut(s![.., .., ..l]).assign(&bs);
        bs_new.slice_mut(s![.., .., l..]).assign(&Qs);

        //QR decomposition
        let nvec: usize = l + dk;
        let bs_flat: Array2<f64> = bs_new.into_shape((n_occ * n_virt, nvec)).unwrap();
        let (Q, R): (Array2<f64>, Array2<f64>) = bs_flat.qr().unwrap();
        bs = Q.into_shape((n_occ, n_virt, nvec)).unwrap();
        l = bs.dim().2;
    }
    let Omega: Array1<f64> = w.slice(s![..k]).to_owned();
    let mut XpY: Array3<f64> = r_canon.slice(s![.., .., ..k]).to_owned();
    let mut XmY: Array3<f64> = l_canon.slice(s![.., .., ..k]).to_owned();

    let t_matrix: Array3<f64> = tensordot(&bs, &Tb, &[Axis(2)], &[Axis(0)])
        .into_dimensionality::<Ix3>()
        .unwrap();
    let mut c_matrix: Array3<f64> = t_matrix.slice(s![.., .., ..k]).to_owned();

    XpY.swap_axes(1, 2);
    XpY.swap_axes(0, 1);
    XmY.swap_axes(1, 2);
    XmY.swap_axes(0, 1);
    c_matrix.swap_axes(1, 2);
    c_matrix.swap_axes(0, 1);

    return (Omega, c_matrix, XmY, XpY);
}

pub fn get_apbv(
    gamma: &ArrayView2<f64>,
    gamma_lr: &Option<ArrayView2<f64>>,
    qtrans_oo: &Option<ArrayView3<f64>>,
    qtrans_vv: &Option<ArrayView3<f64>>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    lc: usize,
) -> (Array3<f64>) {
    let lmax: usize = vs.dim().2;
    let mut us: Array3<f64> = Array::zeros((vs.shape())).into_dimensionality().unwrap();

    for i in 0..lmax {
        let v: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // # matrix product u_ia = sum_jb (A+B)_(ia,jb) v_jb
        // # 1st term in (A+B).v  - KS orbital energy differences
        let mut u: Array2<f64> = omega * &v;
        // 2nd term Coulomb
        let tmp: Array1<f64> = tensordot(&qtrans_ov, &v, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix1>()
            .unwrap();
        let tmp_2: Array1<f64> = gamma.dot(&tmp);
        u = u + 4.0
            * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                .into_dimensionality::<Ix2>()
                .unwrap();

        if lc == 1 {
            // 3rd term - Exchange
            let tmp: Array3<f64> = tensordot(&qtrans_vv.unwrap(), &v, &[Axis(2)], &[Axis(1)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            let tmp_2: Array3<f64> = tensordot(&gamma_lr.unwrap(), &tmp, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            u = u - tensordot(
                &qtrans_oo.unwrap(),
                &tmp_2,
                &[Axis(0), Axis(2)],
                &[Axis(0), Axis(2)],
            )
            .into_dimensionality::<Ix2>()
            .unwrap();

            //4th term - Exchange
            let tmp: Array3<f64> = tensordot(&qtrans_ov, &v, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            let tmp_2: Array3<f64> = tensordot(&gamma_lr.unwrap(), &tmp, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            u = u - tensordot(&qtrans_ov, &tmp_2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
                .into_dimensionality::<Ix2>()
                .unwrap();
        }

        us.slice_mut(s![.., .., i]).assign(&u);
    }
    return us;
}

pub fn get_ambv(
    gamma: &ArrayView2<f64>,
    gamma_lr: &ArrayView2<f64>,
    qtrans_oo: &ArrayView3<f64>,
    qtrans_vv: &ArrayView3<f64>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    lc: usize,
) -> (Array3<f64>) {
    let lmax: usize = vs.dim().2;
    let mut us: Array3<f64> = Array::zeros((vs.shape())).into_dimensionality().unwrap();

    for i in 0..lmax {
        let v: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // # matrix product u_ia = sum_jb (A-B)_(ia,jb) v_jb
        // # 1st term, differences in orbital energies
        let mut u: Array2<f64> = omega * &v;

        if lc == 1 {
            // 2nd term - Coulomb
            let tmp: Array3<f64> = tensordot(&qtrans_ov, &v, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            let tmp_2: Array3<f64> = tensordot(&gamma_lr, &tmp, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            u = u + tensordot(&qtrans_ov, &tmp_2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
                .into_dimensionality::<Ix2>()
                .unwrap();

            //3rd term - Exchange
            let tmp: Array3<f64> = tensordot(&qtrans_vv, &v, &[Axis(2)], &[Axis(1)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            let tmp_2: Array3<f64> = tensordot(&gamma_lr, &tmp, &[Axis(1)], &[Axis(0)])
                .into_dimensionality::<Ix3>()
                .unwrap();
            u = u - tensordot(&qtrans_oo, &tmp_2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
                .into_dimensionality::<Ix2>()
                .unwrap();
        }

        us.slice_mut(s![.., .., i]).assign(&u);
    }
    return us;
}

pub fn initial_expansion_vectors(omega_guess: Array2<f64>, lmax: usize) -> (Array3<f64>) {
    //     The initial guess vectors are the lmax lowest energy
    //     single excitations
    let n_occ: usize = omega_guess.dim().0;
    let n_virt: usize = omega_guess.dim().1;
    let mut bs: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    // flatten omega, python: numpy.ravel(omega)
    let omega_length: usize = omega_guess.iter().len();
    let omega_flat = omega_guess.into_shape(omega_length).unwrap();
    // sort omega, only possible for vectors
    //let mut omega_vec = omega_flat.to_vec();
    //omega_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    //let omega_flat_new = Array::from_vec(omega_vec);
    //let mut indices_new:Array1<usize> = Array::zeros(lmax);
    //for i in 0.. lmax{
    //    for j in 0.. lmax{
    //        if omega_flat[j] == omega_flat_new[i]{
    //            indices_new[i] = j;
    //        }
    //    }
    //}
    let indices_argsort: Vec<usize> = argsort(omega_flat.view());
    //let indices: Array1<usize> = omega_flat_new
    //    .indexed_iter()
    //    .filter_map(|(index, &item)| Some(index))
    //   .collect();

    for j in 0..lmax {
        let idx = indices_argsort[j];
        // row - occupied index
        let i: usize = (idx / n_virt) as usize;
        // col - virtual index
        let a: usize = idx % n_virt;

        bs[[i, a, j]] = 1.0;
    }
    return bs;
}

pub fn reorder_vectors_lambda2(
    Oia: &ArrayView2<f64>,
    w2: &Array1<f64>,
    T: &Array3<f64>,
    l2_treshold: f64,
) -> (Array1<f64>, Array3<f64>) {
    // reorder the expansion vectors so that those with Lambda2 values
    // above a certain threshold come first

    let n_occ: usize = T.dim().0;
    let n_virt: usize = T.dim().1;
    let n_st: usize = T.dim().2;
    let mut l2: Array1<f64> = Array::zeros(n_st);

    for i in 0..n_st {
        let T_temp: Array2<f64> = T
            .slice(s![.., .., i])
            .to_owned()
            .map(|T| ndarray_linalg::Scalar::powi(T, 2));
        l2[i] = tensordot(&T_temp, &Oia, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix0>()
            .unwrap()
            .into_scalar();
    }
    //get indeces
    let over_l2: Array1<_> = l2
        .indexed_iter()
        .filter_map(|(index, &item)| {
            if item > l2_treshold {
                Some(index)
            } else {
                None
            }
        })
        .collect();
    let under_l2: Array1<_> = l2
        .indexed_iter()
        .filter_map(|(index, &item)| {
            if item < l2_treshold {
                Some(index)
            } else {
                None
            }
        })
        .collect();

    let mut T_new: Array3<f64> = Array::zeros((n_occ, n_virt, n_st));
    let mut w2_new: Array1<f64> = Array::zeros(n_st);

    //construct new matrices
    for i in 0..over_l2.len() {
        T_new
            .slice_mut(s![.., .., i])
            .assign(&T.slice(s![.., .., over_l2[i]]));
        w2_new[i] = w2[over_l2[i]];
    }
    let len_over_l2: usize = over_l2.len();
    for i in 0..under_l2.len() {
        T_new
            .slice_mut(s![.., .., i + len_over_l2])
            .assign(&T.slice(s![.., .., under_l2[i]]));
        w2_new[i + len_over_l2] = w2[under_l2[i]];
    }

    return (w2_new, T_new);
}

pub fn norm_special(array: &Array2<f64>) -> (f64) {
    let v: f64 = tensordot(&array, &array, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
        .into_dimensionality::<Ix0>()
        .unwrap()
        .into_scalar();
    return v.sqrt();
}

pub fn matrix_v_product(
    vs: &Array3<f64>,
    lmax: usize,
    n_occ: usize,
    n_virt: usize,
    om: &Array2<f64>,
    wq_ov: &Array3<f64>,
    gamma: &ArrayView2<f64>,
) -> (Array3<f64>) {
    let mut us: Array3<f64> = Array::zeros((n_occ, n_virt, lmax));
    for i in 0..lmax {
        let v: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // # matrix product u = sum_jb (A-B)^(1/2).(A+B).(A-B)^(1/2).v
        // # 1st term in (A+B).v  - KS orbital energy differences
        let mut u: Array2<f64> = Array::zeros((n_occ, n_virt));
        u = om * &v;
        let tmp: Array1<f64> = tensordot(&wq_ov, &v, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix1>()
            .unwrap();
        let tmp2: Array1<f64> = gamma.dot(&tmp);
        u = u + 4.0
            * tensordot(&wq_ov, &tmp2, &[Axis(0)], &[Axis(0)])
                .into_dimensionality::<Ix2>()
                .unwrap();
        us.slice_mut(s![.., .., i]).assign(&u);
    }
    return us;
}

pub fn krylov_solver_zvector(
    a_diag: ArrayView2<f64>,
    b_matrix: ArrayView3<f64>,
    x_0: Option<ArrayView3<f64>>,
    maxiter: Option<usize>,
    conv: Option<f64>,
    g0: ArrayView2<f64>,
    g0_lr: Option<ArrayView2<f64>>,
    qtrans_oo: Option<ArrayView3<f64>>,
    qtrans_vv: Option<ArrayView3<f64>>,
    qtrans_ov: ArrayView3<f64>,
    lc: usize,
) -> (Array3<f64>) {
    // Parameters:
    // ===========
    // A: linear operator, such that A(X) = A.X
    // Adiag: diagonal elements of A-matrix, with dimension (nocc,nvirt)
    // B: right hand side of equation, (nocc,nvirt, k)
    // X0: initial guess vectors or None

    let maxiter: usize = maxiter.unwrap_or(1000);
    let conv: f64 = conv.unwrap_or(1.0e-14);

    let n_occ: usize = b_matrix.dim().0;
    let n_virt: usize = b_matrix.dim().1;
    let k: usize = b_matrix.dim().2;
    // number of vectors
    let kmax: usize = n_occ * n_virt;
    let mut l: usize = k;

    // bs are expansion vectors
    let a_inv: Array2<f64> = 1.0 / &a_diag.to_owned();
    let mut bs: Array3<f64> = Array::zeros((n_occ, n_virt, k));

    if x_0.is_none() {
        for i in 0..k {
            bs.slice_mut(s![.., .., i])
                .assign(&(&a_inv * &b_matrix.slice(s![.., .., i])));
        }
    } else {
        bs = x_0.unwrap().to_owned();
    }

    let mut x_matrix: Array3<f64> = Array::zeros((n_occ, n_virt, k));

    for it in 0..maxiter {
        // representation of A in the basis of expansion vectors
        let a_b: Array2<f64> = tensordot(
            &bs,
            &get_apbv(
                &g0, &g0_lr, &qtrans_oo, &qtrans_vv, &qtrans_ov, &a_diag, &bs, lc,
            ),
            &[Axis(0), Axis(1)],
            &[Axis(0), Axis(1)],
        )
        .into_dimensionality::<Ix2>()
        .unwrap();
        // RHS in basis of expansion vectors
        let b_b: Array2<f64> = tensordot(&bs, &b_matrix, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();
        // solve
        let mut x_b: Array2<f64> = Array2::zeros((k, l));
        for i in 0..k {
            x_b.slice_mut(s![i, ..])
                .assign((&a_b.solve(&b_b.slice(s![.., i])).unwrap()));
        }
        x_b = x_b.reversed_axes();
        // transform solution vector back into canonical basis
        x_matrix = tensordot(&bs, &x_b, &[Axis(2)], &[Axis(0)])
            .into_dimensionality::<Ix3>()
            .unwrap();
        // residual vectors
        let w_res: Array3<f64> = &get_apbv(
            &g0, &g0_lr, &qtrans_oo, &qtrans_vv, &qtrans_ov, &a_diag, &x_matrix, lc,
        ) - &b_matrix;
        let mut norms: Array1<f64> = Array::zeros(k);
        for i in 0..k {
            norms[i] = norm_special(&w_res.slice(s![.., .., i]).to_owned());
        }
        // check if all values of the norms are under the convergence criteria
        let indices_norms: Array1<usize> = norms
            .indexed_iter()
            .filter_map(|(index, &item)| if item < conv { Some(index) } else { None })
            .collect();
        if indices_norms.len() == norms.len() {
            break;
        }
        // # enlarge dimension of subspace by dk vectors
        // # At most k new expansion vectors are added
        let dkmax = (kmax - l).min(k);
        // # count number of non-converged vectors
        // # residual vectors that are zero cannot be used as new expansion vectors
        //1.0e-16
        let eps = 0.01 * conv;
        // version for nc = np.sum(norms > eps)
        let indices_norm_over_eps: Array1<usize> = norms
            .indexed_iter()
            .filter_map(|(index, &item)| if item > eps { Some(index) } else { None })
            .collect();
        let mut norms_over_eps: Array1<f64> = Array::zeros(indices_norm_over_eps.len());
        for i in 0..indices_norm_over_eps.len() {
            norms_over_eps[i] = norms[indices_norm_over_eps[i]];
        }
        let nc: f64 = norms_over_eps.sum();
        let dk: usize = dkmax.min((1.0 + nc) as usize);
        let mut Qs: Array3<f64> = Array::zeros((n_occ, n_virt, dk));
        let mut nb: i32 = 0;

        for i in 0..dkmax {
            if norms[i] > eps {
                Qs.slice_mut(s![.., .., nb])
                    .assign(&((&a_inv) * &w_res.slice(s![.., .., i])));
                nb += 1;
            }
        }
        assert!(nb as usize == dk);
        // new expansion vectors are bs + Qs
        let mut bs_new: Array3<f64> = Array::zeros((n_occ, n_virt, l + dk));
        bs_new.slice_mut(s![.., .., ..l]).assign(&bs);
        bs_new.slice_mut(s![.., .., l..]).assign(&Qs);

        // QR decomposition as in hermitian davidson
        // to receive orthogonalized vectors
        // alternative: implement gram schmidt orthogonalization
        // Alexander also uses this method in hermitian davidson

        let nvec: usize = l + dk;
        let bs_flat: Array2<f64> = bs_new.into_shape((n_occ * n_virt, nvec)).unwrap();
        let (Q, R): (Array2<f64>, Array2<f64>) = bs_flat.qr().unwrap();
        bs = Q.into_shape((n_occ, n_virt, nvec)).unwrap();
        l = bs.dim().2;
    }
    return x_matrix;
}

#[test]
fn excited_energies_tda_routine() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule =
        Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    let S: Array2<f64> = array![
        [
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.3074918525690681,
            0.3074937992389065
        ],
        [
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.1987769748092704
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.3185054221819456
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            -0.3982160222204482,
            0.1327383036929333
        ],
        [
            0.3074918525690681,
            0.0000000000000000,
            0.0000000000000000,
            -0.3982160222204482,
            1.0000000000000000,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.0000000000000000
        ]
    ];
    let f_occ: Array1<f64> = array![
        2.0000000000000000,
        2.0000000000000000,
        2.0000000000000000,
        2.0000000000000000,
        0.0000000000000000,
        0.0000000000000000
    ];
    let orbe: Array1<f64> = array![
        -0.8274698453897238,
        -0.4866977301135242,
        -0.4293504173916476,
        -0.3805317623354740,
        0.4597732058522524,
        0.5075648555895222
    ];
    let orbs: Array2<f64> = array![
        [
            8.7633817073094666e-01,
            -7.3282333485870987e-07,
            -2.5626946551477237e-01,
            6.5297360604596939e-16,
            4.4638746429764842e-05,
            6.5169208620110397e-01
        ],
        [
            1.5609825393253833e-02,
            -1.9781346650257903e-01,
            -3.5949496391505154e-01,
            -8.4834397825097185e-01,
            2.8325035872464288e-01,
            -2.9051011756322170e-01
        ],
        [
            2.5012021798979999e-02,
            -3.1696156822051985e-01,
            -5.7602795979721200e-01,
            5.2944545948125277e-01,
            4.5385928211929161e-01,
            -4.6549177907241801e-01
        ],
        [
            2.0847651645102452e-02,
            5.2838144790877872e-01,
            -4.8012913249889100e-01,
            7.0500141149039645e-17,
            -7.5667687027915898e-01,
            -3.8791257751171815e-01
        ],
        [
            1.6641905232449239e-01,
            -3.7146604214646906e-01,
            2.5136102811674521e-01,
            3.5209333698603627e-16,
            -7.2004450606557047e-01,
            -7.6949321868972731e-01
        ],
        [
            1.6641962261695781e-01,
            3.7146556016199661e-01,
            2.5135992631728726e-01,
            -1.6703880555921563e-15,
            7.1993590540308161e-01,
            -7.6959837575446732e-01
        ]
    ];
    let omega_ref_out: Array1<f64> = array![
        0.5639270376740005,
        0.6133146373942717,
        0.6289193025552757,
        0.6699836563376497,
        0.7241847548895822,
        0.7826629904064589,
        1.0415454883803148,
        1.0853598783803207
    ];

    mol.calculator.set_active_orbitals(f_occ.to_vec());

    let (omega_out, c_ij, XmY, XpY) = get_exc_energies(
        &f_occ.to_vec(),
        &mol,
        None,
        &S,
        &orbe,
        &orbs,
        false,
        Some(String::from("TDA")),
    );
    println!("omega {}", &omega_out);
    println!("omega-ref {}", &omega_ref_out);
    assert!(omega_out.abs_diff_eq(&omega_ref_out, 1e-10));
}

#[test]
fn excited_energies_casida_routine() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        charge,
        multiplicity,
        Some(0.0),
        None,
    );

    let S: Array2<f64> = array![
        [
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.3074918525690681,
            0.3074937992389065
        ],
        [
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.1987769748092704
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.3185054221819456
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            -0.3982160222204482,
            0.1327383036929333
        ],
        [
            0.3074918525690681,
            0.0000000000000000,
            0.0000000000000000,
            -0.3982160222204482,
            1.0000000000000000,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.0000000000000000
        ]
    ];
    let f_occ: Array1<f64> = array![
        2.0000000000000000,
        2.0000000000000000,
        2.0000000000000000,
        2.0000000000000000,
        0.0000000000000000,
        0.0000000000000000
    ];
    let orbe: Array1<f64> = array![
        -0.8688942612299978,
        -0.4499991998359348,
        -0.3563323833221640,
        -0.2833072445490362,
        0.3766541361485592,
        0.4290384545096948
    ];
    let orbs: Array2<f64> = array![
        [
            -8.6192454822470443e-01,
            -1.2183272339283827e-06,
            -2.9726068852099652e-01,
            1.4081526534480036e-16,
            4.3204927822587669e-05,
            6.5350390970911720e-01
        ],
        [
            2.6757514101592716e-03,
            -2.0080751179746489e-01,
            -3.6133406147262681e-01,
            8.4834397825097330e-01,
            2.8113675949218103e-01,
            -2.8862841063402733e-01
        ],
        [
            4.2874248054355704e-03,
            -3.2175900344457298e-01,
            -5.7897479277207053e-01,
            -5.2944545948125010e-01,
            4.5047260810181733e-01,
            -4.6247667201346043e-01
        ],
        [
            3.5735935812310771e-03,
            5.3637854372415550e-01,
            -4.8258565481595922e-01,
            1.1681492606832304e-17,
            -7.5102779853479884e-01,
            -3.8540269278998796e-01
        ],
        [
            -1.7925702667916332e-01,
            -3.6380704327446040e-01,
            2.3851989294055637e-01,
            1.2590381109703569e-16,
            -7.2394294209808119e-01,
            -7.7069762107663153e-01
        ],
        [
            -1.7925784113437207e-01,
            3.6380666541133738e-01,
            2.3851861974981367e-01,
            -1.2849138889170433e-16,
            7.2383785715164428e-01,
            -7.7079977605732564e-01
        ]
    ];
    let omega_ref_out: Array1<f64> = array![
        0.6599613806975956,
        0.7123456990587312,
        0.7456810724193218,
        0.7930925652349430,
        0.8714866033195208,
        0.9348736014086754,
        1.2756452171930386,
        1.3231856682449914
    ];
    let XmY_ref: Array3<f64> = array![
        [
            [2.2128648618044417e-18, -4.2837885645157593e-18],
            [-2.8331089290168361e-17, 1.0956852790925327e-17],
            [-2.1117244243003086e-17, 2.5711253844323120e-17],
            [-1.0000000000000000e+00, -1.2449955808069987e-16]
        ],
        [
            [8.4566420156591629e-18, -6.9515054799941099e-18],
            [-6.1028158643554143e-17, 4.9783268042835159e-17],
            [-2.5743903384582700e-16, 6.5166726178582824e-17],
            [-1.9800776586924378e-16, 9.9999999999999956e-01]
        ],
        [
            [2.1571381149268650e-02, -3.0272950956508287e-07],
            [2.9991274782833736e-05, 1.4821884853200859e-01],
            [9.9507889372419522e-01, 3.2471746776120414e-06],
            [-2.1086226126538834e-17, 2.5417238012172805e-16]
        ],
        [
            [1.1109010931404383e-06, -1.2585514282200026e-02],
            [-3.2069605724195532e-01, 2.8289343610151735e-05],
            [9.1191885305385101e-06, -9.4937779050850379e-01],
            [-1.7561409942356875e-17, 4.5339333385871467e-17]
        ],
        [
            [9.1609974781150986e-06, 6.0701873653884228e-02],
            [-9.6788599240179385e-01, 1.0919490802974424e-05],
            [2.9445559546949760e-05, 3.4280398643310944e-01],
            [4.4324175033345075e-17, -9.2400447203945273e-17]
        ],
        [
            [-1.0110298012914951e-01, 9.9804621423379319e-06],
            [1.5702586865564740e-05, 1.0113993758545037e+00],
            [-1.7694318737760151e-01, 2.6156157292938292e-05],
            [1.9857946600048738e-17, -1.0466006324463316e-16]
        ],
        [
            [1.0046988096296809e+00, 9.6514695321655601e-06],
            [1.5280031987727591e-05, 1.3343212650520786e-01],
            [-6.0845305188056441e-02, 1.2489695971976725e-07],
            [7.5122763960793728e-18, -3.9132864065320797e-17]
        ],
        [
            [9.3758930983432614e-06, -1.0067757866575058e+00],
            [-8.1915557416614687e-02, 1.5848191461390946e-05],
            [-1.1132783883529338e-06, 5.1182023937158247e-02],
            [1.2955170604817585e-17, -2.2231889566890241e-17]
        ]
    ];
    let XpY_ref: Array3<f64> = array![
        [
            [4.1763508636254349e-18, -8.4248404334045637e-18],
            [-3.5486909017295750e-17, 1.4594014828813940e-17],
            [-2.3453880501508304e-17, 3.0597046386914307e-17],
            [-1.0000000000000002e+00, -1.1534413748233340e-16]
        ],
        [
            [1.4786580341155389e-17, -1.2666022125834313e-17],
            [-7.0821135018403943e-17, 6.1432766736513571e-17],
            [-2.6489854805594991e-16, 7.1847203400656705e-17],
            [-2.1372459741218609e-16, 9.9999999999999989e-01]
        ],
        [
            [3.6031757025213461e-02, -5.2693108222979895e-07],
            [3.3247977273821135e-05, 1.7472610444659498e-01],
            [9.7813856605377358e-01, 3.4200094269070232e-06],
            [-1.8662261150095090e-17, 2.4280970577917310e-16]
        ],
        [
            [1.7446652974159811e-06, -2.0596777031237985e-02],
            [-3.3426673906292786e-01, 3.1354975875636876e-05],
            [8.4280732848682095e-06, -9.4013443503879601e-01],
            [-1.4613492623422834e-17, 4.0723215116316388e-17]
        ],
        [
            [1.3093105142174889e-05, 9.0405231040811093e-02],
            [-9.1809349842438848e-01, 1.1014103424754594e-05],
            [2.4765955235883321e-05, 3.0892988258405030e-01],
            [3.3565913282383666e-17, -7.5527335597635306e-17]
        ],
        [
            [-1.3470126301599275e-01, 1.3856384770590849e-05],
            [1.3884867212394084e-05, 9.5099287606167959e-01],
            [-1.3873209262143607e-01, 2.1973326807669002e-05],
            [1.4991749835052334e-17, -7.8178779166510453e-17]
        ],
        [
            [9.8099453932497549e-01, 9.8200956597785595e-06],
            [9.9018827855579505e-06, 9.1947088357042614e-02],
            [-3.4961749454181061e-02, 7.6894757708179213e-08],
            [3.8051933329030105e-18, -2.1807050492333845e-17]
        ],
        [
            [8.8257671639771870e-06, -9.8756150574886581e-01],
            [-5.1176316697330339e-02, 1.0528497535975099e-05],
            [-6.1670714139410095e-07, 3.0378857620768730e-02],
            [6.4616119150035461e-18, -1.1968683832692411e-17]
        ]
    ];

    println!("valorbs {:?}", mol.calculator.valorbs);
    println!("atomic numbers {:?}", mol.atomic_numbers);

    mol.calculator.set_active_orbitals(f_occ.to_vec());

    let (omega_out, c_ij, XmY, XpY) =
        get_exc_energies( &f_occ.to_vec(),&mol, None, &S, &orbe, &orbs, false, None);
    println!("omega_out{}", &omega_out);
    println!("omega_diff {}", &omega_out - &omega_ref_out);
    assert!(omega_out.abs_diff_eq(&omega_ref_out, 1e-10));
    assert!((&XpY * &XpY).abs_diff_eq(&(&XpY_ref * &XpY_ref), 1e-10));
    assert!((&XmY * &XmY).abs_diff_eq(&(&XmY_ref * &XmY_ref), 1e-10));
}

#[test]
fn excited_energies_hermitian_davidson_routine() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        charge,
        multiplicity,
        Some(0.0),
        None,
    );

    let S: Array2<f64> = array![
        [
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.3074918525690681,
            0.3074937992389065
        ],
        [
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.1987769748092704
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.3185054221819456
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            -0.3982160222204482,
            0.1327383036929333
        ],
        [
            0.3074918525690681,
            0.0000000000000000,
            0.0000000000000000,
            -0.3982160222204482,
            1.0000000000000000,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.0000000000000000
        ]
    ];
    let f_occ: Array1<f64> = array![
        2.0000000000000000,
        2.0000000000000000,
        2.0000000000000000,
        2.0000000000000000,
        0.0000000000000000,
        0.0000000000000000
    ];
    let orbe: Array1<f64> = array![
        -0.8688942612301258,
        -0.4499991998360209,
        -0.3563323833222918,
        -0.2833072445491910,
        0.3766541361485015,
        0.4290384545096518
    ];
    let orbs: Array2<f64> = array![
        [
            -8.6192454822475639e-01,
            -1.2183272343139559e-06,
            -2.9726068852089849e-01,
            2.6222307203584133e-16,
            4.3204927822809713e-05,
            6.5350390970909367e-01
        ],
        [
            2.6757514101551499e-03,
            -2.0080751179749709e-01,
            -3.6133406147264924e-01,
            8.4834397825097341e-01,
            2.8113675949215844e-01,
            -2.8862841063399913e-01
        ],
        [
            4.2874248054290296e-03,
            -3.2175900344462377e-01,
            -5.7897479277210717e-01,
            -5.2944545948124977e-01,
            4.5047260810178097e-01,
            -4.6247667201341525e-01
        ],
        [
            3.5735935812255637e-03,
            5.3637854372423877e-01,
            -4.8258565481599014e-01,
            3.4916084620212056e-16,
            -7.5102779853473878e-01,
            -3.8540269278994982e-01
        ],
        [
            -1.7925702667910837e-01,
            -3.6380704327437935e-01,
            2.3851989294050652e-01,
            -2.0731761365694774e-16,
            -7.2394294209812204e-01,
            -7.7069762107665973e-01
        ],
        [
            -1.7925784113431714e-01,
            3.6380666541125695e-01,
            2.3851861974976313e-01,
            -9.2582148396003538e-17,
            7.2383785715168458e-01,
            -7.7079977605735461e-01
        ]
    ];
    let omega_ref_out: Array1<f64> = array![
        0.6599613806976925,
        0.7123456990588427,
        0.7456810724193917,
        0.7930925652350208
    ];
    let XmY_ref: Array3<f64> = array![
        [
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [1.0000000000000000e+00, 0.0000000000000000e+00]
        ],
        [
            [1.5749879276249671e-16, -1.1816552815381077e-16],
            [-4.3162985565594223e-16, 1.6579200843862024e-16],
            [1.5322713275954383e-15, 1.4406376429873231e-15],
            [0.0000000000000000e+00, 9.9999999999999989e-01]
        ],
        [
            [-2.1571381149267595e-02, 3.0272950957152545e-07],
            [-2.9991274781835619e-05, -1.4821884853203318e-01],
            [-9.9507889372419100e-01, -3.2471746790841304e-06],
            [0.0000000000000000e+00, 1.8197135800568093e-15]
        ],
        [
            [-1.1109010931921683e-06, 1.2585514282188594e-02],
            [3.2069605724216904e-01, -2.8289343610101130e-05],
            [-9.1191885320066824e-06, 9.4937779050842697e-01],
            [0.0000000000000000e+00, -1.1660604437441687e-15]
        ]
    ];
    let XpY_ref: Array3<f64> = array![
        [
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [1.0000000000000000e+00, 0.0000000000000000e+00]
        ],
        [
            [2.7538927963428562e-16, -2.1530403716360706e-16],
            [-5.0089199746684879e-16, 2.0458805099784026e-16],
            [1.5766701880603285e-15, 1.5883226278758355e-15],
            [0.0000000000000000e+00, 9.9999999999999989e-01]
        ],
        [
            [-3.6031757025210311e-02, 5.2693108223585217e-07],
            [-3.3247977273169895e-05, -1.7472610444661563e-01],
            [-9.7813856605377103e-01, -3.4200094284606595e-06],
            [0.0000000000000000e+00, 1.7383640140772968e-15]
        ],
        [
            [-1.7446652975011026e-06, 2.0596777031218448e-02],
            [3.3426673906312915e-01, -3.1354975875524647e-05],
            [-8.4280732861148150e-06, 9.4013443503872907e-01],
            [0.0000000000000000e+00, -1.0473407245945591e-15]
        ]
    ];

    println!("valorbs {:?}", mol.calculator.valorbs);
    println!("atomic numbers {:?}", mol.atomic_numbers);

    mol.calculator.set_active_orbitals(f_occ.to_vec());

    let (omega_out, c_ij, XmY, XpY) =
        get_exc_energies(&f_occ.to_vec(), &mol, Some(4), &S, &orbe, &orbs, false,None);
    println!("omega_out{}", &omega_out);
    println!("omega_diff {}", &omega_out - &omega_ref_out);
    assert!(omega_out.abs_diff_eq(&omega_ref_out, 1e-10));
    assert!((&XpY * &XpY).abs_diff_eq(&(&XpY_ref * &XpY_ref), 1e-10));
    assert!((&XmY * &XmY).abs_diff_eq(&(&XmY_ref * &XmY_ref), 1e-10));
}

#[test]
fn excited_energies_non_hermitian_davidson_routine() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mut mol: Molecule =
        Molecule::new(atomic_numbers, positions, charge, multiplicity, None, None);

    let S: Array2<f64> = array![
        [
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.3074918525690681,
            0.3074937992389065
        ],
        [
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.1987769748092704
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            -0.3185054221819456
        ],
        [
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            1.0000000000000000,
            -0.3982160222204482,
            0.1327383036929333
        ],
        [
            0.3074918525690681,
            0.0000000000000000,
            0.0000000000000000,
            -0.3982160222204482,
            1.0000000000000000,
            0.0268024699984349
        ],
        [
            0.3074937992389065,
            -0.1987769748092704,
            -0.3185054221819456,
            0.1327383036929333,
            0.0268024699984349,
            1.0000000000000000
        ]
    ];
    let f_occ: Array1<f64> = array![
        2.0000000000000000,
        2.0000000000000000,
        2.0000000000000000,
        2.0000000000000000,
        0.0000000000000000,
        0.0000000000000000
    ];
    let orbe: Array1<f64> = array![
        -0.8274698453897238,
        -0.4866977301135242,
        -0.4293504173916476,
        -0.3805317623354740,
        0.4597732058522524,
        0.5075648555895222
    ];
    let orbs: Array2<f64> = array![
        [
            8.7633817073094666e-01,
            -7.3282333485870987e-07,
            -2.5626946551477237e-01,
            6.5297360604596939e-16,
            4.4638746429764842e-05,
            6.5169208620110397e-01
        ],
        [
            1.5609825393253833e-02,
            -1.9781346650257903e-01,
            -3.5949496391505154e-01,
            -8.4834397825097185e-01,
            2.8325035872464288e-01,
            -2.9051011756322170e-01
        ],
        [
            2.5012021798979999e-02,
            -3.1696156822051985e-01,
            -5.7602795979721200e-01,
            5.2944545948125277e-01,
            4.5385928211929161e-01,
            -4.6549177907241801e-01
        ],
        [
            2.0847651645102452e-02,
            5.2838144790877872e-01,
            -4.8012913249889100e-01,
            7.0500141149039645e-17,
            -7.5667687027915898e-01,
            -3.8791257751171815e-01
        ],
        [
            1.6641905232449239e-01,
            -3.7146604214646906e-01,
            2.5136102811674521e-01,
            3.5209333698603627e-16,
            -7.2004450606557047e-01,
            -7.6949321868972731e-01
        ],
        [
            1.6641962261695781e-01,
            3.7146556016199661e-01,
            2.5135992631728726e-01,
            -1.6703880555921563e-15,
            7.1993590540308161e-01,
            -7.6959837575446732e-01
        ]
    ];
    let omega_ref_out: Array1<f64> = array![
        0.5639270376740008,
        0.6133146373942723,
        0.6284386094838724,
        0.6699518396823421
    ];
    let XmY_ref: Array3<f64> = array![
        [
            [-4.9249277103966668e-17, 3.7892928547413248e-17],
            [-1.0517031522236221e-15, 1.9719482746270877e-16],
            [-8.2761071936219020e-16, 2.3620763020793745e-16],
            [-9.9999999999555733e-01, 2.9807538851008475e-06]
        ],
        [
            [-8.3914788009812633e-16, 2.4995950188707747e-16],
            [-1.3173024417428260e-16, 9.3595560741998794e-16],
            [-1.8411762384703860e-15, 6.5209187692439883e-16],
            [2.9807538851008459e-06, 9.9999999999555689e-01]
        ],
        [
            [3.4096336997072256e-02, -6.1258454882382015e-07],
            [-3.9780204908382161e-05, -1.7222258226335058e-01],
            [-9.9024005503008039e-01, -2.9144620375845891e-06],
            [8.0825795534993625e-16, -1.6572860696634805e-15]
        ],
        [
            [2.5364612087415368e-06, -1.4682256488048162e-02],
            [4.6763207818763614e-01, -3.7130880369572095e-05],
            [-1.5649698452839685e-05, 8.8583534829328858e-01],
            [-2.7850331681112818e-16, -5.1544058725789784e-16]
        ]
    ];
    let XpY_ref: Array3<f64> = array![
        [
            [-7.7629394175718453e-17, 6.9019789333783619e-17],
            [-1.1170333561598300e-15, 2.6281326590700557e-16],
            [-8.6456668799848029e-16, 2.7642681141955694e-16],
            [-9.9999999999555789e-01, 2.9807538851008471e-06]
        ],
        [
            [-5.1523873457254240e-16, 1.7783283240356011e-16],
            [-1.7075704089918643e-16, 9.2558982183857637e-16],
            [-1.8507807229027217e-15, 6.7637778744855361e-16],
            [2.9807538851008514e-06, 9.9999999999555778e-01]
        ],
        [
            [4.7420378519552261e-02, -8.2900487899391707e-07],
            [-4.2985498977029308e-05, -2.0184926468961226e-01],
            [-9.7311771078766396e-01, -3.1099380179331866e-06],
            [7.6326337681660012e-16, -1.5645471139288654e-15]
        ],
        [
            [3.5195127523310302e-06, -1.9781980905012764e-02],
            [4.7198640639971629e-01, -4.0288485620713528e-05],
            [-1.4408580151172960e-05, 8.7938867144102506e-01],
            [-2.8941857292646342e-16, -5.0494354860756587e-16]
        ]
    ];

    println!("valorbs {:?}", mol.calculator.valorbs);
    println!("atomic numbers {:?}", mol.atomic_numbers);

    mol.calculator.set_active_orbitals(f_occ.to_vec());

    let (omega_out, c_ij, XmY, XpY) =
        get_exc_energies(&f_occ.to_vec(), &mol, Some(4), &S, &orbe, &orbs, false, None);
    println!("omega_out{}", &omega_out);
    println!("omega_diff {}", &omega_out - &omega_ref_out);
    assert!(omega_out.abs_diff_eq(&omega_ref_out, 1e-10));
    assert!((&XpY * &XpY).abs_diff_eq(&(&XpY_ref * &XpY_ref), 1e-10));
    assert!((&XmY * &XmY).abs_diff_eq(&(&XmY_ref * &XmY_ref), 1e-10));
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
    let omega_ref: Array1<f64> = array![
        0.3837776010960228,
        0.4376185583677501,
        0.4777844855653459,
        0.529392732956824
    ];
    let c_ij_ref: Array3<f64> = array![
        [
            [7.1048609280539423e-16, -1.2491679807793276e-17],
            [-9.9999999999684230e-01, 2.5130863202938214e-06]
        ],
        [
            [1.0807435052922551e-16, -7.0715300399956809e-16],
            [2.5130863202798792e-06, 9.9999999999684219e-01]
        ],
        [
            [-9.9999999999915401e-01, 1.3008338757459926e-06],
            [-7.1048583744966613e-16, 1.0807705593029713e-16]
        ],
        [
            [-1.3008338757459926e-06, -9.9999999999915401e-01],
            [1.2488978235912678e-17, -7.0715289477295315e-16]
        ]
    ];

    // test m matrix
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    // let multiplicity: Option<u8> = Some(1);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(
        atomic_numbers.clone(),
        positions,
        charge,
        multiplicity,
        None,
        None,
    );

    let m: Array4<f64> = m_atomwise(&atomic_numbers[..], 3, mol.calculator.spin_couplings.view());
    // println!("M_matrix {}", m);

    let (omega, c_ij): (Array1<f64>, Array3<f64>) = tda(
        gamma.view(),
        gamma_lr.view(),
        q_trans_ov.view(),
        q_trans_oo.view(),
        q_trans_vv.view(),
        omega_0.view(),
        df.view(),
        mol.multiplicity,
        2,
        2,
        None,
    );
    println!("omega {}", omega);
    println!("omega_ref {}", omega_ref);
    assert!(omega.abs_diff_eq(&omega_ref, 1e-14));
    assert!(c_ij.abs_diff_eq(&c_ij_ref, 1e-14));


    let (omega_spin, c_ij_spin): (Array1<f64>, Array3<f64>) = tda(
        gamma.view(),
        gamma_lr.view(),
        q_trans_ov.view(),
        q_trans_oo.view(),
        q_trans_vv.view(),
        omega_0.view(),
        df.view(),
        mol.multiplicity,
        2,
        2,
        Some(m.view()),
    );
    println!("omega {}", omega);
    println!("omega_spin {}", omega);
    println!("c_ij {}", c_ij);
    println!("c_ij_spin {}", c_ij_spin);
}

#[test]
fn casida_routine() {
    let active_occupied_orbs: Vec<usize> = vec![2, 3];
    let active_virtual_orbs: Vec<usize> = vec![4, 5];

    let q_trans_ov: Array3<f64> = array![
        [
            [2.6230102760843588e-05, 3.7065690068981005e-01],
            [-2.7525960675946095e-16, 4.6732329761403721e-16]
        ],
        [
            [-1.7348148585808076e-01, -1.8531670208019752e-01],
            [-3.0454108812507206e-17, -3.2441273916737827e-17]
        ],
        [
            [1.7345525575531995e-01, -1.8534019860961240e-01],
            [2.5834633276981204e-16, -2.7612790453676845e-16]
        ]
    ];
    let q_trans_oo: Array3<f64> = array![
        [
            [8.3509736370957022e-01, -2.7316210319124923e-16],
            [-2.7316210319124923e-16, 1.0000000000000002e+00]
        ],
        [
            [8.2451688036773718e-02, 1.0848820325105576e-17],
            [1.0848820325105576e-17, 6.1352482879196894e-34]
        ],
        [
            [8.2450948253655232e-02, 1.2600837578629461e-16],
            [1.2600837578629461e-16, 1.9194032338495546e-31]
        ]
    ];
    let q_trans_vv: Array3<f64> = array![
        [
            [4.1303024748498873e-01, -5.9780479099297290e-06],
            [-5.9780479099019734e-06, 3.2642073579136882e-01]
        ],
        [
            [2.9352849855153712e-01, 3.1440135449565659e-01],
            [3.1440135449565659e-01, 3.3674597810671997e-01]
        ],
        [
            [2.9344125396347287e-01, -3.1439537644774673e-01],
            [-3.1439537644774673e-01, 3.3683328610191055e-01]
        ]
    ];
    let orbe: Array1<f64> = array![
        -0.8688947761291697,
        -0.4499995482542979,
        -0.3563328959810791,
        -0.2833078663602301,
        0.3766539028637694,
        0.4290382785534344
    ];
    let df: Array2<f64> = array![
        [2.0000000000000000, 2.0000000000000000],
        [2.0000000000000000, 2.0000000000000000]
    ];
    let omega_0: Array2<f64> = array![
        [0.7329867988448485, 0.7853711745345135],
        [0.6599617692239996, 0.7123461449136645]
    ];
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
    let omega_ref: Array1<f64> = array![
        0.3837835356343971,
        0.4376253291429422,
        0.4774964635767988,
        0.5291687613328181
    ];
    let c_ij_ref: Array3<f64> = array![
        [
            [4.7223315823513485e-16, -7.4981668357365175e-17],
            [-9.9999999999684264e-01, 2.5129944999369611e-06]
        ],
        [
            [2.9883738040491793e-16, -6.5359985443162102e-16],
            [2.5129944998536033e-06, 9.9999999999684230e-01]
        ],
        [
            [9.9999999999916322e-01, -1.2936531432711354e-06],
            [4.7223250425470150e-16, -2.9883941265480345e-16]
        ],
        [
            [-1.2936531432711352e-06, -9.9999999999916322e-01],
            [7.4979415220774157e-17, -6.5359965632186451e-16]
        ]
    ];
    let XmY_ref: Array3<f64> = array![
        [
            [4.2111943079100728e-16, -6.5114982514486065e-17],
            [-9.9999999999684130e-01, 2.5129944998727014e-06]
        ],
        [
            [2.9139693953327924e-16, -6.4476630021043831e-16],
            [2.5129944999671646e-06, 9.9999999999684253e-01]
        ],
        [
            [1.0176480455825823e+00, -1.3052877770497048e-06],
            [5.0072430918042775e-16, -3.1215602389124885e-16]
        ],
        [
            [-1.3246100614773253e-06, -1.0148191825376125e+00],
            [8.7676274449557100e-17, -6.7299910891989426e-16]
        ]
    ];
    let XpY_ref: Array3<f64> = array![
        [
            [4.6639739281269028e-16, -8.6398264263935805e-17],
            [-9.9999999999684286e-01, 2.5129944999159119e-06]
        ],
        [
            [3.0696981526418628e-16, -6.6317163323567491e-16],
            [2.5129944998949877e-06, 9.9999999999684230e-01]
        ],
        [
            [9.8265800670392522e-01, -1.2826311376664284e-06],
            [4.4628464004494760e-16, -2.8609089320744545e-16]
        ],
        [
            [-1.2639212061172028e-06, -9.8539721874173636e-01],
            [6.4161949064153771e-17, -6.3535071248490153e-16]
        ]
    ];

    let R_ref: Array2<f64> = array![
        [
            2.2800287272843536e-01,
            6.7291574249027377e-08,
            -8.0918247294654886e-18,
            4.5208604071418659e-18
        ],
        [
            6.7291574245366408e-08,
            2.8001957797042176e-01,
            7.4756755570666384e-18,
            -1.0527565605136828e-17
        ],
        [
            -4.3611325146181644e-18,
            7.4756759226502550e-18,
            1.4728980222431723e-01,
            1.1114001261467662e-07
        ],
        [
            4.3449084290177176e-18,
            -1.0527565620779261e-17,
            1.1114001261503358e-07,
            1.9151592870718942e-01
        ]
    ];

    let (omega, c_ij, XmY, XpY): (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = casida(
        gamma.view(),
        gamma_lr.view(),
        q_trans_ov.view(),
        q_trans_oo.view(),
        q_trans_vv.view(),
        omega_0.view(),
        df.view(),
        1,
        2,
        2,
        None,
    );

    assert!(omega.abs_diff_eq(&omega_ref, 1e-14));
    assert!((&c_ij * &c_ij).abs_diff_eq(&(&c_ij_ref * &c_ij_ref), 1e-14));
    assert!((&XmY * &XmY).abs_diff_eq(&(&XmY_ref * &XmY_ref), 1e-14));
    assert!((&XpY * &XpY).abs_diff_eq(&(&XpY_ref * &XpY_ref), 1e-14));
}

#[test]
fn hermitian_davidson_routine() {
    let active_occupied_orbs: Vec<usize> = vec![2, 3];
    let active_virtual_orbs: Vec<usize> = vec![4, 5];
    let qtrans_ov: Array3<f64> = array![
        [
            [2.6230102760982366e-05, 3.7065690068980978e-01],
            [1.9991274594610739e-16, 1.5626227190095965e-16]
        ],
        [
            [-1.7348148585808104e-01, -1.8531670208019754e-01],
            [-1.2772046586865093e-16, -1.3655275046819219e-16]
        ],
        [
            [1.7345525575532011e-01, -1.8534019860961229e-01],
            [3.1533607568650388e-17, -3.3646128491374340e-17]
        ]
    ];
    let qtrans_oo: Array3<f64> = array![
        [
            [8.3509736370957066e-01, -1.0278107627479084e-17],
            [-1.0278107627479084e-17, 1.0000000000000000e+00]
        ],
        [
            [8.2451688036773829e-02, 6.5469849226515001e-17],
            [6.5469849226515001e-17, 5.0578106550731340e-32]
        ],
        [
            [8.2450948253655273e-02, 1.3061299549224103e-17],
            [1.3061299549224103e-17, 1.8388700816055480e-33]
        ]
    ];
    let qtrans_vv: Array3<f64> = array![
        [
            [4.1303024748498973e-01, -5.9780479099574846e-06],
            [-5.9780479099574846e-06, 3.2642073579136843e-01]
        ],
        [
            [2.9352849855153762e-01, 3.1440135449565670e-01],
            [3.1440135449565670e-01, 3.3674597810671958e-01]
        ],
        [
            [2.9344125396347337e-01, -3.1439537644774673e-01],
            [-3.1439537644774673e-01, 3.3683328610190999e-01]
        ]
    ];
    let orbe: Array1<f64> = array![
        -0.8688947761291694,
        -0.4499995482542977,
        -0.3563328959810789,
        -0.2833078663602301,
        0.3766539028637694,
        0.4290382785534342
    ];
    let df: Array2<f64> = array![
        [2.0000000000000000, 2.0000000000000000],
        [2.0000000000000000, 2.0000000000000000]
    ];
    let omega0: Array2<f64> = array![
        [0.7329867988448483, 0.7853711745345131],
        [0.6599617692239994, 0.7123461449136643]
    ];
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
    let omega_ref: Array1<f64> = array![
        0.6599617692239994,
        0.7123461449136644,
        0.7524123662424658,
        0.8028450373713394
    ];
    let c_ij_ref: Array3<f64> = array![
        [
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [1.0000000000000000e+00, 0.0000000000000000e+00]
        ],
        [
            [1.9562325544462273e-15, 6.8513481021464656e-16],
            [0.0000000000000000e+00, -1.0000000000000002e+00]
        ],
        [
            [9.9999999999975908e-01, 6.9448266247001835e-07],
            [0.0000000000000000e+00, 2.0815153876780652e-15]
        ],
        [
            [-6.9448266231396227e-07, 9.9999999999975908e-01],
            [0.0000000000000000e+00, 9.8876903715205364e-16]
        ]
    ];
    let XmY_ref: Array3<f64> = array![
        [
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [1.0000000000000000e+00, 0.0000000000000000e+00]
        ],
        [
            [1.9284924570627492e-15, 6.5250537593655418e-16],
            [0.0000000000000000e+00, -1.0000000000000004e+00]
        ],
        [
            [1.0131643171258662e+00, 6.7975418789671723e-07],
            [0.0000000000000000e+00, 2.1392525039507682e-15]
        ],
        [
            [-7.2682389246313837e-07, 1.0110633895644352e+00],
            [0.0000000000000000e+00, 1.0496999648733084e-15]
        ]
    ];
    let XpY_ref: Array3<f64> = array![
        [
            [0.0000000000000000e+00, 0.0000000000000000e+00],
            [1.0000000000000000e+00, 0.0000000000000000e+00]
        ],
        [
            [1.9843716749111940e-15, 7.1939592450729834e-16],
            [0.0000000000000000e+00, -1.0000000000000000e+00]
        ],
        [
            [9.8700673039523112e-01, 7.0953026411472090e-07],
            [0.0000000000000000e+00, 2.0253365608496104e-15]
        ],
        [
            [-6.6358050864315742e-07, 9.8905766969795728e-01],
            [0.0000000000000000e+00, 9.3137490858980512e-16]
        ]
    ];
    let Oia: Array2<f64> = array![
        [0.9533418513628125, 0.9300153388504570],
        [0.9184048617688078, 0.8886374235432982]
    ];

    let bs_first_ref: Array3<f64> = array![
        [
            [
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000
            ]
        ],
        [
            [
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ],
            [
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000
            ]
        ]
    ];
    let rbs_ref: Array3<f64> = array![
        [
            [
                1.2569009705584559e-17,
                8.4381698891878669e-18,
                5.6612436887462425e-01,
                -5.4472293013708636e-08
            ],
            [
                1.3288865429416234e-17,
                1.1202300248704291e-17,
                -5.4472293009588668e-08,
                6.4456015403174938e-01
            ]
        ],
        [
            [
                4.3554953683727143e-01,
                7.4934563263903860e-33,
                1.2569009705584556e-17,
                1.3288865429416238e-17
            ],
            [
                7.4934563263903874e-33,
                5.0743703017335928e-01,
                8.4381698891878731e-18,
                1.1202300248704302e-17
            ]
        ]
    ];

    let n_occ: usize = qtrans_oo.dim().1;
    let n_virt: usize = qtrans_vv.dim().1;

    let (omega, c_ij, XmY, XpY): (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
        hermitian_davidson(
            gamma.view(),
            qtrans_ov.view(),
            omega0.view(),
            (&omega0 * 0.0).view(),
            n_occ,
            n_virt,
            None,
            None,
            Oia.view(),
            1,
            None,
            None,
            None,
            None,
            None,
        );

    println!("omega {}", omega);
    println!("omega_diff {}", &omega - &omega_ref);
    assert!(omega.abs_diff_eq(&omega_ref, 1e-14));
    assert!((&c_ij * &c_ij).abs_diff_eq(&(&c_ij_ref * &c_ij_ref), 1e-14));
    assert!((&XmY * &XmY).abs_diff_eq(&(&XmY_ref * &XmY_ref), 1e-14));
    assert!((&XpY * &XpY).abs_diff_eq(&(&XpY_ref * &XpY_ref), 1e-14));
}

#[test]
fn non_hermitian_davidson_routine() {
    let active_occupied_orbs: Vec<usize> = vec![2, 3];
    let active_virtual_orbs: Vec<usize> = vec![4, 5];
    let qtrans_ov: Array3<f64> = array![
        [
            [2.6230102760982366e-05, 3.7065690068980978e-01],
            [1.9991274594610739e-16, 1.5626227190095965e-16]
        ],
        [
            [-1.7348148585808104e-01, -1.8531670208019754e-01],
            [-1.2772046586865093e-16, -1.3655275046819219e-16]
        ],
        [
            [1.7345525575532011e-01, -1.8534019860961229e-01],
            [3.1533607568650388e-17, -3.3646128491374340e-17]
        ]
    ];
    let qtrans_oo: Array3<f64> = array![
        [
            [8.3509736370957066e-01, -1.0278107627479084e-17],
            [-1.0278107627479084e-17, 1.0000000000000000e+00]
        ],
        [
            [8.2451688036773829e-02, 6.5469849226515001e-17],
            [6.5469849226515001e-17, 5.0578106550731340e-32]
        ],
        [
            [8.2450948253655273e-02, 1.3061299549224103e-17],
            [1.3061299549224103e-17, 1.8388700816055480e-33]
        ]
    ];
    let qtrans_vv: Array3<f64> = array![
        [
            [4.1303024748498973e-01, -5.9780479099574846e-06],
            [-5.9780479099574846e-06, 3.2642073579136843e-01]
        ],
        [
            [2.9352849855153762e-01, 3.1440135449565670e-01],
            [3.1440135449565670e-01, 3.3674597810671958e-01]
        ],
        [
            [2.9344125396347337e-01, -3.1439537644774673e-01],
            [-3.1439537644774673e-01, 3.3683328610190999e-01]
        ]
    ];
    let orbe: Array1<f64> = array![
        -0.8688947761291694,
        -0.4499995482542977,
        -0.3563328959810789,
        -0.2833078663602301,
        0.3766539028637694,
        0.4290382785534342
    ];
    let df: Array2<f64> = array![
        [2.0000000000000000, 2.0000000000000000],
        [2.0000000000000000, 2.0000000000000000]
    ];
    let omega0: Array2<f64> = array![
        [0.7329867988448483, 0.7853711745345131],
        [0.6599617692239994, 0.7123461449136643]
    ];
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
    let omega_ref: Array1<f64> = array![
        0.3837835356343960,
        0.4376253291429426,
        0.4774964635767985,
        0.5291687613328180
    ];
    let c_ij_ref: Array3<f64> = array![
        [
            [-5.2180034485911912e-17, 5.6082224559370855e-17],
            [-9.9999999999684241e-01, 2.5129945001804835e-06]
        ],
        [
            [-1.2390345436978378e-16, 1.1895052146880136e-16],
            [2.5129945001804835e-06, 9.9999999999684241e-01]
        ],
        [
            [9.9999999999916311e-01, -1.2936531445026286e-06],
            [-5.2179795667563561e-17, 1.2390373936410043e-16]
        ],
        [
            [1.2936531448551817e-06, 9.9999999999916300e-01],
            [5.6081858134677710e-17, -1.1895050209136492e-16]
        ]
    ];
    let XmY_ref: Array3<f64> = array![
        [
            [-5.6772347066809759e-17, 4.8468634033969528e-17],
            [-9.9999999999684230e-01, 2.5129945001804835e-06]
        ],
        [
            [-1.2112549430193502e-16, 1.2734724110626606e-16],
            [2.5129945001804840e-06, 9.9999999999684253e-01]
        ],
        [
            [1.0176480455825818e+00, -1.3052877785037311e-06],
            [-5.5847456961446496e-17, 1.2899211967600493e-16]
        ],
        [
            [1.3246100628325871e-06, 1.0148191825376121e+00],
            [6.5903739366506471e-17, -1.1147999851952230e-16]
        ]
    ];
    let XpY_ref: Array3<f64> = array![
        [
            [-4.7146468204433370e-17, 6.4891796215541190e-17],
            [-9.9999999999684286e-01, 2.5129945001804844e-06]
        ],
        [
            [-1.2675488178927564e-16, 1.0985208286048134e-16],
            [2.5129945001804848e-06, 9.9999999999684253e-01]
        ],
        [
            [9.8265800670392589e-01, -1.2826311389546797e-06],
            [-6.5578260912368928e-17, 1.1902524116707512e-16]
        ],
        [
            [1.2639212075700780e-06, 9.8539721874173636e-01],
            [4.7830296835723485e-17, -1.2548758415483460e-16]
        ]
    ];

    let n_occ: usize = qtrans_oo.dim().1;
    let n_virt: usize = qtrans_vv.dim().1;

    let (omega, c_ij, XmY, XpY): (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
        non_hermitian_davidson(
            gamma.view(),
            gamma_lr.view(),
            qtrans_oo.view(),
            qtrans_vv.view(),
            qtrans_ov.view(),
            omega0.view(),
            n_occ,
            n_virt,
            None,
            None,
            None,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
        );

    println!("omega {}", omega);
    println!("omega_diff {}", &omega - &omega_ref);
    assert!(omega.abs_diff_eq(&omega_ref, 1e-14));
    assert!((&c_ij * &c_ij).abs_diff_eq(&(&c_ij_ref * &c_ij_ref), 1e-14));
    assert!((&XmY * &XmY).abs_diff_eq(&(&XmY_ref * &XmY_ref), 1e-14));
    assert!((&XpY * &XpY).abs_diff_eq(&(&XpY_ref * &XpY_ref), 1e-14));
}


#[test]
fn ethylene_tda() {
    // Define ethylene
    let atomic_numbers: Vec<u8> = vec![6, 6, 1, 1, 1, 1];
    let mut positions: Array2<f64> = array![
        [ 0.6579, -0.0045,  0.0639],
        [-0.6579,  0.0045, -0.0639],
        [ 1.1610,  0.0661,  1.0238],
        [ 1.3352, -0.0830, -0.7815],
        [-1.3355,  0.0830,  0.7812],
        [-1.1608, -0.0661, -1.0239]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(3);
    let mut mol: Molecule = Molecule::new(
        atomic_numbers.clone(),
        positions,
        charge,
        multiplicity,
        None,
        Some((2, 2)),
    );

    let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
        run_scc(&mol, None, None, None);

    mol.calculator.set_active_orbitals(f.clone());

    let active_occ: Vec<usize> = mol.calculator.active_occ.clone().unwrap();
    let active_virt: Vec<usize> = mol.calculator.active_virt.clone().unwrap();
    let n_occ = active_occ.len();
    let n_virt = active_virt.len();

    let gamma: Array2<f64> = (&mol.calculator.g0).to_owned();
    let gamma_lr: Array2<f64> = (&mol.calculator.g0_lr).to_owned();

    let (q_trans_ov, q_trans_oo, q_trans_vv): (Array3<f64>, Array3<f64>, Array3<f64>) = trans_charges(
        &mol.atomic_numbers,
        &mol.calculator.valorbs,
        orbs.view(),
        s.view(),
        &active_occ[..],
        &active_virt[..],
    );

    let omega_0: Array2<f64> = get_orbital_en_diff(
        orbe.view(),
        n_occ,
        n_virt,
        &active_occ[..],
        &active_virt[..],
    );

    let df: Array2<f64> = get_orbital_occ_diff(
        Array::from(f.clone()).view(),
        n_occ,
        n_virt,
        &active_occ[..],
        &active_virt[..],
    );

    // let m: Array4<f64> = m_atomwise(&atomic_numbers[..], mol.n_atoms,
    //                                 mol.calculator.spin_couplings.view());
    //
    // let (omega, c_ij): (Array1<f64>, Array3<f64>) = tda(
    //     gamma.view(),
    //     gamma_lr.view(),
    //     q_trans_ov.view(),
    //     q_trans_oo.view(),
    //     q_trans_vv.view(),
    //     omega_0.view(),
    //     df.view(),
    //     mol.multiplicity,
    //     n_occ,
    //     n_virt,
    // );
    //
    //
    // let (omega_spin, c_ij_spin): (Array1<f64>, Array3<f64>) = tda_spin(
    //     gamma.view(),
    //     gamma_lr.view(),
    //     m.view(),
    //     q_trans_ov.view(),
    //     q_trans_oo.view(),
    //     q_trans_vv.view(),
    //     omega_0.view(),
    //     df.view(),
    //     mol.multiplicity,
    //     n_occ,
    //     n_virt,
    // );
    // println!("omega {}", omega);
    // println!("omega_spin {}", omega_spin);

    let (omega, c_ij, XmY, XpY) =
        get_exc_energies(&f.to_vec(), &mol, None, &s, &orbe, &orbs, false, None);

    let (omega_magn, c_ij_magn, XmY_magn, XpY_magn) =
        get_exc_energies(&f.to_vec(), &mol, None, &s, &orbe, &orbs, true, None);


    println!("omega: {}", &omega);
    println!("omega_magn: {}", &omega_magn);

    assert_eq!(1, 2);

}