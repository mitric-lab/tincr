use crate::calculator::{get_gamma_gradient_matrix, lambda2_calc_oia};
use crate::defaults;
use crate::gradients;
use crate::io::GeneralConfig;
use crate::scc_routine::*;
use crate::test::{get_benzene_molecule, get_water_molecule};
use crate::transition_charges::trans_charges;
use crate::Molecule;
use approx::{AbsDiffEq, RelativeEq};
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::{stack, Array2, Array4, ArrayView1, ArrayView2, ArrayView3};
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use peroxide::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::ops::{AddAssign, DivAssign};
use std::time::Instant;
use arpack_interface::*;
use ndarray_linalg::krylov::arnoldi_mgs;

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
    let mut gamma0_lr: Array2<f64> = Array::zeros((molecule.g0.shape()))
        .into_dimensionality::<Ix2>()
        .unwrap();

    let r_lr = molecule
        .calculator
        .r_lr
        .unwrap_or(defaults::LONG_RANGE_RADIUS);

    if r_lr == 0.0 {
        gamma0_lr = &molecule.g0 * 0.0;
    } else {
        gamma0_lr = molecule.g0_lr.clone();
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

    let mut omega_out: Array1<f64> = Array::zeros((n_occ * n_virt));
    let mut c_ij: Array3<f64> = Array::zeros((n_occ * n_virt, n_occ, n_virt));
    let mut XpY: Array3<f64> = Array::zeros((n_occ * n_virt, n_occ, n_virt));
    let mut XmY: Array3<f64> = Array::zeros((n_occ * n_virt, n_occ, n_virt));

    // check if complete nstates is used
    if complete_states {
        // check if Tamm-Dancoff is demanded
        if response_method.is_some() && response_method.unwrap() == "TDA" {
            println!("TDA routine called!");
            let tmp: (Array1<f64>, Array3<f64>) = tda(
                (&molecule.g0).view(),
                gamma0_lr.view(),
                qtrans_ov.view(),
                qtrans_oo.view(),
                qtrans_vv.view(),
                omega.view(),
                df.view(),
                molecule.multiplicity,
                n_occ,
                n_virt,
                molecule.calculator.spin_couplings.view(),
            );
            omega_out = tmp.0;
            c_ij = tmp.1;
        } else {
            println!("Casida routine called!");
            let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = casida(
                (&molecule.g0).view(),
                gamma0_lr.view(),
                qtrans_ov.view(),
                qtrans_oo.view(),
                qtrans_vv.view(),
                omega.view(),
                df.view(),
                molecule.multiplicity,
                n_occ,
                n_virt,
                molecule.calculator.spin_couplings.view(),
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
            println!("Hermitian Davidson routine called!");
            // calculate o_ia
            let o_ia: Array2<f64> =
                lambda2_calc_oia(molecule, &active_occ, &active_virt, &qtrans_oo, &qtrans_vv);
            // use hermitian davidson routine, only possible with lc off
            let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = hermitian_davidson(
                (&molecule.g0).view(),
                qtrans_ov.view(),
                omega.view(),
                (0.0 * &omega).view(),
                n_occ,
                n_virt,
                None,
                None,
                o_ia.view(),
                molecule.multiplicity,
                molecule.calculator.spin_couplings.view(),
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
            println!("non-Hermitian Davidson routine called!");
            let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = non_hermitian_davidson(
                (&molecule.g0).view(),
                (&molecule.g0_lr).view(),
                qtrans_oo.view(),
                qtrans_vv.view(),
                qtrans_ov.view(),
                omega.view(),
                n_occ,
                n_virt,
                None,
                None,
                None,
                molecule.multiplicity,
                molecule.calculator.spin_couplings.view(),
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

// pub fn build_a_matrix(
//     gamma: ArrayView2<f64>,
//     gamma_lr: ArrayView2<f64>,
//     q_trans_ov: ArrayView3<f64>,
//     q_trans_oo: ArrayView3<f64>,
//     q_trans_vv: ArrayView3<f64>,
//     omega: ArrayView2<f64>,
//     df: ArrayView2<f64>,
//     multiplicity: u8,
//     m: Option<ArrayView4<f64>>,
// ) -> (Array2<f64>) {
//     let n_occ: usize = q_trans_oo.dim().1;
//     let n_virt: usize = q_trans_vv.dim().1;
//     let mut k_lr_a: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     let mut k_a: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     let mut k_singlet: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     // K_lr_A = np.tensordot(qtrans_oo, np.tensordot(gamma_lr, qtrans_vv, axes=(1,0)),axes=(0,0))
//
//     k_lr_a = tensordot(
//         &q_trans_oo,
//         &tensordot(&gamma_lr, &q_trans_vv, &[Axis(1)], &[Axis(0)]),
//         &[Axis(0)],
//         &[Axis(0)],
//     )
//     .into_dimensionality::<Ix4>()
//     .unwrap();
//
//     // K_lr_A = np.swapaxes(K_lr_A, 1, 2)
//     // swap axes still missing
//     k_lr_a.swap_axes(1, 2);
//     k_a.assign(&-k_lr_a);
//
//     if multiplicity == 1 {
//         //K_singlet = 2.0*np.tensordot(qtrans_ov, np.tensordot(gamma, qtrans_ov, axes=(1,0)),axes=(0,0))
//         //K_A += K_singlet
//         k_singlet = 2.0
//             * tensordot(
//                 &q_trans_ov,
//                 &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]),
//                 &[Axis(0)],
//                 &[Axis(0)],
//             )
//             .into_dimensionality::<Ix4>()
//             .unwrap();
//         // println!("k_singlet_NO_M {}", k_singlet);
//
//         k_a = k_a + k_singlet;
//     }
//
//     let mut k_coupling: Array2<f64> = k_a.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
//
//     // println!("k_coupling_no_M {}", k_coupling);
//
//     let k_m_coupling_red: Array2<f64> = if m.is_some() {
//         // Manipulate q_trans_ov
//         let mut q_trans_ovs: Array4<f64> = q_trans_ov.to_owned().insert_axis(Axis(3));
//         q_trans_ovs = stack![Axis(3), q_trans_ovs, q_trans_ovs];
//
//         // Get the correction matrix K_m
//         let delta_st: Array2<f64> = Array::eye(2);
//         let k_m: Array6<f64> = 2.0
//             * einsum(
//                 "aijs,bklt,abst,st->sijtkl",
//                 &[&q_trans_ovs, &q_trans_ovs, &m.unwrap(), &delta_st],
//             )
//             .unwrap()
//             .into_dimensionality::<Ix6>()
//             .unwrap()
//             .as_standard_layout()
//             .to_owned();
//
//         let k_m_coupling: Array2<f64> = k_m
//             .into_shape((2 * n_occ * n_virt, 2 * n_occ * n_virt))
//             .unwrap();
//         // println!("K_m_coupling {}", k_m_coupling);
//         // assert!(k_m_coupling.abs_diff_eq(&k_m_coupling_2, 1e-16));
//
//         k_m_coupling
//             .slice(s![0..n_occ * n_virt, 0..n_occ * n_virt])
//             .to_owned()
//     } else {
//         Array::zeros(k_coupling.raw_dim())
//     };
//
//     k_coupling = k_coupling + k_m_coupling_red;
//
//     let mut df_half: Array2<f64> =
//         Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
//     let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
//     return df_half.dot(&omega) + &df_half.dot(&k_coupling.dot(&df_half));
// }

pub fn build_a_matrix(
    gamma: ArrayView2<f64>,
    gamma_lr: ArrayView2<f64>,
    q_trans_ov: ArrayView3<f64>,
    q_trans_oo: ArrayView3<f64>,
    q_trans_vv: ArrayView3<f64>,
    omega: ArrayView2<f64>,
    df: ArrayView2<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array2<f64>) {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;

    // Calculate k_a_lr
    // K_A^{lr} = K_{ov,o'v'}°{lr} = - sum_{A,B} q_A^{oo'} \gamma_{AB}^{lr} q_B^{vv'}
    //                             = - sum_{A,B} q_A^{ik} \gamma_{AB}^{lr} q_B^{jl}
    // o = i, o' = k, v = j, v' = l
    // equivalent to einsum("aik,bjl,ab->ijkl", &[&q_trans_oo, &q_trans_vv, &gamma_lr])
    let mut k_a: Array4<f64> = -1.0
        * tensordot(
            &q_trans_oo,
            &tensordot(&gamma_lr, &q_trans_vv, &[Axis(1)], &[Axis(0)]), // ab,bjl->ajl
            &[Axis(0)],
            &[Axis(0)],
        ) // aik,ajl->ikjl
        .into_dimensionality::<Ix4>()
        .unwrap();
    k_a.swap_axes(1, 2); // ikjl->ijkl
    k_a = k_a.as_standard_layout().to_owned();

    if multiplicity == 1 {
        // calculate coulomb integrals for singlets
        // 2 * K_A =  2 K_{ov,o'v'} = 2 sum_{A,B} q_A^{ov} \gamma_{AB} q_B^{o'v'}
        //                          = 2 sum_{A,B} q_A^{ij} \gamma_{AB} q_B^{kl}
        // equivalent to einsum("aij,bkl,ab->ijkl", &[&q_trans_ov, &q_trans_ov, &gamma])
        let k_singlet: Array4<f64> = 2.0
            * tensordot(
                &q_trans_ov,
                &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkl->akl
                &[Axis(0)],
                &[Axis(0)],
            ) // aij,akl->ijkl
            .into_dimensionality::<Ix4>()
            .unwrap();
        k_a = k_a + k_singlet;
    } else if multiplicity == 3 {
        // calculate magnetic corrections for triplets
        // K_A^{magn} = 2 sum_{A,B} q_A^{ov} \delta_{AB} W_A q_B^{o'v'}
        //            = 2 sum_{A,B} q_A^{ij} \delta_{AB} W_A q_B^{kl}
        // equivalent to einsum("aij,bkl,ab,a->ijkl", &[&q_trans_ov, &q_trans_ov, &delta_ab, &spin_couplings])
        let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings); // ab,a->ab
        let k_triplet: Array4<f64> = 2.0
            * tensordot(
                &q_trans_ov,
                &tensordot(&spin_couplings_diag, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkl->akl
                &[Axis(0)],
                &[Axis(0)],
            ) // aij,akl->ijkl
            .into_dimensionality::<Ix4>()
            .unwrap();

        k_a = k_a + k_triplet;
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
    }

    let k_coupling: Array2<f64> = k_a.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();

    // println!("{:?}", k_coupling);

    let mut df_half: Array2<f64> =
        Array2::from_diag(&df.mapv(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
    let omega: Array2<f64> = Array2::from_diag(&omega.into_shape((n_occ * n_virt)).unwrap());
    return df_half.dot(&omega) + &df_half.dot(&k_coupling.dot(&df_half));
}

// pub fn build_b_matrix(
//     gamma: ArrayView2<f64>,
//     gamma_lr: ArrayView2<f64>,
//     q_trans_ov: ArrayView3<f64>,
//     q_trans_oo: ArrayView3<f64>,
//     q_trans_vv: ArrayView3<f64>,
//     omega: ArrayView2<f64>,
//     df: ArrayView2<f64>,
//     multiplicity: u8,
//     spin_couplings: ArrayView1<f64>,
// ) -> (Array2<f64>) {
//     let n_occ: usize = q_trans_oo.dim().1;
//     let n_virt: usize = q_trans_vv.dim().1;
//
//     let mut k_lr_b: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     let mut k_b: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     let mut k_singlet: Array4<f64> = Array4::zeros([n_occ, n_virt, n_occ, n_virt]);
//     //K_lr_B = np.tensordot(qtrans_ov, np.tensordot(gamma_lr, qtrans_ov, axes=(1,0)),axes=(0,0))
//     k_lr_b = tensordot(
//         &q_trans_ov,
//         &tensordot(&gamma_lr, &q_trans_ov, &[Axis(1)], &[Axis(0)]),
//         &[Axis(0)],
//         &[Axis(0)],
//     )
//     .into_dimensionality::<Ix4>()
//     .unwrap();
//     //# got K_ia_jb but we need K_ib_ja
//     //K_lr_B = np.swapaxes(K_lr_B, 1, 3)
//     k_lr_b.swap_axes(1, 3);
//     k_b.assign(&(-1.0 * k_lr_b));
//
//     if multiplicity == 1 {
//         //K_singlet = 2.0*np.tensordot(qtrans_ov, np.tensordot(gamma, qtrans_ov, axes=(1,0)),axes=(0,0))
//         //K_A += K_singlet
//         k_singlet = 2.0
//             * tensordot(
//                 &q_trans_ov,
//                 &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]),
//                 &[Axis(0)],
//                 &[Axis(0)],
//             )
//             .into_dimensionality::<Ix4>()
//             .unwrap();
//
//         println!("k_singlet: {:?}", &k_singlet);
//
//         k_b = k_b + k_singlet;
//     }
//     let mut k_coupling: Array2<f64> = k_b.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();
//
//     let mut df_half: Array2<f64> =
//         Array2::from_diag(&df.map(|x| x / 2.0).into_shape((n_occ * n_virt)).unwrap());
//     return df_half.dot(&k_coupling.dot(&df_half));
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
    spin_couplings: ArrayView1<f64>,
) -> (Array2<f64>) {
    let n_occ: usize = q_trans_oo.dim().1;
    let n_virt: usize = q_trans_vv.dim().1;

    // Calculate k_b_lr
    // K_B^{lr} = K_{ov,v'o'}°{lr} = - sum_{A,B} q_A^{ov'} \gamma_{AB}^{lr} q_B^{o'v}
    //            K_{ijlk}^{lr}    = - sum_{A,B} q_A^{il} \gamma_{AB}^{lr} q_B^{kj}
    // o = i, o' = k, v = j, v' = l
    // equivalent to einsum("ail,bkj,ab->ijkl", &[&q_trans_ov, &q_trans_ov, &gamma_lr])
    let mut k_b: Array4<f64> = -1.0
        * tensordot(
            &q_trans_ov,
            &tensordot(&gamma_lr, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkj->akj
            &[Axis(0)],
            &[Axis(0)],
        ) // ail,akj->ilkj
        .into_dimensionality::<Ix4>()
        .unwrap();
    k_b.swap_axes(1, 3); // ilkj->ijkl, ijkl is the correct order for all matrices
    k_b = k_b.as_standard_layout().to_owned();

    if multiplicity == 1 {
        // calculate coulomb integrals for singlets
        // 2 * K_B =  2 K_{ov,v'o'} = 2 sum_{A,B} q_A^{ov} \gamma_{AB} q_B^{o'v'} = 2 * K_A
        //                          = 2 sum_{A,B} q_A^{ij} \gamma_{AB} q_B^{kl}
        // equivalent to einsum("aij,bkl,ab->ijkl", &[&q_trans_ov, &q_trans_ov, &gamma])
        let k_singlet: Array4<f64> = 2.0
            * tensordot(
                &q_trans_ov,
                &tensordot(&gamma, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkl->akl
                &[Axis(0)],
                &[Axis(0)],
            ) // aij,akl->ijkl
            .into_dimensionality::<Ix4>()
            .unwrap();

        k_b = k_b + k_singlet;
    } else if multiplicity == 3 {
        // calculate magnetic corrections for triplets
        // K_B^{magn} = 2 sum_{A,B} q_A^{ov} \delta_{AB} W_A q_B^{o'v'} = K_A^{magn}
        //            = 2 sum_{A,B} q_A^{ij} \delta_{AB} W_A q_B^{kl}
        // equivalent to einsum("aij,bkl,ab,a->ijkl", &[&q_trans_ov, &q_trans_ov, &delta_ab, &spin_couplings])
        let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings); // ab,a->ab
        let k_triplet: Array4<f64> = 2.0
            * tensordot(
                &q_trans_ov,
                &tensordot(&spin_couplings_diag, &q_trans_ov, &[Axis(1)], &[Axis(0)]), // ab,bkl->akl
                &[Axis(0)],
                &[Axis(0)],
            ) // aij,akl->ijkl
            .into_dimensionality::<Ix4>()
            .unwrap();

        k_b = k_b + k_triplet;
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
    }

    let mut k_coupling: Array2<f64> = k_b.into_shape((n_occ * n_virt, n_occ * n_virt)).unwrap();

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

    // let excited_timer: Instant = Instant::now();
    let (omega2, F): (Array1<f64>, Array2<f64>) = R.eigh(UPLO::Lower).unwrap();

    // println!("{:>68} {:>8.6} s","elapsed time calculate eigh:",excited_timer.elapsed().as_secs_f32());
    // drop(excited_timer);
    // let excited_timer: Instant = Instant::now();
    //
    // let v:Array1<f64> = Array1::ones(R.dim().0);
    // let test = arnoldi_mgs(R,v,1e-9);
    // let tmp_test:(Array1<f64>,Array2<f64>) = test.1.eigh(UPLO::Lower).unwrap();
    // println!("{:>68} {:>8.6} s","elapsed time calculate arnoldi + eigh:",excited_timer.elapsed().as_secs_f32());
    // drop(excited_timer);
    //
    // println!("omega2 {}",omega2);
    // println!("omega ar {}",tmp_test.0);

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
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
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
    let conv: f64 = conv.unwrap_or(1.0e-8);
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
    let mut r_bs_old: Array3<f64> = Array::zeros(bs.raw_dim());

    for it in 0..maxiter {
        let lmax: usize = bs.dim().2;
        let mut r_bs: Array3<f64> = Array3::zeros(bs.raw_dim());
        let r_bs_alt: Array3<f64> = matrix_v_product(
            &bs,
            lmax,
            n_occ,
            n_virt,
            &om,
            &wq_ov,
            &gamma,
            multiplicity,
            spin_couplings,
        );
        if it < 2 {
            r_bs = matrix_v_product_fortran(
                &bs,
                lmax,
                n_occ,
                n_virt,
                &om,
                &wq_ov,
                &gamma,
                multiplicity,
                spin_couplings,
            );
        } else {
            let r_bs_new_vec: Array3<f64> = matrix_v_product_fortran(
                &bs.slice(s![.., .., l - (2 * nstates)..l]).to_owned(),
                (2 * nstates),
                n_occ,
                n_virt,
                &om,
                &wq_ov,
                &gamma,
                multiplicity,
                spin_couplings,
            );
            r_bs.slice_mut(s![.., .., ..l - (2 * nstates)])
                .assign(&r_bs_old.slice(s![.., .., ..l - (2 * nstates)]));
            r_bs.slice_mut(s![.., .., l - (2 * nstates)..l])
                .assign(&r_bs_new_vec);
            r_bs.slice_mut(s![.., .., 0..nstates])
                .assign(&(r_bs_old.slice(s![.., .., 0..nstates]).to_owned() * (-1.0)));
        }
        r_bs_old = r_bs.clone();
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

        let W_res: Array3<f64> = matrix_v_product_fortran(
            &T,
            lmax,
            n_occ,
            n_virt,
            &om,
            &wq_ov,
            &gamma,
            multiplicity,
            spin_couplings,
        ) - &T * &w2_new;

        // let W_res: Array3<f64> = matrix_v_product(&T, lmax, n_occ, n_virt, &om, &wq_ov, &gamma, multiplicity, spin_couplings)
        //     - einsum("k,ijk->ijk", &[&w2_new, &T])
        //         .unwrap()
        //         .into_dimensionality::<Ix3>()
        //         .unwrap();
        // println!("einsum: {}", now.elapsed().as_micros());

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
    let mut Omega: Vec<f64> = w.to_vec();
    Omega.sort_by(|&i, &j| i.partial_cmp(&j).unwrap());
    let Omega: Array1<f64> = Array::from(Omega).slice(s![..k]).to_owned();
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
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
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
    let maxiter: usize = maxiter.unwrap_or(100);
    let conv: f64 = conv.unwrap_or(1.0e-5);
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

    let mut bp_old: Array3<f64> = bs.clone();
    let mut bm_old: Array3<f64> = bs.clone();
    let mut l_prev: usize = l;

    for it in 0..maxiter {
        let lmax: usize = bs.dim().2;
        println!("Iteration {}", it);
        // println!("BS {}",bs.slice(s![0,5,..]));

        if XpYguess.is_none() || it > 0 {
            let mut bp: Array3<f64> = Array3::zeros((n_occ, n_virt, l));
            let mut bm: Array3<f64> = Array3::zeros((n_occ, n_virt, l));

            if it < 2 {
                bp = get_apbv_fortran(
                    &gamma,
                    &gamma_lr,
                    &qtrans_oo,
                    &qtrans_vv,
                    &qtrans_ov,
                    &omega,
                    &bs,
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    l,
                    multiplicity,
                    spin_couplings,
                );
                bm = get_ambv_fortran(
                    &gamma,
                    &gamma_lr,
                    &qtrans_oo,
                    &qtrans_vv,
                    &qtrans_ov,
                    &omega,
                    &bs,
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    l,
                );
            } else {
                let bp_new_vec: Array3<f64> = get_apbv_fortran(
                    &gamma,
                    &gamma_lr,
                    &qtrans_oo,
                    &qtrans_vv,
                    &qtrans_ov,
                    &omega,
                    &bs.slice(s![.., .., l - (3 * nstates)..l]).to_owned(),
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    (3 * nstates),
                    multiplicity,
                    spin_couplings,
                );
                bp.slice_mut(s![.., .., ..l - (3 * nstates)])
                    .assign(&bp_old.slice(s![.., .., ..l - (3 * nstates)]));
                bp.slice_mut(s![.., .., l - (3 * nstates)..l])
                    .assign(&bp_new_vec);
                bp.slice_mut(s![.., .., 0..nstates])
                    .assign(&(bp_old.slice(s![.., .., 0..nstates]).to_owned() * (-1.0)));
                let bm_new_vec: Array3<f64> = get_ambv_fortran(
                    &gamma,
                    &gamma_lr,
                    &qtrans_oo,
                    &qtrans_vv,
                    &qtrans_ov,
                    &omega,
                    &bs.slice(s![.., .., l - (3 * nstates)..l]).to_owned(),
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    (3 * nstates),
                );
                bm.slice_mut(s![.., .., ..l - (3 * nstates)])
                    .assign(&bm_old.slice(s![.., .., ..l - (3 * nstates)]));
                bm.slice_mut(s![.., .., l - (3 * nstates)..l])
                    .assign(&bm_new_vec);
                bm.slice_mut(s![.., .., 0..nstates])
                    .assign(&(bm_old.slice(s![.., .., 0..nstates]).to_owned() * (-1.0)));
            }

            bp_old = bp.clone();
            bm_old = bm.clone();

            // XINCHENG PLEASE CHECK THIS
            //             let bp_alt: Array3<f64> = get_apbv_fortran(
            //                 &gamma,
            //                 &gamma_lr,
            //                 &qtrans_oo,
            //                 &qtrans_vv,
            //                 &qtrans_ov,
            //                 &omega,
            //                 &bs,
            //                 qtrans_ov.dim().0,
            //                 n_occ,
            //                 n_virt,
            //                 l,
            //                 multiplicity,
            //                 spin_couplings,
            //             );
            // XINCHENG PLEASE CHECK THIS

            //             let bm_alt: Array3<f64> = get_ambv_fortran(
            //                 &gamma,
            //                 &gamma_lr,
            //                 &qtrans_oo,
            //                 &qtrans_vv,
            //                 &qtrans_ov,
            //                 &omega,
            //                 &bs,
            //                 qtrans_ov.dim().0,
            //                 n_occ,
            //                 n_virt,
            //                 l,
            //             );

            // # evaluate (A+B).b and (A-B).b
            //let bp_alt: Array3<f64> = get_apbv(
            //     &gamma,
            //     &Some(gamma_lr),
            //     &Some(qtrans_oo),
            //     &Some(qtrans_vv),
            //     &qtrans_ov,
            //     &omega,
            //     &bs,
            //     lc,
            //     multiplicity,
            //     spin_couplings,
            // );
            //let bm_alt: Array3<f64> = get_ambv(
            //    &gamma, &gamma_lr, &qtrans_oo, &qtrans_vv, &qtrans_ov, &omega, &bs, lc,
            //);

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
                panic!(
                    "Hmm... It seems that Mh is not hermitian. The currect error is {:e}\n\
                        and should be lower than 1.0e-10. If you know what you are doing, try\n\
                        lowering the tolerance. Otherwise check you input!",
                    err
                );
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
                panic!("Hmm, it seems that (X+Y) and (X-Y) vectors are not orthonormal. The error\n\
                        is {:e} and should be smaller than 1.0e-3. Maybe your molecule just doesn't like you?" , err);
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
        //let wl = get_apbv(
        //    &gamma,
        //    &Some(gamma_lr),
        //    &Some(qtrans_oo),
        //    &Some(qtrans_vv),
        //    &qtrans_ov,
        //    &omega,
        //    &r_canon,
        //    lc,
        //    multiplicity,
        //    spin_couplings,
        //) - &l_canon * &w;
        let wl = get_apbv_fortran(
            &gamma,
            &gamma_lr,
            &qtrans_oo,
            &qtrans_vv,
            &qtrans_ov,
            &omega,
            &r_canon,
            qtrans_ov.dim().0,
            n_occ,
            n_virt,
            l,
            multiplicity,
            spin_couplings,
        ) - &l_canon * &w;
        //let wr = get_ambv(
        //    &gamma, &gamma_lr, &qtrans_oo, &qtrans_vv, &qtrans_ov, &omega, &l_canon, lc,
        //) - &r_canon * &w;
        let wr = get_ambv_fortran(
            &gamma,
            &gamma_lr,
            &qtrans_oo,
            &qtrans_vv,
            &qtrans_ov,
            &omega,
            &l_canon,
            qtrans_ov.dim().0,
            n_occ,
            n_virt,
            l,
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
        println!("Norms davidson {}", norms);
        if indices_norms.len() == norms.len() && it > 0 {
            break;
        }

        //  enlarge dimension of subspace by dk vectors
        //  At most 2*k new expansion vectors are added
        let dkmax = (kmax - l).min(2 * k);
        // # count number of non-converged vectors
        // # residual vectors that are zero cannot be used as new expansion vectors
        // 1.0e-16
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
        let mut nb: usize = 0;
        // select new expansion vectors among the non-converged left residual vectors
        for i in 0..k {
            if nb == dk {
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
            if nb == dk {
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
        l_prev = l;
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
    let mut Omega: Vec<f64> = w.to_vec();
    Omega.sort_by(|&i, &j| i.partial_cmp(&j).unwrap());
    let Omega: Array1<f64> = Array::from(Omega).slice(s![..k]).to_owned();
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
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let lmax: usize = vs.dim().2;
    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());

    for i in 0..lmax {
        let v: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // # matrix product u_ia = sum_jb (A+B)_(ia,jb) v_jb
        // # 1st term in (A+B).v: KS orbital energy differences
        let mut u: Array2<f64> = omega * &v;

        // 2nd term Coulomb
        let tmp: Array1<f64> = tensordot(&qtrans_ov, &v, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix1>()
            .unwrap();

        if multiplicity == 1 {
            let tmp_2: Array1<f64> = gamma.dot(&tmp);
            let u_singlet: Array2<f64> = 4.0
                * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
            u = u + u_singlet;
        //println!("Iteration {} for the tensordot routine",i);
        //println!("Value of u after 2nd term {}",u);
        } else if multiplicity == 3 {
            let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings);
            let tmp_2: Array1<f64> = spin_couplings_diag.dot(&tmp);
            let u_triplet: Array2<f64> = 4.0
                * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
            u = u + u_triplet;
        } else {
            panic!("Currently only singlets and triplets are supported, you wished a multiplicity of {}!", multiplicity);
        }

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

pub fn get_apbv_fortran(
    gamma: &ArrayView2<f64>,
    gamma_lr: &ArrayView2<f64>,
    qtrans_oo: &ArrayView3<f64>,
    qtrans_vv: &ArrayView3<f64>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    n_at: usize,
    n_occ: usize,
    n_virt: usize,
    n_vec: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let tmp_q_vv: Array2<f64> = qtrans_vv
        .to_owned()
        .into_shape((n_virt * n_at, n_virt))
        .unwrap();
    let tmp_q_oo: Array2<f64> = qtrans_oo
        .to_owned()
        .into_shape((n_at * n_occ, n_occ))
        .unwrap();
    let mut tmp_q_ov_swapped: Array3<f64> = qtrans_ov.to_owned();
    tmp_q_ov_swapped.swap_axes(1, 2);
    tmp_q_ov_swapped = tmp_q_ov_swapped.as_standard_layout().to_owned();
    let tmp_q_ov_shape_1: Array2<f64> =
        tmp_q_ov_swapped.into_shape((n_at * n_virt, n_occ)).unwrap();
    let mut tmp_q_ov_swapped_2: Array3<f64> = qtrans_ov.to_owned();
    tmp_q_ov_swapped_2.swap_axes(0, 1);
    tmp_q_ov_swapped_2 = tmp_q_ov_swapped_2.as_standard_layout().to_owned();
    let tmp_q_ov_shape_2: Array2<f64> = tmp_q_ov_swapped_2
        .into_shape((n_occ, n_at * n_virt))
        .unwrap();
    //let tmp_q_oo: Array2<f64> = qtrans_oo
    //    .to_owned()
    //    .into_shape((n_at * n_occ, n_occ))
    //    .unwrap();
    let tmp_q_ov_shape_1_new: Array2<f64> = qtrans_ov
        .to_owned()
        .into_shape((n_occ, n_at * n_virt))
        .unwrap()
        .reversed_axes();
    let tmp_q_ov_shape_2_new: Array2<f64> = qtrans_ov
        .to_owned()
        .into_shape((n_at * n_virt, n_occ))
        .unwrap()
        .reversed_axes();

    // println!("qtrans ov{}", qtrans_ov.clone());
    // println!("Compare shapes");
    // println!("Old q_ov {:?}", tmp_q_ov_shape_1);
    // println!("New q_ov {:?}", tmp_q_ov_shape_1_new);

    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());

    let gamma_equiv: Array2<f64> = if multiplicity == 1 {
        gamma.to_owned()
    } else if multiplicity == 3 {
        Array2::from_diag(&spin_couplings)
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
        Array::zeros(gamma.raw_dim())
    };

    for i in (0..n_vec) {
        let vl: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // 1st term - KS orbital energy differences
        let mut u_l: Array2<f64> = omega * &vl;

        // 2nd term - Coulomb
        let mut tmp21: Array1<f64> = Array1::zeros(n_at);

        //for at in (0..n_at) {
        //    let tmp:Array2<f64> = qtrans_ov.clone().slice(s![at, .., ..]).to_owned() * vl.clone();
        //    tmp21[at] = tmp.sum();
        //}
        let tmp21: Vec<f64> = (0..n_at)
            .into_par_iter()
            .map(|at| {
                let tmp: Array2<f64> = &qtrans_ov.slice(s![at, .., ..]).to_owned() * &vl;
                tmp.sum()
            })
            .collect();
        let tmp21: Array1<f64> = Array::from(tmp21);

        let tmp22: Array1<f64> = 4.0 * gamma_equiv.dot(&tmp21);

        // for at in (0..n_at).into_iter() {
        //     u_l = u_l + qtrans_ov.slice(s![at, .., ..]).to_owned() * tmp22[at];
        // }
        let mut tmp: Vec<Array2<f64>> = (0..n_at)
            .into_par_iter()
            .map(|at| qtrans_ov.slice(s![at, .., ..]).to_owned() * tmp22[at])
            .collect();
        for i in tmp.iter() {
            u_l = u_l + i;
        }
        //u_l = u_l + tmp;

        // 3rd term - Exchange
        let tmp31: Array3<f64> = tmp_q_vv
            .dot(&vl.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();

        let tmp31_reshaped: Array2<f64> = tmp31.into_shape((n_at, n_virt * n_occ)).unwrap();
        let mut tmp32: Array3<f64> = gamma_lr
            .dot(&tmp31_reshaped)
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        tmp32.swap_axes(1, 2);
        tmp32 = tmp32.as_standard_layout().to_owned();

        let tmp33: Array2<f64> = tmp_q_oo
            .t()
            .dot(&tmp32.into_shape((n_at * n_occ, n_virt)).unwrap());
        u_l = u_l - tmp33;

        // 4th term - Exchange
        let tmp41: Array3<f64> = tmp_q_ov_shape_1
            .dot(&vl)
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        let tmp41_reshaped: Array2<f64> = tmp41.into_shape((n_at, n_virt * n_virt)).unwrap();
        let mut tmp42: Array3<f64> = gamma_lr
            .dot(&tmp41_reshaped)
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        tmp42.swap_axes(1, 2);
        tmp42 = tmp42.as_standard_layout().to_owned();

        let tmp43: Array2<f64> =
            tmp_q_ov_shape_2.dot(&tmp42.into_shape((n_at * n_virt, n_virt)).unwrap());
        u_l = u_l - tmp43;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}

pub fn get_apbv_fortran_no_lc(
    gamma: &ArrayView2<f64>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    n_at: usize,
    n_occ: usize,
    n_virt: usize,
    n_vec: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());

    let gamma_equiv: Array2<f64> = if multiplicity == 1 {
        gamma.to_owned()
    } else if multiplicity == 3 {
        Array2::from_diag(&spin_couplings)
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
        Array::zeros(gamma.raw_dim())
    };

    for i in (0..n_vec) {
        let vl: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // 1st term - KS orbital energy differences
        let mut u_l: Array2<f64> = omega * &vl;

        // 2nd term - Coulomb
        let mut tmp21: Array1<f64> = Array1::zeros(n_at);

        //for at in (0..n_at) {
        //    let tmp:Array2<f64> = qtrans_ov.clone().slice(s![at, .., ..]).to_owned() * vl.clone();
        //    tmp21[at] = tmp.sum();
        //}
        let tmp21: Vec<f64> = (0..n_at)
            .into_par_iter()
            .map(|at| {
                let tmp: Array2<f64> = &qtrans_ov.slice(s![at, .., ..]) * &vl;
                tmp.sum()
            })
            .collect();
        let tmp21: Array1<f64> = Array::from(tmp21);
        let tmp22: Array1<f64> = 4.0 * gamma_equiv.dot(&tmp21);

        // for at in (0..n_at).into_iter() {
        //     u_l = u_l + qtrans_ov.slice(s![at, .., ..]).to_owned() * tmp22[at];
        // }
        let mut tmp: Vec<Array2<f64>> = (0..n_at)
            .into_par_iter()
            .map(|at| qtrans_ov.slice(s![at, .., ..]).to_owned() * tmp22[at])
            .collect();
        for i in tmp.iter() {
            u_l = u_l + i;
        }
        //u_l = u_l + tmp;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}

pub fn get_ambv_fortran(
    gamma: &ArrayView2<f64>,
    gamma_lr: &ArrayView2<f64>,
    qtrans_oo: &ArrayView3<f64>,
    qtrans_vv: &ArrayView3<f64>,
    qtrans_ov: &ArrayView3<f64>,
    omega: &ArrayView2<f64>,
    vs: &Array3<f64>,
    n_at: usize,
    n_occ: usize,
    n_virt: usize,
    n_vec: usize,
) -> (Array3<f64>) {
    let tmp_q_vv: Array2<f64> = qtrans_vv
        .to_owned()
        .into_shape((n_virt * n_at, n_virt))
        .unwrap();
    let mut tmp_q_oo_swapped: Array3<f64> = qtrans_oo.to_owned();
    tmp_q_oo_swapped.swap_axes(0, 1);
    tmp_q_oo_swapped = tmp_q_oo_swapped.as_standard_layout().to_owned();
    let tmp_q_oo: Array2<f64> = tmp_q_oo_swapped.into_shape((n_occ, n_at * n_occ)).unwrap();
    let mut tmp_q_ov_swapped: Array3<f64> = qtrans_ov.to_owned();
    tmp_q_ov_swapped.swap_axes(1, 2);
    tmp_q_ov_swapped = tmp_q_ov_swapped.as_standard_layout().to_owned();
    let tmp_q_ov_shape_1: Array2<f64> =
        tmp_q_ov_swapped.into_shape((n_at * n_virt, n_occ)).unwrap();
    let mut tmp_q_ov_swapped_2: Array3<f64> = qtrans_ov.to_owned();
    //println!("Before swap {}",tmp_q_ov_swapped_2.clone().into_shape((n_occ,n_at,n_virt)).unwrap());
    tmp_q_ov_swapped_2.swap_axes(0, 1);
    //let mut tmp_q_ov_swapped_3:Array3<f64> = tmp_q_ov_swapped_2.to_owned();
    //let tmp_q_ov_shape_3: Array3<f64> = ArrayView::from_shape((n_occ,n_at,n_virt),&tmp_q_ov_swapped_3.clone().as_slice_memory_order().unwrap()).unwrap().to_owned();//.into_shape((n_occ, n_at * n_virt)).unwrap().to_owned();
    //tmp_q_ov_swapped_2 = tmp_q_ov_swapped_3.as_standard_layout().to_owned();
    tmp_q_ov_swapped_2 = tmp_q_ov_swapped_2.as_standard_layout().to_owned();
    let tmp_q_ov_shape_2: Array2<f64> = tmp_q_ov_swapped_2
        .clone()
        .into_shape((n_occ, n_at * n_virt))
        .unwrap();

    // println!("shape after swap {}",tmp_q_ov_swapped_3);
    // println!("shape after standard layout {}",tmp_q_ov_swapped_2);
    // println!("shape Array::from_shape() {}",tmp_q_ov_shape_3);

    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());

    for i in (0..n_vec) {
        let vl: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // 1st term - KS orbital energy differences
        let mut u_l: Array2<f64> = omega * &vl;
        // 2nd term - Coulomb
        let tmp21: Array3<f64> = tmp_q_ov_shape_1
            .dot(&vl)
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();

        let mut tmp22: Array3<f64> = gamma_lr
            .dot(&tmp21.into_shape((n_at, n_virt * n_virt)).unwrap())
            .into_shape((n_at, n_virt, n_virt))
            .unwrap();
        tmp22.swap_axes(1, 2);
        tmp22 = tmp22.as_standard_layout().to_owned();

        let tmp23: Array2<f64> =
            tmp_q_ov_shape_2.dot(&tmp22.into_shape((n_at * n_virt, n_virt)).unwrap());
        u_l = u_l + tmp23;

        // 3rd term - Exchange
        let tmp31: Array3<f64> = tmp_q_vv
            .dot(&vl.t())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        let mut tmp32: Array3<f64> = gamma_lr
            .dot(&tmp31.into_shape((n_at, n_virt * n_occ)).unwrap())
            .into_shape((n_at, n_virt, n_occ))
            .unwrap();
        tmp32.swap_axes(1, 2);
        tmp32 = tmp32.as_standard_layout().to_owned();

        let tmp33: Array2<f64> = tmp_q_oo.dot(&tmp32.into_shape((n_at * n_occ, n_virt)).unwrap());

        u_l = u_l - tmp33;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}

// pub fn get_apbv_single_vector(
//     gamma: &ArrayView2<f64>,
//     gamma_lr: &Option<ArrayView2<f64>>,
//     qtrans_oo: &Option<ArrayView3<f64>>,
//     qtrans_vv: &Option<ArrayView3<f64>>,
//     qtrans_ov: &ArrayView3<f64>,
//     omega: &ArrayView2<f64>,
//     vs: &Array2<f64>,
//     lc: usize,
//     multiplicity: u8,
//     spin_couplings: ArrayView1<f64>,
// ) -> (Array2<f64>) {
//
//     let v: Array2<f64> = vs.clone();
//     // # matrix product u_ia = sum_jb (A+B)_(ia,jb) v_jb
//     // # 1st term in (A+B).v: KS orbital energy differences
//     let mut u: Array2<f64> = omega * &v;
//
//     // 2nd term Coulomb
//     let tmp: Array1<f64> = tensordot(&qtrans_ov, &v, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//
//     if multiplicity == 1 {
//         let tmp_2: Array1<f64> = gamma.dot(&tmp);
//         let u_singlet: Array2<f64> = 4.0
//             * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
//             .into_dimensionality::<Ix2>()
//             .unwrap();
//         u = u + u_singlet;
//     } else if multiplicity == 3 {
//         let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings);
//         let tmp_2: Array1<f64> = spin_couplings_diag.dot(&tmp);
//         let u_triplet: Array2<f64> = 4.0
//             * tensordot(&qtrans_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
//             .into_dimensionality::<Ix2>()
//             .unwrap();
//         u = u + u_triplet;
//     } else {
//         panic!("Currently only singlets and triplets are supported, you wished a multiplicity of {}!", multiplicity);
//     }
//
//     if lc == 1 {
//         // 3rd term - Exchange
//         let tmp: Array3<f64> = tensordot(&qtrans_vv.unwrap(), &v, &[Axis(2)], &[Axis(1)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         let tmp_2: Array3<f64> = tensordot(&gamma_lr.unwrap(), &tmp, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         u = u - tensordot(
//             &qtrans_oo.unwrap(),
//             &tmp_2,
//             &[Axis(0), Axis(2)],
//             &[Axis(0), Axis(2)],
//         )
//             .into_dimensionality::<Ix2>()
//             .unwrap();
//
//         //4th term - Exchange
//         let tmp: Array3<f64> = tensordot(&qtrans_ov, &v, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         let tmp_2: Array3<f64> = tensordot(&gamma_lr.unwrap(), &tmp, &[Axis(1)], &[Axis(0)])
//             .into_dimensionality::<Ix3>()
//             .unwrap();
//         u = u - tensordot(&qtrans_ov, &tmp_2, &[Axis(0), Axis(2)], &[Axis(0), Axis(2)])
//             .into_dimensionality::<Ix2>()
//             .unwrap();
//     }
//
//     return u;
// }

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
            // 2nd term - Exchange
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
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
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

        if multiplicity == 1 {
            let tmp_2: Array1<f64> = gamma.dot(&tmp);
            let u_singlet: Array2<f64> = 4.0
                * tensordot(&wq_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
            u = u + u_singlet;
        } else if multiplicity == 3 {
            let spin_couplings_diag: Array2<f64> = Array2::from_diag(&spin_couplings);
            let tmp_2: Array1<f64> = spin_couplings_diag.dot(&tmp);
            let u_triplet: Array2<f64> = 4.0
                * tensordot(&wq_ov, &tmp_2, &[Axis(0)], &[Axis(0)])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
            u = u + u_triplet;
        } else {
            panic!("Currently only singlets and triplets are supported, you wished a multiplicity of {}!", multiplicity);
        }

        // let tmp2: Array1<f64> = gamma.dot(&tmp);
        // u = u + 4.0
        //     * tensordot(&wq_ov, &tmp2, &[Axis(0)], &[Axis(0)])
        //         .into_dimensionality::<Ix2>()
        //         .unwrap();

        us.slice_mut(s![.., .., i]).assign(&u);
    }
    return us;
}

pub fn matrix_v_product_fortran(
    vs: &Array3<f64>,
    n_vec: usize,
    n_occ: usize,
    n_virt: usize,
    om: &Array2<f64>,
    wq_ov: &Array3<f64>,
    gamma: &ArrayView2<f64>,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
) -> (Array3<f64>) {
    let mut us: Array3<f64> = Array::zeros(vs.raw_dim());
    let n_at: usize = wq_ov.dim().0;

    let gamma_equiv: Array2<f64> = if multiplicity == 1 {
        gamma.to_owned()
    } else if multiplicity == 3 {
        Array2::from_diag(&spin_couplings)
    } else {
        panic!(
            "Currently only singlets and triplets are supported, you wished a multiplicity of {}!",
            multiplicity
        );
        Array::zeros(gamma.raw_dim())
    };

    for i in (0..n_vec) {
        let vl: Array2<f64> = vs.slice(s![.., .., i]).to_owned();
        // 1st term - KS orbital energy differences
        let mut u_l: Array2<f64> = om * &vl;

        // 2nd term - Coulomb
        let mut tmp21: Array1<f64> = Array1::zeros(n_at);

        //for at in (0..n_at) {
        //    let tmp:Array2<f64> = qtrans_ov.clone().slice(s![at, .., ..]).to_owned() * vl.clone();
        //    tmp21[at] = tmp.sum();
        //}
        let tmp21: Vec<f64> = (0..n_at)
            .into_par_iter()
            .map(|at| {
                let tmp: Array2<f64> = &wq_ov.slice(s![at, .., ..]) * &vl;
                tmp.sum()
            })
            .collect();
        let tmp21: Array1<f64> = Array::from(tmp21);

        let tmp22: Array1<f64> = 4.0 * gamma_equiv.dot(&tmp21);

        // for at in (0..n_at).into_iter() {
        //     u_l = u_l + qtrans_ov.slice(s![at, .., ..]).to_owned() * tmp22[at];
        // }
        let mut tmp: Vec<Array2<f64>> = (0..n_at)
            .into_par_iter()
            .map(|at| wq_ov.slice(s![at, .., ..]).to_owned() * tmp22[at])
            .collect();
        for i in tmp.iter() {
            u_l = u_l + i;
        }
        //u_l = u_l + tmp;

        us.slice_mut(s![.., .., i]).assign(&u_l);
    }
    return us;
}

pub fn krylov_solver_zvector(
    a_diag: ArrayView2<f64>,
    b_matrix: ArrayView3<f64>,
    x_0: Option<Array3<f64>>,
    maxiter: Option<usize>,
    conv: Option<f64>,
    g0: ArrayView2<f64>,
    g0_lr: Option<ArrayView2<f64>>,
    qtrans_oo: Option<ArrayView3<f64>>,
    qtrans_vv: Option<ArrayView3<f64>>,
    qtrans_ov: ArrayView3<f64>,
    lc: usize,
    multiplicity: u8,
    spin_couplings: ArrayView1<f64>,
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
        bs = x_0.unwrap();
    }

    let mut x_matrix: Array3<f64> = Array::zeros((n_occ, n_virt, k));
    let mut temp_old: Array3<f64> = bs.clone();

    for it in 0..maxiter {
        // representation of A in the basis of expansion vectors
        let mut temp: Array3<f64> = Array3::zeros((n_occ, n_virt, l));
        if it == 0 {
            // temp = get_apbv(
            //     &g0,
            //     &g0_lr,
            //     &qtrans_oo,
            //     &qtrans_vv,
            //     &qtrans_ov,
            //     &a_diag,
            //     &bs,
            //     lc,
            //     multiplicity,
            //     spin_couplings,
            // );
            if lc == 1 {
                temp = get_apbv_fortran(
                    &g0,
                    &g0_lr.clone().unwrap(),
                    &qtrans_oo.clone().unwrap(),
                    &qtrans_vv.clone().unwrap(),
                    &qtrans_ov,
                    &a_diag,
                    &bs,
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    l,
                    multiplicity,
                    spin_couplings,
                );
            } else {
                temp = get_apbv_fortran_no_lc(
                    &g0,
                    &qtrans_ov,
                    &a_diag,
                    &bs,
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    l,
                    multiplicity,
                    spin_couplings,
                );
            }
        } else {
            //let temp_new_vec_alt: Array3<f64> = get_apbv(
            //    &g0,
            //    &g0_lr,
            //    &qtrans_oo,
            //    &qtrans_vv,
            //    &qtrans_ov,
            //    &a_diag,
            //    &bs.slice(s![.., .., l - 2..l]).to_owned(),
            //    lc,
            //    multiplicity,
            //    spin_couplings,
            //);
            let mut temp_new_vec: Array3<f64> = Array3::zeros((n_occ, n_virt, (l - 2..l).len()));
            if lc == 1 {
                temp_new_vec = get_apbv_fortran(
                    &g0,
                    &g0_lr.clone().unwrap(),
                    &qtrans_oo.clone().unwrap(),
                    &qtrans_vv.clone().unwrap(),
                    &qtrans_ov,
                    &a_diag,
                    &bs.slice(s![.., .., l - 2..l]).to_owned(),
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    (l - 2..l).len(),
                    multiplicity,
                    spin_couplings,
                );
            } else {
                temp_new_vec = get_apbv_fortran_no_lc(
                    &g0,
                    &qtrans_ov,
                    &a_diag,
                    &bs.slice(s![.., .., l - 2..l]).to_owned(),
                    qtrans_ov.dim().0,
                    n_occ,
                    n_virt,
                    (l - 2..l).len(),
                    multiplicity,
                    spin_couplings,
                );
            }
            // println!("Temp new alt {}",temp_new_vec_alt);
            // println!("Temp vec new {}",temp_new_vec)
            temp.slice_mut(s![.., .., ..l - 1]).assign(&temp_old);
            temp.slice_mut(s![.., .., l - 2..l]).assign(&temp_new_vec);
        }
        // let temp:Array3<f64> = get_apbv(
        //         &g0,
        //         &g0_lr,
        //         &qtrans_oo,
        //         &qtrans_vv,
        //         &qtrans_ov,
        //         &a_diag,
        //         &bs,
        //         lc,
        //         multiplicity,
        //         spin_couplings,
        //     );

        temp_old = temp.clone();

        // println!("Temp {}",temp);
        // println!("Bs {}",bs);

        let a_b: Array2<f64> = tensordot(&bs, &temp, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)])
            .into_dimensionality::<Ix2>()
            .unwrap();

        // let a_b: Array2<f64> = tensordot(
        //     &bs,
        //     &get_apbv(
        //         &g0,
        //         &g0_lr,
        //         &qtrans_oo,
        //         &qtrans_vv,
        //         &qtrans_ov,
        //         &a_diag,
        //         &bs,
        //         lc,
        //         multiplicity,
        //         spin_couplings,
        //     ),
        //     &[Axis(0), Axis(1)],
        //     &[Axis(0), Axis(1)],
        // )
        // .into_dimensionality::<Ix2>()
        // .unwrap();
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
        let mut w_res: Array3<f64> = Array3::zeros((x_matrix.raw_dim()));
        if lc == 1 {
            w_res = get_apbv_fortran(
                &g0,
                &g0_lr.clone().unwrap(),
                &qtrans_oo.clone().unwrap(),
                &qtrans_vv.clone().unwrap(),
                &qtrans_ov,
                &a_diag,
                &x_matrix,
                qtrans_ov.dim().0,
                n_occ,
                n_virt,
                x_matrix.dim().2,
                multiplicity,
                spin_couplings,
            );
        } else {
            w_res = get_apbv_fortran_no_lc(
                &g0,
                &qtrans_ov,
                &a_diag,
                &x_matrix,
                qtrans_ov.dim().0,
                n_occ,
                n_virt,
                x_matrix.dim().2,
                multiplicity,
                spin_couplings,
            );
        }
        w_res = &w_res - &b_matrix;
        // let w_res: Array3<f64> = &get_apbv(
        //     &g0,
        //     &g0_lr,
        //     &qtrans_oo,
        //     &qtrans_vv,
        //     &qtrans_ov,
        //     &a_diag,
        //     &x_matrix,
        //     lc,
        //     multiplicity,
        //     spin_couplings,
        // ) - &b_matrix;

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
        // let mut norms_over_eps: Array1<f64> = Array::zeros(indices_norm_over_eps.len());
        // for i in 0..indices_norm_over_eps.len() {
        //     norms_over_eps[i] = norms[indices_norm_over_eps[i]];
        // }
        //let nc: f64 = norms_over_eps.sum();
        let nc: usize = indices_norm_over_eps.len();
        let dk: usize = dkmax.min(nc);

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
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
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
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        charge,
        multiplicity,
        Some(0.0),
        None,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
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
        get_exc_energies(&f_occ.to_vec(), &mol, None, &S, &orbe, &orbs, false, None);
    assert!( 1 == 2);
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
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mut mol: Molecule = Molecule::new(
        atomic_numbers,
        positions,
        charge,
        multiplicity,
        Some(0.0),
        None,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
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

    let (omega_out, c_ij, XmY, XpY) = get_exc_energies(
        &f_occ.to_vec(),
        &mol,
        Some(4),
        &S,
        &orbe,
        &orbs,
        false,
        None,
    );
    println!("omega_out{}", &omega_out);
    println!("omega_diff {}", &omega_out - &omega_ref_out);
    assert!(omega_out.abs_diff_eq(&omega_ref_out, 1e-10));
    assert!((&XpY * &XpY).abs_diff_eq(&(&XpY_ref * &XpY_ref), 1e-10));
    assert!((&XmY * &XmY).abs_diff_eq(&(&XmY_ref * &XmY_ref), 1e-10));
}

#[test]
fn excited_energies_non_hermitian_davidson_routine() {
    let mut mol: Molecule = get_water_molecule();

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

    let (omega_out, c_ij, XmY, XpY) = get_exc_energies(
        &f_occ.to_vec(),
        &mol,
        Some(4),
        &S,
        &orbe,
        &orbs,
        false,
        None,
    );
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

    // test W correction
    let mol: Molecule = get_water_molecule();

    let spin_couplings: ArrayView1<f64> = mol.calculator.spin_couplings.view();
    let spin_couplings_null: ArrayView1<f64> =
        Array::zeros(mol.calculator.spin_couplings.raw_dim()).view();

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
        spin_couplings,
    );
    println!("omega {}", omega);
    println!("omega_ref {}", omega_ref);
    assert!(omega.abs_diff_eq(&omega_ref, 1e-14));
    assert!(c_ij.abs_diff_eq(&c_ij_ref, 1e-14));
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

    // test W correction
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
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mol: Molecule = Molecule::new(
        atomic_numbers.clone(),
        positions,
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    );

    // Only for testing purposes
    let spin_couplings: Array1<f64> = mol.calculator.spin_couplings.clone();
    let spin_couplings_null: Array1<f64> =
        Array::zeros(mol.calculator.spin_couplings.clone().raw_dim());

    let (omega, c_ij, XmY, XpY): (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = casida(
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
        spin_couplings_null.view(),
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

    // test W correction
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
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mol: Molecule = Molecule::new(
        atomic_numbers.clone(),
        positions,
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    );

    // Only for testing purposes
    let spin_couplings: Array1<f64> = mol.calculator.spin_couplings.clone();
    let spin_couplings_null: Array1<f64> =
        Array::zeros(mol.calculator.spin_couplings.clone().raw_dim());

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
            mol.calculator.spin_couplings.view(),
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
fn test_apbv_fortran() {
    let bs: Array3<f64> = array![
        [[0., 0., 1., 0.], [0., 0., 0., 1.]],
        [[1., 0., 0., 0.], [0., 1., 0., 0.]]
    ];

    let bp_ref: Array3<f64> = array![
        [
            [
                -0.0000000000000000015255571718482793,
                0.000000000000000009672565771884146,
                0.49449894023814805,
                0.00000007705985670975216
            ],
            [
                0.000000000000000016762517988397043,
                -0.000000000000000004135597331086377,
                0.0000000770598567199761,
                0.5449686680522884
            ]
        ],
        [
            [
                0.3837835356347358,
                0.0000001353041309525262,
                -0.000000000000000001525557171848286,
                0.000000000000000016762517988397055
            ],
            [
                0.0000001353041309525262,
                0.43762532914260255,
                0.000000000000000009672565771884152,
                -0.000000000000000004135597331086383
            ]
        ]
    ];

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
    let omega0: Array2<f64> = array![
        [0.7329867988448483, 0.7853711745345131],
        [0.6599617692239994, 0.7123461449136643]
    ];

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
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mol: Molecule = Molecule::new(
        atomic_numbers.clone(),
        positions,
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    );

    // Only for testing purposes
    let spin_couplings: Array1<f64> = mol.calculator.spin_couplings.clone();
    let spin_couplings_null: Array1<f64> =
        Array::zeros(mol.calculator.spin_couplings.clone().raw_dim());

    let n_occ: usize = qtrans_oo.dim().1;
    let n_virt: usize = qtrans_vv.dim().1;

    let bp: Array3<f64> = get_apbv_fortran(
        &gamma.view(),
        &gamma_lr.view(),
        &qtrans_oo.view(),
        &qtrans_vv.view(),
        &qtrans_ov.view(),
        &omega0.view(),
        &bs,
        3,
        2,
        2,
        4,
        multiplicity.unwrap(),
        spin_couplings.view(),
    );
    let bp_ref: Array3<f64> = get_apbv(
        &gamma.view(),
        &Some(gamma_lr.view()),
        &Some(qtrans_oo.view()),
        &Some(qtrans_vv.view()),
        &qtrans_ov.view(),
        &omega0.view(),
        &bs,
        1,
        1,
        spin_couplings_null.view(),
    );

    println!("BP diff {}", &bp - &bp_ref);
    assert!(bp.abs_diff_eq(&bp_ref, 1e-8));
}

#[test]
fn test_apbv_fortran_no_lc() {
    let bs: Array3<f64> = array![
        [[0., 0., 1., 0.], [0., 0., 0., 1.]],
        [[1., 0., 0., 0.], [0., 1., 0., 0.]]
    ];

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
    let omega0: Array2<f64> = array![
        [0.7329867988448483, 0.7853711745345131],
        [0.6599617692239994, 0.7123461449136643]
    ];

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
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mol: Molecule = Molecule::new(
        atomic_numbers.clone(),
        positions,
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    );

    // Only for testing purposes
    let spin_couplings: Array1<f64> = mol.calculator.spin_couplings.clone();
    let spin_couplings_null: Array1<f64> =
        Array::zeros(mol.calculator.spin_couplings.clone().raw_dim());

    let n_occ: usize = qtrans_oo.dim().1;
    let n_virt: usize = qtrans_vv.dim().1;

    let bp: Array3<f64> = get_apbv_fortran_no_lc(
        &gamma.view(),
        &qtrans_ov.view(),
        &omega0.view(),
        &bs,
        3,
        2,
        2,
        4,
        multiplicity.unwrap(),
        spin_couplings.view(),
    );
    let bp_ref: Array3<f64> = get_apbv(
        &gamma.view(),
        &Some(gamma_lr.view()),
        &Some(qtrans_oo.view()),
        &Some(qtrans_vv.view()),
        &qtrans_ov.view(),
        &omega0.view(),
        &bs,
        0,
        1,
        spin_couplings_null.view(),
    );

    println!("BP diff {}", &bp - &bp_ref);
    assert!(bp.abs_diff_eq(&bp_ref, 1e-12));
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

    // test W correction
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
    let config: GeneralConfig = toml::from_str("").unwrap();
    let mol: Molecule = Molecule::new(
        atomic_numbers.clone(),
        positions,
        charge,
        multiplicity,
        None,
        None,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    );

    // Only for testing purposes
    let spin_couplings: Array1<f64> = mol.calculator.spin_couplings.clone();
    let spin_couplings_null: Array1<f64> =
        Array::zeros(mol.calculator.spin_couplings.clone().raw_dim());

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
            spin_couplings.view(),
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
    assert!(1 == 2);
}

#[test]
fn benzene_excitations() {
    let nstates: Option<usize> = Some(4);

    // Test molecule without lc
    let mut mol: Molecule = get_benzene_molecule();
    mol.calculator.r_lr = Some(0.0);
    mol.calculator.active_orbitals = Some((4, 4));
    println!(
        "r_lr = {}\n",
        mol.calculator.r_lr.unwrap_or(defaults::LONG_RANGE_RADIUS)
    );

    let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
        run_scc(&mut mol);
    mol.calculator.set_active_orbitals(f.clone());

    // singlets
    mol.multiplicity = 1;
    println!("multiplicity = {}", mol.multiplicity);

    // should call tda
    let (omega_tda, c_ij_tda, XmY_tda, XpY_tda) = get_exc_energies(
        &f.to_vec(),
        &mol,
        None,
        &s,
        &orbe,
        &orbs,
        false,
        Some(String::from("TDA")),
    );
    println!("omega_TDA: {}", &omega_tda);
    // println!("c_ij_TDA: {:?}", &c_ij_tda);

    // should call cadisa
    let (omega_casida, c_ij_casida, XmY_casida, XpY_casida) = get_exc_energies(
        &f.to_vec(),
        &mol,
        None,
        &s,
        &orbe,
        &orbs,
        false,
        Some(String::from("casida")),
    );
    println!("omega_casida: {}", &omega_casida);
    // println!("c_ij_casida: {:?}", &c_ij_casida);

    // should call hermitian davidson
    let (omega_davidson, c_ij_davidson, XmY_davidson, XpY_davidson) =
        get_exc_energies(&f.to_vec(), &mol, nstates, &s, &orbe, &orbs, false, None);
    println!("omega_davidson (hermitian): {}\n", &omega_davidson);
    // println!("c_ij_davidson (hermitian): {:?}", &c_ij_davidson);

    assert!(omega_casida
        .slice(s![0..nstates.unwrap()])
        .abs_diff_eq(&omega_davidson, 1e-12));

    // triplets
    mol.multiplicity = 3;
    println!("multiplicity = {}", mol.multiplicity);

    // should call tda
    let (omega_tda, c_ij_tda, XmY_tda, XpY_tda) = get_exc_energies(
        &f.to_vec(),
        &mol,
        None,
        &s,
        &orbe,
        &orbs,
        false,
        Some(String::from("TDA")),
    );
    println!("omega_TDA: {}", &omega_tda);
    // println!("c_ij_TDA: {:?}", &c_ij_tda);

    // should call cadisa
    let (omega_casida, c_ij_casida, XmY_casida, XpY_casida) = get_exc_energies(
        &f.to_vec(),
        &mol,
        None,
        &s,
        &orbe,
        &orbs,
        false,
        Some(String::from("casida")),
    );
    println!("omega_casida: {}", &omega_casida);
    // println!("c_ij_casida: {:?}", &c_ij_casida);

    // should call hermitian davidson
    let (omega_davidson, c_ij_davidson, XmY_davidson, XpY_davidson) =
        get_exc_energies(&f.to_vec(), &mol, nstates, &s, &orbe, &orbs, false, None);
    println!("omega_davidson (hermitian): {}\n", &omega_davidson);
    // println!("c_ij_davidson (hermitian): {:?}", &c_ij_davidson);

    assert!(omega_casida
        .slice(s![0..nstates.unwrap()])
        .abs_diff_eq(&omega_davidson, 1e-12));

    println!("------------------------------------------------");

    // Test molecule with lc
    let mut mol: Molecule = get_benzene_molecule();
    mol.calculator.r_lr = None;
    mol.calculator.active_orbitals = Some((3, 3));
    println!(
        "r_lr = {}\n",
        mol.calculator.r_lr.unwrap_or(defaults::LONG_RANGE_RADIUS)
    );

    let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
        run_scc(&mut mol);
    mol.calculator.set_active_orbitals(f.clone());

    // singlets
    mol.multiplicity = 1;
    println!("multiplicity = {}", mol.multiplicity);

    // should call tda
    let (omega_tda, c_ij_tda, XmY_tda, XpY_tda) = get_exc_energies(
        &f.to_vec(),
        &mol,
        None,
        &s,
        &orbe,
        &orbs,
        false,
        Some(String::from("TDA")),
    );
    println!("omega_TDA: {}", &omega_tda);
    // println!("c_ij_TDA: {:?}", &c_ij_tda);

    // should call cadisa
    let (omega_casida, c_ij_casida, XmY_casida, XpY_casida) = get_exc_energies(
        &f.to_vec(),
        &mol,
        None,
        &s,
        &orbe,
        &orbs,
        false,
        Some(String::from("casida")),
    );
    println!("omega_casida: {}", &omega_casida);
    // println!("c_ij_casida: {:?}", &c_ij_casida);

    // should call hermitian davidson
    let (omega_davidson, c_ij_davidson, XmY_davidson, XpY_davidson) =
        get_exc_energies(&f.to_vec(), &mol, nstates, &s, &orbe, &orbs, false, None);
    println!("omega_davidson (non-hermitian): {}\n", &omega_davidson);
    // println!("c_ij_davidson (non-hermitian): {:?}", &c_ij_davidson);

    assert!(omega_casida
        .slice(s![0..nstates.unwrap()])
        .abs_diff_eq(&omega_davidson, 1e-12));

    // triplets
    mol.multiplicity = 3;
    println!("multiplicity = {}", mol.multiplicity);

    // should call tda
    let (omega_tda, c_ij_tda, XmY_tda, XpY_tda) = get_exc_energies(
        &f.to_vec(),
        &mol,
        None,
        &s,
        &orbe,
        &orbs,
        false,
        Some(String::from("TDA")),
    );
    println!("omega_TDA: {}", &omega_tda);
    // println!("c_ij_TDA: {:?}", &c_ij_tda);

    // should call cadisa
    let (omega_casida, c_ij_casida, XmY_casida, XpY_casida) = get_exc_energies(
        &f.to_vec(),
        &mol,
        None,
        &s,
        &orbe,
        &orbs,
        false,
        Some(String::from("casida")),
    );
    println!("omega_casida: {}", &omega_casida);
    // println!("c_ij_casida: {:?}", &c_ij_casida);

    // should call hermitian davidson
    let (omega_davidson, c_ij_davidson, XmY_davidson, XpY_davidson) =
        get_exc_energies(&f.to_vec(), &mol, nstates, &s, &orbe, &orbs, false, None);
    println!("omega_davidson (non-hermitian): {}\n", &omega_davidson);
    // println!("c_ij_davidson (hermitian): {:?}", &c_ij_davidson);

    assert!(omega_casida
        .slice(s![0..nstates.unwrap()])
        .abs_diff_eq(&omega_davidson, 1e-12));

    // let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
    //     run_scc(&mol);
    //
    // mol.calculator.set_active_orbitals(f.clone());
    //
    // let active_occ: Vec<usize> = mol.calculator.active_occ.clone().unwrap();
    // let active_virt: Vec<usize> = mol.calculator.active_virt.clone().unwrap();
    // let n_occ = active_occ.len();
    // let n_virt = active_virt.len();
    //
    // let gamma: Array2<f64> = (&mol.calculator.g0).to_owned();
    // let gamma_lr: Array2<f64> = (&mol.calculator.g0_lr).to_owned();
    //
    // let (q_trans_ov, q_trans_oo, q_trans_vv): (Array3<f64>, Array3<f64>, Array3<f64>) =
    //     trans_charges(
    //         &mol.atomic_numbers,
    //         &mol.calculator.valorbs,
    //         orbs.view(),
    //         s.view(),
    //         &active_occ[..],
    //         &active_virt[..],
    //     );

    // println!("q_trans_oo {}", q_trans_oo);
    // println!("q_trans_ov {}", q_trans_ov);
    // println!("q_trans_vv {}", q_trans_vv);

    // let omega_0: Array2<f64> = get_orbital_en_diff(
    //     orbe.view(),
    //     n_occ,
    //     n_virt,
    //     &active_occ[..],
    //     &active_virt[..],
    // );
    //
    // let df: Array2<f64> = get_orbital_occ_diff(
    //     Array::from(f.clone()).view(),
    //     n_occ,
    //     n_virt,
    //     &active_occ[..],
    //     &active_virt[..],
    // );
    //
    // let spin_couplings: Array1<f64> = mol.calculator.spin_couplings.clone();
    // let spin_couplings_null: Array1<f64> = Array::zeros(mol.calculator.spin_couplings.clone().raw_dim());
    //
    // let nstates: Option<usize> = Some(4);

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
    //     spin_couplings_null.view(),
    // );
    //
    // let (omega_magn, c_ij_magn): (Array1<f64>, Array3<f64>) = tda(
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
    //     spin_couplings.view(),
    // );

    // let (omega, c_ij, XmY, XpY): (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = casida(
    //     gamma.view(),
    //     gamma_lr.view(),
    //     q_trans_ov.view(),
    //     q_trans_oo.view(),
    //     q_trans_vv.view(),
    //     omega_0.view(),
    //     df.view(),
    //      mol.multiplicity,
    //      n_occ,
    //      n_virt,
    //     spin_couplings_null.view(),
    // );
    //
    // let (omega_magn, c_ij_magn, XmY_magn, XpY_magn): (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) = casida(
    //     gamma.view(),
    //     gamma_lr.view(),
    //     q_trans_ov.view(),
    //     q_trans_oo.view(),
    //     q_trans_vv.view(),
    //     omega_0.view(),
    //     df.view(),
    //      mol.multiplicity,
    //      n_occ,
    //      n_virt,
    //     spin_couplings.view(),
    // );
}

#[test]
fn test_get_apbv() {
    let bs: Array3<f64> = array![
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 1.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [1., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ]
    ];
    let g0: Array2<f64> = array![
        [
            0.3674262412520011,
            0.2940917219818045,
            0.2112579256828361,
            0.2940916759143259,
            0.2112579117455455,
            0.1868097112430854,
            0.3451174614381897,
            0.2366619953401044,
            0.2366630563678311,
            0.1554579146160162,
            0.1364904303606813,
            0.1554575493584852
        ],
        [
            0.2940917219818045,
            0.3674262412520011,
            0.294090995478244,
            0.2112583250193026,
            0.1868098160502532,
            0.2112579117455455,
            0.2366631387176211,
            0.3451168577403124,
            0.1554579930590613,
            0.1364905989185344,
            0.1554576014191266,
            0.2366622583239489
        ],
        [
            0.2112579256828361,
            0.294090995478244,
            0.3674262412520011,
            0.1868098038927,
            0.2112583250193026,
            0.2940916759143259,
            0.1554576592046108,
            0.2366624781463542,
            0.1364905705533957,
            0.155457926429148,
            0.2366629484773236,
            0.3451168718829348
        ],
        [
            0.2940916759143259,
            0.2112583250193026,
            0.1868098038927,
            0.3674262412520011,
            0.2940909954782439,
            0.2112579256828361,
            0.2366629484773237,
            0.1554574761291892,
            0.3451177096814089,
            0.236662923037203,
            0.1554576592046108,
            0.1364902803064406
        ],
        [
            0.2112579117455455,
            0.1868098160502532,
            0.2112583250193026,
            0.2940909954782439,
            0.3674262412520011,
            0.2940917219818044,
            0.1554576014191266,
            0.1364902825190264,
            0.2366625846250634,
            0.3451177710286033,
            0.2366631387176212,
            0.1554575535130214
        ],
        [
            0.1868097112430854,
            0.2112579117455455,
            0.2940916759143259,
            0.2112579256828361,
            0.2940917219818044,
            0.3674262412520011,
            0.1364904303606813,
            0.1554576037057185,
            0.1554578081039086,
            0.2366628711360083,
            0.3451174614381897,
            0.2366621711453395
        ],
        [
            0.3451174614381897,
            0.2366631387176211,
            0.1554576592046108,
            0.2366629484773237,
            0.1554576014191266,
            0.1364904303606813,
            0.4720158398964135,
            0.2120636439861317,
            0.2120644266659269,
            0.1231165849476988,
            0.1066221453892915,
            0.1231164148937178
        ],
        [
            0.2366619953401044,
            0.3451168577403124,
            0.2366624781463542,
            0.1554574761291892,
            0.1364902825190264,
            0.1554576037057185,
            0.2120636439861317,
            0.4720158398964135,
            0.1231163300519153,
            0.1066221205641489,
            0.1231164236604586,
            0.2120646896143979
        ],
        [
            0.2366630563678311,
            0.1554579930590613,
            0.1364905705533957,
            0.3451177096814089,
            0.2366625846250634,
            0.1554578081039086,
            0.2120644266659269,
            0.1231163300519153,
            0.4720158398964135,
            0.2120646896143979,
            0.123116539239937,
            0.1066221062107855
        ],
        [
            0.1554579146160162,
            0.1364905989185344,
            0.155457926429148,
            0.236662923037203,
            0.3451177710286033,
            0.2366628711360083,
            0.1231165849476988,
            0.1066221205641489,
            0.2120646896143979,
            0.4720158398964135,
            0.2120643166743429,
            0.1231163300519153
        ],
        [
            0.1364904303606813,
            0.1554576014191266,
            0.2366629484773236,
            0.1554576592046108,
            0.2366631387176212,
            0.3451174614381897,
            0.1066221453892915,
            0.1231164236604586,
            0.123116539239937,
            0.2120643166743429,
            0.4720158398964135,
            0.212063693364872
        ],
        [
            0.1554575493584852,
            0.2366622583239489,
            0.3451168718829348,
            0.1364902803064406,
            0.1554575535130214,
            0.2366621711453395,
            0.1231164148937178,
            0.2120646896143979,
            0.1066221062107855,
            0.1231163300519153,
            0.212063693364872,
            0.4720158398964135
        ]
    ];
    let g0lr: Array2<f64> = array![
        [
            0.2615511628331274,
            0.2323495876478454,
            0.1895763999074472,
            0.2323495672724683,
            0.1895763914101345,
            0.1737637997635455,
            0.2541906040581033,
            0.2064301012503937,
            0.2064306794916344,
            0.1515000901420384,
            0.1349602869736729,
            0.1514997858052974
        ],
        [
            0.2323495876478454,
            0.2615511628331274,
            0.2323492663192074,
            0.1895766433753961,
            0.1737638717160474,
            0.1895763914101345,
            0.2064307243706847,
            0.2541903856179916,
            0.1515001555016049,
            0.134960440189187,
            0.151499829182852,
            0.2064302445721806
        ],
        [
            0.1895763999074472,
            0.2323492663192074,
            0.2615511628331274,
            0.1737638633696129,
            0.1895766433753961,
            0.2323495672724683,
            0.1514998773304012,
            0.2064303643715708,
            0.1349604144058846,
            0.1515000999848651,
            0.2064306206933584,
            0.2541903907353175
        ],
        [
            0.2323495672724683,
            0.1895766433753961,
            0.1737638633696129,
            0.2615511628331274,
            0.2323492663192073,
            0.1895763999074471,
            0.2064306206933585,
            0.151499724789745,
            0.2541906938815215,
            0.2064306068289718,
            0.1514998773304012,
            0.1349601505774632
        ],
        [
            0.1895763914101345,
            0.1737638717160474,
            0.1895766433753961,
            0.2323492663192073,
            0.2615511628331274,
            0.2323495876478454,
            0.151499829182852,
            0.1349601525886586,
            0.2064304224005721,
            0.2541907160791556,
            0.2064307243706848,
            0.1514997892669075
        ],
        [
            0.1737637997635455,
            0.1895763914101345,
            0.2323495672724683,
            0.1895763999074471,
            0.2323495876478454,
            0.2615511628331274,
            0.1349602869736729,
            0.1514998310880677,
            0.1515000013949935,
            0.2064305785437932,
            0.2541906040581032,
            0.2064301970613195
        ],
        [
            0.2541906040581033,
            0.2064307243706847,
            0.1514998773304012,
            0.2064306206933585,
            0.151499829182852,
            0.1349602869736729,
            0.2923649998054588,
            0.1949612872956466,
            0.1949617838858273,
            0.1227575138839422,
            0.1065593574365394,
            0.1227573491359603
        ],
        [
            0.2064301012503937,
            0.2541903856179916,
            0.2064303643715708,
            0.151499724789745,
            0.1349601525886586,
            0.1514998310880677,
            0.1949612872956466,
            0.2923649998054588,
            0.1227572669413572,
            0.1065593328114777,
            0.122757357629164,
            0.1949619507193638
        ],
        [
            0.2064306794916344,
            0.1515001555016049,
            0.1349604144058846,
            0.2541906938815215,
            0.2064304224005721,
            0.1515000013949935,
            0.1949617838858273,
            0.1227572669413572,
            0.2923649998054588,
            0.1949619507193638,
            0.1227574696023529,
            0.1065593185737966
        ],
        [
            0.1515000901420384,
            0.134960440189187,
            0.1515000999848651,
            0.2064306068289718,
            0.2541907160791556,
            0.2064305785437932,
            0.1227575138839422,
            0.1065593328114777,
            0.1949619507193638,
            0.2923649998054588,
            0.194961714099133,
            0.1227572669413572
        ],
        [
            0.1349602869736729,
            0.151499829182852,
            0.2064306206933584,
            0.1514998773304012,
            0.2064307243706848,
            0.2541906040581032,
            0.1065593574365394,
            0.122757357629164,
            0.1227574696023529,
            0.194961714099133,
            0.2923649998054588,
            0.1949613186252518
        ],
        [
            0.1514997858052974,
            0.2064302445721806,
            0.2541903907353175,
            0.1349601505774632,
            0.1514997892669075,
            0.2064301970613195,
            0.1227573491359603,
            0.1949619507193638,
            0.1065593185737966,
            0.1227572669413572,
            0.1949613186252518,
            0.2923649998054588
        ]
    ];
    let q_oo: Array3<f64> = array![
        [
            [
                1.6379434708238599e-01,
                -4.2868443242224854e-03,
                -1.3583749111476578e-01,
                1.1151830824929513e-03,
                8.3868459470681028e-02,
                1.2299618299434698e-03,
                -2.7284324666983217e-05,
                -9.5928309862857825e-16,
                1.3406686174307816e-02,
                -1.6364476585151352e-01,
                -3.6129313738347186e-03,
                2.8367233680579975e-03,
                7.7218662544643921e-02,
                -8.3242738063132053e-17,
                -3.0489136700530535e-17
            ],
            [
                -4.2868443242224837e-03,
                3.5710858805168985e-02,
                6.2278660206898486e-03,
                -6.6559067599101795e-02,
                -4.8297343544338430e-03,
                3.5369533343521031e-03,
                7.8707616062201863e-02,
                -3.3003413295587873e-16,
                -3.4551189117054677e-03,
                6.1435111402208514e-04,
                3.2093993911680406e-02,
                7.0888067888587356e-02,
                -1.8466360058650893e-03,
                -5.2356758208005343e-16,
                3.7015572175842926e-17
            ],
            [
                -1.3583749111476576e-01,
                6.2278660206898486e-03,
                2.3226023798564599e-01,
                -2.2461888446020505e-04,
                -1.7968834472608466e-01,
                1.1090300240109847e-01,
                -2.5062568951567136e-03,
                3.8859953651367645e-15,
                -1.0847705914308479e-01,
                4.3197750176734377e-02,
                -6.9651166891431787e-05,
                -1.3801433368062117e-03,
                2.3705942506673860e-02,
                1.8597829042762203e-16,
                -2.6351534099743624e-17
            ],
            [
                1.1151830824929509e-03,
                -6.6559067599101795e-02,
                -2.2461888446020494e-04,
                1.2273483700497753e-01,
                2.6815739117957765e-04,
                -1.3460725562407382e-03,
                -1.4353571849472169e-01,
                1.1193841213928545e-16,
                1.2582181741336066e-03,
                1.3407517959353327e-03,
                -5.8787819910876553e-02,
                -1.2815651999232017e-01,
                4.1935752024622238e-03,
                1.0693946754512777e-17,
                -9.4359907606144524e-17
            ],
            [
                8.3868459470681028e-02,
                -4.8297343544338438e-03,
                -1.7968834472608466e-01,
                2.6815739117957765e-04,
                1.4374459458946665e-01,
                -1.0316987231806167e-01,
                1.8778038623695073e-03,
                -2.6060172667341267e-15,
                9.6361101990568146e-02,
                3.4758010448901824e-03,
                8.4496105548964237e-04,
                9.7177974315203978e-05,
                -4.2657714571115633e-02,
                8.6599430067031060e-16,
                5.8421116021271141e-17
            ],
            [
                1.2299618299434663e-03,
                3.5369533343521031e-03,
                1.1090300240109847e-01,
                -1.3460725562407382e-03,
                -1.0316987231806167e-01,
                1.0381115071522781e-01,
                7.4208447662857391e-07,
                2.9274733089909148e-15,
                -9.1016389647090704e-02,
                -8.5074407282583253e-02,
                -1.8711794461963420e-03,
                2.9869161588972817e-03,
                8.0566371761064626e-02,
                1.5781670809893709e-16,
                -4.6825300885929762e-17
            ],
            [
                -2.7284324666982784e-05,
                7.8707616062201863e-02,
                -2.5062568951567136e-03,
                -1.4353571849472169e-01,
                1.8778038623695071e-03,
                7.4208447662857391e-07,
                1.6627188910863760e-01,
                6.8478535434726328e-17,
                -1.9640760088655666e-06,
                -1.5050033961951578e-03,
                6.8397993395069065e-02,
                1.4740302846660008e-01,
                -5.4640701593214229e-03,
                3.3991150681791236e-16,
                1.1931972951044577e-16
            ],
            [
                -9.5928309862857825e-16,
                -3.3003413295587873e-16,
                3.8859953651367645e-15,
                1.1193841213928548e-16,
                -2.6060172667341267e-15,
                2.9274733089909148e-15,
                6.8478535434726328e-17,
                1.6668436498765807e-01,
                2.6028434666163583e-15,
                -1.8032519450899099e-15,
                -1.4535477925682697e-16,
                1.8071194551563452e-16,
                1.7465383167453963e-15,
                2.3635702761789396e-01,
                6.4092515252490035e-03
            ],
            [
                1.3406686174307819e-02,
                -3.4551189117054677e-03,
                -1.0847705914308479e-01,
                1.2582181741336066e-03,
                9.6361101990568132e-02,
                -9.1016389647090717e-02,
                -1.9640760088655666e-06,
                2.6028434666163583e-15,
                7.9799421909205234e-02,
                6.1740031894731033e-02,
                1.3572046008206002e-03,
                -2.3905940850740435e-03,
                -6.4448539720624781e-02,
                7.2421106626008929e-15,
                2.3769938784026101e-16
            ],
            [
                -1.6364476585151352e-01,
                6.1435111402208557e-04,
                4.3197750176734377e-02,
                1.3407517959353327e-03,
                3.4758010448901824e-03,
                -8.5074407282583253e-02,
                -1.5050033961951578e-03,
                -1.8032519450899099e-15,
                6.1740031894731033e-02,
                2.2707656111913352e-01,
                4.3789430103046327e-03,
                -6.4844172911823553e-03,
                -1.3880633079683177e-01,
                -5.0890565186596933e-16,
                5.7961768536168884e-17
            ],
            [
                -3.6129313738347190e-03,
                3.2093993911680406e-02,
                -6.9651166891431801e-05,
                -5.8787819910876553e-02,
                8.4496105548964226e-04,
                -1.8711794461963418e-03,
                6.8397993395069065e-02,
                -1.4535477925682697e-16,
                1.3572046008206002e-03,
                4.3789430103046327e-03,
                2.8191473940495856e-02,
                6.0713006408357473e-02,
                -5.3102059083330799e-03,
                -6.0379925264363309e-17,
                4.5264707798717304e-17
            ],
            [
                2.8367233680579975e-03,
                7.0888067888587356e-02,
                -1.3801433368062128e-03,
                -1.2815651999232017e-01,
                9.7177974315203978e-05,
                2.9869161588972817e-03,
                1.4740302846660008e-01,
                1.8071194551563452e-16,
                -2.3905940850740435e-03,
                -6.4844172911823553e-03,
                6.0713006408357473e-02,
                1.3014733238357484e-01,
                -1.2747322981145632e-03,
                3.7608639273869124e-16,
                1.0573237435145474e-16
            ],
            [
                7.7218662544643921e-02,
                -1.8466360058650897e-03,
                2.3705942506673860e-02,
                4.1935752024622238e-03,
                -4.2657714571115633e-02,
                8.0566371761064626e-02,
                -5.4640701593214212e-03,
                1.7465383167453963e-15,
                -6.4448539720624781e-02,
                -1.3880633079683177e-01,
                -5.3102059083330799e-03,
                -1.2747322981145632e-03,
                9.5824548718005914e-02,
                -5.6608833786388265e-17,
                -6.0249693835950733e-17
            ],
            [
                -8.3242738063132053e-17,
                -5.2356758208005343e-16,
                1.8597829042762203e-16,
                1.0693946754512780e-17,
                8.6599430067031060e-16,
                1.5781670809893709e-16,
                3.3991150681791236e-16,
                2.3635702761789396e-01,
                7.2421106626008929e-15,
                -5.0890565186596933e-16,
                -6.0379925264363309e-17,
                3.7608639273869124e-16,
                -5.6608833786388265e-17,
                3.3307940827764249e-01,
                9.0320536958803863e-03
            ],
            [
                -3.0489136700530529e-17,
                3.7015572175842926e-17,
                -2.6351534099743627e-17,
                -9.4359907606144524e-17,
                5.8421116021271153e-17,
                -4.6825300885929762e-17,
                1.1931972951044577e-16,
                6.4092515252490035e-03,
                2.3769938784026101e-16,
                5.7961768536168884e-17,
                4.5264707798717304e-17,
                1.0573237435145474e-16,
                -6.0249693835950733e-17,
                9.0320536958803863e-03,
                2.4492055629248108e-04
            ]
        ],
        [
            [
                1.6515561353786393e-01,
                1.1549638811284266e-01,
                -7.3037930647275179e-02,
                -7.3223000355118834e-02,
                -4.1752204332054926e-02,
                6.2922229083042271e-04,
                -7.9539018772625120e-04,
                2.0834409625112257e-16,
                -1.3935959217057495e-02,
                -8.4748837941278368e-02,
                1.4021668979086335e-01,
                6.4936153603092336e-02,
                -4.1581959238951464e-02,
                -3.0343652430329776e-17,
                3.4843055344801731e-16
            ],
            [
                1.1549638811284266e-01,
                1.7649569741505855e-01,
                -8.9167835910872975e-02,
                -1.1525993653905285e-01,
                -1.0442027111782148e-01,
                -9.4659300922474682e-02,
                -4.1325776457682331e-02,
                3.1762019635241275e-15,
                -9.2338969810435309e-02,
                -3.9057384448783097e-03,
                3.8782628780939243e-02,
                -3.7509479699115701e-02,
                -2.0314329668108665e-02,
                -1.2606464648993085e-17,
                4.3304384223165627e-16
            ],
            [
                -7.3037930647275179e-02,
                -8.9167835910872975e-02,
                9.1006234081528137e-02,
                1.1042460790306327e-01,
                -2.4675016836664503e-03,
                5.8772159725098180e-02,
                -6.6258418524134513e-02,
                -1.7004086425052797e-15,
                5.7525033412064261e-02,
                3.5304281426500689e-02,
                -6.1185175039053415e-03,
                -2.1403614972975743e-02,
                -5.7086596297188569e-02,
                4.7173574737147449e-17,
                -2.4614659451488642e-16
            ],
            [
                -7.3223000355118834e-02,
                -1.1525993653905285e-01,
                1.1042460790306327e-01,
                1.3781995847905978e-01,
                8.0226981314429069e-03,
                8.9595116058060351e-02,
                -7.0351624668702692e-02,
                -3.0044898526809557e-15,
                8.3630990911000749e-02,
                2.2995425545251386e-02,
                1.7691333158225155e-02,
                -2.0175304229257804e-03,
                -7.2995436574802036e-02,
                -2.4172809820671389e-16,
                1.7524719405282028e-16
            ],
            [
                -4.1752204332054926e-02,
                -1.0442027111782147e-01,
                -2.4675016836664503e-03,
                8.0226981314429069e-03,
                1.2747420152371311e-01,
                5.0156249752502526e-02,
                1.2548020495871443e-01,
                -1.6263605339811970e-15,
                4.6748224183900282e-02,
                -4.4804190391673201e-02,
                -2.5144952389467456e-02,
                7.7133067955307402e-02,
                8.3857004809180680e-02,
                2.0697289074248733e-16,
                -6.1534959542647409e-16
            ],
            [
                6.2922229083042618e-04,
                -9.4659300922474682e-02,
                5.8772159725098187e-02,
                8.9595116058060364e-02,
                5.0156249752502519e-02,
                1.0383736791445934e-01,
                1.5218164060358183e-04,
                -3.3603247676329022e-15,
                9.0991278585988752e-02,
                -4.4202522155030392e-02,
                7.2768865451639950e-02,
                6.8352638638156882e-02,
                -4.2824887619531272e-02,
                -1.1123401069222725e-16,
                -1.8666296738959209e-19
            ],
            [
                -7.9539018772625120e-04,
                -4.1325776457682331e-02,
                -6.6258418524134513e-02,
                -7.0351624668702692e-02,
                1.2548020495871443e-01,
                1.5218164060358183e-04,
                1.6646909166515192e-01,
                -3.9011674119157633e-16,
                6.5632742815156103e-05,
                -5.8552711769540793e-02,
                -3.5448998153899562e-02,
                7.8569043102662134e-02,
                1.2499827252330956e-01,
                2.5427329820187207e-17,
                -1.9482135528323169e-16
            ],
            [
                2.0834409625112245e-16,
                3.1762019635241275e-15,
                -1.7004086425052797e-15,
                -3.0044898526809557e-15,
                -1.6263605339811970e-15,
                -3.3603247676329022e-15,
                -3.9011674119157623e-16,
                1.6667368082762721e-01,
                2.4005391685796121e-15,
                1.0946422139464762e-15,
                -1.5904015730608756e-15,
                -2.3818958430140926e-15,
                9.0943567202508624e-16,
                1.2372016760380056e-01,
                -2.0148502831249759e-01
            ],
            [
                -1.3935959217057491e-02,
                -9.2338969810435309e-02,
                5.7525033412064261e-02,
                8.3630990911000749e-02,
                4.6748224183900282e-02,
                9.0991278585988752e-02,
                6.5632742815156103e-05,
                2.4005391685796125e-15,
                7.9724822430861680e-02,
                -3.2069123844300655e-02,
                5.2842349539284905e-02,
                5.4642067725630850e-02,
                -3.4299954726693484e-02,
                3.8599166052511209e-15,
                -6.4667052343486295e-15
            ],
            [
                -8.4748837941278368e-02,
                -3.9057384448783110e-03,
                3.5304281426500689e-02,
                2.2995425545251386e-02,
                -4.4804190391673201e-02,
                -4.4202522155030392e-02,
                -5.8552711769540793e-02,
                1.0946422139464760e-15,
                -3.2069123844300655e-02,
                8.1712230486702314e-02,
                -8.8324342983770857e-02,
                -8.8792962973285261e-02,
                -5.7473948887578436e-03,
                -2.2444629317231850e-16,
                3.4702591184095408e-16
            ],
            [
                1.4021668979086335e-01,
                3.8782628780939243e-02,
                -6.1185175039053415e-03,
                1.7691333158225155e-02,
                -2.5144952389467456e-02,
                7.2768865451639964e-02,
                -3.5448998153899562e-02,
                -1.5904015730608755e-15,
                5.2842349539284898e-02,
                -8.8324342983770857e-02,
                1.7356892873356539e-01,
                8.3861046013320922e-02,
                -8.9913693616330281e-02,
                2.7982999082529401e-16,
                -3.0228487335352118e-16
            ],
            [
                6.4936153603092336e-02,
                -3.7509479699115701e-02,
                -2.1403614972975743e-02,
                -2.0175304229257804e-03,
                7.7133067955307416e-02,
                6.8352638638156882e-02,
                7.8569043102662134e-02,
                -2.3818958430140926e-15,
                5.4642067725630850e-02,
                -8.8792962973285261e-02,
                8.3861046013320922e-02,
                1.0558799970943147e-01,
                1.5536288399676437e-02,
                -1.2091147793891393e-16,
                1.2180639164453494e-16
            ],
            [
                -4.1581959238951464e-02,
                -2.0314329668108661e-02,
                -5.7086596297188569e-02,
                -7.2995436574802050e-02,
                8.3857004809180680e-02,
                -4.2824887619531272e-02,
                1.2499827252330957e-01,
                9.0943567202508605e-16,
                -3.4299954726693484e-02,
                -5.7473948887578436e-03,
                -8.9913693616330281e-02,
                1.5536288399676437e-02,
                1.2050499244028967e-01,
                -2.7857842239766995e-17,
                -6.4635119368729140e-17
            ],
            [
                -3.0343652430329776e-17,
                -1.2606464648993085e-17,
                4.7173574737147449e-17,
                -2.4172809820671389e-16,
                2.0697289074248731e-16,
                -1.1123401069222725e-16,
                2.5427329820187200e-17,
                1.2372016760380056e-01,
                3.8599166052511209e-15,
                -2.2444629317231850e-16,
                2.7982999082529401e-16,
                -1.2091147793891396e-16,
                -2.7857842239766995e-17,
                9.1268221928422488e-02,
                -1.4863524774884185e-01
            ],
            [
                3.4843055344801736e-16,
                4.3304384223165627e-16,
                -2.4614659451488638e-16,
                1.7524719405282028e-16,
                -6.1534959542647409e-16,
                -1.8666296738958285e-19,
                -1.9482135528323166e-16,
                -2.0148502831249759e-01,
                -6.4667052343486295e-15,
                3.4702591184095408e-16,
                -3.0228487335352118e-16,
                1.2180639164453494e-16,
                -6.4635119368729140e-17,
                -1.4863524774884185e-01,
                2.4206055959563702e-01
            ]
        ],
        [
            [
                1.6805031143081511e-01,
                1.2122531542478253e-01,
                6.3549064873208549e-02,
                7.4945261456914464e-02,
                -4.3785818649009496e-02,
                -5.8819659712992800e-04,
                8.3127245886916931e-04,
                -1.7245846813297102e-16,
                1.5134331997618093e-02,
                7.9278855274038329e-02,
                1.4387492762185583e-01,
                -6.8378088214938890e-02,
                -3.5360003913284609e-02,
                2.3645514313750718e-16,
                3.4678797986311536e-16
            ],
            [
                1.2122531542478254e-01,
                1.8616493125220157e-01,
                7.9389603055721125e-02,
                1.1957635558436069e-01,
                -1.0759687698281273e-01,
                -9.7903188895103618e-02,
                -3.7077197719206215e-02,
                -2.5566458303862506e-15,
                9.5398497687431902e-02,
                4.3214239789662280e-03,
                3.9261568494144437e-02,
                3.4708866601949002e-02,
                -1.9460420328926406e-02,
                2.0628444337627366e-16,
                2.7051016298162611e-16
            ],
            [
                6.3549064873208549e-02,
                7.9389603055721125e-02,
                7.7262221314351592e-02,
                1.0158130802274545e-01,
                7.5801590982346925e-03,
                -5.2399594674864824e-02,
                6.9544510217815075e-02,
                -1.3110601010099708e-15,
                5.0865033334305837e-02,
                3.3317407966444869e-02,
                2.0694853964689092e-03,
                -1.8328464729307332e-02,
                6.1038504595600049e-02,
                -1.2843310974949994e-16,
                -3.8512831805069691e-17
            ],
            [
                7.4945261456914464e-02,
                1.1957635558436069e-01,
                1.0158130802274545e-01,
                1.3808406470356807e-01,
                -9.6284997874211305e-03,
                -8.8803453394934256e-02,
                7.3318324896764958e-02,
                -2.5545697033416788e-15,
                8.2706581816232916e-02,
                2.4796340210548429e-02,
                -1.6962597128599124e-02,
                1.8330928902583500e-03,
                7.4893730960604157e-02,
                9.0331752060733983e-17,
                3.9397496984773477e-16
            ],
            [
                -4.3785818649009496e-02,
                -1.0759687698281273e-01,
                7.5801590982346959e-03,
                -9.6284997874211374e-03,
                1.2807396849922484e-01,
                5.2834724705476274e-02,
                1.2336949066816244e-01,
                1.3909084459875064e-15,
                -4.9280901187078292e-02,
                4.5033959879216545e-02,
                -2.2654300366637781e-02,
                -7.0778924313526032e-02,
                8.7100726564817002e-02,
                -3.4291102641523658e-16,
                -2.8646781951815296e-16
            ],
            [
                -5.8819659712992800e-04,
                -9.7903188895103618e-02,
                -5.2399594674864824e-02,
                -8.8803453394934243e-02,
                5.2834724705476267e-02,
                1.0396969591338420e-01,
                -1.5252664655421647e-04,
                2.4892444211950049e-15,
                -9.0945609149229903e-02,
                4.0890274821122986e-02,
                7.4623781455718574e-02,
                -7.1334278651493097e-02,
                -3.7823991595145365e-02,
                1.3581061003135961e-16,
                1.8448022173896816e-16
            ],
            [
                8.3127245886916931e-04,
                -3.7077197719206222e-02,
                6.9544510217815089e-02,
                7.3318324896764958e-02,
                1.2336949066816244e-01,
                -1.5252664655421647e-04,
                1.6686551884540088e-01,
                -2.5752387330500453e-18,
                6.1633334085545860e-05,
                5.9948624643468999e-02,
                -3.2965342532979065e-02,
                -6.9051623650224409e-02,
                1.3069862006099153e-01,
                -3.8022438881262268e-16,
                -2.1849736639610163e-16
            ],
            [
                -1.7245846813297112e-16,
                -2.5566458303862506e-15,
                -1.3110601010099708e-15,
                -2.5545697033416788e-15,
                1.3909084459875068e-15,
                2.4892444211950049e-15,
                -2.5752387330500453e-18,
                1.6665773818042862e-01,
                3.1925406958986572e-15,
                3.5247499967452390e-16,
                2.4236784980076961e-15,
                -2.0238129668030022e-15,
                -1.1684521975703022e-15,
                -1.1262806910707485e-01,
                -2.0788052942373933e-01
            ],
            [
                1.5134331997618100e-02,
                9.5398497687431902e-02,
                5.0865033334305851e-02,
                8.2706581816232916e-02,
                -4.9280901187078285e-02,
                -9.0945609149229903e-02,
                6.1633334085545860e-05,
                3.1925406958986576e-15,
                7.9547773929632748e-02,
                -2.9633214993476577e-02,
                -5.4026629260685397e-02,
                5.6961402756603316e-02,
                3.0136596258387667e-02,
                -3.7117110040904390e-15,
                -6.8036989446539495e-15
            ],
            [
                7.9278855274038329e-02,
                4.3214239789662280e-03,
                3.3317407966444862e-02,
                2.4796340210548429e-02,
                4.5033959879216545e-02,
                4.0890274821122986e-02,
                5.9948624643468999e-02,
                3.5247499967452385e-16,
                -2.9633214993476577e-02,
                7.4013677807107991e-02,
                8.3926207768371813e-02,
                -8.4042549740245692e-02,
                1.5820502494697297e-02,
                4.1514970955721997e-16,
                8.7135852108494298e-16
            ],
            [
                1.4387492762185583e-01,
                3.9261568494144444e-02,
                2.0694853964689022e-03,
                -1.6962597128599124e-02,
                -2.2654300366637781e-02,
                7.4623781455718574e-02,
                -3.2965342532979065e-02,
                2.4236784980076961e-15,
                -5.4026629260685397e-02,
                8.3926207768371813e-02,
                1.8105387991240549e-01,
                -9.4145231753613307e-02,
                -8.2922894278909615e-02,
                -1.2180859077159674e-16,
                -4.4448770238596066e-16
            ],
            [
                -6.8378088214938890e-02,
                3.4708866601948996e-02,
                -1.8328464729307332e-02,
                1.8330928902583500e-03,
                -7.0778924313526032e-02,
                -7.1334278651493097e-02,
                -6.9051623650224409e-02,
                -2.0238129668030022e-15,
                5.6961402756603309e-02,
                -8.4042549740245692e-02,
                -9.4145231753613320e-02,
                1.0346837618489904e-01,
                -1.4148792732293393e-02,
                2.1823336651813671e-16,
                2.8119240683917472e-16
            ],
            [
                -3.5360003913284616e-02,
                -1.9460420328926406e-02,
                6.1038504595600049e-02,
                7.4893730960604157e-02,
                8.7100726564817002e-02,
                -3.7823991595145365e-02,
                1.3069862006099153e-01,
                -1.1684521975703024e-15,
                3.0136596258387667e-02,
                1.5820502494697297e-02,
                -8.2922894278909615e-02,
                -1.4148792732293393e-02,
                1.2281799931052628e-01,
                -2.0203533438536955e-16,
                4.8465252102146409e-17
            ],
            [
                2.3645514313750718e-16,
                2.0628444337627366e-16,
                -1.2843310974949996e-16,
                9.0331752060733983e-17,
                -3.4291102641523658e-16,
                1.3581061003135961e-16,
                -3.8022438881262268e-16,
                -1.1262806910707485e-01,
                -3.7117110040904390e-15,
                4.1514970955721997e-16,
                -1.2180859077159674e-16,
                2.1823336651813674e-16,
                -2.0203533438536955e-16,
                7.5643814919705468e-02,
                1.3961771541464835e-01
            ],
            [
                3.4678797986311536e-16,
                2.7051016298162611e-16,
                -3.8512831805069691e-17,
                3.9397496984773477e-16,
                -2.8646781951815296e-16,
                1.8448022173896816e-16,
                -2.1849736639610166e-16,
                -2.0788052942373933e-01,
                -6.8036989446539495e-15,
                8.7135852108494298e-16,
                -4.4448770238596056e-16,
                2.8119240683917472e-16,
                4.8465252102146402e-17,
                1.3961771541464835e-01,
                2.5769597260809263e-01
            ]
        ],
        [
            [
                1.6532082539072948e-01,
                -1.1992437722774833e-01,
                -6.5597164021504464e-02,
                7.2187022732514552e-02,
                -4.3738326227624996e-02,
                5.5321612274692421e-04,
                8.2278551384630180e-04,
                2.9912455447879012e-16,
                -1.4002215659107945e-02,
                -7.8496122654668363e-02,
                -1.4383930299394967e-01,
                -6.7808603148351645e-02,
                -3.6676210226435066e-02,
                6.2989976109923614e-17,
                -3.6612759152896275e-16
            ],
            [
                -1.1992437722774835e-01,
                1.8727474454423951e-01,
                8.2970482023980122e-02,
                -1.1923458796726252e-01,
                1.0717305973288861e-01,
                9.8211032843224799e-02,
                -3.7002998682875703e-02,
                -3.3247230322601598e-15,
                9.5794126610127442e-02,
                4.3714747621838050e-03,
                3.9233566506738836e-02,
                -3.4772960393737416e-02,
                1.9229625423048947e-02,
                -5.6147817442146639e-17,
                2.2687845129297173e-17
            ],
            [
                -6.5597164021504464e-02,
                8.2970482023980122e-02,
                8.0031107178850647e-02,
                -1.0306767625494934e-01,
                -6.4131769977399805e-03,
                5.2635325414824331e-02,
                6.8760187163182121e-02,
                -1.5481843281372531e-15,
                5.1531496189352675e-02,
                3.4790746424119785e-02,
                5.1743597910006538e-03,
                1.9372396591405664e-02,
                -5.9864469026286816e-02,
                1.9428936903118300e-16,
                2.6860241711140798e-16
            ],
            [
                7.2187022732514552e-02,
                -1.1923458796726252e-01,
                -1.0306767625494934e-01,
                1.3738346463765630e-01,
                -8.2912714294677906e-03,
                -8.8259502579405458e-02,
                -7.3606534557460926e-02,
                2.9665964213496956e-15,
                -8.2377025949925922e-02,
                -2.4939975302940114e-02,
                1.7258370979138665e-02,
                1.5400829432135007e-03,
                7.5249887530541265e-02,
                2.4360446041324211e-18,
                1.8872349541208156e-16
            ],
            [
                -4.3738326227625003e-02,
                1.0717305973288861e-01,
                -6.4131769977399770e-03,
                -8.2912714294677906e-03,
                1.2792418386311311e-01,
                5.2500069308202266e-02,
                -1.2360192571851510e-01,
                -1.1944267131165147e-15,
                4.8930643001496688e-02,
                -4.5233215113210080e-02,
                2.2712030230711125e-02,
                -7.0784855064741720e-02,
                8.7402702351804251e-02,
                2.6453194995712011e-16,
                6.8038036666876598e-16
            ],
            [
                5.5321612274692768e-04,
                9.8211032843224799e-02,
                5.2635325414824331e-02,
                -8.8259502579405458e-02,
                5.2500069308202266e-02,
                1.0385702129383527e-01,
                -1.5367411678336024e-04,
                -2.9970165024828030e-15,
                9.1002556159417963e-02,
                -4.0960946193756079e-02,
                -7.4641709108251547e-02,
                -7.1339699938578430e-02,
                -3.7645382350302459e-02,
                2.2008054240823077e-16,
                1.0415486526463850e-16
            ],
            [
                8.2278551384630180e-04,
                -3.7002998682875703e-02,
                6.8760187163182121e-02,
                -7.3606534557460926e-02,
                -1.2360192571851510e-01,
                -1.5367411678336024e-04,
                1.6646582449044406e-01,
                1.3190168248310549e-16,
                -6.6725237327459885e-05,
                6.0054744229673447e-02,
                -3.2841200594984486e-02,
                6.9100649472039768e-02,
                -1.3046939656117026e-01,
                2.0132901498246391e-16,
                7.6915948913034145e-17
            ],
            [
                2.9912455447879012e-16,
                -3.3247230322601598e-15,
                -1.5481843281372531e-15,
                2.9665964213496956e-15,
                -1.1944267131165147e-15,
                -2.9970165024828030e-15,
                1.3190168248310549e-16,
                1.6667443768820317e-01,
                2.4116553177307385e-15,
                5.2614140067250269e-16,
                1.8106405427397649e-15,
                2.1408559725062992e-15,
                8.3624492192146189e-16,
                1.1261901613644160e-01,
                2.0789433705576890e-01
            ],
            [
                -1.4002215659107951e-02,
                9.5794126610127442e-02,
                5.1531496189352675e-02,
                -8.2377025949925922e-02,
                4.8930643001496701e-02,
                9.1002556159417963e-02,
                -6.6725237327459885e-05,
                2.4116553177307385e-15,
                7.9728346427726887e-02,
                -2.9716618753572777e-02,
                -5.4201919978763689e-02,
                -5.7033490804934923e-02,
                -3.0158278172235920e-02,
                3.5821646583365785e-15,
                6.3887046323563975e-15
            ],
            [
                -7.8496122654668363e-02,
                4.3714747621838050e-03,
                3.4790746424119785e-02,
                -2.4939975302940114e-02,
                -4.5233215113210080e-02,
                -4.0960946193756079e-02,
                6.0054744229673447e-02,
                5.2614140067250279e-16,
                -2.9716618753572777e-02,
                7.4123530450114300e-02,
                8.3942900264558140e-02,
                8.4065386764068994e-02,
                -1.5964872728832468e-02,
                -4.2974814375898342e-16,
                -5.5855485151565748e-16
            ],
            [
                -1.4383930299394970e-01,
                3.9233566506738836e-02,
                5.1743597910006572e-03,
                1.7258370979138665e-02,
                2.2712030230711125e-02,
                -7.4641709108251561e-02,
                -3.2841200594984486e-02,
                1.8106405427397649e-15,
                -5.4201919978763689e-02,
                8.3942900264558140e-02,
                1.8115913883082688e-01,
                9.4073379570184856e-02,
                8.2837728186063761e-02,
                -3.0535884735467492e-16,
                1.1852704271555448e-16
            ],
            [
                -6.7808603148351645e-02,
                -3.4772960393737416e-02,
                1.9372396591405664e-02,
                1.5400829432135005e-03,
                -7.0784855064741720e-02,
                -7.1339699938578430e-02,
                6.9100649472039768e-02,
                2.1408559725062992e-15,
                -5.7033490804934923e-02,
                8.4065386764068994e-02,
                9.4073379570184856e-02,
                1.0337510337612546e-01,
                -1.4267288343278832e-02,
                6.4218419991507191e-18,
                2.8676043964964894e-16
            ],
            [
                -3.6676210226435066e-02,
                1.9229625423048947e-02,
                -5.9864469026286816e-02,
                7.5249887530541265e-02,
                8.7402702351804251e-02,
                -3.7645382350302459e-02,
                -1.3046939656117026e-01,
                8.3624492192146189e-16,
                -3.0158278172235920e-02,
                -1.5964872728832468e-02,
                8.2837728186063761e-02,
                -1.4267288343278834e-02,
                1.2271416036393572e-01,
                -3.0680644334711330e-16,
                -1.2861672818968514e-16
            ],
            [
                6.2989976109923602e-17,
                -5.6147817442146639e-17,
                1.9428936903118300e-16,
                2.4360446041324211e-18,
                2.6453194995712011e-16,
                2.2008054240823077e-16,
                2.0132901498246393e-16,
                1.1261901613644160e-01,
                3.5821646583365785e-15,
                -4.2974814375898342e-16,
                -3.0535884735467487e-16,
                6.4218419991507191e-18,
                -3.0680644334711325e-16,
                7.5624075243336822e-02,
                1.3960177760071107e-01
            ],
            [
                -3.6612759152896275e-16,
                2.2687845129297173e-17,
                2.6860241711140803e-16,
                1.8872349541208156e-16,
                6.8038036666876598e-16,
                1.0415486526463850e-16,
                7.6915948913034145e-17,
                2.0789433705576890e-01,
                6.3887046323563975e-15,
                -5.5855485151565748e-16,
                1.1852704271555448e-16,
                2.8676043964964894e-16,
                -1.2861672818968517e-16,
                1.3960177760071107e-01,
                2.5770439171005838e-01
            ]
        ],
        [
            [
                1.6821753237314213e-01,
                -1.1693589989871697e-01,
                7.1201841899072665e-02,
                -7.6145679758584939e-02,
                -4.1898999469727201e-02,
                -6.6429095108164801e-04,
                -8.0378833974380226e-04,
                -2.1375929166113345e-16,
                1.5200685473241458e-02,
                8.5534021178335723e-02,
                -1.4027603448626580e-01,
                6.5544974374260184e-02,
                -4.0337332166702988e-02,
                -1.7513248850076945e-16,
                -2.6823070775407720e-16
            ],
            [
                -1.1693589989871696e-01,
                1.7542685696935728e-01,
                -8.5614360300908729e-02,
                1.1557540846867255e-01,
                1.0484655463888914e-01,
                9.4366971340608882e-02,
                -4.1416168495855901e-02,
                2.8254969627460524e-15,
                -9.1942064250226330e-02,
                -3.8542223518166258e-03,
                3.8824861437070604e-02,
                3.7463542862173994e-02,
                2.0559900606986058e-02,
                4.4572869361989216e-17,
                -3.0160646963841198e-16
            ],
            [
                7.1201841899072665e-02,
                -8.5614360300908729e-02,
                8.7802305015438897e-02,
                -1.0893545171129337e-01,
                3.5517945423694831e-03,
                -5.8539529353194050e-02,
                -6.7046052577335952e-02,
                -1.6002933704586627e-15,
                5.6842067933687061e-02,
                3.3690925480857073e-02,
                -3.0823308761877485e-03,
                2.0453752116090247e-02,
                5.8325542337589686e-02,
                -1.3924842548092886e-16,
                2.8485439808441072e-16
            ],
            [
                -7.6145679758584953e-02,
                1.1557540846867255e-01,
                -1.0893545171129337e-01,
                1.3856924243633117e-01,
                9.3642903216711558e-03,
                9.0158963445419654e-02,
                7.0069319995414742e-02,
                2.4529205025748739e-15,
                -8.3963333782598473e-02,
                -2.2867033388820616e-02,
                -1.7384896208031117e-02,
                -1.7052992848284225e-03,
                -7.2662902212325853e-02,
                1.4401922351649652e-16,
                -4.6748410186952935e-16
            ],
            [
                -4.1898999469727194e-02,
                1.0484655463888914e-01,
                3.5517945423694831e-03,
                9.3642903216711523e-03,
                1.2760198813866216e-01,
                5.0517812473989142e-02,
                -1.2524659682529493e-01,
                2.1063489208181988e-15,
                -4.7113867602088813e-02,
                4.4607411093674838e-02,
                2.5082377233598935e-02,
                7.7109096395686097e-02,
                8.3550569232718086e-02,
                -4.6906461902701445e-16,
                4.7129187096674231e-16
            ],
            [
                -6.6429095108164801e-04,
                9.4366971340608882e-02,
                -5.8539529353194050e-02,
                9.0158963445419668e-02,
                5.0517812473989142e-02,
                1.0399003029376624e-01,
                1.4846086228493746e-04,
                3.3045894765431267e-15,
                -9.0956831591439982e-02,
                4.4137525138055474e-02,
                -7.2750810053657730e-02,
                6.8343132921493382e-02,
                -4.2999206358317089e-02,
                -3.1813568446614377e-16,
                -1.1183039500819342e-16
            ],
            [
                -8.0378833974380226e-04,
                -4.1416168495855894e-02,
                -6.7046052577335952e-02,
                7.0069319995414742e-02,
                -1.2524659682529493e-01,
                1.4846086228493746e-04,
                1.6686274467344098e-01,
                7.9318273256605557e-17,
                -6.3208328523193386e-05,
                -5.8439213754042943e-02,
                -3.5572517479945658e-02,
                -7.8537957714799253e-02,
                -1.2522605818642013e-01,
                -4.3633703888079456e-17,
                2.3667933273942466e-16
            ],
            [
                -2.1375929166113342e-16,
                2.8254969627460524e-15,
                -1.6002933704586623e-15,
                2.4529205025748739e-15,
                2.1063489208181984e-15,
                3.3045894765431263e-15,
                7.9318273256605557e-17,
                1.6665870398596505e-01,
                2.4577887971672098e-15,
                6.7695311728534998e-16,
                -2.1731065694571940e-15,
                2.0807877515867437e-15,
                -1.5048246428484229e-15,
                -1.2372937134170175e-01,
                2.0147124462684712e-01
            ],
            [
                1.5200685473241462e-02,
                -9.1942064250226330e-02,
                5.6842067933687061e-02,
                -8.3963333782598487e-02,
                -4.7113867602088820e-02,
                -9.0956831591439982e-02,
                -6.3208328523193386e-05,
                2.4577887971672098e-15,
                7.9550737171596192e-02,
                -3.1983438195234026e-02,
                5.2670630589557253e-02,
                -5.4573770943924675e-02,
                3.4271660876573698e-02,
                -3.6910727558287612e-15,
                6.5153329288458061e-15
            ],
            [
                8.5534021178335723e-02,
                -3.8542223518166262e-03,
                3.3690925480857073e-02,
                -2.2867033388820620e-02,
                4.4607411093674831e-02,
                4.4137525138055474e-02,
                -5.8439213754042943e-02,
                6.7695311728534998e-16,
                -3.1983438195234026e-02,
                8.1598560165138612e-02,
                -8.8306023382516377e-02,
                8.8761067551815356e-02,
                5.6006950258650472e-03,
                2.5330980386073581e-16,
                -1.0175265568336309e-15
            ],
            [
                -1.4027603448626583e-01,
                3.8824861437070604e-02,
                -3.0823308761877485e-03,
                -1.7384896208031124e-02,
                2.5082377233598935e-02,
                -7.2750810053657744e-02,
                -3.5572517479945658e-02,
                -2.1731065694571940e-15,
                5.2670630589557253e-02,
                -8.8306023382516377e-02,
                1.7347037016242670e-01,
                -8.3920857872463187e-02,
                8.9998395023046310e-02,
                3.5748699087753229e-16,
                2.8409822294138380e-16
            ],
            [
                6.5544974374260184e-02,
                3.7463542862173994e-02,
                2.0453752116090247e-02,
                -1.7052992848284225e-03,
                7.7109096395686097e-02,
                6.8343132921493382e-02,
                -7.8537957714799253e-02,
                2.0807877515867437e-15,
                -5.4573770943924668e-02,
                8.8761067551815356e-02,
                -8.3920857872463187e-02,
                1.0566195325672526e-01,
                1.5424019851358854e-02,
                -2.6450805496382904e-16,
                -2.7573221674746514e-16
            ],
            [
                -4.0337332166702988e-02,
                2.0559900606986058e-02,
                5.8325542337589686e-02,
                -7.2662902212325853e-02,
                8.3550569232718086e-02,
                -4.2999206358317096e-02,
                -1.2522605818642013e-01,
                -1.5048246428484229e-15,
                3.4271660876573698e-02,
                5.6006950258650472e-03,
                8.9998395023046310e-02,
                1.5424019851358854e-02,
                1.2062026665048939e-01,
                2.9599205257621395e-16,
                -2.1363557876746446e-16
            ],
            [
                -1.7513248850076945e-16,
                4.4572869361989216e-17,
                -1.3924842548092886e-16,
                1.4401922351649649e-16,
                -4.6906461902701455e-16,
                -3.1813568446614377e-16,
                -4.3633703888079456e-17,
                -1.2372937134170175e-01,
                -3.6910727558287612e-15,
                2.5330980386073581e-16,
                3.5748699087753229e-16,
                -2.6450805496382904e-16,
                2.9599205257621395e-16,
                9.1290006938361187e-02,
                -1.4864949278950540e-01
            ],
            [
                -2.6823070775407720e-16,
                -3.0160646963841198e-16,
                2.8485439808441072e-16,
                -4.6748410186952935e-16,
                4.7129187096674231e-16,
                -1.1183039500819342e-16,
                2.3667933273942466e-16,
                2.0147124462684712e-01,
                6.5153329288458061e-15,
                -1.0175265568336309e-15,
                2.8409822294138380e-16,
                -2.7573221674746514e-16,
                -2.1363557876746446e-16,
                -1.4864949278950540e-01,
                2.4204918421574978e-01
            ]
        ],
        [
            [
                1.6961716978452382e-01,
                4.4301613890062822e-03,
                1.3755817186045427e-01,
                1.1212169866329402e-03,
                8.7310857859742991e-02,
                -1.2680453887840559e-03,
                -2.7599122869324637e-05,
                8.4070671609898965e-16,
                -1.5734622349342562e-02,
                1.6440873357137212e-01,
                3.6085578647591060e-03,
                2.8687241193583067e-03,
                7.6739192878032422e-02,
                -1.0127984137958530e-16,
                7.6532390424067781e-17
            ],
            [
                4.4301613890062822e-03,
                3.4992982193492436e-02,
                6.1941964099167263e-03,
                6.5858065448687150e-02,
                4.8286203985033142e-03,
                -3.5513563679629304e-03,
                7.8119597569839425e-02,
                -2.2183293532629795e-16,
                -3.4566465424165421e-03,
                6.2052125892856410e-04,
                3.1712123272456920e-02,
                -7.0246721206124393e-02,
                1.8169511866043750e-03,
                4.6899378168692004e-16,
                4.9210673646481418e-17
            ],
            [
                1.3755817186045427e-01,
                6.1941964099167263e-03,
                2.2772766427121863e-01,
                2.2474438436436870e-04,
                1.7753420398147104e-01,
                -1.1144445808821835e-01,
                -2.4944559186034299e-03,
                3.1765555718485974e-15,
                -1.0828882252811971e-01,
                3.9594532449219330e-02,
                -1.4097057539539797e-04,
                1.2969804581894288e-03,
                -2.5549414055550727e-02,
                -3.2136594373919682e-16,
                -5.8765288543064209e-17
            ],
            [
                1.1212169866329407e-03,
                6.5858065448687150e-02,
                2.2474438436436870e-04,
                1.2261237822910795e-01,
                2.6443376952733216e-04,
                -1.3448741711469502e-03,
                1.4381354027233115e-01,
                -3.4487102425751735e-16,
                -1.2532710147131593e-03,
                -1.3365668558251156e-03,
                5.8642708582503270e-02,
                -1.2818986578121122e-01,
                4.1938335840129761e-03,
                5.7445479624539899e-16,
                8.2987790894689600e-17
            ],
            [
                8.7310857859742991e-02,
                4.8286203985033142e-03,
                1.7753420398147104e-01,
                2.6443376952733216e-04,
                1.4238085132675421e-01,
                -1.0283915163695007e-01,
                -1.8773905202350860e-03,
                2.8408956605565509e-15,
                -9.5711527454328330e-02,
                -3.5748722360765851e-03,
                -8.4514080365469583e-04,
                9.7086511922289849e-05,
                -4.2555814986798120e-02,
                -4.5081591056008862e-16,
                -7.8203179618682746e-17
            ],
            [
                -1.2680453887840594e-03,
                -3.5513563679629304e-03,
                -1.1144445808821835e-01,
                -1.3448741711469502e-03,
                -1.0283915163695008e-01,
                1.0401634055368661e-01,
                2.1167181766248641e-06,
                -2.2574401186594329e-15,
                9.0928363575779950e-02,
                8.5123068705690322e-02,
                1.8754910326763848e-03,
                2.9910100457472017e-03,
                8.0726669301552215e-02,
                4.5658976391688168e-17,
                1.0538855299023720e-16
            ],
            [
                -2.7599122869325071e-05,
                7.8119597569839425e-02,
                -2.4944559186034299e-03,
                1.4381354027233115e-01,
                -1.8773905202350864e-03,
                2.1167181766244304e-06,
                1.6706450975196979e-01,
                1.6842795633390283e-16,
                1.8979460200164999e-07,
                -1.5059824324313176e-03,
                6.8425215100507142e-02,
                -1.4785149663881011e-01,
                5.4791054869044089e-03,
                -1.9421678746292124e-16,
                7.2387295338695061e-17
            ],
            [
                8.4070671609898955e-16,
                -2.2183293532629795e-16,
                3.1765555718485974e-15,
                -3.4487102425751739e-16,
                2.8408956605565505e-15,
                -2.2574401186594329e-15,
                1.6842795633390283e-16,
                1.6665107433011761e-01,
                3.3918282694805225e-15,
                -1.7335000242680056e-15,
                2.5924691151977646e-16,
                -3.3013451732040357e-16,
                -1.6210936619596411e-15,
                -2.3633877090935962e-01,
                -6.4092754716283708e-03
            ],
            [
                -1.5734622349342562e-02,
                -3.4566465424165421e-03,
                -1.0828882252811969e-01,
                -1.2532710147131593e-03,
                -9.5711527454328330e-02,
                9.0928363575779936e-02,
                1.8979460200164999e-07,
                3.3918282694805225e-15,
                7.9471039998722154e-02,
                6.1659609972182144e-02,
                1.3580601139716868e-03,
                2.3881159963174332e-03,
                6.4415479237389370e-02,
                -7.6039195652441848e-15,
                -1.2123500820871050e-16
            ],
            [
                1.6440873357137212e-01,
                6.2052125892856410e-04,
                3.9594532449219330e-02,
                -1.3365668558251156e-03,
                -3.5748722360765851e-03,
                8.5123068705690322e-02,
                -1.5059824324313176e-03,
                -1.7335000242680056e-15,
                6.1659609972182144e-02,
                2.2699987660302440e-01,
                4.3823151293521422e-03,
                6.4903742894505938e-03,
                1.3898140552155647e-01,
                8.2842578378436081e-16,
                1.8162661551300958e-16
            ],
            [
                3.6085578647591064e-03,
                3.1712123272456920e-02,
                -1.4097057539539794e-04,
                5.8642708582503270e-02,
                -8.4514080365469572e-04,
                1.8754910326763850e-03,
                6.8425215100507142e-02,
                2.5924691151977646e-16,
                1.3580601139716866e-03,
                4.3823151293521422e-03,
                2.8080006308698138e-02,
                -6.0633673435180391e-02,
                5.3125707082171212e-03,
                -3.8198706851405733e-16,
                2.5268226053757853e-17
            ],
            [
                2.8687241193583067e-03,
                -7.0246721206124393e-02,
                1.2969804581894288e-03,
                -1.2818986578121122e-01,
                9.7086511922289849e-05,
                2.9910100457472017e-03,
                -1.4785149663881011e-01,
                -3.3013451732040352e-16,
                2.3881159963174332e-03,
                6.4903742894505938e-03,
                -6.0633673435180384e-02,
                1.3031844877284826e-01,
                -1.2695132995706104e-03,
                3.5029298484027290e-16,
                -5.3932615601267076e-17
            ],
            [
                7.6739192878032422e-02,
                1.8169511866043754e-03,
                -2.5549414055550727e-02,
                4.1938335840129752e-03,
                -4.2555814986798120e-02,
                8.0726669301552215e-02,
                5.4791054869044080e-03,
                -1.6210936619596411e-15,
                6.4415479237389370e-02,
                1.3898140552155647e-01,
                5.3125707082171212e-03,
                -1.2695132995706104e-03,
                9.6079710749591499e-02,
                3.0166102098489626e-16,
                1.2523902985814448e-16
            ],
            [
                -1.0127984137958530e-16,
                4.6899378168692004e-16,
                -3.2136594373919682e-16,
                5.7445479624539899e-16,
                -4.5081591056008862e-16,
                4.5658976391688181e-17,
                -1.9421678746292124e-16,
                -2.3633877090935962e-01,
                -7.6039195652441848e-15,
                8.2842578378436071e-16,
                -3.8198706851405733e-16,
                3.5029298484027290e-16,
                3.0166102098489626e-16,
                3.3309447269253140e-01,
                9.0331938271073983e-03
            ],
            [
                7.6532390424067781e-17,
                4.9210673646481430e-17,
                -5.8765288543064197e-17,
                8.2987790894689600e-17,
                -7.8203179618682746e-17,
                1.0538855299023720e-16,
                7.2387295338695061e-17,
                -6.4092754716283708e-03,
                -1.2123500820871050e-16,
                1.8162661551300956e-16,
                2.5268226053757856e-17,
                -5.3932615601267076e-17,
                1.2523902985814448e-16,
                9.0331938271073983e-03,
                2.4497131416869602e-04
            ]
        ],
        [
            [
                -2.0758429106043555e-05,
                2.4797318250483150e-05,
                7.7798672666858689e-04,
                -9.6914779585483320e-06,
                -7.4359555418644218e-04,
                7.1914970482870959e-04,
                -8.0999908282928031e-09,
                2.2735000297200223e-17,
                -7.3940594839795907e-04,
                -6.7721216631841548e-04,
                -1.4900652625053403e-05,
                2.7263049207415055e-05,
                7.3576889096224358e-04,
                -4.2901618074003432e-19,
                -3.0295444585716667e-19
            ],
            [
                2.4797318250483150e-05,
                6.8811779340058080e-05,
                2.1579117635491356e-03,
                -2.8180368215769136e-05,
                -2.1620568334257085e-03,
                2.0775273804689723e-03,
                -2.6319549974990580e-08,
                7.5554063831090881e-17,
                -2.4598562420297841e-03,
                -2.3310921118930364e-03,
                -5.1291702918866309e-05,
                1.0146068612848627e-04,
                2.7382078460827294e-03,
                -2.0021784100996869e-18,
                -7.6648082417667487e-19
            ],
            [
                7.7798672666858689e-04,
                2.1579117635491356e-03,
                6.7671304777922853e-02,
                -8.8373134637858555e-04,
                -6.7801718084694790e-02,
                6.5150834610453540e-02,
                -8.2538769818846231e-07,
                2.3694005514351866e-15,
                -7.7141918630087875e-02,
                -7.3104110712279069e-02,
                -1.6085311757725366e-03,
                3.1818795770005640e-03,
                8.5872153627117614e-02,
                -6.2791036928292303e-17,
                -2.4036236026184773e-17
            ],
            [
                -9.6914779585483320e-06,
                -2.8180368215769136e-05,
                -8.8373134637858555e-04,
                1.1533415380622271e-05,
                8.8486847241682118e-04,
                -8.5034521476023365e-04,
                1.0756976251701285e-08,
                -3.0871374900117737e-17,
                1.0050844119292314e-03,
                9.5210524945483891e-04,
                2.0949447012109371e-05,
                -4.1405788120691837e-05,
                -1.1174539999543307e-03,
                8.1538084414171198e-19,
                3.1431317481107229e-19
            ],
            [
                -7.4359555418644218e-04,
                -2.1620568334257085e-03,
                -6.7801718084694790e-02,
                8.8486847241682118e-04,
                6.7889015252411700e-02,
                -6.5240309532475949e-02,
                8.2529983536069256e-07,
                -2.3685233181921697e-15,
                7.7112403034513596e-02,
                7.3047754773537116e-02,
                1.6072908632877394e-03,
                -3.1767528487296522e-03,
                -8.5733790829079087e-02,
                6.2558169073747267e-17,
                2.4114724057904909e-17
            ],
            [
                7.1914970482870959e-04,
                2.0775273804689723e-03,
                6.5150834610453540e-02,
                -8.5034521476023365e-04,
                -6.5240309532475949e-02,
                6.2694222456939397e-02,
                -7.9324899589859019e-07,
                2.2766215348431047e-15,
                -7.4120464968107674e-02,
                -7.0217191675903623e-02,
                -1.5450092035495251e-03,
                3.0540009373282899e-03,
                8.2420978711606835e-02,
                -6.0157916835266894e-17,
                -2.3167749730567886e-17
            ],
            [
                -8.0999908282928031e-09,
                -2.6319549974990580e-08,
                -8.2538769818846231e-07,
                1.0756976251701285e-08,
                8.2529983536069256e-07,
                -7.9324899589859019e-07,
                1.0002297359780358e-11,
                -2.8688920666048604e-20,
                9.3400451065692998e-07,
                8.8401765718601449e-07,
                1.9451288162143751e-08,
                -3.8373659815167591e-08,
                -1.0356232290632352e-06,
                7.5216642110955776e-22,
                2.9441523686075582e-22
            ],
            [
                2.2735000297200223e-17,
                7.5554063831090881e-17,
                2.3694005514351866e-15,
                -3.0871374900117737e-17,
                -2.3685233181921697e-15,
                2.2766215348431047e-15,
                -2.8688920666048604e-20,
                8.2277446613380248e-29,
                -2.6786336576893144e-15,
                -2.5348641395347306e-15,
                -5.5775321479476736e-17,
                1.0999514957984505e-16,
                2.9685344238979015e-15,
                -2.1541077279398434e-30,
                -8.4562716378518894e-31
            ],
            [
                -7.3940594839795907e-04,
                -2.4598562420297841e-03,
                -7.7141918630087875e-02,
                1.0050844119292314e-03,
                7.7112403034513596e-02,
                -7.4120464968107674e-02,
                9.3400451065692998e-07,
                -2.6786336576893144e-15,
                8.7205872601875470e-02,
                8.2524667091726234e-02,
                1.8158132219166139e-03,
                -3.5809265850436436e-03,
                -9.6641568920198928e-02,
                7.0124715105853419e-17,
                2.7532271902407861e-17
            ],
            [
                -6.7721216631841548e-04,
                -2.3310921118930364e-03,
                -7.3104110712279069e-02,
                9.5210524945483891e-04,
                7.3047754773537116e-02,
                -7.0217191675903623e-02,
                8.8401765718601449e-07,
                -2.5348641395347306e-15,
                8.2524667091726234e-02,
                7.8076013438382899e-02,
                1.7179280614313332e-03,
                -3.3861224534915491e-03,
                -9.1384218807020204e-02,
                6.6222583611703708e-17,
                2.6112215292223707e-17
            ],
            [
                -1.4900652625053403e-05,
                -5.1291702918866309e-05,
                -1.6085311757725366e-03,
                2.0949447012109371e-05,
                1.6072908632877394e-03,
                -1.5450092035495251e-03,
                1.9451288162143751e-08,
                -5.5775321479476736e-17,
                1.8158132219166139e-03,
                1.7179280614313332e-03,
                3.7800045034606331e-05,
                -7.4505767951912042e-05,
                -2.0107516766867100e-03,
                1.4571125005621857e-18,
                5.7455494663717032e-19
            ],
            [
                2.7263049207415055e-05,
                1.0146068612848627e-04,
                3.1818795770005640e-03,
                -4.1405788120691837e-05,
                -3.1767528487296522e-03,
                3.0540009373282899e-03,
                -3.8373659815167591e-08,
                1.0999514957984505e-16,
                -3.5809265850436436e-03,
                -3.3861224534915491e-03,
                -7.4505767951912042e-05,
                1.4668789724513255e-04,
                3.9587930970370238e-03,
                -2.8605407222036140e-18,
                -1.1385256599652650e-18
            ],
            [
                7.3576889096224358e-04,
                2.7382078460827294e-03,
                8.5872153627117614e-02,
                -1.1174539999543307e-03,
                -8.5733790829079087e-02,
                8.2420978711606835e-02,
                -1.0356232290632352e-06,
                2.9685344238979015e-15,
                -9.6641568920198928e-02,
                -9.1384218807020204e-02,
                -2.0107516766867100e-03,
                3.9587930970370238e-03,
                1.0683937175075978e-01,
                -7.7199875171508024e-17,
                -3.0726385411583857e-17
            ],
            [
                -4.2901618074003432e-19,
                -2.0021784100996869e-18,
                -6.2791036928292303e-17,
                8.1538084414171198e-19,
                6.2558169073747267e-17,
                -6.0157916835266894e-17,
                7.5216642110955776e-22,
                -2.1541077279398434e-30,
                7.0124715105853419e-17,
                6.6222583611703708e-17,
                1.4571125005621857e-18,
                -2.8605407222036140e-18,
                -7.7199875171508024e-17,
                5.5375111117654401e-32,
                2.2565285045392916e-32
            ],
            [
                -3.0295444585716667e-19,
                -7.6648082417667487e-19,
                -2.4036236026184773e-17,
                3.1431317481107229e-19,
                2.4114724057904909e-17,
                -2.3167749730567886e-17,
                2.9441523686075582e-22,
                -8.4562716378518894e-31,
                2.7532271902407861e-17,
                2.6112215292223707e-17,
                5.7455494663717032e-19,
                -1.1385256599652650e-18,
                -3.0726385411583857e-17,
                2.2565285045392916e-32,
                8.5135954811495435e-33
            ]
        ],
        [
            [
                -2.7152734914200900e-05,
                -3.5866135082936277e-04,
                2.2308973584474640e-04,
                3.3481048942582436e-04,
                1.8727112870112769e-04,
                3.7515272927797678e-04,
                4.0203551610720547e-07,
                -1.2369234802995659e-17,
                3.3176198728373076e-04,
                -1.5115458377064410e-04,
                2.4896010084288960e-04,
                2.3921842206266924e-04,
                -1.5003005733134746e-04,
                -5.2519377155440827e-19,
                1.3881441347770744e-19
            ],
            [
                -3.5866135082936277e-04,
                4.9286220331408039e-02,
                -3.0611169507965552e-02,
                -5.0259501532934951e-02,
                -2.8114686413510587e-02,
                -5.5605923542446777e-02,
                -7.4686856873740239e-05,
                2.3969432349187363e-15,
                -6.5803812247752336e-02,
                3.2405784619073140e-02,
                -5.3370163011294505e-02,
                -6.2152720421990826e-02,
                3.8974969804581747e-02,
                4.7166507973000615e-17,
                -5.3702048590004478e-18
            ],
            [
                2.2308973584474640e-04,
                -3.0611169507965552e-02,
                1.9012281998761572e-02,
                3.1216006308681356e-02,
                1.7461936789057518e-02,
                3.4536598707710663e-02,
                4.6388809152391903e-05,
                -1.4887730861072430e-15,
                4.0871702809671812e-02,
                -2.0127838202948355e-02,
                3.3149205054769620e-02,
                3.8604775752831666e-02,
                -2.4208432797531720e-02,
                -2.9292627451658689e-17,
                3.3342776609082759e-18
            ],
            [
                3.3481048942582436e-04,
                -5.0259501532934951e-02,
                3.1216006308681356e-02,
                5.1219852258643110e-02,
                2.8651878057419596e-02,
                5.6673345111644902e-02,
                7.6014703835767969e-05,
                -2.4390037768192124e-15,
                6.6950388893826346e-02,
                -3.2957722413245891e-02,
                5.4279187024448505e-02,
                6.3158627615386140e-02,
                -3.9605778828644356e-02,
                -4.8287045607146580e-17,
                5.5799112718578914e-18
            ],
            [
                1.8727112870112769e-04,
                -2.8114686413510587e-02,
                1.7461936789057518e-02,
                2.8651878057419596e-02,
                1.6027576797207198e-02,
                3.1702510913070803e-02,
                4.2521812509903680e-05,
                -1.3643522975513347e-15,
                3.7451317540968636e-02,
                -1.8436182871001166e-02,
                3.0363172731114087e-02,
                3.5330201015110932e-02,
                -2.2155011611300804e-02,
                -2.7011427497377070e-17,
                3.1214117769715298e-18
            ],
            [
                3.7515272927797678e-04,
                -5.5605923542446777e-02,
                3.4536598707710663e-02,
                5.6673345111644902e-02,
                3.1702510913070803e-02,
                6.2706732601298679e-02,
                8.4123335387573732e-05,
                -2.6992617667510283e-15,
                7.4095685905712180e-02,
                -3.6477089369045548e-02,
                6.0075348458746444e-02,
                6.9911047620444255e-02,
                -4.3840111677467118e-02,
                -5.3394750074696436e-17,
                6.1576439438884289e-18
            ],
            [
                4.0203551610720547e-07,
                -7.4686856873740239e-05,
                4.6388809152391903e-05,
                7.6014703835767969e-05,
                4.2521812509903680e-05,
                8.4123335387573732e-05,
                1.1250578324681167e-07,
                -3.6081398098251083e-18,
                9.9017920822207646e-05,
                -4.8704315556341074e-05,
                8.0212843342384639e-05,
                9.3171575872874851e-05,
                -5.8426489644442179e-05,
                -7.2339595168288967e-20,
                8.6119092841992330e-21
            ],
            [
                -1.2369234802995659e-17,
                2.3969432349187363e-15,
                -1.4887730861072430e-15,
                -2.4390037768192124e-15,
                -1.3643522975513347e-15,
                -2.6992617667510283e-15,
                -3.6081398098251083e-18,
                1.1570597208837455e-28,
                -3.1751696936588039e-15,
                1.5615619715632757e-15,
                -2.5717914466004841e-15,
                -2.9863591976436110e-15,
                1.8727011636733648e-15,
                2.3248729237141688e-30,
                -2.7817145188071470e-31
            ],
            [
                3.3176198728373076e-04,
                -6.5803812247752336e-02,
                4.0871702809671812e-02,
                6.6950388893826346e-02,
                3.7451317540968636e-02,
                7.4095685905712180e-02,
                9.9017920822207646e-05,
                -3.1751696936588039e-15,
                8.7130018400215614e-02,
                -4.2847679251231348e-02,
                7.0567358007958492e-02,
                8.1929242529538704e-02,
                -5.1376607290143469e-02,
                -6.3873017820587996e-17,
                7.6628848228224361e-18
            ],
            [
                -1.5115458377064410e-04,
                3.2405784619073140e-02,
                -2.0127838202948355e-02,
                -3.2957722413245891e-02,
                -1.8436182871001166e-02,
                -3.6477089369045548e-02,
                -4.8704315556341074e-05,
                1.5615619715632757e-15,
                -4.2847679251231348e-02,
                2.1066016935581345e-02,
                -3.4694375876639691e-02,
                -4.0259403020333193e-02,
                2.5246080127899716e-02,
                3.1529684047939981e-17,
                -3.8146351496236257e-18
            ],
            [
                2.4896010084288960e-04,
                -5.3370163011294505e-02,
                3.3149205054769620e-02,
                5.4279187024448505e-02,
                3.0363172731114087e-02,
                6.0075348458746444e-02,
                8.0212843342384639e-05,
                -2.5717914466004841e-15,
                7.0567358007958492e-02,
                -3.4694375876639691e-02,
                5.7139407079317750e-02,
                6.6304681138626087e-02,
                -4.1578691356240885e-02,
                -5.1927170190894495e-17,
                6.2823870103485601e-18
            ],
            [
                2.3921842206266924e-04,
                -6.2152720421990826e-02,
                3.8604775752831666e-02,
                6.3158627615386140e-02,
                3.5330201015110932e-02,
                6.9911047620444255e-02,
                9.3171575872874851e-05,
                -2.9863591976436110e-15,
                8.1929242529538704e-02,
                -4.0259403020333193e-02,
                6.6304681138626087e-02,
                7.6852694368924585e-02,
                -4.8193230131123449e-02,
                -6.0782293083730611e-17,
                7.4861425295396645e-18
            ],
            [
                -1.5003005733134746e-04,
                3.8974969804581747e-02,
                -2.4208432797531720e-02,
                -3.9605778828644356e-02,
                -2.2155011611300804e-02,
                -4.3840111677467118e-02,
                -5.8426489644442179e-05,
                1.8727011636733648e-15,
                -5.1376607290143469e-02,
                2.5246080127899716e-02,
                -4.1578691356240885e-02,
                -4.8193230131123449e-02,
                3.0221288251321997e-02,
                3.8115474389966138e-17,
                -4.6943722434971171e-18
            ],
            [
                -5.2519377155440827e-19,
                4.7166507973000615e-17,
                -2.9292627451658689e-17,
                -4.8287045607146580e-17,
                -2.7011427497377070e-17,
                -5.3394750074696436e-17,
                -7.2339595168288967e-20,
                2.3248729237141688e-30,
                -6.3873017820587996e-17,
                3.1529684047939981e-17,
                -5.1927170190894495e-17,
                -6.0782293083730611e-17,
                3.8115474389966138e-17,
                4.4025541992574107e-32,
                -4.5295022900910753e-33
            ],
            [
                1.3881441347770744e-19,
                -5.3702048590004478e-18,
                3.3342776609082759e-18,
                5.5799112718578914e-18,
                3.1214117769715298e-18,
                6.1576439438884289e-18,
                8.6119092841992330e-21,
                -2.7817145188071470e-31,
                7.6628848228224361e-18,
                -3.8146351496236257e-18,
                6.2823870103485601e-18,
                7.4861425295396645e-18,
                -4.6943722434971171e-18,
                -4.5295022900910753e-33,
                2.5092426542865505e-34
            ]
        ],
        [
            [
                -2.7661735636431038e-05,
                3.3445475273813092e-04,
                1.7965028031510224e-04,
                -2.9336591117997194e-04,
                1.7434159125725925e-04,
                3.3427184268804079e-04,
                -3.3369494201890000e-07,
                -9.7826244100245314e-18,
                2.8333538171328935e-04,
                -1.1795173575356775e-04,
                -2.1506459208538132e-04,
                -2.0190592228549044e-04,
                -1.0668221687465098e-04,
                3.7944226146238744e-19,
                -1.4671589770696934e-19
            ],
            [
                3.3445475273813092e-04,
                5.3040332818227058e-02,
                2.8438294262388038e-02,
                -5.1357282942500748e-02,
                3.0523533237458742e-02,
                5.7690580842527567e-02,
                -7.4734459702723553e-05,
                -2.3287593824162010e-15,
                6.8265695138050031e-02,
                -3.1148918615489005e-02,
                -5.6788865771146085e-02,
                -6.7290224919097336e-02,
                -3.5547891848212110e-02,
                1.1586618425732152e-16,
                -5.6755838225385404e-17
            ],
            [
                1.7965028031510224e-04,
                2.8438294262388038e-02,
                1.5247573801142850e-02,
                -2.7536239823700865e-02,
                1.6365806225286050e-02,
                3.0931912759218742e-02,
                -4.0071409528838077e-05,
                -1.2486499236450895e-15,
                3.6603198996778649e-02,
                -1.6701779454858963e-02,
                -3.0449696048054243e-02,
                -3.6081016343708805e-02,
                -1.9060778196845943e-02,
                6.2127094126187037e-17,
                -3.0432738222493787e-17
            ],
            [
                -2.9336591117997194e-04,
                -5.1357282942500748e-02,
                -2.7536239823700865e-02,
                4.9696440590168735e-02,
                -2.9536416167070763e-02,
                -5.5829766625776989e-02,
                7.2222207366689643e-05,
                2.2498464266496256e-15,
                -6.5948910051625523e-02,
                3.0080194248555604e-02,
                5.4840453212387494e-02,
                6.4927330850916690e-02,
                3.4299649170400903e-02,
                -1.1183008303782904e-16,
                5.4738718718837978e-17
            ],
            [
                1.7434159125725925e-04,
                3.0523533237458742e-02,
                1.6365806225286050e-02,
                -2.9536416167070763e-02,
                1.7554574717478375e-02,
                3.3181679276820156e-02,
                -4.2924252923072098e-05,
                -1.3371642284144330e-15,
                3.9195794902017314e-02,
                -1.7877728460508599e-02,
                -3.2593630328676251e-02,
                -3.8588590556312419e-02,
                -2.0385484840361373e-02,
                6.6464559468978988e-17,
                -3.2533127708123909e-17
            ],
            [
                3.3427184268804079e-04,
                5.7690580842527567e-02,
                3.0931912759218742e-02,
                -5.5829766625776989e-02,
                3.3181679276820156e-02,
                6.2719290500725208e-02,
                -8.1150382495613871e-05,
                -2.5280724382804927e-15,
                7.4104992163964920e-02,
                -3.3802098018884813e-02,
                -6.1626007277271161e-02,
                -7.2969390025452802e-02,
                -3.8548085052872147e-02,
                1.2567656682443580e-16,
                -6.1522549879462025e-17
            ],
            [
                -3.3369494201890000e-07,
                -7.4734459702723553e-05,
                -4.0071409528838077e-05,
                7.2222207366689643e-05,
                -4.2924252923072098e-05,
                -8.1150382495613871e-05,
                1.0466616469157377e-07,
                3.2586010350389612e-18,
                -9.5507381716149268e-05,
                4.3526657648475712e-05,
                7.9355328846893321e-05,
                9.3785107711520541e-05,
                4.9544626483388846e-05,
                -1.6163424039651356e-19,
                7.8993949129593416e-20
            ],
            [
                -9.7826244100245314e-18,
                -2.3287593824162010e-15,
                -1.2486499236450895e-15,
                2.2498464266496256e-15,
                -1.3371642284144330e-15,
                -2.5280724382804927e-15,
                3.2586010350389612e-18,
                1.0143813826704820e-28,
                -2.9730114003189502e-15,
                1.3546881384971713e-15,
                2.4697908925903676e-15,
                2.9177883093272917e-15,
                1.5414043286563922e-15,
                -5.0293357476180580e-30,
                2.4571214468590641e-30
            ],
            [
                2.8333538171328935e-04,
                6.8265695138050031e-02,
                3.6603198996778649e-02,
                -6.5948910051625523e-02,
                3.9195794902017314e-02,
                7.4104992163964920e-02,
                -9.5507381716149268e-05,
                -2.9730114003189502e-15,
                8.7134448933058245e-02,
                -3.9702534116521818e-02,
                -7.2383420104250057e-02,
                -8.5506944669614102e-02,
                -4.5171468064355472e-02,
                1.4739039304730715e-16,
                -7.2004164924960174e-17
            ],
            [
                -1.1795173575356775e-04,
                -3.1148918615489005e-02,
                -1.6701779454858963e-02,
                3.0080194248555604e-02,
                -1.7877728460508599e-02,
                -3.3802098018884813e-02,
                4.3526657648475712e-05,
                1.3546881384971713e-15,
                -3.9702534116521818e-02,
                1.8085984372921766e-02,
                3.2973354702924174e-02,
                3.8931268812954084e-02,
                2.0566554323153262e-02,
                -6.7119024308331809e-17,
                3.2774369179920309e-17
            ],
            [
                -2.1506459208538132e-04,
                -5.6788865771146085e-02,
                -3.0449696048054243e-02,
                5.4840453212387494e-02,
                -3.2593630328676251e-02,
                -6.1626007277271161e-02,
                7.9355328846893321e-05,
                2.4697908925903676e-15,
                -7.2383420104250057e-02,
                3.2973354702924174e-02,
                6.0115175245656682e-02,
                7.0977350368805273e-02,
                3.7495811875696547e-02,
                -1.2236769371581940e-16,
                5.9752446885804202e-17
            ],
            [
                -2.0190592228549044e-04,
                -6.7290224919097336e-02,
                -3.6081016343708805e-02,
                6.4927330850916690e-02,
                -3.8588590556312419e-02,
                -7.2969390025452802e-02,
                9.3785107711520541e-05,
                2.9177883093272917e-15,
                -8.5506944669614102e-02,
                3.8931268812954084e-02,
                7.0977350368805273e-02,
                8.3707030526152151e-02,
                4.4220664826059991e-02,
                -1.4437151056700215e-16,
                7.0426402050416978e-17
            ],
            [
                -1.0668221687465098e-04,
                -3.5547891848212110e-02,
                -1.9060778196845943e-02,
                3.4299649170400903e-02,
                -2.0385484840361373e-02,
                -3.8548085052872147e-02,
                4.9544626483388846e-05,
                1.5414043286563922e-15,
                -4.5171468064355472e-02,
                2.0566554323153262e-02,
                3.7495811875696547e-02,
                4.4220664826059991e-02,
                2.3360847759494319e-02,
                -7.6268412862344935e-17,
                3.7204804034463862e-17
            ],
            [
                3.7944226146238744e-19,
                1.1586618425732152e-16,
                6.2127094126187037e-17,
                -1.1183008303782904e-16,
                6.6464559468978988e-17,
                1.2567656682443580e-16,
                -1.6163424039651356e-19,
                -5.0293357476180580e-30,
                1.4739039304730715e-16,
                -6.7119024308331809e-17,
                -1.2236769371581940e-16,
                -1.4437151056700215e-16,
                -7.6268412862344935e-17,
                2.4896638027752218e-31,
                -1.2149176911595176e-31
            ],
            [
                -1.4671589770696934e-19,
                -5.6755838225385404e-17,
                -3.0432738222493787e-17,
                5.4738718718837978e-17,
                -3.2533127708123909e-17,
                -6.1522549879462025e-17,
                7.8993949129593416e-20,
                2.4571214468590641e-30,
                -7.2004164924960174e-17,
                3.2774369179920309e-17,
                5.9752446885804202e-17,
                7.0426402050416978e-17,
                3.7204804034463862e-17,
                -1.2149176911595176e-31,
                5.9233776374965636e-32
            ]
        ],
        [
            [
                -2.8364950558585005e-05,
                -2.7223135718336031e-04,
                1.6914882386920489e-04,
                -3.1126956931989448e-04,
                -1.7452736449508749e-04,
                -3.3839151925134370e-04,
                -5.3209152527668026e-07,
                -1.7306217152731798e-17,
                5.1199728764975789e-04,
                -2.6441733486538111e-04,
                4.3568278660769918e-04,
                -5.5841318578315645e-04,
                3.5104552162380358e-04,
                1.8136962508188412e-18,
                2.0615168337928166e-18
            ],
            [
                -2.7223135718336031e-04,
                4.8862399727323383e-02,
                -3.0301026499761727e-02,
                5.0274747627627153e-02,
                2.8191652661960477e-02,
                5.5407359930591302e-02,
                7.0974438611208729e-05,
                2.2570417420351497e-15,
                -6.5445916391947759e-02,
                3.2237034130883649e-02,
                -5.3114390015609858e-02,
                6.1885356923292487e-02,
                -3.8901415446817536e-02,
                -2.1249164455938413e-16,
                -2.0829119042476830e-16
            ],
            [
                1.6914882386920489e-04,
                -3.0301026499761727e-02,
                1.8790563580943043e-02,
                -3.1176521122374248e-02,
                -1.7482288940938267e-02,
                -3.4359422857273017e-02,
                -4.4011849508291900e-05,
                -1.3996064127648093e-15,
                4.0583330274619425e-02,
                -1.9990206285199992e-02,
                3.2936268282921358e-02,
                -3.8374589690867364e-02,
                2.4122440487687899e-02,
                1.3176542817160398e-16,
                1.2915760582648720e-16
            ],
            [
                -3.1126956931989448e-04,
                5.0274747627627153e-02,
                -3.1176521122374248e-02,
                5.1695406078448553e-02,
                2.8988307721995503e-02,
                5.6977911803162688e-02,
                7.2883370710202986e-05,
                2.3173453485263681e-15,
                -6.7183880035571172e-02,
                3.3080365053249641e-02,
                -5.4503854672115402e-02,
                6.3451295551066905e-02,
                -3.9885746035321197e-02,
                -2.1797666295906974e-16,
                -2.1337188448712683e-16
            ],
            [
                -1.7452736449508749e-04,
                2.8191652661960477e-02,
                -1.7482288940938267e-02,
                2.8988307721995503e-02,
                1.6255254534111474e-02,
                3.1950481178533316e-02,
                4.0869558050651321e-05,
                1.2994582424742361e-15,
                -3.7673564989765193e-02,
                1.8549922488478102e-02,
                -3.0563214109671680e-02,
                3.5580550714136350e-02,
                -2.2366080915649976e-02,
                -1.2223116576000534e-16,
                -1.1964918784326647e-16
            ],
            [
                -3.3839151925134370e-04,
                5.5407359930591302e-02,
                -3.4359422857273017e-02,
                5.6977911803162688e-02,
                3.1950481178533316e-02,
                6.2799485549637615e-02,
                8.0345420077234707e-05,
                2.5546632618914678e-15,
                -7.4065729558892165e-02,
                3.6470796052717740e-02,
                -6.0089998575862169e-02,
                6.9962403930171030e-02,
                -4.3978659437528184e-02,
                -2.4032830965437697e-16,
                -2.3529568362527763e-16
            ],
            [
                -5.3209152527668026e-07,
                7.0974438611208729e-05,
                -4.4011849508291900e-05,
                7.2883370710202986e-05,
                4.0869558050651321e-05,
                8.0345420077234707e-05,
                1.0246794399165805e-07,
                3.2567972591500112e-18,
                -9.4388591024474280e-05,
                4.6437587664945477e-05,
                -7.6511402311096166e-05,
                8.8913734496208667e-05,
                -5.5891464020547745e-05,
                -3.0577142385790632e-19,
                -2.9842965590424281e-19
            ],
            [
                -1.7306217152731798e-17,
                2.2570417420351497e-15,
                -1.3996064127648093e-15,
                2.3173453485263681e-15,
                1.2994582424742361e-15,
                2.5546632618914678e-15,
                3.2567972591500112e-18,
                1.0350765128309801e-28,
                -2.9997295589636691e-15,
                1.4756574284946586e-15,
                -2.4313193409466595e-15,
                2.8247706346223473e-15,
                -1.7756597316327545e-15,
                -9.7156454743217242e-30,
                -9.4786744277229338e-30
            ],
            [
                5.1199728764975789e-04,
                -6.5445916391947759e-02,
                4.0583330274619425e-02,
                -6.7183880035571172e-02,
                -3.7673564989765193e-02,
                -7.4065729558892165e-02,
                -9.4388591024474280e-05,
                -2.9997295589636691e-15,
                8.6930924441823071e-02,
                -4.2759747418503580e-02,
                7.0451709873103202e-02,
                -8.1835186329401982e-02,
                5.1441847839682489e-02,
                2.8150343720227419e-16,
                2.7453974286240010e-16
            ],
            [
                -2.6441733486538111e-04,
                3.2237034130883649e-02,
                -1.9990206285199992e-02,
                3.3080365053249641e-02,
                1.8549922488478102e-02,
                3.6470796052717740e-02,
                4.6437587664945477e-05,
                1.4756574284946586e-15,
                -4.2759747418503580e-02,
                2.1027706122526389e-02,
                -3.4645607766739139e-02,
                4.0222618901430016e-02,
                -2.5284050843158459e-02,
                -1.3840403893177175e-16,
                -1.3486297863240595e-16
            ],
            [
                4.3568278660769918e-04,
                -5.3114390015609858e-02,
                3.2936268282921358e-02,
                -5.4503854672115402e-02,
                -3.0563214109671680e-02,
                -6.0089998575862169e-02,
                -7.6511402311096166e-05,
                -2.4313193409466595e-15,
                7.0451709873103202e-02,
                -3.4645607766739139e-02,
                5.7082695094831999e-02,
                -6.6271433613364128e-02,
                4.1658408674289153e-02,
                2.2803680238465496e-16,
                2.2220226706146495e-16
            ],
            [
                -5.5841318578315645e-04,
                6.1885356923292487e-02,
                -3.8374589690867364e-02,
                6.3451295551066905e-02,
                3.5580550714136350e-02,
                6.9962403930171030e-02,
                8.8913734496208667e-05,
                2.8247706346223473e-15,
                -8.1835186329401982e-02,
                4.0222618901430016e-02,
                -6.6271433613364128e-02,
                7.6852007524729773e-02,
                -4.8309344844806097e-02,
                -2.6462292674748425e-16,
                -2.5736396438363357e-16
            ],
            [
                3.5104552162380358e-04,
                -3.8901415446817536e-02,
                2.4122440487687899e-02,
                -3.9885746035321197e-02,
                -2.2366080915649976e-02,
                -4.3978659437528184e-02,
                -5.5891464020547745e-05,
                -1.7756597316327545e-15,
                5.1441847839682489e-02,
                -2.5284050843158459e-02,
                4.1658408674289153e-02,
                -4.8309344844806097e-02,
                3.0367362843623406e-02,
                1.6634265373106711e-16,
                1.6177941507109125e-16
            ],
            [
                1.8136962508188412e-18,
                -2.1249164455938413e-16,
                1.3176542817160398e-16,
                -2.1797666295906974e-16,
                -1.2223116576000534e-16,
                -2.4032830965437697e-16,
                -3.0577142385790632e-19,
                -9.7156454743217242e-30,
                2.8150343720227419e-16,
                -1.3840403893177175e-16,
                2.2803680238465496e-16,
                -2.6462292674748425e-16,
                1.6634265373106711e-16,
                9.1080368907029761e-31,
                8.8682022161443247e-31
            ],
            [
                2.0615168337928166e-18,
                -2.0829119042476830e-16,
                1.2915760582648720e-16,
                -2.1337188448712683e-16,
                -1.1964918784326647e-16,
                -2.3529568362527763e-16,
                -2.9842965590424281e-19,
                -9.4786744277229338e-30,
                2.7453974286240010e-16,
                -1.3486297863240595e-16,
                2.2220226706146495e-16,
                -2.5736396438363357e-16,
                1.6177941507109125e-16,
                8.8682022161443247e-31,
                8.6073607211494013e-31
            ]
        ],
        [
            [
                -2.3130857420080659e-05,
                2.1728769920650176e-05,
                6.8213798771928980e-04,
                9.3383129460677798e-06,
                7.1340534450999602e-04,
                -6.8445962432288647e-04,
                -2.7675079889748370e-09,
                2.3589215907508942e-17,
                -9.2040895882173300e-04,
                -8.9814923732315808e-04,
                -1.9780173460922614e-05,
                -4.1372052303058799e-05,
                -1.1160135697779450e-03,
                1.2881856602402279e-18,
                -1.6839373157562350e-18
            ],
            [
                2.1728769920650176e-05,
                6.9202489245596055e-05,
                2.1713693474344365e-03,
                2.8256577655441650e-05,
                2.1585668556074753e-03,
                -2.0853764106660679e-03,
                -8.4800796515198652e-09,
                6.2312273879690932e-17,
                -2.4617918225428997e-03,
                -2.3391012688203420e-03,
                -5.1515508471538980e-05,
                -1.0194903054544144e-04,
                -2.7500884727812243e-03,
                2.5469037995631199e-18,
                -4.0208991502674056e-18
            ],
            [
                6.8213798771928980e-04,
                2.1713693474344365e-03,
                6.8131141028333300e-02,
                8.8660229747308469e-04,
                6.7729020291207087e-02,
                -6.5432591084651187e-02,
                -2.6607857004567388e-07,
                1.9551272908087134e-15,
                -7.7242003242403406e-02,
                -7.3392144770822093e-02,
                -1.6163616835904472e-03,
                -3.1987484463724594e-03,
                -8.6286658992269505e-02,
                7.9908540469429867e-17,
                -1.2615894457355590e-16
            ],
            [
                9.3383129460677798e-06,
                2.8256577655441650e-05,
                8.8660229747308469e-04,
                1.1530398922906069e-05,
                8.8082574500473066e-04,
                -8.5103362847373872e-04,
                -3.4609320567207339e-09,
                2.5380570867774869e-17,
                -1.0028982266668146e-03,
                -9.5254847910749619e-04,
                -2.0978583580934731e-05,
                -4.1481896729447551e-05,
                -1.1189796521184404e-03,
                1.0323441067083996e-18,
                -1.6352455442692101e-18
            ],
            [
                7.1340534450999602e-04,
                2.1585668556074753e-03,
                6.7729020291207087e-02,
                8.8082574500473066e-04,
                6.7287697307612818e-02,
                -6.5011835282552083e-02,
                -2.6438621636746485e-07,
                1.9388590720355060e-15,
                -7.6612881926894846e-02,
                -7.2766561547904673e-02,
                -1.6025844638870288e-03,
                -3.1688596362341079e-03,
                -8.5480407914839518e-02,
                7.8861890268792058e-17,
                -1.2491861240754237e-16
            ],
            [
                -6.8445962432288647e-04,
                -2.0853764106660679e-03,
                -6.5432591084651187e-02,
                -8.5103362847373872e-04,
                -6.5011835282552083e-02,
                6.2812193676886979e-02,
                2.5543833403136082e-07,
                -1.8737572053568771e-15,
                7.4038583094511196e-02,
                7.0325262267411537e-02,
                1.5488181169898856e-03,
                3.0628997072333730e-03,
                8.2622124338970621e-02,
                -7.6265490195607580e-17,
                1.2074991943508394e-16
            ],
            [
                -2.7675079889748370e-09,
                -8.4800796515198652e-09,
                -2.6607857004567388e-07,
                -3.4609320567207339e-09,
                -2.6438621636746485e-07,
                2.5543833403136082e-07,
                1.0387826640965679e-12,
                -7.6216649622090469e-21,
                3.0115205709534300e-07,
                2.8606060824930016e-07,
                6.3000950985134198e-09,
                1.2460071627267870e-08,
                3.3611207706707392e-07,
                -3.1038771360929940e-22,
                4.9124603136925661e-22
            ],
            [
                2.3589215907508942e-17,
                6.2312273879690932e-17,
                1.9551272908087134e-15,
                2.5380570867774869e-17,
                1.9388590720355060e-15,
                -1.8737572053568771e-15,
                -7.6216649622090469e-21,
                5.5567526935131001e-29,
                -2.1968748566456104e-15,
                -2.0842152774434586e-15,
                -4.5902038007683417e-17,
                -9.0540798435363091e-17,
                -2.4423503138256881e-15,
                2.2276961663406122e-30,
                -3.5639411651414102e-30
            ],
            [
                -9.2040895882173300e-04,
                -2.4617918225428997e-03,
                -7.7242003242403406e-02,
                -1.0028982266668146e-03,
                -7.6612881926894846e-02,
                7.4038583094511196e-02,
                3.0115205709534300e-07,
                -2.1968748566456104e-15,
                8.6849461028202513e-02,
                8.2404846166550721e-02,
                1.8148557632874989e-03,
                3.5806315312962179e-03,
                9.6588019879632420e-02,
                -8.8198502376233117e-17,
                1.4096414628695031e-16
            ],
            [
                -8.9814923732315808e-04,
                -2.3391012688203420e-03,
                -7.3392144770822093e-02,
                -9.5254847910749619e-04,
                -7.2766561547904673e-02,
                7.0325262267411537e-02,
                2.8606060824930016e-07,
                -2.0842152774434586e-15,
                8.2404846166550721e-02,
                7.8168921368532013e-02,
                1.7215654914802790e-03,
                3.3948011091142639e-03,
                9.1575221475482763e-02,
                -8.3417811037047030e-17,
                1.3360659176533099e-16
            ],
            [
                -1.9780173460922614e-05,
                -5.1515508471538980e-05,
                -1.6163616835904472e-03,
                -2.0978583580934731e-05,
                -1.6025844638870288e-03,
                1.5488181169898856e-03,
                6.3000950985134198e-09,
                -4.5902038007683417e-17,
                1.8148557632874989e-03,
                1.7215654914802790e-03,
                3.7915167425489832e-05,
                7.4765959563960130e-05,
                2.0168219243831605e-03,
                -1.8371687551616679e-18,
                2.9425073028495009e-18
            ],
            [
                -4.1372052303058799e-05,
                -1.0194903054544144e-04,
                -3.1987484463724594e-03,
                -4.1481896729447551e-05,
                -3.1688596362341079e-03,
                3.0628997072333730e-03,
                1.2460071627267870e-08,
                -9.0540798435363091e-17,
                3.5806315312962179e-03,
                3.3948011091142639e-03,
                7.4765959563960130e-05,
                1.4726555635979541e-04,
                3.9725086289956581e-03,
                -3.5994303326064330e-18,
                5.7918784494747837e-18
            ],
            [
                -1.1160135697779450e-03,
                -2.7500884727812243e-03,
                -8.6286658992269505e-02,
                -1.1189796521184404e-03,
                -8.5480407914839518e-02,
                8.2622124338970621e-02,
                3.3611207706707392e-07,
                -2.4423503138256881e-15,
                9.6588019879632420e-02,
                9.1575221475482763e-02,
                2.0168219243831605e-03,
                3.9725086289956581e-03,
                1.0715896641062136e-01,
                -9.7095153808832507e-17,
                1.5623672272369783e-16
            ],
            [
                1.2881856602402279e-18,
                2.5469037995631199e-18,
                7.9908540469429867e-17,
                1.0323441067083996e-18,
                7.8861890268792058e-17,
                -7.6265490195607580e-17,
                -3.1038771360929940e-22,
                2.2276961663406122e-30,
                -8.8198502376233117e-17,
                -8.3417811037047030e-17,
                -1.8371687551616679e-18,
                -3.5994303326064330e-18,
                -9.7095153808832507e-17,
                8.5768967642241357e-32,
                -1.4111108987650791e-31
            ],
            [
                -1.6839373157562350e-18,
                -4.0208991502674056e-18,
                -1.2615894457355590e-16,
                -1.6352455442692101e-18,
                -1.2491861240754237e-16,
                1.2074991943508394e-16,
                4.9124603136925661e-22,
                -3.5639411651414102e-30,
                1.4096414628695031e-16,
                1.3360659176533099e-16,
                2.9425073028495009e-18,
                5.7918784494747837e-18,
                1.5623672272369783e-16,
                -1.4111108987650791e-31,
                2.2769877717316992e-31
            ]
        ],
        [
            [
                -2.8730891824526938e-05,
                2.4516839115984446e-04,
                1.3149359639319741e-04,
                2.7017401123551201e-04,
                -1.6086379779437507e-04,
                -2.9759043974606880e-04,
                4.7862074092904132e-07,
                -1.3827093215007680e-17,
                4.6381383091341759e-04,
                -2.2299851825460923e-04,
                -4.0680389270727460e-04,
                5.3532558762327330e-04,
                2.8356155409575375e-04,
                -7.1976600961211668e-19,
                -1.0013541279068228e-18
            ],
            [
                2.4516839115984446e-04,
                5.2606961674938911e-02,
                2.8144669335829452e-02,
                5.1385723242065334e-02,
                -3.0598361823303367e-02,
                -5.7485279533119521e-02,
                7.3409401173332308e-05,
                -1.9739840089208571e-15,
                6.7905856783447099e-02,
                -3.0991517071160008e-02,
                -5.6532516393589714e-02,
                6.7026760708479932e-02,
                3.5501126903407128e-02,
                -8.5476876712698777e-17,
                -1.0775282666510071e-16
            ],
            [
                1.3149359639319741e-04,
                2.8144669335829452e-02,
                1.5057364965862767e-02,
                2.7490970226829198e-02,
                -1.6369890494501861e-02,
                -3.0754237560202367e-02,
                3.9272450386401350e-05,
                -1.0560284474404586e-15,
                3.6327940593216007e-02,
                -1.6579564497767631e-02,
                -3.0243259495363328e-02,
                3.5856793024520245e-02,
                1.8991765811004202e-02,
                -4.5726502212220861e-17,
                -5.7642155639956475e-17
            ],
            [
                2.7017401123551201e-04,
                5.1385723242065334e-02,
                2.7490970226829198e-02,
                5.0161291767733311e-02,
                -2.9869272226698031e-02,
                -5.6120288247549235e-02,
                7.1564978417355566e-05,
                -1.9234040203765278e-15,
                6.6178054853978291e-02,
                -3.0191335663056013e-02,
                -5.5072855900512072e-02,
                6.5241673315293958e-02,
                3.4555620855144770e-02,
                -8.3162310938118203e-17,
                -1.0473846353288866e-16
            ],
            [
                -1.6086379779437507e-04,
                -3.0598361823303367e-02,
                -1.6369890494501861e-02,
                -2.9869272226698031e-02,
                1.7786093450244803e-02,
                3.3417641161445938e-02,
                -4.2614456457496599e-05,
                1.1453207076373889e-15,
                -3.9406741493309180e-02,
                1.7977901340577140e-02,
                3.2793990347793442e-02,
                -3.8849198146934341e-02,
                -2.0576697289375414e-02,
                4.9520347422049437e-17,
                6.2368265506285735e-17
            ],
            [
                -2.9759043974606880e-04,
                -5.7485279533119521e-02,
                -3.0754237560202367e-02,
                -5.6120288247549235e-02,
                3.3417641161445938e-02,
                6.2786468530152406e-02,
                -8.0081104511444876e-05,
                2.1524356034357763e-15,
                -7.4056434570614393e-02,
                3.3787327710205900e-02,
                6.1632409149017367e-02,
                -7.3020681343947277e-02,
                -3.8675820022030655e-02,
                9.3083860556402501e-17,
                1.1724882531330641e-16
            ],
            [
                4.7862074092904132e-07,
                7.3409401173332308e-05,
                3.9272450386401350e-05,
                7.1564978417355566e-05,
                -4.2614456457496599e-05,
                -8.0081104511444876e-05,
                1.0181402433060093e-07,
                -2.7334136257008614e-18,
                9.4084665707473957e-05,
                -4.2887528954856463e-05,
                -7.8232255028033189e-05,
                9.2512458039615268e-05,
                4.8999674039905918e-05,
                -1.1780843901279980e-19,
                -1.4808067006903761e-19
            ],
            [
                -1.3827093215007680e-17,
                -1.9739840089208571e-15,
                -1.0560284474404586e-15,
                -1.9234040203765278e-15,
                1.1453207076373889e-15,
                2.1524356034357763e-15,
                -2.7334136257008614e-18,
                7.3353449546323271e-29,
                -2.5252233277622452e-15,
                1.1507319559932281e-15,
                2.0990792003317992e-15,
                -2.4805245521922937e-15,
                -1.3138211626755883e-15,
                3.1575814038301421e-30,
                3.9659091561640061e-30
            ],
            [
                4.6381383091341759e-04,
                6.7905856783447099e-02,
                3.6327940593216007e-02,
                6.6178054853978291e-02,
                -3.9406741493309180e-02,
                -7.4056434570614393e-02,
                9.4084665707473957e-05,
                -2.5252233277622452e-15,
                8.6927132727081441e-02,
                -3.9616798552349557e-02,
                -7.2266012366201199e-02,
                8.5419452878606944e-02,
                4.5242812802586296e-02,
                -1.0874951977796115e-16,
                -1.3662680618684936e-16
            ],
            [
                -2.2299851825460923e-04,
                -3.0991517071160008e-02,
                -1.6579564497767631e-02,
                -3.0191335663056013e-02,
                1.7977901340577140e-02,
                3.3787327710205900e-02,
                -4.2887528954856463e-05,
                1.1507319559932281e-15,
                -3.9616798552349557e-02,
                1.8050921130834462e-02,
                3.2927135581243452e-02,
                -3.8900061950295106e-02,
                -2.0603590904053559e-02,
                4.9510354496015146e-17,
                6.2165897292996150e-17
            ],
            [
                -4.0680389270727460e-04,
                -5.6532516393589714e-02,
                -3.0243259495363328e-02,
                -5.5072855900512072e-02,
                3.2793990347793442e-02,
                6.1632409149017367e-02,
                -7.8232255028033189e-05,
                2.0990792003317992e-15,
                -7.2266012366201199e-02,
                3.2927135581243452e-02,
                6.0063209479315038e-02,
                -7.0958527016286341e-02,
                -3.7583499555195396e-02,
                9.0312981055937568e-17,
                1.1339816607685286e-16
            ],
            [
                5.3532558762327330e-04,
                6.7026760708479932e-02,
                3.5856793024520245e-02,
                6.5241673315293958e-02,
                -3.8849198146934341e-02,
                -7.3020681343947277e-02,
                9.2512458039615268e-05,
                -2.4805245521922937e-15,
                8.5419452878606944e-02,
                -3.8900061950295106e-02,
                -7.0958527016286341e-02,
                8.3735100442984878e-02,
                4.4350626846058894e-02,
                -1.0650766576271278e-16,
                -1.3356276780109109e-16
            ],
            [
                2.8356155409575375e-04,
                3.5501126903407128e-02,
                1.8991765811004202e-02,
                3.4555620855144770e-02,
                -2.0576697289375414e-02,
                -3.8675820022030655e-02,
                4.8999674039905918e-05,
                -1.3138211626755883e-15,
                4.5242812802586296e-02,
                -2.0603590904053559e-02,
                -3.7583499555195396e-02,
                4.4350626846058894e-02,
                2.3490484751341859e-02,
                -5.6412176765498529e-17,
                -7.0741936580631414e-17
            ],
            [
                -7.1976600961211668e-19,
                -8.5476876712698777e-17,
                -4.5726502212220861e-17,
                -8.3162310938118203e-17,
                4.9520347422049437e-17,
                9.3083860556402501e-17,
                -1.1780843901279980e-19,
                3.1575814038301421e-30,
                -1.0874951977796115e-16,
                4.9510354496015146e-17,
                9.0312981055937568e-17,
                -1.0650766576271278e-16,
                -5.6412176765498529e-17,
                1.3542649917545624e-31,
                1.6970823852878447e-31
            ],
            [
                -1.0013541279068228e-18,
                -1.0775282666510071e-16,
                -5.7642155639956475e-17,
                -1.0473846353288866e-16,
                6.2368265506285735e-17,
                1.1724882531330641e-16,
                -1.4808067006903761e-19,
                3.9659091561640061e-30,
                -1.3662680618684936e-16,
                6.2165897292996150e-17,
                1.1339816607685286e-16,
                -1.3356276780109109e-16,
                -7.0741936580631414e-17,
                1.6970823852878447e-31,
                2.1236461865927713e-31
            ]
        ]
    ];
    let q_vv: Array3<f64> = array![
        [
            [
                2.8818882498412001e-04,
                9.7969398708003842e-03,
                6.9580092529778880e-03,
                -1.0081606696778560e-19,
                6.3534846702971105e-18,
                5.3101967796549979e-18,
                -5.6723892074414782e-18,
                -5.6716134833051544e-18,
                -8.1370227407579583e-21,
                9.6789237831498232e-18,
                -2.8887827691619197e-17,
                -1.3002370179017173e-17,
                -1.7043255594447397e-17,
                1.3390439333472129e-17,
                2.4462374083553045e-17
            ],
            [
                9.7969398708003842e-03,
                3.3304563713253599e-01,
                2.3653651772006642e-01,
                -2.0393177941455004e-18,
                6.5433228473023123e-18,
                5.0753176637642027e-17,
                5.1924195142218023e-17,
                -8.3878603944265280e-17,
                -7.6582817252990655e-17,
                2.4950388087172137e-17,
                -2.4899292994200828e-17,
                4.9534973135611224e-18,
                -6.0031564925952561e-17,
                8.0226387226031550e-18,
                6.1826782330258727e-17
            ],
            [
                6.9580092529778880e-03,
                2.3653651772006642e-01,
                1.6665748022088653e-01,
                -3.5227779671275138e-18,
                -2.8834700400915946e-19,
                6.4142385923091068e-17,
                3.8121842622698370e-17,
                -8.1864524707412044e-17,
                -5.3331338620062981e-17,
                7.5056075479498252e-17,
                -3.6042025051617210e-18,
                4.3982195044482464e-18,
                -1.5339720414457688e-16,
                9.7344861691479542e-17,
                3.3708162776668762e-17
            ],
            [
                -1.0081606696778560e-19,
                -2.0393177941455004e-18,
                -3.5227779671275138e-18,
                6.2659058185430591e-02,
                8.8661497514951670e-02,
                -2.0658987202979815e-03,
                6.9095049004295764e-02,
                5.3267586612642543e-03,
                -1.0200473451350101e-01,
                4.5743678492911354e-04,
                -1.7140440684505206e-02,
                3.6441236785950668e-02,
                -1.0600383199902853e-03,
                2.2739352529949595e-06,
                2.9031024679507179e-02
            ],
            [
                6.3534846702971105e-18,
                6.5433228473023123e-18,
                -2.8834700400915946e-19,
                8.8661497514951670e-02,
                1.3915037897405427e-01,
                -2.7887403192481297e-03,
                8.2210721834477252e-02,
                6.9402450704778382e-03,
                -1.3926673904603962e-01,
                3.1372128639906779e-03,
                -8.4628032910200185e-02,
                2.4605816294986788e-02,
                -2.3731521461622879e-03,
                1.3722722851969025e-03,
                8.8014530707428984e-02
            ],
            [
                5.3101967796549987e-18,
                5.0753176637642033e-17,
                6.4142385923091068e-17,
                -2.0658987202979815e-03,
                -2.7887403192481297e-03,
                1.9640487584586780e-02,
                -1.9124112052241635e-03,
                -1.4290903767900669e-02,
                2.5100990630843377e-03,
                3.7698301583799901e-02,
                2.9747481025896240e-03,
                -2.6547896745144103e-03,
                -7.1630784421812421e-02,
                5.9234958952144243e-02,
                -2.0479194243867389e-03
            ],
            [
                -5.6723892074414790e-18,
                5.1924195142218030e-17,
                3.8121842622698370e-17,
                6.9095049004295764e-02,
                8.2210721834477238e-02,
                -1.9124112052241635e-03,
                9.2372207772050657e-02,
                6.0888863023758786e-03,
                -1.1655091228849736e-01,
                -1.2568466463638447e-03,
                4.7178603585610379e-02,
                6.8199282652862125e-02,
                -1.9994206972128348e-03,
                1.5200968811283303e-05,
                -1.9569211589108248e-02
            ],
            [
                -5.6716134833051544e-18,
                -8.3878603944265280e-17,
                -8.1864524707412032e-17,
                5.3267586612642543e-03,
                6.9402450704778382e-03,
                -1.4290903767900669e-02,
                6.0888863023758795e-03,
                1.0211765864825511e-02,
                -8.1559973393114677e-03,
                -2.6155721925238552e-02,
                -1.0724165752884797e-03,
                4.9230847136180478e-03,
                4.9337026598921287e-02,
                -4.0441740497489123e-02,
                1.5871676614768273e-03
            ],
            [
                -8.1370227407594990e-21,
                -7.6582817252990655e-17,
                -5.3331338620062968e-17,
                -1.0200473451350099e-01,
                -1.3926673904603962e-01,
                2.5100990630843377e-03,
                -1.1655091228849736e-01,
                -8.1559973393114677e-03,
                1.6590279354702434e-01,
                -1.5623622310143145e-03,
                7.5486213477153202e-03,
                -6.6556449346996380e-02,
                4.5111563588504716e-03,
                -2.1083169778851971e-03,
                -3.0703297358435223e-02
            ],
            [
                9.6789237831498232e-18,
                2.4950388087172137e-17,
                7.5056075479498252e-17,
                4.5743678492911365e-04,
                3.1372128639906779e-03,
                3.7698301583799901e-02,
                -1.2568466463638447e-03,
                -2.6155721925238552e-02,
                -1.5623622310143150e-03,
                7.0326358916740384e-02,
                -5.1303252557556274e-03,
                -6.5992929538834424e-03,
                -1.3248412140008650e-01,
                1.0852042946837541e-01,
                5.4758129237766733e-03
            ],
            [
                -2.8887827691619197e-17,
                -2.4899292994200828e-17,
                -3.6042025051617241e-18,
                -1.7140440684505206e-02,
                -8.4628032910200185e-02,
                2.9747481025896244e-03,
                4.7178603585610379e-02,
                -1.0724165752884797e-03,
                7.5486213477153202e-03,
                -5.1303252557556274e-03,
                2.6153656737471698e-01,
                1.0271859709026825e-01,
                -6.5283635161164661e-03,
                2.8953372508005639e-03,
                -2.0457858195862919e-01
            ],
            [
                -1.3002370179017173e-17,
                4.9534973135611224e-18,
                4.3982195044482495e-18,
                3.6441236785950668e-02,
                2.4605816294986788e-02,
                -2.6547896745144103e-03,
                6.8199282652862125e-02,
                4.9230847136180478e-03,
                -6.6556449346996380e-02,
                -6.5992929538834424e-03,
                1.0271859709026825e-01,
                6.9457297036971163e-02,
                5.2493917573003682e-03,
                -5.9382933834411884e-03,
                -7.0192237007539571e-02
            ],
            [
                -1.7043255594447397e-17,
                -6.0031564925952573e-17,
                -1.5339720414457688e-16,
                -1.0600383199902857e-03,
                -2.3731521461622862e-03,
                -7.1630784421812421e-02,
                -1.9994206972128344e-03,
                4.9337026598921287e-02,
                4.5111563588504707e-03,
                -1.3248412140008650e-01,
                -6.5283635161164661e-03,
                5.2493917573003682e-03,
                2.5049109887290477e-01,
                -2.0478057674781555e-01,
                2.0454396636802137e-03
            ],
            [
                1.3390439333472126e-17,
                8.0226387226031550e-18,
                9.7344861691479542e-17,
                2.2739352529949595e-06,
                1.3722722851969016e-03,
                5.9234958952144243e-02,
                1.5200968811283303e-05,
                -4.0441740497489123e-02,
                -2.1083169778851971e-03,
                1.0852042946837541e-01,
                2.8953372508005639e-03,
                -5.9382933834411884e-03,
                -2.0478057674781555e-01,
                1.6700014527180229e-01,
                -2.0597897919033201e-06
            ],
            [
                2.4462374083553045e-17,
                6.1826782330258714e-17,
                3.3708162776668756e-17,
                2.9031024679507179e-02,
                8.8014530707428984e-02,
                -2.0479194243867389e-03,
                -1.9569211589108244e-02,
                1.5871676614768273e-03,
                -3.0703297358435223e-02,
                5.4758129237766733e-03,
                -2.0457858195862919e-01,
                -7.0192237007539571e-02,
                2.0454396636802137e-03,
                -2.0597897919033201e-06,
                1.6188458723696764e-01
            ]
        ],
        [
            [
                2.4136624572419621e-01,
                -1.4899138889706470e-01,
                -2.0136767050528498e-01,
                -1.8953673267553254e-17,
                -6.0985782741800561e-17,
                -6.6437797048312645e-17,
                7.1919377191810367e-17,
                -3.1744614012275501e-17,
                3.9847861294970852e-17,
                8.5507295281355625e-17,
                5.3216514311024950e-17,
                -1.1987874542559676e-16,
                -2.9226028841721717e-17,
                -1.2423367206426497e-16,
                -2.3765123244592033e-17
            ],
            [
                -1.4899138889706470e-01,
                9.1969918572602685e-02,
                1.2430091421509321e-01,
                -1.9528504323373064e-17,
                2.2169551564908034e-17,
                -1.0241652970774045e-17,
                -1.7606623564574123e-17,
                -2.8044911844654282e-17,
                -4.2504032818045496e-17,
                -7.3930272520934680e-17,
                -7.9881871518377530e-17,
                2.5049012649678728e-17,
                4.3313824912320437e-17,
                3.0050833286355613e-17,
                5.7076146152377439e-17
            ],
            [
                -2.0136767050528498e-01,
                1.2430091421509321e-01,
                1.6666137335151332e-01,
                6.8443066240540447e-18,
                8.5803796946423408e-17,
                3.2057227637679860e-17,
                -3.7805672462371502e-17,
                -7.4671486507993004e-18,
                -1.4351095681210815e-17,
                1.6541327081781380e-17,
                -9.6430971981939445e-17,
                -6.0008037988850591e-17,
                9.0681811231133278e-17,
                -3.9964078205022179e-17,
                -1.7945557811767000e-17
            ],
            [
                -1.8953673267553245e-17,
                -1.9528504323373064e-17,
                6.8443066240540447e-18,
                6.2698781672706982e-02,
                4.6125199510522637e-02,
                7.5768898978904722e-02,
                -6.9001687089435063e-02,
                8.5858969762740933e-02,
                5.5522838902479839e-02,
                1.4794911498070006e-02,
                9.1732608195520225e-03,
                1.9116028822744043e-02,
                3.0938110629031686e-02,
                2.5857420884851934e-05,
                -2.9064471271856823e-02
            ],
            [
                -6.0985782741800573e-17,
                2.2169551564908034e-17,
                8.5803796946423408e-17,
                4.6125199510522637e-02,
                5.1939135511487433e-02,
                5.3127933720416180e-02,
                -4.2713234667981090e-02,
                5.4349600261513711e-02,
                4.9581558596233762e-02,
                5.4595857399054738e-02,
                -4.6933920769035489e-03,
                -4.5456495420962265e-02,
                4.2788459706629117e-02,
                -5.0588750620652670e-02,
                -4.5813731396445512e-02
            ],
            [
                -6.6437797048312645e-17,
                -1.0241652970774047e-17,
                3.2057227637679866e-17,
                7.5768898978904722e-02,
                5.3127933720416180e-02,
                1.0681704764042999e-01,
                -7.0075530252018570e-02,
                1.0407344790975753e-01,
                5.8562551514006300e-02,
                5.1397554456941698e-02,
                5.4864579191014597e-02,
                4.2426283843633408e-02,
                -1.9376688848754715e-03,
                3.0808547777948146e-02,
                -7.5188071181677862e-02
            ],
            [
                7.1919377191810367e-17,
                -1.7606623564574123e-17,
                -3.7805672462371502e-17,
                -6.9001687089435076e-02,
                -4.2713234667981090e-02,
                -7.0075530252018570e-02,
                9.2122361625286447e-02,
                -9.7927387521861503e-02,
                -6.3338471943078054e-02,
                3.9864327619970094e-02,
                2.4436412627968883e-02,
                -3.5886212495117788e-02,
                -5.8327442129929123e-02,
                1.4283680466502940e-04,
                -1.9597669622539077e-02
            ],
            [
                -3.1744614012275501e-17,
                -2.8044911844654282e-17,
                -7.4671486507993004e-18,
                8.5858969762740933e-02,
                5.4349600261513711e-02,
                1.0407344790975755e-01,
                -9.7927387521861503e-02,
                1.2035586358100768e-01,
                7.1476948407088181e-02,
                -1.8890813989619788e-03,
                1.5856229263326284e-02,
                5.2369589106087973e-02,
                3.3628168292668202e-02,
                2.2034871432242677e-02,
                -2.5766868546375287e-02
            ],
            [
                3.9847861294970852e-17,
                -4.2504032818045496e-17,
                -1.4351095681210815e-17,
                5.5522838902479839e-02,
                4.9581558596233768e-02,
                5.8562551514006307e-02,
                -6.3338471943078054e-02,
                7.1476948407088181e-02,
                5.6033941866111006e-02,
                1.5336754747816136e-02,
                -1.6613599610390092e-02,
                -1.6312704702407751e-02,
                5.2642858852737118e-02,
                -3.4024996250012859e-02,
                -1.6773522332627022e-02
            ],
            [
                8.5507295281355625e-17,
                -7.3930272520934680e-17,
                1.6541327081781374e-17,
                1.4794911498070006e-02,
                5.4595857399054731e-02,
                5.1397554456941705e-02,
                3.9864327619970101e-02,
                -1.8890813989619823e-03,
                1.5336754747816136e-02,
                2.0862603040296276e-01,
                8.4160773539486783e-02,
                -1.0609755078061052e-01,
                -3.8876242448727089e-02,
                -5.7021375410821973e-02,
                -1.7409150162514037e-01
            ],
            [
                5.3216514311024950e-17,
                -7.9881871518377517e-17,
                -9.6430971981939445e-17,
                9.1732608195520208e-03,
                -4.6933920769035489e-03,
                5.4864579191014597e-02,
                2.4436412627968890e-02,
                1.5856229263326284e-02,
                -1.6613599610390092e-02,
                8.4160773539486769e-02,
                1.2320405460867467e-01,
                6.8365717772004883e-02,
                -1.0607296013441127e-01,
                9.3524173720688136e-02,
                -1.0671987615724896e-01
            ],
            [
                -1.1987874542559676e-16,
                2.5049012649678728e-17,
                -6.0008037988850591e-17,
                1.9116028822744047e-02,
                -4.5456495420962265e-02,
                4.2426283843633388e-02,
                -3.5886212495117788e-02,
                5.2369589106087980e-02,
                -1.6312704702407751e-02,
                -1.0609755078061052e-01,
                6.8365717772004883e-02,
                2.0025862759388580e-01,
                -7.9830352858922329e-02,
                1.7382011139162193e-01,
                3.7781297307010547e-02
            ],
            [
                -2.9226028841721717e-17,
                4.3313824912320437e-17,
                9.0681811231133278e-17,
                3.0938110629031689e-02,
                4.2788459706629117e-02,
                -1.9376688848754724e-03,
                -5.8327442129929123e-02,
                3.3628168292668209e-02,
                5.2642858852737125e-02,
                -3.8876242448727089e-02,
                -1.0607296013441127e-01,
                -7.9830352858922329e-02,
                1.1984418047932781e-01,
                -1.0728352352600261e-01,
                6.0748866037579702e-02
            ],
            [
                -1.2423367206426497e-16,
                3.0050833286355613e-17,
                -3.9964078205022179e-17,
                2.5857420884851934e-05,
                -5.0588750620652670e-02,
                3.0808547777948142e-02,
                1.4283680466502940e-04,
                2.2034871432242677e-02,
                -3.4024996250012859e-02,
                -5.7021375410821973e-02,
                9.3524173720688150e-02,
                1.7382011139162193e-01,
                -1.0728352352600261e-01,
                1.6682235423436173e-01,
                -3.4096632025994861e-05
            ],
            [
                -2.3765123244592036e-17,
                5.7076146152377439e-17,
                -1.7945557811767000e-17,
                -2.9064471271856823e-02,
                -4.5813731396445512e-02,
                -7.5188071181677862e-02,
                -1.9597669622539077e-02,
                -2.5766868546375287e-02,
                -1.6773522332627022e-02,
                -1.7409150162514037e-01,
                -1.0671987615724896e-01,
                3.7781297307010547e-02,
                6.0748866037579681e-02,
                -3.4096632025994861e-05,
                1.6192048236607437e-01
            ]
        ],
        [
            [
                2.5834465750836899e-01,
                1.3918487136448590e-01,
                -2.0833578496961291e-01,
                -4.0691263277970048e-17,
                -6.4324960177601800e-17,
                -1.4149410595128223e-17,
                2.5279476563704285e-17,
                5.3245511181441322e-17,
                1.0219977709079797e-17,
                -8.7556030379427488e-17,
                -2.9333501348567423e-17,
                1.4407263806903844e-16,
                4.7333693591467601e-17,
                1.2746354588189224e-16,
                1.5263001944527959e-17
            ],
            [
                1.3918487136448590e-01,
                7.4986758400109321e-02,
                -1.1224223178533083e-01,
                -3.0100719744118598e-17,
                -3.5710061475034221e-18,
                -2.8787389532144350e-17,
                3.0387234560608153e-17,
                2.7160733079276991e-17,
                -8.4564151436832342e-18,
                6.2216908690746616e-17,
                -3.5473827479680448e-17,
                -9.0863250291723714e-19,
                3.0240954241547834e-17,
                1.2694332227169417e-17,
                -7.8971075344722757e-17
            ],
            [
                -2.0833578496961291e-01,
                -1.1224223178533083e-01,
                1.6667105838329380e-01,
                1.9375097697386488e-17,
                1.9498968818714796e-17,
                -1.2232550138456571e-17,
                -5.3430911769611754e-17,
                -2.0512099296586475e-18,
                6.9875501708910175e-18,
                -2.3797915473866089e-17,
                -1.9940709698301215e-17,
                4.4237367207539469e-17,
                3.8730631788694179e-18,
                3.2918046602747647e-17,
                3.2833038867983402e-17
            ],
            [
                -4.0691263277970048e-17,
                -3.0100719744118604e-17,
                1.9375097697386488e-17,
                6.2777372083001950e-02,
                -4.2570816149016144e-02,
                7.7840118292950428e-02,
                6.8751208655656509e-02,
                -9.1259848101491389e-02,
                4.6741967281949709e-02,
                -1.5204494518784169e-02,
                8.1571716287189740e-03,
                -1.7266456230256338e-02,
                3.2075407788398069e-02,
                2.0425528197465931e-05,
                2.9095253803582109e-02
            ],
            [
                -6.4324960177601812e-17,
                -3.5710061475034267e-18,
                1.9498968818714799e-17,
                -4.2570816149016144e-02,
                4.7083542564912685e-02,
                -5.0285362858482174e-02,
                -3.9121332768246496e-02,
                5.4027223478117517e-02,
                -4.1605759733116102e-02,
                5.1539569158007587e-02,
                9.6604030165364299e-03,
                -4.9905846307538221e-02,
                -4.0233623176698166e-02,
                -5.1934765033750434e-02,
                -4.2220180012417154e-02
            ],
            [
                -1.4149410595128220e-17,
                -2.8787389532144362e-17,
                -1.2232550138456571e-17,
                7.7840118292950428e-02,
                -5.0285362858482174e-02,
                1.1157603176334951e-01,
                7.1637173349873365e-02,
                -1.1223373973946270e-01,
                4.9816969046077103e-02,
                -5.6730849876226536e-02,
                5.1248686648520704e-02,
                -4.0589509169056681e-02,
                2.4612602815296016e-03,
                -2.8394556065574648e-02,
                7.7287072006692104e-02
            ],
            [
                2.5279476563704288e-17,
                3.0387234560608153e-17,
                -5.3430911769611754e-17,
                6.8751208655656509e-02,
                -3.9121332768246503e-02,
                7.1637173349873365e-02,
                9.1449270617790734e-02,
                -1.0364245450205935e-01,
                5.3075794769774629e-02,
                4.1367464616361593e-02,
                -2.2434353741960671e-02,
                -3.2376744805241081e-02,
                5.9849827105214637e-02,
                -1.5512054276273397e-04,
                -1.9681680046826883e-02
            ],
            [
                5.3245511181441316e-17,
                2.7160733079276997e-17,
                -2.0512099296586475e-18,
                -9.1259848101491389e-02,
                5.4027223478117517e-02,
                -1.1223373973946270e-01,
                -1.0364245450205935e-01,
                1.3457850090636139e-01,
                -6.3924999047643385e-02,
                3.0686221687673335e-04,
                -1.3576704806875516e-02,
                4.8155349188440173e-02,
                -4.1752799584928224e-02,
                1.8411903055103627e-02,
                -2.7371171932947330e-02
            ],
            [
                1.0219977709079800e-17,
                -8.4564151436832342e-18,
                6.9875501708910175e-18,
                4.6741967281949709e-02,
                -4.1605759733116102e-02,
                4.9816969046077103e-02,
                5.3075794769774629e-02,
                -6.3924999047643385e-02,
                4.2537217176178498e-02,
                -1.4081044532993511e-02,
                -1.8873646440928957e-02,
                2.4396859624269272e-02,
                4.7904446216793503e-02,
                3.6011232183443487e-02,
                1.3893611930368296e-02
            ],
            [
                -8.7556030379427488e-17,
                6.2216908690746616e-17,
                -2.3797915473866089e-17,
                -1.5204494518784169e-02,
                5.1539569158007587e-02,
                -5.6730849876226536e-02,
                4.1367464616361593e-02,
                3.0686221687674028e-04,
                -1.4081044532993508e-02,
                2.1764335616617261e-01,
                -8.0163344850826823e-02,
                -9.8647932025295190e-02,
                4.9933295084932133e-02,
                -5.1752292990548530e-02,
                -1.7975121326595891e-01
            ],
            [
                -2.9333501348567423e-17,
                -3.5473827479680442e-17,
                -1.9940709698301215e-17,
                8.1571716287189740e-03,
                9.6604030165364299e-03,
                5.1248686648520704e-02,
                -2.2434353741960674e-02,
                -1.3576704806875516e-02,
                -1.8873646440928957e-02,
                -8.0163344850826823e-02,
                1.1306785517618541e-01,
                -7.9056994288267277e-02,
                -9.8641712204693033e-02,
                -9.4766335165041349e-02,
                9.7559262903077509e-02
            ],
            [
                1.4407263806903844e-16,
                -9.0863250291724022e-19,
                4.4237367207539463e-17,
                -1.7266456230256338e-02,
                -4.9905846307538221e-02,
                -4.0589509169056681e-02,
                -3.2376744805241088e-02,
                4.8155349188440159e-02,
                2.4396859624269279e-02,
                -9.8647932025295204e-02,
                -7.9056994288267290e-02,
                2.1057033547230000e-01,
                7.6063599576592644e-02,
                1.8043170056863350e-01,
                3.3294608952833690e-02
            ],
            [
                4.7333693591467595e-17,
                3.0240954241547840e-17,
                3.8730631788694179e-18,
                3.2075407788398069e-02,
                -4.0233623176698166e-02,
                2.4612602815296016e-03,
                5.9849827105214637e-02,
                -4.1752799584928231e-02,
                4.7904446216793503e-02,
                4.9933295084932126e-02,
                -9.8641712204693033e-02,
                7.6063599576592644e-02,
                1.1082010540586404e-01,
                9.7249442527735791e-02,
                -6.2370515426258516e-02
            ],
            [
                1.2746354588189226e-16,
                1.2694332227169411e-17,
                3.2918046602747647e-17,
                2.0425528197474604e-05,
                -5.1934765033750434e-02,
                -2.8394556065574648e-02,
                -1.5512054276273397e-04,
                1.8411903055103620e-02,
                3.6011232183443487e-02,
                -5.1752292990548530e-02,
                -9.4766335165041349e-02,
                1.8043170056863350e-01,
                9.7249442527735777e-02,
                1.6650107676768042e-01,
                -5.0250146845648436e-07
            ],
            [
                1.5263001944527959e-17,
                -7.8971075344722757e-17,
                3.2833038867983402e-17,
                2.9095253803582109e-02,
                -4.2220180012417154e-02,
                7.7287072006692117e-02,
                -1.9681680046826883e-02,
                -2.7371171932947330e-02,
                1.3893611930368296e-02,
                -1.7975121326595889e-01,
                9.7559262903077509e-02,
                3.3294608952833690e-02,
                -6.2370515426258516e-02,
                -5.0250146845301491e-07,
                1.6202291679668526e-01
            ]
        ],
        [
            [
                2.5833567916599603e-01,
                1.3919541742077982e-01,
                2.0832566652006512e-01,
                1.5153836381509874e-17,
                -2.4331802306054807e-17,
                8.1990886037399129e-17,
                2.2986876485175229e-17,
                8.3911461075436495e-17,
                -4.9011613607166157e-17,
                1.0790974092865695e-16,
                2.5997101967287825e-17,
                8.0240479380274754e-17,
                1.0855025493950811e-16,
                -7.3358956583752246e-17,
                6.5925251476932934e-17
            ],
            [
                1.3919541742077982e-01,
                7.5000728871275041e-02,
                1.1224920424626014e-01,
                5.2446281370388930e-17,
                3.4606005717315450e-17,
                -2.2169277518304281e-17,
                -1.6889747010597222e-17,
                -1.2026218547032070e-17,
                1.1119549754166031e-17,
                -3.6832991950271076e-17,
                4.2591222030752945e-17,
                4.8118731729710600e-18,
                4.9571645466362670e-17,
                -7.8900966769903429e-18,
                -5.5279339892309396e-17
            ],
            [
                2.0832566652006512e-01,
                1.1224920424626014e-01,
                1.6666068732112949e-01,
                2.1966127085860711e-18,
                -2.0962374003541760e-17,
                7.1805348423880577e-17,
                4.0756188747428330e-17,
                8.6197734458724817e-17,
                -4.7037549627779082e-17,
                4.7237924483862943e-17,
                3.9452906650065372e-17,
                4.6422797036142164e-17,
                1.0921600828967911e-16,
                -5.4224162372769665e-17,
                1.8207222745342902e-17
            ],
            [
                1.5153836381509874e-17,
                5.2446281370388924e-17,
                2.1966127085860758e-18,
                6.2670843951042471e-02,
                4.2543214400352886e-02,
                -7.7817552548235802e-02,
                -6.8931406618643720e-02,
                -9.1184275305073892e-02,
                4.6302336167075929e-02,
                -1.5255992010915001e-02,
                8.3598145990461600e-03,
                1.7281835985773302e-02,
                -3.1996612158250247e-02,
                -2.1034750064246460e-05,
                -2.9052953316016237e-02
            ],
            [
                -2.4331802306054807e-17,
                3.4606005717315450e-17,
                -2.0962374003541763e-17,
                4.2543214400352893e-02,
                4.7123618390874956e-02,
                -5.0353136397778026e-02,
                -3.9374405833254619e-02,
                -5.4070854145618741e-02,
                4.1418831013990028e-02,
                -5.1683099132190227e-02,
                -1.0015434560885290e-02,
                -4.9792710220079320e-02,
                -4.0009234190343308e-02,
                5.1965841178139280e-02,
                -4.2272701736176699e-02
            ],
            [
                8.1990886037399129e-17,
                -2.2169277518304278e-17,
                7.1805348423880577e-17,
                -7.7817552548235802e-02,
                -5.0353136397778026e-02,
                1.1164784983362228e-01,
                7.1921849462649651e-02,
                1.1228041665789455e-01,
                -4.9428510724192617e-02,
                5.6723645939035242e-02,
                -5.1633375916051563e-02,
                -4.0196285747603512e-02,
                2.3981559610694852e-03,
                2.8424860966776298e-02,
                7.7240289554541791e-02
            ],
            [
                2.2986876485175229e-17,
                -1.6889747010597222e-17,
                4.0756188747428330e-17,
                -6.8931406618643720e-02,
                -3.9374405833254612e-02,
                7.1921849462649651e-02,
                9.1991789613710975e-02,
                1.0394917672616390e-01,
                -5.2791793435170077e-02,
                -4.1108692234892434e-02,
                2.2284949175312697e-02,
                -3.2416312976546013e-02,
                6.0287191357834835e-02,
                -1.5663688368107442e-04,
                -1.9612235096918701e-02
            ],
            [
                8.3911461075436495e-17,
                -1.2026218547032070e-17,
                8.6197734458724817e-17,
                -9.1184275305073892e-02,
                -5.4070854145618741e-02,
                1.1228041665789455e-01,
                1.0394917672616390e-01,
                1.3455839513805970e-01,
                -6.3333422448848709e-02,
                4.4562773182395612e-04,
                -1.4022972415509127e-02,
                -4.8069067951391639e-02,
                4.1815825581343752e-02,
                1.8377764206892435e-02,
                2.7360240780915382e-02
            ],
            [
                -4.9011613607166157e-17,
                1.1119549754166035e-17,
                -4.7037549627779075e-17,
                4.6302336167075936e-02,
                4.1418831013990028e-02,
                -4.9428510724192617e-02,
                -5.2791793435170077e-02,
                -6.3333422448848709e-02,
                4.1967760314116795e-02,
                -1.4456566847898630e-02,
                -1.8968348800649217e-02,
                -2.4457967984869079e-02,
                -4.7525433392079507e-02,
                3.6124788581915562e-02,
                -1.4013140680584018e-02
            ],
            [
                1.0790974092865694e-16,
                -3.6832991950271076e-17,
                4.7237924483862943e-17,
                -1.5255992010915004e-02,
                -5.1683099132190234e-02,
                5.6723645939035242e-02,
                -4.1108692234892427e-02,
                4.4562773182395612e-04,
                -1.4456566847898634e-02,
                2.1733697786744113e-01,
                -7.9137365417101160e-02,
                9.9541726043170214e-02,
                -5.0353898791116833e-02,
                -5.1919817442190347e-02,
                1.7953914082174752e-01
            ],
            [
                2.5997101967287822e-17,
                4.2591222030752951e-17,
                3.9452906650065372e-17,
                8.3598145990461600e-03,
                -1.0015434560885290e-02,
                -5.1633375916051556e-02,
                2.2284949175312693e-02,
                -1.4022972415509125e-02,
                -1.8968348800649217e-02,
                -7.9137365417101160e-02,
                1.1448497977984012e-01,
                7.9836368276600456e-02,
                9.9379767488267087e-02,
                -9.6443749798876127e-02,
                -9.7271637500804087e-02
            ],
            [
                8.0240479380274754e-17,
                4.8118731729710600e-18,
                4.6422797036142170e-17,
                1.7281835985773302e-02,
                -4.9792710220079320e-02,
                -4.0196285747603519e-02,
                -3.2416312976546013e-02,
                -4.8069067951391625e-02,
                -2.4457967984869076e-02,
                9.9541726043170214e-02,
                7.9836368276600456e-02,
                2.0926967807694963e-01,
                7.4621261859518562e-02,
                -1.7975895705703390e-01,
                3.4213911810318763e-02
            ],
            [
                1.0855025493950811e-16,
                4.9571645466362670e-17,
                1.0921600828967910e-16,
                -3.1996612158250254e-02,
                -4.0009234190343308e-02,
                2.3981559610694852e-03,
                6.0287191357834835e-02,
                4.1815825581343752e-02,
                -4.7525433392079500e-02,
                -5.0353898791116833e-02,
                9.9379767488267101e-02,
                7.4621261859518562e-02,
                1.1085415382610733e-01,
                -9.7007387849454341e-02,
                -6.2830840761114976e-02
            ],
            [
                -7.3358956583752246e-17,
                -7.8900966769903475e-18,
                -5.4224162372769665e-17,
                -2.1034750064246460e-05,
                5.1965841178139280e-02,
                2.8424860966776294e-02,
                -1.5663688368107442e-04,
                1.8377764206892442e-02,
                3.6124788581915562e-02,
                -5.1919817442190347e-02,
                -9.6443749798876127e-02,
                -1.7975895705703390e-01,
                -9.7007387849454341e-02,
                1.6681668961960147e-01,
                3.4017042886448634e-05
            ],
            [
                6.5925251476932934e-17,
                -5.5279339892309403e-17,
                1.8207222745342895e-17,
                -2.9052953316016233e-02,
                -4.2272701736176699e-02,
                7.7240289554541791e-02,
                -1.9612235096918701e-02,
                2.7360240780915382e-02,
                -1.4013140680584018e-02,
                1.7953914082174752e-01,
                -9.7271637500804087e-02,
                3.4213911810318763e-02,
                -6.2830840761114976e-02,
                3.4017042886441695e-05,
                1.6191981274103645e-01
            ]
        ],
        [
            [
                2.4137707489609331e-01,
                -1.4898214069933680e-01,
                2.0137781097930216e-01,
                1.1653230150598241e-17,
                4.9375969935323944e-17,
                -3.3656873123789743e-17,
                1.7764770386979121e-17,
                1.6615469328498670e-16,
                -2.8085471162275653e-17,
                -1.3578123829071158e-16,
                -1.9330121907561594e-17,
                -1.0603743075865970e-16,
                2.2353858650742440e-17,
                7.7584690146487823e-17,
                -8.8601549973232043e-17
            ],
            [
                -1.4898214069933680e-01,
                9.1954375769431024e-02,
                -1.2429386123805022e-01,
                5.1774908570993895e-18,
                -1.5295567194016469e-17,
                1.2636991113420516e-17,
                2.4652087819345400e-17,
                -7.1850936360981245e-17,
                2.1785354604982065e-17,
                -3.8928231372374254e-18,
                -1.3389366154873859e-18,
                -1.5154184034511707e-17,
                -1.8204252853156609e-17,
                7.4174985572055250e-18,
                -6.6899349338655394e-18
            ],
            [
                2.0137781097930216e-01,
                -1.2429386123805022e-01,
                1.6667065419439858e-01,
                8.7709548188337076e-18,
                1.9410919280015220e-17,
                -1.9370633426791091e-17,
                7.6844977247880734e-18,
                1.2643270345781938e-16,
                -1.0812380225274249e-17,
                -6.8690629504672535e-17,
                -4.2668030345290713e-17,
                -7.2782323007986121e-18,
                -1.7884011479686417e-17,
                -8.5170423975487533e-18,
                -5.4237163422969022e-17
            ],
            [
                1.1653230150598240e-17,
                5.1774908570993895e-18,
                8.7709548188337076e-18,
                6.2748806393992135e-02,
                -4.6122716413170610e-02,
                -7.5769075576018419e-02,
                6.8680832362703198e-02,
                8.5942150843723170e-02,
                5.5954804541817275e-02,
                1.4739591804686035e-02,
                8.9484924078081958e-03,
                -1.9091151483715259e-02,
                -3.1021851161818718e-02,
                -2.6075632893557410e-05,
                2.9083886207596894e-02
            ],
            [
                4.9375969935323950e-17,
                -1.5295567194016469e-17,
                1.9410919280015220e-17,
                -4.6122716413170610e-02,
                5.1879085432497647e-02,
                5.3059283426335299e-02,
                -4.2362068534618318e-02,
                -5.4328019422235627e-02,
                -4.9777324927023761e-02,
                -5.4420843340158107e-02,
                4.3724587719599466e-03,
                -4.5558751715687719e-02,
                4.3020335713749039e-02,
                5.0562913783429822e-02,
                -4.5759881428950822e-02
            ],
            [
                -3.3656873123789749e-17,
                1.2636991113420516e-17,
                -1.9370633426791091e-17,
                -7.5769075576018405e-02,
                5.3059283426335299e-02,
                1.0679414177726465e-01,
                -6.9684862454632027e-02,
                -1.0410552008239808e-01,
                -5.8964427754469549e-02,
                -5.1448500624110230e-02,
                -5.4471769944168649e-02,
                4.2820412784553348e-02,
                -1.8846060001349585e-03,
                -3.0771368273686282e-02,
                -7.5251487543960507e-02
            ],
            [
                1.7764770386979121e-17,
                2.4652087819345400e-17,
                7.6844977247880719e-18,
                6.8680832362703198e-02,
                -4.2362068534618318e-02,
                -6.9684862454632027e-02,
                9.1319754074639450e-02,
                9.7555971224266527e-02,
                6.3502386192815216e-02,
                -4.0104936649664712e-02,
                -2.4623909141792261e-02,
                -3.5770527652445656e-02,
                -5.7846718755825158e-02,
                1.4227033866598231e-04,
                -1.9696153127749999e-02
            ],
            [
                1.6615469328498670e-16,
                -7.1850936360981245e-17,
                1.2643270345781938e-16,
                8.5942150843723170e-02,
                -5.4328019422235634e-02,
                -1.0410552008239810e-01,
                9.7555971224266513e-02,
                1.2051863905007026e-01,
                7.2099592331811138e-02,
                -2.0005771109523323e-03,
                1.5358479886310851e-02,
                -5.2485191735174526e-02,
                -3.3614724769155974e-02,
                2.2043298509534515e-02,
                2.5770766426238427e-02
            ],
            [
                -2.8085471162275653e-17,
                2.1785354604982065e-17,
                -1.0812380225274251e-17,
                5.5954804541817275e-02,
                -4.9777324927023761e-02,
                -5.8964427754469556e-02,
                6.3502386192815230e-02,
                7.2099592331811152e-02,
                5.6733744823683624e-02,
                1.4944139198302276e-02,
                -1.6586331545035936e-02,
                1.6215647869348981e-02,
                -5.3010009948145205e-02,
                -3.3903483698637077e-02,
                1.6664896009982326e-02
            ],
            [
                -1.3578123829071156e-16,
                -3.8928231372374277e-18,
                -6.8690629504672547e-17,
                1.4739591804686035e-02,
                -5.4420843340158107e-02,
                -5.1448500624110230e-02,
                -4.0104936649664719e-02,
                -2.0005771109523323e-03,
                1.4944139198302273e-02,
                2.0876109462991529e-01,
                8.5299512972112107e-02,
                1.0514782966239777e-01,
                3.8558530291986026e-02,
                -5.6718358094553042e-02,
                1.7427469356077235e-01
            ],
            [
                -1.9330121907561594e-17,
                -1.3389366154873851e-18,
                -4.2668030345290713e-17,
                8.9484924078081958e-03,
                4.3724587719599466e-03,
                -5.4471769944168649e-02,
                -2.4623909141792265e-02,
                1.5358479886310851e-02,
                -1.6586331545035936e-02,
                8.5299512972112107e-02,
                1.2193906318955158e-01,
                -6.7674188532061239e-02,
                1.0527171333541688e-01,
                9.1872468798088788e-02,
                1.0703317492656196e-01
            ],
            [
                -1.0603743075865971e-16,
                -1.5154184034511704e-17,
                -7.2782323007986136e-18,
                -1.9091151483715266e-02,
                -4.5558751715687712e-02,
                4.2820412784553348e-02,
                -3.5770527652445649e-02,
                -5.2485191735174526e-02,
                1.6215647869348981e-02,
                1.0514782966239777e-01,
                -6.7674188532061239e-02,
                2.0145729362382303e-01,
                -8.1319215462917052e-02,
                -1.7449333720677199e-01,
                3.6876724967234820e-02
            ],
            [
                2.2353858650742440e-17,
                -1.8204252853156606e-17,
                -1.7884011479686417e-17,
                -3.1021851161818721e-02,
                4.3020335713749039e-02,
                -1.8846060001349585e-03,
                -5.7846718755825158e-02,
                -3.3614724769155974e-02,
                -5.3010009948145198e-02,
                3.8558530291986033e-02,
                1.0527171333541686e-01,
                -8.1319215462917052e-02,
                1.1995385725339547e-01,
                1.0754760330450190e-01,
                6.0324264975644963e-02
            ],
            [
                7.7584690146487811e-17,
                7.4174985572055250e-18,
                -8.5170423975487548e-18,
                -2.6075632893557410e-05,
                5.0562913783429822e-02,
                -3.0771368273686275e-02,
                1.4227033866598231e-04,
                2.2043298509534515e-02,
                -3.3903483698637077e-02,
                -5.6718358094553042e-02,
                9.1872468798088788e-02,
                -1.7449333720677199e-01,
                1.0754760330450192e-01,
                1.6649839432061139e-01,
                3.1837205803561530e-06
            ],
            [
                -8.8601549973232043e-17,
                -6.6899349338655394e-18,
                -5.4237163422969022e-17,
                2.9083886207596891e-02,
                -4.5759881428950828e-02,
                -7.5251487543960507e-02,
                -1.9696153127750006e-02,
                2.5770766426238427e-02,
                1.6664896009982326e-02,
                1.7427469356077238e-01,
                1.0703317492656197e-01,
                3.6876724967234820e-02,
                6.0324264975644949e-02,
                3.1837205803561530e-06,
                1.6202353812404047e-01
            ]
        ],
        [
            [
                2.8815388035945410e-04,
                9.7963009403356678e-03,
                -6.9580312774469438e-03,
                1.3366409271398685e-17,
                -8.6781417514160231e-18,
                1.6150831675803601e-17,
                -2.4373888113125300e-17,
                1.1402393882948064e-17,
                -2.3089272222851394e-17,
                -2.9409452253191486e-17,
                4.2764147680063988e-17,
                -3.0438248644025507e-17,
                -5.3639629755200135e-17,
                -4.5229170657265458e-17,
                3.2839594470842625e-17
            ],
            [
                9.7963009403356678e-03,
                3.3304258125404546e-01,
                -2.3655054315803850e-01,
                3.7790106372357909e-17,
                -4.3375094205204485e-17,
                5.9455966757826886e-17,
                1.6549577058993943e-17,
                -1.1002774343595919e-17,
                6.9664429243087846e-18,
                -5.0487653413478457e-18,
                2.4332811665557681e-17,
                -2.2054535729637351e-17,
                -4.3860012045904919e-17,
                -3.8392518806855063e-17,
                1.9933076450136562e-17
            ],
            [
                -6.9580312774469438e-03,
                -2.3655054315803850e-01,
                1.6667874652877718e-01,
                -2.2716296222329801e-17,
                3.6712957982183901e-17,
                -1.8809850378794706e-17,
                -3.1873671305095376e-17,
                2.4486716118976562e-17,
                -1.7783827840242799e-17,
                -4.3840194978390144e-17,
                4.0367277719515961e-17,
                -1.8802621353048196e-17,
                -5.4297193928729931e-17,
                -4.4383690064290809e-17,
                3.2649256895225490e-17
            ],
            [
                1.3366409271398686e-17,
                3.7790106372357909e-17,
                -2.2716296222329804e-17,
                6.2784867205581907e-02,
                -8.8693401708749148e-02,
                2.0526447070420640e-03,
                -6.8585090371772725e-02,
                5.3160153283753767e-03,
                -1.0251584788358778e-01,
                4.6897638900133490e-04,
                -1.7501568529108433e-02,
                -3.6325521728083815e-02,
                1.0528520413935760e-03,
                -2.6797719916591897e-06,
                -2.9118126512393393e-02
            ],
            [
                -8.6781417514160247e-18,
                -4.3375094205204485e-17,
                3.6712957982183901e-17,
                -8.8693401708749148e-02,
                1.3896443706355083e-01,
                -2.7598780806366453e-03,
                8.1361903811767139e-02,
                -6.9161864084714055e-03,
                1.3965203833165216e-01,
                -3.1524054755032479e-03,
                8.4787278004641170e-02,
                2.3857105064297078e-02,
                -2.3647315645219935e-03,
                -1.3757312749401460e-03,
                8.8055993047317871e-02
            ],
            [
                1.6150831675803601e-17,
                5.9455966757826886e-17,
                -1.8809850378794706e-17,
                2.0526447070420640e-03,
                -2.7598780806366453e-03,
                1.9660293311965449e-02,
                -1.8862747065752671e-03,
                1.4321693607930853e-02,
                -2.4947534728597567e-03,
                -3.7925134272927617e-02,
                -2.9837474226249586e-03,
                -2.6344882571909907e-03,
                -7.1647992303614766e-02,
                -5.9155179334568361e-02,
                -2.0403000745596598e-03
            ],
            [
                -2.4373888113125300e-17,
                1.6549577058993943e-17,
                -3.1873671305095382e-17,
                -6.8585090371772725e-02,
                8.1361903811767139e-02,
                -1.8862747065752671e-03,
                9.1071395868696420e-02,
                -6.0206958548374382e-03,
                1.1616442822704925e-01,
                1.2502312691157441e-03,
                -4.6604104906601829e-02,
                6.8240204611723171e-02,
                -1.9625764765177941e-03,
                1.4736253369960865e-05,
                -1.9726949363933464e-02
            ],
            [
                1.1402393882948064e-17,
                -1.1002774343595919e-17,
                2.4486716118976562e-17,
                5.3160153283753767e-03,
                -6.9161864084714055e-03,
                1.4321693607930853e-02,
                -6.0206958548374391e-03,
                1.0240773228395178e-02,
                -8.1621213900179707e-03,
                -2.6337489049742393e-02,
                -1.1388121639847627e-03,
                -4.9017263160878377e-03,
                -4.9401023693055435e-02,
                -4.0429576655207175e-02,
                -1.6068412053105463e-03
            ],
            [
                -2.3089272222851394e-17,
                6.9664429243087830e-18,
                -1.7783827840242796e-17,
                -1.0251584788358777e-01,
                1.3965203833165216e-01,
                -2.4947534728597567e-03,
                1.1616442822704925e-01,
                -8.1621213900179707e-03,
                1.6728989529979651e-01,
                -1.5848482721673093e-03,
                7.8626550179505192e-03,
                6.6909553344912095e-02,
                -4.5173357456152882e-03,
                -2.0997332791684014e-03,
                3.0643199165970497e-02
            ],
            [
                -2.9409452253191486e-17,
                -5.0487653413478457e-18,
                -4.3840194978390144e-17,
                4.6897638900133490e-04,
                -3.1524054755032479e-03,
                -3.7925134272927617e-02,
                1.2502312691157441e-03,
                -2.6337489049742393e-02,
                -1.5848482721673091e-03,
                7.1079163192150671e-02,
                -5.0292570403116365e-03,
                6.6587355873195609e-03,
                1.3314655282453663e-01,
                1.0889111215005008e-01,
                -5.4480436372396340e-03
            ],
            [
                4.2764147680063988e-17,
                2.4332811665557681e-17,
                4.0367277719515961e-17,
                -1.7501568529108433e-02,
                8.4787278004641170e-02,
                -2.9837474226249586e-03,
                -4.6604104906601829e-02,
                -1.1388121639847627e-03,
                7.8626550179505192e-03,
                -5.0292570403116365e-03,
                2.5953993639722683e-01,
                -1.0430541224725150e-01,
                6.5927413740306745e-03,
                2.9180590898176010e-03,
                2.0400262992539608e-01
            ],
            [
                -3.0438248644025507e-17,
                -2.2054535729637351e-17,
                -1.8802621353048196e-17,
                -3.6325521728083815e-02,
                2.3857105064297071e-02,
                -2.6344882571909907e-03,
                6.8240204611723171e-02,
                -4.9017263160878377e-03,
                6.6909553344912095e-02,
                6.6587355873195609e-03,
                -1.0430541224725150e-01,
                7.1256072809497847e-02,
                5.2152407559574732e-03,
                5.9394698228325826e-03,
                -7.1975820334816448e-02
            ],
            [
                -5.3639629755200142e-17,
                -4.3860012045904919e-17,
                -5.4297193928729931e-17,
                1.0528520413935756e-03,
                -2.3647315645219970e-03,
                -7.1647992303614766e-02,
                -1.9625764765177950e-03,
                -4.9401023693055435e-02,
                -4.5173357456152873e-03,
                1.3314655282453663e-01,
                6.5927413740306745e-03,
                5.2152407559574740e-03,
                2.5031125606432492e-01,
                2.0431308346523294e-01,
                2.0828944793567098e-03
            ],
            [
                -4.5229170657265458e-17,
                -3.8392518806855063e-17,
                -4.4383690064290809e-17,
                -2.6797719916591897e-06,
                -1.3757312749401460e-03,
                -5.9155179334568368e-02,
                1.4736253369960648e-05,
                -4.0429576655207175e-02,
                -2.0997332791684014e-03,
                1.0889111215005008e-01,
                2.9180590898176010e-03,
                5.9394698228325826e-03,
                2.0431308346523294e-01,
                1.6636130033916013e-01,
                2.6408211771522527e-07
            ],
            [
                3.2839594470842625e-17,
                1.9933076450136565e-17,
                3.2649256895225490e-17,
                -2.9118126512393393e-02,
                8.8055993047317871e-02,
                -2.0403000745596589e-03,
                -1.9726949363933460e-02,
                -1.6068412053105463e-03,
                3.0643199165970497e-02,
                -5.4480436372396340e-03,
                2.0400262992539608e-01,
                -7.1975820334816448e-02,
                2.0828944793567115e-03,
                2.6408211771522527e-07,
                1.6207963067601569e-01
            ]
        ],
        [
            [
                2.6275440971166617e-33,
                2.5037166971616607e-33,
                2.2516575810950620e-33,
                -1.6942570333169244e-17,
                -2.2121920673440847e-17,
                5.1565418344628850e-19,
                -1.4791036920821054e-17,
                -1.1118506643272432e-18,
                2.1288199828695582e-17,
                -7.1736649168466734e-20,
                2.6985010002801872e-18,
                -6.2395214702276523e-18,
                1.8100987723134096e-19,
                2.1769000834890853e-23,
                -4.2156772660919164e-18
            ],
            [
                2.5037166971616607e-33,
                1.0112976228194033e-33,
                1.0637375187588277e-33,
                -1.3417012428968998e-17,
                -1.7093189587628252e-17,
                3.9843847289234750e-19,
                -1.0836380146355983e-17,
                -8.0681746783739300e-19,
                1.5447603848452611e-17,
                -4.1667917432489777e-20,
                1.5689830837939726e-18,
                -4.1983696582280812e-18,
                1.2180581572751782e-19,
                1.0962071777794100e-23,
                -2.3316114725992698e-18
            ],
            [
                2.2516575810950620e-33,
                1.0637375187588277e-33,
                1.0780574584211125e-33,
                -1.2372330672242660e-17,
                -1.5819719204926734e-17,
                3.6875384948539488e-19,
                -1.0111037907722917e-17,
                -7.5394508104580176e-19,
                1.4435325847808916e-17,
                -4.0468225251457530e-20,
                1.5235201456634888e-18,
                -3.9717796498408032e-18,
                1.1523021918606916e-19,
                1.0956196127024681e-23,
                -2.2860298671856400e-18
            ],
            [
                -1.6942570333169244e-17,
                -1.3417012428968998e-17,
                -1.2372330672242660e-17,
                1.0383568483443890e-01,
                1.3473420653630030e-01,
                -3.1406113708514010e-03,
                8.8909866949382493e-02,
                6.6680092680211286e-03,
                -1.2766947640671111e-01,
                4.0960860637700189e-04,
                -1.5411268977482536e-02,
                3.6766259965530755e-02,
                -1.0666174449197150e-03,
                -1.2096057172185921e-07,
                2.3838826409410416e-02
            ],
            [
                -2.2121920673440847e-17,
                -1.7093189587628252e-17,
                -1.5819719204926734e-17,
                1.3473420653630030e-01,
                1.7468870915881007e-01,
                -4.0719388148053055e-03,
                1.1508136350855001e-01,
                8.6282273205593799e-03,
                -1.6520084112146471e-01,
                5.2656571688893873e-04,
                -1.9812256482087960e-02,
                4.7464961358241416e-02,
                -1.3769984447296313e-03,
                -1.5491086977805776e-07,
                3.0604706322810475e-02
            ],
            [
                5.1565418344628850e-19,
                3.9843847289234750e-19,
                3.6875384948539488e-19,
                -3.1406113708514010e-03,
                -4.0719388148053055e-03,
                9.4915611837076658e-05,
                -2.6825113830502767e-03,
                -2.0112135338581070e-04,
                3.8507813385099399e-03,
                -1.2274105008081069e-05,
                4.6181835839662847e-04,
                -1.1063943888663390e-03,
                3.2097431626553971e-05,
                3.6109337888195403e-09,
                -7.1338767860006948e-04
            ],
            [
                -1.4791036920821054e-17,
                -1.0836380146355983e-17,
                -1.0111037907722917e-17,
                8.8909866949382493e-02,
                1.1508136350855001e-01,
                -2.6825113830502767e-03,
                7.5541049956942580e-02,
                5.6600727585662571e-03,
                -1.0837079180721015e-01,
                3.4056701110776310e-04,
                -1.2814753423645731e-02,
                3.0982764524776545e-02,
                -8.9884091803538700e-04,
                -9.9360058028675966e-08,
                1.9736349938753776e-02
            ],
            [
                -1.1118506643272432e-18,
                -8.0681746783739300e-19,
                -7.5394508104580176e-19,
                6.6680092680211286e-03,
                8.6282273205593799e-03,
                -2.0112135338581070e-04,
                5.6600727585662571e-03,
                4.2404462564741359e-04,
                -8.1189845871023035e-03,
                2.5449947547166920e-05,
                -9.5763351279755946e-04,
                2.3191286843570946e-03,
                -6.7280302765164142e-05,
                -7.4137328454414551e-09,
                1.4740776795818976e-03
            ],
            [
                2.1288199828695582e-17,
                1.5447603848452611e-17,
                1.4435325847808916e-17,
                -1.2766947640671111e-01,
                -1.6520084112146471e-01,
                3.8507813385099399e-03,
                -1.0837079180721015e-01,
                -8.1189845871023035e-03,
                1.5545040950779296e-01,
                -4.8727628801181587e-04,
                1.8335287753679892e-02,
                -4.4403211947157906e-02,
                1.2881827422225149e-03,
                1.4194636077332907e-07,
                -2.8223338669636173e-02
            ],
            [
                -7.1736649168466734e-20,
                -4.1667917432489777e-20,
                -4.0468225251457530e-20,
                4.0960860637700189e-04,
                5.2656571688893873e-04,
                -1.2274105008081069e-05,
                3.4056701110776310e-04,
                2.5449947547166920e-05,
                -4.8727628801181587e-04,
                1.4403224570745436e-06,
                -5.4210775867009898e-05,
                1.3642559150744743e-04,
                -3.9579341380388600e-06,
                -4.0440892184457906e-10,
                8.2369266042777269e-05
            ],
            [
                2.6985010002801872e-18,
                1.5689830837939726e-18,
                1.5235201456634888e-18,
                -1.5411268977482536e-02,
                -1.9812256482087960e-02,
                4.6181835839662847e-04,
                -1.2814753423645731e-02,
                -9.5763351279755946e-04,
                1.8335287753679892e-02,
                -5.4210775867009898e-05,
                2.0403797013479612e-03,
                -5.1338865516670027e-03,
                1.4894260405288242e-04,
                1.5223734554406932e-08,
                -3.1003916954206430e-03
            ],
            [
                -6.2395214702276523e-18,
                -4.1983696582280812e-18,
                -3.9717796498408032e-18,
                3.6766259965530755e-02,
                4.7464961358241416e-02,
                -1.1063943888663390e-03,
                3.0982764524776545e-02,
                2.3191286843570946e-03,
                -4.4403211947157906e-02,
                1.3642559150744743e-04,
                -5.1338865516670027e-03,
                1.2595904217467684e-02,
                -3.6542288782159295e-04,
                -3.9260795831770255e-08,
                7.8684055928711569e-03
            ],
            [
                1.8100987723134096e-19,
                1.2180581572751782e-19,
                1.1523021918606916e-19,
                -1.0666174449197150e-03,
                -1.3769984447296313e-03,
                3.2097431626553971e-05,
                -8.9884091803538700e-04,
                -6.7280302765164142e-05,
                1.2881827422225149e-03,
                -3.9579341380388600e-06,
                1.4894260405288242e-04,
                -3.6542288782159295e-04,
                1.0601373551071568e-05,
                1.1390368088605987e-09,
                -2.2827666492428951e-04
            ],
            [
                2.1769000834890853e-23,
                1.0962071777794100e-23,
                1.0956196127024681e-23,
                -1.2096057172185921e-07,
                -1.5491086977805776e-07,
                3.6109337888195403e-09,
                -9.9360058028675966e-08,
                -7.4137328454414551e-09,
                1.4194636077332907e-07,
                -4.0440892184457906e-10,
                1.5223734554406932e-08,
                -3.9260795831770255e-08,
                1.1390368088605987e-09,
                1.1074839868616413e-13,
                -2.2932550453204621e-08
            ],
            [
                -4.2156772660919164e-18,
                -2.3316114725992698e-18,
                -2.2860298671856400e-18,
                2.3838826409410416e-02,
                3.0604706322810475e-02,
                -7.1338767860006948e-04,
                1.9736349938753776e-02,
                1.4740776795818976e-03,
                -2.8223338669636173e-02,
                8.2369266042777269e-05,
                -3.1003916954206430e-03,
                7.8684055928711569e-03,
                -2.2827666492428951e-04,
                -2.2932550453204621e-08,
                4.6969856934081797e-03
            ]
        ],
        [
            [
                -1.1880051181097467e-33,
                -7.9551559592662319e-34,
                -3.7956167211573413e-34,
                6.4942864390357590e-18,
                4.6266132146714596e-18,
                7.5962560171149768e-18,
                -6.5177850065283884e-18,
                8.0275370519065918e-18,
                5.1928406467655393e-18,
                1.3697147229353144e-18,
                8.4482648748965586e-19,
                1.6308363921486223e-18,
                2.6516037444058895e-18,
                -2.8630671207975711e-21,
                -2.6015070380907871e-18
            ],
            [
                -7.9551559592662319e-34,
                1.8128889373915109e-33,
                4.1712277641120716e-33,
                -1.3733692341252731e-17,
                -9.2767590752854580e-18,
                -1.5231249308461191e-17,
                1.1776612476053879e-17,
                -1.4255252935843029e-17,
                -9.2213551611887420e-18,
                -1.7798279695964808e-18,
                -1.0989630456087438e-18,
                -2.5605892963165894e-18,
                -4.1620680436182940e-18,
                4.1387000700431325e-21,
                3.1943989945493427e-18
            ],
            [
                -3.7956167211573413e-34,
                4.1712277641120716e-33,
                8.2280732250012886e-33,
                -3.2041023124528483e-17,
                -2.1869319152587701e-17,
                -3.5906571797891338e-17,
                2.8370799709372448e-17,
                -3.4472245016541567e-17,
                -2.2299237710405568e-17,
                -4.6508248296671859e-18,
                -2.8708142464152733e-18,
                -6.3703459913843607e-18,
                -1.0355318010725861e-17,
                1.0510825125995450e-20,
                8.4823413357667667e-18
            ],
            [
                6.4942864390357590e-18,
                -1.3733692341252731e-17,
                -3.2041023124528483e-17,
                1.0389837339666627e-01,
                7.0107695901658940e-02,
                1.1510786544886310e-01,
                -8.8803838607628374e-02,
                1.0745252790517464e-01,
                6.9508256299295954e-02,
                1.3304071342049809e-02,
                8.2149376521255587e-03,
                1.9243593166284836e-02,
                3.1278944535958891e-02,
                -3.1034441783910251e-05,
                -2.3834304183152511e-02
            ],
            [
                4.6266132146714596e-18,
                -9.2767590752854580e-18,
                -2.1869319152587701e-17,
                7.0107695901658940e-02,
                4.7269208575134440e-02,
                7.7609999547636033e-02,
                -5.9774026680801909e-02,
                7.2304865475409041e-02,
                4.6772138764578877e-02,
                8.8946834158914988e-03,
                5.4923927572280568e-03,
                1.2919396106706324e-02,
                2.0999336615407038e-02,
                -2.0799578474276861e-05,
                -1.5912220548241266e-02
            ],
            [
                7.5962560171149768e-18,
                -1.5231249308461191e-17,
                -3.5906571797891338e-17,
                1.1510786544886310e-01,
                7.7609999547636033e-02,
                1.2742570081606683e-01,
                -9.8141333765768807e-02,
                1.1871537855338304e-01,
                7.6793893782397385e-02,
                1.4603949140075401e-02,
                9.0178166512089059e-03,
                2.1212008849201720e-02,
                3.4478245791744036e-02,
                -3.4150275718588081e-05,
                -2.6125865419765706e-02
            ],
            [
                -6.5177850065283884e-18,
                1.1776612476053879e-17,
                2.8370799709372448e-17,
                -8.8803838607628374e-02,
                -5.9774026680801909e-02,
                -9.8141333765768807e-02,
                7.5315610879063244e-02,
                -9.1046233516763497e-02,
                -5.8895428350729431e-02,
                -1.1044764653996040e-02,
                -6.8204407162652941e-03,
                -1.6188198434867421e-02,
                -2.6312151130045910e-02,
                2.5965609215123145e-05,
                1.9697166490528297e-02
            ],
            [
                8.0275370519065918e-18,
                -1.4255252935843029e-17,
                -3.4472245016541567e-17,
                1.0745252790517464e-01,
                7.2304865475409041e-02,
                1.1871537855338304e-01,
                -9.1046233516763497e-02,
                1.1004982200619320e-01,
                7.1188349840683618e-02,
                1.3316539066659619e-02,
                8.2234083347991292e-03,
                1.9549831509162622e-02,
                3.1776046478249553e-02,
                -3.1336669537813862e-05,
                -2.3735184941770664e-02
            ],
            [
                5.1928406467655393e-18,
                -9.2213551611887420e-18,
                -2.2299237710405568e-17,
                6.9508256299295954e-02,
                4.6772138764578877e-02,
                7.6793893782397385e-02,
                -5.8895428350729431e-02,
                7.1188349840683618e-02,
                4.6049880504753338e-02,
                8.6141123146954539e-03,
                5.3195025324443944e-03,
                1.2646270108959801e-02,
                2.0555085920733674e-02,
                -2.0270859395072423e-05,
                -1.5353651828973242e-02
            ],
            [
                1.3697147229353144e-18,
                -1.7798279695964808e-18,
                -4.6508248296671859e-18,
                1.3304071342049809e-02,
                8.8946834158914988e-03,
                1.4603949140075401e-02,
                -1.1044764653996040e-02,
                1.3316539066659619e-02,
                8.6141123146954539e-03,
                1.5218942574595177e-03,
                9.4004929658358775e-04,
                2.3196260262938091e-03,
                3.7700966681006075e-03,
                -3.6622366605767022e-06,
                -2.6766300998852582e-03
            ],
            [
                8.4482648748965586e-19,
                -1.0989630456087438e-18,
                -2.8708142464152733e-18,
                8.2149376521255587e-03,
                5.4923927572280568e-03,
                9.0178166512089059e-03,
                -6.8204407162652941e-03,
                8.2234083347991292e-03,
                5.3195025324443944e-03,
                9.4004929658358775e-04,
                5.8065252156097049e-04,
                1.4325642283776910e-03,
                2.3283523874813514e-03,
                -2.2618850212542531e-06,
                -1.6534081636199304e-03
            ],
            [
                1.6308363921486223e-18,
                -2.5605892963165894e-18,
                -6.3703459913843607e-18,
                1.9243593166284836e-02,
                1.2919396106706324e-02,
                2.1212008849201720e-02,
                -1.6188198434867421e-02,
                1.9549831509162622e-02,
                1.2646270108959801e-02,
                2.3196260262938091e-03,
                1.4325642283776910e-03,
                3.4492904576506203e-03,
                5.6063330052181139e-03,
                -5.5001628840239396e-06,
                -4.1159727023025250e-03
            ],
            [
                2.6516037444058895e-18,
                -4.1620680436182940e-18,
                -1.0355318010725861e-17,
                3.1278944535958891e-02,
                2.0999336615407038e-02,
                3.4478245791744036e-02,
                -2.6312151130045910e-02,
                3.1776046478249553e-02,
                2.0555085920733674e-02,
                3.7700966681006075e-03,
                2.3283523874813514e-03,
                5.6063330052181139e-03,
                9.1122996722331623e-03,
                -8.9396136084625945e-06,
                -6.6896260121991451e-03
            ],
            [
                -2.8630671207975711e-21,
                4.1387000700431325e-21,
                1.0510825125995450e-20,
                -3.1034441783910251e-05,
                -2.0799578474276861e-05,
                -3.4150275718588081e-05,
                2.5965609215123145e-05,
                -3.1336669537813862e-05,
                -2.0270859395072423e-05,
                -3.6622366605767022e-06,
                -2.2618850212542531e-06,
                -5.5001628840239396e-06,
                -8.9396136084625945e-06,
                8.7352470407223095e-09,
                6.4753893260441833e-06
            ],
            [
                -2.6015070380907871e-18,
                3.1943989945493427e-18,
                8.4823413357667667e-18,
                -2.3834304183152511e-02,
                -1.5912220548241266e-02,
                -2.6125865419765706e-02,
                1.9697166490528297e-02,
                -2.3735184941770664e-02,
                -1.5353651828973242e-02,
                -2.6766300998852582e-03,
                -1.6534081636199304e-03,
                -4.1159727023025250e-03,
                -6.6896260121991451e-03,
                6.4753893260441833e-06,
                4.6922093844226735e-03
            ]
        ],
        [
            [
                3.1147522973331956e-32,
                -9.1399271693726002e-33,
                2.6877385609372704e-33,
                6.0102101243105587e-17,
                3.7764805310637820e-17,
                -6.9042533708986822e-17,
                -5.2798353772430721e-17,
                -6.8146821005769737e-17,
                3.4615691084926124e-17,
                -8.7805834443153091e-18,
                4.7894429867214625e-18,
                1.0647992594281528e-17,
                -1.9802985694430395e-17,
                2.0777553871958404e-20,
                -1.5470063693232878e-17
            ],
            [
                -9.1399271693726002e-33,
                1.9213877513978013e-33,
                -7.2770943173888322e-35,
                -2.0672133204175870e-17,
                -1.3136655487134097e-17,
                2.4016709987889759e-17,
                1.8791635648029608e-17,
                2.4349592638366613e-17,
                -1.2368589984919780e-17,
                3.3829877799339898e-18,
                -1.8447901832099514e-18,
                -3.9152209578442712e-18,
                7.2820855620388774e-18,
                -7.8037574596352666e-21,
                6.0477319738785193e-18
            ],
            [
                2.6877385609372704e-33,
                -7.2770943173888322e-35,
                -4.4190770929014888e-34,
                8.0435678469372149e-18,
                5.1929028129419106e-18,
                -9.4937510297563560e-18,
                -7.6605933411960274e-18,
                -9.9771611156250008e-18,
                5.0680033034013534e-18,
                -1.5166915152473702e-18,
                8.2683195122156774e-19,
                1.6630173495543210e-18,
                -3.0934331739453928e-18,
                3.3993688407965719e-21,
                -2.7544606730683603e-18
            ],
            [
                6.0102101243105587e-17,
                -2.0672133204175870e-17,
                8.0435678469372149e-18,
                1.0385654667842115e-01,
                6.4669201052501166e-02,
                -1.1822997238428093e-01,
                -8.8714750084860239e-02,
                -1.1412401507212361e-01,
                5.7970029527566275e-02,
                -1.3724470770881729e-02,
                7.4880791711873316e-03,
                1.7390631210673073e-02,
                -3.2340402372677456e-02,
                3.3280237422798571e-05,
                -2.3831552410540986e-02
            ],
            [
                3.7764805310637820e-17,
                -1.3136655487134097e-17,
                5.1929028129419106e-18,
                6.4669201052501166e-02,
                4.0236183479644042e-02,
                -7.3560881897011682e-02,
                -5.5104016900031291e-02,
                -7.0865452996600825e-02,
                3.5996556995687420e-02,
                -8.4673759173536107e-03,
                4.6199221818124178e-03,
                1.0774037376745812e-02,
                -2.0035745559849147e-02,
                2.0580598358472311e-05,
                -1.4682068007889666e-02
            ],
            [
                -6.9042533708986822e-17,
                2.4016709987889759e-17,
                -9.4937510297563560e-18,
                -1.1822997238428093e-01,
                -7.3560881897011682e-02,
                1.3448599935168587e-01,
                1.0074268520255542e-01,
                1.2955818404321326e-01,
                -6.5809902556267619e-02,
                1.5480307160024517e-02,
                -8.4462783792247365e-03,
                -1.9697400750152950e-02,
                3.6629918410511789e-02,
                -3.7626044669774453e-05,
                2.6842197233268295e-02
            ],
            [
                -5.2798353772430721e-17,
                1.8791635648029608e-17,
                -7.6605933411960274e-18,
                -8.8714750084860239e-02,
                -5.5104016900031291e-02,
                1.0074268520255542e-01,
                7.5194862669454118e-02,
                9.6641007429257020e-02,
                -4.9089393395484598e-02,
                1.1386995445421365e-02,
                -6.2132524883314732e-03,
                -1.4620705380316200e-02,
                2.7188721810993299e-02,
                -2.7818625200697437e-05,
                1.9683080338811602e-02
            ],
            [
                -6.8146821005769737e-17,
                2.4349592638366613e-17,
                -9.9771611156250008e-18,
                -1.1412401507212361e-01,
                -7.0865452996600825e-02,
                1.2955818404321326e-01,
                9.6641007429257020e-02,
                1.2418956351536439e-01,
                -6.3082847056021241e-02,
                1.4596226334892437e-02,
                -7.9644319817258995e-03,
                -1.8771945481816613e-02,
                3.4908289628651938e-02,
                -3.5691796031752553e-05,
                2.5216115314836203e-02
            ],
            [
                3.4615691084926124e-17,
                -1.2368589984919780e-17,
                5.0680033034013534e-18,
                5.7970029527566275e-02,
                3.5996556995687420e-02,
                -6.5809902556267619e-02,
                -4.9089393395484598e-02,
                -6.3082847056021241e-02,
                3.2043317327650880e-02,
                -7.4142302797954758e-03,
                4.0455753280307719e-03,
                9.5353189174627387e-03,
                -1.7731868772636050e-02,
                1.8129847153561475e-05,
                -1.2808654307207738e-02
            ],
            [
                -8.7805834443153091e-18,
                3.3829877799339898e-18,
                -1.5166915152473702e-18,
                -1.3724470770881729e-02,
                -8.4673759173536107e-03,
                1.5480307160024517e-02,
                1.1386995445421365e-02,
                1.4596226334892437e-02,
                -7.4142302797954758e-03,
                1.6203221805494996e-03,
                -8.8433668442267457e-04,
                -2.1634372001822911e-03,
                4.0228798134450995e-03,
                -4.0477325979662467e-06,
                2.7620839220173473e-03
            ],
            [
                4.7894429867214625e-18,
                -1.8447901832099514e-18,
                8.2683195122156774e-19,
                7.4880791711873316e-03,
                4.6199221818124178e-03,
                -8.4462783792247365e-03,
                -6.2132524883314732e-03,
                -7.9644319817258995e-03,
                4.0455753280307719e-03,
                -8.8433668442267457e-04,
                4.8265129177368552e-04,
                1.1805732166540800e-03,
                -2.1952591663576822e-03,
                2.2089668248916456e-06,
                -1.5075713042478492e-03
            ],
            [
                1.0647992594281528e-17,
                -3.9152209578442712e-18,
                1.6630173495543210e-18,
                1.7390631210673073e-02,
                1.0774037376745812e-02,
                -1.9697400750152950e-02,
                -1.4620705380316200e-02,
                -1.8771945481816613e-02,
                9.5353189174627387e-03,
                -2.1634372001822911e-03,
                1.1805732166540800e-03,
                2.8181846202681550e-03,
                -5.2405824713714962e-03,
                5.3287412924859985e-06,
                -3.7207795477229499e-03
            ],
            [
                -1.9802985694430395e-17,
                7.2820855620388774e-18,
                -3.0934331739453928e-18,
                -3.2340402372677456e-02,
                -2.0035745559849147e-02,
                3.6629918410511789e-02,
                2.7188721810993299e-02,
                3.4908289628651938e-02,
                -1.7731868772636050e-02,
                4.0228798134450995e-03,
                -2.1952591663576822e-03,
                -5.2405824713714962e-03,
                9.7451751892796961e-03,
                -9.9089417289777283e-06,
                6.9186373505031838e-03
            ],
            [
                2.0777553871958404e-20,
                -7.8037574596352666e-21,
                3.3993688407965719e-21,
                3.3280237422798571e-05,
                2.0580598358472311e-05,
                -3.7626044669774453e-05,
                -2.7818625200697437e-05,
                -3.5691796031752553e-05,
                1.8129847153561475e-05,
                -4.0477325979662467e-06,
                2.2089668248916456e-06,
                5.3287412924859985e-06,
                -9.9089417289777283e-06,
                1.0030167338271124e-08,
                -6.9353363394701153e-06
            ],
            [
                -1.5470063693232878e-17,
                6.0477319738785193e-18,
                -2.7544606730683603e-18,
                -2.3831552410540986e-02,
                -1.4682068007889666e-02,
                2.6842197233268295e-02,
                1.9683080338811602e-02,
                2.5216115314836203e-02,
                -1.2808654307207738e-02,
                2.7620839220173473e-03,
                -1.5075713042478492e-03,
                -3.7207795477229499e-03,
                6.9186373505031838e-03,
                -6.9353363394701153e-06,
                4.6930414701879340e-03
            ]
        ],
        [
            [
                2.4900593911989887e-35,
                -1.3836975073402895e-34,
                -6.4950024471561617e-35,
                1.6191290533149288e-18,
                -1.0873623956446152e-18,
                -1.7871990779959146e-18,
                1.3630619586138524e-18,
                1.6568535919912299e-18,
                1.0783838086139216e-18,
                1.9796314360794567e-19,
                1.2070265365614712e-19,
                -2.9514269899181928e-19,
                -4.7741443661349930e-19,
                5.0339255969607621e-22,
                3.5573725862383163e-19
            ],
            [
                -1.3836975073402895e-34,
                4.2541861076171813e-34,
                2.6964541937239585e-34,
                -8.3311212968796520e-18,
                5.7075732402887293e-18,
                9.3810622693092246e-18,
                -7.4574317352833788e-18,
                -9.1305742784909508e-18,
                -5.9427435401582574e-18,
                -1.2657197313711488e-18,
                -7.7217900992922617e-19,
                1.7176407281599238e-18,
                2.7780530502018073e-18,
                -3.0659924414110319e-21,
                -2.3433070202666441e-18
            ],
            [
                -6.4950024471561617e-35,
                2.6964541937239585e-34,
                1.4515954778490887e-34,
                -4.0462644168689821e-18,
                2.7472865376216375e-18,
                4.5154769628933281e-18,
                -3.5242969104954000e-18,
                -4.3013971845148646e-18,
                -2.7996167626236396e-18,
                -5.6038268844942279e-19,
                -3.4179525670253915e-19,
                7.9045381517063724e-19,
                1.2785216118949487e-18,
                -1.3844361901816302e-21,
                -1.0252893419524298e-18
            ],
            [
                1.6191290533149288e-18,
                -8.3311212968796520e-18,
                -4.0462644168689821e-18,
                1.0398969544682068e-01,
                -7.0055041201783019e-02,
                -1.1514320548695742e-01,
                8.8404528074685573e-02,
                1.0758664820029523e-01,
                7.0024102000840999e-02,
                1.3193599322661721e-02,
                8.0452939364222074e-03,
                -1.9341714650934788e-02,
                -3.1285921488032692e-02,
                3.3253593468433361e-05,
                2.3842216595625288e-02
            ],
            [
                -1.0873623956446152e-18,
                5.7075732402887293e-18,
                2.7472865376216375e-18,
                -7.0055041201783019e-02,
                4.7156798511257189e-02,
                7.7507398393077270e-02,
                -5.9408386410861436e-02,
                -7.2277264943445746e-02,
                -4.7042554229292641e-02,
                -8.8061403659297193e-03,
                -5.3697346501071679e-03,
                1.2963929400845054e-02,
                2.0969738756175062e-02,
                -2.2244095410446802e-05,
                -1.5891601824335445e-02
            ],
            [
                -1.7871990779959146e-18,
                9.3810622693092246e-18,
                4.5154769628933281e-18,
                -1.1514320548695742e-01,
                7.7507398393077270e-02,
                1.2739195609293538e-01,
                -9.7644197616714862e-02,
                -1.1879560275889942e-01,
                -7.7319591290551881e-02,
                -1.4473835440959780e-02,
                -8.8257343173694172e-03,
                2.1307627216720124e-02,
                3.4466045207345225e-02,
                -3.6560572622674770e-05,
                -2.6119542605319283e-02
            ],
            [
                1.3630619586138524e-18,
                -7.4574317352833788e-18,
                -3.5242969104954000e-18,
                8.8404528074685573e-02,
                -5.9408386410861436e-02,
                -9.7644197616714862e-02,
                7.4574382932441299e-02,
                9.0670392650039067e-02,
                5.9013953483180580e-02,
                1.0892932951982298e-02,
                6.6418202750863240e-03,
                -1.6182561468555151e-02,
                -2.6176330790916209e-02,
                2.7647262526394525e-05,
                1.9597895290863959e-02
            ],
            [
                1.6568535919912299e-18,
                -9.1305742784909508e-18,
                -4.3013971845148646e-18,
                1.0758664820029523e-01,
                -7.2277264943445746e-02,
                -1.1879560275889942e-01,
                9.0670392650039067e-02,
                1.1022793301642166e-01,
                7.1743222896910222e-02,
                1.3209047242187468e-02,
                8.0539560322638818e-03,
                -1.9655663085390292e-02,
                -3.1794362986121975e-02,
                3.3554838323488032e-05,
                2.3751793030151992e-02
            ],
            [
                1.0783838086139216e-18,
                -5.9427435401582574e-18,
                -2.7996167626236396e-18,
                7.0024102000840999e-02,
                -4.7042554229292641e-02,
                -7.7319591290551881e-02,
                5.9013953483180580e-02,
                7.1743222896910222e-02,
                4.6694970056840181e-02,
                8.5972751918213797e-03,
                5.2420189888796910e-03,
                -1.2793134294706080e-02,
                -2.0693759026987932e-02,
                2.1839588228234133e-05,
                1.5459155316654821e-02
            ],
            [
                1.9796314360794567e-19,
                -1.2657197313711488e-18,
                -5.6038268844942279e-19,
                1.3193599322661721e-02,
                -8.8061403659297193e-03,
                -1.4473835440959780e-02,
                1.0892932951982298e-02,
                1.3209047242187468e-02,
                8.5972751918213797e-03,
                1.4939368749236389e-03,
                9.1067488591622318e-04,
                -2.3090151844735751e-03,
                -3.7351679076657795e-03,
                3.8724489196383372e-06,
                2.6513919054287800e-03
            ],
            [
                1.2070265365614712e-19,
                -7.7217900992922617e-19,
                -3.4179525670253915e-19,
                8.0452939364222074e-03,
                -5.3697346501071679e-03,
                -8.8257343173694172e-03,
                6.6418202750863240e-03,
                8.0539560322638818e-03,
                5.2420189888796910e-03,
                9.1067488591622318e-04,
                5.5512911665511690e-04,
                -1.4077598043409777e-03,
                -2.2772567037796631e-03,
                2.3607758318024850e-06,
                1.6161439402520468e-03
            ],
            [
                -2.9514269899181928e-19,
                1.7176407281599238e-18,
                7.9045381517063724e-19,
                -1.9341714650934788e-02,
                1.2963929400845054e-02,
                2.1307627216720124e-02,
                -1.6182561468555151e-02,
                -1.9655663085390292e-02,
                -1.2793134294706080e-02,
                -2.3090151844735751e-03,
                -1.4077598043409777e-03,
                3.4807623414085070e-03,
                5.6304620607030616e-03,
                -5.9059526738048927e-06,
                -4.1337289161564230e-03
            ],
            [
                -4.7741443661349930e-19,
                2.7780530502018073e-18,
                1.2785216118949487e-18,
                -3.1285921488032692e-02,
                2.0969738756175062e-02,
                3.4466045207345225e-02,
                -2.6176330790916209e-02,
                -3.1794362986121975e-02,
                -2.0693759026987932e-02,
                -3.7351679076657795e-03,
                -2.2772567037796631e-03,
                5.6304620607030616e-03,
                9.1078041627304165e-03,
                -9.5535778144522890e-06,
                -6.6869798024056185e-03
            ],
            [
                5.0339255969607621e-22,
                -3.0659924414110319e-21,
                -1.3844361901816302e-21,
                3.3253593468433361e-05,
                -2.2244095410446802e-05,
                -3.6560572622674770e-05,
                2.7647262526394525e-05,
                3.3554838323488032e-05,
                2.1839588228234133e-05,
                3.8724489196383372e-06,
                2.3607758318024850e-06,
                -5.9059526738048927e-06,
                -9.5535778144522890e-06,
                9.9664345426653185e-09,
                6.9049023722258362e-06
            ],
            [
                3.5573725862383163e-19,
                -2.3433070202666441e-18,
                -1.0252893419524298e-18,
                2.3842216595625288e-02,
                -1.5891601824335445e-02,
                -2.6119542605319283e-02,
                1.9597895290863959e-02,
                2.3751793030151992e-02,
                1.5459155316654821e-02,
                2.6513919054287800e-03,
                1.6161439402520468e-03,
                -4.1337289161564230e-03,
                -6.6869798024056185e-03,
                6.9049023722258362e-06,
                4.6910784657474471e-03
            ]
        ],
        [
            [
                2.1269408446467923e-34,
                -1.6253502365894958e-33,
                -1.7756820014729020e-33,
                6.1097282544082607e-18,
                -7.7701346402602710e-18,
                1.7973353394243797e-19,
                -4.8941005028919931e-18,
                3.6611680642451311e-19,
                -7.0616564127352395e-18,
                1.9644079149844693e-20,
                -7.2856489090013290e-19,
                -1.9089707957827610e-18,
                5.5513874790709594e-20,
                1.6391255866988714e-23,
                -1.0632179565817721e-18
            ],
            [
                -1.6253502365894958e-33,
                4.1037581827307522e-33,
                5.7413883527255425e-33,
                -2.2310864912014070e-17,
                2.9199748459702164e-17,
                -6.7543450570403855e-19,
                1.9561306949366043e-17,
                -1.4796321164534017e-18,
                2.8538685078587920e-17,
                -1.0156629182594278e-19,
                3.7719068378718873e-18,
                8.4064398812425566e-18,
                -2.4449924432051137e-19,
                -8.3870612308112678e-23,
                5.8368873534248483e-18
            ],
            [
                -1.7756820014729020e-33,
                5.7413883527255425e-33,
                7.4565402079646860e-33,
                -2.8062086051520785e-17,
                3.6465420722823106e-17,
                -8.4349882907630346e-19,
                2.4068923977249360e-17,
                -1.8158792245927245e-18,
                3.5024241030881503e-17,
                -1.1830423962839903e-19,
                4.3923951135954572e-18,
                1.0119023251342569e-17,
                -2.9429997987773429e-19,
                -9.7888535779815032e-23,
                6.7228466595759221e-18
            ],
            [
                6.1097282544082607e-18,
                -2.2310864912014070e-17,
                -2.8062086051520785e-17,
                1.0404763348543397e-01,
                -1.3474403134815904e-01,
                3.1168246494311653e-03,
                -8.8298096530807751e-02,
                6.6531461850346148e-03,
                -1.2832450887602964e-01,
                4.2197929093243500e-04,
                -1.5665104886420242e-02,
                -3.6717074830875517e-02,
                1.0678560554101626e-03,
                3.4953239652916074e-07,
                -2.3834987621220535e-02
            ],
            [
                -7.7701346402602710e-18,
                2.9199748459702164e-17,
                3.6465420722823106e-17,
                -1.3474403134815904e-01,
                1.7435832050146099e-01,
                -4.0331596355153267e-03,
                1.1406512426356089e-01,
                -8.5920827471777893e-03,
                1.6572238581212750e-01,
                -5.4147753150234599e-04,
                2.0100566667421285e-02,
                4.7309082545455156e-02,
                -1.3759020015509361e-03,
                -4.4863124369850617e-07,
                3.0539606710517773e-02
            ],
            [
                1.7973353394243797e-19,
                -6.7543450570403855e-19,
                -8.4349882907630346e-19,
                3.1168246494311653e-03,
                -4.0331596355153267e-03,
                9.3292804138734469e-05,
                -2.6384897056664926e-03,
                1.9874716381031599e-04,
                -3.8333958288270971e-03,
                1.2525126810096079e-05,
                -4.6495399980788226e-04,
                -1.0943260314261291e-03,
                3.1826560447234225e-05,
                1.0377464088700633e-08,
                -7.0642318849678998e-04
            ],
            [
                -4.8941005028919931e-18,
                1.9561306949366043e-17,
                2.4068923977249360e-17,
                -8.8298096530807751e-02,
                1.1406512426356089e-01,
                -2.6384897056664926e-03,
                7.4353591444240155e-02,
                -5.5971756409790600e-03,
                1.0795730876924506e-01,
                -3.4788240591531372e-04,
                1.2913056624031104e-02,
                3.0667379208975880e-02,
                -8.9189976885768198e-04,
                -2.8839511335905925e-07,
                1.9557431428525694e-02
            ],
            [
                3.6611680642451311e-19,
                -1.4796321164534017e-18,
                -1.8158792245927245e-18,
                6.6531461850346148e-03,
                -8.5920827471777893e-03,
                1.9874716381031599e-04,
                -5.5971756409790600e-03,
                4.2129478267844100e-04,
                -8.1258586703714629e-03,
                2.6119427376232763e-05,
                -9.6951504560720725e-04,
                -2.3062692181630685e-03,
                6.7073156434475589e-05,
                2.1655284105992616e-08,
                -1.4675305971897919e-03
            ],
            [
                -7.0616564127352395e-18,
                2.8538685078587920e-17,
                3.5024241030881503e-17,
                -1.2832450887602964e-01,
                1.6572238581212750e-01,
                -3.8333958288270971e-03,
                1.0795730876924506e-01,
                -8.1258586703714629e-03,
                1.5673011352368363e-01,
                -5.0378861753152041e-04,
                1.8699898983539926e-02,
                4.4482964399271574e-02,
                -1.2936966822934826e-03,
                -4.1768465717809468e-07,
                2.8305590658828657e-02
            ],
            [
                1.9644079149844693e-20,
                -1.0156629182594278e-19,
                -1.1830423962839903e-19,
                4.2197929093243500e-04,
                -5.4147753150234599e-04,
                1.2525126810096079e-05,
                -3.4788240591531372e-04,
                2.6119427376232763e-05,
                -5.0378861753152041e-04,
                1.5308965973324863e-06,
                -5.6807427103774941e-05,
                -1.4022589507750065e-04,
                4.0780503100458020e-06,
                1.2722760402462506e-09,
                -8.4841391541911763e-05
            ],
            [
                -7.2856489090013290e-19,
                3.7719068378718873e-18,
                4.3923951135954572e-18,
                -1.5665104886420242e-02,
                2.0100566667421285e-02,
                -4.6495399980788226e-04,
                1.2913056624031104e-02,
                -9.6951504560720725e-04,
                1.8699898983539926e-02,
                -5.6807427103774941e-05,
                2.1079662403256207e-03,
                5.2044436966961118e-03,
                -1.5135563555693438e-04,
                -4.7211342405620767e-08,
                3.1479927937086705e-03
            ],
            [
                -1.9089707957827610e-18,
                8.4064398812425566e-18,
                1.0119023251342569e-17,
                -3.6717074830875517e-02,
                4.7309082545455156e-02,
                -1.0943260314261291e-03,
                3.0667379208975880e-02,
                -2.3062692181630685e-03,
                4.4482964399271574e-02,
                -1.4022589507750065e-04,
                5.2044436966961118e-03,
                1.2539071178292593e-02,
                -3.6466920826119093e-04,
                -1.1635395537268063e-07,
                7.8420838315070417e-03
            ],
            [
                5.5513874790709594e-20,
                -2.4449924432051137e-19,
                -2.9429997987773429e-19,
                1.0678560554101626e-03,
                -1.3759020015509361e-03,
                3.1826560447234225e-05,
                -8.9189976885768198e-04,
                6.7073156434475589e-05,
                -1.2936966822934826e-03,
                4.0780503100458020e-06,
                -1.5135563555693438e-04,
                -3.6466920826119093e-04,
                1.0605540630499667e-05,
                3.3838111347074588e-09,
                -2.2806170259148838e-04
            ],
            [
                1.6391255866988714e-23,
                -8.3870612308112678e-23,
                -9.7888535779815032e-23,
                3.4953239652916074e-07,
                -4.4863124369850617e-07,
                1.0377464088700633e-08,
                -2.8839511335905925e-07,
                2.1655284105992616e-08,
                -4.1768465717809468e-07,
                1.2722760402462506e-09,
                -4.7211342405620767e-08,
                -1.1635395537268063e-07,
                3.3838111347074588e-09,
                1.0572355700685464e-12,
                -7.0551270418318240e-08
            ],
            [
                -1.0632179565817721e-18,
                5.8368873534248483e-18,
                6.7228466595759221e-18,
                -2.3834987621220535e-02,
                3.0539606710517773e-02,
                -7.0642318849678998e-04,
                1.9557431428525694e-02,
                -1.4675305971897919e-03,
                2.8305590658828657e-02,
                -8.4841391541911763e-05,
                3.1479927937086705e-03,
                7.8420838315070417e-03,
                -2.2806170259148838e-04,
                -7.0551270418318240e-08,
                4.6854160256993545e-03
            ]
        ],
        [
            [
                2.3553149267636458e-33,
                -1.0438233539047311e-33,
                -7.8579656921547233e-33,
                -2.3441360386668561e-17,
                1.4263351203713735e-17,
                -2.6093255193186617e-17,
                -1.8635217023604232e-17,
                2.3853096167402805e-17,
                -1.2213107649242517e-17,
                2.3172311084936394e-18,
                -1.2475239212774278e-18,
                3.4139533191648033e-18,
                -6.3146276163904831e-18,
                6.3170935943146834e-21,
                -3.8667022529987959e-18
            ],
            [
                -1.0438233539047311e-33,
                -1.5142813910942575e-33,
                -1.2842602385487934e-33,
                -5.5972017933671754e-18,
                3.2382217736928730e-18,
                -5.9238973944444715e-18,
                -3.7335243760921375e-18,
                4.6596447655163859e-18,
                -2.3858243522070970e-18,
                1.4255961258222559e-19,
                -7.5919299239822280e-20,
                5.2497482917731963e-19,
                -9.7180425733692675e-19,
                7.0954822569992410e-22,
                -9.6596685049564519e-20
            ],
            [
                -7.8579656921547233e-33,
                -1.2842602385487934e-33,
                1.4722541001507331e-32,
                3.9660854683263287e-17,
                -2.4536311210695337e-17,
                4.4886711176076357e-17,
                3.3255926717347976e-17,
                -4.2855208020030693e-17,
                2.1942389699521796e-17,
                -4.9109489694873499e-18,
                2.6459015834951910e-18,
                -6.4758478014066378e-18,
                1.1976180202039769e-17,
                -1.2614134762557022e-20,
                8.5354519431721869e-18
            ],
            [
                -2.3441360386668561e-17,
                -5.5972017933671754e-18,
                3.9660854683263287e-17,
                1.0403233666646444e-01,
                -6.4655008095409208e-02,
                1.1827996400945054e-01,
                8.8493384256423738e-02,
                -1.1423608767594046e-01,
                5.8490232958803928e-02,
                -1.3605217738126452e-02,
                7.3313328626561942e-03,
                -1.7497667013090534e-02,
                3.2358271895496796e-02,
                -3.4494690317996897e-05,
                2.3845187619459005e-02
            ],
            [
                1.4263351203713735e-17,
                3.2382217736928730e-18,
                -2.4536311210695337e-17,
                -6.4655008095409208e-02,
                4.0150581836315639e-02,
                -7.3451517083987491e-02,
                -5.4861641622559972e-02,
                7.0799699057472604e-02,
                -3.6250290457332579e-02,
                8.3774532088038041e-03,
                -4.5141707194151824e-03,
                1.0819475516990125e-02,
                -2.0008483708104634e-02,
                2.1286300217016820e-05,
                -1.4662451833618442e-02
            ],
            [
                -2.6093255193186617e-17,
                -5.9238973944444715e-18,
                4.4886711176076357e-17,
                1.1827996400945054e-01,
                -7.3451517083987491e-02,
                1.3437228341211754e-01,
                1.0036390307457202e-01,
                -1.2952098023394296e-01,
                6.6316286883093462e-02,
                -1.5325689087454648e-02,
                8.2582110275167871e-03,
                -1.9793138675297632e-02,
                3.6603501966163708e-02,
                -3.8941118426182169e-05,
                2.6823438322264369e-02
            ],
            [
                -1.8635217023604232e-17,
                -3.7335243760921375e-18,
                3.3255926717347976e-17,
                8.8493384256423738e-02,
                -5.4861641622559972e-02,
                1.0036390307457202e-01,
                7.4693722545683655e-02,
                -9.6331560054167792e-02,
                4.9322919778103415e-02,
                -1.1239396323126538e-02,
                6.0559721305881425e-03,
                -1.4648367785248313e-02,
                2.7089640393297337e-02,
                -2.8693430437991996e-05,
                1.9611975359592807e-02
            ],
            [
                2.3853096167402805e-17,
                4.6596447655163859e-18,
                -4.2855208020030693e-17,
                -1.1423608767594046e-01,
                7.0799699057472604e-02,
                -1.2952098023394296e-01,
                -9.6331560054167792e-02,
                1.2422340428497536e-01,
                -6.3603882937176615e-02,
                1.4456997517531465e-02,
                -7.7895870149116099e-03,
                1.8872880586358068e-02,
                -3.4902238400242355e-02,
                3.6939334617756023e-05,
                -2.5212563669607061e-02
            ],
            [
                -1.2213107649242517e-17,
                -2.3858243522070970e-18,
                2.1942389699521796e-17,
                5.8490232958803928e-02,
                -3.6250290457332579e-02,
                6.6316286883093462e-02,
                4.9322919778103415e-02,
                -6.3603882937176615e-02,
                3.2565956052367694e-02,
                -7.4021643832225242e-03,
                3.9883664447633175e-03,
                -9.6631459880872940e-03,
                1.7870373476420152e-02,
                -1.8913397345817687e-05,
                1.2909152095658870e-02
            ],
            [
                2.3172311084936394e-18,
                1.4255961258222559e-19,
                -4.9109489694873499e-18,
                -1.3605217738126452e-02,
                8.3774532088038041e-03,
                -1.5325689087454648e-02,
                -1.1239396323126538e-02,
                1.4456997517531465e-02,
                -7.4021643832225242e-03,
                1.5878942926301478e-03,
                -8.5536324271009425e-04,
                2.1531111288336465e-03,
                -3.9820442515763636e-03,
                4.1389721603886038e-06,
                -2.7332623800194094e-03
            ],
            [
                -1.2475239212774278e-18,
                -7.5919299239822280e-20,
                2.6459015834951910e-18,
                7.3313328626561942e-03,
                -4.5141707194151824e-03,
                8.2582110275167871e-03,
                6.0559721305881425e-03,
                -7.7895870149116099e-03,
                3.9883664447633175e-03,
                -8.5536324271009425e-04,
                4.6076460213972839e-04,
                -1.1600228570128647e-03,
                2.1453901716661453e-03,
                -2.2297655051898954e-06,
                1.4722622909746931e-03
            ],
            [
                3.4139533191648033e-18,
                5.2497482917731963e-19,
                -6.4758478014066378e-18,
                -1.7497667013090534e-02,
                1.0819475516990125e-02,
                -1.9793138675297632e-02,
                -1.4648367785248313e-02,
                1.8872880586358068e-02,
                -9.6631459880872940e-03,
                2.1531111288336465e-03,
                -1.1600228570128647e-03,
                2.8474825714829757e-03,
                -5.2660461259965100e-03,
                5.5388531753824927e-06,
                -3.7384939532378644e-03
            ],
            [
                -6.3146276163904831e-18,
                -9.7180425733692675e-19,
                1.1976180202039769e-17,
                3.2358271895496796e-02,
                -2.0008483708104634e-02,
                3.6603501966163708e-02,
                2.7089640393297337e-02,
                -3.4902238400242355e-02,
                1.7870373476420152e-02,
                -3.9820442515763636e-03,
                2.1453901716661453e-03,
                -5.2660461259965100e-03,
                9.7388621596507779e-03,
                -1.0243563894341781e-05,
                6.9141978627290952e-03
            ],
            [
                6.3170935943146834e-21,
                7.0954822569992410e-22,
                -1.2614134762557022e-20,
                -3.4494690317996897e-05,
                2.1286300217016820e-05,
                -3.8941118426182169e-05,
                -2.8693430437991996e-05,
                3.6939334617756023e-05,
                -1.8913397345817687e-05,
                4.1389721603886038e-06,
                -2.2297655051898954e-06,
                5.5388531753824927e-06,
                -1.0243563894341781e-05,
                1.0713766282307096e-08,
                -7.1573938360278508e-06
            ],
            [
                -3.8667022529987959e-18,
                -9.6596685049564519e-20,
                8.5354519431721869e-18,
                2.3845187619459005e-02,
                -1.4662451833618442e-02,
                2.6823438322264369e-02,
                1.9611975359592807e-02,
                -2.5212563669607061e-02,
                1.2909152095658870e-02,
                -2.7332623800194094e-03,
                1.4722622909746931e-03,
                -3.7384939532378644e-03,
                6.9141978627290952e-03,
                -7.1573938360278508e-06,
                4.6903010197161549e-03
            ]
        ]
    ];
    let q_ov: Array3<f64> = array![
        [
            [
                1.5858086485954589e-17,
                -2.1698265939773277e-16,
                -1.6484836330571115e-16,
                4.0711789317526836e-05,
                6.0149427446544480e-02,
                -1.3865377876193614e-03,
                -7.2972282680777517e-02,
                -1.3043023978948848e-03,
                2.4366182402976000e-02,
                7.9108488340276350e-03,
                -2.9461064526900904e-01,
                -1.3241132242456505e-01,
                3.8105603218842200e-03,
                3.4612533735647530e-05,
                2.4152186951326138e-01
            ],
            [
                -2.0876392270688272e-17,
                -5.1511695658019344e-16,
                -4.1087120239823738e-16,
                3.2161652207716982e-03,
                2.3120024709204893e-03,
                -3.2776754518957098e-02,
                5.5635339005052284e-03,
                2.6810480383329373e-02,
                -4.7495782160310368e-03,
                -7.0481649563321036e-02,
                4.6639168054546255e-03,
                9.3656967221670585e-03,
                1.3474833165919556e-01,
                -1.1440751623483843e-01,
                -4.4042326041545841e-03
            ],
            [
                -9.5284195878132702e-18,
                2.8328594184103062e-16,
                2.0901485800746188e-16,
                1.0091786233700401e-01,
                9.5675450147296032e-02,
                -1.1919644818845011e-03,
                1.7558554707413057e-01,
                9.2309696023380704e-03,
                -1.9259030172282010e-01,
                -3.3411645067036840e-03,
                2.0828527055133833e-01,
                1.7233032506922147e-01,
                -9.3104143601221608e-03,
                3.6457972207627133e-03,
                -1.4073269716298148e-01
            ],
            [
                9.0220528368878598e-18,
                -1.5030937728790403e-17,
                5.8215319387672488e-17,
                -1.2098148378706349e-03,
                -5.8318413943516528e-05,
                5.6858860131978856e-02,
                -1.8118469587165313e-03,
                -4.5415846940850817e-02,
                -1.4454559584064358e-04,
                1.2038152493252710e-01,
                1.8431535101228417e-03,
                -8.2459309724555192e-03,
                -2.3054421679724663e-01,
                1.9490793540940593e-01,
                7.5081212919873332e-04
            ],
            [
                3.4874422463410436e-17,
                8.3427970640575346e-16,
                6.0247170026697439e-16,
                -9.2997466938054996e-02,
                -1.0555788073620670e-01,
                1.7189143154091938e-03,
                -1.4026347146419699e-01,
                -8.3007836506146116e-03,
                1.7011828614974134e-01,
                1.2334810209090733e-03,
                -1.0473592324459607e-01,
                -1.1944990034275796e-01,
                6.4906614035668882e-03,
                -2.5510839292780356e-03,
                5.7600701143527944e-02
            ],
            [
                4.2360645596812191e-18,
                1.4813488731043304e-16,
                1.0362210748333213e-16,
                9.3109720724170258e-02,
                1.3189682379204540e-01,
                -3.0753599694414730e-03,
                1.0919162211761316e-01,
                8.3520344755042483e-03,
                -1.5989776201070793e-01,
                5.2233732910954072e-04,
                -1.9753602881487184e-02,
                6.3810458745715143e-02,
                -1.8483907632642679e-03,
                -2.7685183132345609e-06,
                4.2891623803173518e-02
            ],
            [
                2.6072451238839859e-19,
                3.7376083471188027e-16,
                1.9185294812933604e-16,
                -3.3486303674503184e-06,
                -1.4657310347585377e-03,
                -6.3160229333168089e-02,
                -1.8181707324561695e-05,
                4.9625447951631388e-02,
                2.5880831497496204e-03,
                -1.3195188387656917e-01,
                -3.5143498019928306e-03,
                7.3237978213364962e-03,
                2.5249271160565989e-01,
                -2.1277176758497063e-01,
                -2.5621493027691566e-06
            ],
            [
                7.1303418893942250e-03,
                2.4239516254675850e-01,
                1.7576084249497245e-01,
                2.4394291004837063e-15,
                3.2631500506737726e-15,
                3.2337596268835250e-17,
                3.1096230585361018e-15,
                1.0675424188756390e-16,
                -4.3002392221032799e-15,
                1.5512507531372029e-16,
                3.5041841900701322e-16,
                2.0406524101655899e-15,
                -3.8251535094496760e-16,
                2.4408201403031792e-16,
                4.6410313097476152e-16
            ],
            [
                2.2005754116822187e-16,
                7.4334368099208845e-15,
                5.3944441213548073e-15,
                -7.8748670718968586e-02,
                -1.0895299262258316e-01,
                2.5409962579496584e-03,
                -9.4541018946988270e-02,
                -7.0712543528762498e-03,
                1.3535597987559186e-01,
                -1.7318175000622651e-04,
                6.7020354324379972e-03,
                -5.7726186213153403e-02,
                1.6712910543951829e-03,
                3.4615231592214149e-06,
                -2.8483337631895062e-02
            ],
            [
                -3.2090476233357915e-17,
                -3.9350046088059029e-16,
                -2.7693378679990656e-16,
                -7.3139579217500347e-02,
                -1.5755324484161526e-01,
                4.2520885635089455e-03,
                -2.0446386422150610e-02,
                -5.8401605437416182e-03,
                1.0334775791770497e-01,
                -6.1571678287666272e-03,
                2.7596248700248505e-01,
                6.7232327743682915e-02,
                -4.3127436117821796e-03,
                1.9846247582075427e-03,
                -2.4507117089829208e-01
            ],
            [
                -6.2206576120102809e-18,
                -4.5491730064754686e-17,
                -6.4600727552133041e-17,
                -1.6103256021100798e-03,
                -4.0811820130242822e-03,
                -2.6393768855784989e-02,
                -4.5666861028281243e-04,
                2.0822113172758193e-02,
                3.3657953787709673e-03,
                -5.5819616351835806e-02,
                4.5906893279315138e-03,
                4.5729528596470132e-03,
                1.0652343540487574e-01,
                -8.9929092292278967e-02,
                -5.3947049301360161e-03
            ],
            [
                2.5272365307224967e-18,
                3.9596128494377832e-16,
                2.1464468055627879e-16,
                2.5023950940706368e-03,
                3.1945682497674854e-03,
                -5.4387752498549366e-02,
                1.8051280532916649e-03,
                4.2371845986139578e-02,
                -1.6996355068819181e-03,
                -1.1207155906175113e-01,
                -7.8846716801877492e-03,
                5.9432994473423446e-03,
                2.1448252047542798e-01,
                -1.8029377117752673e-01,
                4.6660770395865504e-03
            ],
            [
                6.0953426763854530e-18,
                -1.2524401519577492e-16,
                -9.4336807712842367e-17,
                6.7590801742109977e-02,
                1.2022675851722525e-01,
                -7.8780769708619754e-04,
                4.9112320947766276e-02,
                3.9218703456228412e-03,
                -1.0526759060123228e-01,
                7.6920020742162667e-03,
                -1.3201431120271562e-01,
                -7.7352377985651704e-03,
                -7.7251838850187736e-03,
                6.6801455108631592e-03,
                1.2596668334334193e-01
            ],
            [
                9.9239889747012346e-03,
                3.3736480575529493e-01,
                2.4300574670488168e-01,
                -1.2957104909578316e-18,
                4.7728202486855856e-17,
                5.4542440168051426e-17,
                3.1341523922584514e-18,
                -8.5575955381210712e-17,
                -6.0363105297967818e-17,
                3.3330157069308416e-17,
                -2.1476513881091791e-16,
                -8.1391766264986833e-17,
                -6.1857458560215335e-17,
                1.2440989484856233e-17,
                2.1338042414219877e-16
            ],
            [
                2.6910691599025594e-04,
                9.1482570840608346e-03,
                6.5895393331113043e-03,
                -4.8665350548995898e-17,
                -8.3248859381536038e-17,
                -3.8315307662387369e-17,
                -3.9186670538429545e-17,
                2.6378659116065724e-17,
                7.8185118778034907e-17,
                -8.8525745574547003e-17,
                7.7225870788487272e-17,
                2.4654365373780546e-18,
                1.6506983518598226e-16,
                -1.4015476464540083e-16,
                -8.0005846213867990e-17
            ]
        ],
        [
            [
                -3.3246238224097441e-16,
                1.7685076892511234e-16,
                3.1491460951008339e-16,
                -4.3507093817221447e-04,
                3.1317863170966048e-02,
                5.0809429953712765e-02,
                7.3830509429139241e-02,
                -2.1536096491118113e-02,
                -1.3529462852435736e-02,
                2.5171704592638200e-01,
                1.5355808038449795e-01,
                -7.2292350775360448e-02,
                -1.1429400569136407e-01,
                -9.1567242124906273e-04,
                -2.4224478149015100e-01
            ],
            [
                -4.3323293014138257e-16,
                2.9206143196568547e-16,
                4.3111176090311878e-16,
                -8.6103893990245456e-02,
                -2.8114120352553187e-02,
                -7.9082032499168509e-02,
                1.4886880007598871e-01,
                -1.4561811223744967e-01,
                -7.7567769361799307e-02,
                1.6766179454270666e-01,
                5.9122595778190901e-02,
                -1.3726344400804463e-01,
                -8.8108202162484134e-02,
                -5.9680708420331255e-02,
                -1.1774606770556713e-01
            ],
            [
                2.0737620346614598e-16,
                -1.2550225159341192e-16,
                -1.5546584447075408e-16,
                5.3444935545243898e-02,
                4.9772104508353322e-02,
                2.8711604470456287e-02,
                -9.3292825695362874e-02,
                7.3763452797258100e-02,
                7.4347644553763415e-02,
                -6.3473194891713630e-02,
                -1.0908867559606419e-01,
                -4.7875007434310279e-02,
                1.3845856797178391e-01,
                -9.6498331779546045e-02,
                7.5594624135694677e-02
            ],
            [
                -2.2834188271899024e-16,
                1.3213245684134270e-16,
                2.1420360857051309e-16,
                8.0726223178842066e-02,
                7.1509131314784133e-02,
                6.3783892885963270e-02,
                -1.2161701185084157e-01,
                1.1212159984319124e-01,
                9.9043402401586988e-02,
                -4.5684557654527086e-02,
                -9.7901287910524257e-02,
                -4.1319549972841874e-02,
                1.4806044372192112e-01,
                -9.5705307273054693e-02,
                4.9798757861272178e-02
            ],
            [
                6.6714128030694715e-16,
                -4.4882505339892085e-16,
                -6.6414398308906507e-16,
                4.5234695617459617e-02,
                -1.5733127550344853e-02,
                6.9626537156771473e-02,
                -6.7962246967350606e-02,
                9.1151851619643703e-02,
                1.1674702531508652e-02,
                -9.8429659560447635e-02,
                6.4239651815216153e-02,
                2.0179722675798545e-01,
                -5.5793184825151873e-02,
                1.7009521692352891e-01,
                2.8018436779905190e-02
            ],
            [
                -9.4804596529472723e-18,
                -3.3312566180014109e-17,
                -6.9971637344955602e-18,
                9.3150008391274269e-02,
                6.8603962226794934e-02,
                1.1266899846083292e-01,
                -1.0905236460206251e-01,
                1.3456415348599771e-01,
                8.7031713847710718e-02,
                1.7076139785136290e-02,
                1.0601924114228851e-02,
                3.3479863551127031e-02,
                5.4286018190938645e-02,
                -8.9353346872725337e-06,
                -4.2849377803948520e-02
            ],
            [
                2.6928808635602294e-16,
                -1.9922595354139515e-16,
                -3.4038668685767178e-16,
                1.7539996912864161e-04,
                -5.3921133761719475e-02,
                3.3014291043718419e-02,
                -4.8911860620906578e-05,
                2.7293835391915718e-02,
                -4.1660848024305072e-02,
                -6.9611233623355237e-02,
                1.1375169301275458e-01,
                2.1469105744734940e-01,
                -1.3224740090241827e-01,
                2.1278715348337990e-01,
                9.3695402667438865e-05
            ],
            [
                2.0634608803132459e-01,
                -1.2737406878130658e-01,
                -1.7575727955098541e-01,
                -2.9282629146067677e-15,
                -1.9462995803138552e-15,
                -3.6140355898506472e-15,
                3.5600305171727078e-15,
                -4.3435973848321993e-15,
                -2.5344913659985177e-15,
                1.4086711169415609e-16,
                -5.0016308664969320e-16,
                -2.0337179524131009e-15,
                -1.3628828872478525e-15,
                -8.9916217582694613e-16,
                9.9846679508217625e-16
            ],
            [
                6.6074475590658713e-15,
                -4.1108433359959753e-15,
                -5.6411477594176015e-15,
                7.8737204571001113e-02,
                5.6669567878380490e-02,
                9.3020627885458312e-02,
                -9.4344649810335204e-02,
                1.1383228582004450e-01,
                7.3640790799057151e-02,
                6.0408513968204861e-03,
                3.7547335174791426e-03,
                3.0207767939688983e-02,
                4.9146415020526170e-02,
                -9.3142353654054877e-05,
                -2.8514377626366112e-02
            ],
            [
                -3.9104863827758262e-16,
                2.8288557548999982e-16,
                3.6334832782543272e-16,
                -3.8058238676151775e-02,
                -2.3235188050510440e-02,
                -8.1690044449033744e-02,
                1.0585218339788479e-02,
                -5.4960411479890177e-02,
                -1.4206735105644755e-02,
                -9.6795160670261018e-02,
                -1.1578880502580566e-01,
                -5.8513741339279918e-02,
                7.8135987120400166e-02,
                -7.6922382542580714e-02,
                1.2716443185869131e-01
            ],
            [
                3.0591336492064450e-16,
                -2.3127097234028212e-16,
                -2.1957477923446888e-16,
                6.2617531505433741e-02,
                8.1857431733715263e-02,
                1.0798654323940249e-01,
                -1.7482531145429567e-02,
                6.8448928936114026e-02,
                5.7329460938184411e-02,
                2.1599736196458968e-01,
                9.8307185990462370e-02,
                -7.8124329840146126e-02,
                -2.1146373254540579e-02,
                -4.6527229083915621e-02,
                -2.0956029328020631e-01
            ],
            [
                -8.9098701708515524e-17,
                4.0268991719860504e-18,
                2.6481381117368596e-17,
                5.7399257020246529e-02,
                2.8336862930259450e-02,
                1.0209861062335662e-01,
                -4.1567122334706531e-02,
                8.7330248167381205e-02,
                2.9712359876112957e-02,
                6.3900604837450511e-02,
                1.1000639879288181e-01,
                9.3216121632960044e-02,
                -6.5692067762396181e-02,
                9.5945295750758192e-02,
                -1.0670207856642706e-01
            ],
            [
                1.1156205326921241e-16,
                -7.1818893761899724e-17,
                -1.7847856570112469e-16,
                -3.5920258069484748e-02,
                -7.2574474547768900e-02,
                -3.0649687931906689e-02,
                2.6126642271552473e-02,
                -2.7521185499779076e-02,
                -6.0528186591782857e-02,
                -1.1001768438835849e-01,
                4.5203623906207711e-02,
                1.5686172545344745e-01,
                -9.1500899461875618e-02,
                1.5274934271320637e-01,
                6.7082253235682343e-02
            ],
            [
                1.5033910270345754e-01,
                -9.2801861552826659e-02,
                -1.2720599029158947e-01,
                -1.0181384023563023e-16,
                -3.7280750196839617e-17,
                -1.2687613849911988e-16,
                2.1416897797727547e-16,
                -1.8591376837305698e-16,
                -4.0474153375084147e-17,
                3.0567477559665842e-16,
                7.8887232762277137e-17,
                -3.1738312597112985e-16,
                -7.7845609453562065e-17,
                -2.2248579911456059e-16,
                -1.8526513766309099e-16
            ],
            [
                -2.4483531196210123e-01,
                1.5113282117102383e-01,
                2.0716176372398301e-01,
                8.6110027912539922e-18,
                3.9640451903971435e-17,
                7.4190983865610400e-17,
                -4.8033455396476494e-17,
                2.3248447633975700e-17,
                -6.9454817908085778e-17,
                -7.1284914577702887e-17,
                1.3102183218350126e-17,
                1.8431472744312211e-16,
                -4.5810362267160729e-17,
                2.0241883475146543e-16,
                -1.1425364143051164e-17
            ]
        ],
        [
            [
                3.6533070820604459e-16,
                1.2300275102138372e-16,
                -2.4454493509374712e-16,
                -1.4685780454131870e-03,
                -2.7873507541189255e-02,
                5.1693284638404058e-02,
                -7.5554297876226514e-02,
                2.4451066156259769e-02,
                -1.2938458252076804e-02,
                -2.6121607841981082e-01,
                1.4251103568224591e-01,
                6.3004117239719326e-02,
                -1.2025727830298558e-01,
                -9.9782600497230756e-04,
                2.4381818996029395e-01
            ],
            [
                5.0613180975380950e-16,
                2.1996233907115408e-16,
                -3.4317413530843447e-16,
                -8.8979826712778001e-02,
                2.7104017624191575e-02,
                -8.1776026717510136e-02,
                -1.5345292700854410e-01,
                1.5728204579769989e-01,
                -6.6541640754835721e-02,
                -1.7452614828383803e-01,
                5.7137577582171376e-02,
                1.2784812985408969e-01,
                -1.0285659337959267e-01,
                5.4077547990371816e-02,
                1.2190994452928142e-01
            ],
            [
                -4.8658729896100978e-17,
                -2.7173364549110856e-17,
                2.8179525769604763e-17,
                -4.7635852181434624e-02,
                4.7359311736886123e-02,
                -2.6500554226300967e-02,
                -8.1259983523890980e-02,
                7.0247760149572014e-02,
                -6.2286598288436065e-02,
                -5.2679757193835185e-02,
                9.8873976710602426e-02,
                -6.7209675176425523e-02,
                -1.2661311769563149e-01,
                -1.0059615797946025e-01,
                6.2774028030184864e-02
            ],
            [
                4.5135713037644336e-16,
                2.4345881373136374e-16,
                -3.7911395347931129e-16,
                -8.0016442880907146e-02,
                6.8874959397202021e-02,
                -6.5619926694092306e-02,
                -1.1996953522222024e-01,
                1.2025232945430377e-01,
                -8.7581526400567128e-02,
                -4.9850475759024418e-02,
                9.6602866532540826e-02,
                -5.4736768270826283e-02,
                -1.4661224757481950e-01,
                -9.9006164982247610e-02,
                4.9742982876396301e-02
            ],
            [
                -5.6357246332402325e-16,
                -2.5512833000493558e-16,
                3.7140428383957747e-16,
                4.7569153198878188e-02,
                1.6960135070019644e-02,
                7.0751286076015496e-02,
                7.1523461980357944e-02,
                -9.5513819010985115e-02,
                5.1623487761141215e-03,
                9.6247081135749632e-02,
                6.4880394905324962e-02,
                -2.0372594553046489e-01,
                -4.0256127566649459e-02,
                -1.6714226495631090e-01,
                -2.9432019197585872e-02
            ],
            [
                9.7521411008241682e-18,
                -5.4162897848265452e-18,
                -2.1953694698807293e-17,
                9.3267417314522552e-02,
                -6.3285096176398686e-02,
                1.1573702979278984e-01,
                1.0869553984793512e-01,
                -1.4305329752583276e-01,
                7.3261408619633067e-02,
                -1.7364042674702602e-02,
                9.2970420932930228e-03,
                -3.0325843886622121e-02,
                5.6250762302325036e-02,
                -6.9941050859154141e-06,
                4.2787606731544608e-02
            ],
            [
                -4.5947961901994336e-16,
                -1.9955869184085038e-16,
                2.7799516173819595e-16,
                -1.6956165705075271e-04,
                5.5633388852419724e-02,
                3.0236879677739921e-02,
                -2.3327778485248557e-05,
                -2.2440000770949813e-02,
                -4.4463238449060631e-02,
                6.2980811134919434e-02,
                1.1570836199610783e-01,
                -2.2312008858512367e-01,
                -1.2049065214442091e-01,
                -2.1284046992193517e-01,
                1.0815960597729268e-04
            ],
            [
                -2.1347005531513996e-01,
                -1.1500847520267737e-01,
                1.7575403978780216e-01,
                2.1911155723243443e-15,
                -1.5924336124541745e-15,
                2.7214695075657335e-15,
                2.3295096030272331e-15,
                -3.2375255698385166e-15,
                1.7550801030325873e-15,
                -1.0318747573198183e-15,
                2.7187524035211094e-16,
                -1.5765722500971461e-16,
                1.2733092676219860e-15,
                3.9180028003596291e-16,
                1.5049647530801161e-15
            ],
            [
                -6.8319430761881718e-15,
                -3.6764637763851205e-15,
                5.6407139485984310e-15,
                -7.8691555107729710e-02,
                5.2130847028634159e-02,
                -9.5393262725052233e-02,
                -9.3883175565218760e-02,
                1.2081639968144454e-01,
                -6.1856043206578878e-02,
                5.8978868372595236e-03,
                -3.1569941076336622e-03,
                2.7449634971895120e-02,
                -5.0718386870776518e-02,
                9.5814301219958320e-05,
                -2.8362825214823285e-02
            ],
            [
                7.4214770507055558e-16,
                3.7971320934897264e-16,
                -6.2742356455218723e-16,
                3.5209815465512009e-02,
                -1.5992703560067819e-02,
                7.7612996876073037e-02,
                9.7087440859161319e-03,
                -5.2731699472146010e-02,
                6.3169575354621423e-03,
                -9.3328068818685550e-02,
                1.0584012362426447e-01,
                -6.6930507404126649e-02,
                -7.3199892289699961e-02,
                -7.8569087920105654e-02,
                1.1791890670929089e-01
            ],
            [
                -5.2624015470862123e-16,
                -3.6138072862368708e-16,
                4.8516481642984165e-16,
                6.4291369773025214e-02,
                -7.7463025443156447e-02,
                1.1518273623665311e-01,
                1.7646481749503236e-02,
                -7.6430822256737543e-02,
                5.0363326451919405e-02,
                -2.2526165488867370e-01,
                9.1831379612943104e-02,
                7.3188402114056181e-02,
                -2.8106311500861694e-02,
                4.3210970182111702e-02,
                2.1492140120266318e-01
            ],
            [
                5.0019956801521314e-16,
                2.8526961157683229e-16,
                -3.7858451712126414e-16,
                -5.9896337387689097e-02,
                2.8720252404218501e-02,
                -1.0555115416014671e-01,
                -4.3178948016922132e-02,
                9.2186967167087547e-02,
                -2.4946691486603619e-02,
                7.7792923699632741e-02,
                -1.0168956682597627e-01,
                8.5401376745609103e-02,
                5.3917773797881242e-02,
                8.4245097319724238e-02,
                -1.1162220767798846e-01
            ],
            [
                -6.8033499262419421e-17,
                1.9320312953251306e-17,
                -3.0106979009717002e-17,
                -3.1793418662623550e-02,
                6.9210117818128025e-02,
                -2.6397294209952099e-02,
                -2.2791165896248817e-02,
                2.7125704577522176e-02,
                -5.5861008624617625e-02,
                1.0171358188142701e-01,
                5.7414371031115879e-02,
                -1.6908896714095850e-01,
                -8.7123743464999820e-02,
                -1.5930594378475704e-01,
                -5.8949360506207069e-02
            ],
            [
                1.4159906035658112e-01,
                7.6287458417280768e-02,
                -1.1581031209651185e-01,
                8.0468166079692019e-17,
                -1.6109498796137204e-16,
                1.0877230577614266e-16,
                1.0789298011432686e-16,
                -9.7719860136086076e-17,
                1.2010837055571434e-16,
                -2.0201098970418406e-16,
                -6.2782524396251810e-17,
                2.6158572715015218e-16,
                1.5304225044291533e-16,
                2.5601599274868751e-16,
                1.2887297535201990e-16
            ],
            [
                2.6135290419009011e-01,
                1.4080565761025854e-01,
                -2.1375393231000894e-01,
                1.0934365206915525e-16,
                -9.1452248298480079e-17,
                2.1300385714907671e-16,
                1.9805253173563273e-16,
                -2.0435442105180705e-16,
                6.8198138260480112e-17,
                -3.4568556321206519e-17,
                1.4467483467617685e-16,
                -2.0331375279517677e-16,
                -2.8537300962347932e-17,
                -1.5928679733093698e-16,
                8.8597872013822301e-17
            ]
        ],
        [
            [
                -7.5302744832095902e-18,
                5.6575309519910192e-17,
                1.1820184956524063e-17,
                -5.0706170271373405e-04,
                2.8908404747588677e-02,
                -5.2146891578109353e-02,
                7.3924888346570281e-02,
                2.2985508967723035e-02,
                -1.1261408371031890e-02,
                -2.5967651719932705e-01,
                1.3993299314836707e-01,
                -6.5672115666036077e-02,
                1.1835569950804856e-01,
                9.6368452251687636e-04,
                -2.4232607824744590e-01
            ],
            [
                2.2953748891540509e-16,
                1.2448975611828377e-16,
                1.6315315989648291e-16,
                8.9305434224886626e-02,
                2.7453007063371452e-02,
                -8.2146972682258929e-02,
                -1.5434040094893198e-01,
                -1.5780036708384440e-01,
                6.6114080670923767e-02,
                1.7443199823633326e-01,
                -5.6614664263182971e-02,
                1.2805220067397513e-01,
                -1.0427168579153173e-01,
                -5.3394788503600632e-02,
                1.2214367489188223e-01
            ],
            [
                -2.7730467196077466e-16,
                -1.3264825250928792e-16,
                -2.3939225117612610e-16,
                4.7851587961002584e-02,
                4.6731371953738665e-02,
                -2.5824262325017073e-02,
                -8.3586901806621178e-02,
                -7.1246399235334851e-02,
                6.2205421846347170e-02,
                5.8224191130939199e-02,
                -1.0215512768849205e-01,
                -6.4108671235671141e-02,
                -1.2906716970796817e-01,
                1.0010488931957155e-01,
                6.7948416168370135e-02
            ],
            [
                -1.5608765300125358e-16,
                -1.1836803570027291e-16,
                -1.0680256551376726e-16,
                -7.9500719437894329e-02,
                -6.8933050500666088e-02,
                6.5222197575888430e-02,
                1.1972119046822548e-01,
                1.1943165787776056e-01,
                -8.6777900252503787e-02,
                -4.8360905946172136e-02,
                9.7109562441505162e-02,
                5.5163591628143785e-02,
                1.4650385055105772e-01,
                -1.0011193920733617e-01,
                -4.9046083674729339e-02
            ],
            [
                -4.5377078008352985e-16,
                -2.5514997659749507e-16,
                -3.9121635943935373e-16,
                4.7327031689210729e-02,
                -1.7155119432391102e-02,
                -7.0567586422039752e-02,
                -7.1070858651875846e-02,
                -9.5082658915466994e-02,
                4.4286173929187558e-03,
                9.6137508657410276e-02,
                6.6897221832191239e-02,
                2.0335359335937836e-01,
                4.0161533480280433e-02,
                -1.6754155500334311e-01,
                2.9312026992585311e-02
            ],
            [
                -1.9677390374076338e-16,
                -5.2705284414238221e-17,
                -1.7514880598338130e-16,
                9.3138385657064171e-02,
                6.3295228963254652e-02,
                -1.1575564238742678e-01,
                -1.0897609006230273e-01,
                -1.4295896010183631e-01,
                7.2601315359193369e-02,
                -1.7613897460777107e-02,
                9.6698547677587943e-03,
                3.0282844662365319e-02,
                -5.6148249156428770e-02,
                5.9587259068905052e-06,
                -4.2848614467885140e-02
            ],
            [
                -3.4367764277426680e-16,
                -1.5522142090102882e-16,
                -2.7817306019872578e-16,
                -1.6981024455265531e-04,
                5.5397085489480413e-02,
                3.0474663842836432e-02,
                2.9000298809974540e-05,
                2.2813301679832504e-02,
                4.4265734694977348e-02,
                -6.3397361925436713e-02,
                -1.1732180012202492e-01,
                -2.2201035169434752e-01,
                -1.1956087228758497e-01,
                2.1278137600423064e-01,
                -8.6202631993020917e-05
            ],
            [
                -2.1347702729651705e-01,
                -1.1502489317568373e-01,
                -1.7575731251389404e-01,
                -2.7372618706500536e-15,
                -1.8631586755688404e-15,
                3.2678277715655644e-15,
                3.1734826152006586e-15,
                4.0764341932949165e-15,
                -2.1087531743315923e-15,
                4.2093226318635189e-16,
                -1.7667465565860002e-16,
                -7.7388389890137073e-16,
                1.6549561532602863e-15,
                -1.1211991194986196e-16,
                1.1543294672849006e-15
            ],
            [
                -6.6101234800925353e-15,
                -3.5194693066724258e-15,
                -5.4565140838084746e-15,
                7.8722134215148828e-02,
                5.2283804376171944e-02,
                -9.5565844402623251e-02,
                -9.4270660917916332e-02,
                -1.2092734201315178e-01,
                6.1428587504204643e-02,
                -6.2383461639279290e-03,
                3.4272325699156482e-03,
                2.7309704976930305e-02,
                -5.0817569207738417e-02,
                9.0773370921039265e-05,
                -2.8520441163414789e-02
            ],
            [
                6.9396041832219547e-16,
                3.3659417217062050e-16,
                5.6960711260586289e-16,
                -3.5254180140903875e-02,
                -1.6019532425649855e-02,
                7.7692186569055577e-02,
                9.7735461073994792e-03,
                5.2832668892166285e-02,
                -6.2259965918653147e-03,
                9.3031121946492795e-02,
                -1.0653811876778262e-01,
                -6.6253176057835136e-02,
                -7.3397372842993788e-02,
                7.8889846156196444e-02,
                1.1782092565003638e-01
            ],
            [
                3.1709857555646527e-16,
                7.6820992542171841e-17,
                2.5055771790728838e-16,
                -6.4208608317440685e-02,
                -7.7532195546661534e-02,
                1.1520484527915796e-01,
                1.7879594837763524e-02,
                7.6436742007911715e-02,
                -5.0240015093226328e-02,
                2.2524937953630125e-01,
                -9.1403041638002780e-02,
                7.3961299460897620e-02,
                -2.8893655238926091e-02,
                -4.3107129067821881e-02,
                2.1495583901631662e-01
            ],
            [
                -1.8442472593341304e-16,
                -1.4652221959198221e-16,
                -1.4753458515226375e-16,
                -5.9881760185392492e-02,
                -2.8756872613531850e-02,
                1.0554361716763662e-01,
                4.3327046788118530e-02,
                9.2212546649812557e-02,
                -2.4678818319437721e-02,
                7.7462224729152873e-02,
                -1.0237317145264684e-01,
                -8.4611485604269843e-02,
                -5.3911482811882766e-02,
                8.4371909185208885e-02,
                1.1137826216484770e-01
            ],
            [
                4.4891389546898542e-16,
                1.8745994141450529e-16,
                3.6434749627051334e-16,
                -3.1565487098741352e-02,
                -6.9130936576269564e-02,
                2.6247159340302056e-02,
                2.2961515790908944e-02,
                2.6922446868983888e-02,
                -5.5667460034356846e-02,
                1.0196613004022241e-01,
                5.8765207656569438e-02,
                1.6864827820956493e-01,
                8.6469738474670488e-02,
                -1.5943033141683660e-01,
                5.8995349635593053e-02
            ],
            [
                -1.4157811356312530e-01,
                -7.6284581216876673e-02,
                -1.1579156027227104e-01,
                8.4637455794450942e-17,
                8.8798743054185042e-17,
                -1.7504765160756805e-16,
                -9.6275805994938687e-17,
                -1.7858823944542475e-16,
                9.7065546600201640e-17,
                -1.6256538019162900e-16,
                3.6543956398090591e-17,
                -4.3977463193550635e-17,
                -7.9246059785318839e-17,
                4.7197662200926650e-17,
                -1.5573874003823685e-16
            ],
            [
                -2.6135262882337051e-01,
                -1.4082103043973040e-01,
                -2.1375072328509639e-01,
                -9.1927266768166166e-17,
                -1.7429900603479763e-16,
                7.8355839307661517e-17,
                -8.7797001221602353e-17,
                -4.8718654200101354e-17,
                -5.3211903121672913e-17,
                5.4380846027419714e-16,
                -1.4702435079061028e-16,
                3.6750701595363425e-16,
                -1.2713436114024328e-16,
                -2.5065346457733652e-16,
                4.7029484334883870e-16
            ]
        ],
        [
            [
                -1.1260570365319008e-16,
                -1.6808617205243523e-18,
                -8.3723937256217354e-17,
                -1.5395528155088302e-03,
                -3.0222266725408307e-02,
                -5.0284393000650886e-02,
                -7.5646505853871587e-02,
                -2.3135710208589299e-02,
                -1.5460446164226600e-02,
                2.5333911874475440e-01,
                1.5631761490282370e-01,
                7.0012113503352347e-02,
                1.1644764489193424e-01,
                9.5174617771745518e-04,
                2.4390058387472000e-01
            ],
            [
                -2.8804628744879898e-16,
                2.5174648640119317e-16,
                -2.7244204840199817e-16,
                8.5736267680662920e-02,
                -2.7726232074975002e-02,
                -7.8751294030452432e-02,
                1.4780068667561355e-01,
                1.4524629404963624e-01,
                7.7915173438120627e-02,
                -1.6765524836500120e-01,
                -5.9680343522573738e-02,
                -1.3708884705722010e-01,
                -8.6688768977992853e-02,
                6.0356876694492613e-02,
                -1.1748623773273403e-01
            ],
            [
                1.6643442126148734e-16,
                -1.1893042009638383e-16,
                1.1793891415024426e-16,
                -5.3203066258205277e-02,
                5.0406435414876612e-02,
                2.9359294048269171e-02,
                -9.0822102614974226e-02,
                -7.2867303749317935e-02,
                -7.4361129169829315e-02,
                5.8133884821162815e-02,
                1.0557924856681297e-01,
                -5.1114913606470420e-02,
                1.3610023261267834e-01,
                9.6952063616207396e-02,
                7.0406993519057340e-02
            ],
            [
                -2.8789666900828032e-16,
                1.9763002709807030e-16,
                -2.1914040431840865e-16,
                8.1214842474188365e-02,
                -7.1456217015516910e-02,
                -6.4206353527014759e-02,
                1.2171565227930026e-01,
                1.1302291692625958e-01,
                9.9948181781224735e-02,
                -4.7209200182642491e-02,
                -9.7542991432832654e-02,
                4.0860526884568510e-02,
                -1.4812177584147784e-01,
                -9.4602991562765512e-02,
                -5.0492794581351974e-02
            ],
            [
                3.8879390731081256e-16,
                -1.7872219325528028e-16,
                2.7419563787332699e-16,
                4.5460931065288354e-02,
                1.5531016279820148e-02,
                -6.9825424625463983e-02,
                6.8323135640931010e-02,
                9.1601607666213170e-02,
                1.2455841967291610e-02,
                -9.8362370752239608e-02,
                6.2216745316944988e-02,
                -2.0221610422948649e-01,
                5.5885425419722808e-02,
                1.6967751974528991e-01,
                -2.8124025463678277e-02
            ],
            [
                8.8042947090533337e-17,
                -3.9167996095004832e-17,
                7.4121397442097910e-17,
                9.3255623341251162e-02,
                -6.8585686741107932e-02,
                -1.1269788650401379e-01,
                1.0862014523663041e-01,
                1.3476492811185700e-01,
                8.7728921906179966e-02,
                1.6836910020042752e-02,
                1.0208460698630229e-02,
                -3.3550374033087173e-02,
                -5.4407210371759093e-02,
                1.0690213822786565e-05,
                4.2786414241403406e-02
            ],
            [
                4.4003291833182961e-16,
                -3.2134234497087933e-16,
                4.4261469077098537e-16,
                1.7417164234265550e-04,
                -5.4170579713593806e-02,
                3.2785880728052413e-02,
                3.9257671931330895e-05,
                -2.6926727645940847e-02,
                4.1890546412177057e-02,
                6.9037367004912059e-02,
                -1.1219008329947605e-01,
                2.1576177021283163e-01,
                -1.3322739429273756e-01,
                -2.1283688248507587e-01,
                -1.1827199081644590e-04
            ],
            [
                2.0634145147215149e-01,
                -1.2735764028349150e-01,
                1.7575433389020739e-01,
                2.8073026277609418e-15,
                -1.9436063248712034e-15,
                -3.4893292859194260e-15,
                3.2042376473410293e-15,
                4.1893573682607585e-15,
                2.5163748093002838e-15,
                4.7714491284902165e-16,
                5.9013386028661465e-16,
                -1.3725374302300710e-15,
                -1.2895553719188308e-15,
                4.0408268078193999e-16,
                1.4050764558185481e-15
            ],
            [
                6.5098447369852781e-15,
                -4.0349085711996628e-15,
                5.5465845396732288e-15,
                -7.8675781359818078e-02,
                5.6497930232230480e-02,
                9.2883601411266573e-02,
                -9.3809508630296731e-02,
                -1.1380904867222588e-01,
                -7.4069516832501126e-02,
                -5.7266198047372374e-03,
                -3.4663785335892188e-03,
                3.0346058349762381e-02,
                4.9045063628438447e-02,
                -9.2540961951589895e-05,
                -2.8367896972361677e-02
            ],
            [
                -9.3853415534931523e-16,
                5.6956330975177242e-16,
                -8.1645170677605798e-16,
                3.7991468298353370e-02,
                -2.3196649690399650e-02,
                -8.1622336689805453e-02,
                1.0442156102961319e-02,
                5.4860939848926593e-02,
                1.4304730859017728e-02,
                9.7145193066785276e-02,
                1.1513946390430350e-01,
                -5.9208906544909845e-02,
                7.7920168302978862e-02,
                7.6586974243120678e-02,
                1.2726436864927088e-01
            ],
            [
                -3.0685947303352804e-17,
                7.4627934255384571e-17,
                -5.0076335778773925e-17,
                -6.2658145984900646e-02,
                8.1779868448094234e-02,
                1.0798215464586894e-01,
                -1.7152885400442928e-02,
                -6.8450532035628259e-02,
                -5.7456046745956030e-02,
                -2.1596199469433447e-01,
                -9.8770019278232105e-02,
                -7.7414031447489068e-02,
                -2.0390636543881291e-02,
                4.6633993275322319e-02,
                -2.0952744989051880e-01
            ],
            [
                -1.7696538592535730e-16,
                1.1687516388746236e-16,
                -1.7778937871364839e-16,
                5.7359136768989245e-02,
                -2.8281605861563155e-02,
                -1.0211023334646646e-01,
                4.1315332543929890e-02,
                8.7309684819331643e-02,
                2.9970298784003108e-02,
                6.4318078734539028e-02,
                1.0926078980724209e-01,
                -9.4017984800039311e-02,
                6.5776319857220281e-02,
                9.5804396127033287e-02,
                1.0695687974572474e-01
            ],
            [
                -4.9182601851649034e-16,
                3.5004528743391337e-16,
                -4.7150195433113244e-16,
                -3.6132483216187936e-02,
                7.2651262714248308e-02,
                3.0803497344088351e-02,
                -2.5903955279181524e-02,
                -2.7739320321290530e-02,
                -6.0759682945121511e-02,
                -1.0967180693521378e-01,
                4.3937447063632513e-02,
                -1.5723525781517622e-01,
                9.2170526390438490e-02,
                1.5263171823120292e-01,
                -6.7044928214110622e-02
            ],
            [
                -1.5036042357269927e-01,
                9.2805124124921867e-02,
                -1.2722475025145127e-01,
                -2.3995996413482949e-16,
                1.0807727256116830e-16,
                4.1594685131031962e-16,
                -1.8433707169898794e-16,
                -4.4838024751139308e-16,
                -1.2483104757065690e-16,
                -1.9482397853507051e-16,
                -3.7284382320068269e-16,
                3.5574259169909794e-16,
                -2.1382227942242294e-16,
                -3.4343516300712273e-16,
                -3.6930947350356437e-16
            ],
            [
                2.4483504825714114e-01,
                -1.5111654053367440e-01,
                2.0716270638712997e-01,
                -1.6789476333259159e-16,
                2.7778590042574283e-16,
                1.9691156195622999e-16,
                -1.1771111618752472e-16,
                -3.9581940213273577e-17,
                -2.3353384987191346e-16,
                -5.0652696264571082e-16,
                -6.3991425438455222e-17,
                -3.6760307911807721e-16,
                1.6979381351811083e-16,
                3.2947645966292302e-16,
                -4.1957310029459312e-16
            ]
        ],
        [
            [
                -3.4837581344947472e-17,
                -2.0713077777161702e-16,
                1.2282893459897133e-16,
                -2.0113633800059663e-03,
                -5.8171501626961684e-02,
                1.3603309137765343e-03,
                7.6496573417273325e-02,
                -1.4605233587524982e-03,
                2.8827527201092162e-02,
                7.9254848250763493e-03,
                -2.9770428914516067e-01,
                1.3856897292422088e-01,
                -4.0565109601400050e-03,
                -3.6594052894287077e-05,
                -2.4464591592677667e-01
            ],
            [
                -3.5120279757502885e-17,
                -3.9503464364756935e-16,
                2.4997063426114324e-16,
                -3.2278217618364000e-03,
                2.3187870553471688e-03,
                -3.2491006130372924e-02,
                5.5612628556366674e-03,
                -2.6624234500396728e-02,
                4.8105576317888973e-03,
                7.0242246300497135e-02,
                -4.6266290720317384e-03,
                9.4074632030465444e-03,
                1.3350353995048056e-01,
                1.1317018877187221e-01,
                -4.4172255984154711e-03
            ],
            [
                -3.9299296336868509e-17,
                9.0358854044858025e-17,
                -8.4079178175248717e-17,
                -1.0120979728265170e-01,
                9.7054795243001951e-02,
                -1.2069764024291653e-03,
                1.7337727776904832e-01,
                -9.1735332495958763e-03,
                1.9354127364528731e-01,
                3.1223081664293732e-03,
                -2.0096984937758228e-01,
                1.7165192176277447e-01,
                -9.2467436585729739e-03,
                -3.6107686266516471e-03,
                -1.3599342231327188e-01
            ],
            [
                -5.1943112011519159e-17,
                -4.1998620768058933e-16,
                2.4033460724469440e-16,
                -1.2150469169030778e-03,
                5.1798126931384275e-05,
                -5.6875265598281294e-02,
                1.8313613978154443e-03,
                -4.5500231528534218e-02,
                -1.3061405309055911e-04,
                1.2104627718920434e-01,
                1.8936038788351737e-03,
                8.2697683069016936e-03,
                2.3046303130572121e-01,
                1.9452204904462969e-01,
                -7.4963050404845688e-04
            ],
            [
                -2.2513327237014257e-17,
                3.1905794629924299e-16,
                -2.4364976403190932e-16,
                -9.2591485468190696e-02,
                1.0490354011257549e-01,
                -1.6829233919952316e-03,
                1.3861860491999767e-01,
                -8.2132624877881266e-03,
                1.7007481501906210e-01,
                1.1693138502573749e-03,
                -1.0317483893366527e-01,
                1.1975323440927334e-01,
                -6.4844000174656049e-03,
                -2.5375678917456349e-03,
                -5.7346406095202085e-02
            ],
            [
                1.6937169963230825e-17,
                -5.4276585644374130e-17,
                3.5919287106551832e-17,
                9.3293095484914795e-02,
                -1.3186631248435704e-01,
                3.0494362935797300e-03,
                -1.0848210210614204e-01,
                8.3323312411043796e-03,
                -1.6073030952154727e-01,
                5.4220575008415628e-04,
                -2.0020896586264089e-02,
                -6.3849018306567890e-02,
                1.8597793252157317e-03,
                3.3405570184219524e-06,
                -4.2739749686116560e-02
            ],
            [
                -3.2306261646646249e-17,
                3.9764766518194962e-16,
                -3.6130842184152788e-16,
                -2.8932347814134007e-06,
                -1.4722203689900883e-03,
                -6.3343230188096400e-02,
                1.7115549315974490e-05,
                -4.9842127599262175e-02,
                -2.5890040845599881e-03,
                1.3301224178734303e-01,
                3.5685901384516581e-03,
                7.3540602393694600e-03,
                2.5303600863219072e-01,
                2.1288059257578879e-01,
                3.7071871995184025e-06
            ],
            [
                7.1292012187867161e-03,
                2.4236991489640097e-01,
                -1.7575462410810230e-01,
                -1.8909418556967992e-15,
                2.4287684251739136e-15,
                2.1453530752002761e-17,
                2.5212066387648241e-15,
                -1.6088113105645057e-16,
                3.3906512413315323e-15,
                -5.2914838263871429e-17,
                -7.0881005217842023e-16,
                1.7963126435340843e-15,
                -2.1344205562705922e-16,
                -1.3518440252366773e-16,
                -2.4856479372103418e-17
            ],
            [
                2.4762255687887222e-16,
                7.8124905379726353e-15,
                -5.6633483669165750e-15,
                7.8658821822829905e-02,
                -1.0861884062295131e-01,
                2.5123948543525126e-03,
                -9.3612932193854831e-02,
                7.0312893846585026e-03,
                -1.3563892624106616e-01,
                1.9413310702533172e-04,
                -7.0911823218376829e-03,
                -5.7597955595939436e-02,
                1.6739324971062628e-03,
                2.1288314159314398e-07,
                -2.8388019807796788e-02
            ],
            [
                -4.6179363964063283e-17,
                -1.1520351772873803e-15,
                8.0489026944726272e-16,
                7.3382464569764938e-02,
                -1.5760499635762842e-01,
                4.2307406719357434e-03,
                -2.0071169171002518e-02,
                5.8610610420961667e-03,
                -1.0371794822854374e-01,
                6.1194032205147035e-03,
                -2.7505135199230701e-01,
                6.9377508069766197e-02,
                -4.3426961539435940e-03,
                -1.9686607876105103e-03,
                -2.4509693221832207e-01
            ],
            [
                -5.0520269620935091e-18,
                4.5978740877847058e-16,
                -3.6170254135727727e-16,
                1.6148574890709694e-03,
                -4.0853117150794656e-03,
                -2.6357939427886310e-02,
                -4.3520411619518652e-04,
                -2.0824577044595009e-02,
                -3.3726767627668252e-03,
                5.6030244333763357e-02,
                -4.5561546167735062e-03,
                4.6194394528647964e-03,
                1.0630238519032428e-01,
                8.9596756721413456e-02,
                -5.3948203092582508e-03
            ],
            [
                2.2390151815022479e-17,
                -5.3126204728849190e-16,
                4.4493937388142393e-16,
                2.5168507432738346e-03,
                -3.1970038485614749e-03,
                5.4449870205881455e-02,
                -1.8199289282211503e-03,
                4.2480964878346968e-02,
                -1.7238280267829632e-03,
                -1.1277279216362314e-01,
                -7.9228677316530282e-03,
                -5.9202296155185553e-03,
                -2.1456608849623182e-01,
                -1.8006912087653731e-01,
                -4.6779553421877722e-03
            ],
            [
                -1.3843045985278631e-17,
                -4.4560399923882787e-16,
                3.0534977066949448e-16,
                6.7818282869967161e-02,
                -1.2035775062776716e-01,
                7.7033942021851806e-04,
                -4.8682538987325899e-02,
                3.9243674353535429e-03,
                -1.0580523233426825e-01,
                7.7144788176055981e-03,
                -1.3193630402992446e-01,
                8.7649369230787082e-03,
                7.7105545263105625e-03,
                6.6745717738493134e-03,
                -1.2613321007547648e-01
            ],
            [
                -9.9236148989153682e-03,
                -3.3737094552779445e-01,
                2.4302686620694197e-01,
                -4.5808678340920964e-17,
                1.5454218415366451e-16,
                -1.3156460559832272e-16,
                -1.2982440453042574e-16,
                -3.6652499461304225e-17,
                -3.7966664019577918e-17,
                1.3152584282959422e-16,
                4.6364184990689818e-16,
                -1.8636351647726355e-16,
                3.1437485712424009e-16,
                2.5963699417243572e-16,
                3.7882620094491226e-16
            ],
            [
                -2.6911857774989550e-04,
                -9.1491648919383786e-03,
                6.5906461508812453e-03,
                8.6573558394965453e-17,
                -1.4457623544161894e-16,
                -2.6877430297241858e-17,
                -7.5945103948474261e-17,
                -1.3988772917392896e-17,
                -1.4224634288035689e-16,
                6.1353269691806494e-17,
                -1.1923250100016569e-16,
                -8.4412419754727172e-18,
                1.1151237442935368e-16,
                9.2830254136706814e-17,
                -1.2188173050428888e-16
            ]
        ],
        [
            [
                6.4540954959181795e-20,
                -1.1374298245222873e-19,
                -8.2624668507882212e-20,
                -6.8451820530203759e-05,
                -3.5137065694926793e-05,
                8.1931834163096614e-07,
                5.2035800071814978e-05,
                4.9008998363212972e-06,
                -9.3865890519759547e-05,
                1.6406950551201738e-06,
                -6.1517388436031062e-05,
                6.9496628077039786e-05,
                -2.0148121495778532e-06,
                -7.1238466839554464e-10,
                1.1134074323865156e-04
            ],
            [
                4.5052986363159500e-19,
                5.3698909066929805e-19,
                4.7084213680510043e-19,
                -3.1187237415534865e-03,
                -4.1054510716287814e-03,
                9.5696443685291415e-05,
                -2.7913787158112737e-03,
                -2.1043750279973737e-04,
                4.0291887681605043e-03,
                -1.4391370581046903e-05,
                5.4123380585876471e-04,
                -1.2067465616080436e-03,
                3.5007181670272019e-05,
                4.4989831994032155e-09,
                -8.5489452863587426e-04
            ],
            [
                1.4129608378967144e-17,
                1.6843420705527852e-17,
                1.4768425313123823e-17,
                -9.7814527987067440e-02,
                -1.2876254028534920e-01,
                3.0014039763779860e-03,
                -8.7549139862115460e-02,
                -6.6001989009250913e-03,
                1.2637218651410065e-01,
                -4.5138896027079295e-04,
                1.6975932329608544e-02,
                -3.7849102455467361e-02,
                1.0979856333759549e-03,
                1.4111410537864312e-07,
                -2.6814146590875665e-02
            ],
            [
                -1.8297719723542738e-19,
                -2.1511395906035511e-19,
                -1.8888312283025040e-19,
                1.2607234399224707e-03,
                1.6587429625346439e-03,
                -3.8664648085447796e-05,
                1.1266269834583221e-03,
                8.4919303429449187e-05,
                -1.6259259940615636e-03,
                5.7870615990440696e-06,
                -2.1764398394794954e-04,
                4.8632039244798290e-04,
                -1.4107958128776643e-05,
                -1.8060171494902058e-09,
                3.4355367799821548e-04
            ],
            [
                -1.4038542053130903e-17,
                -1.6504459643668763e-17,
                -1.4491891555384787e-17,
                9.6726943385440356e-02,
                1.2726442464406834e-01,
                -2.9664838393091489e-03,
                8.6438786402134982e-02,
                6.5153092688247389e-03,
                -1.2474679225579122e-01,
                4.4400584271318670e-04,
                -1.6698491485552212e-02,
                3.7312284110618078e-02,
                -1.0824142872786968e-03,
                -1.3856495865362274e-07,
                2.6358793988692598e-02
            ],
            [
                1.3504919521528805e-17,
                1.5907058107950117e-17,
                1.3964613022312634e-17,
                -9.3109664703460826e-02,
                -1.2251381807795304e-01,
                2.8557490175649589e-03,
                -8.3224134746679723e-02,
                -6.2731600332327948e-03,
                1.2011043371447908e-01,
                -4.2771025154060468e-04,
                1.6085605609445108e-02,
                -3.5932073093866250e-02,
                1.0423747247547822e-03,
                1.3351109439801764e-07,
                -2.5393593967444308e-02
            ],
            [
                -1.6776040111293029e-22,
                -1.9106374134648144e-22,
                -1.6832598809744125e-22,
                1.1436545900637760e-06,
                1.5029300646005125e-06,
                -3.5032719855401143e-08,
                1.0183319655982071e-06,
                7.6724787266909388e-08,
                -1.4690269339218450e-06,
                5.1861677263545482e-09,
                -1.9505121155522622e-07,
                4.3804551697316936e-07,
                -1.2707564331910076e-08,
                -1.6119813325767564e-12,
                3.0742821415499271e-07
            ],
            [
                4.7987679061987948e-31,
                5.4312454275449784e-31,
                4.7880970425045190e-31,
                -3.2646430036522934e-15,
                -4.2892216293711718e-15,
                9.9980106282976203e-17,
                -2.9048423742433850e-15,
                -2.1884339700107516e-16,
                4.1901296087117241e-15,
                -1.4768738870618589e-17,
                5.5545412645453372e-16,
                -1.2486882597294202e-15,
                3.6224081421834420e-17,
                4.5867793717903437e-21,
                -8.7521205092843101e-16
            ],
            [
                -1.5621015993643022e-17,
                -1.7674634784162769e-17,
                -1.5582162882574894e-17,
                1.0626073395177529e-01,
                1.3960816793146608e-01,
                -3.2542127006529638e-03,
                9.4546434884092312e-02,
                7.1228591936711889e-03,
                -1.3637927135274705e-01,
                4.8065247072171598e-04,
                -1.8077405180418716e-02,
                4.0640824634090120e-02,
                -1.1789784786951671e-03,
                -1.4927225481980090e-07,
                2.8483610023606911e-02
            ],
            [
                -1.4725943164114517e-17,
                -1.6506392771057554e-17,
                -1.4566924114070470e-17,
                9.9863534696175288e-02,
                1.3115773584790574e-01,
                -3.0572366176797776e-03,
                8.8760438083082108e-02,
                6.6861421914787399e-03,
                -1.2801755567899345e-01,
                4.5009203661053267e-04,
                -1.6928180742474156e-02,
                3.8114473227694984e-02,
                -1.1056908136665043e-03,
                -1.3961246899173079e-07,
                2.6660839710369255e-02
            ],
            [
                -3.2401843571654296e-19,
                -3.6319246117991356e-19,
                -3.2051821759797796e-19,
                2.1973179081820012e-03,
                2.8858901770820577e-03,
                -6.7268995360964746e-05,
                1.9530131985449736e-03,
                1.4711647965953521e-04,
                -2.8167950344169966e-03,
                9.9034506325116441e-06,
                -3.7247360405073403e-04,
                8.3863976097184778e-04,
                -2.4328718242169568e-05,
                -3.0719148063736717e-09,
                5.8662280724842411e-04
            ],
            [
                6.3365778983363518e-19,
                6.9554603790806885e-19,
                6.1522554989714600e-19,
                -4.2679148226086611e-03,
                -5.6010163719807660e-03,
                1.3055757035061518e-04,
                -3.7844649339698125e-03,
                -2.8499836468819751e-04,
                5.4567756998415611e-03,
                -1.9081559366033252e-05,
                7.1768177296599000e-04,
                -1.6213498375043205e-03,
                4.7035031301038773e-05,
                5.9027424619529755e-09,
                -1.1291607354447408e-03
            ],
            [
                1.7101065248938123e-17,
                1.8771277889199201e-17,
                1.6603604054085463e-17,
                -1.1518183438780119e-01,
                -1.5115937022832693e-01,
                3.5234676712650216e-03,
                -1.0213455122360211e-01,
                -7.6914914558206737e-03,
                1.4726661227241197e-01,
                -5.1497001925161171e-04,
                1.9368678928008368e-02,
                -4.3756732595513263e-02,
                1.2693739746109057e-03,
                1.5930223255612718e-07,
                -3.0473604765237659e-02
            ],
            [
                -1.2145328632286016e-32,
                -1.2597944965272536e-32,
                -1.1214626196590846e-32,
                8.0347601420468234e-17,
                1.0522714489146754e-16,
                -2.4528060338153123e-18,
                7.0798160550511720e-17,
                5.3277246303557773e-18,
                -1.0200817991632901e-16,
                3.5149188589371322e-19,
                -1.3220817962698181e-17,
                3.0143925531940276e-17,
                -8.7447424236840543e-19,
                -1.0791735936862597e-22,
                2.0743066416537407e-17
            ],
            [
                -5.1064516646713563e-33,
                -6.2581202593271729e-33,
                -5.4718312152035934e-33,
                3.5689333703210309e-17,
                4.7030523351371280e-17,
                -1.0962626734468735e-18,
                3.2045375235591903e-17,
                2.4167294991023355e-18,
                -4.6272479264431260e-17,
                1.6645068528911439e-19,
                -6.2597459889721819e-18,
                1.3895928472570480e-17,
                -4.0311358243929622e-19,
                -5.2214911107131505e-23,
                9.9002138563427947e-18
            ]
        ],
        [
            [
                -1.8726758750711608e-19,
                -5.0892876883671699e-20,
                8.0738325226411349e-20,
                4.4933215360784626e-04,
                3.3622403040411324e-04,
                5.5202996410822337e-04,
                -5.1470170418193109e-04,
                6.4184208061713590e-04,
                4.1519572959538393e-04,
                1.3024190362300874e-04,
                8.0294347676008223e-05,
                1.4104861864937040e-04,
                2.2937254049814688e-04,
                -2.5895034965344962e-07,
                -2.5327857487037039e-04
            ],
            [
                -1.0846913037955499e-19,
                -1.0825689952600685e-17,
                -2.0322340335461634e-17,
                8.3489663496981198e-02,
                5.7153453008095989e-02,
                9.3838486361607854e-02,
                -7.4591942444802772e-02,
                9.0727473218911447e-02,
                5.8689368441862733e-02,
                1.2489233416134955e-02,
                7.7086578379851611e-03,
                1.6893976731326279e-02,
                2.7462527477022335e-02,
                -2.8023859103998477e-05,
                -2.2868001155457372e-02
            ],
            [
                6.9446754318290936e-20,
                6.7251584221434435e-18,
                1.2622740495380704e-17,
                -5.1866256472565424e-02,
                -3.5505737318516715e-02,
                -5.8295771612330724e-02,
                4.6339979391128883e-02,
                -5.6364295169417392e-02,
                -3.6460674729507106e-02,
                -7.7593787325314235e-03,
                -4.7892757469737684e-03,
                -1.0495593476620791e-02,
                -1.7061438226447210e-02,
                1.7410447493790901e-05,
                1.4207719443420412e-02
            ],
            [
                -8.4843438708439768e-20,
                1.0904697011208523e-17,
                2.0653865293379381e-17,
                -8.4039907139610903e-02,
                -5.7500369576155078e-02,
                -9.4408084042983689e-02,
                7.4965833838708751e-02,
                -9.1165820748083146e-02,
                -5.8972920620230768e-02,
                -1.2506038954601940e-02,
                -7.7191284409712044e-03,
                -1.6953218538358447e-02,
                -2.7558739839580498e-02,
                2.8096169501951947e-05,
                2.2883386650739218e-02
            ],
            [
                -4.7578577595499580e-20,
                6.0999022937067979e-18,
                1.1553533385641798e-17,
                -4.7010460556834707e-02,
                -3.2164687217260755e-02,
                -5.2810208292944430e-02,
                4.1934509998060095e-02,
                -5.0996474952894694e-02,
                -3.2988361695257575e-02,
                -6.9956203546942975e-03,
                -4.3179213650359552e-03,
                -9.4833033665308458e-03,
                -1.5415827311551302e-02,
                1.5716439229176274e-05,
                1.2800485249871029e-02
            ],
            [
                -6.4005340339459492e-20,
                1.2085290818128101e-17,
                2.2861611324328006e-17,
                -9.3147600197957703e-02,
                -6.3736487839514688e-02,
                -1.0464697341226756e-01,
                8.3108358586878428e-02,
                -1.0107047706064412e-01,
                -6.5379999094111546e-02,
                -1.3871500684219716e-02,
                -8.5619200049170986e-03,
                -1.8798561221765521e-02,
                -3.0558498008867438e-02,
                3.1158443106378970e-05,
                2.5384284661451081e-02
            ],
            [
                -7.2950634985749746e-22,
                1.5788561730684936e-20,
                3.0476686326034291e-20,
                -1.2149406131786043e-04,
                -8.3033551372765685e-05,
                -1.3633024712526447e-04,
                1.0800777605237092e-04,
                -1.3129662317992101e-04,
                -8.4932534118802835e-05,
                -1.7874740010172232e-05,
                -1.1033170502372613e-05,
                -2.4345829840512185e-05,
                -3.9575706689700126e-05,
                4.0266325406149293e-08,
                3.2658592379503114e-05
            ],
            [
                2.6782449235446686e-32,
                -5.0438219984165573e-31,
                -9.7689220443592337e-31,
                3.8801971302737370e-15,
                2.6513364407252330e-15,
                4.3531482793542492e-15,
                -3.4473674300055689e-15,
                4.1903991502782957e-15,
                2.7106653666620751e-15,
                5.6969543012786883e-16,
                3.5164588312029499e-16,
                7.7660555336977124e-16,
                1.2624204666390686e-15,
                -1.2839813795654195e-18,
                -1.0405986252873162e-15
            ],
            [
                -7.8463339573908861e-19,
                1.3812874184385250e-17,
                2.6801209350302833e-17,
                -1.0624645179679992e-01,
                -7.2590273587637866e-02,
                -1.1918375351963087e-01,
                9.4363719678684801e-02,
                -1.1469813141308793e-01,
                -7.4195377763411841e-02,
                -1.5581923913367560e-02,
                -9.6180065914497955e-03,
                -2.1251021455323055e-02,
                -3.4544828619793168e-02,
                3.5127907964286816e-05,
                2.8457606119453343e-02
            ],
            [
                4.6364586368500300e-19,
                -6.7490329119934503e-18,
                -1.3170957669904682e-17,
                5.1888068531395465e-02,
                3.5438907708449006e-02,
                5.8186062905651911e-02,
                -4.6036036587191381e-02,
                5.5949479328542724e-02,
                3.6192329462286066e-02,
                7.5826549107110078e-03,
                4.6804662845532276e-03,
                1.0356857003191128e-02,
                1.6835663582300934e-02,
                -1.7108980156679029e-05,
                -1.3841860357332088e-02
            ],
            [
                -7.6347627042963186e-19,
                1.1115287486605720e-17,
                2.1691729601804213e-17,
                -8.5456843839813695e-02,
                -5.8365984119589182e-02,
                -9.5829331178432814e-02,
                7.5818938449450091e-02,
                -9.2145869370719724e-02,
                -5.9606875756886196e-02,
                -1.2488264387031833e-02,
                -7.7085006005793086e-03,
                -1.7057216735343609e-02,
                -2.7727481717581336e-02,
                2.8177636894988495e-05,
                2.2796881796704575e-02
            ],
            [
                -1.2095286038758924e-18,
                1.2723472707867661e-17,
                2.5146894855735368e-17,
                -9.7718802991349316e-02,
                -6.6689145690372764e-02,
                -1.0949488719720829e-01,
                8.6493834411258380e-02,
                -1.0509082685231949e-01,
                -6.7980640377132973e-02,
                -1.4166563173714633e-02,
                -8.7446202292202624e-03,
                -1.9414352554488985e-02,
                -3.1558982629054211e-02,
                3.2025859874246072e-05,
                2.5833242013173819e-02
            ],
            [
                7.5834929176972991e-19,
                -7.9787720862914091e-18,
                -1.5769257900917730e-17,
                6.1278598848119051e-02,
                4.1820195404091688e-02,
                6.8663311407697866e-02,
                -5.4239602283547739e-02,
                6.5901641332399258e-02,
                4.2630131612496069e-02,
                8.8837724451637859e-03,
                5.4837023126361540e-03,
                1.2174605489634265e-02,
                1.9790418625939197e-02,
                -2.0083212489756833e-05,
                -1.6199892501977866e-02
            ],
            [
                -1.2535103878875171e-33,
                -1.1152863999895362e-32,
                -1.9858839220869974e-32,
                8.6360368534173469e-17,
                5.9293650277362441e-17,
                9.7352374162034018e-17,
                -7.7849112021699567e-17,
                9.4785829063674621e-17,
                6.1314642701403172e-17,
                1.3303980328336922e-17,
                8.2109640162888278e-18,
                1.7781315944212319e-17,
                2.8905498594395413e-17,
                -2.9648488789177509e-20,
                -2.4450319003869528e-17
            ],
            [
                6.4199652605705077e-34,
                1.6140947283480076e-33,
                2.4393312862857205e-33,
                -1.2638649186628060e-17,
                -8.7478192156800802e-18,
                -1.4362754072976764e-17,
                1.1671234490177205e-17,
                -1.4248875568335218e-17,
                -9.2172597961147457e-18,
                -2.1018183068304485e-18,
                -1.2969778684917164e-18,
                -2.7253825943912897e-18,
                -4.4306202973026027e-18,
                4.6045862148110266e-21,
                3.8980716111192261e-18
            ]
        ],
        [
            [
                -3.5525735255418317e-20,
                1.5558395724004190e-19,
                -1.3969177691250383e-19,
                5.1079847701288885e-04,
                3.4909494104925327e-04,
                -6.3821574702447258e-04,
                -5.6926523976069253e-04,
                -7.5292490228065056e-04,
                3.8245978761106033e-04,
                -1.4388153335238911e-04,
                7.8388086148705393e-05,
                1.3874795057737868e-04,
                -2.5815848017947808e-04,
                3.0202698629246871e-07,
                -2.7018039752689472e-04
            ],
            [
                -4.1768039103598945e-17,
                1.0753502258280369e-17,
                -2.1896278356712717e-18,
                -8.6593689461086698e-02,
                -5.4701964444370386e-02,
                1.0000737530360654e-01,
                7.7318636726487464e-02,
                9.9983312601459429e-02,
                -5.0787338024034623e-02,
                1.3367907678519315e-02,
                -7.2906714876571780e-03,
                -1.5840957357280812e-02,
                2.9461999399818285e-02,
                -3.1234572277132908e-05,
                2.3724963066557643e-02
            ],
            [
                -2.2397010018452720e-17,
                5.7646696124940535e-18,
                -1.1726124766266302e-18,
                -4.6440019265857155e-02,
                -2.9336843023514546e-02,
                5.3634283443401730e-02,
                4.1467073663135130e-02,
                5.3622643369909739e-02,
                -2.7238058529091988e-02,
                7.1698977143566473e-03,
                -3.9103618822326934e-03,
                -8.4959720513374196e-03,
                1.5801339299853896e-02,
                -1.6752334450869063e-05,
                1.2725083435603919e-02
            ],
            [
                4.0209427764057595e-17,
                -1.0502133100486618e-17,
                2.2490150751485817e-18,
                8.2764070672277104e-02,
                5.2255716722369026e-02,
                -9.5535097333001917e-02,
                -7.3783348456894507e-02,
                -9.5394526414774306e-02,
                4.8456421123847168e-02,
                -1.2710142238845729e-02,
                6.9320202089565384e-03,
                1.5094009811382718e-02,
                -2.8072670899849224e-02,
                2.9732647338108741e-05,
                -2.2542405632939626e-02
            ],
            [
                -2.3897823261634833e-17,
                6.2418552436709040e-18,
                -1.3367431522238095e-18,
                -4.9189158994799004e-02,
                -3.1057118149242492e-02,
                5.6779334235482426e-02,
                4.3851583546933776e-02,
                5.6695723773288721e-02,
                -2.8799051371564445e-02,
                7.5539803444039828e-03,
                -4.1198866244491318e-03,
                -8.9707951940929277e-03,
                1.6684378997327533e-02,
                -1.7670933670424881e-05,
                1.3397551241407832e-02
            ],
            [
                -4.5204179662400282e-17,
                1.1783297776294643e-17,
                -2.5063655866906946e-18,
                -9.3138260914321994e-02,
                -5.8810041279233657e-02,
                1.0751786228478416e-01,
                8.3050047453784137e-02,
                1.0737815168539835e-01,
                -5.4543601367214901e-02,
                1.4313777970237581e-02,
                -7.8066181987297709e-03,
                -1.6993281752745275e-02,
                3.1605058650444273e-02,
                -3.3478483712757125e-05,
                2.5388973526520038e-02
            ],
            [
                5.7798882267000863e-20,
                -1.5557382105806741e-20,
                3.6668631417617005e-21,
                1.1712844728592775e-04,
                7.3868958351771413e-05,
                -1.3504894435954068e-04,
                -1.0405987383026184e-04,
                -1.3448572607496464e-04,
                6.8313094793145658e-05,
                -1.7781255578103357e-05,
                9.6980333092468734e-06,
                2.1217478460615261e-05,
                -3.9461096699239145e-05,
                4.1704333499048914e-08,
                -3.1489147747641937e-05
            ],
            [
                1.7963287548465084e-30,
                -4.8658964751580965e-31,
                1.1686356391038563e-31,
                3.6279268958611911e-15,
                2.2874420983199294e-15,
                -4.1819549050739730e-15,
                -3.2207046537434459e-15,
                -4.1620364840085769e-15,
                2.1141394353725169e-15,
                -5.4935477767298455e-16,
                2.9962412904769636e-16,
                6.5621277175011036e-16,
                -1.2204478771694743e-15,
                1.2892101644933824e-18,
                -9.7253770451996437e-16
            ],
            [
                -5.2631637928083474e-17,
                1.4274107187078759e-17,
                -3.4402827073761667e-18,
                -1.0622782332605149e-01,
                -6.6974449839478117e-02,
                1.2244424954923422e-01,
                9.4290435275146939e-02,
                1.2184714189337179e-01,
                -6.1893221261677886e-02,
                1.6077568218046231e-02,
                -8.7688924966615317e-03,
                -1.9208817031673198e-02,
                3.5725228831082506e-02,
                -3.7734589544696164e-05,
                2.8460729295381421e-02
            ],
            [
                2.3928539416318019e-17,
                -6.5465339333391706e-18,
                1.6176786104519140e-18,
                4.8068391607060619e-02,
                3.0295578684586683e-02,
                -5.5387085795223397e-02,
                -4.2621494103351532e-02,
                -5.5071029405908856e-02,
                2.7973763587644921e-02,
                -7.2491780400900417e-03,
                3.9538196936523195e-03,
                8.6739524411034528e-03,
                -1.6132076726298746e-02,
                1.7027975557598857e-05,
                -1.2826555387418150e-02
            ],
            [
                4.3625265653760440e-17,
                -1.1935188638143603e-17,
                2.9491630684106945e-18,
                8.7636227775426859e-02,
                5.5233618947568563e-02,
                -1.0097939448857064e-01,
                -7.7705766450133829e-02,
                -1.0040326369749104e-01,
                5.1000629416895683e-02,
                -1.3216440918751822e-02,
                7.2084619270422375e-03,
                1.5814013040517758e-02,
                -2.9411375515698883e-02,
                3.1044765880184626e-05,
                -2.3384926731473240e-02
            ],
            [
                5.1287334766421487e-17,
                -1.4298289495228925e-17,
                3.7183263174910413e-18,
                1.0196296051055041e-01,
                6.4213496082186822e-02,
                -1.1739663022953933e-01,
                -9.0196198053763457e-02,
                -1.1651029411588905e-01,
                5.9182312180395637e-02,
                -1.5254751844347806e-02,
                8.3203494952818983e-03,
                1.8314067650919551e-02,
                -3.4060849151114320e-02,
                3.5898421051901909e-05,
                -2.6962935903419485e-02
            ],
            [
                2.7094082253332703e-17,
                -7.5534016557619511e-18,
                1.9642227940645017e-18,
                5.3865414817280265e-02,
                3.3922990545923515e-02,
                -6.2018812555410911e-02,
                -4.7649302438998412e-02,
                -6.1550658047205609e-02,
                3.1265136592118298e-02,
                -8.0588903274422027e-03,
                4.3955341792374873e-03,
                9.6750641262652697e-03,
                -1.7993867197838475e-02,
                1.8964651376462863e-05,
                -1.4244185273682944e-02
            ],
            [
                -8.8554261746099193e-32,
                2.4526334148675527e-32,
                -6.2681451350700009e-33,
                -1.7669699250838142e-16,
                -1.1130934339105128e-16,
                2.0349836193541792e-16,
                1.5643593367004786e-16,
                2.0209449184709104e-16,
                -1.0265547894566411e-16,
                2.6510561984247685e-17,
                -1.4459470870810643e-17,
                -3.1789526768960658e-17,
                5.9122887350114786e-17,
                -6.2345745609507223e-20,
                4.6875294021776307e-17
            ],
            [
                4.3077745225120111e-32,
                -1.2129397859396367e-32,
                3.2359298819187907e-33,
                8.5163390595049977e-17,
                5.3611037370996068e-17,
                -9.8012971262667944e-17,
                -7.5238747402499511e-17,
                -9.7174651560866008e-17,
                4.9360617108382332e-17,
                -1.2685859975586414e-17,
                6.9192801989318473e-18,
                1.5257949384141268e-17,
                -2.8376930157008480e-17,
                2.9883225683950587e-20,
                -2.2409338296801529e-17
            ]
        ],
        [
            [
                2.7673291244648180e-20,
                -1.1275389759937158e-20,
                -3.4315286505200417e-20,
                1.5230368348188141e-03,
                -1.0695529535741939e-03,
                -1.7579443698327213e-03,
                1.4663218277121658e-03,
                1.8096611136961102e-03,
                1.1778386143717282e-03,
                2.8873422254586641e-04,
                1.7623071236643308e-04,
                -3.6018700521632633e-04,
                -5.8248149852187464e-04,
                6.7092004695047079e-07,
                5.4740500109793473e-04
            ],
            [
                -1.3688829924236675e-18,
                4.5959526651932453e-18,
                2.7705054128913954e-18,
                -8.3170441484138685e-02,
                5.6842144084826254e-02,
                9.3426643965946937e-02,
                -7.3907759381676069e-02,
                -9.0414296903301716e-02,
                -5.8847232317354177e-02,
                -1.2334884448451358e-02,
                -7.5247243910241852e-03,
                1.6905044867964028e-02,
                2.7342019134365687e-02,
                -3.0028682431113980e-05,
                -2.2768902874179556e-02
            ],
            [
                8.4866360120116619e-19,
                -2.8502451532291889e-18,
                -1.7178642175010304e-18,
                5.1564758164769540e-02,
                -3.5241187183949932e-02,
                -5.7922970615010498e-02,
                4.5820750094194077e-02,
                5.6054168024885401e-02,
                3.6483529288785996e-02,
                7.6467982898734957e-03,
                4.6648217300290402e-03,
                -1.0480382609303885e-02,
                -1.6950847462696050e-02,
                1.8616115696587706e-05,
                1.4115028151002833e-02
            ],
            [
                -1.3875393290957976e-18,
                4.7437448639594748e-18,
                2.8308923588287533e-18,
                -8.4469118820209077e-02,
                5.7699833560521399e-02,
                9.4836345187222032e-02,
                -7.4944031839777978e-02,
                -9.1665475604694568e-02,
                -5.9661578148013795e-02,
                -1.2461925582221091e-02,
                -7.6021276584407899e-03,
                1.7116211919945029e-02,
                2.7683642391336138e-02,
                -3.0371326136208135e-05,
                -2.2988353593706677e-02
            ],
            [
                -7.7807690679118736e-19,
                2.6600547930747547e-18,
                1.5874389556282326e-18,
                -4.7366829921647653e-02,
                3.2355725785148344e-02,
                5.3180374887541203e-02,
                -4.2025617730533066e-02,
                -5.1402344548134055e-02,
                -3.3455834663594308e-02,
                -6.9881760837705986e-03,
                -4.2629854454269778e-03,
                9.5981023619665408e-03,
                1.5523904111530977e-02,
                -1.7031070722993525e-05,
                -1.2891007056127083e-02
            ],
            [
                -1.5323185065536740e-18,
                5.2258068813753126e-18,
                3.1228434332057112e-18,
                -9.3257786736854406e-02,
                6.3707788180580247e-02,
                1.0471111460525119e-01,
                -8.2759475578818617e-02,
                -1.0122720010088471e-01,
                -6.5884941457031873e-02,
                -1.3768459972297331e-02,
                -8.3991653171335348e-03,
                1.8905074207722788e-02,
                3.0576923065955108e-02,
                -3.3550443431457937e-05,
                -2.5400790844146544e-02
            ],
            [
                -1.8966682879296903e-21,
                6.7413319197839589e-21,
                3.9379138499401097e-21,
                -1.1596171123607013e-04,
                7.9122048989267225e-05,
                1.3004619984118395e-04,
                -1.0253069006766822e-04,
                -1.2535736737545689e-04,
                -8.1590354850298488e-05,
                -1.6910638494802059e-05,
                -1.0315676689695786e-05,
                2.3338624331030829e-05,
                3.7747983332679371e-05,
                -4.1314441255994116e-08,
                -3.1149259730202768e-05
            ],
            [
                -6.0057025965994943e-32,
                2.1456431884596591e-31,
                1.2498525223102532e-31,
                -3.6740082170487643e-15,
                2.5064345167343588e-15,
                4.1196136509979998e-15,
                -3.2469572888514591e-15,
                -3.9696219293030174e-15,
                -2.5836763362684103e-15,
                -5.3493445565404732e-16,
                -3.2631467727570477e-16,
                7.3875634124920864e-16,
                1.1948685239415902e-15,
                -1.3073372976508185e-18,
                -9.8514779389594227e-16
            ],
            [
                1.7346032043024906e-18,
                -6.2264591072456069e-18,
                -3.6176807577157085e-18,
                1.0617171358306399e-01,
                -7.2420892214895782e-02,
                -1.1903206926043669e-01,
                9.3790546608683800e-02,
                1.1465952288600981e-01,
                7.4627534516712585e-02,
                1.5436176117885775e-02,
                9.4161683435982548e-03,
                -2.1330592930356704e-02,
                -3.4500246067757964e-02,
                3.7736396172883886e-05,
                2.8422385546308000e-02
            ],
            [
                -8.4622433421945158e-19,
                3.0728596938480950e-18,
                1.7742588433760579e-18,
                -5.1864199613038273e-02,
                3.5364848767712405e-02,
                5.8126193687811155e-02,
                -4.5767633391372735e-02,
                -5.5944372743209431e-02,
                -3.6412070767329371e-02,
                -7.5135028742391247e-03,
                -4.5832454182128017e-03,
                1.0398138313797180e-02,
                1.6818054652219763e-02,
                -1.8382056093717143e-05,
                -1.3828176487235864e-02
            ],
            [
                1.3942406228525893e-18,
                -5.0629170740282161e-18,
                -2.9232899816951498e-18,
                8.5451674200234001e-02,
                -5.8267250337194827e-02,
                -9.5768922997512787e-02,
                7.5406852285712034e-02,
                9.2174056925024395e-02,
                5.9992598346253007e-02,
                1.2379226901141070e-02,
                7.5513426075048875e-03,
                -1.7131974580081567e-02,
                -2.7709429976681431e-02,
                3.0286252797291778e-05,
                2.2783253713888962e-02
            ],
            [
                -1.5904066537747260e-18,
                5.9233293770629029e-18,
                3.3739366560870483e-18,
                -9.7761691097452849e-02,
                6.6609618752423633e-02,
                1.0948054581398242e-01,
                -8.6066815137354424e-02,
                -1.0517562999965363e-01,
                -6.8454831191528565e-02,
                -1.4049537278350449e-02,
                -8.5700637213658986e-03,
                1.9508960813931570e-02,
                3.1554138792497723e-02,
                -3.4431617340335870e-05,
                -2.5830797479332299e-02
            ],
            [
                9.9972040719569311e-19,
                -3.7234433877298600e-18,
                -2.1208563941421616e-18,
                6.1452573877211300e-02,
                -4.1870491403826843e-02,
                -6.8818953443087624e-02,
                5.4101126102281315e-02,
                6.6112807794302411e-02,
                4.3030320781280357e-02,
                8.8314225838693004e-03,
                5.3870708773108679e-03,
                -1.2263203271910626e-02,
                -1.9834722268789733e-02,
                2.1643458153072408e-05,
                1.6237011874576466e-02
            ],
            [
                5.5304363920928044e-33,
                -2.0288816147574313e-32,
                -1.1650375570817675e-32,
                3.3935488578768101e-16,
                -2.3132548271039155e-16,
                -3.8020999293333487e-16,
                2.9918123763767993e-16,
                3.6566618466646703e-16,
                2.3799825600383481e-16,
                4.9004410284983794e-17,
                2.9892509675727282e-17,
                -6.7909659341812403e-17,
                -1.0983797863612211e-16,
                1.1997312884813823e-19,
                9.0152816334158338e-17
            ],
            [
                5.2308050291211452e-33,
                -2.0023796533509035e-32,
                -1.1240846027062157e-32,
                3.2258704555376948e-16,
                -2.1960563356154062e-16,
                -3.6094697086363114e-16,
                2.8325603582866941e-16,
                3.4604037662100999e-16,
                2.2522457266579036e-16,
                4.5947090009424700e-17,
                2.8026595157125709e-17,
                -6.4042028093193677e-17,
                -1.0358325012825755e-16,
                1.1282058490489209e-19,
                8.4378331537311986e-17
            ]
        ],
        [
            [
                2.1618950136212349e-19,
                -2.3082592375793451e-19,
                -4.6716480974599582e-19,
                2.0442057960895106e-03,
                -2.7408314239961620e-03,
                6.3400018225053800e-05,
                -1.9262026575539864e-03,
                1.4687943227559671e-04,
                -2.8329333372813883e-03,
                1.1670530311559606e-05,
                -4.3369231250477001e-04,
                -8.8401162933039452e-04,
                2.5713598422819472e-05,
                9.5880719777545368e-09,
                -6.8970847204369377e-04
            ],
            [
                2.4427783276674595e-19,
                -5.4090967130053575e-19,
                -7.9149159497019757e-19,
                3.1308102322153502e-03,
                -4.1132620331078250e-03,
                9.5146097364984822e-05,
                -2.7772211082060308e-03,
                2.1035524884911966e-04,
                -4.0572591570225015e-03,
                1.4821850550219326e-05,
                -5.5051207400388957e-04,
                -1.2070457953866180e-03,
                3.5107193901775490e-05,
                1.2227641683671177e-08,
                -8.5637247391602274e-04
            ],
            [
                7.6630694764336779e-18,
                -1.6971425268186402e-17,
                -2.4832084039374267e-17,
                9.8223038370683399e-02,
                -1.2904488834147237e-01,
                2.9850073723398972e-03,
                -8.7128540278318886e-02,
                6.5993714015985484e-03,
                -1.2728638938849232e-01,
                4.6498300987032545e-04,
                -1.7270361800164294e-02,
                -3.7867565820065320e-02,
                1.1013865037254563e-03,
                3.8359937432449136e-07,
                -2.6865468206819929e-02
            ],
            [
                9.7572412339069817e-20,
                -2.1991384936042181e-19,
                -3.1977760451439858e-19,
                1.2618522653993266e-03,
                -1.6569647727786648e-03,
                3.8328146223450519e-05,
                -1.1175843023641043e-03,
                8.4633969911069565e-05,
                -1.6323910967589485e-03,
                5.9428390820677383e-06,
                -2.2072496659330656e-04,
                -4.8499968348681853e-04,
                1.4106293105549934e-05,
                4.9033073535160285e-09,
                -3.4312377623646939e-04
            ],
            [
                7.4535502700401148e-18,
                -1.6799510326552992e-17,
                -2.4428078164137810e-17,
                9.6393708722403659e-02,
                -1.2657654071343505e-01,
                2.9279102609172615e-03,
                -8.5372848048196448e-02,
                6.4652319233936204e-03,
                -1.2469918454775726e-01,
                4.5397487712339367e-04,
                -1.6861231904742202e-02,
                -3.7049321595074930e-02,
                1.0775854220782051e-03,
                3.7456485244540810e-07,
                -2.6211287150649340e-02
            ],
            [
                -7.2229725283363255e-18,
                1.6239574042965267e-17,
                2.3634515038560136e-17,
                -9.3293793970853134e-02,
                1.2251485180062130e-01,
                -2.8339572007760588e-03,
                8.2645508800390760e-02,
                -6.2588506159861855e-03,
                1.2071856816661988e-01,
                -4.3969614223681223e-04,
                1.6330937264890412e-02,
                3.5873282824423369e-02,
                -1.0433804576196316e-03,
                -3.6277733903791860e-07,
                2.5389363729737713e-02
            ],
            [
                -2.9445498682610405e-23,
                6.6069617898986313e-23,
                9.6224010156363048e-23,
                -3.7993479648539760e-07,
                4.9896568927158212e-07,
                -1.1541845014898658e-08,
                3.3663038838440578e-07,
                -2.5493977495003334e-08,
                4.9171909684531888e-07,
                -1.7917067792791393e-09,
                6.6546647943822049e-08,
                1.4614353176073957e-07,
                -4.2506102034646908e-09,
                -1.4782509276548709e-12,
                1.0346674631700220e-07
            ],
            [
                2.0121733701165109e-31,
                -4.7886061792593294e-31,
                -6.8331344983236492e-31,
                2.6765324793224076e-15,
                -3.5090224450046692e-15,
                8.1169055231930375e-17,
                -2.3590865332305480e-15,
                1.7855240763966178e-16,
                -3.4438605027059331e-15,
                1.2403494420371124e-17,
                -4.6065926369151929e-16,
                -1.0190224409731541e-15,
                2.9638234498226350e-17,
                1.0237909021850292e-20,
                -7.1457498549793135e-16
            ],
            [
                -8.0032918247063728e-18,
                1.8941998492365486e-17,
                2.7080115175802554e-17,
                -1.0615142524413200e-01,
                1.3919039893437360e-01,
                -3.2196869262866648e-03,
                9.3607293379358844e-02,
                -7.0852655048785221e-03,
                1.3665827546016784e-01,
                -4.9273181164750869e-04,
                1.8299894324955875e-02,
                4.0453391386350471e-02,
                -1.1765863613374217e-03,
                -4.0668690746087454e-07,
                2.8393051775037627e-02
            ],
            [
                -7.4944687588122395e-18,
                1.7949894701995810e-17,
                2.5558146000298712e-17,
                -1.0002456836293469e-01,
                1.3111098205492810e-01,
                -3.0327973531476204e-03,
                8.8111112179179371e-02,
                -6.6684341773069825e-03,
                1.2861859149508173e-01,
                -4.6264404702759808e-04,
                1.7182253907290952e-02,
                3.8039189719085584e-02,
                -1.1063677268999391e-03,
                -3.8188689995597241e-07,
                2.6646281654958342e-02
            ],
            [
                -1.6505699576361336e-19,
                3.9532258173030494e-19,
                5.6288568619887665e-19,
                -2.2029163108212421e-03,
                2.8875564381964842e-03,
                -6.6793592620004416e-05,
                1.9405386428412643e-03,
                -1.4686405520394211e-04,
                2.8326661727837740e-03,
                -1.0189181510689830e-05,
                3.7841858383510742e-04,
                8.3776682735471272e-04,
                -2.4366401812924557e-05,
                -8.4106015013173484e-09,
                5.8685265423781010e-04
            ],
            [
                -3.1627573802752010e-19,
                7.7780757029740864e-19,
                1.0976918836726290e-18,
                -4.2806626634835348e-03,
                5.6067031278938190e-03,
                -1.2969158216523234e-04,
                3.7619370887771088e-03,
                -2.8463346617900345e-04,
                5.4899198850720029e-03,
                -1.9642644596662958e-05,
                7.2949491131372123e-04,
                1.6203882899211038e-03,
                -4.7128750253164262e-05,
                -1.6217113978099396e-08,
                1.1300902023104961e-03
            ],
            [
                -8.5315933459575651e-18,
                2.0981467286354427e-17,
                2.9610404226311857e-17,
                -1.1547154205104218e-01,
                1.5124169599519591e-01,
                -3.4984507643360191e-03,
                1.0147885093444464e-01,
                -7.6780330989677115e-03,
                1.4809146357002320e-01,
                -5.2986361008385203e-04,
                1.9678246777520350e-02,
                4.3710236731532223e-02,
                -1.2713056762873394e-03,
                -4.3745934663441927e-07,
                3.0484373156555535e-02
            ],
            [
                6.7154927147430146e-33,
                -1.8912862700922415e-32,
                -2.5564048690760125e-32,
                9.7919445137201522e-17,
                -1.2774725380623869e-16,
                2.9549852421560902e-18,
                -8.5019913188538730e-17,
                6.4236392220725027e-18,
                -1.2389735598069254e-16,
                4.3106903663120591e-19,
                -1.6007042967539348e-17,
                -3.6187805740187771e-17,
                1.0524989448745892e-18,
                3.5627002710140455e-22,
                -2.4654835390608337e-17
            ],
            [
                -1.2230863643290724e-32,
                3.0570649349150847e-32,
                4.2912164275929886e-32,
                -1.6698085379239809e-16,
                2.1860369481125876e-16,
                -5.0566357116732565e-18,
                1.4653433492878678e-16,
                -1.1085131066884336e-17,
                2.1380653679279860e-16,
                -7.6248089785350302e-19,
                2.8316824517444141e-17,
                6.3028295491946809e-17,
                -1.8331648345364488e-18,
                -6.2958706398143627e-22,
                4.3837558850670105e-17
            ]
        ],
        [
            [
                -7.0918620952896047e-19,
                -3.7775440833712521e-19,
                6.9733689352554050e-19,
                1.4619936514975636e-03,
                -9.4821699972827596e-04,
                1.7346876766685792e-03,
                1.4129271916055207e-03,
                -1.8503012917724837e-03,
                9.4737113192585427e-04,
                -2.8830852928576683e-04,
                1.5550685098443778e-04,
                -3.1450936408796836e-04,
                5.8145888455252018e-04,
                -6.7362722517491100e-07,
                5.3055401620285658e-04
            ],
            [
                -2.6908314207327024e-17,
                -1.0518619905566219e-17,
                3.5656004306675828e-17,
                8.6316056296120797e-02,
                -5.4422381330117856e-02,
                9.9560738406508337e-02,
                7.6748709373740759e-02,
                -9.9592513072093256e-02,
                5.0992448880220767e-02,
                -1.3195679993548941e-02,
                7.1135630008130847e-03,
                -1.5865471273028498e-02,
                2.9336718315147080e-02,
                -3.2329910779116230e-05,
                2.3623352185338758e-02
            ],
            [
                -1.4389495221690314e-17,
                -5.6238328332344219e-18,
                1.9070079312952299e-17,
                4.6167337069078530e-02,
                -2.9108272851349887e-02,
                5.3250906352127869e-02,
                4.1048865789646546e-02,
                -5.3266635040970657e-02,
                2.7273095979892331e-02,
                -7.0571788475773441e-03,
                3.8044022031179358e-03,
                -8.4853629663235782e-03,
                1.5690219090020611e-02,
                -1.7290713103031542e-05,
                1.2633841390614501e-02
            ],
            [
                -2.5683776511583663e-17,
                -9.9366400362975846e-18,
                3.4282461805830790e-17,
                8.3223338002765879e-02,
                -5.2445261805282473e-02,
                9.5943767916183434e-02,
                7.3882693663306520e-02,
                -9.5856156137918619e-02,
                4.9079396854408396e-02,
                -1.2656285704377820e-02,
                6.8226978213494268e-03,
                -1.5249961505420843e-02,
                2.8198684647960964e-02,
                -3.1042016625522199e-05,
                2.2642898567407959e-02
            ],
            [
                1.5294067517592531e-17,
                5.9170809782675869e-18,
                -2.0414250599269681e-17,
                -4.9557061799153528e-02,
                3.1229631907249259e-02,
                -5.7131730360384547e-02,
                -4.3995039626262639e-02,
                5.7079619314519645e-02,
                -2.9225387302671695e-02,
                7.5364810225851502e-03,
                -4.0627348662097733e-03,
                9.0809292591861428e-03,
                -1.6791534826409740e-02,
                1.8484677129218901e-05,
                -1.3483250432747193e-02
            ],
            [
                2.8823416250306085e-17,
                1.1167342070641528e-17,
                -3.8434580861855984e-17,
                -9.3267144389748882e-02,
                5.8778787635268531e-02,
                -1.0753037098087698e-01,
                -8.2817054947226604e-02,
                1.0745034643855525e-01,
                -5.5015748163202580e-02,
                1.4193936331164055e-02,
                -7.6516215597149505e-03,
                1.7097628303300954e-02,
                -3.1615187501694328e-02,
                3.4808211620919379e-05,
                -2.5396139924289275e-02
            ],
            [
                -3.4881140121649038e-20,
                -1.3175882075619353e-20,
                4.7328370475082793e-20,
                1.1560576075554476e-04,
                -7.2768814560426236e-05,
                1.3312379512580579e-04,
                1.0227565186510105e-04,
                -1.3264052140630667e-04,
                6.7913403034667216e-05,
                -1.7377262191568857e-05,
                9.3673946262672026e-06,
                -2.1039903415397195e-05,
                3.8905167542145929e-05,
                -4.2724545132358941e-08,
                3.1043496405800581e-05
            ],
            [
                9.1925579647704593e-31,
                3.4376996197966124e-31,
                -1.2556481992718971e-30,
                -3.0747004651805911e-15,
                1.9345086019724223e-15,
                -3.5390036146379798e-15,
                -2.7163963680539247e-15,
                3.5223096013672702e-15,
                -1.8034612909875900e-15,
                4.6001027518589251e-16,
                -2.4797046670717135e-16,
                5.5805822372183369e-16,
                -1.0319162450843193e-15,
                1.1321139647574896e-18,
                -8.2129306927709163e-16
            ],
            [
                -3.1854225610608453e-17,
                -1.1956159064368358e-17,
                4.3405400947061279e-17,
                1.0619109940968088e-01,
                -6.6823267493710509e-02,
                1.2224695957642125e-01,
                9.3863516238643460e-02,
                -1.2171845690297978e-01,
                6.2321188502248756e-02,
                -1.5914464704072685e-02,
                8.5787950432036665e-03,
                -1.9292809032271253e-02,
                3.5674664574550319e-02,
                -3.9152518266481021e-05,
                2.8419515656870505e-02
            ],
            [
                1.4316742738342846e-17,
                5.3321108553220259e-18,
                -1.9608513408017133e-17,
                -4.8062977157732972e-02,
                3.0234261862289419e-02,
                -5.5310768369146418e-02,
                -4.2438495223258182e-02,
                5.5025816518992590e-02,
                -2.8173824484821117e-02,
                7.1772570979557046e-03,
                -3.8689124699669520e-03,
                8.7138848278302018e-03,
                -1.6113033492615069e-02,
                1.7670652834022110e-05,
                -1.2811058884016693e-02
            ],
            [
                2.6115075663364841e-17,
                9.7261716461647083e-18,
                -3.5767992128474602e-17,
                -8.7672138596286475e-02,
                5.5150583430049062e-02,
                -1.0089285986491386e-01,
                -7.7412363441330720e-02,
                1.0037297093890755e-01,
                -5.1392067311554886e-02,
                1.3092044235710559e-02,
                -7.0572883120806496e-03,
                1.5895039086750440e-02,
                -2.9391861726973771e-02,
                3.2233092112802460e-05,
                -2.3368656049467074e-02
            ],
            [
                -2.9927035964210095e-17,
                -1.0948373121747371e-17,
                4.1465203192901362e-17,
                1.0206656901084540e-01,
                -6.4155857160739585e-02,
                1.1736714763286765e-01,
                8.9910198519562029e-02,
                -1.1654587486937001e-01,
                5.9672778482942536e-02,
                -1.5119904275025100e-02,
                8.1502468613647242e-03,
                -1.8418812168862750e-02,
                3.4058811646604049e-02,
                -3.7288677874845015e-05,
                2.6960584539156559e-02
            ],
            [
                -1.5850582433479487e-17,
                -5.7986170602662771e-18,
                2.1961882847986760e-17,
                5.4059351331193155e-02,
                -3.3979997610853400e-02,
                6.2163231418207635e-02,
                4.7620660061950912e-02,
                -6.1728149931120552e-02,
                3.1605496303049659e-02,
                -8.0081725621544333e-03,
                4.3167325004016292e-03,
                -9.7554483113989534e-03,
                1.8039109962839951e-02,
                -1.9749767453731638e-05,
                1.4279510090943179e-02
            ],
            [
                3.7440207401483606e-32,
                1.3553995682077151e-32,
                -5.2219744391981901e-32,
                -1.2884639909026967e-16,
                8.0953472469373713e-17,
                -1.4809679969709227e-16,
                -1.1334936632945227e-16,
                1.4690612375742418e-16,
                -7.5217566474856641e-17,
                1.9000326063356524e-17,
                -1.0241836003906764e-17,
                2.3190228054290058e-17,
                -4.2881921972162431e-17,
                4.6903758343623528e-20,
                -3.3859940466531102e-17
            ],
            [
                4.5360263462211036e-32,
                1.6051753773862062e-32,
                -6.4157049233278398e-32,
                -1.5908983950530539e-16,
                9.9864728254761686e-17,
                -1.8269313081446890e-16,
                -1.3956825280617953e-16,
                1.8082878683749259e-16,
                -9.2586357832455883e-17,
                2.3238188056401320e-17,
                -1.2525890076876639e-17,
                2.8476713523940663e-17,
                -5.2657703384465019e-17,
                5.7481508204130603e-20,
                -4.1360926005708478e-17
            ]
        ]
    ];
    let omega: Array2<f64> = array![
        [
            0.5317313499297132,
            0.5317344489938158,
            0.6734942275339232,
            0.8381128471319967,
            0.8846464625423937,
            0.8846495046208143,
            1.0135043222508973,
            1.0306505739649499,
            1.0306540893642984,
            1.2589182415700959,
            1.2589373656047711,
            1.2903533914691798,
            1.2903673445584736,
            1.4559329212259793,
            1.5868580587082635
        ],
        [
            0.5295858880029307,
            0.5295889870670334,
            0.6713487656071407,
            0.8359673852052143,
            0.8825010006156112,
            0.8825040426940318,
            1.0113588603241148,
            1.0285051120381676,
            1.0285086274375157,
            1.2567727796433137,
            1.2567919036779887,
            1.2882079295423976,
            1.2882218826316911,
            1.4537874592991968,
            1.584712596781481
        ],
        [
            0.5295854396044922,
            0.5295885386685949,
            0.6713483172087022,
            0.8359669368067759,
            0.8825005522171727,
            0.8825035942955932,
            1.0113584119256762,
            1.028504663639729,
            1.0285081790390773,
            1.2567723312448751,
            1.2567914552795503,
            1.288207481143959,
            1.2882214342332525,
            1.4537870109007582,
            1.5847121483830424
        ],
        [
            0.5231437840026195,
            0.5231468830667222,
            0.6649066616068295,
            0.8295252812049032,
            0.8760588966153,
            0.8760619386937205,
            1.0049167563238035,
            1.0220630080378563,
            1.0220665234372046,
            1.2503306756430024,
            1.2503497996776773,
            1.2817658255420863,
            1.28177977863138,
            1.4473453552988857,
            1.5782704927811699
        ],
        [
            0.5231436845673584,
            0.5231467836314611,
            0.6649065621715684,
            0.8295251817696421,
            0.8760587971800389,
            0.8760618392584594,
            1.0049166568885424,
            1.0220629086025952,
            1.0220664240019435,
            1.2503305762077412,
            1.2503497002424164,
            1.2817657261068252,
            1.2817796791961187,
            1.4473452558636244,
            1.5782703933459086
        ],
        [
            0.5186331099158599,
            0.5186362089799625,
            0.6603959875200699,
            0.8250146071181435,
            0.8715482225285404,
            0.871551264606961,
            1.000406082237044,
            1.0175523339510968,
            1.0175558493504449,
            1.2458200015562428,
            1.2458391255909178,
            1.2772551514553268,
            1.2772691045446203,
            1.442834681212126,
            1.5737598186944102
        ],
        [
            0.4885344410389381,
            0.4885375401030408,
            0.6302973186431482,
            0.7949159382412218,
            0.8414495536516187,
            0.8414525957300392,
            0.9703074133601222,
            0.987453665074175,
            0.9874571804735233,
            1.215721332679321,
            1.215740456713996,
            1.2471564825784049,
            1.2471704356676985,
            1.4127360123352042,
            1.5436611498174884
        ],
        [
            0.4722972410105424,
            0.4723003400746451,
            0.6140601186147524,
            0.778678738212826,
            0.8252123536232229,
            0.8252153957016435,
            0.9540702133317265,
            0.9712164650457792,
            0.9712199804451275,
            1.1994841326509253,
            1.1995032566856003,
            1.2309192825500093,
            1.2309332356393028,
            1.3964988123068085,
            1.5274239497890927
        ],
        [
            0.4715165459178972,
            0.4715196449819998,
            0.6132794235221072,
            0.7778980431201807,
            0.8244316585305776,
            0.8244347006089983,
            0.9532895182390813,
            0.970435769953134,
            0.9704392853524823,
            1.1987034375582799,
            1.1987225615929551,
            1.2301385874573638,
            1.2301525405466576,
            1.3957181172141633,
            1.5266432546964475
        ],
        [
            0.4548460478141776,
            0.4548491468782803,
            0.5966089254183877,
            0.7612275450164613,
            0.8077611604268582,
            0.8077642025052787,
            0.9366190201353617,
            0.9537652718494145,
            0.9537687872487628,
            1.1820329394545606,
            1.1820520634892355,
            1.2134680893536445,
            1.213482042442938,
            1.3790476191104437,
            1.5099727565927279
        ],
        [
            0.4548449657303829,
            0.4548480647944855,
            0.5966078433345929,
            0.7612264629326665,
            0.8077600783430634,
            0.8077631204214839,
            0.9366179380515669,
            0.9537641897656197,
            0.953767705164968,
            1.1820318573707658,
            1.1820509814054407,
            1.2134670072698497,
            1.2134809603591432,
            1.3790465370266489,
            1.5099716745089331
        ],
        [
            0.4204439609140203,
            0.420447059978123,
            0.5622068385182304,
            0.7268254581163039,
            0.7733590735267009,
            0.7733621156051214,
            0.9022169332352044,
            0.9193631849492572,
            0.9193667003486055,
            1.1476308525544032,
            1.1476499765890782,
            1.1790660024534871,
            1.1790799555427807,
            1.3446455322102864,
            1.4755706696925706
        ],
        [
            0.4204423344841589,
            0.4204454335482615,
            0.5622052120883689,
            0.7268238316864425,
            0.7733574470968394,
            0.7733604891752599,
            0.9022153068053429,
            0.9193615585193957,
            0.919365073918744,
            1.1476292261245418,
            1.1476483501592167,
            1.1790643760236257,
            1.1790783291129192,
            1.3446439057804249,
            1.4755690432627091
        ],
        [
            0.3827429810550408,
            0.3827460801191435,
            0.5245058586592508,
            0.6891244782573245,
            0.7356580936677213,
            0.7356611357461418,
            0.8645159533762248,
            0.8816622050902776,
            0.8816657204896259,
            1.1099298726954236,
            1.1099489967300986,
            1.1413650225945076,
            1.1413789756838013,
            1.306944552351307,
            1.4378696898335912
        ],
        [
            0.382741116063357,
            0.3827442151274597,
            0.5245039936675671,
            0.6891226132656406,
            0.7356562286760375,
            0.735659270754458,
            0.864514088384541,
            0.8816603400985938,
            0.8816638554979421,
            1.1099280077037399,
            1.1099471317384149,
            1.1413631576028238,
            1.1413771106921173,
            1.306942687359623,
            1.4378678248419072
        ]
    ];
    let bp: Array3<f64> = array![
        [
            [2.3023599800652536e-17, 3.6747443705203717e-17],
            [-9.7739308272008557e-18, 4.2437160122386481e-17],
            [2.2544084264510903e-17, 3.0665262292594968e-17],
            [5.9886902424296696e-05, -2.2954068749118321e-06],
            [-3.2274413658530801e-02, 2.5787086322352276e-03],
            [2.5785766809598883e-03, 3.2273343027525138e-02],
            [-4.3425790472374284e-03, -1.0047707078250992e-05],
            [-2.0448438020988138e-06, 8.8610904279912486e-05],
            [1.1223749843643563e-04, 1.9541667090048655e-06],
            [-1.6370519908534553e-05, -5.4046081324718442e-04],
            [-3.0906374306721718e-05, -2.3529190764679658e-05],
            [7.1799068857738874e-02, -6.1571499797024865e-03],
            [-6.1573230893629798e-03, -7.1799354833610732e-02],
            [-5.4580892877915683e-05, -5.8810739027698850e-04],
            [1.4794482909805149e-02, 3.4046422380848429e-05]
        ],
        [
            [-2.6424747371139717e-19, 1.5772536707806506e-17],
            [2.2241150473580200e-17, 1.6624734892244189e-17],
            [3.5540787554799554e-17, 4.0088220661750258e-17],
            [3.1897540113745397e-04, 3.6006154867263915e-03],
            [-1.0809169055372644e-06, -1.3982500560234812e-05],
            [6.5157750302989592e-06, -1.4333784703508823e-06],
            [3.7210404196449942e-06, 8.8636135453755427e-05],
            [2.0039076028057004e-03, 1.8593445905518494e-04],
            [9.2483147933239418e-05, -7.2424889892090862e-03],
            [-5.0858571546600860e-02, -1.0764404890003729e-04],
            [-3.4345200107376711e-05, 4.0923022931986089e-02],
            [-1.3053887892577887e-06, -6.3619602161869848e-05],
            [4.0059163278587107e-05, -1.8295292780121097e-06],
            [6.0516662005621363e-02, -5.3628132469741678e-03],
            [-6.3915637405469877e-06, -1.0851013913591723e-04]
        ],
        [
            [-6.7195320277007152e-18, 4.5751011362573354e-17],
            [-3.3190330281174369e-17, 1.8435442676186427e-17],
            [-3.1144640027600519e-17, -8.1211388909210858e-18],
            [3.6004693096917516e-03, -3.1902837701900527e-04],
            [3.7272531303366406e-04, -2.6783129628006955e-05],
            [-2.8225300281027252e-05, -3.3046108982919681e-04],
            [5.2501785997312311e-05, -6.3641548626609118e-06],
            [-3.0337662451992032e-04, 7.2413499727570080e-03],
            [1.2480184216333044e-02, 2.0994872839814535e-04],
            [1.3654045792664462e-04, -4.0915221525244461e-02],
            [-3.0990603819619038e-02, -6.1765692333619338e-05],
            [-7.3933436517481521e-04, 5.9166445895487453e-05],
            [6.0351571949801910e-05, 7.0073013389379546e-04],
            [-5.3629961135504556e-03, -6.0520029298724777e-02],
            [-2.6292245298661227e-04, 7.8746035202907627e-06]
        ],
        [
            [2.4480249182358735e-17, 4.2435015083875454e-17],
            [-3.6928700411533102e-17, 5.2067038281689024e-17],
            [-8.6745308493502914e-18, 3.4343479688412253e-17],
            [1.7010694813510141e-07, -3.5094299195501111e-06],
            [6.5549074071181729e-04, 5.9803641595923170e-03],
            [4.4359788279442949e-03, 3.0139479276096448e-04],
            [-1.0417817290404273e-03, -2.3959650644518375e-02],
            [3.1882453185069770e-05, -8.4302934737802426e-06],
            [4.3226019061197440e-07, -4.0985462101052195e-05],
            [-1.7370410001345954e-04, 1.4896302782535075e-05],
            [-1.5434513354971313e-05, -1.7021229228614612e-04],
            [-1.9425737288410808e-03, -4.0483807333425451e-02],
            [3.3361686220639772e-02, -1.6591321026634340e-03],
            [1.3838210951597207e-04, -6.3616431955470371e-06],
            [2.0032696124001261e-03, 4.6066171953997787e-02]
        ],
        [
            [-4.8709945663628144e-17, -3.5365197887865362e-18],
            [-6.8583356347177578e-17, -4.2827975042793416e-17],
            [-3.3819071364126838e-17, -2.5289934865715385e-17],
            [-3.3715492299406236e-05, 1.7499415605731067e-06],
            [1.6394162174838484e-02, -2.5359610388943441e-04],
            [-1.0038816593227126e-04, 5.9781573199233010e-03],
            [2.3960800159113827e-02, -1.0418569962580143e-03],
            [5.2725769259329841e-06, -2.0995262258391434e-06],
            [3.5121722005197753e-05, -3.1026184426751618e-06],
            [7.2151318291288624e-06, 2.9040694541140888e-06],
            [-1.5022764359278948e-04, 6.9911197734078001e-06],
            [-4.7608363978579427e-02, 1.6262501999962618e-03],
            [-1.3426490035426319e-03, -4.0485251012275257e-02],
            [8.7195816857697783e-06, 1.3586833507039930e-04],
            [-4.6062170139505212e-02, 2.0027916796534905e-03]
        ],
        [
            [-1.4782741326779699e-19, 3.4974591705482571e-17],
            [-3.6068946519198117e-17, 1.1328411847880617e-17],
            [2.6248824240355656e-17, 1.6114096904509761e-17],
            [-7.0915430706794402e-06, 8.2618156460601000e-07],
            [1.5437512415264615e-03, -1.2335943760667482e-04],
            [-1.2335784826906539e-04, -1.5435843216235756e-03],
            [-8.9371500060289252e-03, -2.0529623592150173e-05],
            [-1.8247396787463011e-06, -6.2429000129288204e-06],
            [-3.1342609795349807e-05, 6.8099606569289872e-08],
            [2.3509253022752606e-06, 3.7558286392258902e-05],
            [-3.9216235690028162e-05, 3.9327032920170738e-06],
            [-1.2943435600016861e-02, 1.1099054110970440e-03],
            [1.1099976254042966e-03, 1.2943296869971033e-02],
            [3.0529831788259939e-06, 1.7386142944034311e-05],
            [-2.1608449870307040e-03, -4.9227957559577415e-06]
        ],
        [
            [-2.1123457223237102e-17, -2.6939954115291541e-17],
            [-6.0078048296249552e-18, 1.2379869708983457e-17],
            [1.4128501922734102e-17, -5.0416991713534675e-17],
            [1.9127118359724076e-07, -9.7095044115778255e-07],
            [7.2526139983149912e-07, 2.2973209814848572e-05],
            [-7.9166391760074339e-06, 1.0268296658094700e-07],
            [1.1572996149974090e-05, 4.8430036600693175e-05],
            [-2.6422596150101878e-02, 2.8780349861862223e-03],
            [-2.8780771815579542e-03, -2.6423103860471912e-02],
            [7.0493483442931867e-02, -5.8819523028035593e-03],
            [5.8821914328950464e-03, 7.0492902128614407e-02],
            [-1.7999256603430658e-05, -3.5256560576490958e-04],
            [-7.7352253639723090e-05, 1.1210414476870573e-05],
            [-1.2885530409875470e-02, -2.9551295812585421e-05],
            [3.8571611770605859e-06, 8.3916665109958388e-05]
        ],
        [
            [6.1446632463843695e-08, 3.1567343028633564e-07],
            [-1.8744693215958750e-06, 1.4835064729597072e-07],
            [-1.3637381366724204e-02, -3.1305648274123679e-05],
            [-5.3905908611057267e-16, 1.7803568327881093e-17],
            [-5.2281718311714361e-17, 8.3185648566980177e-18],
            [7.7791342904696255e-18, 4.6546258134174491e-17],
            [-3.8262254752852817e-17, -2.2997018817090156e-17],
            [8.8725092807557649e-17, 2.4378099682652554e-16],
            [-2.6707515354023422e-16, 5.8455622430319489e-17],
            [-1.4466168438677793e-16, 8.1515632926989628e-17],
            [-3.5422066616887896e-16, -5.5426876815658915e-17],
            [1.0746324412494148e-16, -2.1430315134537333e-17],
            [7.7289111293124013e-18, -2.6785445217168154e-17],
            [1.0567632427529497e-16, -1.2138466744312306e-16],
            [4.8893620942583117e-17, 5.7754651244930237e-18]
        ],
        [
            [-1.0841657394385991e-17, 1.0180764749654401e-17],
            [5.8217503416591654e-17, -4.1311723012081671e-17],
            [-4.2677184810255710e-16, -3.6582551706786941e-17],
            [1.5120736879711157e-02, 3.4711511742654966e-05],
            [-3.4063635587777696e-06, 1.0333612671369757e-06],
            [-2.2721141403926615e-06, -1.0390147398370297e-05],
            [-3.6291991402311735e-05, 3.9256157681431814e-06],
            [-8.5023771259733379e-04, -7.8061978185007541e-03],
            [7.8056841955498370e-03, -8.5019195307052021e-04],
            [-5.8264533460171148e-04, -6.9754388596469912e-03],
            [6.9763532187570015e-03, -5.8274700695998705e-04],
            [-2.8895862805161654e-06, -1.8337672092480842e-07],
            [6.4067807430991108e-07, 1.4859185892855675e-05],
            [1.2517543234273824e-07, 2.5913761141256597e-09],
            [2.1358872874258084e-05, -3.7456894573517641e-07]
        ],
        [
            [-3.6003304352677840e-17, 3.7222237914610299e-17],
            [-1.0262196425164587e-16, -3.6908903899963320e-17],
            [4.0174654209679340e-17, 1.3975175679467678e-17],
            [-1.9527056955830971e-02, 1.5366810918475936e-03],
            [-2.7312666008388886e-05, 1.7823952198877183e-06],
            [1.5163323687384517e-06, 1.5064470741676939e-07],
            [-6.2618434525443800e-06, -3.2743133356136905e-07],
            [2.0783624565986255e-04, 4.4548277861119886e-03],
            [-5.0682197921906313e-03, 5.5755982631513698e-05],
            [-4.5688549353208215e-04, -5.3650133254125446e-02],
            [-4.5212080869416013e-02, 4.3702428899875662e-04],
            [2.2187111240712029e-04, -1.1870389950088475e-05],
            [-2.8000279811831938e-06, 6.9291882633722157e-05],
            [-3.7593884769572140e-03, -4.7773794786455132e-02],
            [1.6631908097225478e-05, -7.6616686782091104e-07]
        ],
        [
            [-1.2555151861076840e-16, -4.4264667846116255e-17],
            [7.6559350875723081e-17, -1.2271875973643359e-16],
            [-5.4662686484825625e-17, 9.9298547602941602e-18],
            [-1.5368299017244451e-03, -1.9528301486997163e-02],
            [2.4245125685579128e-07, -7.9994175528619316e-06],
            [2.2510357626480404e-05, 7.1222787056263170e-07],
            [-2.2627923565686492e-06, 3.1884711435104657e-05],
            [1.3977548758199270e-02, 9.9428697943552274e-05],
            [3.6325335140164651e-04, -4.4532861724384653e-03],
            [-6.2096830128600372e-02, 3.9826051990690670e-04],
            [3.7720129385179074e-04, 5.3651309402554567e-02],
            [-9.1655859749142053e-06, -2.0108636805315702e-04],
            [1.2071517018075878e-04, -6.0087499324356503e-07],
            [4.7767001269441653e-02, -3.7587410553603122e-03],
            [-5.0573640251322383e-08, -3.8965334120955042e-06]
        ],
        [
            [8.8779774067761586e-18, 7.9987401527232673e-17],
            [-1.1362356286977052e-16, 6.2111242494099898e-17],
            [1.1850954426631282e-17, 2.3771488544899174e-17],
            [4.2570594262839045e-06, 2.9699346381736880e-05],
            [-2.2528319329272184e-05, -2.6822175347484142e-02],
            [3.5698099923428400e-02, 1.2063120645217432e-04],
            [3.1486756436243726e-03, 3.3554276666032626e-02],
            [-1.2987363408115127e-06, 7.3529654797226118e-06],
            [7.0937240680430530e-06, 8.2953094591851906e-05],
            [-1.1679386950576298e-05, -8.3339415244652881e-06],
            [-8.9912856567758714e-07, 1.6131182717258469e-04],
            [-4.9126878383543623e-04, 4.2306983578112738e-02],
            [-3.1463846032292284e-02, -3.7921450498763776e-04],
            [-6.7910106997248754e-06, 7.9896119465315552e-07],
            [-4.6163367543073696e-03, -4.9196209107354977e-02]
        ],
        [
            [-9.1371849445093699e-17, 2.8817121929994532e-17],
            [-3.3521299506549767e-17, -9.9917067246894955e-17],
            [6.1944433015313657e-18, -2.4630744792694901e-17],
            [-5.5122353850405037e-06, 1.9997215134764934e-06],
            [1.7944742091668145e-02, 7.9891797752371443e-05],
            [-2.2288895890939641e-04, 2.6820646604667316e-02],
            [3.3555655279495937e-02, -3.1487839418062073e-03],
            [5.5154385294638175e-06, -1.9883286853927817e-06],
            [9.8556256016180122e-05, -1.0043094208134727e-05],
            [6.1391355359822347e-06, -7.4381568510917262e-05],
            [-2.1811742949396501e-04, -1.5623013120431156e-07],
            [-5.3156528249587706e-02, -4.2899841002145764e-04],
            [3.1746537266515770e-04, -4.2309823839802460e-02],
            [1.3450799904706628e-06, 2.2337629415870863e-05],
            [-4.9192778008768381e-02, 4.6160968159121279e-03]
        ],
        [
            [1.7537182940494979e-05, 2.4960175600520615e-01],
            [-9.2152753904794868e-02, 1.7409144179193377e-05],
            [8.7406020546969479e-07, 2.3553830955817838e-08],
            [2.6703601839140972e-17, -1.3956311479356804e-17],
            [-4.5557412001080802e-17, 3.5689992381350126e-17],
            [-6.3428089055131929e-17, 3.3177745730668415e-18],
            [-5.1602575585554793e-17, -1.3964243405773376e-17],
            [2.2514637623148325e-17, -5.3682417495373534e-17],
            [2.4275198453397685e-17, -4.8498809367780007e-18],
            [-2.7146894125061033e-17, 6.8785922965776893e-17],
            [5.9154173334244046e-17, 3.0649335753226153e-17],
            [5.9388544341773172e-17, -2.3399245669543643e-17],
            [4.5569476947121203e-17, 4.0203783632630663e-17],
            [5.3923147679138636e-17, 4.7152193844505523e-17],
            [3.8063649178088206e-17, 3.2643182272371116e-17]
        ],
        [
            [2.5721075900185020e-01, 1.7537182940491509e-05],
            [-1.7407918865145963e-05, 9.9764154034942848e-02],
            [-1.7052297330471047e-08, -1.5588695625589222e-06],
            [4.7365185141800725e-19, -4.2414703157474518e-17],
            [4.1864500804835808e-17, 2.4245163042601600e-17],
            [-3.7564531283309628e-18, 5.5309716817184373e-17],
            [4.7124274902313477e-17, 3.0618738723150088e-18],
            [-9.1219550089703115e-18, -3.2216215523897516e-17],
            [1.1885595292199298e-17, -2.3118139192977947e-17],
            [-4.7212793280756071e-17, -1.7929113637564196e-17],
            [-1.2300102560408291e-17, 3.9903610256102210e-17],
            [-9.0069918925151556e-17, 7.8982306509231256e-18],
            [-7.3004646284082392e-18, -1.1665872959989743e-17],
            [5.7955130367724465e-17, -1.5841923797104876e-17],
            [-6.4732854585917048e-17, -1.3641420964507075e-17]
        ]
    ];
    let spin_couplings: Array1<f64> = Array::zeros(12);

    let bp_fortran: Array3<f64> = get_apbv_fortran(
        &g0.view(),
        &g0lr.view(),
        &q_oo.view(),
        &q_vv.view(),
        &q_ov.view(),
        &omega.view(),
        &bs,
        12,
        15,
        15,
        2,
        1,
        spin_couplings.view(),
    );

    println!("bp fortran {}", bp_fortran.clone());
    assert!(bp_fortran.abs_diff_eq(&bp, 1e-12));
}
