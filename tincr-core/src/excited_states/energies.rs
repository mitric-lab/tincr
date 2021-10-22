use crate::initialization::Molecule;
use ndarray::{Array2, Array3, Array1, Array};
use crate::excited_states::transition_charges::trans_charges;
use crate::defaults;
use crate::excited_states::helpers::{get_orbital_en_diff, get_orbital_occ_diff};
use crate::excited_states::tda::tda;
use crate::excited_states::casida::casida;
use crate::calculator::lambda2_calc_oia;
use crate::excited_states::davidson::{hermitian_davidson, non_hermitian_davidson};

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

    let n_at: usize = molecule.atoms.len();

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


