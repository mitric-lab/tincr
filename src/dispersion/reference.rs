use std::cmp;
use crate::dispersion::reference_inc::*;
use crate::dispersion::data::{get_effective_charge, get_hardness};

pub const MAX_NREF: usize = 7;
const MAX_CN: usize = 19;
pub const TRAPZD_POINTS: usize = 23;

// Get number of references for a given atomic number
pub fn get_nref(atomic_number: u8) -> usize {
    if atomic_number > 0 && atomic_number <= (REFN.len() as u8) {
        REFN[atomic_number as usize] as usize
    } else {
        0
    }
}

// Get the reference coordination numbers for an atomic number
pub fn get_refcn(atomic_number: u8) -> [f64; MAX_NREF] {
    // coordination number cn
    REFCOVCN[atomic_number as usize]
}

// Get the number of gaussian weights for an atomic number
pub fn get_refgw(atomic_number: u8) -> [usize; MAX_NREF] {
    let mut cnc: [usize; MAX_CN + 1] = [0; MAX_CN + 1];
    cnc[0] = 1;
    let mut ngw: [usize; MAX_NREF] = [0; MAX_NREF];

    let ref_: usize = get_nref(atomic_number);

    for ir in 0..ref_ {
        let icn = cmp::min(REFCN[atomic_number as usize][ir].round() as usize, MAX_CN);
        cnc[icn] = cnc[icn] + 1;
    }

    for ir in 0..ref_ {
        let icn = cnc[cmp::min(REFCN[atomic_number as usize][ir].round() as usize, MAX_CN)];
        ngw[ir] = icn*(icn + 1)/2;
    }
    return ngw;
}

// Get the reference partial charges for an atomic number
pub fn get_refq(atomic_number: u8) -> [f64; MAX_NREF] {
    // partial charge q
    CLSQ[atomic_number as usize]
}

// Get the reference polarizability for an atomic number
pub fn get_refalpha(ga: f64, gc: f64, atomic_number: u8) -> [[f64; TRAPZD_POINTS]; MAX_NREF] {
    // \alpha^{A,ref}(iw) = (1/m) * [\alpha^{AmXn}(iw) - (n/l)*\alpha{X_l}(iw)*\zeta(z^X, z^{X,ref})]
    let mut alpha: [[f64; TRAPZD_POINTS]; MAX_NREF] = [[0.0; TRAPZD_POINTS]; MAX_NREF];

    let ref_: usize = get_nref(atomic_number);
    let atomic_number_usize = atomic_number as usize;

    for ir in 0..ref_ {
        let is = REFSYS[atomic_number_usize][ir];  // Atomic number in the reference system
        let iz = get_effective_charge(is);
        // aiw = sscale(is)*secaiw(:, is) * zeta(ga, get_hardness(is)*gc, iz, clsh(ir, num)+iz)
        let aiw: [f64; TRAPZD_POINTS] = array_scalar_mult(SECAIW[is as usize],
                                                          SSCALE[is as usize] * zeta(
                                                              ga,
                                                              get_hardness(is)*gc,
                                                              iz,
                                                              CLSH[atomic_number_usize][ir] + iz));
        // alpha(:, ir) = max(ascale(ir, num)*(alphaiw(:, ir, num) - hcount(ir, num)*aiw), 0.0_wp)
        alpha[ir] = array_scalar_comp(
            array_scalar_mult(
                array_substraction(
                    ALPHAIW[atomic_number_usize][ir],
                    array_scalar_mult(aiw, HCOUNT[atomic_number_usize][ir])
                ), ASCALE[atomic_number_usize][ir]),
        0.0);
    }
    return alpha;
}

// Charge scaling function
pub fn zeta(a: f64, c: f64, qref: f64, qmod: f64) -> f64 {
    if qmod < 0.0 {
        a.exp()
    } else {
        (a * (1.0 - (c * (1.0 - qref/qmod)).exp())).exp()
    }
}

fn new_aiw_from<F: Iterator<Item=f64>>(src: F) -> [f64; TRAPZD_POINTS] {
    let mut result = [0.0; TRAPZD_POINTS];
    for (rref, val) in result.iter_mut().zip(src) {
        *rref = val;
    }
    result
}

fn array_scalar_mult(a: [f64; TRAPZD_POINTS], scalar: f64) -> [f64; TRAPZD_POINTS] {
    new_aiw_from(a.iter().map(|a| a * scalar))
}

fn array_substraction(a: [f64; TRAPZD_POINTS], b: [f64; TRAPZD_POINTS]) -> [f64; TRAPZD_POINTS] {
    new_aiw_from(a.iter().zip(&b).map(|(a, b)| a - b))
}

fn array_scalar_comp(a: [f64; TRAPZD_POINTS], scalar: f64) -> [f64; TRAPZD_POINTS] {
    new_aiw_from(a.iter().map(|a| a.max(scalar)))
}