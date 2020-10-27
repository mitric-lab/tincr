use crate::parameters::*;
use ndarray::prelude::*;
use std::collections::HashMap;
use peroxide::prelude::CubicSpline;

const SQRT3: f64 = 1.7320508075688772;
pub fn get_h0_and_s_mu_nu(
    skt: &SlaterKosterTable,
    li: i8,
    mi: i8,
    posi: ArrayView1<f64>,
    lj: i8,
    mj: i8,
    posj: ArrayView1<f64>,
) -> (f64, f64) {
    let (r, x, y, z): (f64, f64, f64, f64) = directional_cosines(posi, posj);
    let s: f64 = slako_transformation(r, x, y, z, &skt.s_spline, li, mi, lj, mj);
    let h: f64 = slako_transformation(r, x, y, z, &skt.h_spline, li, mi, lj, mj);
    return (s, h);
}

/// compute directional cosines for the vector going from
/// pos1 to pos2
/// Returns:
/// ========
/// r: length of vector
/// x,y,z: directional cosines
pub fn directional_cosines(pos1: ArrayView1<f64>, pos2: ArrayView1<f64>) -> (f64, f64, f64, f64) {
    let xc: f64 = pos2[0] - pos1[0];
    let yc: f64 = pos2[1] - pos1[1];
    let zc: f64 = pos2[2] - pos1[2];
    let r: f64 = (xc.powi(2) + yc.powi(2) + zc.powi(2)).sqrt();
    // directional cosines
    let x: f64;
    let y: f64;
    let z: f64;
    if r > 0.0 {
        x = xc / r;
        y = yc / r;
        z = zc / r;
    } else {
        x = 0.0;
        y = 0.0;
        z = 1.0;
    }
    return (r, x, y, z);
}

/// transformation rules for matrix elements
pub fn slako_transformation(
    r: f64,
    x: f64,
    y: f64,
    z: f64,
    s_or_h: &HashMap<u8, CubicSpline>,
    l1: i8,
    m1: i8,
    l2: i8,
    m2: i8,
) -> f64 {
    // x,y,z are directional cosines, r is the distance between the two centers
    // length of array sor_h
    // values of the N Slater-Koster tables for S or H0 evaluated at distance r
    // orbital qm numbers for center 1 and center 2
    // Local Variables

    // Result S(x,y,z) or H(x,y,z) after applying SK rules
    // index that encodes the tuple (l1,m1,l2,m2)

    // First we need to transform the tuple (l1,m1,l2,m2) into a unique integer
    // so that the compiler can build a branching table for each case.
    // Valid ranges for qm numbers: 0 <= l1,l2 <= lmax, -lmax <= m1,m2 <= lmax

    //transformation rules for matrix elements
    //# x,y,z are directional cosines, r is the distance between the two centers
    let value = match (l1, m1, l2, m2) {
        (0, 0, 0, 0) => s_or_h[&0].eval(r),
        (0, 0, 1, -1) => y * s_or_h[&2].eval(r),
        (0, 0, 1, 0) => z * s_or_h[&2].eval(r),
        (0, 0, 1, 1) => x * s_or_h[&2].eval(r),
        (0, 0, 2, -2) => x * y * s_or_h[&3].eval(r) * SQRT3,
        (0, 0, 2, -1) => y * z * s_or_h[&3].eval(r) * SQRT3,
        (0, 0, 2, 0) => -((x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&3].eval(r)) / 2.,
        (0, 0, 2, 1) => x * z * s_or_h[&3].eval(r) * SQRT3,
        (0, 0, 2, 2) => ((x - y) * (x + y) * s_or_h[&3].eval(r) * SQRT3) / 2.,
        (1, -1, 0, 0) => y * s_or_h[&4].eval(r),
        (1, -1, 1, -1) => {
            (x.powi(2) + z.powi(2)) * s_or_h[&5].eval(r) + y.powi(2) * s_or_h[&6].eval(r)
        }
        (1, -1, 1, 0) => y * z * (-s_or_h[&5].eval(r) + s_or_h[&6].eval(r)),
        (1, -1, 1, 1) => x * y * (-s_or_h[&5].eval(r) + s_or_h[&6].eval(r)),
        (1, -1, 2, -2) => {
            x * ((x.powi(2) - y.powi(2) + z.powi(2)) * s_or_h[&7].eval(r)
                + y.powi(2) * s_or_h[&8].eval(r) * SQRT3)
        }
        (1, -1, 2, -1) => {
            z * ((x.powi(2) - y.powi(2) + z.powi(2)) * s_or_h[&7].eval(r)
                + y.powi(2) * s_or_h[&8].eval(r) * SQRT3)
        }
        (1, -1, 2, 0) => {
            -(y * ((x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&8].eval(r)
                + 2.0 * z.powi(2) * s_or_h[&7].eval(r) * SQRT3))
                / 2.
        }
        (1, -1, 2, 1) => x * y * z * (-2.0 * s_or_h[&7].eval(r) + s_or_h[&8].eval(r) * SQRT3),
        (1, -1, 2, 2) => {
            -(y * (2.0 * (2.0 * x.powi(2) + z.powi(2)) * s_or_h[&7].eval(r)
                + (-x.powi(2) + y.powi(2)) * s_or_h[&8].eval(r) * SQRT3))
                / 2.
        }
        (1, 0, 0, 0) => z * s_or_h[&4].eval(r),
        (1, 0, 1, -1) => y * z * (-s_or_h[&5].eval(r) + s_or_h[&6].eval(r)),
        (1, 0, 1, 0) => (x.powi(2) + y.powi(2)) * s_or_h[&5].eval(r) + z.powi(2) * s_or_h[&6].eval(r),
        (1, 0, 1, 1) => x * z * (-s_or_h[&5].eval(r) + s_or_h[&6].eval(r)),
        (1, 0, 2, -2) => x * y * z * (-2.0 * s_or_h[&7].eval(r) + s_or_h[&8].eval(r) * SQRT3),
        (1, 0, 2, -1) => {
            y * ((x.powi(2) + y.powi(2) - z.powi(2)) * s_or_h[&7].eval(r)
                + z.powi(2) * s_or_h[&8].eval(r) * SQRT3)
        }
        (1, 0, 2, 0) => {
            z.powi(3) * s_or_h[&8].eval(r)
                - ((x.powi(2) + y.powi(2))
                    * z
                    * (s_or_h[&8].eval(r) - 2.0 * s_or_h[&7].eval(r) * SQRT3))
                    / 2.
        }
        (1, 0, 2, 1) => {
            x * ((x.powi(2) + y.powi(2) - z.powi(2)) * s_or_h[&7].eval(r)
                + z.powi(2) * s_or_h[&8].eval(r) * SQRT3)
        }
        (1, 0, 2, 2) => {
            -((x - y) * (x + y) * z * (2.0 * s_or_h[&7].eval(r) - s_or_h[&8].eval(r) * SQRT3)) / 2.
        }
        (1, 1, 0, 0) => x * s_or_h[&4].eval(r),
        (1, 1, 1, -1) => x * y * (-s_or_h[&5].eval(r) + s_or_h[&6].eval(r)),
        (1, 1, 1, 0) => x * z * (-s_or_h[&5].eval(r) + s_or_h[&6].eval(r)),
        (1, 1, 1, 1) => (y.powi(2) + z.powi(2)) * s_or_h[&5].eval(r) + x.powi(2) * s_or_h[&6].eval(r),
        (1, 1, 2, -2) => {
            y * ((-x.powi(2) + y.powi(2) + z.powi(2)) * s_or_h[&7].eval(r)
                + x.powi(2) * s_or_h[&8].eval(r) * SQRT3)
        }
        (1, 1, 2, -1) => x * y * z * (-2.0 * s_or_h[&7].eval(r) + s_or_h[&8].eval(r) * SQRT3),
        (1, 1, 2, 0) => {
            -(x * ((x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&8].eval(r)
                + 2.0 * z.powi(2) * s_or_h[&7].eval(r) * SQRT3))
                / 2.
        }
        (1, 1, 2, 1) => {
            z * ((-x.powi(2) + y.powi(2) + z.powi(2)) * s_or_h[&7].eval(r)
                + x.powi(2) * s_or_h[&8].eval(r) * SQRT3)
        }
        (1, 1, 2, 2) => {
            x * (2.0 * y.powi(2) + z.powi(2)) * s_or_h[&7].eval(r)
                + (x * (x - y) * (x + y) * s_or_h[&8].eval(r) * SQRT3) / 2.
        }
        (2, -2, 0, 0) => x * y * s_or_h[&9].eval(r) * SQRT3,
        (2, -2, 1, -1) => {
            x * ((x.powi(2) - y.powi(2) + z.powi(2)) * s_or_h[&10].eval(r)
                + y.powi(2) * s_or_h[&11].eval(r) * SQRT3)
        }
        (2, -2, 1, 0) => x * y * z * (-2.0 * s_or_h[&10].eval(r) + s_or_h[&11].eval(r) * SQRT3),
        (2, -2, 1, 1) => {
            y * ((-x.powi(2) + y.powi(2) + z.powi(2)) * s_or_h[&10].eval(r)
                + x.powi(2) * s_or_h[&11].eval(r) * SQRT3)
        }
        (2, -2, 2, -2) => {
            (x.powi(2) + z.powi(2)) * (y.powi(2) + z.powi(2)) * s_or_h[&12].eval(r)
                + ((x.powi(2) - y.powi(2)).powi(2) + (x.powi(2) + y.powi(2)) * z.powi(2))
                    * s_or_h[&13].eval(r)
                + 3.0 * x.powi(2) * y.powi(2) * s_or_h[&14].eval(r)
        }
        (2, -2, 2, -1) => {
            x * z
                * (-((x.powi(2) + z.powi(2)) * s_or_h[&12].eval(r))
                    + (x.powi(2) - 3.0 * y.powi(2) + z.powi(2)) * s_or_h[&13].eval(r)
                    + 3.0 * y.powi(2) * s_or_h[&14].eval(r))
        }
        (2, -2, 2, 0) => {
            (x * y
                * ((x.powi(2) + y.powi(2) + 2.0 * z.powi(2)) * s_or_h[&12].eval(r)
                    - 4.0 * z.powi(2) * s_or_h[&13].eval(r)
                    - (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&14].eval(r))
                * SQRT3)
                / 2.
        }
        (2, -2, 2, 1) => {
            y * z
                * (-((y.powi(2) + z.powi(2)) * (s_or_h[&12].eval(r) - s_or_h[&13].eval(r)))
                    + 3.0 * x.powi(2) * (-s_or_h[&13].eval(r) + s_or_h[&14].eval(r)))
        }
        (2, -2, 2, 2) => {
            (x * (x - y)
                * y
                * (x + y)
                * (s_or_h[&12].eval(r) - 4.0 * s_or_h[&13].eval(r) + 3.0 * s_or_h[&14].eval(r)))
                / 2.
        }
        (2, -1, 0, 0) => y * z * s_or_h[&9].eval(r) * SQRT3,
        (2, -1, 1, -1) => {
            z * ((x.powi(2) - y.powi(2) + z.powi(2)) * s_or_h[&10].eval(r)
                + y.powi(2) * s_or_h[&11].eval(r) * SQRT3)
        }
        (2, -1, 1, 0) => {
            y * ((x.powi(2) + y.powi(2) - z.powi(2)) * s_or_h[&10].eval(r)
                + z.powi(2) * s_or_h[&11].eval(r) * SQRT3)
        }
        (2, -1, 1, 1) => x * y * z * (-2.0 * s_or_h[&10].eval(r) + s_or_h[&11].eval(r) * SQRT3),
        (2, -1, 2, -2) => {
            x * z
                * (-((x.powi(2) + z.powi(2)) * s_or_h[&12].eval(r))
                    + (x.powi(2) - 3.0 * y.powi(2) + z.powi(2)) * s_or_h[&13].eval(r)
                    + 3.0 * y.powi(2) * s_or_h[&14].eval(r))
        }
        (2, -1, 2, -1) => {
            (x.powi(2) + y.powi(2)) * (x.powi(2) + z.powi(2)) * s_or_h[&12].eval(r)
                + ((y.powi(2) - z.powi(2)).powi(2) + x.powi(2) * (y.powi(2) + z.powi(2)))
                    * s_or_h[&13].eval(r)
                + 3.0 * y.powi(2) * z.powi(2) * s_or_h[&14].eval(r)
        }
        (2, -1, 2, 0) => {
            -(y * z
                * ((x.powi(2) + y.powi(2)) * s_or_h[&12].eval(r)
                    - 2.0 * (x.powi(2) + y.powi(2) - z.powi(2)) * s_or_h[&13].eval(r)
                    + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&14].eval(r))
                * SQRT3)
                / 2.
        }
        (2, -1, 2, 1) => {
            x * y
                * (-((x.powi(2) + y.powi(2)) * s_or_h[&12].eval(r))
                    + (x.powi(2) + y.powi(2) - 3.0 * z.powi(2)) * s_or_h[&13].eval(r)
                    + 3.0 * z.powi(2) * s_or_h[&14].eval(r))
        }
        (2, -1, 2, 2) => {
            (y * z
                * ((3.0 * x.powi(2) + y.powi(2) + 2.0 * z.powi(2)) * s_or_h[&12].eval(r)
                    - 2.0 * (3.0 * x.powi(2) - y.powi(2) + z.powi(2)) * s_or_h[&13].eval(r)
                    + 3.0 * (x - y) * (x + y) * s_or_h[&14].eval(r)))
                / 2.
        }
        (2, 0, 0, 0) => -((x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&9].eval(r)) / 2.,
        (2, 0, 1, -1) => {
            -(y * ((x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&11].eval(r)
                + 2.0 * z.powi(2) * s_or_h[&10].eval(r) * SQRT3))
                / 2.
        }
        (2, 0, 1, 0) => {
            z.powi(3) * s_or_h[&11].eval(r)
                - ((x.powi(2) + y.powi(2))
                    * z
                    * (s_or_h[&11].eval(r) - 2.0 * s_or_h[&10].eval(r) * SQRT3))
                    / 2.
        }
        (2, 0, 1, 1) => {
            -(x * ((x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&11].eval(r)
                + 2.0 * z.powi(2) * s_or_h[&10].eval(r) * SQRT3))
                / 2.
        }
        (2, 0, 2, -2) => {
            (x * y
                * ((x.powi(2) + y.powi(2) + 2.0 * z.powi(2)) * s_or_h[&12].eval(r)
                    - 4.0 * z.powi(2) * s_or_h[&13].eval(r)
                    - (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&14].eval(r))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, -1) => {
            -(y * z
                * ((x.powi(2) + y.powi(2)) * s_or_h[&12].eval(r)
                    - 2.0 * (x.powi(2) + y.powi(2) - z.powi(2)) * s_or_h[&13].eval(r)
                    + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&14].eval(r))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, 0) => {
            (3.0 * (x.powi(2) + y.powi(2))
                * ((x.powi(2) + y.powi(2)) * s_or_h[&12].eval(r)
                    + 4.0 * z.powi(2) * s_or_h[&13].eval(r))
                + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)).powi(2) * s_or_h[&14].eval(r))
                / 4.
        }
        (2, 0, 2, 1) => {
            -(x * z
                * ((x.powi(2) + y.powi(2)) * s_or_h[&12].eval(r)
                    - 2.0 * (x.powi(2) + y.powi(2) - z.powi(2)) * s_or_h[&13].eval(r)
                    + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&14].eval(r))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, 2) => {
            ((x - y)
                * (x + y)
                * ((x.powi(2) + y.powi(2) + 2.0 * z.powi(2)) * s_or_h[&12].eval(r)
                    - 4.0 * z.powi(2) * s_or_h[&13].eval(r)
                    - (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&14].eval(r))
                * SQRT3)
                / 4.
        }
        (2, 1, 0, 0) => x * z * s_or_h[&9].eval(r) * SQRT3,
        (2, 1, 1, -1) => x * y * z * (-2.0 * s_or_h[&10].eval(r) + s_or_h[&11].eval(r) * SQRT3),
        (2, 1, 1, 0) => {
            x * ((x.powi(2) + y.powi(2) - z.powi(2)) * s_or_h[&10].eval(r)
                + z.powi(2) * s_or_h[&11].eval(r) * SQRT3)
        }
        (2, 1, 1, 1) => {
            z * ((-x.powi(2) + y.powi(2) + z.powi(2)) * s_or_h[&10].eval(r)
                + x.powi(2) * s_or_h[&11].eval(r) * SQRT3)
        }
        (2, 1, 2, -2) => {
            y * z
                * (-((y.powi(2) + z.powi(2)) * (s_or_h[&12].eval(r) - s_or_h[&13].eval(r)))
                    + 3.0 * x.powi(2) * (-s_or_h[&13].eval(r) + s_or_h[&14].eval(r)))
        }
        (2, 1, 2, -1) => {
            x * y
                * (-((x.powi(2) + y.powi(2)) * s_or_h[&12].eval(r))
                    + (x.powi(2) + y.powi(2) - 3.0 * z.powi(2)) * s_or_h[&13].eval(r)
                    + 3.0 * z.powi(2) * s_or_h[&14].eval(r))
        }
        (2, 1, 2, 0) => {
            -(x * z
                * ((x.powi(2) + y.powi(2)) * s_or_h[&12].eval(r)
                    - 2.0 * (x.powi(2) + y.powi(2) - z.powi(2)) * s_or_h[&13].eval(r)
                    + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&14].eval(r))
                * SQRT3)
                / 2.
        }
        (2, 1, 2, 1) => {
            (x.powi(2) + y.powi(2)) * (y.powi(2) + z.powi(2)) * s_or_h[&12].eval(r)
                + (x.powi(4)
                    + x.powi(2) * (y.powi(2) - 2.0 * z.powi(2))
                    + z.powi(2) * (y.powi(2) + z.powi(2)))
                    * s_or_h[&13].eval(r)
                + 3.0 * x.powi(2) * z.powi(2) * s_or_h[&14].eval(r)
        }
        (2, 1, 2, 2) => {
            -(x * z
                * ((x.powi(2) + 3.0 * y.powi(2) + 2.0 * z.powi(2)) * s_or_h[&12].eval(r)
                    + 2.0 * (x.powi(2) - 3.0 * y.powi(2) - z.powi(2)) * s_or_h[&13].eval(r)
                    + 3.0 * (-x.powi(2) + y.powi(2)) * s_or_h[&14].eval(r)))
                / 2.
        }
        (2, 2, 0, 0) => ((x - y) * (x + y) * s_or_h[&9].eval(r) * SQRT3) / 2.,
        (2, 2, 1, -1) => {
            -(y * (2.0 * (2.0 * x.powi(2) + z.powi(2)) * s_or_h[&10].eval(r)
                + (-x.powi(2) + y.powi(2)) * s_or_h[&11].eval(r) * SQRT3))
                / 2.
        }
        (2, 2, 1, 0) => {
            -((x - y) * (x + y) * z * (2.0 * s_or_h[&10].eval(r) - s_or_h[&11].eval(r) * SQRT3)) / 2.
        }
        (2, 2, 1, 1) => {
            x * (2.0 * y.powi(2) + z.powi(2)) * s_or_h[&10].eval(r)
                + (x * (x - y) * (x + y) * s_or_h[&11].eval(r) * SQRT3) / 2.
        }
        (2, 2, 2, -2) => {
            (x * (x - y)
                * y
                * (x + y)
                * (s_or_h[&12].eval(r) - 4.0 * s_or_h[&13].eval(r) + 3.0 * s_or_h[&14].eval(r)))
                / 2.
        }
        (2, 2, 2, -1) => {
            (y * z
                * ((3.0 * x.powi(2) + y.powi(2) + 2.0 * z.powi(2)) * s_or_h[&12].eval(r)
                    - 2.0 * (3.0 * x.powi(2) - y.powi(2) + z.powi(2)) * s_or_h[&13].eval(r)
                    + 3.0 * (x - y) * (x + y) * s_or_h[&14].eval(r)))
                / 2.
        }
        (2, 2, 2, 0) => {
            ((x - y)
                * (x + y)
                * ((x.powi(2) + y.powi(2) + 2.0 * z.powi(2)) * s_or_h[&12].eval(r)
                    - 4.0 * z.powi(2) * s_or_h[&13].eval(r)
                    - (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)) * s_or_h[&14].eval(r))
                * SQRT3)
                / 4.
        }
        (2, 2, 2, 1) => {
            -(x * z
                * ((x.powi(2) + 3.0 * y.powi(2) + 2.0 * z.powi(2)) * s_or_h[&12].eval(r)
                    + 2.0 * (x.powi(2) - 3.0 * y.powi(2) - z.powi(2)) * s_or_h[&13].eval(r)
                    + 3.0 * (-x.powi(2) + y.powi(2)) * s_or_h[&14].eval(r)))
                / 2.
        }
        (2, 2, 2, 2) => {
            (((x.powi(2) - y.powi(2)).powi(2)
                + 4.0 * (x.powi(2) + y.powi(2)) * z.powi(2)
                + 4.0 * z.powi(4))
                * s_or_h[&12].eval(r)
                + 4.0
                    * (4.0 * x.powi(2) * y.powi(2) + (x.powi(2) + y.powi(2)) * z.powi(2))
                    * s_or_h[&13].eval(r)
                + 3.0 * (x.powi(2) - y.powi(2)).powi(2) * s_or_h[&14].eval(r))
                / 4.
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!")
    };
    return value;
}
