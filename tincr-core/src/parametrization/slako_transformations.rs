use crate::{AtomicOrbital, Spline};
use hashbrown::HashMap;
use nalgebra::Vector4;
use ndarray::prelude::*;

const SQRT3: f64 = 1.7320508075688772;

/// transformation rules for matrix elements
pub fn slako_transformation(
    rcos: &Vector4<f64>,
    s_or_h: &HashMap<u8, Spline>,
    orb_a: &AtomicOrbital,
    orb_b: &AtomicOrbital,
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
    let value = match (orb_a.l, orb_a.m, orb_b.l, orb_b.m) {
        (0, 0, 0, 0) => s_or_h[&0].eval(rcos.w),
        (0, 0, 1, -1) => rcos.y * s_or_h[&2].eval(rcos.w),
        (0, 0, 1, 0) => rcos.z * s_or_h[&2].eval(rcos.w),
        (0, 0, 1, 1) => rcos.x * s_or_h[&2].eval(rcos.w),
        (0, 0, 2, -2) => rcos.x * rcos.y * s_or_h[&3].eval(rcos.w) * SQRT3,
        (0, 0, 2, -1) => rcos.y * rcos.z * s_or_h[&3].eval(rcos.w) * SQRT3,
        (0, 0, 2, 0) => {
            -((rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)) * s_or_h[&3].eval(rcos.w))
                / 2.
        }
        (0, 0, 2, 1) => rcos.x * rcos.z * s_or_h[&3].eval(rcos.w) * SQRT3,
        (0, 0, 2, 2) => {
            ((rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&3].eval(rcos.w) * SQRT3) / 2.
        }
        (1, -1, 0, 0) => rcos.y * s_or_h[&4].eval(rcos.w),
        (1, -1, 1, -1) => {
            (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                + rcos.y.powi(2) * s_or_h[&6].eval(rcos.w)
        }
        (1, -1, 1, 0) => rcos.y * rcos.z * (-s_or_h[&5].eval(rcos.w) + s_or_h[&6].eval(rcos.w)),
        (1, -1, 1, 1) => rcos.x * rcos.y * (-s_or_h[&5].eval(rcos.w) + s_or_h[&6].eval(rcos.w)),
        (1, -1, 2, -2) => {
            rcos.x
                * ((rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + rcos.y.powi(2) * s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, -1, 2, -1) => {
            rcos.z
                * ((rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + rcos.y.powi(2) * s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, -1, 2, 0) => {
            -(rcos.x
                * ((rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&8].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&7].eval(rcos.w) * SQRT3))
                / 2.
        }
        (1, -1, 2, 1) => {
            rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&7].eval(rcos.w) + s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, -1, 2, 2) => {
            -(rcos.x
                * (2.0 * (2.0 * rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + (-rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w) * SQRT3))
                / 2.
        }
        (1, 0, 0, 0) => rcos.z * s_or_h[&4].eval(rcos.w),
        (1, 0, 1, -1) => rcos.y * rcos.z * (-s_or_h[&5].eval(rcos.w) + s_or_h[&6].eval(rcos.w)),
        (1, 0, 1, 0) => {
            (rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&5].eval(rcos.w)
                + rcos.z.powi(2) * s_or_h[&6].eval(rcos.w)
        }
        (1, 0, 1, 1) => rcos.x * rcos.z * (-s_or_h[&5].eval(rcos.w) + s_or_h[&6].eval(rcos.w)),
        (1, 0, 2, -2) => {
            rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&7].eval(rcos.w) + s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, 0, 2, -1) => {
            rcos.y
                * ((rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + rcos.z.powi(2) * s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, 0, 2, 0) => {
            rcos.x.powi(3) * s_or_h[&8].eval(rcos.w)
                - ((rcos.x.powi(2) + rcos.y.powi(2))
                    * rcos.z
                    * (s_or_h[&8].eval(rcos.w) - 2.0 * s_or_h[&7].eval(rcos.w) * SQRT3))
                    / 2.
        }
        (1, 0, 2, 1) => {
            rcos.x
                * ((rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + rcos.z.powi(2) * s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, 0, 2, 2) => {
            -((rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * rcos.z
                * (2.0 * s_or_h[&7].eval(rcos.w) - s_or_h[&8].eval(rcos.w) * SQRT3))
                / 2.
        }
        (1, 1, 0, 0) => rcos.x * s_or_h[&4].eval(rcos.w),
        (1, 1, 1, -1) => rcos.x * rcos.y * (-s_or_h[&5].eval(rcos.w) + s_or_h[&6].eval(rcos.w)),
        (1, 1, 1, 0) => rcos.x * rcos.z * (-s_or_h[&5].eval(rcos.w) + s_or_h[&6].eval(rcos.w)),
        (1, 1, 1, 1) => {
            (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                + rcos.x.powi(2) * s_or_h[&6].eval(rcos.w)
        }
        (1, 1, 2, -2) => {
            rcos.y
                * ((-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + rcos.x.powi(2) * s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, 1, 2, -1) => {
            rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&7].eval(rcos.w) + s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, 1, 2, 0) => {
            -(rcos.x
                * ((rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&8].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&7].eval(rcos.w) * SQRT3))
                / 2.
        }
        (1, 1, 2, 1) => {
            rcos.z
                * ((-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + rcos.x.powi(2) * s_or_h[&8].eval(rcos.w) * SQRT3)
        }
        (1, 1, 2, 2) => {
            rcos.x * (2.0 * rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                + (rcos.x * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&8].eval(rcos.w) * SQRT3)
                    / 2.
        }
        (2, -2, 0, 0) => rcos.x * rcos.y * s_or_h[&9].eval(rcos.w) * SQRT3,
        (2, -2, 1, -1) => {
            rcos.x
                * ((rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + rcos.y.powi(2) * s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, -2, 1, 0) => {
            rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&10].eval(rcos.w) + s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, -2, 1, 1) => {
            rcos.y
                * ((-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + rcos.x.powi(2) * s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, -2, 2, -2) => {
            (rcos.x.powi(2) + rcos.z.powi(2))
                * (rcos.x.powi(2) + rcos.z.powi(2))
                * s_or_h[&12].eval(rcos.w)
                + ((rcos.x.powi(2) - rcos.y.powi(2)).powi(2)
                    + (rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2))
                    * s_or_h[&13].eval(rcos.w)
                + 3.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
        }
        (2, -2, 2, -1) => {
            rcos.x
                * rcos.z
                * (-((rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].eval(rcos.w))
                    + (rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w))
        }
        (2, -2, 2, 0) => {
            (rcos.x
                * rcos.y
                * ((rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 4.0 * rcos.z.powi(2) * s_or_h[&13].eval(rcos.w)
                    - (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].eval(rcos.w))
                * SQRT3)
                / 2.
        }
        (2, -2, 2, 1) => {
            rcos.y
                * rcos.z
                * (-((rcos.x.powi(2) + rcos.z.powi(2))
                    * (s_or_h[&12].eval(rcos.w) - s_or_h[&13].eval(rcos.w)))
                    + 3.0 * rcos.x.powi(2) * (-s_or_h[&13].eval(rcos.w) + s_or_h[&14].eval(rcos.w)))
        }
        (2, -2, 2, 2) => {
            (rcos.x
                * (rcos.x - rcos.y)
                * rcos.y
                * (rcos.x + rcos.y)
                * (s_or_h[&12].eval(rcos.w) - 4.0 * s_or_h[&13].eval(rcos.w)
                    + 3.0 * s_or_h[&14].eval(rcos.w)))
                / 2.
        }
        (2, -1, 0, 0) => rcos.y * rcos.z * s_or_h[&9].eval(rcos.w) * SQRT3,
        (2, -1, 1, -1) => {
            rcos.z
                * ((rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + rcos.y.powi(2) * s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, -1, 1, 0) => {
            rcos.y
                * ((rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + rcos.z.powi(2) * s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, -1, 1, 1) => {
            rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&10].eval(rcos.w) + s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, -1, 2, -2) => {
            rcos.x
                * rcos.z
                * (-((rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].eval(rcos.w))
                    + (rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w))
        }
        (2, -1, 2, -1) => {
            (rcos.x.powi(2) + rcos.y.powi(2))
                * (rcos.x.powi(2) + rcos.z.powi(2))
                * s_or_h[&12].eval(rcos.w)
                + ((rcos.x.powi(2) - rcos.z.powi(2)).powi(2)
                    + rcos.x.powi(2) * (rcos.x.powi(2) + rcos.z.powi(2)))
                    * s_or_h[&13].eval(rcos.w)
                + 3.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
        }
        (2, -1, 2, 0) => {
            -(rcos.x
                * rcos.z
                * ((rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].eval(rcos.w))
                * SQRT3)
                / 2.
        }
        (2, -1, 2, 1) => {
            rcos.x
                * rcos.y
                * (-((rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w))
                    + (rcos.x.powi(2) + rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w))
        }
        (2, -1, 2, 2) => {
            (rcos.x
                * rcos.z
                * ((3.0 * rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (3.0 * rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&14].eval(rcos.w)))
                / 2.
        }
        (2, 0, 0, 0) => {
            -((rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)) * s_or_h[&9].eval(rcos.w))
                / 2.
        }
        (2, 0, 1, -1) => {
            -(rcos.x
                * ((rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&11].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&10].eval(rcos.w) * SQRT3))
                / 2.
        }
        (2, 0, 1, 0) => {
            rcos.x.powi(3) * s_or_h[&11].eval(rcos.w)
                - ((rcos.x.powi(2) + rcos.y.powi(2))
                    * rcos.z
                    * (s_or_h[&11].eval(rcos.w) - 2.0 * s_or_h[&10].eval(rcos.w) * SQRT3))
                    / 2.
        }
        (2, 0, 1, 1) => {
            -(rcos.x
                * ((rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&11].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&10].eval(rcos.w) * SQRT3))
                / 2.
        }
        (2, 0, 2, -2) => {
            (rcos.x
                * rcos.y
                * ((rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 4.0 * rcos.z.powi(2) * s_or_h[&13].eval(rcos.w)
                    - (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].eval(rcos.w))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, -1) => {
            -(rcos.x
                * rcos.z
                * ((rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].eval(rcos.w))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, 0) => {
            (3.0 * (rcos.x.powi(2) + rcos.y.powi(2))
                * ((rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    + 4.0 * rcos.z.powi(2) * s_or_h[&13].eval(rcos.w))
                + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)).powi(2)
                    * s_or_h[&14].eval(rcos.w))
                / 4.
        }
        (2, 0, 2, 1) => {
            -(rcos.x
                * rcos.z
                * ((rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].eval(rcos.w))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, 2) => {
            ((rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * ((rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 4.0 * rcos.z.powi(2) * s_or_h[&13].eval(rcos.w)
                    - (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].eval(rcos.w))
                * SQRT3)
                / 4.
        }
        (2, 1, 0, 0) => rcos.x * rcos.z * s_or_h[&9].eval(rcos.w) * SQRT3,
        (2, 1, 1, -1) => {
            rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&10].eval(rcos.w) + s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, 1, 1, 0) => {
            rcos.x
                * ((rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + rcos.z.powi(2) * s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, 1, 1, 1) => {
            rcos.z
                * ((-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + rcos.x.powi(2) * s_or_h[&11].eval(rcos.w) * SQRT3)
        }
        (2, 1, 2, -2) => {
            rcos.y
                * rcos.z
                * (-((rcos.x.powi(2) + rcos.z.powi(2))
                    * (s_or_h[&12].eval(rcos.w) - s_or_h[&13].eval(rcos.w)))
                    + 3.0 * rcos.x.powi(2) * (-s_or_h[&13].eval(rcos.w) + s_or_h[&14].eval(rcos.w)))
        }
        (2, 1, 2, -1) => {
            rcos.x
                * rcos.y
                * (-((rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w))
                    + (rcos.x.powi(2) + rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w))
        }
        (2, 1, 2, 0) => {
            -(rcos.x
                * rcos.z
                * ((rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].eval(rcos.w))
                * SQRT3)
                / 2.
        }
        (2, 1, 2, 1) => {
            (rcos.x.powi(2) + rcos.y.powi(2))
                * (rcos.x.powi(2) + rcos.z.powi(2))
                * s_or_h[&12].eval(rcos.w)
                + (rcos.x.powi(4)
                    + rcos.x.powi(2) * (rcos.x.powi(2) - 2.0 * rcos.z.powi(2))
                    + rcos.z.powi(2) * (rcos.x.powi(2) + rcos.z.powi(2)))
                    * s_or_h[&13].eval(rcos.w)
                + 3.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
        }
        (2, 1, 2, 2) => {
            -(rcos.x
                * rcos.z
                * ((rcos.x.powi(2) + 3.0 * rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * (-rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&14].eval(rcos.w)))
                / 2.
        }
        (2, 2, 0, 0) => {
            ((rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&9].eval(rcos.w) * SQRT3) / 2.
        }
        (2, 2, 1, -1) => {
            -(rcos.x
                * (2.0 * (2.0 * rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + (-rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w) * SQRT3))
                / 2.
        }
        (2, 2, 1, 0) => {
            -((rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * rcos.z
                * (2.0 * s_or_h[&10].eval(rcos.w) - s_or_h[&11].eval(rcos.w) * SQRT3))
                / 2.
        }
        (2, 2, 1, 1) => {
            rcos.x * (2.0 * rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                + (rcos.x
                    * (rcos.x - rcos.y)
                    * (rcos.x + rcos.y)
                    * s_or_h[&11].eval(rcos.w)
                    * SQRT3)
                    / 2.
        }
        (2, 2, 2, -2) => {
            (rcos.x
                * (rcos.x - rcos.y)
                * rcos.y
                * (rcos.x + rcos.y)
                * (s_or_h[&12].eval(rcos.w) - 4.0 * s_or_h[&13].eval(rcos.w)
                    + 3.0 * s_or_h[&14].eval(rcos.w)))
                / 2.
        }
        (2, 2, 2, -1) => {
            (rcos.x
                * rcos.z
                * ((3.0 * rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (3.0 * rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&14].eval(rcos.w)))
                / 2.
        }
        (2, 2, 2, 0) => {
            ((rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * ((rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 4.0 * rcos.z.powi(2) * s_or_h[&13].eval(rcos.w)
                    - (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].eval(rcos.w))
                * SQRT3)
                / 4.
        }
        (2, 2, 2, 1) => {
            -(rcos.x
                * rcos.z
                * ((rcos.x.powi(2) + 3.0 * rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * (-rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&14].eval(rcos.w)))
                / 2.
        }
        (2, 2, 2, 2) => {
            (((rcos.x.powi(2) - rcos.y.powi(2)).powi(2)
                + 4.0 * (rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2)
                + 4.0 * rcos.z.powi(4))
                * s_or_h[&12].eval(rcos.w)
                + 4.0
                    * (4.0 * rcos.x.powi(2) * rcos.y.powi(2)
                        + (rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2))
                    * s_or_h[&13].eval(rcos.w)
                + 3.0 * (rcos.x.powi(2) - rcos.y.powi(2)).powi(2) * s_or_h[&14].eval(rcos.w))
                / 4.
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!"),
    };
    value
}

/// transformation rules for matrircos.x elements
pub fn slako_transformation_gradients(
    rcos: &Vector4<f64>,
    s_or_h: &HashMap<u8, Spline>,
    orb_a: &AtomicOrbital,
    orb_b: &AtomicOrbital,
) -> Array1<f64> {
    // TODO: The splines are evaulated multiple times at the same position. This is an unnecessarrcos.x
    // load and could be implemented in a more efficient way

    // rcos.x,rcos.x,rcos.x are directional cosines, r is the distance between the two centers
    // length of array sor_h
    // values of the N Slater-Koster tables for S or H0 evaluated at distance r
    // orbital qm numbers for center 1 and center 2
    // Local Variables

    // rcos.wesult S(rcos.x,rcos.x,rcos.x) or H(rcos.x,rcos.x,rcos.x) after applrcos.xing SK rules
    // indercos.x that encodes the tuple (l1,m1,l2,m2)

    // First we need to transform the tuple (l1,m1,l2,m2) into a unique integer
    // so that the compiler can build a branching table for each case.
    // Valid ranges for qm numbers: 0 <= l1,l2 <= lmarcos.x, -lmarcos.x <= m1,m2 <= lmarcos.x

    //transformation rules for matrircos.x elements
    //# rcos.x,rcos.x,rcos.x are directional cosines, r is the distance between the two centers
    let grad0: f64 = match (orb_a.l, orb_a.m, orb_b.l, orb_b.m) {
        (0, 0, 0, 0) => rcos.x * s_or_h[&0].deriv(rcos.w),
        (0, 0, 1, -1) => {
            rcos.x * rcos.y * (-(s_or_h[&2].eval(rcos.w) / rcos.w) + s_or_h[&2].deriv(rcos.w))
        }
        (0, 0, 1, 0) => {
            rcos.x * rcos.z * (-(s_or_h[&2].eval(rcos.w) / rcos.w) + s_or_h[&2].deriv(rcos.w))
        }
        (0, 0, 1, 1) => {
            -(((-1.0 + rcos.x.powi(2)) * s_or_h[&2].eval(rcos.w)) / rcos.w)
                + rcos.x.powi(2) * s_or_h[&2].deriv(rcos.w)
        }
        (0, 0, 2, -2) => {
            (SQRT3
                * rcos.y
                * ((1.0 - 2.0 * rcos.x.powi(2)) * s_or_h[&3].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, -1) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&3].eval(rcos.w) + rcos.w * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, 0) => {
            (rcos.x
                * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                * s_or_h[&3].eval(rcos.w))
                / rcos.w
                - (rcos.x
                    * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&3].deriv(rcos.w))
                    / 2.
        }
        (0, 0, 2, 1) => {
            (SQRT3
                * rcos.z
                * ((1.0 - 2.0 * rcos.x.powi(2)) * s_or_h[&3].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, 2) => {
            (SQRT3
                * rcos.x
                * (2.0 * (1.0 - rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&3].eval(rcos.w)
                    + rcos.w * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&3].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, -1, 0, 0) => {
            rcos.x * rcos.y * (-(s_or_h[&4].eval(rcos.w) / rcos.w) + s_or_h[&4].deriv(rcos.w))
        }
        (1, -1, 1, -1) => {
            (rcos.x
                * (-2.0 * (-1.0 + rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + rcos.w * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&5].deriv(rcos.w)
                    + rcos.y.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w) + rcos.w * s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 1, 0) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (2.0 * s_or_h[&5].eval(rcos.w) - 2.0 * s_or_h[&6].eval(rcos.w)
                    + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 1, 1) => {
            (rcos.x
                * ((-1.0 + 2.0 * rcos.x.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.x.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, -1, 2, -2) => {
            ((-rcos.x.powi(2) + rcos.z.powi(2)
                - 3.0 * rcos.x.powi(2) * (-1.0 + rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2)))
                * s_or_h[&7].eval(rcos.w)
                + rcos.w
                    * rcos.x.powi(2)
                    * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                    * s_or_h[&7].deriv(rcos.w)
                + SQRT3
                    * rcos.y.powi(2)
                    * ((1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&8].eval(rcos.w)
                        + rcos.w * rcos.x.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, -1, 2, -1) => {
            (rcos.x
                * rcos.z
                * ((2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + SQRT3
                        * rcos.y.powi(2)
                        * (-3.0 * s_or_h[&8].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 2, 0) => {
            -(rcos.x
                * rcos.y
                * ((2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + 2.0
                        * rcos.z.powi(2)
                        * (-3.0 * SQRT3 * s_or_h[&7].eval(rcos.w)
                            + 3.0 * s_or_h[&8].eval(rcos.w)
                            + rcos.w * SQRT3 * s_or_h[&8].deriv(rcos.w))
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, -1, 2, 1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.x.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.x.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 2, 2) => {
            (rcos.x
                * rcos.y
                * (2.0
                    * (-4.0 + 6.0 * rcos.x.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + SQRT3
                        * (2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                        * s_or_h[&8].eval(rcos.w)
                    - 2.0
                        * rcos.w
                        * (2.0 * rcos.x.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w
                        * SQRT3
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, 0, 0, 0) => {
            rcos.x * rcos.z * (-(s_or_h[&4].eval(rcos.w) / rcos.w) + s_or_h[&4].deriv(rcos.w))
        }
        (1, 0, 1, -1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (2.0 * s_or_h[&5].eval(rcos.w) - 2.0 * s_or_h[&6].eval(rcos.w)
                    + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 1, 0) => {
            (rcos.x
                * (-2.0 * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + rcos.w * (rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&5].deriv(rcos.w)
                    + rcos.z.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w) + rcos.w * s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 1, 1) => {
            (rcos.z
                * ((-1.0 + 2.0 * rcos.x.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.x.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, 0, 2, -2) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.x.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.x.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 2, -1) => {
            (rcos.x
                * rcos.y
                * ((2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + SQRT3
                        * rcos.z.powi(2)
                        * (-3.0 * s_or_h[&8].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 2, 0) => {
            -(rcos.x
                * rcos.z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + (2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 6.0 * rcos.z.powi(2))
                        * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * (-2.0
                            * SQRT3
                            * (rcos.x.powi(2) + rcos.y.powi(2))
                            * s_or_h[&7].deriv(rcos.w)
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                * s_or_h[&8].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (1, 0, 2, 1) => {
            (((rcos.x - rcos.z) * (rcos.x + rcos.z)
                - 3.0 * rcos.x.powi(2) * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2)))
                * s_or_h[&7].eval(rcos.w)
                + rcos.w
                    * rcos.x.powi(2)
                    * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                    * s_or_h[&7].deriv(rcos.w)
                + SQRT3
                    * rcos.z.powi(2)
                    * ((1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&8].eval(rcos.w)
                        + rcos.w * rcos.x.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, 0, 2, 2) => {
            (rcos.x
                * rcos.z
                * ((-4.0 + 6.0 * rcos.x.powi(2) - 6.0 * rcos.y.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + 2.0 * SQRT3 * s_or_h[&8].eval(rcos.w)
                    - (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * (3.0 * SQRT3 * s_or_h[&8].eval(rcos.w)
                            + 2.0 * rcos.w * s_or_h[&8].deriv(rcos.w)
                            - rcos.w * SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (1, 1, 0, 0) => {
            -(((-1.0 + rcos.x.powi(2)) * s_or_h[&4].eval(rcos.w)) / rcos.w)
                + rcos.x.powi(2) * s_or_h[&4].deriv(rcos.w)
        }
        (1, 1, 1, -1) => {
            (rcos.x
                * ((-1.0 + 2.0 * rcos.x.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.x.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, 1, 1, 0) => {
            (rcos.z
                * ((-1.0 + 2.0 * rcos.x.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.x.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, 1, 1, 1) => {
            (rcos.x
                * (-2.0 * (-1.0 + rcos.x.powi(2)) * s_or_h[&6].eval(rcos.w)
                    - (rcos.x.powi(2) + rcos.z.powi(2))
                        * (2.0 * s_or_h[&5].eval(rcos.w) - rcos.w * s_or_h[&5].deriv(rcos.w))
                    + rcos.w * rcos.x.powi(2) * s_or_h[&6].deriv(rcos.w)))
                / rcos.w
        }
        (1, 1, 2, -2) => {
            (rcos.x
                * rcos.y
                * ((-2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (2.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.x.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, 1, 2, -1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.x.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.x.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 2, 0) => {
            -(2.0 * SQRT3 * (1.0 - 3.0 * rcos.x.powi(2)) * rcos.z.powi(2) * s_or_h[&7].eval(rcos.w)
                + (rcos.x.powi(2)
                    - 2.0 * rcos.z.powi(2)
                    - 3.0
                        * rcos.x.powi(2)
                        * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)))
                    * s_or_h[&8].eval(rcos.w)
                + rcos.w
                    * rcos.x.powi(2)
                    * (2.0 * SQRT3 * rcos.z.powi(2) * s_or_h[&7].deriv(rcos.w)
                        + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                            * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, 1, 2, 1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (2.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.x.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, 1, 2, 2) => {
            (-2.0
                * (-1.0 + 3.0 * rcos.x.powi(2))
                * (2.0 * rcos.y.powi(2) + rcos.z.powi(2))
                * s_or_h[&7].eval(rcos.w)
                - SQRT3
                    * (3.0 * rcos.x.powi(4) + rcos.y.powi(2)
                        - 3.0 * rcos.x.powi(2) * (1.0 + rcos.y.powi(2)))
                    * s_or_h[&8].eval(rcos.w)
                + rcos.w
                    * rcos.x.powi(2)
                    * (2.0 * (2.0 * rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&7].deriv(rcos.w)
                        + SQRT3 * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -2, 0, 0) => {
            (SQRT3
                * rcos.y
                * ((1.0 - 2.0 * rcos.x.powi(2)) * s_or_h[&9].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 1, -1) => {
            ((-rcos.x.powi(2) + rcos.z.powi(2)
                - 3.0 * rcos.x.powi(2) * (-1.0 + rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2)))
                * s_or_h[&10].eval(rcos.w)
                + rcos.w
                    * rcos.x.powi(2)
                    * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                    * s_or_h[&10].deriv(rcos.w)
                + SQRT3
                    * rcos.y.powi(2)
                    * ((1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&11].eval(rcos.w)
                        + rcos.w * rcos.x.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 1, 0) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.x.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.x.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -2, 1, 1) => {
            (rcos.x
                * rcos.y
                * ((-2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (2.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.x.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, -2) => {
            (rcos.x
                * (-2.0
                    * (rcos.x.powi(2) + rcos.z.powi(2))
                    * (-1.0 + 2.0 * rcos.x.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (-2.0 * rcos.x.powi(4)
                            + rcos.z.powi(2)
                            + rcos.x.powi(2)
                                * (2.0 + 4.0 * rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                            - 2.0 * rcos.y.powi(2) * (1.0 + rcos.y.powi(2) + rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, -1) => {
            (rcos.z
                * ((4.0 * rcos.x.powi(4) - rcos.z.powi(2)
                    + rcos.x.powi(2) * (-3.0 + 4.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + (-3.0 * rcos.y.powi(2)
                        + rcos.z.powi(2)
                        + rcos.x.powi(2)
                            * (3.0 - 4.0 * rcos.x.powi(2) + 12.0 * rcos.y.powi(2)
                                - 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, 0) => {
            (SQRT3
                * rcos.y
                * ((-4.0 * rcos.x.powi(4)
                    + rcos.y.powi(2)
                    + 2.0 * rcos.z.powi(2)
                    + rcos.x.powi(2) * (3.0 - 4.0 * rcos.y.powi(2) - 8.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (-1.0 + 4.0 * rcos.x.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    - 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.x.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -2, 2, 1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (4.0 * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 4.0 * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&13].eval(rcos.w)
                    + 6.0
                        * (-1.0 + 2.0 * rcos.x.powi(2))
                        * (s_or_h[&13].eval(rcos.w) - s_or_h[&14].eval(rcos.w))
                    + rcos.w
                        * (-((rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].deriv(rcos.w))
                            + (-3.0 * rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                                * s_or_h[&13].deriv(rcos.w)
                            + 3.0 * rcos.x.powi(2) * s_or_h[&14].deriv(rcos.w))))
                / rcos.w
        }
        (2, -2, 2, 2) => {
            (rcos.x
                * (-((4.0 * rcos.x.powi(4) + rcos.y.powi(2)
                    - rcos.x.powi(2) * (3.0 + 4.0 * rcos.y.powi(2)))
                    * s_or_h[&12].eval(rcos.w))
                    + 4.0
                        * (rcos.x.powi(2)
                            + rcos.x.powi(2)
                                * (-3.0 + 4.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 9.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * rcos.x.powi(2)
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -1, 0, 0) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&9].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 1, -1) => {
            (rcos.x
                * rcos.z
                * ((2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + SQRT3
                        * rcos.y.powi(2)
                        * (-3.0 * s_or_h[&11].eval(rcos.w) + rcos.w * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -1, 1, 0) => {
            (rcos.x
                * rcos.y
                * ((2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + SQRT3
                        * rcos.z.powi(2)
                        * (-3.0 * s_or_h[&11].eval(rcos.w) + rcos.w * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -1, 1, 1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.x.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.x.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -1, 2, -2) => {
            (rcos.z
                * ((4.0 * rcos.x.powi(4) - rcos.z.powi(2)
                    + rcos.x.powi(2) * (-3.0 + 4.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + (-3.0 * rcos.y.powi(2)
                        + rcos.z.powi(2)
                        + rcos.x.powi(2)
                            * (3.0 - 4.0 * rcos.x.powi(2) + 12.0 * rcos.y.powi(2)
                                - 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 2, -1) => {
            (rcos.x
                * (2.0
                    * (rcos.x.powi(2)
                        - 2.0 * rcos.x.powi(2) * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2))
                        + rcos.z.powi(2)
                        - 2.0 * (rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (-2.0 * rcos.y.powi(4) + rcos.z.powi(2)
                            - 2.0 * rcos.z.powi(2) * (rcos.x.powi(2) + rcos.z.powi(2))
                            + rcos.y.powi(2)
                                * (1.0 - 2.0 * rcos.x.powi(2) + 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 2, 0) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * ((-2.0 + 4.0 * rcos.x.powi(2) + 4.0 * rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    + (4.0 - 8.0 * rcos.x.powi(2) - 8.0 * rcos.y.powi(2) + 8.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - 2.0 * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -1, 2, 1) => {
            (rcos.x
                * ((4.0 * rcos.x.powi(4) - rcos.y.powi(2)
                    + rcos.x.powi(2) * (-3.0 + 4.0 * rcos.y.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + (rcos.x.powi(2) - 3.0 * rcos.z.powi(2)
                        + rcos.x.powi(2)
                            * (3.0 - 4.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2)
                                + 12.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 2, 2) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (-2.0
                    * (-3.0 + 6.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (-3.0 + 6.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2)
                            + 2.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 0, 0) => {
            (rcos.x
                * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                * s_or_h[&9].eval(rcos.w))
                / rcos.w
                - (rcos.x
                    * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&9].deriv(rcos.w))
                    / 2.
        }
        (2, 0, 1, -1) => {
            -(rcos.x
                * rcos.y
                * ((2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + 2.0
                        * rcos.z.powi(2)
                        * (-3.0 * SQRT3 * s_or_h[&10].eval(rcos.w)
                            + 3.0 * s_or_h[&11].eval(rcos.w)
                            + rcos.w * SQRT3 * s_or_h[&11].deriv(rcos.w))
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 1, 0) => {
            -(rcos.x
                * rcos.z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + (2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 6.0 * rcos.z.powi(2))
                        * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (-2.0
                            * SQRT3
                            * (rcos.x.powi(2) + rcos.y.powi(2))
                            * s_or_h[&10].deriv(rcos.w)
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                * s_or_h[&11].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, 0, 1, 1) => {
            -(2.0
                * SQRT3
                * (1.0 - 3.0 * rcos.x.powi(2))
                * rcos.z.powi(2)
                * s_or_h[&10].eval(rcos.w)
                + (rcos.x.powi(2)
                    - 2.0 * rcos.z.powi(2)
                    - 3.0
                        * rcos.x.powi(2)
                        * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)))
                    * s_or_h[&11].eval(rcos.w)
                + rcos.w
                    * rcos.x.powi(2)
                    * (2.0 * SQRT3 * rcos.z.powi(2) * s_or_h[&10].deriv(rcos.w)
                        + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                            * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, -2) => {
            (SQRT3
                * rcos.y
                * ((-4.0 * rcos.x.powi(4)
                    + rcos.y.powi(2)
                    + 2.0 * rcos.z.powi(2)
                    + rcos.x.powi(2) * (3.0 - 4.0 * rcos.y.powi(2) - 8.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (-1.0 + 4.0 * rcos.x.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    - 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.x.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, -1) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * ((-2.0 + 4.0 * rcos.x.powi(2) + 4.0 * rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    + (4.0 - 8.0 * rcos.x.powi(2) - 8.0 * rcos.y.powi(2) + 8.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - 2.0 * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, 0) => {
            (rcos.x
                * (-12.0
                    * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2))
                    * (rcos.x.powi(2) + rcos.y.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 24.0
                        * (-1.0 + 2.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 16.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 16.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 16.0 * rcos.z.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 6.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 12.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 12.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)).powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 0, 2, 1) => {
            (SQRT3
                * rcos.z
                * ((4.0 * rcos.x.powi(4) - rcos.y.powi(2)
                    + rcos.x.powi(2) * (-3.0 + 4.0 * rcos.y.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * ((rcos.x - rcos.z) * (rcos.x + rcos.z)
                            + rcos.x.powi(2)
                                * (3.0 - 4.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2)
                                    + 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    - 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.x.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, 2) => {
            (SQRT3
                * rcos.x
                * (4.0
                    * (-rcos.x.powi(4)
                        + rcos.y.powi(4)
                        + rcos.z.powi(2)
                        + 2.0 * rcos.y.powi(2) * rcos.z.powi(2)
                        + rcos.x.powi(2) * (1.0 - 2.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 8.0
                        * (-1.0 + 2.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    - 4.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 8.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 1, 0, 0) => {
            (SQRT3
                * rcos.z
                * ((1.0 - 2.0 * rcos.x.powi(2)) * s_or_h[&9].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 1, -1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.x.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.x.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, 1, 1, 0) => {
            (((rcos.x - rcos.z) * (rcos.x + rcos.z)
                - 3.0 * rcos.x.powi(2) * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2)))
                * s_or_h[&10].eval(rcos.w)
                + rcos.w
                    * rcos.x.powi(2)
                    * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                    * s_or_h[&10].deriv(rcos.w)
                + SQRT3
                    * rcos.z.powi(2)
                    * ((1.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&11].eval(rcos.w)
                        + rcos.w * rcos.x.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 1, 1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (2.0 - 3.0 * rcos.x.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.x.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, -2) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (4.0 * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 4.0 * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&13].eval(rcos.w)
                    + 6.0
                        * (-1.0 + 2.0 * rcos.x.powi(2))
                        * (s_or_h[&13].eval(rcos.w) - s_or_h[&14].eval(rcos.w))
                    + rcos.w
                        * (-((rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].deriv(rcos.w))
                            + (-3.0 * rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                                * s_or_h[&13].deriv(rcos.w)
                            + 3.0 * rcos.x.powi(2) * s_or_h[&14].deriv(rcos.w))))
                / rcos.w
        }
        (2, 1, 2, -1) => {
            (rcos.x
                * ((4.0 * rcos.x.powi(4) - rcos.y.powi(2)
                    + rcos.x.powi(2) * (-3.0 + 4.0 * rcos.y.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + (rcos.x.powi(2) - 3.0 * rcos.z.powi(2)
                        + rcos.x.powi(2)
                            * (3.0 - 4.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2)
                                + 12.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, 0) => {
            (SQRT3
                * rcos.z
                * ((4.0 * rcos.x.powi(4) - rcos.y.powi(2)
                    + rcos.x.powi(2) * (-3.0 + 4.0 * rcos.y.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * ((rcos.x - rcos.z) * (rcos.x + rcos.z)
                            + rcos.x.powi(2)
                                * (3.0 - 4.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2)
                                    + 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    - 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.x.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 1, 2, 1) => {
            (rcos.x
                * (-2.0
                    * (-1.0 + 2.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2))
                    * (rcos.x.powi(2) + rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2)
                            - 2.0 * rcos.z.powi(2)
                            - 2.0
                                * (rcos.x.powi(2) * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2))
                                    + (-2.0 * rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2)
                                    + rcos.z.powi(4)))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, 2) => {
            -(rcos.z
                * ((-4.0 * rcos.x.powi(4)
                    + 3.0 * rcos.y.powi(2)
                    + 2.0 * rcos.z.powi(2)
                    + rcos.x.powi(2) * (3.0 - 12.0 * rcos.y.powi(2) - 8.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (4.0 * rcos.x.powi(4) + 3.0 * rcos.y.powi(2) + rcos.z.powi(2)
                            - rcos.x.powi(2)
                                * (3.0 + 12.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    - 9.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 3.0
                        * rcos.w
                        * rcos.x.powi(2)
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 0, 0) => {
            (SQRT3
                * rcos.x
                * (2.0 * (1.0 - rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&9].eval(rcos.w)
                    + rcos.w * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&9].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 1, -1) => {
            (rcos.x
                * rcos.y
                * (2.0
                    * (-4.0 + 6.0 * rcos.x.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + SQRT3
                        * (2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                        * s_or_h[&11].eval(rcos.w)
                    - 2.0
                        * rcos.w
                        * (2.0 * rcos.x.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w
                        * SQRT3
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 1, 0) => {
            (rcos.x
                * rcos.z
                * ((-4.0 + 6.0 * rcos.x.powi(2) - 6.0 * rcos.y.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3
                        * (2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                        * s_or_h[&11].eval(rcos.w)
                    - rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * (2.0 * s_or_h[&11].deriv(rcos.w) - SQRT3 * s_or_h[&11].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, 2, 1, 1) => {
            (-2.0
                * (-1.0 + 3.0 * rcos.x.powi(2))
                * (2.0 * rcos.y.powi(2) + rcos.z.powi(2))
                * s_or_h[&10].eval(rcos.w)
                - SQRT3
                    * (3.0 * rcos.x.powi(4) + rcos.y.powi(2)
                        - 3.0 * rcos.x.powi(2) * (1.0 + rcos.y.powi(2)))
                    * s_or_h[&11].eval(rcos.w)
                + rcos.w
                    * rcos.x.powi(2)
                    * (2.0 * (2.0 * rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&10].deriv(rcos.w)
                        + SQRT3
                            * (rcos.x - rcos.y)
                            * (rcos.x + rcos.y)
                            * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, -2) => {
            (rcos.x
                * (-((4.0 * rcos.x.powi(4) + rcos.y.powi(2)
                    - rcos.x.powi(2) * (3.0 + 4.0 * rcos.y.powi(2)))
                    * s_or_h[&12].eval(rcos.w))
                    + 4.0
                        * (rcos.x.powi(2)
                            + rcos.x.powi(2)
                                * (-3.0 + 4.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 9.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * rcos.x.powi(2)
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, -1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (-2.0
                    * (-3.0 + 6.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (-3.0 + 6.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2)
                            + 2.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, 0) => {
            (SQRT3
                * rcos.x
                * (4.0
                    * (-rcos.x.powi(4)
                        + rcos.y.powi(4)
                        + rcos.z.powi(2)
                        + 2.0 * rcos.y.powi(2) * rcos.z.powi(2)
                        + rcos.x.powi(2) * (1.0 - 2.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 8.0
                        * (-1.0 + 2.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    - 4.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 8.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 2, 2, 1) => {
            -(rcos.z
                * ((-4.0 * rcos.x.powi(4)
                    + 3.0 * rcos.y.powi(2)
                    + 2.0 * rcos.z.powi(2)
                    + rcos.x.powi(2) * (3.0 - 12.0 * rcos.y.powi(2) - 8.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (4.0 * rcos.x.powi(4) + 3.0 * rcos.y.powi(2) + rcos.z.powi(2)
                            - rcos.x.powi(2)
                                * (3.0 + 12.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    - 9.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 3.0
                        * rcos.w
                        * rcos.x.powi(2)
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, 2) => {
            (rcos.x
                * (-4.0
                    * (rcos.x.powi(4) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)
                        + (rcos.x.powi(2) + 2.0 * rcos.z.powi(2)).powi(2)
                        + rcos.x.powi(2) * (-1.0 - 2.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    - 8.0
                        * ((-1.0 + 2.0 * rcos.x.powi(2)) * rcos.z.powi(2)
                            + 2.0
                                * rcos.y.powi(2)
                                * (-2.0 + 4.0 * rcos.x.powi(2) + rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 12.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 24.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 16.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2)).powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!"),
    };

    let grad1: f64 = match (orb_a.l, orb_a.m, orb_b.l, orb_b.m) {
        (0, 0, 0, 0) => rcos.y * s_or_h[&0].deriv(rcos.w),
        (0, 0, 1, -1) => {
            -(((-1.0 + rcos.y.powi(2)) * s_or_h[&2].eval(rcos.w)) / rcos.w)
                + rcos.y.powi(2) * s_or_h[&2].deriv(rcos.w)
        }
        (0, 0, 1, 0) => {
            rcos.y * rcos.z * (-(s_or_h[&2].eval(rcos.w) / rcos.w) + s_or_h[&2].deriv(rcos.w))
        }
        (0, 0, 1, 1) => {
            rcos.x * rcos.y * (-(s_or_h[&2].eval(rcos.w) / rcos.w) + s_or_h[&2].deriv(rcos.w))
        }
        (0, 0, 2, -2) => {
            (SQRT3
                * rcos.x
                * ((1.0 - 2.0 * rcos.y.powi(2)) * s_or_h[&3].eval(rcos.w)
                    + rcos.w * rcos.y.powi(2) * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, -1) => {
            (SQRT3
                * rcos.z
                * ((1.0 - 2.0 * rcos.y.powi(2)) * s_or_h[&3].eval(rcos.w)
                    + rcos.w * rcos.y.powi(2) * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, 0) => {
            (rcos.x
                * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                * s_or_h[&3].eval(rcos.w))
                / rcos.w
                - (rcos.x
                    * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&3].deriv(rcos.w))
                    / 2.
        }
        (0, 0, 2, 1) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&3].eval(rcos.w) + rcos.w * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, 2) => {
            (SQRT3
                * rcos.y
                * (-2.0 * (1.0 + rcos.x.powi(2) - rcos.y.powi(2)) * s_or_h[&3].eval(rcos.w)
                    + rcos.w * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&3].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, -1, 0, 0) => {
            -(((-1.0 + rcos.y.powi(2)) * s_or_h[&4].eval(rcos.w)) / rcos.w)
                + rcos.y.powi(2) * s_or_h[&4].deriv(rcos.w)
        }
        (1, -1, 1, -1) => {
            (rcos.x
                * (-2.0 * (-1.0 + rcos.y.powi(2)) * s_or_h[&6].eval(rcos.w)
                    - (rcos.x.powi(2) + rcos.z.powi(2))
                        * (2.0 * s_or_h[&5].eval(rcos.w) - rcos.w * s_or_h[&5].deriv(rcos.w))
                    + rcos.w * rcos.y.powi(2) * s_or_h[&6].deriv(rcos.w)))
                / rcos.w
        }
        (1, -1, 1, 0) => {
            (rcos.z
                * ((-1.0 + 2.0 * rcos.y.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.y.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, -1, 1, 1) => {
            (rcos.x
                * ((-1.0 + 2.0 * rcos.y.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.y.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, -1, 2, -2) => {
            (rcos.x
                * rcos.y
                * (-((2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w))
                    + SQRT3 * (2.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.y.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, -1, 2, -1) => {
            (rcos.x
                * rcos.z
                * (-((2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w))
                    + SQRT3 * (2.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.y.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, -1, 2, 0) => {
            -(2.0 * SQRT3 * (1.0 - 3.0 * rcos.y.powi(2)) * rcos.z.powi(2) * s_or_h[&7].eval(rcos.w)
                + rcos.x.powi(2) * s_or_h[&8].eval(rcos.w)
                + (-2.0 * rcos.z.powi(2)
                    - 3.0
                        * rcos.y.powi(2)
                        * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)))
                    * s_or_h[&8].eval(rcos.w)
                + rcos.w
                    * rcos.y.powi(2)
                    * (2.0 * SQRT3 * rcos.z.powi(2) * s_or_h[&7].deriv(rcos.w)
                        + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                            * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, -1, 2, 1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.y.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.y.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 2, 2) => {
            (2.0 * (-1.0 + 3.0 * rcos.y.powi(2))
                * (2.0 * rcos.x.powi(2) + rcos.z.powi(2))
                * s_or_h[&7].eval(rcos.w)
                + SQRT3
                    * (rcos.x.powi(2) - 3.0 * (1.0 + rcos.x.powi(2)) * rcos.y.powi(2)
                        + 3.0 * rcos.y.powi(4))
                    * s_or_h[&8].eval(rcos.w)
                + rcos.w
                    * rcos.y.powi(2)
                    * (-2.0 * (2.0 * rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&7].deriv(rcos.w)
                        + SQRT3 * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, 0, 0, 0) => {
            rcos.y * rcos.z * (-(s_or_h[&4].eval(rcos.w) / rcos.w) + s_or_h[&4].deriv(rcos.w))
        }
        (1, 0, 1, -1) => {
            (rcos.z
                * ((-1.0 + 2.0 * rcos.y.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.y.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, 0, 1, 0) => {
            (rcos.x
                * (-2.0 * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + rcos.w * (rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&5].deriv(rcos.w)
                    + rcos.z.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w) + rcos.w * s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 1, 1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (2.0 * s_or_h[&5].eval(rcos.w) - 2.0 * s_or_h[&6].eval(rcos.w)
                    + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 2, -2) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.y.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.y.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 2, -1) => {
            ((rcos.x.powi(2) * (1.0 - 3.0 * rcos.y.powi(2)) - rcos.z.powi(2)
                + 3.0 * rcos.y.powi(2) * (1.0 - rcos.y.powi(2) + rcos.z.powi(2)))
                * s_or_h[&7].eval(rcos.w)
                + rcos.w
                    * rcos.y.powi(2)
                    * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                    * s_or_h[&7].deriv(rcos.w)
                + SQRT3
                    * rcos.z.powi(2)
                    * ((1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                        + rcos.w * rcos.y.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, 0, 2, 0) => {
            -(rcos.x
                * rcos.z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + (2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 6.0 * rcos.z.powi(2))
                        * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * (-2.0
                            * SQRT3
                            * (rcos.x.powi(2) + rcos.y.powi(2))
                            * s_or_h[&7].deriv(rcos.w)
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                * s_or_h[&8].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (1, 0, 2, 1) => {
            (rcos.x
                * rcos.y
                * ((2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + SQRT3
                        * rcos.z.powi(2)
                        * (-3.0 * s_or_h[&8].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 2, 2) => {
            (rcos.x
                * rcos.z
                * ((4.0 + 6.0 * rcos.x.powi(2) - 6.0 * rcos.y.powi(2)) * s_or_h[&7].eval(rcos.w)
                    - 2.0 * SQRT3 * s_or_h[&8].eval(rcos.w)
                    - (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * (3.0 * SQRT3 * s_or_h[&8].eval(rcos.w)
                            + 2.0 * rcos.w * s_or_h[&8].deriv(rcos.w)
                            - rcos.w * SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (1, 1, 0, 0) => {
            rcos.x * rcos.y * (-(s_or_h[&4].eval(rcos.w) / rcos.w) + s_or_h[&4].deriv(rcos.w))
        }
        (1, 1, 1, -1) => {
            (rcos.x
                * ((-1.0 + 2.0 * rcos.y.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.y.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, 1, 1, 0) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (2.0 * s_or_h[&5].eval(rcos.w) - 2.0 * s_or_h[&6].eval(rcos.w)
                    + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 1, 1) => {
            (rcos.x
                * (-2.0 * (-1.0 + rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + rcos.w * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&5].deriv(rcos.w)
                    + rcos.x.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w) + rcos.w * s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 2, -2) => {
            ((rcos.x.powi(2) * (-1.0 + 3.0 * rcos.y.powi(2)) + rcos.z.powi(2)
                - 3.0 * rcos.y.powi(2) * (-1.0 + rcos.y.powi(2) + rcos.z.powi(2)))
                * s_or_h[&7].eval(rcos.w)
                + rcos.w
                    * rcos.y.powi(2)
                    * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                    * s_or_h[&7].deriv(rcos.w)
                + SQRT3
                    * rcos.x.powi(2)
                    * ((1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                        + rcos.w * rcos.y.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, 1, 2, -1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.y.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.y.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 2, 0) => {
            -(rcos.x
                * rcos.y
                * ((2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + 2.0
                        * rcos.z.powi(2)
                        * (-3.0 * SQRT3 * s_or_h[&7].eval(rcos.w)
                            + 3.0 * s_or_h[&8].eval(rcos.w)
                            + rcos.w * SQRT3 * s_or_h[&8].deriv(rcos.w))
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, 1, 2, 1) => {
            (rcos.x
                * rcos.z
                * ((2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + rcos.w
                        * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + SQRT3
                        * rcos.x.powi(2)
                        * (-3.0 * s_or_h[&8].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 2, 2) => {
            (rcos.x
                * rcos.y
                * (-2.0
                    * (-4.0 + 6.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + SQRT3
                        * (-2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                        * s_or_h[&8].eval(rcos.w)
                    + 2.0
                        * rcos.w
                        * (2.0 * rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w
                        * SQRT3
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -2, 0, 0) => {
            (SQRT3
                * rcos.x
                * ((1.0 - 2.0 * rcos.y.powi(2)) * s_or_h[&9].eval(rcos.w)
                    + rcos.w * rcos.y.powi(2) * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 1, -1) => {
            (rcos.x
                * rcos.y
                * (-((2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w))
                    + SQRT3 * (2.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.y.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 1, 0) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.y.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.y.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -2, 1, 1) => {
            ((rcos.x.powi(2) * (-1.0 + 3.0 * rcos.y.powi(2)) + rcos.z.powi(2)
                - 3.0 * rcos.y.powi(2) * (-1.0 + rcos.y.powi(2) + rcos.z.powi(2)))
                * s_or_h[&10].eval(rcos.w)
                + rcos.w
                    * rcos.y.powi(2)
                    * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                    * s_or_h[&10].deriv(rcos.w)
                + SQRT3
                    * rcos.x.powi(2)
                    * ((1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                        + rcos.w * rcos.y.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, -2) => {
            (rcos.x
                * (-2.0
                    * (rcos.x.powi(2) + rcos.z.powi(2))
                    * (-1.0 + 2.0 * rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (-2.0
                            * (rcos.x - rcos.y)
                            * (rcos.x + rcos.y)
                            * (1.0 + rcos.x.powi(2) - rcos.y.powi(2))
                            + (1.0 - 2.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2))
                                * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, -1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (4.0 * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (3.0 + 2.0 * rcos.x.powi(2) - 6.0 * rcos.y.powi(2)
                            + 2.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, 0) => {
            (SQRT3
                * rcos.x
                * ((rcos.x.powi(2) * (1.0 - 4.0 * rcos.y.powi(2))
                    + 2.0 * rcos.z.powi(2)
                    + rcos.y.powi(2) * (3.0 - 4.0 * rcos.y.powi(2) - 8.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (-1.0 + 4.0 * rcos.y.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    - rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.y.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -2, 2, 1) => {
            (rcos.z
                * ((4.0 * rcos.y.powi(4) - rcos.z.powi(2)
                    + rcos.y.powi(2) * (-3.0 + 4.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + (3.0 * rcos.x.powi(2) * (-1.0 + 4.0 * rcos.y.powi(2))
                        + rcos.z.powi(2)
                        + rcos.y.powi(2) * (3.0 - 4.0 * rcos.y.powi(2) - 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, 2) => {
            (rcos.x
                * ((rcos.x.powi(2) - (3.0 + 4.0 * rcos.x.powi(2)) * rcos.y.powi(2)
                    + 4.0 * rcos.y.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (-rcos.x.powi(2) + (3.0 + 4.0 * rcos.x.powi(2)) * rcos.y.powi(2)
                            - 4.0 * rcos.y.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 9.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * rcos.y.powi(2)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -1, 0, 0) => {
            (SQRT3
                * rcos.z
                * ((1.0 - 2.0 * rcos.y.powi(2)) * s_or_h[&9].eval(rcos.w)
                    + rcos.w * rcos.y.powi(2) * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 1, -1) => {
            (rcos.x
                * rcos.z
                * (-((2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w))
                    + SQRT3 * (2.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.y.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 1, 0) => {
            ((rcos.x.powi(2) * (1.0 - 3.0 * rcos.y.powi(2)) - rcos.z.powi(2)
                + 3.0 * rcos.y.powi(2) * (1.0 - rcos.y.powi(2) + rcos.z.powi(2)))
                * s_or_h[&10].eval(rcos.w)
                + rcos.w
                    * rcos.y.powi(2)
                    * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                    * s_or_h[&10].deriv(rcos.w)
                + SQRT3
                    * rcos.z.powi(2)
                    * ((1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                        + rcos.w * rcos.y.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 1, 1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.y.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.y.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -1, 2, -2) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (4.0 * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (3.0 + 2.0 * rcos.x.powi(2) - 6.0 * rcos.y.powi(2)
                            + 2.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 2, -1) => {
            (rcos.x
                * (-2.0
                    * (-1.0 + 2.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2))
                    * (rcos.x.powi(2) + rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) * (1.0 - 2.0 * rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                            - 2.0
                                * (rcos.x - rcos.z)
                                * (rcos.x + rcos.z)
                                * (-1.0 + rcos.y.powi(2) - rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 2, 0) => {
            (SQRT3
                * rcos.z
                * ((-rcos.x.powi(2)
                    + (-3.0 + 4.0 * rcos.x.powi(2)) * rcos.y.powi(2)
                    + 4.0 * rcos.y.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) + 3.0 * rcos.y.powi(2)
                            - 4.0 * rcos.x.powi(2) * rcos.y.powi(2)
                            - 4.0 * rcos.y.powi(4)
                            + (-1.0 + 4.0 * rcos.y.powi(2)) * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.y.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -1, 2, 1) => {
            (rcos.x
                * ((-rcos.x.powi(2)
                    + (-3.0 + 4.0 * rcos.x.powi(2)) * rcos.y.powi(2)
                    + 4.0 * rcos.y.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + (rcos.x.powi(2) + 3.0 * rcos.y.powi(2)
                        - 4.0 * rcos.x.powi(2) * rcos.y.powi(2)
                        - 4.0 * rcos.y.powi(4)
                        + 3.0 * (-1.0 + 4.0 * rcos.y.powi(2)) * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 2, 2) => {
            (rcos.z
                * ((-4.0 * rcos.y.powi(4)
                    + rcos.x.powi(2) * (3.0 - 12.0 * rcos.y.powi(2))
                    + 2.0 * rcos.z.powi(2)
                    + rcos.y.powi(2) * (3.0 - 8.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + (-8.0 * rcos.y.powi(4)
                        + 6.0 * rcos.x.powi(2) * (-1.0 + 4.0 * rcos.y.powi(2))
                        - 2.0 * rcos.z.powi(2)
                        + rcos.y.powi(2) * (6.0 + 8.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 9.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * rcos.y.powi(2)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 0, 0) => {
            (rcos.x
                * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                * s_or_h[&9].eval(rcos.w))
                / rcos.w
                - (rcos.x
                    * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&9].deriv(rcos.w))
                    / 2.
        }
        (2, 0, 1, -1) => {
            -(2.0
                * SQRT3
                * (1.0 - 3.0 * rcos.y.powi(2))
                * rcos.z.powi(2)
                * s_or_h[&10].eval(rcos.w)
                + (-3.0 * rcos.y.powi(4) + rcos.x.powi(2) * (1.0 - 3.0 * rcos.y.powi(2))
                    - 2.0 * rcos.z.powi(2)
                    + rcos.y.powi(2) * (3.0 + 6.0 * rcos.z.powi(2)))
                    * s_or_h[&11].eval(rcos.w)
                + rcos.w
                    * rcos.y.powi(2)
                    * (2.0 * SQRT3 * rcos.z.powi(2) * s_or_h[&10].deriv(rcos.w)
                        + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                            * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 1, 0) => {
            -(rcos.x
                * rcos.z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + (2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 6.0 * rcos.z.powi(2))
                        * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (-2.0
                            * SQRT3
                            * (rcos.x.powi(2) + rcos.y.powi(2))
                            * s_or_h[&10].deriv(rcos.w)
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                * s_or_h[&11].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, 0, 1, 1) => {
            -(rcos.x
                * rcos.y
                * ((2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + 2.0
                        * rcos.z.powi(2)
                        * (-3.0 * SQRT3 * s_or_h[&10].eval(rcos.w)
                            + 3.0 * s_or_h[&11].eval(rcos.w)
                            + rcos.w * SQRT3 * s_or_h[&11].deriv(rcos.w))
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, -2) => {
            (SQRT3
                * rcos.x
                * ((rcos.x.powi(2) * (1.0 - 4.0 * rcos.y.powi(2))
                    + 2.0 * rcos.z.powi(2)
                    + rcos.y.powi(2) * (3.0 - 4.0 * rcos.y.powi(2) - 8.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (-1.0 + 4.0 * rcos.y.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    - rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.y.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, -1) => {
            (SQRT3
                * rcos.z
                * ((-rcos.x.powi(2)
                    + (-3.0 + 4.0 * rcos.x.powi(2)) * rcos.y.powi(2)
                    + 4.0 * rcos.y.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) + 3.0 * rcos.y.powi(2)
                            - 4.0 * rcos.x.powi(2) * rcos.y.powi(2)
                            - 4.0 * rcos.y.powi(4)
                            + (-1.0 + 4.0 * rcos.y.powi(2)) * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 2.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.y.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, 0) => {
            (rcos.x
                * (-12.0
                    * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2))
                    * (rcos.x.powi(2) + rcos.y.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 24.0
                        * (-1.0 + 2.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 16.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 16.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 16.0 * rcos.z.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 6.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 12.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 12.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)).powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 0, 2, 1) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * ((-2.0 + 4.0 * rcos.x.powi(2) + 4.0 * rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    + (4.0 - 8.0 * rcos.x.powi(2) - 8.0 * rcos.y.powi(2) + 8.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - 2.0 * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, 2) => {
            (SQRT3
                * rcos.y
                * (-4.0
                    * (rcos.x.powi(4) + rcos.y.powi(2) - rcos.y.powi(4)
                        + (1.0 + 2.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2)) * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 8.0
                        * (1.0 + 2.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    + 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 8.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 1, 0, 0) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&9].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 1, -1) => {
            (rcos.x
                * rcos.z
                * ((-2.0 + 6.0 * rcos.y.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.y.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, 1, 1, 0) => {
            (rcos.x
                * rcos.y
                * ((2.0 - 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + SQRT3
                        * rcos.z.powi(2)
                        * (-3.0 * s_or_h[&11].eval(rcos.w) + rcos.w * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, 1, 1, 1) => {
            (rcos.x
                * rcos.z
                * ((2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + rcos.w
                        * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + SQRT3
                        * rcos.x.powi(2)
                        * (-3.0 * s_or_h[&11].eval(rcos.w) + rcos.w * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, 1, 2, -2) => {
            (rcos.z
                * ((4.0 * rcos.y.powi(4) - rcos.z.powi(2)
                    + rcos.y.powi(2) * (-3.0 + 4.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + (3.0 * rcos.x.powi(2) * (-1.0 + 4.0 * rcos.y.powi(2))
                        + rcos.z.powi(2)
                        + rcos.y.powi(2) * (3.0 - 4.0 * rcos.y.powi(2) - 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, -1) => {
            (rcos.x
                * ((-rcos.x.powi(2)
                    + (-3.0 + 4.0 * rcos.x.powi(2)) * rcos.y.powi(2)
                    + 4.0 * rcos.y.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + (rcos.x.powi(2) + 3.0 * rcos.y.powi(2)
                        - 4.0 * rcos.x.powi(2) * rcos.y.powi(2)
                        - 4.0 * rcos.y.powi(4)
                        + 3.0 * (-1.0 + 4.0 * rcos.y.powi(2)) * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, 0) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * ((-2.0 + 4.0 * rcos.x.powi(2) + 4.0 * rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    + (4.0 - 8.0 * rcos.x.powi(2) - 8.0 * rcos.y.powi(2) + 8.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - 2.0 * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 1, 2, 1) => {
            (rcos.x
                * (2.0
                    * (rcos.x.powi(2)
                        + rcos.x.powi(2) * (1.0 - 2.0 * rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        - 2.0 * rcos.y.powi(2) * (-1.0 + rcos.y.powi(2) + rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (-2.0 * rcos.x.powi(4) + rcos.z.powi(2)
                            - 2.0 * rcos.z.powi(2) * (rcos.x.powi(2) + rcos.z.powi(2))
                            + rcos.x.powi(2)
                                * (1.0 - 2.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, 2) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (2.0
                    * (-3.0 + 2.0 * rcos.x.powi(2) + 6.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (3.0 + 2.0 * rcos.x.powi(2)
                            - 6.0 * rcos.y.powi(2)
                            - 2.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - 6.0 * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 6.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 0, 0) => {
            (SQRT3
                * rcos.y
                * (-2.0 * (1.0 + rcos.x.powi(2) - rcos.y.powi(2)) * s_or_h[&9].eval(rcos.w)
                    + rcos.w * (rcos.x - rcos.y) * (rcos.x + rcos.y) * s_or_h[&9].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 1, -1) => {
            (2.0 * (-1.0 + 3.0 * rcos.y.powi(2))
                * (2.0 * rcos.x.powi(2) + rcos.z.powi(2))
                * s_or_h[&10].eval(rcos.w)
                + SQRT3
                    * (rcos.x.powi(2) - 3.0 * (1.0 + rcos.x.powi(2)) * rcos.y.powi(2)
                        + 3.0 * rcos.y.powi(4))
                    * s_or_h[&11].eval(rcos.w)
                + rcos.w
                    * rcos.y.powi(2)
                    * (-2.0 * (2.0 * rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&10].deriv(rcos.w)
                        + SQRT3
                            * (rcos.x - rcos.y)
                            * (rcos.x + rcos.y)
                            * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 1, 0) => {
            -(rcos.x
                * rcos.z
                * ((-4.0 - 6.0 * rcos.x.powi(2) + 6.0 * rcos.y.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3
                        * (2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2))
                        * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * (2.0 * s_or_h[&11].deriv(rcos.w) - SQRT3 * s_or_h[&11].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, 2, 1, 1) => {
            (rcos.x
                * rcos.y
                * (-2.0
                    * (-4.0 + 6.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + SQRT3
                        * (-2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2))
                        * s_or_h[&11].eval(rcos.w)
                    + 2.0
                        * rcos.w
                        * (2.0 * rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w
                        * SQRT3
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, -2) => {
            (rcos.x
                * ((rcos.x.powi(2) - (3.0 + 4.0 * rcos.x.powi(2)) * rcos.y.powi(2)
                    + 4.0 * rcos.y.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (-rcos.x.powi(2) + (3.0 + 4.0 * rcos.x.powi(2)) * rcos.y.powi(2)
                            - 4.0 * rcos.y.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 9.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * rcos.y.powi(2)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, -1) => {
            (rcos.z
                * ((-4.0 * rcos.y.powi(4)
                    + rcos.x.powi(2) * (3.0 - 12.0 * rcos.y.powi(2))
                    + 2.0 * rcos.z.powi(2)
                    + rcos.y.powi(2) * (3.0 - 8.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    + (-8.0 * rcos.y.powi(4)
                        + 6.0 * rcos.x.powi(2) * (-1.0 + 4.0 * rcos.y.powi(2))
                        - 2.0 * rcos.z.powi(2)
                        + rcos.y.powi(2) * (6.0 + 8.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 9.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * rcos.y.powi(2)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, 0) => {
            (SQRT3
                * rcos.y
                * (-4.0
                    * (rcos.x.powi(4) + rcos.y.powi(2) - rcos.y.powi(4)
                        + (1.0 + 2.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2)) * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 8.0
                        * (1.0 + 2.0 * rcos.x.powi(2) - 2.0 * rcos.y.powi(2))
                        * rcos.z.powi(2)
                        * s_or_h[&13].eval(rcos.w)
                    + 4.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    - 4.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 8.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 2, 2, 1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (2.0
                    * (-3.0 + 2.0 * rcos.x.powi(2) + 6.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 4.0
                        * (3.0 + 2.0 * rcos.x.powi(2)
                            - 6.0 * rcos.y.powi(2)
                            - 2.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - 6.0 * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 6.0 * rcos.w * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, 2) => {
            (rcos.x
                * (-4.0
                    * (rcos.x.powi(4)
                        + (-1.0 + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                            * (rcos.x.powi(2) + 2.0 * rcos.z.powi(2))
                        + rcos.x.powi(2) * (1.0 - 2.0 * rcos.y.powi(2) + 4.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    - 8.0
                        * ((-1.0 + 2.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                            + 2.0
                                * rcos.x.powi(2)
                                * (-2.0 + 4.0 * rcos.y.powi(2) + rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 24.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 16.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2)).powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!"),
    };
    let grad2: f64 = match (orb_a.l, orb_a.m, orb_b.l, orb_b.m) {
        (0, 0, 0, 0) => rcos.z * s_or_h[&0].deriv(rcos.w),
        (0, 0, 1, -1) => {
            rcos.y * rcos.z * (-(s_or_h[&2].eval(rcos.w) / rcos.w) + s_or_h[&2].deriv(rcos.w))
        }
        (0, 0, 1, 0) => {
            -(((-1.0 + rcos.z.powi(2)) * s_or_h[&2].eval(rcos.w)) / rcos.w)
                + rcos.z.powi(2) * s_or_h[&2].deriv(rcos.w)
        }
        (0, 0, 1, 1) => {
            rcos.x * rcos.z * (-(s_or_h[&2].eval(rcos.w) / rcos.w) + s_or_h[&2].deriv(rcos.w))
        }
        (0, 0, 2, -2) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&3].eval(rcos.w) + rcos.w * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, -1) => {
            (SQRT3
                * rcos.y
                * ((1.0 - 2.0 * rcos.z.powi(2)) * s_or_h[&3].eval(rcos.w)
                    + rcos.w * rcos.z.powi(2) * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, 0) => {
            (rcos.z
                * (2.0
                    * (2.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&3].eval(rcos.w)
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&3].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (0, 0, 2, 1) => {
            (SQRT3
                * rcos.x
                * ((1.0 - 2.0 * rcos.z.powi(2)) * s_or_h[&3].eval(rcos.w)
                    + rcos.w * rcos.z.powi(2) * s_or_h[&3].deriv(rcos.w)))
                / rcos.w
        }
        (0, 0, 2, 2) => {
            (SQRT3
                * (rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * rcos.z
                * (-2.0 * s_or_h[&3].eval(rcos.w) + rcos.w * s_or_h[&3].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, -1, 0, 0) => {
            rcos.y * rcos.z * (-(s_or_h[&4].eval(rcos.w) / rcos.w) + s_or_h[&4].deriv(rcos.w))
        }
        (1, -1, 1, -1) => {
            (rcos.z
                * (-2.0 * (-1.0 + rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + rcos.w * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&5].deriv(rcos.w)
                    + rcos.y.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w) + rcos.w * s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 1, 0) => {
            (rcos.x
                * ((-1.0 + 2.0 * rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.z.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, -1, 1, 1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (2.0 * s_or_h[&5].eval(rcos.w) - 2.0 * s_or_h[&6].eval(rcos.w)
                    + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 2, -2) => {
            (rcos.x
                * rcos.z
                * ((2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + SQRT3
                        * rcos.y.powi(2)
                        * (-3.0 * s_or_h[&8].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 2, -1) => {
            ((rcos.x.powi(2) - rcos.y.powi(2)
                + 3.0 * (1.0 - rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2)
                - 3.0 * rcos.z.powi(4))
                * s_or_h[&7].eval(rcos.w)
                + rcos.w
                    * rcos.z.powi(2)
                    * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                    * s_or_h[&7].deriv(rcos.w)
                + SQRT3
                    * rcos.y.powi(2)
                    * ((1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&8].eval(rcos.w)
                        + rcos.w * rcos.z.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, -1, 2, 0) => {
            (rcos.x
                * rcos.z
                * (2.0 * SQRT3 * (-2.0 + 3.0 * rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + (4.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 6.0 * rcos.z.powi(2))
                        * s_or_h[&8].eval(rcos.w)
                    - rcos.w
                        * (2.0 * SQRT3 * rcos.z.powi(2) * s_or_h[&7].deriv(rcos.w)
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                * s_or_h[&8].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (1, -1, 2, 1) => {
            (rcos.x
                * rcos.y
                * ((-2.0 + 6.0 * rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.z.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, -1, 2, 2) => {
            (rcos.x
                * rcos.z
                * (2.0
                    * (-2.0 + 6.0 * rcos.x.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + 3.0 * SQRT3 * (-rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    - 2.0
                        * rcos.w
                        * (2.0 * rcos.x.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w
                        * SQRT3
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (1, 0, 0, 0) => {
            -(((-1.0 + rcos.z.powi(2)) * s_or_h[&4].eval(rcos.w)) / rcos.w)
                + rcos.z.powi(2) * s_or_h[&4].deriv(rcos.w)
        }
        (1, 0, 1, -1) => {
            (rcos.x
                * ((-1.0 + 2.0 * rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.z.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, 0, 1, 0) => {
            (rcos.z
                * (-2.0 * (-1.0 + rcos.z.powi(2)) * s_or_h[&6].eval(rcos.w)
                    - (rcos.x.powi(2) + rcos.y.powi(2))
                        * (2.0 * s_or_h[&5].eval(rcos.w) - rcos.w * s_or_h[&5].deriv(rcos.w))
                    + rcos.w * rcos.z.powi(2) * s_or_h[&6].deriv(rcos.w)))
                / rcos.w
        }
        (1, 0, 1, 1) => {
            (rcos.x
                * ((-1.0 + 2.0 * rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.z.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, 0, 2, -2) => {
            (rcos.x
                * rcos.y
                * ((-2.0 + 6.0 * rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.z.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 0, 2, -1) => {
            (rcos.x
                * rcos.z
                * (-((2.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w))
                    + SQRT3 * (2.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.z.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, 0, 2, 0) => {
            (-2.0
                * SQRT3
                * (rcos.x.powi(2) + rcos.y.powi(2))
                * (-1.0 + 3.0 * rcos.z.powi(2))
                * s_or_h[&7].eval(rcos.w)
                + (-rcos.x.powi(2) - rcos.y.powi(2)
                    + 3.0 * (2.0 + rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2)
                    - 6.0 * rcos.z.powi(4))
                    * s_or_h[&8].eval(rcos.w)
                + rcos.w
                    * (rcos.x.powi(2) + rcos.y.powi(2))
                    * rcos.z.powi(2)
                    * (2.0 * SQRT3 * s_or_h[&7].deriv(rcos.w) - s_or_h[&8].deriv(rcos.w))
                + 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&8].deriv(rcos.w))
                / (2. * rcos.w)
        }
        (1, 0, 2, 1) => {
            (rcos.x
                * rcos.z
                * (-((2.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w))
                    + SQRT3 * (2.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.z.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, 0, 2, 2) => {
            ((rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * ((-2.0 + 6.0 * rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.z.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (1, 1, 0, 0) => {
            rcos.x * rcos.z * (-(s_or_h[&4].eval(rcos.w) / rcos.w) + s_or_h[&4].deriv(rcos.w))
        }
        (1, 1, 1, -1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (2.0 * s_or_h[&5].eval(rcos.w) - 2.0 * s_or_h[&6].eval(rcos.w)
                    + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 1, 0) => {
            (rcos.x
                * ((-1.0 + 2.0 * rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + s_or_h[&6].eval(rcos.w)
                    + rcos.z.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w)
                            + rcos.w * (-s_or_h[&5].deriv(rcos.w) + s_or_h[&6].deriv(rcos.w)))))
                / rcos.w
        }
        (1, 1, 1, 1) => {
            (rcos.z
                * (-2.0 * (-1.0 + rcos.y.powi(2) + rcos.z.powi(2)) * s_or_h[&5].eval(rcos.w)
                    + rcos.w * (rcos.x.powi(2) + rcos.z.powi(2)) * s_or_h[&5].deriv(rcos.w)
                    + rcos.x.powi(2)
                        * (-2.0 * s_or_h[&6].eval(rcos.w) + rcos.w * s_or_h[&6].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 2, -2) => {
            (rcos.x
                * rcos.z
                * ((2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + rcos.w
                        * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + SQRT3
                        * rcos.x.powi(2)
                        * (-3.0 * s_or_h[&8].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 2, -1) => {
            (rcos.x
                * rcos.y
                * ((-2.0 + 6.0 * rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + rcos.w
                        * rcos.z.powi(2)
                        * (-2.0 * s_or_h[&7].deriv(rcos.w) + SQRT3 * s_or_h[&9].deriv(rcos.w))))
                / rcos.w
        }
        (1, 1, 2, 0) => {
            (rcos.x
                * rcos.z
                * (2.0 * SQRT3 * (-2.0 + 3.0 * rcos.z.powi(2)) * s_or_h[&7].eval(rcos.w)
                    + (4.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 6.0 * rcos.z.powi(2))
                        * s_or_h[&8].eval(rcos.w)
                    - rcos.w
                        * (2.0 * SQRT3 * rcos.z.powi(2) * s_or_h[&7].deriv(rcos.w)
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                * s_or_h[&8].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (1, 1, 2, 1) => {
            ((-rcos.x.powi(2)
                + rcos.y.powi(2)
                + 3.0 * (1.0 + rcos.x.powi(2) - rcos.y.powi(2)) * rcos.z.powi(2)
                - 3.0 * rcos.z.powi(4))
                * s_or_h[&7].eval(rcos.w)
                + rcos.w
                    * rcos.z.powi(2)
                    * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                    * s_or_h[&7].deriv(rcos.w)
                + SQRT3
                    * rcos.x.powi(2)
                    * ((1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&8].eval(rcos.w)
                        + rcos.w * rcos.z.powi(2) * s_or_h[&8].deriv(rcos.w)))
                / rcos.w
        }
        (1, 1, 2, 2) => {
            (rcos.x
                * rcos.z
                * (-2.0
                    * (-2.0 + 6.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&7].eval(rcos.w)
                    + 3.0 * SQRT3 * (-rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&8].eval(rcos.w)
                    + 2.0
                        * rcos.w
                        * (2.0 * rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&7].deriv(rcos.w)
                    + rcos.w
                        * SQRT3
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&8].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -2, 0, 0) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * (-2.0 * s_or_h[&9].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 1, -1) => {
            (rcos.x
                * rcos.z
                * ((2.0 - 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + SQRT3
                        * rcos.y.powi(2)
                        * (-3.0 * s_or_h[&11].eval(rcos.w) + rcos.w * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -2, 1, 0) => {
            (rcos.x
                * rcos.y
                * ((-2.0 + 6.0 * rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.z.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -2, 1, 1) => {
            (rcos.x
                * rcos.z
                * ((2.0 + 3.0 * rcos.x.powi(2) - 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + rcos.w
                        * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + SQRT3
                        * rcos.x.powi(2)
                        * (-3.0 * s_or_h[&11].eval(rcos.w) + rcos.w * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -2, 2, -2) => {
            (rcos.z
                * (2.0
                    * (rcos.x.powi(2) + rcos.y.powi(2)
                        - 2.0 * rcos.x.powi(2) * rcos.y.powi(2)
                        - 2.0 * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2)
                        - 2.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) - 2.0 * rcos.x.powi(4)
                            + rcos.y.powi(2)
                            + 4.0 * rcos.x.powi(2) * rcos.y.powi(2)
                            - 2.0 * rcos.y.powi(4)
                            - 2.0 * (rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, -1) => {
            (rcos.x
                * ((-rcos.x.powi(2)
                    + (-3.0 + 4.0 * rcos.x.powi(2)) * rcos.z.powi(2)
                    + 4.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + (rcos.x.powi(2) - 3.0 * rcos.y.powi(2)
                        + (3.0 - 4.0 * rcos.x.powi(2) + 12.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                        - 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, 0) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * (-4.0
                    * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 8.0 * s_or_h[&13].eval(rcos.w)
                    + 4.0 * s_or_h[&14].eval(rcos.w)
                    + (rcos.x.powi(2) + rcos.y.powi(2))
                        * (4.0 * s_or_h[&14].eval(rcos.w) + rcos.w * s_or_h[&13].deriv(rcos.w))
                    + 2.0
                        * rcos.z.powi(2)
                        * (8.0 * s_or_h[&13].eval(rcos.w) - 4.0 * s_or_h[&14].eval(rcos.w)
                            + rcos.w
                                * (s_or_h[&13].deriv(rcos.w) - 2.0 * s_or_h[&14].deriv(rcos.w)))
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -2, 2, 1) => {
            (rcos.x
                * ((-rcos.x.powi(2)
                    + (-3.0 + 4.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                    + 4.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + (-3.0 * rcos.x.powi(2)
                        + rcos.y.powi(2)
                        + (3.0 + 12.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                        - 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -2, 2, 2) => {
            (rcos.x
                * (rcos.x - rcos.y)
                * rcos.y
                * (rcos.x + rcos.y)
                * rcos.z
                * (-4.0 * s_or_h[&12].eval(rcos.w) + 16.0 * s_or_h[&13].eval(rcos.w)
                    - 12.0 * s_or_h[&14].eval(rcos.w)
                    + rcos.w
                        * (s_or_h[&12].deriv(rcos.w) - 4.0 * s_or_h[&13].deriv(rcos.w)
                            + 3.0 * s_or_h[&14].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, -1, 0, 0) => {
            (SQRT3
                * rcos.y
                * ((1.0 - 2.0 * rcos.z.powi(2)) * s_or_h[&9].eval(rcos.w)
                    + rcos.w * rcos.z.powi(2) * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 1, -1) => {
            ((rcos.x.powi(2) - rcos.y.powi(2)
                + 3.0 * (1.0 - rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2)
                - 3.0 * rcos.z.powi(4))
                * s_or_h[&10].eval(rcos.w)
                + rcos.w
                    * rcos.z.powi(2)
                    * (rcos.x.powi(2) - rcos.y.powi(2) + rcos.z.powi(2))
                    * s_or_h[&10].deriv(rcos.w)
                + SQRT3
                    * rcos.y.powi(2)
                    * ((1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&11].eval(rcos.w)
                        + rcos.w * rcos.z.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 1, 0) => {
            (rcos.x
                * rcos.z
                * (-((2.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w))
                    + SQRT3 * (2.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.z.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 1, 1) => {
            (rcos.x
                * rcos.y
                * ((-2.0 + 6.0 * rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.z.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, -1, 2, -2) => {
            (rcos.x
                * ((-rcos.x.powi(2)
                    + (-3.0 + 4.0 * rcos.x.powi(2)) * rcos.z.powi(2)
                    + 4.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + (rcos.x.powi(2) - 3.0 * rcos.y.powi(2)
                        + (3.0 - 4.0 * rcos.x.powi(2) + 12.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                        - 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 2, -1) => {
            (rcos.z
                * (-2.0
                    * (rcos.x.powi(2) + rcos.y.powi(2))
                    * (-1.0 + 2.0 * rcos.x.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) * (1.0 - 2.0 * rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                            - 2.0
                                * (rcos.x - rcos.z)
                                * (rcos.x + rcos.z)
                                * (1.0 + rcos.y.powi(2) - rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, -1, 2, 0) => {
            (SQRT3
                * rcos.y
                * ((rcos.x.powi(2) + rcos.y.powi(2))
                    * (-1.0 + 4.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) + rcos.y.powi(2)
                            - (3.0 + 4.0 * rcos.x.powi(2) + 4.0 * rcos.y.powi(2))
                                * rcos.z.powi(2)
                            + 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    - rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 6.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(4) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.z.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, -1, 2, 1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (4.0 * (rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (3.0 + 2.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2)
                            - 6.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * s_or_h[&14].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2))
                        * (-s_or_h[&12].deriv(rcos.w) + s_or_h[&13].deriv(rcos.w))
                    - 3.0
                        * rcos.z.powi(2)
                        * (4.0 * s_or_h[&14].eval(rcos.w)
                            + rcos.w * (s_or_h[&14].deriv(rcos.w) - s_or_h[&15].deriv(rcos.w)))))
                / rcos.w
        }
        (2, -1, 2, 2) => {
            (rcos.x
                * ((3.0 * rcos.x.powi(2) + rcos.y.powi(2)
                    - 2.0 * (-3.0 + 6.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                    - 8.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (-3.0 * rcos.x.powi(2)
                            + rcos.y.powi(2)
                            + (-3.0 + 12.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2))
                                * rcos.z.powi(2)
                            + 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * rcos.z.powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 0, 0) => {
            (rcos.z
                * (2.0
                    * (2.0 + rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&9].eval(rcos.w)
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&9].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 1, -1) => {
            (rcos.x
                * rcos.z
                * (2.0 * SQRT3 * (-2.0 + 3.0 * rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + (4.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 6.0 * rcos.z.powi(2))
                        * s_or_h[&11].eval(rcos.w)
                    - rcos.w
                        * (2.0 * SQRT3 * rcos.z.powi(2) * s_or_h[&10].deriv(rcos.w)
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                * s_or_h[&11].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, 0, 1, 0) => {
            (-2.0
                * SQRT3
                * (rcos.x.powi(2) + rcos.y.powi(2))
                * (-1.0 + 3.0 * rcos.z.powi(2))
                * s_or_h[&10].eval(rcos.w)
                + (-rcos.x.powi(2) - rcos.y.powi(2)
                    + 3.0 * (2.0 + rcos.x.powi(2) + rcos.y.powi(2)) * rcos.z.powi(2)
                    - 6.0 * rcos.z.powi(4))
                    * s_or_h[&11].eval(rcos.w)
                + rcos.w
                    * (rcos.x.powi(2) + rcos.y.powi(2))
                    * rcos.z.powi(2)
                    * (2.0 * SQRT3 * s_or_h[&10].deriv(rcos.w) - s_or_h[&11].deriv(rcos.w))
                + 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&11].deriv(rcos.w))
                / (2. * rcos.w)
        }
        (2, 0, 1, 1) => {
            (rcos.x
                * rcos.z
                * (2.0 * SQRT3 * (-2.0 + 3.0 * rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + (4.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 6.0 * rcos.z.powi(2))
                        * s_or_h[&11].eval(rcos.w)
                    - rcos.w
                        * (2.0 * SQRT3 * rcos.z.powi(2) * s_or_h[&10].deriv(rcos.w)
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                * s_or_h[&11].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, 0, 2, -2) => {
            (SQRT3
                * rcos.x
                * rcos.y
                * rcos.z
                * (-4.0
                    * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 8.0 * s_or_h[&13].eval(rcos.w)
                    + 4.0 * s_or_h[&14].eval(rcos.w)
                    + (rcos.x.powi(2) + rcos.y.powi(2))
                        * (4.0 * s_or_h[&14].eval(rcos.w) + rcos.w * s_or_h[&13].deriv(rcos.w))
                    + 2.0
                        * rcos.z.powi(2)
                        * (8.0 * s_or_h[&13].eval(rcos.w) - 4.0 * s_or_h[&14].eval(rcos.w)
                            + rcos.w
                                * (s_or_h[&13].deriv(rcos.w) - 2.0 * s_or_h[&14].deriv(rcos.w)))
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, -1) => {
            (SQRT3
                * rcos.y
                * ((rcos.x.powi(2) + rcos.y.powi(2))
                    * (-1.0 + 4.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) + rcos.y.powi(2)
                            - (3.0 + 4.0 * rcos.x.powi(2) + 4.0 * rcos.y.powi(2))
                                * rcos.z.powi(2)
                            + 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    - rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 6.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(4) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.z.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, 0) => {
            (rcos.z
                * (-8.0
                    * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                    * s_or_h[&14].eval(rcos.w)
                    - 4.0
                        * (3.0
                            * (rcos.x.powi(2) + rcos.y.powi(2))
                            * ((rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                                + 4.0 * rcos.z.powi(2) * s_or_h[&13].eval(rcos.w))
                            + (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)).powi(2)
                                * s_or_h[&14].eval(rcos.w))
                    + 3.0
                        * (rcos.x.powi(2) + rcos.y.powi(2))
                        * (8.0 * s_or_h[&13].eval(rcos.w)
                            + rcos.w
                                * (rcos.x.powi(2) + rcos.y.powi(2))
                                * s_or_h[&12].deriv(rcos.w)
                            + 4.0 * rcos.w * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w))
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2)).powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 0, 2, 1) => {
            (SQRT3
                * rcos.x
                * ((rcos.x.powi(2) + rcos.y.powi(2))
                    * (-1.0 + 4.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) + rcos.y.powi(2)
                            - (3.0 + 4.0 * rcos.x.powi(2) + 4.0 * rcos.y.powi(2))
                                * rcos.z.powi(2)
                            + 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    - rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 6.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(4) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.z.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 0, 2, 2) => {
            (SQRT3
                * (rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * rcos.z
                * (-4.0
                    * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 8.0 * s_or_h[&13].eval(rcos.w)
                    + 4.0 * s_or_h[&14].eval(rcos.w)
                    + (rcos.x.powi(2) + rcos.y.powi(2))
                        * (4.0 * s_or_h[&14].eval(rcos.w) + rcos.w * s_or_h[&13].deriv(rcos.w))
                    + 2.0
                        * rcos.z.powi(2)
                        * (8.0 * s_or_h[&13].eval(rcos.w) - 4.0 * s_or_h[&14].eval(rcos.w)
                            + rcos.w
                                * (s_or_h[&13].deriv(rcos.w) - 2.0 * s_or_h[&14].deriv(rcos.w)))
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 1, 0, 0) => {
            (SQRT3
                * rcos.x
                * ((1.0 - 2.0 * rcos.z.powi(2)) * s_or_h[&9].eval(rcos.w)
                    + rcos.w * rcos.z.powi(2) * s_or_h[&9].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 1, -1) => {
            (rcos.x
                * rcos.y
                * ((-2.0 + 6.0 * rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.z.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / rcos.w
        }
        (2, 1, 1, 0) => {
            (rcos.x
                * rcos.z
                * (-((2.0 + 3.0 * rcos.x.powi(2) + 3.0 * rcos.y.powi(2) - 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w))
                    + SQRT3 * (2.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w * SQRT3 * rcos.z.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 1, 1) => {
            ((-rcos.x.powi(2)
                + rcos.y.powi(2)
                + 3.0 * (1.0 + rcos.x.powi(2) - rcos.y.powi(2)) * rcos.z.powi(2)
                - 3.0 * rcos.z.powi(4))
                * s_or_h[&10].eval(rcos.w)
                + rcos.w
                    * rcos.z.powi(2)
                    * (-rcos.x.powi(2) + rcos.y.powi(2) + rcos.z.powi(2))
                    * s_or_h[&10].deriv(rcos.w)
                + SQRT3
                    * rcos.x.powi(2)
                    * ((1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&11].eval(rcos.w)
                        + rcos.w * rcos.z.powi(2) * s_or_h[&11].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, -2) => {
            (rcos.x
                * ((-rcos.x.powi(2)
                    + (-3.0 + 4.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                    + 4.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + (-3.0 * rcos.x.powi(2)
                        + rcos.y.powi(2)
                        + (3.0 + 12.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                        - 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, -1) => {
            (rcos.x
                * rcos.y
                * rcos.z
                * (4.0 * (rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&12].eval(rcos.w)
                    - 2.0
                        * (3.0 + 2.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2)
                            - 6.0 * rcos.z.powi(2))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * s_or_h[&14].eval(rcos.w)
                    + rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2))
                        * (-s_or_h[&12].deriv(rcos.w) + s_or_h[&13].deriv(rcos.w))
                    - 3.0
                        * rcos.z.powi(2)
                        * (4.0 * s_or_h[&14].eval(rcos.w)
                            + rcos.w * (s_or_h[&14].deriv(rcos.w) - s_or_h[&15].deriv(rcos.w)))))
                / rcos.w
        }
        (2, 1, 2, 0) => {
            (SQRT3
                * rcos.x
                * ((rcos.x.powi(2) + rcos.y.powi(2))
                    * (-1.0 + 4.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) + rcos.y.powi(2)
                            - (3.0 + 4.0 * rcos.x.powi(2) + 4.0 * rcos.y.powi(2))
                                * rcos.z.powi(2)
                            + 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    - rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 6.0 * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 4.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 8.0 * rcos.z.powi(4) * s_or_h[&14].eval(rcos.w)
                    - rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    - rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - rcos.w
                        * rcos.z.powi(2)
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 1, 2, 1) => {
            (rcos.z
                * (-2.0
                    * (rcos.x.powi(2) + rcos.y.powi(2))
                    * (-1.0 + 2.0 * rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) + 2.0 * rcos.z.powi(2)
                            - 2.0
                                * (rcos.x.powi(4)
                                    + rcos.x.powi(2)
                                        * (1.0 + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                                    + rcos.z.powi(2) * (rcos.x.powi(2) + rcos.z.powi(2))))
                        * s_or_h[&13].eval(rcos.w)
                    + 6.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].deriv(rcos.w)))
                / rcos.w
        }
        (2, 1, 2, 2) => {
            -(rcos.x
                * ((rcos.x.powi(2) + 3.0 * rcos.y.powi(2)
                    - 2.0 * (-3.0 + 2.0 * rcos.x.powi(2) + 6.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                    - 8.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) - 3.0 * rcos.y.powi(2)
                            + (-3.0 - 4.0 * rcos.x.powi(2) + 12.0 * rcos.y.powi(2))
                                * rcos.z.powi(2)
                            + 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    - 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * rcos.z.powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 0, 0) => {
            (SQRT3
                * (rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * rcos.z
                * (-2.0 * s_or_h[&9].eval(rcos.w) + rcos.w * s_or_h[&9].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 1, -1) => {
            (rcos.x
                * rcos.z
                * (2.0
                    * (-2.0 + 6.0 * rcos.x.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + 3.0 * SQRT3 * (-rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    - 2.0
                        * rcos.w
                        * (2.0 * rcos.x.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w
                        * SQRT3
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 1, 0) => {
            ((rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * ((-2.0 + 6.0 * rcos.z.powi(2)) * s_or_h[&10].eval(rcos.w)
                    + SQRT3 * (1.0 - 3.0 * rcos.z.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + rcos.w
                        * rcos.z.powi(2)
                        * (-2.0 * s_or_h[&10].deriv(rcos.w) + SQRT3 * s_or_h[&12].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, 2, 1, 1) => {
            (rcos.x
                * rcos.z
                * (-2.0
                    * (-2.0 + 6.0 * rcos.y.powi(2) + 3.0 * rcos.z.powi(2))
                    * s_or_h[&10].eval(rcos.w)
                    + 3.0 * SQRT3 * (-rcos.x.powi(2) + rcos.y.powi(2)) * s_or_h[&11].eval(rcos.w)
                    + 2.0
                        * rcos.w
                        * (2.0 * rcos.y.powi(2) + rcos.z.powi(2))
                        * s_or_h[&10].deriv(rcos.w)
                    + rcos.w
                        * SQRT3
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * s_or_h[&11].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, -2) => {
            (rcos.x
                * (rcos.x - rcos.y)
                * rcos.y
                * (rcos.x + rcos.y)
                * rcos.z
                * (-4.0 * s_or_h[&12].eval(rcos.w) + 16.0 * s_or_h[&13].eval(rcos.w)
                    - 12.0 * s_or_h[&14].eval(rcos.w)
                    + rcos.w
                        * (s_or_h[&12].deriv(rcos.w) - 4.0 * s_or_h[&13].deriv(rcos.w)
                            + 3.0 * s_or_h[&14].deriv(rcos.w))))
                / (2. * rcos.w)
        }
        (2, 2, 2, -1) => {
            (rcos.x
                * ((3.0 * rcos.x.powi(2) + rcos.y.powi(2)
                    - 2.0 * (-3.0 + 6.0 * rcos.x.powi(2) + 2.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                    - 8.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (-3.0 * rcos.x.powi(2)
                            + rcos.y.powi(2)
                            + (-3.0 + 12.0 * rcos.x.powi(2) - 4.0 * rcos.y.powi(2))
                                * rcos.z.powi(2)
                            + 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    + 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * rcos.z.powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, 0) => {
            (SQRT3
                * (rcos.x - rcos.y)
                * (rcos.x + rcos.y)
                * rcos.z
                * (-4.0
                    * (-1.0 + rcos.x.powi(2) + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                    * s_or_h[&12].eval(rcos.w)
                    - 8.0 * s_or_h[&13].eval(rcos.w)
                    + 4.0 * s_or_h[&14].eval(rcos.w)
                    + (rcos.x.powi(2) + rcos.y.powi(2))
                        * (4.0 * s_or_h[&14].eval(rcos.w) + rcos.w * s_or_h[&13].deriv(rcos.w))
                    + 2.0
                        * rcos.z.powi(2)
                        * (8.0 * s_or_h[&13].eval(rcos.w) - 4.0 * s_or_h[&14].eval(rcos.w)
                            + rcos.w
                                * (s_or_h[&13].deriv(rcos.w) - 2.0 * s_or_h[&14].deriv(rcos.w)))
                    - rcos.w
                        * (rcos.x.powi(2) + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        (2, 2, 2, 1) => {
            -(rcos.x
                * ((rcos.x.powi(2) + 3.0 * rcos.y.powi(2)
                    - 2.0 * (-3.0 + 2.0 * rcos.x.powi(2) + 6.0 * rcos.y.powi(2)) * rcos.z.powi(2)
                    - 8.0 * rcos.z.powi(4))
                    * s_or_h[&12].eval(rcos.w)
                    + 2.0
                        * (rcos.x.powi(2) - 3.0 * rcos.y.powi(2)
                            + (-3.0 - 4.0 * rcos.x.powi(2) + 12.0 * rcos.y.powi(2))
                                * rcos.z.powi(2)
                            + 4.0 * rcos.z.powi(4))
                        * s_or_h[&13].eval(rcos.w)
                    - 3.0 * rcos.x.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 3.0 * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    + 12.0 * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 3.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 2.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 6.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.z.powi(4) * s_or_h[&13].deriv(rcos.w)
                    - 3.0
                        * rcos.w
                        * (rcos.x - rcos.y)
                        * (rcos.x + rcos.y)
                        * rcos.z.powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (2. * rcos.w)
        }
        (2, 2, 2, 2) => {
            (rcos.z
                * (-4.0
                    * (rcos.x.powi(4)
                        - 2.0 * rcos.x.powi(2) * (1.0 + rcos.y.powi(2) - 2.0 * rcos.z.powi(2))
                        + (-2.0 + rcos.y.powi(2) + 2.0 * rcos.z.powi(2))
                            * (rcos.x.powi(2) + 2.0 * rcos.z.powi(2)))
                    * s_or_h[&12].eval(rcos.w)
                    - 8.0
                        * (rcos.x.powi(2) * (-1.0 + 2.0 * rcos.z.powi(2))
                            + rcos.x.powi(2)
                                * (-1.0 + 8.0 * rcos.y.powi(2) + 2.0 * rcos.z.powi(2)))
                        * s_or_h[&13].eval(rcos.w)
                    - 12.0 * rcos.x.powi(4) * s_or_h[&14].eval(rcos.w)
                    + 24.0 * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&14].eval(rcos.w)
                    - 12.0 * rcos.y.powi(4) * s_or_h[&14].eval(rcos.w)
                    + rcos.w * rcos.x.powi(4) * s_or_h[&12].deriv(rcos.w)
                    - 2.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + rcos.w * rcos.y.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&12].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.z.powi(4) * s_or_h[&12].deriv(rcos.w)
                    + 16.0 * rcos.w * rcos.x.powi(2) * rcos.y.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.x.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 4.0 * rcos.w * rcos.y.powi(2) * rcos.z.powi(2) * s_or_h[&13].deriv(rcos.w)
                    + 3.0
                        * rcos.w
                        * (rcos.x.powi(2) - rcos.y.powi(2)).powi(2)
                        * s_or_h[&14].deriv(rcos.w)))
                / (4. * rcos.w)
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!"),
    };
    return array![grad0, grad1, grad2];
}
