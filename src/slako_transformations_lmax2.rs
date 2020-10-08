use ndarray::ArrayView1;

const SQRT3: f64 = 1.7320508075688772;
/// transformation rules for matrix elements
fn slako_transformation(
    //r: f64,
    x: f64,
    y: f64,
    z: f64,
    sor_h: ArrayView1<f64>,
    l1: u8,
    m1: u8,
    l2: u8,
    m2: u8,
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
    let number: usize = l1 * 1000 + (lmax + m1) * 100 + l2 * 10 + (lmax + m2);
    let res: f64 = match number {
        202 => sor_h[0],                                                   // (0,0,0,0)
        211 => y * sor_h[2],                                               // (0,0,1,-1)
        212 => z * sor_h[2],                                               // (0,0,1,0)
        213 => x * sor_h[2],                                               // (0,0,1,1)
        220 => x * y * sor_h[3] * SQRT3,                                   // (0,0,2,-2)
        221 => y * z * sor_h[3] * SQRT3,                                   // (0,0,2,-1)
        222 => -((x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h(4)) / 2., // (0,0,2,0)
        223 => x * z * sor_h[3] * SQRT3,                                   // (0,0,2,1)
        224 => ((x - y) * (x + y) * sor_h[3] * SQRT3) / 2.,                // (0,0,2,2)
        1102 => y * sor_h[4],                                              // (1,-1,0,0)
        1111 => (x.powi(2) + z.powi(2)) * sor_h[5] + y.powi(2) * sor_h[6],  // (1,-1,1,-1)
        1112 => y * z * (-sor_h[5] + sor_h[6]),                             // (1,-1,1,0)
        1113 => x * y * (-sor_h[5] + sor_h[6]),                             // (1,-1,1,1)
        1120 => x * ((x.powi(2) - y.powi(2) + z.powi(2)) * sor_h[7] + y.powi(2) * sor_h[7] * SQRT3), // (1,-1,2,-2)
        1121 => z * ((x.powi(2) - y.powi(2) + z.powi(2)) * sor_h[7] + y.powi(2) * sor_h[8] * SQRT3), // (1,-1,2,-1)
        1122 => {
            -(y * ((x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[8]
                + 2 * z.powi(2) * sor_h[7] * SQRT3))
                / 2.
        } // (1,-1,2,0)
        1123 => x * y * z * (-2 * sor_h[7] + sor_h[8] * SQRT3), // (1,-1,2,1)
        1124 => {
            -(y * (2 * (2 * x.powi(2) + z.powi(2)) * sor_h[7]
                + (-x.powi(2) + y.powi(2)) * sor_h[8] * SQRT3))
                / 2.
        } // (1,-1,2,2)
        1202 => z * sor_h[4],                                  // (1,0,0,0)
        1211 => y * z * (-sor_h[5] + sor_h[6]),                 // (1,0,1,-1)
        1212 => (x.powi(2) + y.powi(2)) * sor_h[5] + z.powi(2) * sor_h[6], // (1,0,1,0)
        1213 => x * z * (-sor_h[5] + sor_h[6]),                 // (1,0,1,1)
        1220 => x * y * z * (-2 * sor_h[7] + sor_h[8] * SQRT3), // (1,0,2,-2)
        1221 => y * ((x.powi(2) + y.powi(2) - z.powi(2)) * sor_h[7] + z.powi(2) * sor_h[8] * SQRT3), // (1,0,2,-1)
        1222 => {
            z * *3 * sor_h[8] - ((x.powi(2) + y.powi(2)) * z * (sor_h[8] - 2 * sor_h[7] * SQRT3)) / 2.
        } // (1,0,2,0)
        1223 => x * ((x.powi(2) + y.powi(2) - z.powi(2)) * sor_h[7] + z.powi(2) * sor_h[8] * SQRT3), // (1,0,2,1)
        1224 => -((x - y) * (x + y) * z * (2 * sor_h[7] - sor_h[8] * SQRT3)) / 2., // (1,0,2,2)
        1302 => x * sor_h[4],                                                     // (1,1,0,0)
        1311 => x * y * (-sor_h[5] + sor_h[6]),                                    // (1,1,1,-1)
        1312 => x * z * (-sor_h[5] + sor_h[6]),                                    // (1,1,1,0)
        1313 => (y.powi(2) + z.powi(2)) * sor_h[5] + x.powi(2) * sor_h[6],         // (1,1,1,1)
        1320 => y * ((-x.powi(2) + y.powi(2) + z.powi(2)) * sor_h[7] + x.powi(2) * sor_h[8] * SQRT3), // (1,1,2,-2)
        1321 => x * y * z * (-2 * sor_h[7] + sor_h[8] * SQRT3), // (1,1,2,-1)
        1322 => {
            -(x * ((x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[8]
                + 2 * z.powi(2) * sor_h[7] * SQRT3))
                / 2.
        } // (1,1,2,0)
        1323 => z * ((-x.powi(2) + y.powi(2) + z.powi(2)) * sor_h[7] + x.powi(2) * sor_h[8] * SQRT3), // (1,1,2,1)
        1324 => {
            x * (2 * y.powi(2) + z.powi(2)) * sor_h[7]
                + (x * (x - y) * (x + y) * sor_h[8] * SQRT3) / 2.
        } // (1,1,2,2)
        2002 => x * y * sor_h[9] * SQRT3, // (2,-2,0,0)
        2011 => x * ((x.powi(2) - y.powi(2) + z.powi(2)) * sor_h[10] + y.powi(2) * sor_h[11] * SQRT3), // (2,-2,1,-1)
        2012 => x * y * z * (-2 * sor_h[10] + sor_h[11] * SQRT3), // (2,-2,1,0)
        2013 => {
            y * ((-x.powi(2) + y.powi(2) + z.powi(2)) * sor_h[10] + x.powi(2) * sor_h[11] * SQRT3)
        } // (2,-2,1,1)
        2020 => {
            (x.powi(2) + z.powi(2)) * (y.powi(2) + z.powi(2)) * sor_h[12]
                + ((x.powi(2) - y.powi(2)).powi(2) + (x.powi(2) + y.powi(2)) * z.powi(2)) * sor_h[13]
                + 3 * x.powi(2) * y.powi(2) * sor_h[14]
        } // (2,-2,2,-2)
        2021 => {
            x * z
                * (-((x.powi(2) + z.powi(2)) * sor_h[12])
                    + (x.powi(2) - 3 * y.powi(2) + z.powi(2)) * sor_h[13]
                    + 3 * y.powi(2) * sor_h[14])
        } // (2,-2,2,-1)
        2022 => {
            (x * y
                * ((x.powi(2) + y.powi(2) + 2 * z.powi(2)) * sor_h[12]
                    - 4 * z.powi(2) * sor_h[13]
                    - (x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[14])
                * SQRT3)
                / 2.
        } // (2,-2,2,0)
        2023 => {
            y * z
                * (-((y.powi(2) + z.powi(2)) * (sor_h[12] - sor_h[13]))
                    + 3 * x.powi(2) * (-sor_h[13] + sor_h[14]))
        } // (2,-2,2,1)
        2024 => (x * (x - y) * y * (x + y) * (sor_h[12] - 4 * sor_h[13] + 3 * sor_h[14])) / 2., // (2,-2,2,2)
        2102 => y * z * sor_h[9] * SQRT3, // (2,-1,0,0)
        2111 => z * ((x.powi(2) - y.powi(2) + z.powi(2)) * sor_h[10] + y.powi(2) * sor_h[11] * SQRT3), // (2,-1,1,-1)
        2112 => y * ((x.powi(2) + y.powi(2) - z.powi(2)) * sor_h[10] + z.powi(2) * sor_h[11] * SQRT3), // (2,-1,1,0)
        2113 => x * y * z * (-2 * sor_h[10] + sor_h[11] * SQRT3), // (2,-1,1,1)
        2120 => {
            x * z
                * (-((x.powi(2) + z.powi(2)) * sor_h[12])
                    + (x.powi(2) - 3 * y.powi(2) + z.powi(2)) * sor_h[13]
                    + 3 * y.powi(2) * sor_h[14])
        } // (2,-1,2,-2)
        2121 => {
            (x.powi(2) + y.powi(2)) * (x.powi(2) + z.powi(2)) * sor_h[12]
                + ((y.powi(2) - z.powi(2)).powi(2) + x.powi(2) * (y.powi(2) + z.powi(2))) * sor_h[13]
                + 3 * y.powi(2) * z.powi(2) * sor_h[14]
        } // (2,-1,2,-1)
        2122 => {
            -(y * z
                * ((x.powi(2) + y.powi(2)) * sor_h[12]
                    - 2 * (x.powi(2) + y.powi(2) - z.powi(2)) * sor_h[13]
                    + (x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[14])
                * SQRT3)
                / 2.
        } // (2,-1,2,0)
        2123 => {
            x * y
                * (-((x.powi(2) + y.powi(2)) * sor_h[12])
                    + (x.powi(2) + y.powi(2) - 3 * z.powi(2)) * sor_h[13]
                    + 3 * z.powi(2) * sor_h[14])
        } // (2,-1,2,1)
        2124 => {
            (y * z
                * ((3 * x.powi(2) + y.powi(2) + 2 * z.powi(2)) * sor_h[12]
                    - 2 * (3 * x.powi(2) - y.powi(2) + z.powi(2)) * sor_h[13]
                    + 3 * (x - y) * (x + y) * sor_h[14]))
                / 2.
        } // (2,-1,2,2)
        2202 => -((x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[9]) / 2., // (2,0,0,0)
        2211 => {
            -(y * ((x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[11]
                + 2 * z.powi(2) * sor_h[10] * SQRT3))
                / 2.
        } // (2,0,1,-1)
        2212 => {
            z * *3 * sor_h[11]
                - ((x.powi(2) + y.powi(2)) * z * (sor_h[11] - 2 * sor_h[10] * SQRT3)) / 2.
        } // (2,0,1,0)
        2213 => {
            -(x * ((x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[11]
                + 2 * z.powi(2) * sor_h[10] * SQRT3))
                / 2.
        } // (2,0,1,1)
        2220 => {
            (x * y
                * ((x.powi(2) + y.powi(2) + 2 * z.powi(2)) * sor_h[12]
                    - 4 * z.powi(2) * sor_h[13]
                    - (x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[14])
                * SQRT3)
                / 2.
        } // (2,0,2,-2)
        2221 => {
            -(y * z
                * ((x.powi(2) + y.powi(2)) * sor_h[12]
                    - 2 * (x.powi(2) + y.powi(2) - z.powi(2)) * sor_h[13]
                    + (x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[14])
                * SQRT3)
                / 2.
        } // (2,0,2,-1)
        2222 => {
            (3 * (x.powi(2) + y.powi(2)).powi(2) * sor_h[12]
                + 12 * (x.powi(2) + y.powi(2)) * z.powi(2) * sor_h[13]
                + (x.powi(2) + y.powi(2) - 2 * z.powi(2)).powi(2) * sor_h[14])
                / 4.
        } // (2,0,2,0)
        2223 => {
            -(x * z
                * ((x.powi(2) + y.powi(2)) * sor_h[12]
                    - 2 * (x.powi(2) + y.powi(2) - z.powi(2)) * sor_h[13]
                    + (x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[14])
                * SQRT3)
                / 2.
        } // (2,0,2,1)
        2224 => {
            ((x - y)
                * (x + y)
                * ((x.powi(2) + y.powi(2) + 2 * z.powi(2)) * sor_h[12]
                    - 4 * z.powi(2) * sor_h[13]
                    - (x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[14])
                * SQRT3)
                / 4.
        } // (2,0,2,2)
        2302 => x * z * sor_h[9] * SQRT3,                        // (2,1,0,0)
        2311 => x * y * z * (-2 * sor_h[10] + sor_h[11] * SQRT3), // (2,1,1,-1)
        2312 => x * ((x.powi(2) + y.powi(2) - z.powi(2)) * sor_h[10] + z.powi(2) * sor_h[11] * SQRT3), // (2,1,1,0)
        2313 => {
            z * ((-x.powi(2) + y.powi(2) + z.powi(2)) * sor_h[10] + x.powi(2) * sor_h[11] * SQRT3)
        } // (2,1,1,1)
        2320 => {
            y * z
                * (-((y.powi(2) + z.powi(2)) * (sor_h[12] - sor_h[13]))
                    + 3 * x.powi(2) * (-sor_h[13] + sor_h[14]))
        } // (2,1,2,-2)
        2321 => {
            x * y
                * (-((x.powi(2) + y.powi(2)) * sor_h[12])
                    + (x.powi(2) + y.powi(2) - 3 * z.powi(2)) * sor_h[13]
                    + 3 * z.powi(2) * sor_h[14])
        } // (2,1,2,-1)
        2322 => {
            -(x * z
                * ((x.powi(2) + y.powi(2)) * sor_h[12]
                    - 2 * (x.powi(2) + y.powi(2) - z.powi(2)) * sor_h[13]
                    + (x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[14])
                * SQRT3)
                / 2.
        } // (2,1,2,0)
        2323 => {
            (x.powi(2) + y.powi(2)) * (y.powi(2) + z.powi(2)) * sor_h[12]
                + (x * *4
                    + x.powi(2) * (y.powi(2) - 2 * z.powi(2))
                    + z.powi(2) * (y.powi(2) + z.powi(2)))
                    * sor_h[13]
                + 3 * x.powi(2) * z.powi(2) * sor_h[14]
        } // (2,1,2,1)
        2324 => {
            -(x * z
                * ((x.powi(2) + 3 * y.powi(2) + 2 * z.powi(2)) * sor_h[12]
                    + 2 * (x.powi(2) - 3 * y.powi(2) - z.powi(2)) * sor_h[13]
                    + 3 * (-x.powi(2) + y.powi(2)) * sor_h[14]))
                / 2.
        } // (2,1,2,2)
        2402 => ((x - y) * (x + y) * sor_h[9] * SQRT3) / 2., // (2,2,0,0)
        2411 => {
            -(y * (2 * (2 * x.powi(2) + z.powi(2)) * sor_h[10]
                + (-x.powi(2) + y.powi(2)) * sor_h[11] * SQRT3))
                / 2.
        } // (2,2,1,-1)
        2412 => -((x - y) * (x + y) * z * (2 * sor_h[10] - sor_h[11] * SQRT3)) / 2., // (2,2,1,0)
        2413 => {
            x * (2 * y.powi(2) + z.powi(2)) * sor_h[10]
                + (x * (x - y) * (x + y) * sor_h[11] * SQRT3) / 2.
        } // (2,2,1,1)
        2420 => (x * (x - y) * y * (x + y) * (sor_h[12] - 4 * sor_h[13] + 3 * sor_h[14])) / 2., // (2,2,2,-2)
        2421 => {
            (y * z
                * ((3 * x.powi(2) + y.powi(2) + 2 * z.powi(2)) * sor_h[12]
                    - 2 * (3 * x.powi(2) - y.powi(2) + z.powi(2)) * sor_h[13]
                    + 3 * (x - y) * (x + y) * sor_h[14]))
                / 2.
        } // (2,2,2,-1)
        2422 => {
            ((x - y)
                * (x + y)
                * ((x.powi(2) + y.powi(2) + 2 * z.powi(2)) * sor_h[12]
                    - 4 * z.powi(2) * sor_h[13]
                    - (x.powi(2) + y.powi(2) - 2 * z.powi(2)) * sor_h[14])
                * SQRT3)
                / 4.
        } // (2,2,2,0)
        2423 => {
            -(x * z
                * ((x.powi(2) + 3 * y.powi(2) + 2 * z.powi(2)) * sor_h[12]
                    + 2 * (x.powi(2) - 3 * y.powi(2) - z.powi(2)) * sor_h[13]
                    + 3 * (-x.powi(2) + y.powi(2)) * sor_h[14]))
                / 2.
        } // (2,2,2,1)
        2424 => {
            (((x - y).powi(2) + 2 * z.powi(2)) * ((x + y).powi(2) + 2 * z.powi(2)) * sor_h[12]
                + 4 * (4 * x.powi(2) * y.powi(2) + (x.powi(2) + y.powi(2)) * z.powi(2)) * sor_h[13]
                + 3 * (x.powi(2) - y.powi(2)).powi(2) * sor_h[14])
                / 4.
        } // (2,2,2,2)
        _ => panic!(
            "BUG: No case for i=",
            i, " which encodes (l1,m1,l2,m2)=(", l1, ",", m1, ",", l2, ",", m2, ") //"
        ),
    };
    return res;
}
