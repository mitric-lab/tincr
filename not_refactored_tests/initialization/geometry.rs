use tincr::initialization::geometry::*;

/// Test of distance matrix and proximity matrix of a water molecule. The xyz geometry of the
/// water molecule is
/// ```no_run
/// 3
//
// O          0.34215        1.17577        0.00000
// H          1.31215        1.17577        0.00000
// H          0.01882        1.65996        0.77583
///```
///
///
#[test]
fn test_distance_matrix() {
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let (dist_matrix, dir_matrix, prox_matrix): (Array2<f64>, Array3<f64>, Array2<bool>) =
        build_geometric_matrices(positions.view(), None);

    let dist_matrix_ref: Array2<f64> = array![
        [0.0000000000000000, 1.8330342089215557, 1.8330287870558954],
        [1.8330342089215557, 0.0000000000000000, 2.9933251510242216],
        [1.8330287870558954, 2.9933251510242216, 0.0000000000000000]
    ];

    let direction: Array3<f64> = array![
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [-1.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.3333308828545918, -0.4991664249199420, -0.7998271080477469]
        ],
        [
            [1.0000000000000000, -0.0000000000000000, -0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.8164964343992185, -0.3056755882657617, -0.4897918000045972]
        ],
        [
            [-0.3333308828545918, 0.4991664249199420, 0.7998271080477469],
            [-0.8164964343992185, 0.3056755882657617, 0.4897918000045972],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        ]
    ];
    assert!(dir_matrix.abs_diff_eq(&direction, 1.0e-14));
    assert!(dist_matrix.abs_diff_eq(&dist_matrix_ref, 1e-05));

    let prox_matrix_ref: Array2<bool> =
        array![[true, true, true], [true, true, true], [true, true, true]];
    assert_eq!(prox_matrix, prox_matrix_ref);
}
