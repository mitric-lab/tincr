use ndarray::prelude::*;
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use std::ops::AddAssign;
use approx::AbsDiffEq;

// computes the Mulliken transition charges between occupied-occupied
// occupied-virtual and virtual-virtual molecular orbitals.
// Point charge approximation of transition densities according to formula (14)
// in Heringer, Niehaus  J Comput Chem 28: 2589-2601 (2007)
pub fn trans_charges(
    atomic_numbers: &[u8],
    valorbs: &HashMap<u8, Vec<(i8, i8, i8)>>,
    orbs: ArrayView2<f64>,
    s: ArrayView2<f64>,
    active_occupied_orbs:&[usize],
    active_virtual_orbs: &[usize],
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let n_atoms: usize = atomic_numbers.len();
    let dim_o: usize = active_occupied_orbs.len();
    let dim_v: usize = active_virtual_orbs.len();
    // transition charges between occupied and virutal orbitals
    let mut q_trans_ov: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_v]);
    // transition charges between occupied and occupied orbitals
    let mut q_trans_oo: Array3<f64> = Array3::zeros([n_atoms, dim_o, dim_o]);
    // transition charges between virtual and virtual orbitals
    let mut q_trans_vv: Array3<f64> = Array3::zeros([n_atoms, dim_v, dim_v]);

    let s_c: Array2<f64> = s.dot(&orbs);

    let mut mu: usize = 0;
    for (atom_a, z_a) in atomic_numbers.iter().enumerate() {
        for _ in valorbs[z_a].iter() {
            // occupied - virtuals
            for (i, occi) in active_occupied_orbs.iter().enumerate() {
                for (a, virta) in active_virtual_orbs.iter().enumerate() {
                    q_trans_ov.slice_mut(s![atom_a, i, a]).add_assign(
                        0.5 * (orbs[[mu, *occi]] * s_c[[mu, *virta]]
                            + orbs[[mu, *virta]] * s_c[[mu, *occi]]),
                    );
                }
            }
            // occupied - occupied
            for (i, occi) in active_occupied_orbs.iter().enumerate() {
                for (j, occj) in active_occupied_orbs.iter().enumerate() {
                    q_trans_oo.slice_mut(s![atom_a, i, j]).add_assign(
                        0.5 * (orbs[[mu, *occi]] * s_c[[mu, *occj]]
                            + orbs[[mu, *occj]] * s_c[[mu, *occi]]),
                    );
                }
            }
            // virtual - virtual
            for (a, virta) in active_virtual_orbs.iter().enumerate() {
                for (b, virtb) in active_virtual_orbs.iter().enumerate() {
                    q_trans_vv.slice_mut(s![atom_a, a, b]).add_assign(
                        0.5 * (orbs[[mu, *virta]] * s_c[[mu, *virtb]]
                            + orbs[[mu, *virtb]] * s_c[[mu, *virta]]),
                    );
                }
            }
            mu += 1;
        }
    }
    return (q_trans_ov, q_trans_oo, q_trans_vv);
}


#[test]
fn transition_charges() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut valorbs: HashMap<u8, Vec<(i8, i8, i8)>> = HashMap::new();
    valorbs.insert(1, vec![(0, 0, 0)]);
    valorbs.insert(8, vec![(1, 0, 0), (1, 1, -1), (1, 1, 0), (1, 1, 1)]);
    let orbs: Array2<f64> = array![
    [-8.6192166213060994e-01, -1.2181834219010016e-06, -2.9726616765213620e-01,
     -2.3769036691661326e-16,  4.3206301264811484e-05,  6.5350522392460086e-01],
    [ 2.6759804936282508e-03, -2.0080575784062019e-01, -3.6133279553418873e-01,
      8.4834397825097274e-01,  2.8113801144299444e-01, -2.8862999414972668e-01],
    [ 4.2877918717272520e-03, -3.2175619303473502e-01, -5.7897276432658473e-01,
     -5.2944545948125077e-01,  4.5047461413457190e-01, -4.6247920932109815e-01],
    [ 3.5738995440957322e-03,  5.3637385901663792e-01, -4.8258396378917212e-01,
     -1.4744753593534519e-16, -7.5103114541738114e-01, -3.8540480519437487e-01],
    [-1.7926008848716424e-01, -3.6381155905729773e-01,  2.3852270484454788e-01,
      1.3358923998855696e-16, -7.2394067400302153e-01, -7.7069603748475568e-01],
    [-1.7926090288676341e-01,  3.6381118097628601e-01,  2.3852143176360416e-01,
      2.5698277778340866e-16,  7.2383558631110978e-01, -7.7079819502973201e-01]
    ];
    let active_occupied_orbs: Vec<usize> = vec![2, 3];
    let active_virtual_orbs: Vec<usize> = vec![4, 5];
    let s: Array2<f64> = array![
    [ 1.0000000000000000,  0.0000000000000000,  0.0000000000000000,
      0.0000000000000000,  0.3074918525690681,  0.3074937992389065],
    [ 0.0000000000000000,  1.0000000000000000,  0.0000000000000000,
      0.0000000000000000,  0.0000000000000000, -0.1987769748092704],
    [ 0.0000000000000000,  0.0000000000000000,  1.0000000000000000,
      0.0000000000000000,  0.0000000000000000, -0.3185054221819456],
    [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
      1.0000000000000000, -0.3982160222204482,  0.1327383036929333],
    [ 0.3074918525690681,  0.0000000000000000,  0.0000000000000000,
     -0.3982160222204482,  1.0000000000000000,  0.0268024699984349],
    [ 0.3074937992389065, -0.1987769748092704, -0.3185054221819456,
      0.1327383036929333,  0.0268024699984349,  1.0000000000000000]
    ];
    let q_trans_ov_ref: Array3<f64> = array![
    [[ 2.6230764031964782e-05,  3.7065733463488038e-01],
     [-4.9209998651226938e-17,  2.3971084358783751e-16]],
    [[-1.7348142939318700e-01, -1.8531691862558541e-01],
     [-7.2728474862656226e-17, -7.7779165808212125e-17]],
    [[ 1.7345519862915512e-01, -1.8534041600929513e-01],
     [ 1.5456547682172723e-16, -1.6527399530138889e-16]]
     ];
    let q_trans_oo_ref: Array3<f64> = array![
    [[ 8.3509500972984507e-01, -3.0814858028948981e-16],
     [-3.0814858028948981e-16,  9.9999999999999978e-01]],
    [[ 8.2452864978581231e-02,  3.8129127163009314e-17],
     [ 3.8129127163009314e-17,  1.6846288898245608e-32]],
    [[ 8.2452125291573627e-02,  7.8185267908421217e-17],
     [ 7.8185267908421217e-17,  7.2763969108729995e-32]]
    ];
    let q_trans_vv_ref: Array3<f64> = array![
    [[ 4.1303771372197096e-01, -5.9782394554452889e-06],
     [-5.9782394554452889e-06,  3.2642696006563388e-01]],
    [[ 2.9352476622180407e-01,  3.1439790351905961e-01],
     [ 3.1439790351905961e-01,  3.3674286510673440e-01]],
    [[ 2.9343752005622487e-01, -3.1439192527960413e-01],
     [-3.1439192527960413e-01,  3.3683017482763289e-01]]
    ];
    let (q_trans_ov, q_trans_oo, q_trans_vv): (Array3<f64>, Array3<f64>, Array3<f64>) =
    trans_charges(&atomic_numbers, &valorbs, orbs.view(), s.view(), &active_occupied_orbs, &active_virtual_orbs);
    assert!(q_trans_ov.abs_diff_eq(&q_trans_ov_ref, 1e-14));
    assert!(q_trans_oo.abs_diff_eq(&q_trans_oo_ref, 1e-14));
    assert!(q_trans_vv.abs_diff_eq(&q_trans_vv_ref, 1e-14));
}