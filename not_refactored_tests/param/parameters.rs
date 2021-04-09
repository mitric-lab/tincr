use approx::AbsDiffEq;
use tincr::initialization::parameters::*;

#[test]
fn test_load_free_pseudo_atom() {
    let path: &Path = Path::new("../../src/param/slaterkoster/free_pseudo_atom/h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {pseudo_atom.z, 1};
}

#[test]
fn test_load_confined_pseudo_atom() {
    let path: &Path = Path::new("../../src/param/slaterkoster/confined_pseudo_atom/h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let pseudo_atom: PseudoAtom = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {pseudo_atom.z, 1};
}

#[test]
fn test_load_slako_tables() {
    let path: &Path = Path::new("../../src/param/slaterkoster/slako_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {slako_table.z1, 1};
    assert_eq! {slako_table.z2, 1};
}

#[test]
fn test_load_repulsive_potential_tables() {
    let path: &Path = Path::new("./src/param/repulsive_potential/reppot_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let reppot_table: RepulsivePotentialTable =
        from_str(&data).expect("RON file was not well-formatted");
    assert_eq! {reppot_table.z1, 1};
    assert_eq! {reppot_table.z2, 1};
}

/// import numpy as np
// from DFTB2 import DFTB2
// import XYZ
// import GammaApproximation
// atomlist = XYZ.read_xyz("h2o.xyz")[0]
// dftb = DFTB2(atomlist)
// dftb.setGeometry(atomlist, charge=0)
//
// dftb.getEnergy()
//
// for key, value in dftb.SKT.items():
//     print(key, value)
// d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
// np.set_printoptions(16)
// print(dftb.SKT[(1,1)].S_spl(0, d))
#[test]
fn test_spline_overlap_integrals() {
    let path: &Path = Path::new("../../src/param/slaterkoster/slako_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    let spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = slako_table.spline_overlap();
    let mut y_values: Vec<f64> = Vec::new();
    let x_values: Vec<f64> = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    ];
    for x in x_values {
        y_values.push(splev_uniform(&spline[&0].0, &spline[&0].1, spline[&0].2, x));
    }
    let y_values: Array1<f64> = Array::from_shape_vec((14), y_values).unwrap();
    let y_values_ref: Array1<f64> = array![
        0.9953396476468342,
        0.9812384772492724,
        0.9583521361528490,
        0.9274743570042232,
        0.8895949507497998,
        0.8458257726181956,
        0.7973774727029854,
        0.7454487849069387,
        0.6912337281934855,
        0.6358463027455588,
        0.5803233164667398,
        0.5255800129748242,
        0.4724037942538298,
        0.4214524357395346
    ];
    assert!(y_values.abs_diff_eq(&y_values_ref, 1e-16));
}

#[test]
fn test_spline_h0() {
    let path: &Path = Path::new("../../src/param/slaterkoster/slako_tables/h_h.ron");
    let data: String = fs::read_to_string(path).expect("Unable to read file");
    let slako_table: SlaterKosterTable = from_str(&data).expect("RON file was not well-formatted");
    let spline: HashMap<u8, (Vec<f64>, Vec<f64>, usize)> = slako_table.spline_hamiltonian();
    let mut y_values: Vec<f64> = Vec::new();
    let x_values: Vec<f64> = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    ];
    for x in x_values {
        y_values.push(splev_uniform(&spline[&0].0, &spline[&0].1, spline[&0].2, x));
    }
    let y_values: Array1<f64> = Array::from_shape_vec((14), y_values).unwrap();
    let y_values_ref: Array1<f64> = array![
        -0.7020075123394368,
        -0.6827454396001111,
        -0.6559933747633552,
        -0.6249508412681278,
        -0.5919398382976603,
        -0.5585510127756821,
        -0.5258067834653534,
        -0.4942857974427148,
        -0.4642429835387089,
        -0.4357098171167648,
        -0.4085694762270275,
        -0.3826380560498423,
        -0.3577041260543390,
        -0.3335812815557643
    ];
    assert!(y_values.abs_diff_eq(&y_values_ref, 1e-16));
}
