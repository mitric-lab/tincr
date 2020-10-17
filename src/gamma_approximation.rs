use std::collections::HashMap;
use std::f32::consts::PI;


/// The decay constants for the gaussian charge fluctuations
/// are determined from the requirement d^2 E_atomic/d n^2 = U_H.
///
/// In the DFTB approximations with long-range correction one has
///
/// U_H = gamma_AA - 1/2 * 1/(2*l+1) gamma^lr_AA
///
/// where l is the angular momentum of the highest valence orbital
///
/// see "Implementation and benchmark of a long-range corrected functional
///          in the DFTB method" by V. Lutsker, B. Aradi and Th. Niehaus
///
/// Here, this equation is solved for sigmaA, the decay constant
/// of a gaussian.
pub fn gaussian_decay(
    hubbard_u: HashMap<u8, f64>,
    valorbs: HashMap<u8, (u8, u8, u8)>,
    r_lr: f64,
    lc_flag: bool,
) -> HashMap<u8, f64> {

    let mut sigmas: HashMap<u8, f64> = HashMap::new();

    // ATTENTION LC_FLAG IS OVERWRITTEN HERE
    let lc_flag = true;
    if lc_flag == false {
        for (z, u) in hubbard_u.iter() {
            sigmas.insert(*z, 1.0 / (*u * PI.sqrt()));
        }
    } else {
        // do something else
        // but it don't understand the dftbaby code
        // at this place, as it always returns sigmas0
        // so this if-else conditions might be superfluous
    }
    return sigmas;
}
