use ndarray::prelude::*;
use crate::initialization::{Atom, Geometry};
use crate::param::slako_transformations::*;
use std::collections::HashMap;
use crate::defaults::PROXIMITY_CUTOFF;
use crate::param::slako::SlaterKoster;


#[cfg(test)]
mod tests {
    use super::*;
    use crate::initialization::Properties;
    use crate::initialization::System;
    use crate::utils::*;
    use approx::AbsDiffEq;

    pub const EPSILON: f64 = 1e-15;

    fn test_h0_and_s(molecule_and_properties: (&str, System, Properties)) {
        let name = molecule_and_properties.0;
        let molecule = molecule_and_properties.1;
        let props = molecule_and_properties.2;
        let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s(molecule.n_orbs, &molecule.atoms,  &molecule.slako);
        let s_ref: Array2<f64> = props.get("S").unwrap().as_array2().unwrap().to_owned();
        let h0_ref: Array2<f64> = props.get("H0").unwrap().as_array2().unwrap().to_owned();

        assert!(
            s_ref.abs_diff_eq(&s, EPSILON),
            "Molecule: {}, S (ref): {}  S: {}",
            name,
            s_ref,
            s
        );

        assert!(
            h0_ref.abs_diff_eq(&h0, EPSILON),
            "Molecule: {}, H0 (ref): {}  H0: {}",
            name,
            h0_ref,
            h0
        );
    }

    #[test]
    fn get_h0_and_s() {
        let names = AVAILAIBLE_MOLECULES;
        for molecule in names.iter() {
            test_h0_and_s(get_molecule(molecule, "no_lc_gs"));
        }
    }

}