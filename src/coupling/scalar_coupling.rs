use crate::initialization::{Atom,System};
use ndarray::prelude::*;

impl System{
    pub fn get_scalar_coupling(
        &self,
        dt:f64,
        old_atoms:&[Atom],
        old_orbs:ArrayView2<f64>,
        old_ci_coeff:ArrayView2<f64>,
    ){
        let n_states:usize = self.config.excited.nstates;
        // scalar coupling matrix
        let s_ci:Array2<f64> = self.ci_overlap(old_atoms,old_orbs,old_ci_coeff,n_states);
        // align phases
        // The eigenvalue solver produces vectors with arbitrary global phases
        // (+1 or -1). The orbitals of the ground state can also change their signs.
        // Eigen states from neighbouring geometries should change continuously.
        let diag = s_ci.diag();
        // get signs of the diagonal
        let mut sign:Array1<f64> = Array1::zeros(diag.len());
        diag.iter().enumerate().for_each(|(idx,val)|
            if val.is_sign_positive(){
                sign[idx] = 1.0;
            }else{
                sign[idx] = -1.0;
            }
        );
        let p:Array2<f64> = Array::from_diag(&sign);
        // align the new CI coefficients with the old coefficients
        let p_exclude_gs:ArrayView2<f64> = p.slice(s![1..,1..]);
        let aligned_coeff:ArrayView2<f64> = self.properties.ci_coefficients().unwrap()
            .dot(&p_exclude_gs);

        // align overlap matrix
        let s_ci = s_ci.dot(&p);

        // The relative signs for the overlap between the ground and excited states at different geometries
        // cannot be deduced from the diagonal elements of Sci. The phases are chosen such that the coupling
        // between S0 and S1-SN changes smoothly for most of the states.
    }
}