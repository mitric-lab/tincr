use crate::initialization::{Atom,System};
use ndarray::prelude::*;
use crate::coupling::helpers::get_sign_of_array;

impl System{
    pub fn get_scalar_coupling(
        &self,
        dt:f64,
        old_atoms:&[Atom],
        old_orbs:ArrayView2<f64>,
        old_ci_coeff:ArrayView2<f64>,
        old_s_ci:ArrayView2<f64>,
    )->(Array2<f64>,Array2<f64>){
        // TODO: save and load arrays from the previous iteration.
        let n_states:usize = self.config.excited.nstates;
        // scalar coupling matrix
        let s_ci:Array2<f64> = self.ci_overlap(old_atoms,old_orbs,old_ci_coeff,n_states);
        // align phases
        // The eigenvalue solver produces vectors with arbitrary global phases
        // (+1 or -1). The orbitals of the ground state can also change their signs.
        // Eigen states from neighbouring geometries should change continuously.
        let diag = s_ci.diag();
        // get signs of the diagonal
        let sign:Array1<f64> = get_sign_of_array(diag);

        let p:Array2<f64> = Array::from_diag(&sign);
        // align the new CI coefficients with the old coefficients
        let p_exclude_gs:ArrayView2<f64> = p.slice(s![1..,1..]);
        let aligned_coeff:Array2<f64> = self.properties.ci_coefficients().unwrap()
            .dot(&p_exclude_gs);

        // align overlap matrix
        let mut s_ci = s_ci.dot(&p);

        // The relative signs for the overlap between the ground and excited states at different geometries
        // cannot be deduced from the diagonal elements of Sci. The phases are chosen such that the coupling
        // between S0 and S1-SN changes smoothly for most of the states.
        let s:Array1<f64> = get_sign_of_array(
            (&old_s_ci.slice(s![0,1..])/&s_ci.slice(s![0,1..])).view());
        let w:Array1<f64> = (&old_s_ci.slice(s![0,1..]) - &s_ci.slice(s![0,1..])).map(|val| val.abs());
        let mean_sign:f64 = ((&w*&s).sum()/w.sum()).signum();
        for i in (1..n_states){
            s_ci[[0,i]] *= mean_sign;
            s_ci[[i,0]] *= mean_sign;
        }

        // coupl[A,B] = <Psi_A(t)|Psi_B(t+dt)> - delta_AB
        //            ~ <Psi_A(t)|d/dR Psi_B(t)>*dR/dt dt
        // The scalar coupling matrix should be more or less anti-symmetric
        // provided the time-step is small enough
        // set diagonal elements of coupl to zero
        let mut coupling:Array2<f64> = s_ci.clone();
        coupling = coupling - Array::from_diag(&s_ci.diag());

        // Because of the finite time-step it will not be completely antisymmetric,
        coupling = 0.5 * (&coupling - &coupling.t());

        // coupl = <Psi_A|d/dR Psi_B>*dR/dt * dt
        coupling = coupling / dt;

        return (coupling, s_ci);
    }
}