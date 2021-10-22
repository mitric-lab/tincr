use ndarray::{Array1, Array2, Array3, Array};
use core::::*;
use crate::gradients::ground_state::*;
use core::::scc_routine;
use core::::scc_routine::RestrictedSCC;

impl<'a> System<'a>{
    pub fn calculate_gradients(&mut self,state:usize)->(Array1<f64>,Array1<f64>){
        // ground state scc routine
        self.prepare_scc();
        self.run_scc();

        let mut ground_state_gradient:Array1<f64> = Array1::zeros(3*self.atoms.len());
        let mut excited_state_gradient:Array1<f64> = Array1::zeros(3*self.atoms.len());

        if state == 0{
            // calculate ground state gradients
            ground_state_gradient = self.ground_state_gradient(false);
        }
        if state > 0{
            // TODO: insert excited state calculation here

            // prepare excited gradient calculation
            // the transition charges and the ground state gradient calculation
            // are mandatory
            ground_state_gradient = self.ground_state_gradient(true);
            self.prepare_excited_grad();

            // calculate the excited state gradient
            if self.config.lc.long_range_correction{
                excited_state_gradient = self.excited_state_gradient_lc(state-1);
            }
            else{
                excited_state_gradient = self.excited_state_gradient_no_lc(state-1);
            }
        }
        return (ground_state_gradient,excited_state_gradient);
    }
}

// pub fn get_gradients(
//     orbe: &Array1<f64>,
//     orbs: &Array2<f64>,
//     s: &Array2<f64>,
//     molecule: &Molecule,
//     XmY: &Option<Array3<f64>>,
//     XpY: &Option<Array3<f64>>,
//     exc_state: Option<usize>,
//     omega: &Option<Array1<f64>>,
//     old_z_vec: Option<Array3<f64>>,
// ) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array3<f64>) {
//     info!("{:^80}", "");
//     info!("{:^80}", "Calculating analytic gradient");
//     info!("{:-^80}", "");
//     let grad_timer = Instant::now();
//     let n_at: usize = molecule.atoms.len();
//     let active_occ: Vec<usize> = molecule.calculator.active_occ.clone().unwrap();
//     let active_virt: Vec<usize> = molecule.calculator.active_virt.clone().unwrap();
//     let full_occ: Vec<usize> = molecule.calculator.full_occ.clone().unwrap();
//     let full_virt: Vec<usize> = molecule.calculator.full_virt.clone().unwrap();
//
//     let n_occ: usize = active_occ.len();
//     let n_virt: usize = active_virt.len();
//
//     let n_occ_full: usize = full_occ.len();
//     let n_virt_full: usize = full_virt.len();
//
//     let r_lr = molecule
//         .calculator
//         .r_lr
//         .unwrap_or(defaults::LONG_RANGE_RADIUS);
//
//     let mut grad_e0: Array1<f64> = Array::zeros((3 * n_at));
//     let mut grad_ex: Array1<f64> = Array::zeros((3 * n_at));
//     let mut grad_vrep: Array1<f64> = Array::zeros((3 * n_at));
//     let mut new_z_vec: Array3<f64> = Array3::zeros((n_occ, n_virt, 1));
//
//     // check if active space is smaller than full space
//     // otherwise this part is unnecessary
//     if (n_occ + n_virt) < orbe.len() {
//         // set arrays of orbitals energies of the active space
//         //let orbe_occ: Array1<f64> = active_occ
//         //    .iter()
//         //    .map(|&active_occ| orbe[active_occ])
//         //    .collect();
//         //let orbe_virt: Array1<f64> = active_virt
//         //    .iter()
//         //    .map(|&active_virt| orbe[active_virt])
//         //    .collect();
//
//         //let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ));
//         //let mut orbs_virt: Array2<f64> = Array::zeros((orbs.dim().0, n_virt));
//
//         //for (i,index) in active_occ.iter().enumerate() {
//         //    orbs_occ
//         //        .slice_mut(s![.., i])
//         //        .assign(&orbs.column(*index));
//         //}
//
//         let orbe_occ: Array1<f64> = full_occ.iter().map(|&full_occ| orbe[full_occ]).collect();
//         let orbe_virt: Array1<f64> = full_virt.iter().map(|&full_virt| orbe[full_virt]).collect();
//
//         let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ_full));
//         let mut orbs_virt: Array2<f64> = Array::zeros((orbs.dim().0, n_virt_full));
//
//         for (i, index) in full_occ.iter().enumerate() {
//             orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
//         }
//
//         let (gradE0, grad_v_rep, grad_s, grad_h0, fdmdO, flrdmdO, g1, g1_ao, g1lr, g1lr_ao): (
//             Array1<f64>,
//             Array1<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//         ) = gradient_lc_gs(&molecule, &orbe_occ, &orbe_virt, &orbs_occ, s, Some(r_lr));
//
//         // set values for return of the gradients
//         grad_e0 = gradE0;
//         grad_vrep = grad_v_rep;
//
//         // if an excited state is specified in the input, calculate gradients for it
//         // otherwise just return ground state
//         if exc_state.is_some() {
//             for (i, index) in full_virt.iter().enumerate() {
//                 orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
//             }
//
//             let nstates: usize = XmY.as_ref().unwrap().dim().0;
//             // get transition charges of the complete range of orbitals
//             let (qtrans_ov, qtrans_oo, qtrans_vv): (Array3<f64>, Array3<f64>, Array3<f64>) =
//                 trans_charges(
//                     &molecule.atomic_numbers,
//                     &molecule.calculator.valorbs,
//                     orbs.view(),
//                     s.view(),
//                     &full_occ[..],
//                     &full_virt[..],
//                 );
//
//             // construct XY matrices for active space
//             let mut XmY_active: Array3<f64> = Array::zeros((nstates, n_occ_full, n_virt_full));
//             let mut XpY_active: Array3<f64> = Array::zeros((nstates, n_occ_full, n_virt_full));
//             for n in 0..nstates {
//                 for (i, occ) in active_occ.iter().enumerate() {
//                     for (j, virt) in active_virt.iter().enumerate() {
//                         XmY_active
//                             .slice_mut(s![n, *occ, *virt - n_occ_full])
//                             .assign(&XmY.as_ref().unwrap().slice(s![n, i, j]));
//                         XpY_active
//                             .slice_mut(s![n, *occ, *virt - n_occ_full])
//                             .assign(&XpY.as_ref().unwrap().slice(s![n, i, j]));
//                     }
//                 }
//             }
//             //check for lc correction
//             if r_lr > 0.0 {
//                 let tmp: (Array1<f64>, Array3<f64>) = gradients_lc_ex(
//                     exc_state.unwrap(),
//                     (&molecule.g0).view(),
//                     g1.view(),
//                     (&molecule.g0_ao).view(),
//                     g1_ao.view(),
//                     (&molecule.g0_lr).view(),
//                     g1lr.view(),
//                     (&molecule.g0_lr_ao).view(),
//                     g1lr_ao.view(),
//                     s.view(),
//                     grad_s.view(),
//                     grad_h0.view(),
//                     XmY_active.view(),
//                     XpY_active.view(),
//                     omega.as_ref().unwrap().view(),
//                     qtrans_oo.view(),
//                     qtrans_vv.view(),
//                     qtrans_ov.view(),
//                     orbe_occ,
//                     orbe_virt,
//                     orbs_occ.view(),
//                     orbs_virt.view(),
//                     fdmdO.view(),
//                     flrdmdO.view(),
//                     molecule.multiplicity,
//                     molecule.calculator.spin_couplings.view(),
//                     None,
//                     old_z_vec,
//                 );
//                 grad_ex = tmp.0;
//                 new_z_vec = tmp.1;
//             } else {
//                 let tmp: (Array1<f64>, Array3<f64>) = gradients_nolc_ex(
//                     exc_state.unwrap(),
//                     (&molecule.g0).view(),
//                     g1.view(),
//                     (&molecule.g0_ao).view(),
//                     g1_ao.view(),
//                     (&molecule.g0_lr).view(),
//                     g1lr.view(),
//                     (&molecule.g0_lr_ao).view(),
//                     g1lr_ao.view(),
//                     s.view(),
//                     grad_s.view(),
//                     grad_h0.view(),
//                     XmY_active.view(),
//                     XpY_active.view(),
//                     omega.as_ref().unwrap().view(),
//                     qtrans_oo.view(),
//                     qtrans_vv.view(),
//                     qtrans_ov.view(),
//                     orbe_occ,
//                     orbe_virt,
//                     orbs_occ.view(),
//                     orbs_virt.view(),
//                     fdmdO.view(),
//                     molecule.multiplicity,
//                     molecule.calculator.spin_couplings.view(),
//                     None,
//                     old_z_vec,
//                 );
//                 grad_ex = tmp.0;
//                 new_z_vec = tmp.1;
//             }
//         }
//     } else {
//         //println!("Full active space");
//         // no active space, use full range of orbitals
//
//         let orbe_occ: Array1<f64> = full_occ.iter().map(|&full_occ| orbe[full_occ]).collect();
//         let orbe_virt: Array1<f64> = full_virt.iter().map(|&full_virt| orbe[full_virt]).collect();
//
//         let mut orbs_occ: Array2<f64> = Array::zeros((orbs.dim().0, n_occ));
//         let mut orbs_virt: Array2<f64> = Array::zeros((orbs.dim().0, n_virt));
//
//         for (i, index) in full_occ.iter().enumerate() {
//             orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
//         }
//
//         let (gradE0, grad_v_rep, grad_s, grad_h0, fdmdO, flrdmdO, g1, g1_ao, g1lr, g1lr_ao): (
//             Array1<f64>,
//             Array1<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//             Array3<f64>,
//         ) = gradient_lc_gs(&molecule, &orbe_occ, &orbe_virt, &orbs_occ, s, Some(r_lr));
//
//         // set values for return of the gradients
//         grad_e0 = gradE0;
//         grad_vrep = grad_v_rep;
//
//         if exc_state.is_some() {
//             for (i, index) in full_virt.iter().enumerate() {
//                 orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
//             }
//             let nstates: usize = XmY.as_ref().unwrap().dim().0;
//             // get transition charges of the complete range of orbitals
//             let (qtrans_ov, qtrans_oo, qtrans_vv): (Array3<f64>, Array3<f64>, Array3<f64>) =
//                 trans_charges(
//                     &molecule.atomic_numbers,
//                     &molecule.calculator.valorbs,
//                     orbs.view(),
//                     s.view(),
//                     &full_occ[..],
//                     &full_virt[..],
//                 );
//
//             if r_lr > 0.0 {
//                 let tmp: (Array1<f64>, Array3<f64>) = gradients_lc_ex(
//                     exc_state.unwrap(),
//                     (&molecule.g0).view(),
//                     g1.view(),
//                     (&molecule.g0_ao).view(),
//                     g1_ao.view(),
//                     (&molecule.g0_lr).view(),
//                     g1lr.view(),
//                     (&molecule.g0_lr_ao).view(),
//                     g1lr_ao.view(),
//                     s.view(),
//                     grad_s.view(),
//                     grad_h0.view(),
//                     XmY.as_ref().unwrap().view(),
//                     XpY.as_ref().unwrap().view(),
//                     omega.as_ref().unwrap().view(),
//                     qtrans_oo.view(),
//                     qtrans_vv.view(),
//                     qtrans_ov.view(),
//                     orbe_occ,
//                     orbe_virt,
//                     orbs_occ.view(),
//                     orbs_virt.view(),
//                     fdmdO.view(),
//                     flrdmdO.view(),
//                     molecule.multiplicity,
//                     molecule.calculator.spin_couplings.view(),
//                     None,
//                     old_z_vec,
//                 );
//                 grad_ex = tmp.0;
//                 new_z_vec = tmp.1;
//             } else {
//                 let tmp: (Array1<f64>, Array3<f64>) = gradients_nolc_ex(
//                     exc_state.unwrap(),
//                     (&molecule.g0).view(),
//                     g1.view(),
//                     (&molecule.g0_ao).view(),
//                     g1_ao.view(),
//                     (&molecule.g0_lr).view(),
//                     g1lr.view(),
//                     (&molecule.g0_lr_ao).view(),
//                     g1lr_ao.view(),
//                     s.view(),
//                     grad_s.view(),
//                     grad_h0.view(),
//                     XmY.as_ref().unwrap().view(),
//                     XpY.as_ref().unwrap().view(),
//                     omega.as_ref().unwrap().view(),
//                     qtrans_oo.view(),
//                     qtrans_vv.view(),
//                     qtrans_ov.view(),
//                     orbe_occ,
//                     orbe_virt,
//                     orbs_occ.view(),
//                     orbs_virt.view(),
//                     fdmdO.view(),
//                     molecule.multiplicity,
//                     molecule.calculator.spin_couplings.view(),
//                     None,
//                     old_z_vec,
//                 );
//                 grad_ex = tmp.0;
//                 new_z_vec = tmp.1;
//             }
//         }
//     }
//     let total_grad: Array2<f64> = (&grad_e0 + &grad_vrep + &grad_ex)
//         .into_shape([molecule.atoms.len(), 3])
//         .unwrap();
//     if log_enabled!(Level::Debug) || molecule.config.jobtype == "force" {
//         info!("{: <45} ", "Gradient in atomic units");
//         info!(
//             "{: <4} {: >18} {: >18} {: >18}",
//             "Atom", "dE/dx", "dE/dy", "dE/dz"
//         );
//         info!("{:-^61} ", "");
//         for (grad_xyz, at) in total_grad.outer_iter().zip(molecule.atomic_numbers.iter()) {
//             info!(
//                 "{: <4} {:>18.10e} {:>18.10e} {:>18.10e}",
//                 ATOM_NAMES[*at as usize], grad_xyz[0], grad_xyz[1], grad_xyz[2]
//             );
//         }
//         info!("{:-^61} ", "");
//     }
//     info!(
//         "{:<25} {:>18.10e}",
//         "Max gradient component:",
//         total_grad.max().unwrap()
//     );
//     info!(
//         "{:<25} {:>18.10e}",
//         "RMS gradient:",
//         total_grad.root_mean_sq_err(&(&total_grad * 0.0)).unwrap()
//     );
//     info!("{:-^80} ", "");
//     info!(
//         "{:>68} {:>8.2} s",
//         "elapsed time:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     info!("{:^80} ", "");
//     drop(grad_timer);
//     return (grad_e0, grad_vrep, grad_ex, new_z_vec);
// }