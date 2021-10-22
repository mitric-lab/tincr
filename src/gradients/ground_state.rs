use core::defaults;
use crate::fmo::scc::helpers::atomvec_to_aomat;
use crate::gradients::helpers::{f_lr, gradient_v_rep};
use core::::*;
use core::::gamma_approximation::{gamma_gradients_ao_wise, gamma_gradients_atomwise};
use ndarray::{Array, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis,s};
use ndarray_einsum_beta::tensordot;
use std::time::Instant;

impl<'a> System<'a> {
    pub fn ground_state_gradient(&mut self, excited_gradients:bool) -> Array1<f64> {
        // for the evaluation of the gradient it is necessary to compute the derivatives
        // of: - H0
        //     - S
        //     - Gamma
        //     - Repulsive Potential
        // the first three properties are calculated here at the beginning and the gradient that
        // originates from the repulsive potential is added at the end to total gradient

        // derivative of H0 and S
        let (grad_s, grad_h0) = self.slako.h0_and_s_gradients(&self.atoms, self.n_orbs());

        // and reshape them into a 2D array. the last two dimension (number of orbitals) are compressed
        // into one dimension to be able to just matrix-matrix products for the computation of the gradient
        let grad_s_2d: ArrayView2<f64> = grad_s.view()
            .into_shape([3 * self.atoms.len(), self.n_orbs() * self.n_orbs()])
            .unwrap();
        let grad_h0_2d: ArrayView2<f64> = grad_h0.view()
            .into_shape([3 * self.atoms.len(), self.n_orbs() * self.n_orbs()])
            .unwrap();

        // derivative of the gamma matrix and transform it in the same way to a 2D array
        let grad_gamma: Array2<f64> =
            gamma_gradients_atomwise(&self.gammafunction, &self.atoms, self.atoms.len())
                .into_shape([3 * self.atoms.len(), self.atoms.len() * self.atoms.len()])
                .unwrap();

        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.data.gamma();
        let p: ArrayView2<f64> = self.data.p();
        let h0: ArrayView2<f64> = self.data.h0();
        let dq: ArrayView1<f64> = self.data.dq();
        let s: ArrayView2<f64> = self.data.s();

        // transform the expression Sum_c_in_X (gamma_AC + gamma_aC) * dq_C
        // into matrix of the dimension (norb, norb) to do an element wise multiplication with P
        let mut coulomb_mat: Array2<f64> =
            atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs(), &self.atoms) * 0.5;

        // The product of the Coulomb interaction matrix and the density matrix flattened as vector.
        let coulomb_x_p: Array1<f64> = (&p * &coulomb_mat)
            .into_shape([self.n_orbs() * self.n_orbs()])
            .unwrap();

        // The density matrix in vector form.
        let p_flat: ArrayView1<f64> = p.into_shape([self.n_orbs() * self.n_orbs()]).unwrap();

        // the gradient part which involves the gradient of the gamma matrix is given by:
        // 1/2 * dq . dGamma / dR . dq
        // the dq's are element wise multiplied into a 2D array and reshaped into a flat one, that
        // has the length of natoms^2. this allows to do only a single matrix vector product of
        // 'grad_gamma' with 'dq_x_dq' and avoids to reshape dGamma multiple times
        let dq_column: ArrayView2<f64> = dq.clone().insert_axis(Axis(1));
        let dq_x_dq: Array1<f64> = (&dq_column.broadcast((self.atoms.len(), self.atoms.len())).unwrap()
            * &dq)
            .into_shape([self.atoms.len() * self.atoms.len()])
            .unwrap();

        // compute the energy weighted density matrix: W = 1/2 * D . (H + H_Coul) . D
        // let w: Array1<f64> = 0.5
        //     * (p.dot(&(&h0 + &(&coulomb_mat * &s))).dot(&p))
        //         .into_shape([self.n_orbs() * self.n_orbs()])
        //         .unwrap();
        let w: Array1<f64> = 0.5 * (p.dot(&self.data.fock()).dot(&p)).into_shape([self.n_orbs() * self.n_orbs()]).unwrap();

        // calculation of the gradient
        // 1st part:  dH0 / dR . P
        let mut gradient: Array1<f64> = grad_h0_2d.dot(&p_flat);

        // 2nd part: dS / dR . W
        gradient -= &grad_s_2d.dot(&w);

        // 3rd part: 1/2 * dS / dR * sum_c_in_X (gamma_ac + gamma_bc) * dq_c
        gradient += &grad_s_2d.dot(&coulomb_x_p);

        // 4th part: 1/2 * dq . dGamma / dR . dq
        gradient += &(grad_gamma.dot(&dq_x_dq));

        // last part: dV_rep / dR
        gradient = gradient + gradient_v_rep(&self.atoms, &self.vrep);

        // long-range corrected part of the gradient
        if self.config.lc.long_range_correction {
            let (g1_lr,g1_lr_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &self.atoms,
                self.atoms.len(),
                self.n_orbs(),
            );

            let diff_p: Array2<f64> = &p - &self.data.p_ref();
            let flr_dmd0:Array3<f64> = f_lr(
                diff_p.view(),
                self.data.s(),
                grad_s.view(),
                self.data.gamma_lr_ao(),
                g1_lr_ao.view(),
                self.atoms.len(),
                self.n_orbs(),
            );
            gradient = gradient
                - 0.25
                    * flr_dmd0.view()
                        .into_shape((3 * self.atoms.len(), self.n_orbs() * self.n_orbs()))
                        .unwrap()
                        .dot(&diff_p.into_shape(self.n_orbs() * self.n_orbs()).unwrap());

            // save necessary properties for the excited gradient calculation with lr-correction
            if excited_gradients{
                self.data.set_grad_gamma_lr(g1_lr);
                self.data.set_grad_gamma_lr_ao(g1_lr_ao);
                self.data.set_flr_dmd0(flr_dmd0);
            }
        }
        // save necessary properties for the excited gradient calculation
        if excited_gradients{
            self.data.set_grad_s(grad_s);
            self.data.set_grad_h0(grad_h0);
            self.data.set_grad_gamma(grad_gamma.into_shape([3 * self.atoms.len(), self.atoms.len(), self.atoms.len()]).unwrap());
        }

        return gradient;
    }
}

// only ground state
// pub fn gradient_lc_gs(
//     molecule: &Molecule,
//     orbe_occ: &Array1<f64>,
//     orbe_virt: &Array1<f64>,
//     orbs_occ: &Array2<f64>,
//     s: &Array2<f64>,
//     r_lc: Option<f64>,
// ) -> (
//     Array1<f64>,
//     Array1<f64>,
//     Array3<f64>,
//     Array3<f64>,
//     Array3<f64>,
//     Array3<f64>,
//     Array3<f64>,
//     Array3<f64>,
//     Array3<f64>,
//     Array3<f64>,
// ) {
//     let grad_timer = Instant::now();
//     let (g1, g1_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
//         &molecule.atomic_numbers.unwrap(),
//         molecule.atoms.len(),
//         molecule.calculator.n_orbs(),
//         molecule.distance_matrix.view(),
//         molecule.directions_matrix.view(),
//         &molecule.calculator.hubbard_u,
//         &molecule.calculator.valorbs,
//         Some(0.0),
//     );
//
//     let (g1lr, g1lr_ao): (Array3<f64>, Array3<f64>) = get_gamma_gradient_matrix(
//         &molecule.atomic_numbers.unwrap(),
//         molecule.atoms.len(),
//         molecule.calculator.n_orbs(),
//         molecule.distance_matrix.view(),
//         molecule.directions_matrix.view(),
//         &molecule.calculator.hubbard_u,
//         &molecule.calculator.valorbs,
//         None,
//     );
//     let n_at: usize = *&molecule.g0.dim().0;
//     let n_orb: usize = *&molecule.g0_ao.dim().0;
//
//     info!(
//         "{:>65} {:>8.3} s",
//         "elapsed time for gammas:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//
//     let grad_timer = Instant::now();
//     let (grad_s, grad_h0): (Array3<f64>, Array3<f64>) = h0_and_s_gradients(
//         &molecule.atomic_numbers.unwrap(),
//         molecule.positions.view(),
//         molecule.calculator.n_orbs(),
//         &molecule.calculator.valorbs,
//         molecule.proximity_matrix.view(),
//         &molecule.calculator.skt,
//         &molecule.calculator.orbital_energies,
//     );
//
//     info!(
//         "{:>65} {:>8.3} s",
//         "elapsed time for h0andS gradients:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//     let grad_timer = Instant::now();
//     let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
//     let ea: Array2<f64> = Array2::from_diag(&orbe_virt);
//
//     // density matrix
//     let d = 2.0 * orbs_occ.dot(&orbs_occ.t());
//     // reference density matrix
//     let d_ref: Array2<f64> = density_matrix_ref(&molecule);
//
//     let diff_d: Array2<f64> = &d - &d_ref;
//     // computing F(D-D0)
//
//     let fdmd0: Array3<f64> = f_v_new(
//         diff_d.view(),
//         s.view(),
//         grad_s.view(),
//         (&molecule.g0_ao).view(),
//         g1_ao.view(),
//         molecule.atoms.len(),
//         molecule.calculator.n_orbs(),
//     );
//
//     info!(
//         "{:>65} {:>8.3} s",
//         "elapsed time for f:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//     let grad_timer = Instant::now();
//
//     let mut flr_dmd0: Array3<f64> = Array::zeros((3 * n_at, n_orb, n_orb));
//     if r_lc.unwrap_or(defaults::LONG_RANGE_RADIUS) > 0.0 {
//         flr_dmd0 = f_lr_new(
//             diff_d.view(),
//             s.view(),
//             grad_s.view(),
//             (&molecule.g0_lr_ao).view(),
//             g1lr_ao.view(),
//             n_at,
//             n_orb,
//         );
//     }
//     info!(
//         "{:>65} {:>8.3} s",
//         "elapsed time for f_lr:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//     let grad_timer = Instant::now();
//     // energy weighted density matrix
//     let d_en = 2.0 * orbs_occ.dot(&ei.dot(&orbs_occ.t()));
//
//     // at the time of writing the code there is no tensordot/einsum functionality availaible
//     // in ndarray or other packages. therefore we use indexed loops at the moment
//     // tensordot grad_h0, d, axes=([1,2], [0,1])
//     // tensordot fdmd0, diff_d, axes=([1,2], [0,1])
//     // tensordot grad_s, d_en, axes=([1,2], [0,1])
//     let mut grad_e0: Array1<f64> = Array1::zeros([3 * molecule.atoms.len()]);
//     for i in 0..(3 * molecule.atoms.len()) {
//         grad_e0[i] += (&grad_h0.slice(s![i, .., ..]) * &d).sum();
//         grad_e0[i] += 0.5 * (&fdmd0.slice(s![i, .., ..]) * &diff_d).sum();
//         grad_e0[i] -= (&grad_s.slice(s![i, .., ..]) * &d_en).sum();
//     }
//     if r_lc.unwrap_or(defaults::LONG_RANGE_RADIUS) > 0.0 {
//         grad_e0 = grad_e0
//             - 0.25
//             * tensordot(&flr_dmd0, &diff_d, &[Axis(1), Axis(2)], &[Axis(0), Axis(1)])
//             .into_dimensionality::<Ix1>()
//             .unwrap();
//     }
//
//     info!(
//         "{:>65} {:>8.3} s",
//         "time tensordots gradients:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//     let grad_timer = Instant::now();
//     let grad_v_rep: Array1<f64> = gradient_v_rep(
//         &molecule.atomic_numbers,
//         molecule.distance_matrix.view(),
//         molecule.directions_matrix.view(),
//         &molecule.calculator.v_rep,
//     );
//
//     info!(
//         "{:>65} {:>8.3} s",
//         "time grad_v_rep gradients:",
//         grad_timer.elapsed().as_secs_f32()
//     );
//     drop(grad_timer);
//
//     return (
//         grad_e0, grad_v_rep, grad_s, grad_h0, fdmd0, flr_dmd0, g1, g1_ao, g1lr, g1lr_ao,
//     );
