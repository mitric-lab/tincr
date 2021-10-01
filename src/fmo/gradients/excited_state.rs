use crate::fmo::{Monomer, Pair, SuperSystem, ExcitedStateMonomerGradient};
use crate::excited_states::{trans_charges};
use crate::gradients::helpers::{f_lr, f_v, h_minus, h_plus_no_lr, zvector_lc, zvector_no_lc, Hplus, HplusType, h_a_nolr, tda_zvector_no_lc, Hav, tda_zvector_lc};
use crate::utils::ToOwnedF;
use ndarray::{s, Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, Axis};
use ndarray_linalg::{into_col, into_row, IntoTriangular, Solve, UPLO};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_gradients_ao_wise, gamma_ao_wise, gamma_ao_wise_from_gamma_atomwise};
use crate::scc::h0_and_s::h0_and_s_gradients;

impl ExcitedStateMonomerGradient for Monomer{
    fn prepare_excited_gradient(&mut self,atoms:&[Atom]){
        // check if occ and virt indices exist
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        if (self.properties.contains_key("occ_indices") == false) || (self.properties.contains_key("virt_indices") == true) {
            // calculate the number of electrons
            let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
            // get the indices of the occupied and virtual orbitals
            (0..self.n_orbs).for_each(|index| {
                if index < (n_elec / 2) {
                    occ_indices.push(index)
                } else {
                    virt_indices.push(index)
                }
            });

            self.properties.set_occ_indices(occ_indices.clone());
            self.properties.set_virt_indices(virt_indices.clone());
        }
        else{
            occ_indices = self.properties.occ_indices().unwrap().to_vec();
            virt_indices = self.properties.virt_indices().unwrap().to_vec();
        }
        // calculate transition charges if they don't exist
        if self.properties.contains_key("q_ov") == false{
            let tmp: (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
                self.n_atoms,
                atoms,
                self.properties.orbs().unwrap(),
                self.properties.s().unwrap(),
                &occ_indices,
                &virt_indices,
            );

            self.properties.set_q_ov(tmp.0);
            self.properties.set_q_oo(tmp.1);
            self.properties.set_q_vv(tmp.2);
        }

        // prepare the grad gamma_lr ao matrix
        if self.gammafunction_lc.is_some(){
            // calculate the gamma gradient matrix in AO basis
            let (g1_lr,g1_lr_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.properties.set_grad_gamma_lr_ao(g1_lr_ao);
        }
        // prepare gamma and grad gamma AO matrix
        let g0_ao:Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
            self.properties.gamma().unwrap(),
            atoms,
            self.n_orbs
        );
        let (g1,g1_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
            &self.gammafunction,
            atoms,
            self.n_atoms,
            self.n_orbs,
        );
        self.properties.set_grad_gamma(g1);
        self.properties.set_gamma_ao(g0_ao);
        self.properties.set_grad_gamma_ao(g1_ao);

        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&atoms, self.n_orbs, &self.slako);
        self.properties.set_grad_s(grad_s);
        self.properties.set_grad_h0(grad_h0);
    }

    fn tda_gradient_nolc(&mut self, state: usize)->Array1<f64>{
        // get occ and virt indices from properties
        let occ_indices:&[usize] = self.properties.occ_indices().unwrap();
        let virt_indices:&[usize] = self.properties.virt_indices().unwrap();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // excitation energy of the state
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states:usize = omega_state.len();
        let omega_state:f64 = omega_state[state];
        // take state specific values from the excitation vectors
        let x_state:ArrayView3<f64> = self.properties.ci_coefficients().unwrap()
            .into_shape([n_states,n_occ,n_virt]).unwrap();
        let x_state:ArrayView2<f64> = x_state.slice(s![state,..,..]);


        // calculate the vectors u, v and t
        // vectors U, V and T
        let u_ab: Array2<f64> = 2.0 *x_state.t().dot(&x_state);
        let u_ij: Array2<f64> = 2.0 *x_state.dot(&x_state.t());
        let v_ab: Array2<f64> = 2.0 *ei.dot(&x_state).t().dot(&x_state);
        let v_ij: Array2<f64> = 2.0 *x_state.dot(&ea).dot(&x_state.t());
        let t_ab: Array2<f64> = x_state.t().dot(&x_state);
        let t_ij: Array2<f64> = x_state.dot(&x_state.t());

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = self.properties.q_ov().unwrap().into_shape((self.n_atoms,n_occ,n_virt)).unwrap();
        let qtrans_oo: ArrayView3<f64> = self.properties.q_oo().unwrap().into_shape((self.n_atoms,n_occ,n_occ)).unwrap();
        let qtrans_vv: ArrayView3<f64> = self.properties.q_vv().unwrap().into_shape((self.n_atoms,n_virt,n_virt)).unwrap();

        // set gamma matrix
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> =
            h_a_nolr(g0, qtrans_oo, qtrans_vv, t_ab.view());
        let hplus_tij: Array2<f64> =
            h_a_nolr(g0, qtrans_oo, qtrans_oo, t_ij.view());

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> = x_state
            .dot(&h_a_nolr(g0, qtrans_vv, qtrans_ov, x_state).t());
        q_ia = q_ia + h_a_nolr(g0, qtrans_ov, qtrans_vv, t_ab.view());
        q_ia = q_ia - h_a_nolr(g0, qtrans_ov, qtrans_oo, t_ij.view());

        // calculate q_ai
        let q_ai: Array2<f64> = x_state.t().dot(&h_a_nolr(
            g0,
            qtrans_oo,
            qtrans_ov,
            x_state,
        ));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape((n_occ * n_virt)).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = tda_zvector_no_lc(
            omega_input.view(),
            r_matrix.view(),
            g0,
            qtrans_ov,
            1,
            Array1::zeros(self.n_atoms).view(),
        );

        // calculate w_ij
        let mut w_ij: Array2<f64> =
            q_ij + h_a_nolr(g0, qtrans_oo, qtrans_ov, z_ia.view());
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia); //+ h_a_nolr(g0, qtrans_ov, qtrans_ov, z_ia.view());

        // w_ai
        let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
        }

        // build w matrix: combine w_ij, w_ia, w_ai and w_ab
        let length: usize = n_occ + n_virt;
        let mut w_matrix: Array2<f64> = Array::zeros((length, length));
        for i in 0..w_ij.dim().0 {
            w_matrix
                .slice_mut(s![i, ..w_ij.dim().1])
                .assign(&w_ij.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![i, w_ij.dim().1..])
                .assign(&w_ia.slice(s![i, ..]));
        }
        for i in 0..w_ai.dim().0 {
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
                .assign(&w_ai.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
                .assign(&w_ab.slice(s![i, ..]));
        }

        // get arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let grad_h: Array3<f64> = &grad_h + &f_dmd0;

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut gradExc: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        gradExc = gradExc
            + grad_h
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(
                &(t_vv - t_oo + z_ao)
                    .into_shape(self.n_orbs * self.n_orbs)
                    .unwrap(),
            );
        // - gradS * W
        gradExc = gradExc
            - grad_s
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        gradExc = gradExc
            + 2.0
            * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&x_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());

        return gradExc;
    }

    fn tda_gradient_lc(&mut self, state: usize)->Array1<f64>{
        // get occ and virt indices from properties
        let occ_indices:&[usize] = self.properties.occ_indices().unwrap();
        let virt_indices:&[usize] = self.properties.virt_indices().unwrap();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // excitation energy of the state
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states:usize = omega_state.len();
        let omega_state:f64 = omega_state[state];
        // take state specific values from the excitation vectors
        let x_state:ArrayView3<f64> = self.properties.ci_coefficients().unwrap()
            .into_shape([n_states,n_occ,n_virt]).unwrap();
        let x_state:ArrayView2<f64> = x_state.slice(s![state,..,..]);

        // calculate the vectors u, v and t
        // vectors U, V and T
        let u_ab: Array2<f64> = 2.0 *x_state.t().dot(&x_state);
        let u_ij: Array2<f64> = 2.0 *x_state.dot(&x_state.t());
        let v_ab: Array2<f64> = 2.0 *ei.dot(&x_state).t().dot(&x_state);
        let v_ij: Array2<f64> = 2.0 *x_state.dot(&ea).dot(&x_state.t());
        let t_ab: Array2<f64> = x_state.t().dot(&x_state);
        let t_ij: Array2<f64> = x_state.dot(&x_state.t());

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = self.properties.q_ov().unwrap().into_shape((self.n_atoms,n_occ,n_virt)).unwrap();
        let qtrans_oo: ArrayView3<f64> = self.properties.q_oo().unwrap().into_shape((self.n_atoms,n_occ,n_occ)).unwrap();
        let qtrans_vv: ArrayView3<f64> = self.properties.q_vv().unwrap().into_shape((self.n_atoms,n_virt,n_virt)).unwrap();
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hav: Hav = Hav::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrix
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> = hav.compute(g0, g0_lr, t_ab.view(), HplusType::Tab);
        let hplus_tij: Array2<f64> = hav.compute(g0, g0_lr, t_ij.view(), HplusType::Tij);

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> = x_state.dot(
            &hav
                .compute(g0, g0_lr, x_state, HplusType::Qia_Xpy)
                .t(),
        );
        q_ia = q_ia + hav.compute(g0, g0_lr, t_ab.view(), HplusType::Qia_Tab);
        q_ia = q_ia - hav.compute(g0, g0_lr, t_ij.view(), HplusType::Qia_Tij);

        // calculate q_ai
        let mut q_ai: Array2<f64> =
            x_state
                .t()
                .dot(&hav.compute(g0, g0_lr, x_state, HplusType::Qai));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape((n_occ * n_virt)).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = tda_zvector_lc(
            omega_input.view(),
            r_matrix.view(),
            g0,
            g0_lr,
            qtrans_oo,
            qtrans_vv,
            qtrans_ov,
            1,
            Array1::zeros(self.n_atoms).view(),
        );

        // calculate w_ij
        let mut w_ij: Array2<f64> =
            q_ij + hav.compute(g0, g0_lr, z_ia.view(), HplusType::Wij);
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia); //+ h_a_nolr(g0, qtrans_ov, qtrans_ov, z_ia.view());

        // w_ai
        let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
        }

        // build w matrix: combine w_ij, w_ia, w_ai and w_ab
        let length: usize = n_occ + n_virt;
        let mut w_matrix: Array2<f64> = Array::zeros((length, length));
        for i in 0..w_ij.dim().0 {
            w_matrix
                .slice_mut(s![i, ..w_ij.dim().1])
                .assign(&w_ij.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![i, w_ij.dim().1..])
                .assign(&w_ia.slice(s![i, ..]));
        }
        for i in 0..w_ai.dim().0 {
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
                .assign(&w_ai.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
                .assign(&w_ab.slice(s![i, ..]));
        }

        // get arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_dmd0:Array3<f64> = f_lr(
            diff_p.view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        let grad_h: Array3<f64> = &grad_h + &f_dmd0- 0.5 * &flr_dmd0;

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            x_ao.t(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut gradExc: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        gradExc = gradExc
            + grad_h
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(
                &(t_vv - t_oo + z_ao)
                    .into_shape(self.n_orbs * self.n_orbs)
                    .unwrap(),
            );
        // - gradS * W
        gradExc = gradExc
            - grad_s
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        gradExc = gradExc
            + 2.0
            * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&x_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - sum (X) F_lr (X)(X)
        gradExc = gradExc
            - flr_p
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&x_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        return gradExc;
    }

    fn excited_gradient_lc(&mut self, state:usize)->Array1<f64>{
        // get occ and virt indices from properties
        let occ_indices:&[usize] = self.properties.occ_indices().unwrap();
        let virt_indices:&[usize] = self.properties.virt_indices().unwrap();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // take state specific values from the excitation vectors
        let xmy_state: ArrayView3<f64> = self
            .properties
            .xmy()
            .unwrap();
        let xmy_state: ArrayView2<f64> = xmy_state.slice(s![state,..,..]);
        let xpy_state: ArrayView3<f64> = self
            .properties
            .xpy()
            .unwrap();
        let xpy_state: ArrayView2<f64> = xpy_state.slice(s![state,..,..]);
        // excitation energy of the state
        let omega_state: f64 = self.properties.tddft_eigenvalues().unwrap()[state];

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = xpy_state.t().dot(&xmy_state) + xmy_state.t().dot(&xpy_state);
        let u_ij: Array2<f64> = xpy_state.dot(&xmy_state.t()) + xmy_state.dot(&xpy_state.t());

        let v_ab: Array2<f64> =
            ei.dot(&xpy_state).t().dot(&xpy_state) + ei.dot(&xmy_state).t().dot(&xmy_state);
        let v_ij: Array2<f64> =
            xpy_state.dot(&ea).dot(&xpy_state.t()) + xmy_state.dot(&ea).dot(&xmy_state.t());

        let t_ab: Array2<f64> =
            0.5 * (xpy_state.t().dot(&xpy_state) + xmy_state.t().dot(&xmy_state));
        let t_ij: Array2<f64> =
            0.5 * (xpy_state.dot(&xpy_state.t()) + xmy_state.dot(&xmy_state.t()));

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = self.properties.q_ov().unwrap().into_shape((self.n_atoms,n_occ,n_virt)).unwrap();
        let qtrans_oo: ArrayView3<f64> = self.properties.q_oo().unwrap().into_shape((self.n_atoms,n_occ,n_occ)).unwrap();
        let qtrans_vv: ArrayView3<f64> = self.properties.q_vv().unwrap().into_shape((self.n_atoms,n_virt,n_virt)).unwrap();
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hplus: Hplus = Hplus::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());
        // set gamma matrices
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> = hplus.compute(g0, g0_lr, t_ab.view(), HplusType::Tab);
        let hplus_tij: Array2<f64> = hplus.compute(g0, g0_lr, t_ij.view(), HplusType::Tij);

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> = xpy_state.dot(
            &hplus
                .compute(g0, g0_lr, xpy_state, HplusType::Qia_Xpy)
                .t(),
        );
        q_ia = q_ia
            + xmy_state.dot(
            &h_minus(
                g0_lr,
                qtrans_vv,
                qtrans_vo.view(),
                qtrans_vo.view(),
                qtrans_vv,
                xmy_state,
            )
                .t(),
        );
        q_ia = q_ia + hplus.compute(g0, g0_lr, t_ab.view(), HplusType::Qia_Tab);
        q_ia = q_ia - hplus.compute(g0, g0_lr, t_ij.view(), HplusType::Qia_Tij);

        // calculate q_ai
        let mut q_ai: Array2<f64> =
            xpy_state
                .t()
                .dot(&hplus.compute(g0, g0_lr, xpy_state, HplusType::Qai));
        q_ai = q_ai
            + xmy_state.t().dot(&h_minus(
            g0_lr,
            qtrans_ov,
            qtrans_oo,
            qtrans_oo,
            qtrans_ov,
            xmy_state,
        ));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape((n_occ * n_virt)).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = zvector_lc(
            omega_input.view(),
            r_matrix.view(),
            g0,
            g0_lr,
            qtrans_oo,
            qtrans_vv,
            qtrans_ov,
            1,
            Array1::zeros(self.n_atoms).view(),
        );

        // calculate w_ij
        let mut w_ij: Array2<f64> = q_ij + hplus.compute(g0, g0_lr, z_ia.view(), HplusType::Wij);
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
        }

        // build w matrix: combine w_ij, w_ia, w_ai and w_ab
        let length: usize = n_occ + n_virt;
        let mut w_matrix: Array2<f64> = Array::zeros((length, length));
        for i in 0..w_ij.dim().0 {
            w_matrix
                .slice_mut(s![i, ..w_ij.dim().1])
                .assign(&w_ij.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![i, w_ij.dim().1..])
                .assign(&w_ia.slice(s![i, ..]));
        }
        for i in 0..w_ai.dim().0 {
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
                .assign(&w_ai.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
                .assign(&w_ab.slice(s![i, ..]));
        }

        // get arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate F[diff_p] and F_lr[diff_p] matrix
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_dmd0:Array3<f64> = f_lr(
            diff_p.view(),
            s,
            grad_s,
            self.properties.gamma_lr_ao().unwrap(),
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );
        // calculate gradH: gradH0 + gradHexc
        let grad_h: Array3<f64> = &grad_h + &(f_dmd0 - 0.5 * flr_dmd0);

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));
        let xmy_ao: Array2<f64> = orbs_occ.dot(&xmy_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            xpy_ao.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            (&xpy_ao + &xpy_ao.t()).view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_m = -f_lr(
            (&xmy_ao - &xmy_ao.t()).view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut gradExc: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        gradExc = gradExc
            + grad_h
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(
                &(t_vv - t_oo + z_ao)
                    .into_shape(self.n_orbs * self.n_orbs)
                    .unwrap(),
            );
        // - gradS * W
        gradExc = gradExc
            - grad_s
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        gradExc = gradExc
            + 2.0
            * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&xpy_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X+Y) F_lr (X+Y)(X+Y)
        gradExc = gradExc
            - 0.5
            * flr_p
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&xpy_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - 0.5 * sum (X-Y) F_lr (X-Y)(X-Y)
        gradExc = gradExc
            - 0.5
            * flr_m
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&xmy_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());

        return gradExc;
    }
    fn excited_gradient_no_lc(&mut self, state: usize)->Array1<f64> {
        // get occ and virt indices from properties
        let occ_indices:&[usize] = self.properties.occ_indices().unwrap();
        let virt_indices:&[usize] = self.properties.virt_indices().unwrap();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // take state specific values from the excitation vectors
        let xmy_state: ArrayView3<f64> = self
            .properties
            .xmy()
            .unwrap();
        let xmy_state: ArrayView2<f64> = xmy_state.slice(s![state,..,..]);
        let xpy_state: ArrayView3<f64> = self
            .properties
            .xpy()
            .unwrap();
        let xpy_state: ArrayView2<f64> = xpy_state.slice(s![state,..,..]);
        // excitation energy of the state
        let omega_state: f64 = self.properties.tddft_eigenvalues().unwrap()[state];

        // calculate the vectors u, v and t
        let u_ab: Array2<f64> = xpy_state.t().dot(&xmy_state) + xmy_state.t().dot(&xpy_state);
        let u_ij: Array2<f64> = xpy_state.dot(&xmy_state.t()) + xmy_state.dot(&xpy_state.t());

        let v_ab: Array2<f64> =
            ei.dot(&xpy_state).t().dot(&xpy_state) + ei.dot(&xmy_state).t().dot(&xmy_state);
        let v_ij: Array2<f64> =
            xpy_state.dot(&ea).dot(&xpy_state.t()) + xmy_state.dot(&ea).dot(&xmy_state.t());

        let t_ab: Array2<f64> =
            0.5 * (xpy_state.t().dot(&xpy_state) + xmy_state.t().dot(&xmy_state));
        let t_ij: Array2<f64> =
            0.5 * (xpy_state.dot(&xpy_state.t()) + xmy_state.dot(&xmy_state.t()));

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = self.properties.q_ov().unwrap().into_shape((self.n_atoms,n_occ,n_virt)).unwrap();
        let qtrans_oo: ArrayView3<f64> = self.properties.q_oo().unwrap().into_shape((self.n_atoms,n_occ,n_occ)).unwrap();
        let qtrans_vv: ArrayView3<f64> = self.properties.q_vv().unwrap().into_shape((self.n_atoms,n_virt,n_virt)).unwrap();

        // set gamma matrix
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> =
            h_plus_no_lr(g0, qtrans_oo, qtrans_vv, t_ab.view());
        let hplus_tij: Array2<f64> =
            h_plus_no_lr(g0, qtrans_oo, qtrans_oo, t_ij.view());

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> = xpy_state
            .dot(&h_plus_no_lr(g0, qtrans_vv, qtrans_ov, xpy_state).t());
        q_ia = q_ia + h_plus_no_lr(g0, qtrans_ov, qtrans_vv, t_ab.view());
        q_ia = q_ia - h_plus_no_lr(g0, qtrans_ov, qtrans_oo, t_ij.view());

        // calculate q_ai
        let q_ai: Array2<f64> = xpy_state.t().dot(&h_plus_no_lr(
            g0,
            qtrans_oo,
            qtrans_ov,
            xpy_state,
        ));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape((n_occ * n_virt)).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = zvector_no_lc(
            omega_input.view(),
            r_matrix.view(),
            g0,
            qtrans_ov,
            1,
            Array1::zeros(self.n_atoms).view(),
        );

        // calculate w_ij
        let mut w_ij: Array2<f64> =
            q_ij + h_plus_no_lr(g0, qtrans_oo, qtrans_ov, z_ia.view());
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia);

        // w_ai
        let w_ai: Array2<f64> = w_ia.clone().reversed_axes();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
        }

        // build w matrix: combine w_ij, w_ia, w_ai and w_ab
        let length: usize = n_occ + n_virt;
        let mut w_matrix: Array2<f64> = Array::zeros((length, length));
        for i in 0..w_ij.dim().0 {
            w_matrix
                .slice_mut(s![i, ..w_ij.dim().1])
                .assign(&w_ij.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![i, w_ij.dim().1..])
                .assign(&w_ia.slice(s![i, ..]));
        }
        for i in 0..w_ai.dim().0 {
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
                .assign(&w_ai.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
                .assign(&w_ab.slice(s![i, ..]));
        }

        // get arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let grad_h: Array3<f64> = &grad_h + &f_dmd0;

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let xpy_ao: Array2<f64> = orbs_occ.dot(&xpy_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            xpy_ao.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut gradExc: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        gradExc = gradExc
            + grad_h
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(
                &(t_vv - t_oo + z_ao)
                    .into_shape(self.n_orbs * self.n_orbs)
                    .unwrap(),
            );
        // - gradS * W
        gradExc = gradExc
            - grad_s
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        gradExc = gradExc
            + 2.0
            * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&xpy_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());

        return gradExc;
    }
}

impl Pair{
    pub fn prepare_ct_state(
        &mut self,
        atoms:&[Atom],
        m_i:&Monomer,
        m_j:&Monomer,
        ct_ind_i: usize,
        ct_ind_j: usize,
        ct_energy:f64,
        hole_i:bool,
    ){
        // check if occ and virt indices exist
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        if (self.properties.contains_key("occ_indices") == false) || (self.properties.contains_key("virt_indices") == true) {
            // calculate the number of electrons
            let n_elec: usize = atoms.iter().fold(0, |n, atom| n + atom.n_elec);
            // get the indices of the occupied and virtual orbitals
            (0..self.n_orbs).for_each(|index| {
                if index < (n_elec / 2) {
                    occ_indices.push(index)
                } else {
                    virt_indices.push(index)
                }
            });

            self.properties.set_occ_indices(occ_indices.clone());
            self.properties.set_virt_indices(virt_indices.clone());
        }
        else{
            occ_indices = self.properties.occ_indices().unwrap().to_vec();
            virt_indices = self.properties.virt_indices().unwrap().to_vec();
        }
        let nocc:usize = occ_indices.len();
        let nvirt:usize = virt_indices.len();
        let nocc_i:usize = m_i.properties.occ_indices().unwrap().len();
        let nocc_j:usize = m_j.properties.occ_indices().unwrap().len();
        let nvirt_i:usize = m_i.properties.virt_indices().unwrap().len();
        let nvirt_j:usize = m_j.properties.virt_indices().unwrap().len();

        // set ct_energy
        let cis_energy:Array1<f64> = Array::from(vec!(ct_energy));
        // save in properties
        self.properties.set_ci_eigenvalues(cis_energy);

        // get the overlap matrices between the monomers and the dimer
        let s_i_ij:ArrayView2<f64> = self.properties.s_i_ij().unwrap();
        let s_j_ij:ArrayView2<f64> = self.properties.s_j_ij().unwrap();

        // if the hole is on fragment I
        if hole_i == true{
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_occ:ArrayView2<f64> = s_i_ij.slice(s![..nocc_i,..nocc]);
            let s_j_ij_virt:ArrayView2<f64> = s_j_ij.slice(s![nocc_j..,nocc..]);

            // Create matrix for the CT. Only the matrix element, which corresponds to the charge
            // transfer is set to the value 1.0. Everything else is set to null.
            let mut ct_coefficients:Array2<f64> = Array2::zeros([nocc_i,nvirt_j]);
            ct_coefficients[[nocc_i-1-ct_ind_i,ct_ind_j]] = 1.0;

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff:Array2<f64> = s_i_ij_occ.t().dot(&ct_coefficients.dot(&s_j_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff:Array3<f64> = Array3::zeros([1,nocc,nvirt]);
            cis_coeff.slice_mut(s![0,..,..]).assign(&transformed_ct_coeff);
            // save in properties
            self.properties.set_ci_coefficients(cis_coeff.into_shape([1,nocc*nvirt]).unwrap());
        }
        // if the hole is on fragment J
        else{
            // reduce overlap matrices to occupied and virtual blocks
            let s_i_ij_virt:ArrayView2<f64> = s_i_ij.slice(s![nocc_i..,nocc..]);
            let s_j_ij_occ:ArrayView2<f64> = s_j_ij.slice(s![..nocc_j,..nocc]);

            // Create matrix for the CT. Only the matrix element, which corresponds to the charge
            // transfer is set to the value 1.0. Everything else is set to null.
            let mut ct_coefficients:Array2<f64> = Array2::zeros([nocc_j,nvirt_i]);
            ct_coefficients[[nocc_j-1-ct_ind_j,ct_ind_i]] = 1.0;

            // transform the CT matrix using the reduced overlap matrices between the monomers
            // and the dimer
            let transformed_ct_coeff:Array2<f64> = s_j_ij_occ.t().dot(&ct_coefficients.dot(&s_i_ij_virt));

            // set cis coefficient for the CT transition
            let mut cis_coeff:Array3<f64> = Array3::zeros([1,nocc,nvirt]);
            cis_coeff.slice_mut(s![0,..,..]).assign(&transformed_ct_coeff);
            // save in properties
            self.properties.set_ci_coefficients(cis_coeff.into_shape([1,nocc*nvirt]).unwrap());
        }

        // calculate transition charges
        let tmp: (Array2<f64>, Array2<f64>, Array2<f64>) = trans_charges(
            self.n_atoms,
            atoms,
            self.properties.orbs().unwrap(),
            self.properties.s().unwrap(),
            &occ_indices,
            &virt_indices,
        );
        self.properties.set_q_ov(tmp.0);
        self.properties.set_q_oo(tmp.1);
        self.properties.set_q_vv(tmp.2);

        // prepare the grad gamma_lr ao matrix
        if self.gammafunction_lc.is_some(){
            // calculate the gamma gradient matrix in AO basis
            let (g1_lr,g1_lr_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                atoms,
                self.n_atoms,
                self.n_orbs,
            );
            self.properties.set_grad_gamma_lr_ao(g1_lr_ao);
        }
        // prepare gamma and grad gamma AO matrix
        let g0_ao:Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
            self.properties.gamma().unwrap(),
            atoms,
            self.n_orbs
        );
        let (g1,g1_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
            &self.gammafunction,
            atoms,
            self.n_atoms,
            self.n_orbs,
        );
        self.properties.set_grad_gamma(g1);
        self.properties.set_gamma_ao(g0_ao);
        self.properties.set_grad_gamma_ao(g1_ao);

        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&atoms, self.n_orbs, &self.slako);
        self.properties.set_grad_s(grad_s);
        self.properties.set_grad_h0(grad_h0);
    }

    pub fn tda_gradient_lc(&mut self, state: usize)->Array1<f64>{
        // get occ and virt indices from properties
        let occ_indices:&[usize] = self.properties.occ_indices().unwrap();
        let virt_indices:&[usize] = self.properties.virt_indices().unwrap();

        // set the occupied and virtual orbital energies
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();
        let orbe_occ: Array1<f64> = occ_indices.iter().map(|&occ| orbe[occ]).collect();
        let orbe_virt: Array1<f64> = virt_indices.iter().map(|&virt| orbe[virt]).collect();

        // transform the energies to a diagonal 2d matrix
        let ei: Array2<f64> = Array2::from_diag(&orbe_occ);
        let ea: Array2<f64> = Array2::from_diag(&orbe_virt);

        let n_occ: usize = orbe_occ.len();
        let n_virt: usize = orbe_virt.len();

        // excitation energy of the state
        let omega_state: ArrayView1<f64> = self.properties.ci_eigenvalues().unwrap();
        let n_states:usize = omega_state.len();
        let omega_state:f64 = omega_state[state];
        // take state specific values from the excitation vectors
        let x_state:ArrayView3<f64> = self.properties.ci_coefficients().unwrap()
            .into_shape([n_states,n_occ,n_virt]).unwrap();
        let x_state:ArrayView2<f64> = x_state.slice(s![state,..,..]);

        // calculate the vectors u, v and t
        // vectors U, V and T
        let u_ab: Array2<f64> = 2.0 *x_state.t().dot(&x_state);
        let u_ij: Array2<f64> = 2.0 *x_state.dot(&x_state.t());
        let v_ab: Array2<f64> = 2.0 *ei.dot(&x_state).t().dot(&x_state);
        let v_ij: Array2<f64> = 2.0 *x_state.dot(&ea).dot(&x_state.t());
        let t_ab: Array2<f64> = x_state.t().dot(&x_state);
        let t_ij: Array2<f64> = x_state.dot(&x_state.t());

        // get the transition charges
        let qtrans_ov: ArrayView3<f64> = self.properties.q_ov().unwrap().into_shape((self.n_atoms,n_occ,n_virt)).unwrap();
        let qtrans_oo: ArrayView3<f64> = self.properties.q_oo().unwrap().into_shape((self.n_atoms,n_occ,n_occ)).unwrap();
        let qtrans_vv: ArrayView3<f64> = self.properties.q_vv().unwrap().into_shape((self.n_atoms,n_virt,n_virt)).unwrap();
        let qtrans_vo: Array3<f64> = qtrans_ov
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // create struct hplus
        let hav: Hav = Hav::new(qtrans_ov, qtrans_vv, qtrans_oo, qtrans_vo.view());

        // set gamma matrix
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        // compute hplus of tab and tij
        let hplus_tab: Array2<f64> = hav.compute(g0, g0_lr, t_ab.view(), HplusType::Tab);
        let hplus_tij: Array2<f64> = hav.compute(g0, g0_lr, t_ij.view(), HplusType::Tij);

        // calculate q_ij
        let g_ij: Array2<f64> = hplus_tab - hplus_tij;
        let q_ij: Array2<f64> = omega_state * u_ij - v_ij + g_ij;

        // calculate q_ab
        let q_ab: Array2<f64> = omega_state * u_ab + v_ab;

        // calculate q_ia
        let mut q_ia: Array2<f64> = x_state.dot(
            &hav
                .compute(g0, g0_lr, x_state, HplusType::Qia_Xpy)
                .t(),
        );
        q_ia = q_ia + hav.compute(g0, g0_lr, t_ab.view(), HplusType::Qia_Tab);
        q_ia = q_ia - hav.compute(g0, g0_lr, t_ij.view(), HplusType::Qia_Tij);

        // calculate q_ai
        let mut q_ai: Array2<f64> =
            x_state
                .t()
                .dot(&hav.compute(g0, g0_lr, x_state, HplusType::Qai));

        // calculate right hand side of the z-vector equation
        let r_ia: Array2<f64> = &q_ai.t() - &q_ia;

        // input for zvector routine
        let omega_input: Array2<f64> = into_col(Array::ones(orbe_occ.len()))
            .dot(&into_row(orbe_virt.clone()))
            - into_col(orbe_occ.clone()).dot(&into_row(Array::ones(orbe_virt.len())));
        let r_ia_flat: Array1<f64> = r_ia.t().to_owned_f().into_shape((n_occ * n_virt)).unwrap();
        let r_matrix: Array2<f64> = r_ia_flat.into_shape((n_occ, n_virt)).unwrap();

        // calculate the z-vector
        let z_ia: Array2<f64> = tda_zvector_lc(
            omega_input.view(),
            r_matrix.view(),
            g0,
            g0_lr,
            qtrans_oo,
            qtrans_vv,
            qtrans_ov,
            1,
            Array1::zeros(self.n_atoms).view(),
        );

        // calculate w_ij
        let mut w_ij: Array2<f64> =
            q_ij + hav.compute(g0, g0_lr, z_ia.view(), HplusType::Wij);
        for i in 0..w_ij.dim().0 {
            w_ij[[i, i]] = w_ij[[i, i]] / 2.0;
        }
        // w_ia
        let w_ia: Array2<f64> = &q_ai.t() + &ei.dot(&z_ia); //+ h_a_nolr(g0, qtrans_ov, qtrans_ov, z_ia.view());

        // w_ai
        let w_ai: Array2<f64> = &q_ai + &ei.dot(&z_ia).t();

        // w_ab
        let mut w_ab: Array2<f64> = q_ab;
        for i in 0..w_ab.dim().0 {
            w_ab[[i, i]] = w_ab[[i, i]] / 2.0;
        }

        // build w matrix: combine w_ij, w_ia, w_ai and w_ab
        let length: usize = n_occ + n_virt;
        let mut w_matrix: Array2<f64> = Array::zeros((length, length));
        for i in 0..w_ij.dim().0 {
            w_matrix
                .slice_mut(s![i, ..w_ij.dim().1])
                .assign(&w_ij.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![i, w_ij.dim().1..])
                .assign(&w_ia.slice(s![i, ..]));
        }
        for i in 0..w_ai.dim().0 {
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, ..w_ai.dim().1])
                .assign(&w_ai.slice(s![i, ..]));
            w_matrix
                .slice_mut(s![w_ij.dim().0 + i, w_ai.dim().1..])
                .assign(&w_ab.slice(s![i, ..]));
        }

        // get arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let grad_h: ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let grad_s: ArrayView3<f64> = self.properties.grad_s().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_dmd0:Array3<f64> = f_lr(
            diff_p.view(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        let grad_h: Array3<f64> = &grad_h + &f_dmd0- 0.5 * &flr_dmd0;

        // set the occupied and virtuals orbital coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, n_occ));
        let mut orbs_virt: Array2<f64> = Array::zeros((self.n_orbs, n_virt));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }
        for (i, index) in virt_indices.iter().enumerate() {
            orbs_virt.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // transform t and z vectors to AO basis
        let t_oo: Array2<f64> = orbs_occ.dot(&t_ij.dot(&orbs_occ.t()));
        let t_vv: Array2<f64> = orbs_virt.dot(&t_ab.dot(&orbs_virt.t()));
        let z_ao: Array2<f64> = orbs_occ.dot(&z_ia.dot(&orbs_virt.t()));

        // transform w matrix and excited state vectors to AO basis
        let w_triangular: Array2<f64> = w_matrix.into_triangular(UPLO::Upper);
        let w_ao: Array2<f64> = orbs.dot(&w_triangular.dot(&orbs.t()));
        let x_ao: Array2<f64> = orbs_occ.dot(&x_state.dot(&orbs_virt.t()));

        // calculate contributions to the excited gradient
        let f: Array3<f64> = f_v(
            x_ao.view(),
            s,
            grad_s,
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_p = f_lr(
            x_ao.t(),
            s,
            grad_s,
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // assemble the excited gradient
        let mut gradExc: Array1<f64> = Array::zeros(3 * self.n_atoms);
        // gradH * (T + Z)
        gradExc = gradExc
            + grad_h
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(
                &(t_vv - t_oo + z_ao)
                    .into_shape(self.n_orbs * self.n_orbs)
                    .unwrap(),
            );
        // - gradS * W
        gradExc = gradExc
            - grad_s
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&w_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        // 2.0 * sum (X+Y) F (X+Y)
        gradExc = gradExc
            + 2.0
            * f.into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&x_ao.view().into_shape(self.n_orbs * self.n_orbs).unwrap());
        // - sum (X) F_lr (X)(X)
        gradExc = gradExc
            - flr_p
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap()
            .dot(&x_ao.into_shape(self.n_orbs * self.n_orbs).unwrap());
        return gradExc;
    }
}