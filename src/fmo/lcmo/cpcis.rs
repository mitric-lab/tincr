use crate::excited_states::trans_charges;
use crate::fmo::{ExcitedStateMonomerGradient, Monomer};
use crate::gradients::helpers::{f_lr, f_v};
use crate::initialization::{Atom, MO};
use crate::scc::h0_and_s::h0_and_s_gradients;
use ndarray::prelude::*;
use std::ops::SubAssign;

impl Monomer {
    pub fn cpcis_a_matrix(&self, le_state: usize) -> Array2<f64> {
        // Calculate the coulomb integral
        // Reference to the o-v transition charges.
        let qov: ArrayView2<f64> = self.properties.q_ov().unwrap();
        // Reference to the unscreened Gamma matrix.
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        // Reference to the energy differences of the orbital energies.
        let omega: ArrayView1<f64> = self.properties.omega().unwrap();
        // The sum of one-electron part and Coulomb part is computed and retzurned.
        let coulomb: Array2<f64> = 2.0 * qov.t().dot(&gamma.dot(&qov));

        // Calculate the exchange integral
        // Number of occupied orbitals.
        let n_occ: usize = self.properties.occ_indices().unwrap().len();
        // Number of virtual orbitals.
        let n_virt: usize = self.properties.virt_indices().unwrap().len();
        // Reference to the o-o transition charges.
        let qoo: ArrayView2<f64> = self.properties.q_oo().unwrap();
        // Reference to the v-v transition charges.
        let qvv: ArrayView2<f64> = self.properties.q_vv().unwrap();
        // Reference to the screened Gamma matrix.
        let gamma_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
        // The exchange part to the CIS Hamiltonian is computed.
        let exchange = qoo
            .t()
            .dot(&gamma_lr.dot(&qvv))
            .into_shape((n_occ, n_occ, n_virt, n_virt))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape([n_occ * n_virt, n_occ * n_virt])
            .unwrap()
            .to_owned();

        // get orbital energy differences - energy of the tda state
        let tda_energy = self.properties.ci_eigenvalue(le_state).unwrap();
        let tda_coeff: ArrayView2<f64> = self
            .properties
            .ci_coefficient(le_state)
            .unwrap()
            .into_shape([n_occ, n_virt])
            .unwrap();
        let energies: Array2<f64> = Array2::from_diag(&omega.map(|val| val - tda_energy));

        let mut coefficient_matrix: Array4<f64> = Array4::zeros((n_occ, n_virt, n_occ, n_virt));
        for i in 0..n_occ {
            for a in 0..n_virt {
                for j in 0..n_occ {
                    for b in 0..n_virt {
                        coefficient_matrix[[i, a, j, b]] +=
                            2.0 * tda_coeff[[i, a]] * tda_coeff[[j, b]];
                    }
                }
            }
        }

        energies + coulomb - exchange
            + coefficient_matrix
                .into_shape([n_occ * n_virt, n_occ * n_virt])
                .unwrap()
    }

    pub fn cpcis_b_matrix(&mut self, le_state: usize, atoms: &[Atom], u_mat: ArrayView3<f64>)->Array3<f64> {
        self.prepare_excited_gradient(atoms);
        let tda_grad: Array1<f64> = self.tda_gradient_lc(le_state);

        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&atoms, self.n_orbs, &self.slako);

        // get necessary arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g0: ArrayView2<f64> = self.properties.gamma().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let orbe: ArrayView1<f64> = self.properties.orbe().unwrap();

        // calculate grad_Hxc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s.view(),
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        // calulcate gradH
        let mut grad_h: Array3<f64> = &grad_h0 + &f_dmd0;

        // add the lc-gradient of the hamiltonian
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();

        let flr_dmd0: Array3<f64> = f_lr(
            diff_p.view(),
            s,
            grad_s.view(),
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );
        grad_h = grad_h - 0.5 * &flr_dmd0;

        // get MO coefficient for the occupied or virtual orbital
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();
        let nocc: usize = occ_indices.len();
        let nvirt: usize = virt_indices.len();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, nocc));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // calculate transition charges
        let (qov, qoo, qvv) =
            trans_charges(self.n_atoms, atoms, orbs, s, occ_indices, virt_indices);
        // virtual-occupied transition charges
        let qvo: Array2<f64> = qov
            .clone()
            .into_shape([self.n_atoms, nocc, nvirt])
            .unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([self.n_atoms, nvirt * nocc])
            .unwrap();

        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();
        // calculate B matrix B_ij with i = nvirt, j = nocc
        // for the CPHF iterations
        let mut b_mat: Array3<f64> = Array3::zeros([3 * self.n_atoms, self.n_orbs, self.n_orbs]);

        // create orbital energy matrix
        let mut orbe_matrix: Array2<f64> = Array2::zeros((self.n_orbs, self.n_orbs));
        for mu in 0..self.n_orbs {
            for nu in 0..self.n_orbs {
                orbe_matrix[[mu, nu]] = orbe[nu];
            }
        }

        // calculate A_matrix ij,kl i = nvirt, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vo: Array2<f64> = Array2::zeros([nvirt * nocc, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_vo = a_mat_vo
            - qvv
                .t()
                .dot(&g0_lr.dot(&qoo))
                .into_shape([nvirt, nvirt, nocc, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nocc, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_vo = a_mat_vo
            - qvo
                .t()
                .dot(&g0_lr.dot(&qov))
                .into_shape([nvirt, nocc, nocc, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nocc, nvirt * nocc])
                .unwrap();

        // calculate A_matrix ij,kl i = nocc, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_oo: Array2<f64> = Array2::zeros([nocc * nocc, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_oo = a_mat_oo
            - qov
                .t()
                .dot(&g0_lr.dot(&qoo))
                .into_shape([nocc, nvirt, nocc, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nocc, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_oo = a_mat_oo
            - qoo
                .t()
                .dot(&g0_lr.dot(&qov))
                .into_shape([nocc, nocc, nocc, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nocc, nvirt * nocc])
                .unwrap();

        // calculate A_matrix ij,kl i = nocc, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_ov: Array2<f64> = Array2::zeros([nocc * nvirt, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_ov = a_mat_ov
            - qov
                .t()
                .dot(&g0_lr.dot(&qvo))
                .into_shape([nocc, nvirt, nvirt, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nvirt, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_ov = a_mat_ov
            - qoo
                .t()
                .dot(&g0_lr.dot(&qvv))
                .into_shape([nocc, nocc, nvirt, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nvirt, nvirt * nocc])
                .unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vv: Array2<f64> = Array2::zeros([nvirt * nvirt, nvirt * nocc]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_vv = a_mat_vv
            - qvv
                .t()
                .dot(&g0_lr.dot(&qvo))
                .into_shape([nvirt, nvirt, nvirt, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nvirt, nvirt * nocc])
                .unwrap();
        // integral (il|jk)
        a_mat_vv = a_mat_vv
            - qvo
                .t()
                .dot(&g0_lr.dot(&qvv))
                .into_shape([nvirt, nocc, nvirt, nvirt])
                .unwrap()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nvirt, nvirt * nocc])
                .unwrap();

        let mut a_matrix: Array3<f64> = Array3::zeros([self.n_orbs, self.n_orbs, nvirt * nocc]);
        a_matrix
            .slice_mut(s![..nocc, ..nocc, ..])
            .assign(&a_mat_oo.into_shape([nocc, nocc, nvirt * nocc]).unwrap());
        a_matrix
            .slice_mut(s![..nocc, nocc.., ..])
            .assign(&a_mat_ov.into_shape([nocc, nvirt, nvirt * nocc]).unwrap());
        a_matrix
            .slice_mut(s![nocc.., ..nocc, ..])
            .assign(&a_mat_vo.into_shape([nvirt, nocc, nvirt * nocc]).unwrap());
        a_matrix
            .slice_mut(s![nocc.., nocc.., ..])
            .assign(&a_mat_vv.into_shape([nvirt, nvirt, nvirt * nocc]).unwrap());
        let a_mat: Array2<f64> = a_matrix
            .into_shape([self.n_orbs * self.n_orbs, nvirt * nocc])
            .unwrap();

        // Calculate integrals partwise before iteration over gradient
        // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
        let integral_vo_2d: Array2<f64> = (2.0 * qvo.t().dot(&g0.dot(&qoo))
            - qvo
                .t()
                .dot(&g0_lr.dot(&qoo))
                .into_shape([nvirt, nocc, nocc, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nocc, nocc * nocc])
                .unwrap());
        // integral (ij|kl) - (ik|jl), i = nocc, j = nocc
        let integral_oo_2d: Array2<f64> = (2.0 * qoo.t().dot(&g0.dot(&qoo))
            - qoo
                .t()
                .dot(&g0_lr.dot(&qoo))
                .into_shape([nocc, nocc, nocc, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nocc, nocc * nocc])
                .unwrap());
        // integral (ij|kl) - (ik|jl), i = nvirt, j = nvirt
        let integral_vv_2d: Array2<f64> = (2.0 * qvv.t().dot(&g0.dot(&qoo))
            - qvo
                .t()
                .dot(&g0_lr.dot(&qvo))
                .into_shape([nvirt, nocc, nvirt, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nvirt * nvirt, nocc * nocc])
                .unwrap());
        // integral (ij|kl) - (ik|jl), i = nocc, j = nvirt
        let integral_ov_2d: Array2<f64> = (2.0 * qov.t().dot(&g0.dot(&qoo))
            - qoo
                .t()
                .dot(&g0_lr.dot(&qvo))
                .into_shape([nocc, nocc, nvirt, nocc])
                .unwrap()
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned()
                .into_shape([nocc * nvirt, nocc * nocc])
                .unwrap());

        // Calculate the B matrix
        for nc in 0..3 * self.n_atoms {
            let ds_mo: Array1<f64> = orbs_occ
                .t()
                .dot(&grad_s.slice(s![nc, .., ..]).dot(&orbs_occ))
                .into_shape([nocc * nocc])
                .unwrap();

            // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
            let integral_vo: Array2<f64> = integral_vo_2d
                .dot(&ds_mo)
                .into_shape([nvirt, nocc])
                .unwrap();
            // integral (ij|kl) - (ik|jl), i = nocc, j = nocc
            let integral_oo: Array2<f64> =
                integral_oo_2d.dot(&ds_mo).into_shape([nocc, nocc]).unwrap();
            // integral (ij|kl) - (ik|jl), i = nvirt, j = nvirt
            let integral_vv: Array2<f64> = integral_vv_2d
                .dot(&ds_mo)
                .into_shape([nvirt, nvirt])
                .unwrap();
            // integral (ij|kl) - (ik|jl), i = nocc, j = nvirt
            let integral_ov: Array2<f64> = integral_ov_2d
                .dot(&ds_mo)
                .into_shape([nocc, nvirt])
                .unwrap();

            let gradh_mo: Array2<f64> = orbs.t().dot(&grad_h.slice(s![nc, .., ..]).dot(&orbs));
            let grads_mo: Array2<f64> = orbs.t().dot(&grad_s.slice(s![nc, .., ..]).dot(&orbs));

            let a_dot_u: Array2<f64> = a_mat
                .dot(
                    &u_mat
                        .slice(s![nc, nocc.., ..nocc])
                        .to_owned()
                        .into_shape([nvirt * nocc])
                        .unwrap(),
                )
                .into_shape([self.n_orbs, self.n_orbs])
                .unwrap();

            b_mat
                .slice_mut(s![nc, .., ..])
                .assign(&(&gradh_mo - &grads_mo * &orbe_matrix + &a_dot_u));
            b_mat
                .slice_mut(s![nc, ..nocc, ..nocc])
                .sub_assign(&integral_oo);
            b_mat
                .slice_mut(s![nc, ..nocc, nocc..])
                .sub_assign(&integral_ov);
            b_mat
                .slice_mut(s![nc, nocc.., ..nocc])
                .sub_assign(&integral_vo);
            b_mat
                .slice_mut(s![nc, nocc.., nocc..])
                .sub_assign(&integral_vv);
        }
        // calculate the derivative of the two electron integrals
        // First: calculate the derivative of the MO coefficients
        let mut dc_mo: Array3<f64> = Array3::zeros([3 * self.n_atoms, self.n_orbs, self.n_orbs]);
        // iterate over gradient dimensions of both monomers
        for nat in 0..3 * self.n_atoms {
            for orb in 0..self.n_orbs {
                dc_mo
                    .slice_mut(s![nat, orb, ..])
                    .assign(&u_mat.slice(s![nat, .., orb]).dot(&orbs.t()));
            }
        }
        let two_electron_integral_derivative: Array5<f64> = two_electron_integral_derivative(
            s,
            g0,
            g0,
            grad_s.view(),
            g1_ao,
            g1lr_ao,
            dc_mo.view(),
            orbs,
            nocc,
            nvirt,
            self.n_orbs,
            self.n_atoms,
        );

        // create CPCIS mixed derivative matrix
        let mut cpcis_mat: Array3<f64> = Array3::zeros([3 * self.n_atoms, nvirt, nocc]);

        let tda_coeff: ArrayView2<f64> = self
            .properties
            .ci_coefficient(le_state)
            .unwrap()
            .into_shape([nocc, nvirt])
            .unwrap();

        for nc in 0..3 * self.n_atoms {
            for i in 0..nocc {
                for a in 0..nvirt {
                    for j in 0..nocc {
                        for b in 0..nvirt {
                            let fock_val_ba = if i == j { b_mat[[nc, b, a]] } else { 0.0 };
                            let fock_val_ji = if a == b { b_mat[[nc, j, i]] } else { 0.0 };
                            let le_grad = if i == j && a == b { tda_grad[nc] } else { 0.0 };
                            cpcis_mat[[nc, b, j]] +=
                                (two_electron_integral_derivative[[nc, a, i, j, b]] + fock_val_ba
                                    - fock_val_ji
                                    - le_grad)
                                    * tda_coeff[[i, a]];
                        }
                    }
                }
            }
        }

        cpcis_mat
    }
}

fn two_electron_integral_derivative(
    s: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    ds: ArrayView3<f64>,
    dg: ArrayView3<f64>,
    dg_lr: ArrayView3<f64>,
    dc_mo: ArrayView3<f64>,
    orbs: ArrayView2<f64>,
    nocc: usize,
    nvirt: usize,
    n_orbs: usize,
    n_atoms: usize,
) -> Array5<f64> {
    // calculate the two electron integrals
    let two_electron_integrals: Array4<f64> = coulomb_exchange_integral(s, g0, g0_lr, n_orbs);
    // calculate the derivative of the two electron integrals
    let two_electron_derivative: Array5<f64> =
        f_monomer_coulomb_exchange_loop(s, ds, g0, g0_lr, dg, dg_lr, n_atoms, n_orbs);
    // transform the derivative from the AO basis to the MO basis using MO coefficients
    // from [nc,norbs,norbs,norbs,norbs] to [nc,nvirt,nocc,nocc,nvirt]
    let two_electron_mo_basis: Array5<f64> = two_electron_derivative
        .view()
        .into_shape([3 * n_atoms * n_orbs * n_orbs * n_orbs, n_orbs])
        .unwrap()
        .dot(&orbs.slice(s![.., nocc..]))
        .into_shape([3 * n_atoms, n_orbs, n_orbs, n_orbs, nvirt])
        .unwrap()
        .permuted_axes([0, 1, 2, 4, 3])
        .to_owned()
        .into_shape([3 * n_atoms * n_orbs * n_orbs * nvirt, n_orbs])
        .unwrap()
        .dot(&orbs.slice(s![.., ..nocc]))
        .into_shape([3 * n_atoms, n_orbs, n_orbs, nvirt, nocc])
        .unwrap()
        .permuted_axes([0, 1, 4, 3, 2])
        .to_owned()
        .into_shape([3 * n_atoms * n_orbs * nocc * nvirt, n_orbs])
        .unwrap()
        .dot(&orbs.slice(s![.., ..nocc]))
        .into_shape([3 * n_atoms, n_orbs, nocc, nvirt, nocc])
        .unwrap()
        .permuted_axes([0, 4, 2, 3, 1])
        .to_owned()
        .into_shape([3 * n_atoms * nocc * nocc * nvirt, n_orbs])
        .unwrap()
        .dot(&orbs.slice(s![.., nocc..]))
        .into_shape([3 * n_atoms, nocc, nocc, nvirt, nvirt])
        .unwrap()
        .permuted_axes([0, 4, 1, 2, 3])
        .to_owned();

    // build the product between the derivatives of the MO coefficients and the two electron integrals
    let cphf_integral: Array5<f64> = cphf_two_electron_integral(
        nocc,
        nvirt,
        n_orbs,
        n_atoms,
        dc_mo,
        orbs,
        two_electron_integrals.view(),
    );

    let final_integral: Array5<f64> = two_electron_mo_basis + cphf_integral;
    final_integral
}

fn cphf_two_electron_integral(
    nocc: usize,
    nvirt: usize,
    n_orbs: usize,
    n_atoms: usize,
    dc_mo: ArrayView3<f64>,
    orbs: ArrayView2<f64>,
    two_electron_integrals: ArrayView4<f64>,
) -> Array5<f64> {
    let dc_mo_occs: ArrayView3<f64> = dc_mo.slice(s![.., ..nocc, ..]);
    let dc_mo_virts: ArrayView3<f64> = dc_mo.slice(s![.., nocc.., ..]);
    let orbs_occ: ArrayView2<f64> = orbs.slice(s![.., ..nocc]);
    let orbs_virt: ArrayView2<f64> = orbs.slice(s![.., nocc..]);

    let mut cphf_integral: Array5<f64> = Array5::zeros((3 * n_atoms, nvirt, nocc, nocc, nvirt));
    let mut cphf_integral_dot:Array5<f64> = Array5::zeros((3 * n_atoms, nvirt, nocc, nocc, nvirt));

    for nc in 0..3 * n_atoms {
        let term_1:Array2<f64> = &dc_mo_virts.slice(s![nc,..,..]) + 3.0* &orbs_virt.t();
        let term_2:Array2<f64> = &dc_mo_occs.slice(s![nc,..,..]) + 3.0* &orbs_occ.t();

        let tmp:Array4<f64> = two_electron_integrals
            .into_shape([n_orbs * n_orbs * n_orbs, n_orbs])
            .unwrap()
            .dot(&term_1.t())
            .into_shape([n_orbs, n_orbs, n_orbs, nvirt])
            .unwrap()
            .permuted_axes([0,1,3,2])
            .to_owned()
            .into_shape([n_orbs * n_orbs * nvirt, n_orbs])
            .unwrap()
            .dot(&term_2.t())
            .into_shape([n_orbs, n_orbs, nvirt, nocc])
            .unwrap()
            .permuted_axes([0,3,2,1])
            .to_owned()
            .into_shape([n_orbs * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&term_2.t())
            .into_shape([n_orbs, nocc, nvirt, nocc])
            .unwrap()
            .permuted_axes([3,1,2,0])
            .to_owned()
            .into_shape([nocc * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&term_1.t())
            .into_shape([nocc, nocc, nvirt, nvirt])
            .unwrap()
            .permuted_axes([3,0,1,2])
            .to_owned();

        cphf_integral_dot.slice_mut(s![nc,..,..,..,..]).assign(&tmp);
    }

    for nc in 0..3 * n_atoms {
        for a in 0..nvirt {
            for i in 0..nocc {
                for j in 0..nocc {
                    for b in 0..nvirt {
                        for mu in 0..n_orbs {
                            for nu in 0..n_orbs {
                                for la in 0..n_orbs {
                                    for sig in 0..n_orbs {
                                        cphf_integral[[nc, a, i, j, b]] += two_electron_integrals
                                            [[mu, nu, la, sig]]
                                            // first
                                            * (dc_mo_virts[[nc, a, mu]]
                                            * orbs_occ[[nu, i]]
                                            * orbs_occ[[la, j]]
                                            * orbs_virt[[sig, b]]
                                            // second term
                                            + orbs_virt[[mu, a]]
                                            * dc_mo_occs[[nc, i, nu]]
                                            * orbs_occ[[la, j]]
                                            * orbs_virt[[sig, b]]
                                            // third term
                                            + orbs_virt[[mu, a]]
                                            * orbs_occ[[nu, i]]
                                            * dc_mo_occs[[nc, j, la]]
                                            * orbs_virt[[sig, b]]
                                            // fourth term
                                            + orbs_virt[[mu, a]]
                                            * orbs_occ[[nu, i]]
                                            * orbs_occ[[la, j]]
                                            * dc_mo_virts[[nc, b, sig]]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(cphf_integral.abs_diff_eq(&cphf_integral_dot,1e-10),"Loop version NOT equal to dot version");

    return cphf_integral;
}

fn coulomb_exchange_integral(
    s: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    n_orbs: usize,
) -> Array4<f64> {
    let mut coulomb_integral: Array4<f64> = Array4::zeros((n_orbs, n_orbs, n_orbs, n_orbs));
    let mut exchange_integral: Array4<f64> = Array4::zeros((n_orbs, n_orbs, n_orbs, n_orbs));
    for mu in 0..n_orbs {
        for nu in 0..n_orbs {
            for la in 0..n_orbs {
                for sig in 0..n_orbs {
                    coulomb_integral[[mu, nu, la, sig]] += 0.25
                        * s[[mu, nu]]
                        * s[[la, sig]]
                        * (g0[[mu, la]] + g0[[mu, sig]] + g0[[nu, la]] + g0[[nu, sig]]);

                    exchange_integral[[mu, nu, la, sig]] += 0.25
                        * s[[mu, la]]
                        * s[[nu, sig]]
                        * (g0_lr[[mu, nu]] + g0_lr[[mu, sig]] + g0_lr[[la, nu]] + g0_lr[[la, sig]]);
                }
            }
        }
    }
    let result: Array4<f64> = 2.0 * coulomb_integral - exchange_integral;
    result
}

fn f_monomer_coulomb_exchange_loop(
    s: ArrayView2<f64>,
    ds: ArrayView3<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    dg: ArrayView3<f64>,
    dg_lr: ArrayView3<f64>,
    n_atoms: usize,
    n_orbs: usize,
) -> Array5<f64> {
    let mut coulomb_integral: Array5<f64> =
        Array5::zeros([3 * n_atoms, n_orbs, n_orbs, n_orbs, n_orbs]);
    let mut exchange_integral: Array5<f64> =
        Array5::zeros([3 * n_atoms, n_orbs, n_orbs, n_orbs, n_orbs]);

    for nc in 0..3 * n_atoms {
        for mu in 0..n_orbs {
            for nu in 0..n_orbs {
                for la in 0..n_orbs {
                    for sig in 0..n_orbs {
                        exchange_integral[[nc, mu, nu, la, sig]] += 0.25
                            * ((ds[[nc, mu, la]] * s[[nu, sig]] + s[[mu, la]] * ds[[nc, nu, sig]])
                                * (g0_lr[[mu, nu]]
                                    + g0_lr[[mu, sig]]
                                    + g0_lr[[la, nu]]
                                    + g0_lr[[la, sig]])
                                + s[[mu, la]]
                                    * s[[nu, sig]]
                                    * (dg_lr[[nc, mu, nu]]
                                        + dg_lr[[nc, mu, sig]]
                                        + dg_lr[[nc, la, nu]]
                                        + dg_lr[[nc, la, sig]]));

                        coulomb_integral[[nc, mu, nu, la, sig]] += 0.25
                            * ((ds[[nc, mu, nu]] * s[[la, sig]] + s[[mu, nu]] * ds[[nc, la, sig]])
                                * (g0[[mu, la]] + g0[[mu, sig]] + g0[[nu, la]] + g0[[nu, sig]])
                                + s[[mu, nu]]
                                    * s[[la, sig]]
                                    * (dg[[nc, mu, nu]]
                                        + dg[[nc, mu, sig]]
                                        + dg[[nc, nu, la]]
                                        + dg[[nc, nu, sig]]));
                    }
                }
            }
        }
    }
    return 2.0 * coulomb_integral - exchange_integral;
}
