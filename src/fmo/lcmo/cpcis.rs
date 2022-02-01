use crate::excited_states::trans_charges;
use crate::fmo::{ExcitedStateMonomerGradient, Monomer, SuperSystem};
use crate::gradients::helpers::{f_lr, f_v};
use crate::initialization::{Atom, MO};
use crate::scc::h0_and_s::h0_and_s_gradients;
use ndarray::prelude::*;
use std::ops::SubAssign;
use ndarray_linalg::Inverse;

impl SuperSystem {
    pub fn cpcis_routine(
        &mut self,
        ind: usize,
        le_state: usize,
        umat: ArrayView3<f64>,
        nocc:usize,
        nvirt:usize,
    )->Array3<f64> {
        // get mutable monomer I
        let m_i: &mut Monomer = &mut self.monomers[ind];
        // get atoms
        let atoms:&[Atom] = &self.atoms[m_i.slice.atom_as_range()];
        // create cpcis B matrix
        let fock_derivative: Array3<f64> =
            m_i.fock_derivative(atoms, umat);

        // output is [3*natoms,nvirt,nocc]
        let cpcis_coefficients:Array3<f64> = m_i.calculate_cpcis_terms(
            nocc,
            nvirt,
            umat,
            le_state,
            fock_derivative.view()
        );
        // switch axes 1 and 2
        let cpcis_coefficients:Array3<f64> = cpcis_coefficients
            .permuted_axes([0,2,1]).as_standard_layout().to_owned();

        //Other version
        // create cpcis A matrix
        let amat: Array2<f64> = m_i.cpcis_a_matrix(le_state);
        // invert the A matrix
        let a_inv:Array2<f64> = amat.inv().unwrap();
        // create b matrix
        let bmat:Array3<f64> = m_i.cpcis_b_matrix(le_state,atoms,umat);
        // solve the equation
        // sum_i,a A^-1_i,a,j,b B_i,a
        let mut cpcis_coefficients_2:Array3<f64> = Array3::zeros([3*m_i.n_atoms,nocc,nvirt]);
        for nc in 0..3*m_i.n_atoms{
            let tmp:Array1<f64> = bmat.slice(s![nc,..,..]).t().into_shape([nocc*nvirt]).unwrap().dot(&a_inv.t());
            // let tmp:Array1<f64> = a_inv.dot(&bmat.slice(s![nc,..,..]).t().into_shape([nocc*nvirt]).unwrap());
            cpcis_coefficients_2.slice_mut(s![nc,..,..]).assign(&tmp.into_shape([nocc,nvirt]).unwrap());
        }
        // let cpcis_coefficients = solve_cpcis_pople(amat_i.view(),bmat_i.view(),nocc,nvirt,m_i.n_atoms);

        return cpcis_coefficients_2;
    }
}

impl Monomer {
    pub fn calculate_cpcis_terms(
        &self,
        nocc:usize,
        nvirt:usize,
        umat: ArrayView3<f64>,
        le_state: usize,
        fock_derivative:ArrayView3<f64>,
    )->Array3<f64>{
        // MO coefficients
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        // calculate the derivative of the MO coefficients
        let mut dc_mo: Array3<f64> = Array3::zeros([3 * self.n_atoms, self.n_orbs, self.n_orbs]);
        // iterate over gradient dimensions of both monomers
        for nat in 0..3 * self.n_atoms {
            for orb in 0..self.n_orbs {
                dc_mo
                    .slice_mut(s![nat, .., orb])
                    .assign(&umat.slice(s![nat, .., orb]).dot(&orbs.t()));
            }
        }
        // occupied and virtual MO coefficients
        let orbs_occ:ArrayView2<f64> = orbs.slice(s![..,..nocc]);
        let orbs_virt:ArrayView2<f64> = orbs.slice(s![..,nocc..]);
        // occupied and virtual derivatives of MO coefficients
        let dc_mo_occs:ArrayView3<f64> = dc_mo.slice(s![..,..,..nocc]);
        let dc_mo_virts:ArrayView3<f64> = dc_mo.slice(s![..,..,nocc..]);

        // calculate the two electron integrals
        let integrals: Array4<f64> = coulomb_exchange_integral(
            self.properties.s().unwrap(),
            self.properties.gamma_ao().unwrap(),
            self.properties.gamma_lr_ao().unwrap(),
            self.n_orbs
        );
        let integrals_2d:Array2<f64> = integrals
            .into_shape([self.n_orbs*self.n_orbs,self.n_orbs*self.n_orbs]).unwrap();

        // calculate the derivative of the two electron integrals
        let integral_derivatives:Array5<f64> = f_monomer_coulomb_exchange_loop(
            self.properties.s().unwrap(),
            self.properties.grad_s().unwrap(),
            self.properties.gamma_ao().unwrap(),
            self.properties.gamma_lr_ao().unwrap(),
            self.properties.grad_gamma_ao().unwrap(),
            self.properties.grad_gamma_lr_ao().unwrap(),
            self.n_atoms,
            self.n_orbs
        );
        // get CIS coefficients and the CIS energy for the LE state
        let cis_energy:f64 = self.properties.ci_eigenvalue(le_state).unwrap();
        let cis_coeff:ArrayView1<f64> = self.properties.ci_coefficient(le_state).unwrap();
        let cis_coeff:ArrayView2<f64> = cis_coeff.into_shape([nocc,nvirt]).unwrap();
        // CIS coefficients in AO basis
        let cis_coeff_ao:Array2<f64> = orbs_virt.dot(&cis_coeff.t().dot(&orbs_occ.t()));

        // Fock matrix in AO basis
        let fock_mat:ArrayView2<f64> = self.properties.h_coul_x().unwrap();

        // calculate necessary terms for l_b
        let integral_dot_cis_ao:Array2<f64> = integrals_2d.dot(&cis_coeff_ao.view()
            .into_shape([self.n_orbs*self.n_orbs]).unwrap()).into_shape([self.n_orbs,self.n_orbs]).unwrap();

        // assemble l_b
        // Equation (22) from https://doi.org/10.1080/00268979909483096
        let mut l_b:Array3<f64> = Array3::zeros([3*self.n_atoms,nvirt,nocc]);
        for nc in 0..3*self.n_atoms{
            let dc_mo_o:ArrayView2<f64> = dc_mo_occs.slice(s![nc,..,..]);
            let dc_mo_v:ArrayView2<f64> = dc_mo_virts.slice(s![nc,..,..]);
            let integral_derivs_2d:ArrayView2<f64> = integral_derivatives.slice(s![nc,..,..,..,..])
                .into_shape([self.n_orbs*self.n_orbs,self.n_orbs*self.n_orbs]).unwrap();

            // b_w * w
            let term_1:Array2<f64> = cis_energy * &cis_coeff.t();

            // (C^x_v F C_v + C_v F^X C_V + C_v F C^x_v) b_w
            let term_2:Array2<f64> = (dc_mo_v.t().dot(&fock_mat.dot(&orbs_virt))
                + orbs_virt.t().dot(&fock_derivative.slice(s![nc,..,..]).dot(&orbs_virt))
                +  orbs_virt.t().dot(&fock_mat.dot(&dc_mo_v))).dot(&cis_coeff.t());

            // b_w(C^x_o F C_o + C_o F^X C_o + C_o F C^x_o)
            let term_3:Array2<f64> = cis_coeff.t().dot(&(dc_mo_o.t().dot(&fock_mat.dot(&orbs_occ))
                + orbs_occ.t().dot(&fock_derivative.slice(s![nc,..,..]).dot(&orbs_occ))
                +  orbs_occ.t().dot(&fock_mat.dot(&dc_mo_o))));

            // C_v [(mu nu||la sig) C_v b_w C^x_o +(mu nu||la sig) C^x_v b_w C_o + (mu nu||la sig)^x C_v b_w C_o] C_o
            let term_4:Array2<f64> = orbs_virt.t().dot(&(
                (integrals_2d.dot(&(orbs_virt.dot(&cis_coeff.t()).dot(&dc_mo_o.t())).into_shape([self.n_orbs*self.n_orbs]).unwrap())
                + integrals_2d.dot(&(dc_mo_v.dot(&cis_coeff.t()).dot(&orbs_occ.t())).into_shape([self.n_orbs*self.n_orbs]).unwrap())
                + integral_derivs_2d.dot(&cis_coeff_ao.view().into_shape([self.n_orbs*self.n_orbs]).unwrap()))
                    .into_shape([self.n_orbs,self.n_orbs]).unwrap()
                )).dot(&orbs_occ);

            // C^x_v [(mu nu||la sig) C_v b_w C_o ] C_o + C_v [(mu nu||la sig) C_v b_w C_o ] C^x_o
            let term_5:Array2<f64> = dc_mo_v.t().dot(&integral_dot_cis_ao).dot(&orbs_occ)
                + orbs_virt.t().dot(&integral_dot_cis_ao).dot(&dc_mo_o);

            l_b.slice_mut(s![nc,..,..]).assign(&(term_1 - term_2 - term_3 + term_4 + term_5));
        }

        let fock_terms:Array2<f64> = orbs_virt.t().dot(&fock_mat.dot(&orbs_virt)).dot(&cis_coeff.t()) -
            cis_coeff.t().dot(&orbs_occ.t().dot(&fock_mat.dot(&orbs_occ)));

        let cis_der:Array3<f64> = solve_cpcis_iterative(
            fock_terms.view(),
            l_b.view(),
            orbs_occ,
            orbs_virt,
            integrals_2d.view(),
            cis_energy,
            self.n_atoms,
            nocc,
            nvirt
        );
        return cis_der;
    }

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

    pub fn fock_derivative(
        &mut self,
        atoms: &[Atom],
        u_mat: ArrayView3<f64>,
    ) -> Array3<f64> {
        // self.prepare_excited_gradient(atoms);
        // let tda_grad: Array1<f64> = self.tda_gradient_lc(le_state);

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
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

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


        // calculate B matrix B_ij with i = nvirt, j = nocc
        // for the CPHF iterations
        let mut b_mat: Array3<f64> = Array3::zeros([3 * self.n_atoms, self.n_orbs, self.n_orbs]);
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

            // b_mat
            //     .slice_mut(s![nc, .., ..])
            //     .assign(&(&gradh_mo - &grads_mo * &orbe_matrix + &a_dot_u));
            // b_mat
            //     .slice_mut(s![nc, ..nocc, ..nocc])
            //     .sub_assign(&integral_oo);
            // b_mat
            //     .slice_mut(s![nc, ..nocc, nocc..])
            //     .sub_assign(&integral_ov);
            // b_mat
            //     .slice_mut(s![nc, nocc.., ..nocc])
            //     .sub_assign(&integral_vo);
            // b_mat
            //     .slice_mut(s![nc, nocc.., nocc..])
            //     .sub_assign(&integral_vv);
            let u_2d:ArrayView2<f64> = u_mat.slice(s![nc,..,..]);
            // loop version
            for i in 0..nvirt{
                for j in 0..nocc{
                    b_mat[[nc,nocc+i,j]] = gradh_mo[[nocc+i,j]] +a_dot_u[[nocc+i,j]]
                        + u_2d[[j,nocc+i]] * orbe[j] + u_2d[[nocc+i,j]] *orbe[nocc+i] - integral_vo[[i,j]];
                }
                for j in 0..nvirt{
                    b_mat[[nc,nocc+i,nocc+j]] = gradh_mo[[nocc+i,nocc+j]] +a_dot_u[[nocc+i,nocc+j]]
                        + u_2d[[nocc+j,nocc+i]] * orbe[nocc+j] + u_2d[[nocc+i,nocc+j]] *orbe[nocc+i] - integral_vv[[i,j]];
                }
            }
            for i in 0..nocc{
                for j in 0..nocc{
                    b_mat[[nc,i,j]] = gradh_mo[[i,j]] +a_dot_u[[i,j]] +
                        u_2d[[j,i]] * orbe[j] + u_2d[[i,j]] *orbe[i] - integral_oo[[i,j]];
                }
                for j in 0..nvirt{
                    b_mat[[nc,i,nocc+j]] = gradh_mo[[i,nocc+j]] +a_dot_u[[i,nocc+j]] +
                        u_2d[[nocc+j,i]] * orbe[nocc+j] + u_2d[[i,nocc+j]] *orbe[i] - integral_ov[[i,j]];
                }
            }
        }
        self.properties.set_grad_s(grad_s);

        b_mat
    }

    pub fn cpcis_b_matrix(
        &mut self,
        le_state: usize,
        atoms: &[Atom],
        u_mat: ArrayView3<f64>,
    ) -> Array3<f64> {
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
        let g0_lr: ArrayView2<f64> = self.properties.gamma_lr().unwrap();

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


        // calculate B matrix B_ij with i = nvirt, j = nocc
        // for the CPHF iterations
        let mut b_mat: Array3<f64> = Array3::zeros([3 * self.n_atoms, self.n_orbs, self.n_orbs]);
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

            // b_mat
            //     .slice_mut(s![nc, .., ..])
            //     .assign(&(&gradh_mo - &grads_mo * &orbe_matrix + &a_dot_u));
            // b_mat
            //     .slice_mut(s![nc, ..nocc, ..nocc])
            //     .sub_assign(&integral_oo);
            // b_mat
            //     .slice_mut(s![nc, ..nocc, nocc..])
            //     .sub_assign(&integral_ov);
            // b_mat
            //     .slice_mut(s![nc, nocc.., ..nocc])
            //     .sub_assign(&integral_vo);
            // b_mat
            //     .slice_mut(s![nc, nocc.., nocc..])
            //     .sub_assign(&integral_vv);
            let u_2d:ArrayView2<f64> = u_mat.slice(s![nc,..,..]);
            // loop version
            for i in 0..nvirt{
                for j in 0..nocc{
                    b_mat[[nc,nocc+i,j]] = gradh_mo[[nocc+i,j]] +a_dot_u[[nocc+i,j]]
                        + u_2d[[j,nocc+i]] * orbe[j] + u_2d[[nocc+i,j]] *orbe[nocc+i] - integral_vo[[i,j]];
                }
                for j in 0..nvirt{
                    b_mat[[nc,nocc+i,nocc+j]] = gradh_mo[[nocc+i,nocc+j]] +a_dot_u[[nocc+i,nocc+j]]
                        + u_2d[[nocc+j,nocc+i]] * orbe[nocc+j] + u_2d[[nocc+i,nocc+j]] *orbe[nocc+i] - integral_vv[[i,j]];
                }
            }
            for i in 0..nocc{
                for j in 0..nocc{
                    b_mat[[nc,i,j]] = gradh_mo[[i,j]] +a_dot_u[[i,j]] +
                        u_2d[[j,i]] * orbe[j] + u_2d[[i,j]] *orbe[i] - integral_oo[[i,j]];
                }
                for j in 0..nvirt{
                    b_mat[[nc,i,nocc+j]] = gradh_mo[[i,nocc+j]] +a_dot_u[[i,nocc+j]] +
                        u_2d[[nocc+j,i]] * orbe[nocc+j] + u_2d[[i,nocc+j]] *orbe[i] - integral_ov[[i,j]];
                }
            }
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
        // let two_electron_integral_derivative:Array5<f64> = Array5::zeros([3 * self.n_atoms,nvirt,nocc,nocc,nvirt]);
        let two_electron_integral_derivative: Array5<f64> = two_electron_integral_derivative(
            s,
            g0_ao,
            g0lr_ao,
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
                            cpcis_mat[[nc, b, j]] += -1.0 *
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
        .as_standard_layout()
        .to_owned()
        .into_shape([3 * n_atoms * n_orbs * n_orbs * nvirt, n_orbs])
        .unwrap()
        .dot(&orbs.slice(s![.., ..nocc]))
        .into_shape([3 * n_atoms, n_orbs, n_orbs, nvirt, nocc])
        .unwrap()
        .permuted_axes([0, 1, 4, 3, 2])
        .as_standard_layout()
        .to_owned()
        .into_shape([3 * n_atoms * n_orbs * nocc * nvirt, n_orbs])
        .unwrap()
        .dot(&orbs.slice(s![.., ..nocc]))
        .into_shape([3 * n_atoms, n_orbs, nocc, nvirt, nocc])
        .unwrap()
        .permuted_axes([0, 4, 2, 3, 1])
        .as_standard_layout()
        .to_owned()
        .into_shape([3 * n_atoms * nocc * nocc * nvirt, n_orbs])
        .unwrap()
        .dot(&orbs.slice(s![.., nocc..]))
        .into_shape([3 * n_atoms, nocc, nocc, nvirt, nvirt])
        .unwrap()
        .permuted_axes([0, 4, 1, 2, 3])
        .as_standard_layout()
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
    let mut cphf_integral_dot: Array5<f64> = Array5::zeros((3 * n_atoms, nvirt, nocc, nocc, nvirt));

    for nc in 0..3 * n_atoms {
        let tmp_1: Array4<f64> = two_electron_integrals
            .into_shape([n_orbs * n_orbs * n_orbs, n_orbs])
            .unwrap()
            .dot(&orbs_virt)
            .into_shape([n_orbs, n_orbs, n_orbs, nvirt])
            .unwrap()
            .permuted_axes([0, 1, 3, 2])
            .as_standard_layout()
            .to_owned()
            .into_shape([n_orbs * n_orbs * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_occ)
            .into_shape([n_orbs, n_orbs, nvirt, nocc])
            .unwrap()
            .permuted_axes([0, 3, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([n_orbs * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_occ)
            .into_shape([n_orbs, nocc, nvirt, nocc])
            .unwrap()
            .permuted_axes([3, 1, 2, 0])
            .as_standard_layout()
            .to_owned()
            .into_shape([nocc * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&dc_mo_virts.slice(s![nc, .., ..]).t())
            .into_shape([nocc, nocc, nvirt, nvirt])
            .unwrap()
            .permuted_axes([3, 0, 1, 2])
            .as_standard_layout()
            .to_owned();
        let tmp_2: Array4<f64> = two_electron_integrals
            .into_shape([n_orbs * n_orbs * n_orbs, n_orbs])
            .unwrap()
            .dot(&orbs_virt)
            .into_shape([n_orbs, n_orbs, n_orbs, nvirt])
            .unwrap()
            .permuted_axes([0, 1, 3, 2])
            .as_standard_layout()
            .to_owned()
            .into_shape([n_orbs * n_orbs * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_occ)
            .into_shape([n_orbs, n_orbs, nvirt, nocc])
            .unwrap()
            .permuted_axes([0, 3, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([n_orbs * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&dc_mo_occs.slice(s![nc, .., ..]).t())
            .into_shape([n_orbs, nocc, nvirt, nocc])
            .unwrap()
            .permuted_axes([3, 1, 2, 0])
            .as_standard_layout()
            .to_owned()
            .into_shape([nocc * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_virt)
            .into_shape([nocc, nocc, nvirt, nvirt])
            .unwrap()
            .permuted_axes([3, 0, 1, 2])
            .as_standard_layout()
            .to_owned();
        let tmp_3: Array4<f64> = two_electron_integrals
            .into_shape([n_orbs * n_orbs * n_orbs, n_orbs])
            .unwrap()
            .dot(&orbs_virt)
            .into_shape([n_orbs, n_orbs, n_orbs, nvirt])
            .unwrap()
            .permuted_axes([0, 1, 3, 2])
            .as_standard_layout()
            .to_owned()
            .into_shape([n_orbs * n_orbs * nvirt, n_orbs])
            .unwrap()
            .dot(&dc_mo_occs.slice(s![nc, .., ..]).t())
            .into_shape([n_orbs, n_orbs, nvirt, nocc])
            .unwrap()
            .permuted_axes([0, 3, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([n_orbs * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_occ)
            .into_shape([n_orbs, nocc, nvirt, nocc])
            .unwrap()
            .permuted_axes([3, 1, 2, 0])
            .as_standard_layout()
            .to_owned()
            .into_shape([nocc * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_virt)
            .into_shape([nocc, nocc, nvirt, nvirt])
            .unwrap()
            .permuted_axes([3, 0, 1, 2])
            .as_standard_layout()
            .to_owned();
        let tmp_4: Array4<f64> = two_electron_integrals
            .into_shape([n_orbs * n_orbs * n_orbs, n_orbs])
            .unwrap()
            .dot(&dc_mo_virts.slice(s![nc, .., ..]).t())
            .into_shape([n_orbs, n_orbs, n_orbs, nvirt])
            .unwrap()
            .permuted_axes([0, 1, 3, 2])
            .as_standard_layout()
            .to_owned()
            .into_shape([n_orbs * n_orbs * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_occ)
            .into_shape([n_orbs, n_orbs, nvirt, nocc])
            .unwrap()
            .permuted_axes([0, 3, 2, 1])
            .as_standard_layout()
            .to_owned()
            .into_shape([n_orbs * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_occ)
            .into_shape([n_orbs, nocc, nvirt, nocc])
            .unwrap()
            .permuted_axes([3, 1, 2, 0])
            .as_standard_layout()
            .to_owned()
            .into_shape([nocc * nocc * nvirt, n_orbs])
            .unwrap()
            .dot(&orbs_virt)
            .into_shape([nocc, nocc, nvirt, nvirt])
            .unwrap()
            .permuted_axes([3, 0, 1, 2])
            .as_standard_layout()
            .to_owned();

        cphf_integral_dot
            .slice_mut(s![nc, .., .., .., ..])
            .assign(&(tmp_1 + tmp_2 + tmp_3 + tmp_4));
    }
    // loop version
    // for nc in 0..3 * n_atoms {
    //     for a in 0..nvirt {
    //         for i in 0..nocc {
    //             for j in 0..nocc {
    //                 for b in 0..nvirt {
    //                     for mu in 0..n_orbs {
    //                         for nu in 0..n_orbs {
    //                             for la in 0..n_orbs {
    //                                 for sig in 0..n_orbs {
    //                                     cphf_integral[[nc, a, i, j, b]] += two_electron_integrals
    //                                         [[mu, nu, la, sig]]
    //                                         // first
    //                                         * (dc_mo_virts[[nc, a, mu]]
    //                                         * orbs_occ[[nu, i]]
    //                                         * orbs_occ[[la, j]]
    //                                         * orbs_virt[[sig, b]]
    //                                         // second term
    //                                         + orbs_virt[[mu, a]]
    //                                         * dc_mo_occs[[nc, i, nu]]
    //                                         * orbs_occ[[la, j]]
    //                                         * orbs_virt[[sig, b]]
    //                                         // third term
    //                                         + orbs_virt[[mu, a]]
    //                                         * orbs_occ[[nu, i]]
    //                                         * dc_mo_occs[[nc, j, la]]
    //                                         * orbs_virt[[sig, b]]
    //                                         // fourth term
    //                                         + orbs_virt[[mu, a]]
    //                                         * orbs_occ[[nu, i]]
    //                                         * orbs_occ[[la, j]]
    //                                         * dc_mo_virts[[nc, b, sig]]);
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    // assert!(cphf_integral.abs_diff_eq(&cphf_integral_dot,1e-10),"Loop version NOT equal to dot version");

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

fn solve_cpcis_iterative(
    fock_terms:ArrayView2<f64>,
    lb_term:ArrayView3<f64>,
    orbs_occ:ArrayView2<f64>,
    orbs_virt:ArrayView2<f64>,
    integrals_2d:ArrayView2<f64>,
    energy:f64,
    n_atoms:usize,
    nocc:usize,
    nvirt:usize,
)->Array3<f64>{
    let norbs:usize = nocc + nvirt;
    let mut cis_der:Array3<f64> = Array3::zeros((3*n_atoms,nvirt,nocc));
    let mut iteration:usize = 0;

    'cpcis_loop: for it in 0..500{
        let prev:Array3<f64> = cis_der.clone();

        for nc in 0..3*n_atoms{
            let term_1:Array2<f64> = &fock_terms - &lb_term.slice(s![nc,..,..]);
            let term_2:Array2<f64> = orbs_virt.t().dot(& (integrals_2d
                .dot(&(orbs_virt.dot(&prev.slice(s![nc,..,..]).dot(&orbs_occ.t())))
                    .into_shape([norbs*norbs]).unwrap())).into_shape([norbs,norbs]).unwrap())
                .dot(&orbs_occ);

            cis_der.slice_mut(s![nc,..,..]).assign(&((term_1+term_2)/energy))
        }

        let diff:Array3<f64> = (&prev - &cis_der.view()).map(|val| val.abs());
        let not_converged:Vec<f64> = diff.iter().filter_map(|&item| if item > 1e-9 {Some(item)} else {None}).collect();

        if not_converged.len() == 0{
            println!("CPCIS converged in {} Iterations.",it);
            break 'cpcis_loop;
        }
        iteration = it;
        print!("{}",it);
    }
    println!(" ");
    println!("Number of iterations {}",iteration);

    return cis_der;
}





fn solve_cpcis_pople(
    amat:ArrayView2<f64>,
    bmat:ArrayView3<f64>,
    nocc:usize,
    nvirt:usize,
    nat:usize
)->Array3<f64>{
    let n_orbs:usize = nocc + nvirt;

    let mut u_matrix:Array3<f64> = Array3::zeros([3*nat,nocc,nvirt]);
    // Iteration over the gradient
    for nc in 0..3*nat{
        let b_zero:ArrayView1<f64> = bmat.slice(s![nc,..,..]).into_shape([nocc*nvirt]).unwrap();
        let mut saved_b:Vec<Array1<f64>> = Vec::new();
        let mut saved_u_dot_a:Vec<Array1<f64>> = Vec::new();
        let mut b_prev:Array1<f64> = b_zero.to_owned();
        let mut u_prev:Array1<f64> = b_zero.to_owned();
        let mut iteration:usize = 0;
        saved_b.push(b_zero.to_owned());

        let mut first_term:Array1<f64> = amat.dot(&b_prev.view());
        // let mut first_term:Array1<f64> = amat_new.dot(&b_prev);
        saved_u_dot_a.push(first_term.clone());

        'cphf_loop: for it in 0..50{
            let mut second_term:Array1<f64> = Array1::zeros(nocc*nvirt);

            // Gram Schmidt Orthogonalization
            for b_arr in saved_b.iter(){
                second_term = second_term + b_arr.dot(&first_term)/(b_arr.dot(b_arr)) * b_arr;
            }

            b_prev = &first_term - &second_term;
            saved_b.push(b_prev.clone());

            first_term = amat.dot(&b_prev.view());
            saved_u_dot_a.push(first_term.clone());

            let mut u_mat_1d:Array1<f64> = Array1::zeros((nocc*nvirt));
            // calcula the factors a_n and the contributions to the u matrix
            for (b_arr,u_dot_a) in saved_b.iter().zip(saved_u_dot_a.iter()){
                let a_factor:f64 = b_arr.dot(&b_zero)/(b_arr.dot(b_arr)-b_arr.dot(u_dot_a));
                println!("factor {}",a_factor);
                // println!("a factor {}",a_factor);
                u_mat_1d = u_mat_1d + a_factor * b_arr;
            }
            let diff:Array1<f64> = (&u_prev - &u_mat_1d).map(|val| val.abs());
            let not_converged:Vec<f64> = diff.iter().filter_map(|&item| if item > 1e-14 {Some(item)} else {None}).collect();
            u_prev = u_mat_1d;

            iteration = it;
            if not_converged.len() == 0{
                // println!("CPHF converged in {} Iterations.",it);
                break 'cphf_loop;
            }
        }
        // println!("Number of iterations {}",iteration);
        u_matrix.slice_mut(s![nc,..,..]).assign(&u_prev.into_shape([nocc,nvirt]).unwrap());
    }
    return u_matrix;
}
