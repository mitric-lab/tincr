use crate::excited_states::trans_charges;
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::lcmo::helpers::*;
use crate::fmo::{Monomer, Pair, SuperSystem, GroundStateGradient};
use crate::gradients::helpers::{f_lr, f_v};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{
    gamma_ao_wise, gamma_ao_wise_from_gamma_atomwise, gamma_atomwise_ab, gamma_gradients_ao_wise,
};
use crate::scc::h0_and_s::{h0_and_s_ab, h0_and_s_gradients};
use ndarray::prelude::*;
use ndarray_linalg::trace::Trace;
use std::ops::AddAssign;

impl SuperSystem {
    pub fn ct_gradient(
        &mut self,
        index_i: usize,
        index_j: usize,
        ct_ind_i: usize,
        ct_ind_j: usize,
    ) -> Array1<f64> {
        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        // get atoms of monomer I and J
        let atoms_i: &[Atom] = &self.atoms[m_i.slice.atom_as_range()];
        let atoms_j: &[Atom] = &self.atoms[m_j.slice.atom_as_range()];
        // norbs of both monomers
        let n_orbs_i: usize = m_i.n_orbs;
        let n_orbs_j: usize = m_j.n_orbs;

        // calculate the gradients of the fock matrix elements
        let mut gradh_i = -1.0*m_i.calculate_ct_fock_gradient(atoms_i, ct_ind_i, true);
        let gradh_j = m_j.calculate_ct_fock_gradient(atoms_j, ct_ind_j, false);
        // append the fock matrix gradients
        gradh_i.append(Axis(0), gradh_j.view()).unwrap();

        // reference to the mo coefficients of fragment I
        let c_mo_i: ArrayView2<f64> = m_i.data.orbs();
        // reference to the mo coefficients of fragment J
        let c_mo_j: ArrayView2<f64> = m_j.data.orbs();

        // get pair index
        let pair_index:usize = self.data.index_of_pair(index_i,index_j);
        // get correct pair from pairs vector
        let pair_ij: &mut Pair = &mut self.pairs[pair_index];
        // get pair atoms
        let pair_atoms: Vec<Atom> = get_pair_slice(
            &self.atoms,
            m_i.slice.atom_as_range(),
            m_j.slice.atom_as_range(),
        );
        let (grad_s_pair, grad_h0_pair) =
            h0_and_s_gradients(&pair_atoms, pair_ij.n_orbs, &pair_ij.slako);
        let grad_s_i: ArrayView3<f64> = grad_s_pair.slice(s![.., ..n_orbs_i, ..n_orbs_i]);
        let grad_s_j: ArrayView3<f64> = grad_s_pair.slice(s![.., n_orbs_i.., n_orbs_i..]);

        // calculate the overlap matrix
        if !pair_ij.data.s_is_set() {
            let mut s: Array2<f64> = Array2::zeros([pair_ij.n_orbs, pair_ij.n_orbs]);
            let (s_ab, h0_ab): (Array2<f64>, Array2<f64>) = h0_and_s_ab(
                m_i.n_orbs,
                m_j.n_orbs,
                &pair_atoms[0..m_i.n_atoms],
                &pair_atoms[m_i.n_atoms..],
                &m_i.slako,
            );

            let mu: usize = m_i.n_orbs;
            s.slice_mut(s![0..mu, 0..mu])
                .assign(&m_i.data.s());
            s.slice_mut(s![mu.., mu..])
                .assign(&m_j.data.s());
            s.slice_mut(s![0..mu, mu..]).assign(&s_ab);
            s.slice_mut(s![mu.., 0..mu]).assign(&s_ab.t());

            pair_ij.data.set_s(s);
        }

        // get the gamma matrix
        if !pair_ij.data.gamma_is_set() {
            let a: usize = m_i.n_atoms;
            let mut gamma_pair: Array2<f64> = Array2::zeros([pair_ij.n_atoms, pair_ij.n_atoms]);
            let gamma_ab: Array2<f64> = gamma_atomwise_ab(
                &pair_ij.gammafunction,
                &pair_atoms[0..m_i.n_atoms],
                &pair_atoms[m_j.n_atoms..],
                m_i.n_atoms,
                m_j.n_atoms,
            );
            gamma_pair
                .slice_mut(s![0..a, 0..a])
                .assign(&m_i.data.gamma());
            gamma_pair
                .slice_mut(s![a.., a..])
                .assign(&m_j.data.gamma());
            gamma_pair.slice_mut(s![0..a, a..]).assign(&gamma_ab);
            gamma_pair.slice_mut(s![a.., 0..a]).assign(&gamma_ab.t());

            pair_ij.data.set_gamma(gamma_pair);

            let (gamma_lr, gamma_lr_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &pair_atoms,
                pair_ij.n_atoms,
                pair_ij.n_orbs,
            );
            pair_ij.data.set_gamma_lr(gamma_lr);
            pair_ij.data.set_gamma_lr_ao(gamma_lr_ao);
        }
        // calculate the gamma matrix in AO basis
        let g0_ao: Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
            pair_ij.data.gamma(),
            &pair_atoms,
            pair_ij.n_orbs,
        );
        // calculate the gamma gradient matrix
        let (g1, g1_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
            &pair_ij.gammafunction,
            &pair_atoms,
            pair_ij.n_atoms,
            pair_ij.n_orbs,
        );
        // gamma gradient matrix for the long range correction
        let (g1_lr, g1_lr_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
            pair_ij.gammafunction_lc.as_ref().unwrap(),
            &pair_atoms,
            pair_ij.n_atoms,
            pair_ij.n_orbs,
        );

        let coulomb_gradient: Array1<f64> = f_v_ct(
            c_mo_j,
            m_i.data.s(),
            m_j.data.s(),
            grad_s_i,
            grad_s_j,
            g0_ao.view(),
            g1_ao.view(),
            pair_ij.n_atoms,
            n_orbs_i,
        )
        .into_shape([3 * pair_ij.n_atoms, n_orbs_i * n_orbs_i])
        .unwrap()
        .dot(&c_mo_i.into_shape([n_orbs_i * n_orbs_i]).unwrap());

        let exchange_gradient: Array1<f64> = f_lr_ct(
            c_mo_j.t(),
            pair_ij.data.s(),
            grad_s_pair.view(),
            pair_ij.data.gamma_lr_ao(),
            g1_lr_ao.view(),
            m_i.n_atoms,
            m_j.n_atoms,
            n_orbs_i,
        )
        .into_shape([3 * pair_ij.n_atoms, n_orbs_i * n_orbs_i])
        .unwrap()
        .dot(&c_mo_i.into_shape([n_orbs_i * n_orbs_i]).unwrap());

        // assemble the gradient
        let gradient: Array1<f64> = gradh_i + 2.0 * exchange_gradient - 1.0 * coulomb_gradient;

        return gradient;
    }
}

impl Monomer {
    pub fn calculate_ct_fock_gradient(
        &self,
        atoms: &[Atom],
        ct_index: usize,
        hole: bool,
    ) -> Array1<f64> {
        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&atoms, self.n_orbs, &self.slako);

        // get necessary arrays from properties
        let diff_p: Array2<f64> = &self.data.p() - &self.data.p_ref();
        let g0_ao: ArrayView2<f64> = self.data.gamma_ao();
        let g0:ArrayView2<f64> = self.data.gamma();
        let g1:ArrayView3<f64> = self.data.grad_gamma();
        let g1_ao: ArrayView3<f64> = self.data.grad_gamma_ao();
        let s: ArrayView2<f64> = self.data.s();
        let orbs: ArrayView2<f64> = self.data.orbs();
        let orbe:ArrayView1<f64> = self.data.orbe();

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
        let mut grad_h: Array3<f64> = &grad_h0+ &f_dmd0;

        // add the lc-gradient of the hamiltonian
        if self.gammafunction_lc.is_some(){
            let g1lr_ao: ArrayView3<f64> = self.data.grad_gamma_lr_ao();
            let g0lr_ao: ArrayView2<f64> = self.data.gamma_lr_ao();

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
        }

        // // esp atomwise
        // let esp_atomwise:Array1<f64> = 0.5*g0.dot(&self.data.dq());
        // // get Omega_AB
        // let omega_ab:Array2<f64> = atomwise_to_aowise(esp_atomwise.view(),self.n_orbs,atoms);

        // get MO coefficient for the occupied or virtual orbital
        let mut c_mo: Array1<f64> = Array1::zeros(self.n_orbs);
        let mut ind:usize = 0;
        let occ_indices: &[usize] = self.data.occ_indices();
        let virt_indices: &[usize] = self.data.virt_indices();
        let nocc:usize = occ_indices.len();
        let nvirt:usize = virt_indices.len();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, nocc));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        let mut gradient_term:Array1<f64> = Array1::zeros(3*self.n_atoms);
        // calculate transition charges
        let (qov,qoo,qvv) = trans_charges(self.n_atoms,atoms,orbs,s,occ_indices,virt_indices);
        // virtual-occupied transition charges
        let qvo:Array2<f64> = qov.clone().into_shape([self.n_atoms,nocc,nvirt]).unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned().into_shape([self.n_atoms,nvirt*nocc]).unwrap();

        // check the type of charge transfer of the fragment
        if hole {
            // orbital index
            ind= occ_indices[occ_indices.len() - 1 - ct_index];
            // orbital coefficients
            c_mo = orbs.slice(s![.., ind]).to_owned();

            // calculate last term of the orbital energy derivative
            // as shown in the Paper of H.F.Schaefer https://doi.org/10.1063/1.464483
            // in equation (25)
            // integral (ii|kl)
            let mut integral:Array2<f64> = 2.0* qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap().slice(s![..,ind,ind]).dot(&g0.dot(&qoo))
                .into_shape([nocc,nocc]).unwrap();

            if self.gammafunction_lc.is_some(){
                let g0_lr:ArrayView2<f64> = self.data.gamma_lr();
                // integral (ik|il)
                let term_2:Array2<f64> = qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap().slice(s![..,ind,..]).t()
                    .dot(&g0_lr.dot(&qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap().slice(s![..,ind,..])));
                integral = integral - term_2;

            }
            // loop over gradient
            for n_at in 0..3*self.n_atoms{
                // grads in MO basis of occupied orbitals for one gradient index
                let ds_mo:Array2<f64> = orbs_occ.t().dot(&grad_s.slice(s![n_at,..,..]).dot(&orbs_occ));
                gradient_term[n_at] = ds_mo.dot(&integral.t()).trace().unwrap();
                // gradient_term[n_at] = (ds_mo * integrals).sum();
            }
        } else {
            // orbital index
            ind = virt_indices[ct_index];
            // orbital coefficients
            c_mo = orbs.slice(s![.., ind]).to_owned();

            // calculate last term of the orbital energy derivative
            // as shown in the Paper of H.F.Schaefer https://doi.org/10.1063/1.464483
            // in equation (25)
            // integral (ii|kl)
            let slice_ind:usize = ind - virt_indices[0];
            let mut integral:Array2<f64> = 2.0 * qvv.view().into_shape([self.n_atoms,nvirt,nvirt]).unwrap().slice(s![..,slice_ind,slice_ind]).dot(&g0.dot(&qoo))
                .into_shape([nocc,nocc]).unwrap();

            if self.gammafunction_lc.is_some(){
                let g0_lr:ArrayView2<f64> = self.data.gamma_lr();
                // integral (ik|il)
                let term_2:Array2<f64> = qvo.view().into_shape([self.n_atoms,nvirt,nocc]).unwrap().slice(s![..,slice_ind,..]).t()
                    .dot(&g0_lr.dot(&qvo.view().into_shape([self.n_atoms,nvirt,nocc]).unwrap().slice(s![..,slice_ind,..])));
                integral = integral - term_2;
            }

            // loop over gradient
            for n_at in 0..3*self.n_atoms{
                // grads in MO basis of occupied orbitals for one gradient index
                let ds_mo:Array2<f64> = orbs_occ.t().dot(&grad_s.slice(s![n_at,..,..]).dot(&orbs_occ));
                gradient_term[n_at] = ds_mo.dot(&integral.t()).trace().unwrap();
                // gradient_term[n_at] = (ds_mo * integrals).sum();
            }
        }

        // transform grad_h into MO basis
        let grad_h_mo: Array1<f64> = c_mo.dot(
            &(grad_h
                .into_shape([3 * self.n_atoms * self.n_orbs, self.n_orbs])
                .unwrap()
                .dot(&c_mo)
                .into_shape([3 * self.n_atoms, self.n_orbs])
                .unwrap()
                .t()),
        );
        // transform grad_s into MO basis
        let grad_s_mo:Array1<f64> = c_mo.dot(
            &(grad_s.view()
                .into_shape([3 * self.n_atoms * self.n_orbs, self.n_orbs])
                .unwrap()
                .dot(&c_mo)
                .into_shape([3 * self.n_atoms, self.n_orbs])
                .unwrap()
                .t()),
        );
        let grad = &grad_h_mo - orbe[ind] * &grad_s_mo - &gradient_term;
        // let (u_mat,grad_omega):(Array3<f64>,Array3<f64>) = solve_cphf(
        //     grad_s.view(),
        //     grad_h0.view(),
        //     orbs,
        //     self.data.p(),
        //     s,
        //     atoms,
        //     occ_indices,
        //     self.n_orbs,
        //     self.n_atoms,
        //     orbe,
        //     omega_ab.view(),
        //     g1,
        //     g0,
        //     self.data.dq()
        // );
        // // println!("umat {}",u_mat);
        // // calculate a matrix A_ii,kl = 4 * (ii|kl) - (ik|il) - (il|ik)
        // // k = virt, l = occ
        // // let g0_lr:ArrayView2<f64> = self.data.gamma_lr();
        // let a_mat:Array2<f64> = 4.0* qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap()
        //     .slice(s![..,ind,ind]).dot(&g0.dot(&qvo)).into_shape([nvirt,nocc]).unwrap();
        //     // - qov.view().into_shape([self.n_atoms,nocc,nvirt]).unwrap().slice(s![..,ind,..]).t()
        //     // .dot(&g0_lr.dot(&qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap().slice(s![..,ind,..])))
        //     // - qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap().slice(s![..,ind,..]).t()
        //     // .dot(&g0_lr.dot(&qov.view().into_shape([self.n_atoms,nocc,nvirt]).unwrap().slice(s![..,ind,..]))).t();
        //
        // // calculate gradient term of the umatrix: sum_k sum_l U^a_kl A_ii,kl
        // let mut u_term:Array1<f64> = Array1::zeros(3*self.n_atoms);
        // for nat in 0..3*self.n_atoms{
        //     for (virt_ind,virt) in virt_indices.iter().enumerate(){
        //         for (occ_ind,occ) in occ_indices.iter().enumerate(){
        //             u_term[nat] += u_mat[[nat,*virt,*occ]] * a_mat[[virt_ind,occ_ind]];
        //         }
        //     }
        // }
        // let grad = &grad_h_mo - orbe[ind] * &grad_s_mo - &gradient_term + 1.37*&u_term;
        // let grad_u = &grad_h_mo - orbe[ind] * &grad_s_mo - &gradient_term;// + 1.4*&u_term;
        //
        // // second try for gradient calculation
        // let mut grad_2:Array1<f64> = Array1::zeros(3*self.n_atoms);
        // let mut ei:Array2<f64> = Array2::zeros((1,1));
        // ei[[0,0]] = orbe[ind];
        // let mut orbs_index:Array2<f64> = Array2::zeros((self.n_orbs,1));
        // orbs_index.slice_mut(s![..,0]).assign(&orbs.slice(s![..,ind]));
        // let ei_ao:Array2<f64> = orbs_index.dot(&ei.dot(&orbs_index.t()));
        // println!("ei ao {}",ei_ao);
        // for n_at in 0..3*self.n_atoms{
        //     let term:Array2<f64> = &grad_h0.slice(s![n_at,..,..])
        //         + &(&grad_s.slice(s![n_at,..,..]) * &omega_ab)
        //         + &grad_omega.slice(s![n_at,..,..]) * &s;
        //     grad_2[n_at] = c_mo.dot(&term.dot(&c_mo));
        // }
        // grad_2 = grad_2 - orbe[ind] * &grad_s_mo;
        //
        // // Third try for the gradient calculation
        // let mut grad_omega_aowise:Array3<f64> = Array3::zeros([3*self.n_atoms,self.n_orbs,self.n_orbs]);
        // let dq:ArrayView1<f64> = self.data.dq();
        // for nat in 0..3*self.n_atoms{
        //     let temp:Array1<f64> = 0.5 * g1.slice(s![nat,..,..]).dot(&dq);
        //     grad_omega_aowise.slice_mut(s![nat,..,..]).assign(&atomwise_to_aowise(temp.view(),self.n_orbs,atoms));
        // }
        // let mut grad_3:Array1<f64> = Array1::zeros(3*self.n_atoms);
        //
        // for n_at in 0..3*self.n_atoms{
        //     let mut term_2:Array1<f64> =  Array1::zeros([self.n_atoms]);
        //     let mut mu:usize = 0;
        //     // P_mu,nu * DS_mu,nu
        //     for (index,atom) in atoms.iter().enumerate(){
        //         for _ in 0..atom.n_orbs{
        //             term_2[index] += self.data.p().slice(s![mu,..]).dot(&grad_s.slice(s![n_at,mu,..]));
        //             mu += 1;
        //         }
        //     }
        //     // sum_C (g0_ac + g0_bc) * sum_mu on C,nu P_mu,nu * DS_mu,nu
        //     let term_1:Array1<f64> = g0.dot(&term_2);
        //     // transform to ao basis
        //     let ao_term:Array2<f64> = 0.5 * &s * &atomwise_to_aowise(term_1.view(),self.n_orbs,atoms);
        //
        //     let term:Array2<f64> = &grad_h0.slice(s![n_at,..,..])
        //         + &(&grad_s.slice(s![n_at,..,..]) * &omega_ab)
        //         + &grad_omega_aowise.slice(s![n_at,..,..]) * &s
        //         + ao_term;
        //
        //     grad_3[n_at] = c_mo.dot(&term.dot(&c_mo));
        // }
        // grad_3 = grad_3 - orbe[ind] * &grad_s_mo - &gradient_term;// + &u_term;
        //
        // // 4th try for the gradient calculation
        // let mut grad_4:Array1<f64> = Array1::zeros(3*self.n_atoms);
        //
        // // calculate gradient of charges
        // let p_mat:ArrayView2<f64> = self.data.p();
        // let grad_dq: Array2<f64> = charges_derivatives_contribution(self.n_atoms,self.n_orbs,atoms,p_mat,s,grad_s.view());
        //
        // for n_at in 0..3*self.n_atoms{
        //     // sum_C (g0_ac + g0_bc) * sum_mu on C,nu P_mu,nu * DS_mu,nu
        //     let term_1:Array1<f64> = g0.dot(&grad_dq.slice(s![n_at,..]));
        //     // transform to ao basis
        //     let ao_term:Array2<f64> = 0.5 * &s * &atomwise_to_aowise(term_1.view(),self.n_orbs,atoms);
        //
        //     let term:Array2<f64> = &grad_h0.slice(s![n_at,..,..])
        //         + &(&grad_s.slice(s![n_at,..,..]) * &omega_ab)
        //         + &grad_omega_aowise.slice(s![n_at,..,..]) * &s
        //         + ao_term;
        //
        //     grad_4[n_at] = c_mo.dot(&term.dot(&c_mo));
        // }
        // grad_4 = grad_4 - orbe[ind] * &grad_s_mo - &gradient_term;// + &u_term;
        //
        // println!("grad 1 {}",grad);
        // println!("grad 2 {}",grad_2);
        // println!("grad 3 {}",grad_3);
        // println!("grad 4 {}",grad_4);
        // println!("grad u {}",grad_u);

        return grad;
    }
}

fn f_v_ct(
    v: ArrayView2<f64>,
    s_i: ArrayView2<f64>,
    s_j: ArrayView2<f64>,
    grad_s_i: ArrayView3<f64>,
    grad_s_j: ArrayView3<f64>,
    g0_pair_ao: ArrayView2<f64>,
    g1_pair_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
) -> Array3<f64> {
    let vp: Array2<f64> = &v + &(v.t());
    let s_j_v: Array1<f64> = (&s_j * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_pair_ao.dot(&s_j_v);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let ds_i: ArrayView2<f64> = grad_s_i.slice(s![nc, .., ..]);
        let ds_j: ArrayView2<f64> = grad_s_j.slice(s![nc, .., ..]);
        let dg: ArrayView2<f64> = g1_pair_ao.slice(s![nc, .., ..]);

        let gdsv: Array1<f64> = g0_pair_ao.dot(&(&ds_j * &vp).sum_axis(Axis(0)));
        let dgsv: Array1<f64> = dg.dot(&s_j_v);
        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_i));

        for b in 0..n_orb_i {
            for a in 0..n_orb_i {
                d_f[[a, b]] = ds_i[[a, b]] * (gsv[a] + gsv[b])
                    + s_i[[a, b]] * (dgsv[a] + gdsv[a] + dgsv[b] + gdsv[b]);
            }
        }
        d_f = d_f * 0.25;
        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }

    return f_return;
}

fn f_lr_ct(
    v: ArrayView2<f64>,
    s_ij: ArrayView2<f64>,
    grad_pair_s: ArrayView3<f64>,
    g0_pair_lr_a0: ArrayView2<f64>,
    g1_pair_lr_ao: ArrayView3<f64>,
    n_atoms_i: usize,
    n_atoms_j: usize,
    n_orb_i: usize,
) -> Array3<f64> {
    let g0_lr_ao_i: ArrayView2<f64> = g0_pair_lr_a0.slice(s![..n_orb_i, ..n_orb_i]);
    let g0_lr_ao_j: ArrayView2<f64> = g0_pair_lr_a0.slice(s![n_orb_i.., n_orb_i..]);
    let g0_lr_ao_ij: ArrayView2<f64> = g0_pair_lr_a0.slice(s![..n_orb_i, n_orb_i..]);
    let n_atoms: usize = n_atoms_i + n_atoms_j;

    let sv: Array2<f64> = s_ij.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s_ij.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_ao_j * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_ao_ij).reversed_axes();
    let sgv_t: Array2<f64> = s_ij.dot(&gv).reversed_axes();

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let d_s: ArrayView2<f64> = grad_pair_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
        let d_g_i: ArrayView2<f64> = g1_pair_lr_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
        let d_g_j: ArrayView2<f64> = g1_pair_lr_ao.slice(s![nc, n_orb_i.., n_orb_i..]);
        let d_g_ij: ArrayView2<f64> = g1_pair_lr_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

        let d_sv_t: Array2<f64> = d_s.dot(&v_t);
        let d_sv: Array2<f64> = d_s.dot(&v);
        let d_gv: Array2<f64> = &d_g_j * &v;

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_i));
        // 1st term
        d_f = d_f + &g0_lr_ao_i * &(d_s.dot(&t_sv));
        // 2nd term
        d_f = d_f + (&d_sv_t * &g0_lr_ao_ij).dot(&s_ij);
        // 3rd term
        d_f = d_f + d_s.dot(&svg_t);
        // 4th term
        d_f = d_f + d_s.dot(&sgv_t);
        // 5th term
        d_f = d_f + &g0_lr_ao_i * &(s_ij.dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t * &g0_lr_ao_ij).dot(&d_s.t());
        // 7th term
        d_f = d_f + s_ij.dot(&(&d_sv * &g0_lr_ao_ij).t());
        // 8th term
        d_f = d_f + s_ij.dot(&(d_s.dot(&gv)).t());
        // 9th term
        d_f = d_f + &d_g_i * &(s_ij.dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t * &d_g_ij).dot(&s_ij);
        // 11th term
        d_f = d_f + s_ij.dot(&(&sv * &d_g_ij).t());
        // 12th term
        d_f = d_f + s_ij.dot(&(s_ij.dot(&d_gv)).t());
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

fn atomwise_to_aowise(esp_atomwise: ArrayView1<f64>, n_orbs: usize, atoms: &[Atom]) -> Array2<f64> {
    let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
    let mut mu: usize = 0;
    for (atom, esp_at) in atoms.iter().zip(esp_atomwise.iter()) {
        for _ in 0..atom.n_orbs {
            esp_ao_row[mu] = *esp_at;
            mu = mu + 1;
        }
    }
    let esp_ao_column: Array2<f64> = esp_ao_row.clone().insert_axis(Axis(1));
    let esp_ao: Array2<f64> = &esp_ao_column.broadcast((n_orbs, n_orbs)).unwrap() + &esp_ao_row;
    return esp_ao;
}

fn charge_derivative(
    p:ArrayView2<f64>,
    s:ArrayView2<f64>,
    grad_s:ArrayView3<f64>,
    orbs:ArrayView2<f64>,
    u_mat:ArrayView3<f64>,
    atoms:&[Atom],
    n_at:usize,
    n_orbs:usize,
    occ_indices:&[usize],
)->Array2<f64>{
    let mut charge_deriv:Array2<f64> = Array2::zeros([3*n_at,n_at]);
    let sc:Array2<f64> = s.dot(&orbs);

    for dir in 0..3{
        let mut mu:usize = 0;
        for (a,at_a) in atoms.iter().enumerate(){
            let a_dir:usize = 3*a + dir;

            let mut term_1:f64 = 0.0;
            let mut term_2:f64 = 0.0;
            for _ in 0..at_a.n_orbs{
                term_1 += p.slice(s![mu,..]).dot(&grad_s.slice(s![a_dir,mu,..]));

                for m in 0..n_orbs{
                    for (occ_ind,occ) in occ_indices.iter().enumerate(){
                        term_2 += 2.0 * u_mat[[a_dir,m,*occ]] * (orbs[[mu,m]] * sc[[mu,*occ]] + sc[[mu,m]] * orbs[[mu,*occ]]);
                    }
                }

                mu += 1;
            }
            charge_deriv[[a_dir,a]] = term_1 + term_2;
        }
    }
    return charge_deriv;
}

fn solve_cphf(
    grad_s:ArrayView3<f64>,
    grad_h0:ArrayView3<f64>,
    orbs:ArrayView2<f64>,
    p:ArrayView2<f64>,
    s:ArrayView2<f64>,
    atoms:&[Atom],
    occ_indices:&[usize],
    norbs:usize,
    n_at:usize,
    orbe:ArrayView1<f64>,
    omega_shift:ArrayView2<f64>,
    grad_gamma:ArrayView3<f64>,
    gamma:ArrayView2<f64>,
    dq:ArrayView1<f64>,
)->(Array3<f64>,Array3<f64>){
    // calculate initial U matrix
    // calculate grad_omega
    let mut grad_omega_atomwise:Array2<f64> = Array2::zeros([3*n_at,n_at]);
    let mut u_matrix:Array3<f64> = Array3::zeros([3*n_at,norbs,norbs]);

    for nat in 0..3*n_at{
        for orb_i in 0..norbs{
            u_matrix[[nat,orb_i,orb_i]] = -0.5 * orbs.slice(s![..,orb_i])
                .dot(&grad_s.slice(s![nat,..,..]).dot(&orbs.slice(s![..,orb_i])));
        }
        grad_omega_atomwise.slice_mut(s![nat,..]).assign(&grad_gamma.slice(s![nat,..,..]).dot(&dq));
    }

    // convergence
    let mut grad_omega_aowise:Array3<f64> = Array3::zeros([3*n_at,norbs,norbs]);
    let mut old_charge_deriv:Array2<f64> = Array2::zeros([3*n_at,n_at]);
    'cphf_loop: for it in 0..1000{
        println!("Iteration: {}",it);
        let charge_deriv:Array2<f64> = charge_derivative(
            p,
            s,
            grad_s,
            orbs,
            u_matrix.view(),
            atoms,
            n_at,
            norbs,
            occ_indices
        );
        // check convergence
        let diff_charges:Array2<f64> = (&charge_deriv - &old_charge_deriv).mapv(|val| val.abs());
        let not_converged:Vec<f64> = diff_charges.iter().filter_map(|&item| if item > 1e-15 {Some(item)} else {None}).collect();
        if not_converged.len() == 0{
            println!("CPHF converged in {} Iterations.",it);
            break 'cphf_loop;
        }
        println!("Not converged {}",not_converged.len());
        old_charge_deriv = charge_deriv.clone();

        for nat in 0..3*n_at{
            let grad_dq_atomwise:Array1<f64> = gamma.dot(&charge_deriv.slice(s![nat,..]));
            let grad_om_atom:Array1<f64> = 0.5 * (&grad_omega_atomwise.slice(s![nat,..]) + &grad_dq_atomwise);
            grad_omega_aowise.slice_mut(s![nat,..,..]).assign(&atomwise_to_aowise(grad_om_atom.view(),norbs,atoms));
        }

        for nat in 0..3*n_at{
            let term_1:Array2<f64> = &grad_h0.slice(s![nat,..,..]) + &s *&grad_omega_aowise.slice(s![nat,..,..]);

            for orb_i in 0..norbs{
                for orb_j in 0..norbs{
                    if orb_i != orb_j{
                        let term_2:Array2<f64> = &grad_s.slice(s![nat,..,..]) * &(&omega_shift - orbe[orb_j]);
                        u_matrix[[nat,orb_i,orb_j]] = 1.0/(orbe[orb_j] - orbe[orb_i]) *
                            orbs.slice(s![..,orb_i]).dot(&(&term_1+&term_2).dot(&orbs.slice(s![..,orb_j])));
                    }
                }
            }
        }
    }
    return (u_matrix,grad_omega_aowise);
}

fn charges_derivatives_contribution(
    n_atoms: usize,
    n_orbs: usize,
    atoms:&[Atom],
    p_mat: ArrayView2<f64>,
    s_mat: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
) -> (Array2<f64>) {
    // calculate w_mat
    let mut w_mat: Array3<f64> = Array3::zeros((3 * n_atoms, n_orbs, n_orbs));
    for a in (0..3 * n_atoms).into_iter() {
        w_mat
            .slice_mut(s![a, .., ..])
            .assign(&p_mat.dot(&grad_s.slice(s![a, .., ..]).dot(&p_mat)));
    }
    w_mat = -0.5 * w_mat;

    let mut matrix: Array2<f64> = Array2::zeros((3 * n_atoms, n_atoms));
    for a in (0..3 * n_atoms).into_iter() {
        let mut mu: usize = 0;
        for (c, atom_c) in atoms.iter().enumerate() {
            for _ in 0..atom_c.n_orbs {
                matrix[[a, c]] += p_mat.slice(s![mu, ..]).dot(&grad_s.slice(s![a, mu, ..]))
                + w_mat.slice(s![a, mu, ..]).dot(&s_mat.slice(s![mu, ..]));

                mu = mu + 1;
            }
        }
    }
    return matrix;
}