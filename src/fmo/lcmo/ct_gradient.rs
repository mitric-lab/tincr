use crate::excited_states::trans_charges;
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::lcmo::helpers::*;
use crate::fmo::{Monomer, Pair, SuperSystem};
use crate::gradients::helpers::{f_lr, f_v};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{
    gamma_ao_wise, gamma_ao_wise_from_gamma_atomwise, gamma_atomwise_ab, gamma_gradients_ao_wise,
};
use crate::scc::h0_and_s::{h0_and_s_ab, h0_and_s_gradients};
use ndarray::prelude::*;
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
        let mut gradh_i = m_i.calculate_ct_fock_gradient(atoms_i, ct_ind_i, true);
        let gradh_j = m_j.calculate_ct_fock_gradient(atoms_j, ct_ind_j, false);
        // append the fock matrix gradients
        gradh_i.append(Axis(0), gradh_j.view()).unwrap();

        // reference to the mo coefficients of fragment I
        let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
        // reference to the mo coefficients of fragment J
        let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

        // get pair from indices
        // TODO: Take pair index from hashmap with monomer and pair indices
        let pair_ij: &mut Pair = &mut self.pairs[0];
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
        if pair_ij.properties.s().is_none() {
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
                .assign(&m_i.properties.s().unwrap());
            s.slice_mut(s![mu.., mu..])
                .assign(&m_j.properties.s().unwrap());
            s.slice_mut(s![0..mu, mu..]).assign(&s_ab);
            s.slice_mut(s![mu.., 0..mu]).assign(&s_ab.t());

            pair_ij.properties.set_s(s);
        }

        // get the gamma matrix
        if pair_ij.properties.gamma().is_none() {
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
                .assign(&m_i.properties.gamma().unwrap());
            gamma_pair
                .slice_mut(s![a.., a..])
                .assign(&m_j.properties.gamma().unwrap());
            gamma_pair.slice_mut(s![0..a, a..]).assign(&gamma_ab);
            gamma_pair.slice_mut(s![a.., 0..a]).assign(&gamma_ab.t());

            pair_ij.properties.set_gamma(gamma_pair);

            let (gamma_lr, gamma_lr_ao): (Array2<f64>, Array2<f64>) = gamma_ao_wise(
                self.gammafunction_lc.as_ref().unwrap(),
                &pair_atoms,
                pair_ij.n_atoms,
                pair_ij.n_orbs,
            );
            pair_ij.properties.set_gamma_lr(gamma_lr);
            pair_ij.properties.set_gamma_lr_ao(gamma_lr_ao);
        }
        // calculate the gamma matrix in AO basis
        let g0_ao: Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
            pair_ij.properties.gamma().unwrap(),
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
            m_i.properties.s().unwrap(),
            m_j.properties.s().unwrap(),
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
            pair_ij.properties.s().unwrap(),
            grad_s_pair.view(),
            pair_ij.properties.gamma_lr_ao().unwrap(),
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
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let g1lr_ao: ArrayView3<f64> = self.properties.grad_gamma_lr_ao().unwrap();
        let g0lr_ao: ArrayView2<f64> = self.properties.gamma_lr_ao().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();

        // calculate gradH: gradH0 + gradHexc
        let f_dmd0: Array3<f64> = f_v(
            diff_p.view(),
            s,
            grad_s.view(),
            g0_ao,
            g1_ao,
            self.n_atoms,
            self.n_orbs,
        );
        let flr_dmd0: Array3<f64> = f_lr(
            diff_p.view(),
            s,
            grad_s.view(),
            g0lr_ao,
            g1lr_ao,
            self.n_atoms,
            self.n_orbs,
        );

        // derivative of a fock matrix element in AO basis
        let grad_h: Array3<f64> = &grad_h0 + &f_dmd0 - 0.5 * &flr_dmd0;

        // get MO coefficient for the occupied or virtual orbital
        let mut c_mo: Array1<f64> = Array1::zeros(self.n_orbs);
        if hole {
            let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
            let ind: usize = occ_indices[occ_indices.len() - 1 - ct_index];
            c_mo = orbs.slice(s![.., ind]).to_owned();
        } else {
            let virt_indices: &[usize] = self.properties.virt_indices().unwrap();
            let ind: usize = virt_indices[ct_index];
            c_mo = orbs.slice(s![.., ind]).to_owned();
        }

        // transform grad_h into MO basis
        let grad: Array1<f64> = c_mo.dot(
            &(grad_h
                .into_shape([3 * self.n_atoms * self.n_orbs, self.n_orbs])
                .unwrap()
                .dot(&c_mo)
                .into_shape([3 * self.n_atoms, self.n_orbs])
                .unwrap()
                .t()),
        );

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
