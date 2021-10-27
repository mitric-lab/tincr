use crate::fmo::{
    BasisState, ChargeTransfer, LocallyExcited, Monomer, PairType, Particle, SuperSystem, LRC,
};
use crate::initialization::{Atom};
use crate::fmo::{Pair,ESDPair};
use ndarray::prelude::*;
use crate::fmo::helpers::get_pair_slice;
use std::net::UdpSocket;

impl SuperSystem {
    pub fn exciton_coupling_gradient<'a>(&mut self, lhs: &'a BasisState<'a>, rhs: &'a BasisState<'a>){
        match (lhs, rhs) {
            // Coupling between two LE states.
            (BasisState::LE(ref a), BasisState::LE(ref b)) => self.le_le_coupling_grad(a, b),
            // Coupling between LE and CT state.
            (BasisState::LE(ref a), BasisState::CT(ref b)) => self.le_ct_coupling_grad(a, b),
            // Coupling between CT and LE state.
            (BasisState::CT(ref a), BasisState::LE(ref b)) => self.ct_le_coupling_grad(a, b),
            // Coupling between CT and CT
            (BasisState::CT(ref a), BasisState::CT(ref b)) => {
                self.ct_ct_coupling_grad(a, b)
            },
        }
    }

    pub fn le_le_coupling_grad<'a>(&mut self, i: &'a LocallyExcited<'a>, j: &'a LocallyExcited<'a>) {
        // Check if the ESD approximation is used or not.
        let type_pair: PairType = self
            .properties
            .type_of_pair(i.monomer.index, j.monomer.index);

        // transform the CI coefficients of the monomers to the AO basis
        let nocc_i = i.monomer.properties.n_occ().unwrap();
        let nvirt_i = i.monomer.properties.n_virt().unwrap();
        let cis_c_i = i.tdm.into_shape([nocc_i,nvirt_i]).unwrap();
        let tdm_i:Array2<f64> = i.occs.dot(&cis_c_i.dot(&i.virts.t()));
        let n_orbs_i: usize = i.monomer.n_orbs;

        let nocc_j = j.monomer.properties.n_occ().unwrap();
        let nvirt_j = j.monomer.properties.n_virt().unwrap();
        let cis_c_j = j.tdm.into_shape([nocc_j,nvirt_j]).unwrap();
        let tdm_j:Array2<f64> = j.occs.dot(&cis_c_j.dot(&j.virts.t()));

        let n_atoms:usize = i.monomer.n_atoms + j.monomer.n_atoms;
        let mut gradient:Array1<f64> = Array1::zeros(3*n_atoms);

        if type_pair == PairType::Pair{
            // calculate the coulomb and exchange contribution of the gradient
            // calculate F[tdm_j] and F_lr[tdm_j]

            // get the index of the pair
            let pair_index:usize = self.properties.index_of_pair(i.monomer.index,j.monomer.index);
            // get the pair from pairs vector
            let pair: &mut Pair = &mut self.pairs[pair_index];
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                i.monomer.slice.atom_as_range(),
                j.monomer.slice.atom_as_range(),
            );
            // calculate S,dS, gamma_AO and dgamma_AO of the pair
            pair.prepare_lcmo_gradient(&pair_atoms,i.monomer,j.monomer);
            let grad_s_pair = pair.properties.grad_s().unwrap();
            let grad_s_i: ArrayView3<f64> = grad_s_pair.slice(s![.., ..n_orbs_i, ..n_orbs_i]);
            let grad_s_j: ArrayView3<f64> = grad_s_pair.slice(s![.., n_orbs_i.., n_orbs_i..]);

            // Coulomb: S, dS, gamma_AO and dgamma_AO of the pair necessary
            let coulomb_gradient: Array1<f64> = f_pair_coulomb(
                tdm_j.view(),
                i.monomer.properties.s().unwrap(),
                j.monomer.properties.s().unwrap(),
                grad_s_i,
                grad_s_j,
                pair.properties.gamma_ao().unwrap(),
                pair.properties.grad_gamma_ao().unwrap(),
                pair.n_atoms,
                n_orbs_i,
            )
                .into_shape([3 * pair.n_atoms, n_orbs_i * n_orbs_i])
                .unwrap()
                .dot(&tdm_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

            // Exchange: S, dS, gamma_AO_lr and dgamma_AO_lr of the pair necessary
            let exchange_gradient: Array1<f64> = f_lr_pair_exchange(
                tdm_j.t(),
                pair.properties.s().unwrap(),
                grad_s_pair.view(),
                pair.properties.gamma_lr_ao().unwrap(),
                pair.properties.grad_gamma_lr_ao().unwrap(),
                i.monomer.n_atoms,
                j.monomer.n_atoms,
                n_orbs_i,
            )
                .into_shape([3 * pair.n_atoms, n_orbs_i * n_orbs_i])
                .unwrap()
                .dot(&tdm_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

            gradient = 2.0 * coulomb_gradient - exchange_gradient;

        }
        else{
            // calculate only the coulomb contribution of the gradient
            // calculate F[tdm_j]
            let pair_index:usize = self.properties.index_of_esd_pair(i.monomer.index,j.monomer.index);
            // get correct pair from pairs vector
            let esd_pair: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // get pair atoms
            let esd_pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                i.monomer.slice.atom_as_range(),
                j.monomer.slice.atom_as_range(),
            );
            esd_pair.prepare_scc(&esd_pair_atoms,i.monomer,j.monomer);
            esd_pair.run_scc(&esd_pair_atoms,self.config.scf);
            esd_pair.prepare_lcmo_gradient(&esd_pair_atoms);

            let grad_s_pair = esd_pair.properties.grad_s().unwrap();
            let grad_s_i: ArrayView3<f64> = grad_s_pair.slice(s![.., ..n_orbs_i, ..n_orbs_i]);
            let grad_s_j: ArrayView3<f64> = grad_s_pair.slice(s![.., n_orbs_i.., n_orbs_i..]);

            // Coulomb: S, dS, gamma_AO and dgamma_AO necessary
            let coulomb_gradient: Array1<f64> = f_pair_coulomb(
                tdm_j.view(),
                i.monomer.properties.s().unwrap(),
                j.monomer.properties.s().unwrap(),
                grad_s_i,
                grad_s_j,
                esd_pair.properties.gamma_ao().unwrap(),
                esd_pair.properties.grad_gamma_ao().unwrap(),
                esd_pair.n_atoms,
                n_orbs_i,
            )
                .into_shape([3 * esd_pair.n_atoms, n_orbs_i * n_orbs_i])
                .unwrap()
                .dot(&tdm_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

            gradient = 2.0 * coulomb_gradient;
        }
    }

    pub fn ct_ct_coupling_grad<'a>(&self, state_1: &'a ChargeTransfer<'a>, state_2: &'a ChargeTransfer<'a>){

    }

    pub fn le_ct_coupling_grad<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>){

    }

    pub fn ct_le_coupling_grad<'a>(&self, i: &'a ChargeTransfer<'a>, j: &'a LocallyExcited<'a>){
        self.le_ct_coupling_grad(j, i)
    }

    pub fn le_ct_1e_coupling_grad<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) {

    }

    pub fn le_ct_2e_coupling_grad<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) {

    }
}

fn f_pair_coulomb(
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
    let gsv: Array1<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]).dot(&s_j_v);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let ds_i: ArrayView2<f64> = grad_s_i.slice(s![nc, .., ..]);
        let ds_j: ArrayView2<f64> = grad_s_j.slice(s![nc, .., ..]);
        let dg: ArrayView2<f64> = g1_pair_ao.slice(s![nc, .., ..]);

        let gdsv: Array1<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]).dot(&(&ds_j * &vp).sum_axis(Axis(0)));
        let dgsv: Array1<f64> = dg.slice(s![..n_orb_i,n_orb_i..]).dot(&s_j_v);
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

fn f_lr_pair_exchange(
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
    let s_ij_outer:ArrayView2<f64> = s_ij.slice(s![..n_orb_i,n_orb_i..]);
    let n_atoms: usize = n_atoms_i + n_atoms_j;

    let sv: Array2<f64> = s_ij_outer.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s_ij_outer.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_ao_j * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_ao_ij).reversed_axes();
    let sgv_t: Array2<f64> = s_ij_outer.dot(&gv).reversed_axes();

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
        d_f = d_f + (&d_sv_t * &g0_lr_ao_ij).dot(&s_ij_outer.t());
        // 3rd term
        d_f = d_f + d_s.dot(&svg_t);
        // 4th term
        d_f = d_f + d_s.dot(&sgv_t);
        // 5th term
        d_f = d_f + &g0_lr_ao_i * &(s_ij_outer.dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t * &g0_lr_ao_ij).dot(&d_s.t());
        // 7th term
        d_f = d_f + s_ij_outer.dot(&(&d_sv * &g0_lr_ao_ij).t());
        // 8th term
        d_f = d_f + s_ij_outer.dot(&(d_s.dot(&gv)).t());
        // 9th term
        d_f = d_f + &d_g_i * &(s_ij_outer.dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t * &d_g_ij).dot(&s_ij_outer.t());
        // 11th term
        d_f = d_f + s_ij_outer.dot(&(&sv * &d_g_ij).t());
        // 12th term
        d_f = d_f + s_ij_outer.dot(&(s_ij_outer.dot(&d_gv)).t());
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}