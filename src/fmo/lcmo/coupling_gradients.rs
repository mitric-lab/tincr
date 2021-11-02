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
            // monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];

            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );
            // calculate S,dS, gamma_AO and dgamma_AO of the pair
            pair.prepare_lcmo_gradient(&pair_atoms,m_i,m_j);
            let grad_s_pair = pair.properties.grad_s().unwrap();
            let grad_s_i: ArrayView3<f64> = grad_s_pair.slice(s![.., ..n_orbs_i, ..n_orbs_i]);
            let grad_s_j: ArrayView3<f64> = grad_s_pair.slice(s![.., n_orbs_i.., n_orbs_i..]);

            // Coulomb: S, dS, gamma_AO and dgamma_AO of the pair necessary
            let coulomb_gradient: Array1<f64> = f_le_le_coulomb(
                tdm_j.view(),
                m_i.properties.s().unwrap(),
                m_j.properties.s().unwrap(),
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
            let exchange_gradient: Array1<f64> = f_lr_le_le_exchange(
                tdm_j.t(),
                pair.properties.s().unwrap(),
                grad_s_pair.view(),
                pair.properties.gamma_lr_ao().unwrap(),
                pair.properties.grad_gamma_lr_ao().unwrap(),
                m_i.n_atoms,
                m_j.n_atoms,
                n_orbs_i,
            )
                .into_shape([3 * pair.n_atoms, n_orbs_i * n_orbs_i])
                .unwrap()
                .dot(&tdm_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

            gradient = 2.0 * coulomb_gradient - exchange_gradient;
            // TODO: reset specific part of the properties
            pair.properties.reset_gradient();
        }
        else{
            // calculate only the coulomb contribution of the gradient
            // calculate F[tdm_j]
            let pair_index:usize = self.properties.index_of_esd_pair(i.monomer.index,j.monomer.index);
            // get correct pair from pairs vector
            let esd_pair: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // monomers
            let m_i: &Monomer = &self.monomers[esd_pair.i];
            let m_j: &Monomer = &self.monomers[esd_pair.j];

            // get pair atoms
            let esd_pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );
            esd_pair.prepare_scc(&esd_pair_atoms,m_i,m_j);
            esd_pair.run_scc(&esd_pair_atoms,self.config.scf);
            esd_pair.prepare_lcmo_gradient(&esd_pair_atoms);

            let grad_s_pair = esd_pair.properties.grad_s().unwrap();
            let grad_s_i: ArrayView3<f64> = grad_s_pair.slice(s![.., ..n_orbs_i, ..n_orbs_i]);
            let grad_s_j: ArrayView3<f64> = grad_s_pair.slice(s![.., n_orbs_i.., n_orbs_i..]);

            // Coulomb: S, dS, gamma_AO and dgamma_AO necessary
            let coulomb_gradient: Array1<f64> = f_le_le_coulomb(
                tdm_j.view(),
                m_i.properties.s().unwrap(),
                m_j.properties.s().unwrap(),
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
            // reset properties of the esd_pair
            esd_pair.properties.reset();
        }
    }

    pub fn ct_ct_coupling_grad<'a>(&mut self, state_1: &'a ChargeTransfer<'a>, state_2: &'a ChargeTransfer<'a>){

    }

    pub fn le_ct_coupling_grad<'a>(&mut self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>){
        // self.le_ct_1e_coupling_grad(i,j) + self.le_ct_2e_coupling_grad(i,j);
    }

    pub fn ct_le_coupling_grad<'a>(&mut self, i: &'a ChargeTransfer<'a>, j: &'a LocallyExcited<'a>){
        self.le_ct_coupling_grad(j, i)
    }

    pub fn le_ct_1e_coupling_grad<'a>(&mut self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) {

    }

    pub fn le_ct_2e_coupling_grad<'a>(&mut self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) {
        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.monomer.index, j.hole.idx);
        // The same for I and K
        let type_ik: PairType = self.properties.type_of_pair(i.monomer.index, j.electron.idx);
        // and J K
        let type_jk: PairType = self.properties.type_of_pair(j.electron.idx, j.hole.idx);

        // transform the CI coefficients of the monomers to the AO basis
        let nocc = i.monomer.properties.n_occ().unwrap();
        let nvirt = i.monomer.properties.n_virt().unwrap();
        let cis_c = i.tdm.into_shape([nocc,nvirt]).unwrap();
        let tdm:Array2<f64> = i.occs.dot(&cis_c.dot(&i.virts.t()));
        let n_orbs_i: usize = i.monomer.n_orbs;

        // < LE I | H | CT J_j -> I_b>
        if i.monomer.index == j.electron.idx {
            // Check if the pair IK is close, so that the overlap is non-zero.
            if type_ik == PairType::Pair {
                // get the index of the pair
                let pair_index:usize = self.properties.index_of_pair(i.monomer.index,j.hole.idx);
                // get the pair from pairs vector
                let pair: &mut Pair = &mut self.pairs[pair_index];
                // monomers
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];
                let n_atoms:usize = m_i.n_atoms + m_j.n_atoms;
                let n_orbs_j:usize = m_j.n_orbs;

                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    m_i.slice.atom_as_range(),
                    m_j.slice.atom_as_range(),
                );
                // calculate S,dS, gamma_AO and dgamma_AO of the pair
                pair.prepare_lcmo_gradient(&pair_atoms,m_i,m_j);

                if pair.i == i.monomer.index{
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms I + 3*n_atoms J]
                    let coulomb_gradient:Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    ).into_shape([3*n_atoms*n_orbs_i,n_orbs_j]).unwrap().dot(&j.hole.mo.c)
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.electron.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient:Array1<f64> = f_lr_le_ct_exchange_hole_j(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    ).into_shape([3*n_atoms*n_orbs_i,n_orbs_j]).unwrap().dot(&j.hole.mo.c)
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.electron.mo.c);;
                }
                else{
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms J + 3*n_atoms I]
                    let coulomb_gradient:Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    ).into_shape([3*n_atoms*n_orbs_i,n_orbs_j]).unwrap().dot(&j.hole.mo.c)
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.electron.mo.c);

                    // calculate the gradient of the exchange integral
                    // calculate the gradient of the exchange integral
                    let exchange_gradient:Array1<f64> = f_lr_le_ct_exchange_hole_j(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    ).into_shape([3*n_atoms*n_orbs_i,n_orbs_j]).unwrap().dot(&j.hole.mo.c)
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.electron.mo.c);;
                }

                pair.properties.reset_gradient();
            }
            else{
                // If overlap IK is zero, the coupling is zero.
            }
        }
        else if i.monomer.index == j.hole.idx {
            // Check if the pair IJ is close, so that the overlap is non-zero.
            if type_ij == PairType::Pair {
                // get the index of the pair
                let pair_index:usize = self.properties.index_of_pair(i.monomer.index,j.electron.idx);
                // get the pair from pairs vector
                let pair: &mut Pair = &mut self.pairs[pair_index];
                // monomers
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];
                let n_atoms:usize = m_i.n_atoms + m_j.n_atoms;
                let n_orbs_j:usize = m_j.n_orbs;

                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    m_i.slice.atom_as_range(),
                    m_j.slice.atom_as_range(),
                );
                // calculate S,dS, gamma_AO and dgamma_AO of the pair
                pair.prepare_lcmo_gradient(&pair_atoms,m_i,m_j);

                if pair.i == i.monomer.index{
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms I + 3*n_atoms J]
                    let coulomb_gradient:Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    ).into_shape([3*n_atoms*n_orbs_i,n_orbs_j]).unwrap().dot(&j.electron.mo.c)
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.hole.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient:Array1<f64> = f_lr_le_ct_exchange_hole_i(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    ).into_shape([3*n_atoms*n_orbs_i,n_orbs_j]).unwrap().dot(&j.electron.mo.c)
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.hole.mo.c);
                }
                else{
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms J + 3*n_atoms I]
                    let coulomb_gradient:Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    ).into_shape([3*n_atoms*n_orbs_i,n_orbs_j]).unwrap().dot(&j.electron.mo.c)
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.hole.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient:Array1<f64> = f_lr_le_ct_exchange_hole_i(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    ).into_shape([3*n_atoms*n_orbs_i,n_orbs_j]).unwrap().dot(&j.electron.mo.c)
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.hole.mo.c);
                }

                pair.properties.reset_gradient();
            }
            else{
                // If overlap IJ is zero, the coupling is zero.
            }
        }
        // < LE I_ia | H | CT K_j -> J_b >
        else{
            // The integral (ia|jb) requires that the overlap between K and J is non-zero.
            // coulomb
            if type_jk == PairType::Pair {

            }
            else{
                // If overlap JK is zero, the integral is zero.
            }
            // exchange
            if type_ik == PairType::Pair && type_ij == PairType::Pair {

            }
            else{
                // If overlap IK or IJ is zero, the integral is zero.
            }
        }
    }
}

fn f_le_ct_coulomb(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_pair_ao: ArrayView2<f64>,
    g1_pair_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
    bool_ij:bool,
) -> Array3<f64> {
    // The pair indices are IJ -> I < J
    let (s_i,s_ij,g_i,g_ij) = if bool_ij{
        let s_i:ArrayView2<f64> = s.slice(s![..n_orb_i,..n_orb_i]);
        let s_ij:ArrayView2<f64> = s.slice(s![..n_orb_i,n_orb_i..]);
        let g_i:ArrayView2<f64> = g0_pair_ao.slice(s![..n_orb_i,..n_orb_i]);
        let g_ij:ArrayView2<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]);

        (s_i,s_ij,g_i,g_ij)
    }else{
        // The pair indices are JI -> J < I
        let s_i:ArrayView2<f64> = s.slice(s![n_orb_j..,n_orb_j..]);
        let s_ij:ArrayView2<f64> = s.slice(s![n_orb_j..,..n_orb_j]);
        let g_i:ArrayView2<f64> = g0_pair_ao.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_pair_ao.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    };


    let si_v: Array1<f64> = (&s_i * &v).sum_axis(Axis(1));
    let gi_sv:Array1<f64> = g_i.dot(&si_v);
    let gij_sv: Array1<f64> = g_ij.dot(&si_v);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_j));

    for nc in 0..3 * n_atoms {
        let (ds_i,ds_ij,dg_i,dg_ij) = if bool_ij{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_pair_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij:ArrayView2<f64> = g1_pair_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i,ds_ij,dg_i,dg_ij)
        }else{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,n_orb_j..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_pair_ao.slice(s![nc, n_orb_j..,n_orb_j..]);
            let dg_ij:ArrayView2<f64> = g1_pair_ao.slice(s![nc, n_orb_j..,..n_orb_j]);

            (ds_i,ds_ij,dg_i,dg_ij)
        };

        let gi_dsv:Array1<f64> = g_i.dot(&(&ds_i * &v).sum_axis(Axis(1)));
        let gij_dsv:Array1<f64> = g_ij.t().dot(&(&ds_i * &v).sum_axis(Axis(1)));
        let dgi_sv:Array1<f64> = dg_i.dot(&si_v);
        let dgij_sv:Array1<f64> = dg_ij.dot(&si_v);

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_j));

        for b in 0..n_orb_i {
            for a in 0..n_orb_j {
                d_f[[b, a]] = 2.0* ds_ij[[b, a]]  * (gi_sv[b] + gij_sv[a])
                    + 2.0 * s_ij[[b, a]] * (gi_dsv[b] + gij_dsv[a] + dgi_sv[b] + dgij_sv[a]);
            }
        }
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }

    return f_return;
}

fn f_lr_le_ct_exchange_hole_i(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
    bool_ij:bool,
) -> Array3<f64> {
    let (s_i,s_ij,g_i,g_ij) = if bool_ij{
        let s_i:ArrayView2<f64> = s.slice(s![..n_orb_i,..n_orb_i]);
        let s_ij:ArrayView2<f64> = s.slice(s![..n_orb_i,n_orb_i..]);
        let g_i:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    }
    else{
        let s_i:ArrayView2<f64> = s.slice(s![n_orb_j..,n_orb_j..]);
        let s_ij:ArrayView2<f64> = s.slice(s![n_orb_j..,..n_orb_j]);
        let g_i:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    };

    // for term 1
    let gi_v: Array2<f64> = &g_i * &v;
    // for term 1
    let gi_v_sij:Array2<f64> = gi_v.dot(&s_ij);
    // for term 2
    let v_si:Array2<f64> = v.dot(&s_i);
    // for term 4, 10
    let v_sij:Array2<f64> = v.dot(&s_ij);
    // for term 5
    let gi_v_si:Array2<f64> = s_i.dot(&gi_v);
    // for term 7, 11, 12
    let vt_si:Array2<f64> = v.t().dot(&s_i);
    // for term 7
    let gi_vt_si:Array2<f64> = &g_i * &vt_si;
    // for term 8
    let si_v:Array2<f64> = s_i.dot(&v);
    // for term 12
    let vt_si_t_sij:Array2<f64> = vt_si.t().dot(&s_ij);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let (ds_i,ds_ij,dg_i,dg_ij) = if bool_ij{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i,ds_ij,dg_i,dg_ij)
        }else{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,n_orb_j..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j..,n_orb_j..]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j..,..n_orb_j]);

            (ds_i,ds_ij,dg_i,dg_ij)
        };

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_j));
        // 1st term
        d_f = d_f + ds_i.dot(&gi_v_sij);
        // 2nd term
        d_f = d_f + (&v_si * &ds_i).t().dot(&g_ij);
        // 3rd term
        d_f = d_f + (&ds_i.dot(&v) * &g_i).dot(&s_ij);
        // 4th term
        d_f = d_f + &ds_i.dot(&v_sij) *&g_ij;
        // 5th term
        d_f = d_f + gi_v_si.dot(&ds_ij);
        // 6th term
        d_f = d_f + s_i.dot(&(&v.dot(&ds_ij) *&g_ij));
        // 7th term
        d_f = d_f + gi_vt_si.t().dot(&ds_ij);
        // 8th term
        d_f = d_f + &si_v.dot(&ds_ij) * &g_ij;
        // 9th term
        d_f = d_f + s_i.dot(&(&dg_i*&v).dot(&s_ij));
        // 10th term
        d_f = d_f + s_i.dot(&(&v_sij * &dg_ij));
        // 11th term
        d_f = d_f + (&vt_si*&dg_i).t().dot(&s_ij);
        // 12th term
        d_f = d_f + &vt_si_t_sij*&dg_ij;
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

fn f_lr_le_ct_exchange_hole_j(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
    bool_ij:bool,
) -> Array3<f64> {
    let (s_i,s_ij,g_i,g_ij) = if bool_ij{
        let s_i:ArrayView2<f64> = s.slice(s![..n_orb_i,..n_orb_i]);
        let s_ij:ArrayView2<f64> = s.slice(s![..n_orb_i,n_orb_i..]);
        let g_i:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    }
    else{
        let s_i:ArrayView2<f64> = s.slice(s![n_orb_j..,n_orb_j..]);
        let s_ij:ArrayView2<f64> = s.slice(s![n_orb_j..,..n_orb_j]);
        let g_i:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,n_orb_j..]);
        let g_ij:ArrayView2<f64> = g0_lr_a0.slice(s![n_orb_j..,..n_orb_j]);

        (s_i,s_ij,g_i,g_ij)
    };

    // for term 1
    let gi_v: Array2<f64> = &g_i * &v;
    // for term 1
    let gi_v_si:Array2<f64> = gi_v.dot(&s_i);
    // for term 4,10
    let v_si:Array2<f64> = v.dot(&s_i);
    // for term 2
    let v_sij:Array2<f64> = v.dot(&s_ij);
    // for term 5
    let gi_v_sij:Array2<f64> = gi_v.t().dot(&s_ij);
    // for term 7, 11, 12
    let vt_sij:Array2<f64> = v.t().dot(&s_ij);
    // for term 7
    let gij_vt_sij:Array2<f64> = &g_ij * &vt_sij;
    // for term 8
    let sij_v:Array2<f64> = s_ij.t().dot(&v);
    // for term 12
    let si_t_vt_sij:Array2<f64> = s_i.t().dot(&vt_sij);

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb_i, n_orb_i));

    for nc in 0..3 * n_atoms {
        let (ds_i,ds_ij,dg_i,dg_ij) = if bool_ij{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, ..n_orb_i, n_orb_i..]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, ..n_orb_i]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, ..n_orb_i, n_orb_i..]);

            (ds_i,ds_ij,dg_i,dg_ij)
        }else{
            let ds_i: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,n_orb_j..]);
            let ds_ij: ArrayView2<f64> = grad_s.slice(s![nc, n_orb_j..,..n_orb_j]);
            let dg_i: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j..,n_orb_j..]);
            let dg_ij: ArrayView2<f64> = g1_lr_ao.slice(s![nc, n_orb_j..,..n_orb_j]);

            (ds_i,ds_ij,dg_i,dg_ij)
        };

        let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_j));
        // 1st term
        d_f = d_f + gi_v_si.t().dot(&ds_ij);
        // 2nd term
        d_f = d_f + g_i.dot(&(&v_sij * &ds_ij));
        // 3rd term
        d_f = d_f + s_i.dot(&(&v.t().dot(&ds_ij) * &g_ij));
        // 4th term
        d_f = d_f + v_si.t().dot(&ds_ij) *&g_ij;
        // 5th term
        d_f = d_f + ds_i.t().dot(&gi_v_sij);
        // 6th term
        d_f = d_f + (&v.dot(&ds_i) *&g_i).t().dot(&s_ij);
        // 7th term
        d_f = d_f + ds_i.t().dot(&gij_vt_sij);
        // 8th term
        d_f = d_f + &ds_i.t().dot(&sij_v.t()) * &g_ij;
        // 9th term
        d_f = d_f + (&dg_i*&v).dot(&s_i).t().dot(&s_ij);
        // 10th term
        d_f = d_f + (&v_si * &dg_i).t().dot(&s_ij);
        // 11th term
        d_f = d_f + s_i.t().dot(&(&vt_sij*&dg_ij));
        // 12th term
        d_f = d_f + &si_t_vt_sij*&dg_ij;
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

fn f_le_le_coulomb(
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

fn f_lr_le_le_exchange(
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