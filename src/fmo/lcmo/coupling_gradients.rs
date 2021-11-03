use crate::fmo::{
    BasisState, ChargeTransfer, LocallyExcited, Monomer, PairType, Particle, SuperSystem, LRC,
};
use crate::initialization::{Atom};
use crate::fmo::{Pair,ESDPair};
use ndarray::prelude::*;
use crate::fmo::helpers::get_pair_slice;
use std::net::UdpSocket;
use crate::fmo::lcmo::helpers::{f_le_ct_coulomb, f_lr_le_ct_exchange_hole_i, f_lr_le_ct_exchange_hole_j, f_le_le_coulomb, f_lr_le_le_exchange, f_coulomb_ct_ct_ijij, f_exchange_ct_ct_ijij, f_coul_ct_ct_ijji, f_exchange_ct_ct_ijji};
use crate::fmo::lcmo::integrals::CTCoupling;
use ndarray_linalg::{into_col, into_row};

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

    pub fn le_ct_coupling_grad<'a>(&mut self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>){
        // self.le_ct_1e_coupling_grad(i,j);
        self.le_ct_2e_coupling_grad(i,j);
    }

    pub fn ct_le_coupling_grad<'a>(&mut self, i: &'a ChargeTransfer<'a>, j: &'a LocallyExcited<'a>){
        self.le_ct_coupling_grad(j, i);
    }

    pub fn le_ct_1e_coupling_grad<'a>(&mut self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) {
        // Neglect the contribution? Depends on the derivative of the Fock matrix
        // -> CPHF necessary
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

                let (grad_i,grad_j) = if pair.i == i.monomer.index{
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
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.electron.mo.c);

                    let gradient:Array1<f64> = 2.0* coulomb_gradient - exchange_gradient;
                    let gradient_i = gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                    let gradient_j = gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                    (gradient_i,gradient_j)
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
                        .into_shape([3*n_atoms,n_orbs_i]).unwrap().dot(&j.electron.mo.c);

                    let gradient:Array1<f64> = 2.0* coulomb_gradient - exchange_gradient;
                    let gradient_j = gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                    let gradient_i = gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                    (gradient_i,gradient_j)
                };

                pair.properties.reset_gradient();
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

                let (grad_i,grad_j) = if pair.i == i.monomer.index{
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

                    let gradient:Array1<f64> = 2.0* coulomb_gradient - exchange_gradient;
                    let gradient_i = gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                    let gradient_j = gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                    (gradient_i,gradient_j)
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

                    let gradient:Array1<f64> = 2.0* coulomb_gradient - exchange_gradient;
                    let gradient_j = gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                    let gradient_i = gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                    (gradient_i,gradient_j)
                };

                pair.properties.reset_gradient();
            }
        }
        // < LE I_ia | H | CT K_j -> J_b >
        else{
            // The integral (ia|jb) requires that the overlap between K and J is non-zero.
            // coulomb
            if type_jk == PairType::Pair {

            }
            // exchange
            if type_ik == PairType::Pair && type_ij == PairType::Pair {

            }
        }
    }

    pub fn ct_ct_coupling_grad<'a>(&mut self, state_1: &'a ChargeTransfer<'a>, state_2: &'a ChargeTransfer<'a>){
        // i -> electron on I, j -> hole on J.
        let (i, j): (&Particle, &Particle) = (&state_1.electron, &state_1.hole);
        // k -> electron on K, l -> hole on L.
        let (k, l): (&Particle, &Particle) = (&state_2.electron, &state_2.hole);

        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.idx, j.idx);
        // I and K
        let type_ik: PairType = self.properties.type_of_pair(i.idx, k.idx);
        // I and L
        let type_il: PairType = self.properties.type_of_pair(i.idx, l.idx);
        // J and K
        let type_jk: PairType = self.properties.type_of_pair(j.idx, k.idx);
        // J and L
        let type_jl: PairType = self.properties.type_of_pair(j.idx, l.idx);
        // K and L
        let type_kl: PairType = self.properties.type_of_pair(k.idx, l.idx);

        // Check how many monomers are involved in this matrix element.
        let kind: CTCoupling = CTCoupling::from((i, j, k, l));

        match kind {
            // electrons i,k on I, holes j,l on J
            CTCoupling::IJIJ => {
                if type_ij == PairType::Pair{
                    // get the index of the pair
                    let pair_index:usize = self.properties.index_of_pair(i.monomer.index,j.monomer.index);
                    // get the pair from pairs vector
                    let pair: &mut Pair = &mut self.pairs[pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];
                    let n_atoms:usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i:usize = m_i.n_orbs;
                    let orbs_j:usize = m_j.n_orbs;

                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    // calculate S,dS, gamma_AO and dgamma_AO of the pair
                    pair.prepare_lcmo_gradient(&pair_atoms,m_i,m_j);

                    // MO coefficients of the virtual orbitals of I in 2D
                    let c_mat_virts:Array2<f64> = into_col(i.mo.c.to_owned())
                        .dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals of J in 2D
                    let c_mat_occs:Array2<f64> = into_col(j.mo.c.to_owned())
                        .dot(&into_row(l.mo.c.to_owned()));

                    // Check if the monomers of the CT have the same order as the pair
                    let (grad_i,grad_j) = if pair.i == i.monomer.index{
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true
                        )
                            .into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let exchange_gradient: Array1<f64> = f_exchange_ct_ct_ijij(
                            c_mat_occs.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true
                        ).into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let gradient:Array1<f64> = 2.0* exchange_gradient - coulomb_gradient;
                        let gradient_i = gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                        let gradient_j = gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                        (gradient_i,gradient_j)
                    }
                    else{
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false
                        )
                            .into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let exchange_gradient: Array1<f64> = f_exchange_ct_ct_ijij(
                            c_mat_occs.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false
                        ).into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let gradient:Array1<f64> = 2.0* exchange_gradient - coulomb_gradient;
                        let gradient_j = gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                        let gradient_i = gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                        (gradient_i,gradient_j)
                    };
                }
                // Only coulomb gradient for ESD
                else{
                    // get the index of the pair
                    let esd_pair_index:usize = self.properties.index_of_esd_pair(i.monomer.index,j.monomer.index);
                    // get the pair from pairs vector
                    let esd_pair: &mut ESDPair = &mut self.esd_pairs[esd_pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[esd_pair.i];
                    let m_j: &Monomer = &self.monomers[esd_pair.j];
                    let n_atoms:usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i:usize = m_i.n_orbs;
                    let orbs_j:usize = m_j.n_orbs;

                    // get pair atoms
                    let esd_pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    esd_pair.prepare_scc(&esd_pair_atoms,m_i,m_j);
                    esd_pair.run_scc(&esd_pair_atoms,self.config.scf);
                    esd_pair.prepare_lcmo_gradient(&esd_pair_atoms);

                    // MO coefficients of the virtual orbitals of I in 2D
                    let c_mat_virts:Array2<f64> = into_col(i.mo.c.to_owned())
                        .dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals of J in 2D
                    let c_mat_occs:Array2<f64> = into_col(j.mo.c.to_owned())
                        .dot(&into_row(l.mo.c.to_owned()));

                    // Check if the monomers of the CT have the same order as the pair
                    let (grad_i,grad_j) = if esd_pair.i == i.monomer.index{
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            esd_pair.properties.s().unwrap(),
                            esd_pair.properties.grad_s().unwrap(),
                            esd_pair.properties.gamma_ao().unwrap(),
                            esd_pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true
                        )
                            .into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let gradient_i = coulomb_gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                        let gradient_j = coulomb_gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                        (gradient_i,gradient_j)
                    }
                    else{
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            esd_pair.properties.s().unwrap(),
                            esd_pair.properties.grad_s().unwrap(),
                            esd_pair.properties.gamma_ao().unwrap(),
                            esd_pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false
                        )
                            .into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let gradient_j = coulomb_gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                        let gradient_i = coulomb_gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                        (gradient_i,gradient_j)
                    };
                }
            }
            // electron i on I, electron k on J, hole j on J, hole l on I
            CTCoupling::IJJI => {
                // Only real pair gradient, ESD gradient is zero
                if type_ij == PairType::Pair{
                    // get the index of the pair
                    let pair_index:usize = self.properties.index_of_pair(i.monomer.index,j.monomer.index);
                    // get the pair from pairs vector
                    let pair: &mut Pair = &mut self.pairs[pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];
                    let n_atoms:usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i:usize = m_i.n_orbs;
                    let orbs_j:usize = m_j.n_orbs;

                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    // calculate S,dS, gamma_AO and dgamma_AO of the pair
                    pair.prepare_lcmo_gradient(&pair_atoms,m_i,m_j);

                    // MO coefficients of the virtual orbitals:
                    // shape: orbs_i, orbs_j
                    let c_mat_virts:Array2<f64> = into_col(i.mo.c.to_owned())
                        .dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals:
                    // shape: orbs_j, orbs_i
                    let c_mat_occs:Array2<f64> = into_col(j.mo.c.to_owned())
                        .dot(&into_row(l.mo.c.to_owned()));

                    if m_j.index == j.idx{
                        // calculate the gradient of the coulomb integral
                        let coulomb_gradient:Array1<f64> = f_coul_ct_ct_ijji(
                            c_mat_virts.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        ).into_shape([3 * n_atoms, orbs_j * orbs_i])
                            .unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_i]).unwrap());

                        // calculate the gradient of the exchange integral
                        let exchange_gradient:Array1<f64> = f_exchange_ct_ct_ijji(
                            c_mat_virts.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        ).into_shape([3 * n_atoms, orbs_j * orbs_i])
                            .unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_i]).unwrap());

                        let gradient:Array1<f64> = 2.0* exchange_gradient - coulomb_gradient;
                    }
                    else{
                        // calculate the gradient of the coulomb integral
                        // c_mats must be transposed if the monomer index J is the CT index I
                        let coulomb_gradient:Array1<f64> = f_coul_ct_ct_ijji(
                            c_mat_virts.t(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        ).into_shape([3 * n_atoms, orbs_j * orbs_i])
                            .unwrap()
                            .dot(&c_mat_occs.t().into_shape([orbs_j * orbs_i]).unwrap());

                        // calculate the gradient of the exchange integral
                        let exchange_gradient:Array1<f64> = f_exchange_ct_ct_ijji(
                            c_mat_virts.t(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        ).into_shape([3 * n_atoms, orbs_j * orbs_i])
                            .unwrap()
                            .dot(&c_mat_occs.t().into_shape([orbs_j * orbs_i]).unwrap());

                        let gradient:Array1<f64> = 2.0* exchange_gradient - coulomb_gradient;
                    }
                }
            }
            // electrons i,k on I, hole j on J, hole l on K
            CTCoupling::IJIK => {

            }
            // electron i on I, electron k on J, hole j on J, hole l on K
            CTCoupling::IJJK => {

            }
            // electron i on I, electron k on K, hole j on J, hole l on I
            CTCoupling::IJKI => {

            }
            // electron i on I, electron k on K, hole j on J, hole l on J
            CTCoupling::IJKJ => {

            }
            // electron i on I, electron k on K, hole j on J, hole l on L
            CTCoupling::IJKL => {

            }
        }
    }
}