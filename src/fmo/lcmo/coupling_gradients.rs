use crate::fmo::helpers::get_pair_slice;
use crate::fmo::lcmo::cis_gradient::{ReducedBasisState, ReducedCT, ReducedLE, ReducedParticle};
use crate::fmo::lcmo::helpers::*;
use crate::fmo::lcmo::integrals::CTCoupling;
use crate::fmo::{
    BasisState, ChargeTransfer, LocallyExcited, Monomer, PairType, Particle, SuperSystem, LRC,
};
use crate::fmo::{ESDPair, Pair};
use crate::initialization::Atom;
use ndarray::prelude::*;
use ndarray_linalg::{into_col, into_row};

impl SuperSystem {
    pub fn exciton_coupling_gradient<'a>(
        &mut self,
        lhs: &'a BasisState<'a>,
        rhs: &'a BasisState<'a>,
    ) -> Array1<f64> {
        match (lhs, rhs) {
            // Coupling between two LE states.
            (BasisState::LE(ref a), BasisState::LE(ref b)) => self.le_le_coupling_grad(a, b),
            // Coupling between LE and CT state.
            (BasisState::LE(ref a), BasisState::CT(ref b)) => self.le_ct_coupling_grad(a, b),
            // Coupling between CT and LE state.
            (BasisState::CT(ref a), BasisState::LE(ref b)) => self.ct_le_coupling_grad(a, b),
            // Coupling between CT and CT
            (BasisState::CT(ref a), BasisState::CT(ref b)) => self.ct_ct_coupling_grad(a, b),
        }
    }

    pub fn exciton_coupling_gradient_new(
        &mut self,
        lhs: &ReducedBasisState,
        rhs: &ReducedBasisState,
    ) -> Array1<f64> {
        match (lhs, rhs) {
            // Coupling between two LE states.
            (ReducedBasisState::LE(ref a), ReducedBasisState::LE(ref b)) => {
                self.le_le_coupling_grad_new(a, b)
            }
            // Coupling between LE and CT state.
            (ReducedBasisState::LE(ref a), ReducedBasisState::CT(ref b)) => {
                self.le_ct_coupling_grad_new(a, b)
            }
            // Coupling between CT and LE state.
            (ReducedBasisState::CT(ref a), ReducedBasisState::LE(ref b)) => {
                self.ct_le_coupling_grad_new(a, b)
            }
            // Coupling between CT and CT
            (ReducedBasisState::CT(ref a), ReducedBasisState::CT(ref b)) => {
                self.ct_ct_coupling_grad_new(a, b)
            }
        }
    }

    pub fn le_le_coupling_grad_new(&mut self, i: &ReducedLE, j: &ReducedLE) -> Array1<f64> {
        // Check if the ESD approximation is used or not.
        let type_pair: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.monomer_index);

        let mol_i: &Monomer = &self.monomers[i.monomer_index];
        let mol_j: &Monomer = &self.monomers[j.monomer_index];

        // transform the CI coefficients of the monomers to the AO basis
        let nocc_i = mol_i.properties.n_occ().unwrap();
        let nvirt_i = mol_i.properties.n_virt().unwrap();
        let cis_c_i: ArrayView2<f64> = mol_i
            .properties
            .ci_coefficient(i.state_index)
            .unwrap()
            .into_shape([nocc_i, nvirt_i])
            .unwrap();
        let occs_i = mol_i.properties.orbs_slice(0, Some(i.homo + 1)).unwrap();
        let virts_i = mol_i.properties.orbs_slice(i.homo + 1, None).unwrap();

        let nocc_j = mol_j.properties.n_occ().unwrap();
        let nvirt_j = mol_j.properties.n_virt().unwrap();
        let cis_c_j: ArrayView2<f64> = mol_j
            .properties
            .ci_coefficient(j.state_index)
            .unwrap()
            .into_shape([nocc_j, nvirt_j])
            .unwrap();
        let occs_j = mol_j.properties.orbs_slice(0, Some(j.homo + 1)).unwrap();
        let virts_j = mol_j.properties.orbs_slice(j.homo + 1, None).unwrap();

        let mut tdm_i:Array2<f64>;
        let mut tdm_j:Array2<f64>;
        if i.monomer_index < j.monomer_index{
            tdm_i = occs_i.dot(&cis_c_i.dot(&virts_i.t()));
            tdm_j = occs_j.dot(&cis_c_j.dot(&virts_j.t()));
        }
        else{
            tdm_j = occs_i.dot(&cis_c_i.dot(&virts_i.t()));
            tdm_i = occs_j.dot(&cis_c_j.dot(&virts_j.t()));
        }

        let n_atoms: usize = mol_i.n_atoms + mol_j.n_atoms;
        let mut gradient: Array1<f64> = Array1::zeros(3 * n_atoms);

        if type_pair == PairType::Pair {
            // calculate the coulomb and exchange contribution of the gradient
            // calculate F[tdm_j] and F_lr[tdm_j]

            // get the index of the pair
            let pair_index: usize = self
                .properties
                .index_of_pair(i.monomer_index, j.monomer_index);
            // get the pair from pairs vector
            let pair: &mut Pair = &mut self.pairs[pair_index];
            // monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];
            let n_orbs_i: usize = m_i.n_orbs;
            let n_orbs_j: usize = m_j.n_orbs;

            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );
            // calculate S,dS, gamma_AO and dgamma_AO of the pair
            pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
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

            let coulomb_integral:Array5<f64> = f_coulomb_loop(
                pair.properties.s().unwrap(),
                pair.properties.grad_s().unwrap(),
                pair.properties.gamma_ao().unwrap(),
                pair.properties.grad_gamma_ao().unwrap(),
                pair.n_atoms,
                n_orbs_i,
                n_orbs_j
            );
            let coulomb_grad:Array1<f64> = coulomb_integral.view()
                .into_shape([3*n_atoms*n_orbs_i*n_orbs_i,n_orbs_j*n_orbs_j]).unwrap()
                .dot(&tdm_j.view().into_shape([n_orbs_j*n_orbs_j]).unwrap())
                .into_shape([3*n_atoms,n_orbs_i*n_orbs_i]).unwrap()
                .dot(&tdm_i.view().into_shape([n_orbs_i*n_orbs_i]).unwrap());

            println!("coulomb gradient: {}",coulomb_gradient.slice(s![0..10]));
            println!("coulomb grad loop: {}",coulomb_grad.slice(s![0..10]));
            assert!(coulomb_gradient.abs_diff_eq(&coulomb_grad,1e-14),"LE-LE coulomb gradient is wrong!");

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

            let exchange_integral:Array5<f64> = f_exchange_loop(
                pair.properties.s().unwrap(),
                pair.properties.grad_s().unwrap(),
                pair.properties.gamma_lr_ao().unwrap(),
                pair.properties.grad_gamma_lr_ao().unwrap(),
                pair.n_atoms,
                n_orbs_i,
                n_orbs_j
            );
            let exchange_grad:Array1<f64> = exchange_integral.view()
                .into_shape([3*n_atoms*n_orbs_i*n_orbs_i,n_orbs_j*n_orbs_j]).unwrap()
                .dot(&tdm_j.view().into_shape([n_orbs_j*n_orbs_j]).unwrap())
                .into_shape([3*n_atoms,n_orbs_i*n_orbs_i]).unwrap()
                .dot(&tdm_i.view().into_shape([n_orbs_i*n_orbs_i]).unwrap());

            println!("exchange gradient: {}",exchange_gradient.slice(s![0..10]));
            println!("exchange grad loop: {}",exchange_grad.slice(s![0..10]));
            assert!(exchange_gradient.abs_diff_eq(&exchange_grad,1e-14),"LE-LE exchange gradient is wrong!");

            gradient = 2.0 * coulomb_gradient - exchange_gradient;
            pair.properties.reset_gradient();
        } else {
            // calculate only the coulomb contribution of the gradient
            // calculate F[tdm_j]
            let pair_index: usize = self
                .properties
                .index_of_esd_pair(i.monomer_index, j.monomer_index);
            // get correct pair from pairs vector
            let esd_pair: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // monomers
            let m_i: &Monomer = &self.monomers[esd_pair.i];
            let m_j: &Monomer = &self.monomers[esd_pair.j];
            let n_orbs_i: usize = m_i.n_orbs;
            let n_orbs_j: usize = m_j.n_orbs;

            // get pair atoms
            let esd_pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );
            esd_pair.prepare_scc(&esd_pair_atoms, m_i, m_j);
            esd_pair.run_scc(&esd_pair_atoms, self.config.scf);
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
        return gradient;
    }

    pub fn le_ct_coupling_grad_new(&mut self, i: &ReducedLE, j: &ReducedCT) -> Array1<f64> {
        // self.le_ct_1e_coupling_grad(i,j);
        self.le_ct_2e_coupling_grad_new(i, j)
    }

    pub fn ct_le_coupling_grad_new(&mut self, i: &ReducedCT, j: &ReducedLE) -> Array1<f64> {
        self.le_ct_coupling_grad_new(j, i)
    }

    pub fn le_ct_1e_coupling_grad_new(&mut self, i: &ReducedLE, j: &ReducedCT) {
        // Neglect the contribution? Depends on the derivative of the Fock matrix
        // -> CPHF necessary
    }

    pub fn le_ct_2e_coupling_grad_new(&mut self, i: &ReducedLE, j: &ReducedCT) -> Array1<f64> {
        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.hole.m_index);
        // The same for I and K
        let type_ik: PairType = self
            .properties
            .type_of_pair(i.monomer_index, j.electron.m_index);
        // and J K
        let type_jk: PairType = self
            .properties
            .type_of_pair(j.electron.m_index, j.hole.m_index);

        let mol_i: &Monomer = &self.monomers[i.monomer_index];
        let nocc = mol_i.properties.n_occ().unwrap();
        let nvirt = mol_i.properties.n_virt().unwrap();
        let cis_c: ArrayView2<f64> = mol_i
            .properties
            .ci_coefficient(i.state_index)
            .unwrap()
            .into_shape([nocc, nvirt])
            .unwrap();
        let occs = mol_i.properties.orbs_slice(0, Some(i.homo + 1)).unwrap();
        let virts = mol_i.properties.orbs_slice(i.homo + 1, None).unwrap();
        let tdm: Array2<f64> = occs.dot(&cis_c.dot(&virts.t()));

        // initialize return value
        let mut return_gradient: Array1<f64> = Array1::zeros(mol_i.n_atoms);

        // < LE I | H | CT J_j -> I_b>
        if i.monomer_index == j.electron.m_index {
            // Check if the pair IK is close, so that the overlap is non-zero.
            if type_ik == PairType::Pair {
                // get the index of the pair
                let pair_index: usize = self
                    .properties
                    .index_of_pair(i.monomer_index, j.hole.m_index);
                // get the pair from pairs vector
                let pair: &mut Pair = &mut self.pairs[pair_index];
                // monomers
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];
                let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                let n_orbs_i: usize = m_i.n_orbs;
                let n_orbs_j: usize = m_j.n_orbs;

                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    m_i.slice.atom_as_range(),
                    m_j.slice.atom_as_range(),
                );
                // calculate S,dS, gamma_AO and dgamma_AO of the pair
                pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                let grad = if pair.i == i.monomer_index {
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms I + 3*n_atoms J]
                    let coulomb_gradient: Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient: Array1<f64> = f_lr_le_ct_exchange_hole_j(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    let gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;

                    gradient
                } else {
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms J + 3*n_atoms I]
                    let coulomb_gradient: Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient: Array1<f64> = f_lr_le_ct_exchange_hole_j(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    let gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;

                    gradient
                };

                pair.properties.reset_gradient();
                return_gradient = grad;
            }
        } else if i.monomer_index == j.hole.m_index {
            // Check if the pair IJ is close, so that the overlap is non-zero.
            if type_ij == PairType::Pair {
                // get the index of the pair
                let pair_index: usize = self
                    .properties
                    .index_of_pair(i.monomer_index, j.electron.m_index);
                // get the pair from pairs vector
                let pair: &mut Pair = &mut self.pairs[pair_index];
                // monomers
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];
                let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                let n_orbs_i: usize = m_i.n_orbs;
                let n_orbs_j: usize = m_j.n_orbs;

                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    m_i.slice.atom_as_range(),
                    m_j.slice.atom_as_range(),
                );
                // calculate S,dS, gamma_AO and dgamma_AO of the pair
                pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                let grad = if pair.i == i.monomer_index {
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms I + 3*n_atoms J]
                    let coulomb_gradient: Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.electron.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.hole.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient: Array1<f64> = f_lr_le_ct_exchange_hole_i(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.electron.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.hole.mo.c);

                    let gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;

                    gradient
                } else {
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms J + 3*n_atoms I]
                    let coulomb_gradient: Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    )
                    .into_shape([3 * n_atoms * n_orbs_j, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient: Array1<f64> = f_lr_le_ct_exchange_hole_i(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    )
                    .into_shape([3 * n_atoms * n_orbs_j, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c);

                    let gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;

                    gradient
                };

                pair.properties.reset_gradient();
                return_gradient = grad;
            }
        }
        return return_gradient;

        // // < LE I_ia | H | CT K_j -> J_b >
        // else{
        //     // neglect this contribution
        //
        //     // // The integral (ia|jb) requires that the overlap between K and J is non-zero.
        //     // // coulomb
        //     // if type_jk == PairType::Pair {
        //     //
        //     // }
        //     // // exchange
        //     // if type_ik == PairType::Pair && type_ij == PairType::Pair {
        //     //
        //     // }
        // }
    }

    pub fn ct_ct_coupling_grad_new(
        &mut self,
        state_1: &ReducedCT,
        state_2: &ReducedCT,
    ) -> Array1<f64> {
        // i -> electron on I, j -> hole on J.
        let (i, j): (&ReducedParticle, &ReducedParticle) = (&state_1.electron, &state_1.hole);
        // k -> electron on K, l -> hole on L.
        let (k, l): (&ReducedParticle, &ReducedParticle) = (&state_2.electron, &state_2.hole);

        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.m_index, j.m_index);
        // I and K
        let type_ik: PairType = self.properties.type_of_pair(i.m_index, k.m_index);
        // I and L
        let type_il: PairType = self.properties.type_of_pair(i.m_index, l.m_index);
        // J and K
        let type_jk: PairType = self.properties.type_of_pair(j.m_index, k.m_index);
        // J and L
        let type_jl: PairType = self.properties.type_of_pair(j.m_index, l.m_index);
        // K and L
        let type_kl: PairType = self.properties.type_of_pair(k.m_index, l.m_index);

        // Check how many monomers are involved in this matrix element.
        let kind: CTCoupling = CTCoupling::from((i, j, k, l));

        // initialize return value
        let mut return_gradient: Array1<f64> = Array1::zeros(1);

        match kind {
            // electrons i,k on I, holes j,l on J
            CTCoupling::IJIJ => {
                if type_ij == PairType::Pair {
                    // get the index of the pair
                    let pair_index: usize = self.properties.index_of_pair(i.m_index, j.m_index);
                    // get the pair from pairs vector
                    let pair: &mut Pair = &mut self.pairs[pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];
                    let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i: usize = m_i.n_orbs;
                    let orbs_j: usize = m_j.n_orbs;

                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    // calculate S,dS, gamma_AO and dgamma_AO of the pair
                    pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                    // MO coefficients of the virtual orbitals of I in 2D
                    let c_mat_virts: Array2<f64> =
                        into_col(i.mo.c.to_owned()).dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals of J in 2D
                    let c_mat_occs: Array2<f64> =
                        into_col(j.mo.c.to_owned()).dot(&into_row(l.mo.c.to_owned()));

                    // Check if the monomers of the CT have the same order as the pair
                    let grad = if pair.i == i.m_index {
                        println!("Index I is pair index I");
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_virts.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_j])
                        .unwrap()
                        .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_j]).unwrap());

                        let coulomb_integral:Array5<f64> = f_coulomb_loop(
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                        );
                        let coulomb_grad:Array1<f64> = coulomb_integral.into_shape([3*n_atoms*orbs_i*orbs_i,orbs_j*orbs_j]).unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_j]).unwrap()).into_shape([3*n_atoms,orbs_i*orbs_i]).unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        println!("coulomb gradient: {}",coulomb_gradient.slice(s![0..10]));
                        println!("coulomb grad loop: {}",coulomb_grad.slice(s![0..10]));
                        assert!(coulomb_gradient.abs_diff_eq(&coulomb_grad,1e-14),"Coulomb gradients are NOT equal!");

                        let exchange_gradient: Array1<f64> = f_exchange_ct_ct_ijij(
                            c_mat_virts.t(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_j])
                        .unwrap()
                        .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_j]).unwrap());

                        let exchange_integral:Array5<f64> = f_exchange_loop(
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                        );
                        let exchange_grad:Array1<f64> = exchange_integral.into_shape([3*n_atoms*orbs_i*orbs_i,orbs_j*orbs_j]).unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_j]).unwrap()).into_shape([3*n_atoms,orbs_i*orbs_i]).unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());
                        assert!(exchange_gradient.abs_diff_eq(&exchange_grad,1e-14),"Coulomb gradients are NOT equal!");

                        // add the coulomb and exchange gradient
                        let gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;

                        gradient
                    } else {
                        println!("Index J is pair index I");
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_virts.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_i * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let coulomb_integral:Array5<f64> = f_coulomb_loop(
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                        );
                        let coulomb_grad:Array1<f64> = coulomb_integral.into_shape([3*n_atoms*orbs_i*orbs_i,orbs_j*orbs_j]).unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_j * orbs_j]).unwrap()).into_shape([3*n_atoms,orbs_i*orbs_i]).unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_i * orbs_i]).unwrap());

                        println!("coulomb gradient: {}",coulomb_gradient.slice(s![0..10]));
                        println!("coulomb grad loop: {}",coulomb_grad.slice(s![0..10]));
                        assert!(coulomb_gradient.abs_diff_eq(&coulomb_grad,1e-14),"Coulomb gradients are NOT equal!");

                        let exchange_gradient: Array1<f64> = f_exchange_ct_ct_ijij(
                            c_mat_virts.t(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_i * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let exchange_integral:Array5<f64> = f_exchange_loop(
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                        );
                        let exchange_grad:Array1<f64> = exchange_integral.into_shape([3*n_atoms*orbs_i*orbs_i,orbs_j*orbs_j]).unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_j * orbs_j]).unwrap()).into_shape([3*n_atoms,orbs_i*orbs_i]).unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_i * orbs_i]).unwrap());
                        assert!(exchange_gradient.abs_diff_eq(&exchange_grad,1e-14),"Coulomb gradients are NOT equal!");

                        // add the coulomb and exchange gradient
                        let gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;

                        gradient
                    };
                    return_gradient = grad;
                }
                // Only coulomb gradient for ESD
                else {
                    // get the index of the pair
                    let esd_pair_index: usize =
                        self.properties.index_of_esd_pair(i.m_index, j.m_index);
                    // get the pair from pairs vector
                    let esd_pair: &mut ESDPair = &mut self.esd_pairs[esd_pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[esd_pair.i];
                    let m_j: &Monomer = &self.monomers[esd_pair.j];
                    let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i: usize = m_i.n_orbs;
                    let orbs_j: usize = m_j.n_orbs;

                    // get pair atoms
                    let esd_pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    esd_pair.prepare_scc(&esd_pair_atoms, m_i, m_j);
                    esd_pair.run_scc(&esd_pair_atoms, self.config.scf);
                    esd_pair.prepare_lcmo_gradient(&esd_pair_atoms);

                    // MO coefficients of the virtual orbitals of I in 2D
                    let c_mat_virts: Array2<f64> =
                        into_col(i.mo.c.to_owned()).dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals of J in 2D
                    let c_mat_occs: Array2<f64> =
                        into_col(j.mo.c.to_owned()).dot(&into_row(l.mo.c.to_owned()));

                    // Check if the monomers of the CT have the same order as the pair
                    let grad = if esd_pair.i == i.m_index {
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            esd_pair.properties.s().unwrap(),
                            esd_pair.properties.grad_s().unwrap(),
                            esd_pair.properties.gamma_ao().unwrap(),
                            esd_pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_i * orbs_i])
                        .unwrap()
                        .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        // let gradient_i = coulomb_gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                        // let gradient_j = coulomb_gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                        coulomb_gradient
                    } else {
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            esd_pair.properties.s().unwrap(),
                            esd_pair.properties.grad_s().unwrap(),
                            esd_pair.properties.gamma_ao().unwrap(),
                            esd_pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_i * orbs_i])
                        .unwrap()
                        .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        // let gradient_j = coulomb_gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                        // let gradient_i = coulomb_gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                        coulomb_gradient
                    };
                    return_gradient = grad;
                }
            }
            // electron i on I, electron k on J, hole j on J, hole l on I
            CTCoupling::IJJI => {
                // Only real pair gradient, ESD gradient is zero
                if type_ij == PairType::Pair {
                    // get the index of the pair
                    let pair_index: usize = self.properties.index_of_pair(i.m_index, j.m_index);
                    // get the pair from pairs vector
                    let pair: &mut Pair = &mut self.pairs[pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];
                    let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i: usize = m_i.n_orbs;
                    let orbs_j: usize = m_j.n_orbs;

                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    // calculate S,dS, gamma_AO and dgamma_AO of the pair
                    pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                    // MO coefficients of the virtual orbitals:
                    // shape: orbs_i, orbs_j
                    let c_mat_virts: Array2<f64> =
                        into_col(i.mo.c.to_owned()).dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals:
                    // shape: orbs_j, orbs_i
                    let c_mat_occs: Array2<f64> =
                        into_col(j.mo.c.to_owned()).dot(&into_row(l.mo.c.to_owned()));

                    let grad = if m_j.index == j.m_index {
                        // calculate the gradient of the coulomb integral
                        let coulomb_gradient: Array1<f64> = f_coul_ct_ct_ijji(
                            c_mat_virts.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_i]).unwrap());
                        // calculate the gradient of the exchange integral
                        let exchange_gradient: Array1<f64> = f_exchange_ct_ct_ijji(
                            c_mat_virts.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_i]).unwrap());

                        let gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;
                        gradient
                    } else {
                        // calculate the gradient of the coulomb integral
                        // c_mats must be transposed if the monomer index J is the CT index I
                        let coulomb_gradient: Array1<f64> = f_coul_ct_ct_ijji(
                            c_mat_virts.t(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.t().into_shape([orbs_j * orbs_i]).unwrap());

                        // calculate the gradient of the exchange integral
                        let exchange_gradient: Array1<f64> = f_exchange_ct_ct_ijji(
                            c_mat_virts.t(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.t().into_shape([orbs_j * orbs_i]).unwrap());

                        let gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;
                        gradient
                    };
                    return_gradient = grad;
                }
            }
            // electrons i,k on I, hole j on J, hole l on K
            CTCoupling::IJIK => {
                // neglect this type of coupling
            }
            // electron i on I, electron k on J, hole j on J, hole l on K
            CTCoupling::IJJK => {
                // neglect this type of coupling
            }
            // electron i on I, electron k on K, hole j on J, hole l on I
            CTCoupling::IJKI => {
                // neglect this type of coupling
            }
            // electron i on I, electron k on K, hole j on J, hole l on J
            CTCoupling::IJKJ => {
                // neglect this type of coupling
            }
            // electron i on I, electron k on K, hole j on J, hole l on L
            CTCoupling::IJKL => {
                // neglect this type of coupling
            }
        }
        return return_gradient;
    }

    pub fn le_le_coupling_grad<'a>(
        &mut self,
        i: &'a LocallyExcited<'a>,
        j: &'a LocallyExcited<'a>,
    ) -> Array1<f64> {
        // Check if the ESD approximation is used or not.
        let type_pair: PairType = self
            .properties
            .type_of_pair(i.monomer.index, j.monomer.index);

        // transform the CI coefficients of the monomers to the AO basis
        let nocc_i = i.monomer.properties.n_occ().unwrap();
        let nvirt_i = i.monomer.properties.n_virt().unwrap();
        let cis_c_i = i.tdm.into_shape([nocc_i, nvirt_i]).unwrap();
        let tdm_i: Array2<f64> = i.occs.dot(&cis_c_i.dot(&i.virts.t()));
        let n_orbs_i: usize = i.monomer.n_orbs;

        let nocc_j = j.monomer.properties.n_occ().unwrap();
        let nvirt_j = j.monomer.properties.n_virt().unwrap();
        let cis_c_j = j.tdm.into_shape([nocc_j, nvirt_j]).unwrap();
        let tdm_j: Array2<f64> = j.occs.dot(&cis_c_j.dot(&j.virts.t()));

        let n_atoms: usize = i.monomer.n_atoms + j.monomer.n_atoms;
        let mut gradient: Array1<f64> = Array1::zeros(3 * n_atoms);

        if type_pair == PairType::Pair {
            // calculate the coulomb and exchange contribution of the gradient
            // calculate F[tdm_j] and F_lr[tdm_j]

            // get the index of the pair
            let pair_index: usize = self
                .properties
                .index_of_pair(i.monomer.index, j.monomer.index);
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
            pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
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
            pair.properties.reset_gradient();
        } else {
            // calculate only the coulomb contribution of the gradient
            // calculate F[tdm_j]
            let pair_index: usize = self
                .properties
                .index_of_esd_pair(i.monomer.index, j.monomer.index);
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
            esd_pair.prepare_scc(&esd_pair_atoms, m_i, m_j);
            esd_pair.run_scc(&esd_pair_atoms, self.config.scf);
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
        return gradient;
    }

    pub fn le_ct_coupling_grad<'a>(
        &mut self,
        i: &'a LocallyExcited<'a>,
        j: &'a ChargeTransfer<'a>,
    ) -> Array1<f64> {
        // self.le_ct_1e_coupling_grad(i,j);
        self.le_ct_2e_coupling_grad(i, j)
    }

    pub fn ct_le_coupling_grad<'a>(
        &mut self,
        i: &'a ChargeTransfer<'a>,
        j: &'a LocallyExcited<'a>,
    ) -> Array1<f64> {
        self.le_ct_coupling_grad(j, i)
    }

    pub fn le_ct_1e_coupling_grad<'a>(
        &mut self,
        i: &'a LocallyExcited<'a>,
        j: &'a ChargeTransfer<'a>,
    ) {
        // Neglect the contribution? Depends on the derivative of the Fock matrix
        // -> CPHF necessary
    }

    pub fn le_ct_2e_coupling_grad<'a>(
        &mut self,
        i: &'a LocallyExcited<'a>,
        j: &'a ChargeTransfer<'a>,
    ) -> Array1<f64> {
        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.monomer.index, j.hole.idx);
        // The same for I and K
        let type_ik: PairType = self
            .properties
            .type_of_pair(i.monomer.index, j.electron.idx);
        // and J K
        let type_jk: PairType = self.properties.type_of_pair(j.electron.idx, j.hole.idx);

        // transform the CI coefficients of the monomers to the AO basis
        let nocc = i.monomer.properties.n_occ().unwrap();
        let nvirt = i.monomer.properties.n_virt().unwrap();
        let cis_c = i.tdm.into_shape([nocc, nvirt]).unwrap();
        let tdm: Array2<f64> = i.occs.dot(&cis_c.dot(&i.virts.t()));
        let n_orbs_i: usize = i.monomer.n_orbs;

        // initialize return value
        let mut return_gradient: Array1<f64> = Array1::zeros(i.monomer.n_atoms);

        // < LE I | H | CT J_j -> I_b>
        if i.monomer.index == j.electron.idx {
            // Check if the pair IK is close, so that the overlap is non-zero.
            if type_ik == PairType::Pair {
                // get the index of the pair
                let pair_index: usize = self.properties.index_of_pair(i.monomer.index, j.hole.idx);
                // get the pair from pairs vector
                let pair: &mut Pair = &mut self.pairs[pair_index];
                // monomers
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];
                let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                let n_orbs_j: usize = m_j.n_orbs;

                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    m_i.slice.atom_as_range(),
                    m_j.slice.atom_as_range(),
                );
                // calculate S,dS, gamma_AO and dgamma_AO of the pair
                pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                let grad = if pair.i == i.monomer.index {
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms I + 3*n_atoms J]
                    let coulomb_gradient: Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient: Array1<f64> = f_lr_le_ct_exchange_hole_j(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    let gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;
                    // let gradient_i = gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                    // let gradient_j = gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                    gradient
                } else {
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms J + 3*n_atoms I]
                    let coulomb_gradient: Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient: Array1<f64> = f_lr_le_ct_exchange_hole_j(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    let gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;
                    // let gradient_j = gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                    // let gradient_i = gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                    gradient
                };

                pair.properties.reset_gradient();
                return_gradient = grad;
            }
        } else if i.monomer.index == j.hole.idx {
            // Check if the pair IJ is close, so that the overlap is non-zero.
            if type_ij == PairType::Pair {
                // get the index of the pair
                let pair_index: usize = self
                    .properties
                    .index_of_pair(i.monomer.index, j.electron.idx);
                // get the pair from pairs vector
                let pair: &mut Pair = &mut self.pairs[pair_index];
                // monomers
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];
                let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                let n_orbs_j: usize = m_j.n_orbs;

                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    m_i.slice.atom_as_range(),
                    m_j.slice.atom_as_range(),
                );
                // calculate S,dS, gamma_AO and dgamma_AO of the pair
                pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                let grad = if pair.i == i.monomer.index {
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms I + 3*n_atoms J]
                    let coulomb_gradient: Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.electron.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.hole.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient: Array1<f64> = f_lr_le_ct_exchange_hole_i(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.electron.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.hole.mo.c);

                    let gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;
                    // let gradient_i = gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                    // let gradient_j = gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                    gradient
                } else {
                    // calculate the gradient of the coulomb integral
                    // the order of the gradient is [3*n_atoms J + 3*n_atoms I]
                    let coulomb_gradient: Array1<f64> = f_le_ct_coulomb(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.electron.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.hole.mo.c);

                    // calculate the gradient of the exchange integral
                    let exchange_gradient: Array1<f64> = f_lr_le_ct_exchange_hole_i(
                        tdm.view(),
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    )
                    .into_shape([3 * n_atoms * n_orbs_i, n_orbs_j])
                    .unwrap()
                    .dot(&j.electron.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.hole.mo.c);

                    let gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;
                    // let gradient_j = gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                    // let gradient_i = gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                    gradient
                };

                pair.properties.reset_gradient();
                return_gradient = grad;
            }
        }
        return return_gradient;

        // // < LE I_ia | H | CT K_j -> J_b >
        // else{
        //     // neglect this contribution
        //
        //     // // The integral (ia|jb) requires that the overlap between K and J is non-zero.
        //     // // coulomb
        //     // if type_jk == PairType::Pair {
        //     //
        //     // }
        //     // // exchange
        //     // if type_ik == PairType::Pair && type_ij == PairType::Pair {
        //     //
        //     // }
        // }
    }

    pub fn ct_ct_coupling_grad<'a>(
        &mut self,
        state_1: &'a ChargeTransfer<'a>,
        state_2: &'a ChargeTransfer<'a>,
    ) -> Array1<f64> {
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

        // initialize return value
        let mut return_gradient: Array1<f64> = Array1::zeros(i.monomer.n_atoms);

        match kind {
            // electrons i,k on I, holes j,l on J
            CTCoupling::IJIJ => {
                if type_ij == PairType::Pair {
                    // get the index of the pair
                    let pair_index: usize = self
                        .properties
                        .index_of_pair(i.monomer.index, j.monomer.index);
                    // get the pair from pairs vector
                    let pair: &mut Pair = &mut self.pairs[pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];
                    let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i: usize = m_i.n_orbs;
                    let orbs_j: usize = m_j.n_orbs;

                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    // calculate S,dS, gamma_AO and dgamma_AO of the pair
                    pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                    // MO coefficients of the virtual orbitals of I in 2D
                    let c_mat_virts: Array2<f64> =
                        into_col(i.mo.c.to_owned()).dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals of J in 2D
                    let c_mat_occs: Array2<f64> =
                        into_col(j.mo.c.to_owned()).dot(&into_row(l.mo.c.to_owned()));

                    // Check if the monomers of the CT have the same order as the pair
                    let grad = if pair.i == i.monomer.index {
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
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
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_i * orbs_i])
                        .unwrap()
                        .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;
                        // let gradient_i = gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                        // let gradient_j = gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                        gradient
                    } else {
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
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
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_i * orbs_i])
                        .unwrap()
                        .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        let gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;
                        // let gradient_j = gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                        // let gradient_i = gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                        gradient
                    };
                    return_gradient = grad;
                }
                // Only coulomb gradient for ESD
                else {
                    // get the index of the pair
                    let esd_pair_index: usize = self
                        .properties
                        .index_of_esd_pair(i.monomer.index, j.monomer.index);
                    // get the pair from pairs vector
                    let esd_pair: &mut ESDPair = &mut self.esd_pairs[esd_pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[esd_pair.i];
                    let m_j: &Monomer = &self.monomers[esd_pair.j];
                    let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i: usize = m_i.n_orbs;
                    let orbs_j: usize = m_j.n_orbs;

                    // get pair atoms
                    let esd_pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    esd_pair.prepare_scc(&esd_pair_atoms, m_i, m_j);
                    esd_pair.run_scc(&esd_pair_atoms, self.config.scf);
                    esd_pair.prepare_lcmo_gradient(&esd_pair_atoms);

                    // MO coefficients of the virtual orbitals of I in 2D
                    let c_mat_virts: Array2<f64> =
                        into_col(i.mo.c.to_owned()).dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals of J in 2D
                    let c_mat_occs: Array2<f64> =
                        into_col(j.mo.c.to_owned()).dot(&into_row(l.mo.c.to_owned()));

                    // Check if the monomers of the CT have the same order as the pair
                    let grad = if esd_pair.i == i.monomer.index {
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            esd_pair.properties.s().unwrap(),
                            esd_pair.properties.grad_s().unwrap(),
                            esd_pair.properties.gamma_ao().unwrap(),
                            esd_pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_i * orbs_i])
                        .unwrap()
                        .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        // let gradient_i = coulomb_gradient.slice(s![..3*m_i.n_atoms]).to_owned();
                        // let gradient_j = coulomb_gradient.slice(s![3*m_i.n_atoms..]).to_owned();

                        coulomb_gradient
                    } else {
                        let coulomb_gradient: Array1<f64> = f_coulomb_ct_ct_ijij(
                            c_mat_occs.view(),
                            esd_pair.properties.s().unwrap(),
                            esd_pair.properties.grad_s().unwrap(),
                            esd_pair.properties.gamma_ao().unwrap(),
                            esd_pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_i * orbs_i])
                        .unwrap()
                        .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        // let gradient_j = coulomb_gradient.slice(s![..3*m_j.n_atoms]).to_owned();
                        // let gradient_i = coulomb_gradient.slice(s![3*m_j.n_atoms..]).to_owned();

                        coulomb_gradient
                    };
                    return_gradient = grad;
                }
            }
            // electron i on I, electron k on J, hole j on J, hole l on I
            CTCoupling::IJJI => {
                // Only real pair gradient, ESD gradient is zero
                if type_ij == PairType::Pair {
                    // get the index of the pair
                    let pair_index: usize = self
                        .properties
                        .index_of_pair(i.monomer.index, j.monomer.index);
                    // get the pair from pairs vector
                    let pair: &mut Pair = &mut self.pairs[pair_index];
                    // monomers
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];
                    let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i: usize = m_i.n_orbs;
                    let orbs_j: usize = m_j.n_orbs;

                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        m_i.slice.atom_as_range(),
                        m_j.slice.atom_as_range(),
                    );
                    // calculate S,dS, gamma_AO and dgamma_AO of the pair
                    pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                    // MO coefficients of the virtual orbitals:
                    // shape: orbs_i, orbs_j
                    let c_mat_virts: Array2<f64> =
                        into_col(i.mo.c.to_owned()).dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals:
                    // shape: orbs_j, orbs_i
                    let c_mat_occs: Array2<f64> =
                        into_col(j.mo.c.to_owned()).dot(&into_row(l.mo.c.to_owned()));

                    let grad = if m_j.index == j.idx {
                        // calculate the gradient of the coulomb integral
                        let coulomb_gradient: Array1<f64> = f_coul_ct_ct_ijji(
                            c_mat_virts.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_i]).unwrap());

                        // calculate the gradient of the exchange integral
                        let exchange_gradient: Array1<f64> = f_exchange_ct_ct_ijji(
                            c_mat_virts.view(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            true,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_i]).unwrap());

                        let gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;
                        gradient
                    } else {
                        // calculate the gradient of the coulomb integral
                        // c_mats must be transposed if the monomer index J is the CT index I
                        let coulomb_gradient: Array1<f64> = f_coul_ct_ct_ijji(
                            c_mat_virts.t(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.t().into_shape([orbs_j * orbs_i]).unwrap());

                        // calculate the gradient of the exchange integral
                        let exchange_gradient: Array1<f64> = f_exchange_ct_ct_ijji(
                            c_mat_virts.t(),
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                            false,
                        )
                        .into_shape([3 * n_atoms, orbs_j * orbs_i])
                        .unwrap()
                        .dot(&c_mat_occs.t().into_shape([orbs_j * orbs_i]).unwrap());

                        let gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;
                        gradient
                    };
                    return_gradient = grad;
                }
            }
            // electrons i,k on I, hole j on J, hole l on K
            CTCoupling::IJIK => {
                // neglect this type of coupling
            }
            // electron i on I, electron k on J, hole j on J, hole l on K
            CTCoupling::IJJK => {
                // neglect this type of coupling
            }
            // electron i on I, electron k on K, hole j on J, hole l on I
            CTCoupling::IJKI => {
                // neglect this type of coupling
            }
            // electron i on I, electron k on K, hole j on J, hole l on J
            CTCoupling::IJKJ => {
                // neglect this type of coupling
            }
            // electron i on I, electron k on K, hole j on J, hole l on L
            CTCoupling::IJKL => {
                // neglect this type of coupling
            }
        }
        return return_gradient;
    }
}
