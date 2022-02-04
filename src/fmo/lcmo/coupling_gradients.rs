use crate::fmo::helpers::get_pair_slice;
use crate::fmo::lcmo::cis_gradient::{ReducedBasisState, ReducedCT, ReducedLE, ReducedParticle};
use crate::fmo::lcmo::helpers::*;
use crate::fmo::lcmo::integrals::CTCoupling;
use crate::fmo::{
    BasisState, ChargeTransfer, ExcitedStateMonomerGradient, LocallyExcited, Monomer, PairType,
    Particle, SuperSystem, LRC,
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

        let n_atoms: usize = mol_i.n_atoms + mol_j.n_atoms;
        let mut gradient: Array1<f64> = Array1::zeros(3 * n_atoms);

        let (state_i, state_j, homo_i, homo_j): (usize, usize, usize, usize) =
            if i.monomer_index < j.monomer_index {
                (i.state_index, j.state_index, i.homo, j.homo)
            } else {
                (j.state_index, i.state_index, j.homo, i.homo)
            };

        if type_pair == PairType::Pair {
            println!("Real pair");
            // calculate the coulomb and exchange contribution of the gradient
            // calculate F[tdm_j] and F_lr[tdm_j]

            // get the index of the pair
            let pair_index: usize = self
                .properties
                .index_of_pair(i.monomer_index, j.monomer_index);
            // get the pair from pairs vector
            let pair: &mut Pair = &mut self.pairs[pair_index];

            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                self.monomers[pair.i].slice.atom_as_range(),
                self.monomers[pair.j].slice.atom_as_range(),
            );
            let atoms_i = self.monomers[pair.i].n_atoms;
            // set necessary arrays for the U matrix calculations
            let monomers: &mut Vec<Monomer> = &mut self.monomers;
            let monomer: &mut Monomer = &mut monomers[pair.i];
            monomer.prepare_u_matrix(&pair_atoms[..atoms_i]);
            let monomer: &mut Monomer = &mut monomers[pair.j];
            monomer.prepare_u_matrix(&pair_atoms[atoms_i..]);
            drop(monomer);
            drop(monomers);

            // monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];
            let n_orbs_i: usize = m_i.n_orbs;
            let n_orbs_j: usize = m_j.n_orbs;

            // calculate S,dS, gamma_AO and dgamma_AO of the pair
            pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
            let grad_s_pair = pair.properties.grad_s().unwrap();
            let grad_s_i: ArrayView3<f64> = grad_s_pair.slice(s![.., ..n_orbs_i, ..n_orbs_i]);
            let grad_s_j: ArrayView3<f64> = grad_s_pair.slice(s![.., n_orbs_i.., n_orbs_i..]);

            // transform the CI coefficients of the monomers to the AO basis
            let nocc_i = m_i.properties.n_occ().unwrap();
            let nvirt_i = m_i.properties.n_virt().unwrap();
            let cis_c_i: ArrayView2<f64> = m_i
                .properties
                .ci_coefficient(state_i)
                .unwrap()
                .into_shape([nocc_i, nvirt_i])
                .unwrap();
            let occs_i = m_i.properties.orbs_slice(0, Some(homo_i + 1)).unwrap();
            let virts_i = m_i.properties.orbs_slice(homo_i + 1, None).unwrap();

            let nocc_j = m_j.properties.n_occ().unwrap();
            let nvirt_j = m_j.properties.n_virt().unwrap();
            let cis_c_j: ArrayView2<f64> = m_j
                .properties
                .ci_coefficient(state_j)
                .unwrap()
                .into_shape([nocc_j, nvirt_j])
                .unwrap();
            let occs_j = m_j.properties.orbs_slice(0, Some(homo_j + 1)).unwrap();
            let virts_j = m_j.properties.orbs_slice(homo_j + 1, None).unwrap();

            let tdm_i: Array2<f64> = occs_i.dot(&cis_c_i.dot(&virts_i.t()));
            let tdm_j: Array2<f64> = occs_j.dot(&cis_c_j.dot(&virts_j.t()));

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

            let coulomb_integral: Array5<f64> = f_coulomb_loop(
                pair.properties.s().unwrap(),
                pair.properties.grad_s().unwrap(),
                pair.properties.gamma_ao().unwrap(),
                pair.properties.grad_gamma_ao().unwrap(),
                pair.n_atoms,
                n_orbs_i,
                n_orbs_j,
            );
            let coulomb_grad: Array1<f64> = coulomb_integral
                .view()
                .into_shape([3 * n_atoms * n_orbs_i * n_orbs_i, n_orbs_j * n_orbs_j])
                .unwrap()
                .dot(&tdm_j.view().into_shape([n_orbs_j * n_orbs_j]).unwrap())
                .into_shape([3 * n_atoms, n_orbs_i * n_orbs_i])
                .unwrap()
                .dot(&tdm_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

            // println!("coulomb gradient: {}",coulomb_gradient.slice(s![0..10]));
            // println!("coulomb grad loop: {}",coulomb_grad.slice(s![0..10]));
            assert!(
                coulomb_gradient.abs_diff_eq(&coulomb_grad, 1e-14),
                "LE-LE coulomb gradient is wrong!"
            );

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

            let exchange_integral: Array5<f64> = f_exchange_loop(
                pair.properties.s().unwrap(),
                pair.properties.grad_s().unwrap(),
                pair.properties.gamma_lr_ao().unwrap(),
                pair.properties.grad_gamma_lr_ao().unwrap(),
                pair.n_atoms,
                n_orbs_i,
                n_orbs_j,
            );
            let exchange_grad: Array1<f64> = exchange_integral
                .view()
                .into_shape([3 * n_atoms * n_orbs_i * n_orbs_i, n_orbs_j * n_orbs_j])
                .unwrap()
                .dot(&tdm_j.view().into_shape([n_orbs_j * n_orbs_j]).unwrap())
                .into_shape([3 * n_atoms, n_orbs_i * n_orbs_i])
                .unwrap()
                .dot(&tdm_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

            // println!("exchange gradient: {}",exchange_gradient.slice(s![0..10]));
            // println!("exchange grad loop: {}",exchange_grad.slice(s![0..10]));
            assert!(
                exchange_gradient.abs_diff_eq(&exchange_grad, 1e-14),
                "LE-LE exchange gradient is wrong!"
            );

            // gradient = 2.0 * coulomb_gradient - exchange_gradient;
            gradient = 2.0 * coulomb_gradient;

            // calculate the cphf correction
            // calculate the U matrix of both monomers using the CPHF equations
            let u_mat_i: Array3<f64> = m_i.calculate_u_matrix(&pair_atoms[..m_i.n_atoms]);
            let u_mat_j: Array3<f64> = m_j.calculate_u_matrix(&pair_atoms[m_i.n_atoms..]);

            // calculate gradients of the MO coefficients for ALL occupied and virtual orbitals
            // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
            let mut dc_mo_i: Array3<f64> =
                Array3::zeros([3 * m_i.n_atoms, m_i.n_orbs, m_i.n_orbs]);
            let mut dc_mo_j: Array3<f64> =
                Array3::zeros([3 * m_j.n_atoms, m_j.n_orbs, m_j.n_orbs]);

            // reference to the mo coefficients of fragment I
            let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            // reference to the mo coefficients of fragment J
            let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

            // iterate over gradient dimensions of both monomers
            for nat in 0..3 * m_i.n_atoms {
                for orb in 0..m_i.n_orbs {
                    dc_mo_i
                        .slice_mut(s![nat, orb, ..])
                        .assign(&u_mat_i.slice(s![nat, .., orb]).dot(&c_mo_i.t()));
                }
            }
            for nat in 0..3 * m_j.n_atoms {
                for orb in 0..m_j.n_orbs {
                    dc_mo_j
                        .slice_mut(s![nat, orb, ..])
                        .assign(&u_mat_j.slice(s![nat, .., orb]).dot(&c_mo_j.t()));
                }
            }

            let mut cphf_tdm_i_1: Array3<f64> =
                Array3::zeros([3 * pair.n_atoms, m_i.n_orbs, m_i.n_orbs]);
            let mut cphf_tdm_j_1: Array3<f64> =
                Array3::zeros([3 * pair.n_atoms, m_j.n_orbs, m_j.n_orbs]);
            // let mut cphf_tdm_i_loop: Array3<f64> =
            //     Array3::zeros([3 * pair.n_atoms, m_i.n_orbs, m_i.n_orbs]);
            // let mut cphf_tdm_j_loop: Array3<f64> =
            //     Array3::zeros([3 * pair.n_atoms, m_j.n_orbs, m_j.n_orbs]);
            for nat in 0..3 * m_i.n_atoms {
                cphf_tdm_i_1.slice_mut(s![nat, .., ..]).assign(
                    &((dc_mo_i
                        .slice(s![nat, ..nocc_i, ..])
                        .t()
                        .dot(&cis_c_i)
                        .dot(&c_mo_i.slice(s![.., nocc_i..]).t()))
                        + (c_mo_i
                            .slice(s![.., ..nocc_i])
                            .dot(&cis_c_i)
                            .dot(&dc_mo_i.slice(s![nat, nocc_i.., ..]))))
                );
            }
            // for nat in 0..3 * m_i.n_atoms {
            //     for orb_i in 0..n_orbs_i{
            //         for orb_j in 0..n_orbs_i{
            //             cphf_tdm_i_loop[[nat, orb_i, orb_j]] +=
            //                 (dc_mo_i
            //                     .slice(s![nat, ..nocc_i, orb_i])
            //                     .dot(&cis_c_i)
            //                     .dot(&c_mo_i.slice(s![orb_j, nocc_i..])))
            //                     + (c_mo_i
            //                     .slice(s![orb_i, ..nocc_i])
            //                     .dot(&cis_c_i)
            //                     .dot(&dc_mo_i.slice(s![nat, nocc_i.., orb_j])));
            //         }
            //     }
            // }
            // assert!(cphf_tdm_i_1.abs_diff_eq(&cphf_tdm_i_loop,1e-10),"CPHF TDMS not equal!");
            for nat in 0..3 * m_j.n_atoms {
                cphf_tdm_j_1
                    .slice_mut(s![3*m_i.n_atoms +nat, .., ..])
                    .assign(
                        &((dc_mo_j
                            .slice(s![nat, ..nocc_j, ..])
                            .t()
                            .dot(&cis_c_j)
                            .dot(&c_mo_j.slice(s![.., nocc_j..]).t()))
                            + (c_mo_j
                                .slice(s![.., ..nocc_j])
                                .dot(&cis_c_j)
                                .dot(&dc_mo_j.slice(s![nat, nocc_j.., ..]))))
                    );
            }
            // for nat in 0..3 * m_j.n_atoms {
            //     for orb_i in 0..n_orbs_j{
            //         for orb_j in 0..n_orbs_j{
            //             cphf_tdm_j_loop[[3*m_i.n_atoms +nat, orb_i, orb_j]] +=
            //                 (dc_mo_j
            //                     .slice(s![nat, ..nocc_j, orb_i])
            //                     .dot(&cis_c_j)
            //                     .dot(&c_mo_j.slice(s![orb_j, nocc_j..])))
            //                     + (c_mo_j
            //                     .slice(s![orb_i, ..nocc_j])
            //                     .dot(&cis_c_j)
            //                     .dot(&dc_mo_j.slice(s![nat, nocc_j.., orb_j])));
            //         }
            //     }
            // }
            // assert!(cphf_tdm_j_1.abs_diff_eq(&cphf_tdm_j_loop,1e-10),"CPHF TDMS not equal!");
            // transform cphf tdm matrices to the shape [grad,norb*norb]
            let cphf_tdm_i: Array2<f64> = cphf_tdm_i_1
                .into_shape([3 * pair.n_atoms, m_i.n_orbs * m_i.n_orbs])
                .unwrap();
            let cphf_tdm_j: Array2<f64> = cphf_tdm_j_1
                .into_shape([3 * pair.n_atoms, m_j.n_orbs * m_j.n_orbs])
                .unwrap();
            let tdm_i_1: ArrayView1<f64> = tdm_i.view().into_shape([m_i.n_orbs * m_i.n_orbs]).unwrap();
            let tdm_j_1: ArrayView1<f64> = tdm_j.view().into_shape([m_j.n_orbs * m_j.n_orbs]).unwrap();

            // calculate coulomb and exchange integrals in AO basis
            let mut coulomb_arr: Array4<f64> = coulomb_integral_loop_ao(
                m_i.properties.s().unwrap(),
                m_j.properties.s().unwrap(),
                pair.properties.gamma_ao().unwrap(),
                m_i.n_orbs,
                m_j.n_orbs,
            );
            let exchange_arr: Array4<f64> = exchange_integral_loop_ao(
                pair.properties.s().unwrap(),
                pair.properties.gamma_lr_ao().unwrap(),
                m_i.n_orbs,
                m_j.n_orbs,
            );
            // coulomb_arr = 2.0 * coulomb_arr - exchange_arr;
            coulomb_arr = 2.0 * coulomb_arr;
            let coulomb_arr_1: ArrayView2<f64> = coulomb_arr.view()
                .into_shape([m_i.n_orbs * m_i.n_orbs, m_j.n_orbs * m_j.n_orbs])
                .unwrap();

            // (d/dr tdm_i * tdm_j + tdm_i * d/dr tdm_j) * [2*exchange - coulomb]
            let mut cphf_grad = cphf_tdm_i.dot(&coulomb_arr_1.dot(&tdm_j_1));
            cphf_grad = cphf_grad + tdm_i_1.dot(&coulomb_arr_1).dot(&cphf_tdm_j.t());

            // let mut cphf_grad_loop:Array1<f64> = Array1::zeros(3*pair.n_atoms);
            // for nat in 0..3*pair.n_atoms{
            //     for mu in 0..n_orbs_i{
            //         for nu in 0..n_orbs_i{
            //             for la in 0..n_orbs_j{
            //                 for sig in 0..n_orbs_j{
            //                     cphf_grad_loop[nat] += (cphf_tdm_i_loop[[nat,mu,nu]] * tdm_j[[la,sig]]
            //                         + tdm_i[[mu,nu]] * cphf_tdm_j_loop[[nat,la,sig]]) *
            //                         coulomb_arr[[mu,nu,la,sig]];
            //                 }
            //             }
            //         }
            //     }
            // }
            // assert!(cphf_grad.abs_diff_eq(&cphf_grad_loop,1e-10),"NOT equal to Loop version!");

            // add cphf term to the gradient
            gradient = gradient + cphf_grad;

            drop(pair);
            drop(m_i);
            drop(m_j);
            let index_i:usize = m_i.index;
            let index_j:usize = m_j.index;
            let cpcis_coeff_i:Array3<f64> =
                self.cpcis_routine(index_i,state_i,u_mat_i.view(),nocc_i,nvirt_i);
            let cpcis_coeff_j:Array3<f64> =
                self.cpcis_routine(index_j,state_j,u_mat_j.view(),nocc_j,nvirt_j);

            let pair: &mut Pair = &mut self.pairs[pair_index];
            // monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];
            // reference to the mo coefficients of fragment I
            let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            // reference to the mo coefficients of fragment J
            let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

            let mut cpcis_tdm_i: Array3<f64> =
                Array3::zeros([3 * pair.n_atoms, n_orbs_i, n_orbs_i]);
            let mut cpcis_tdm_j: Array3<f64> =
                Array3::zeros([3 * pair.n_atoms, n_orbs_j, n_orbs_j]);

            for nat in 0..3 * m_i.n_atoms {
                cpcis_tdm_i.slice_mut(s![nat, .., ..]).assign(
                    &(c_mo_i
                        .slice(s![.., ..nocc_i])
                        .dot(&cpcis_coeff_i.slice(s![nat,..,..]))
                        .dot(&c_mo_i.slice(s![.., nocc_i..]).t()))
                );
            }
            for nat in 0..3 * m_j.n_atoms {
                cpcis_tdm_j.slice_mut(s![3*m_i.n_atoms +nat, .., ..]).assign(
                    &(c_mo_j
                        .slice(s![.., ..nocc_j])
                        .dot(&cpcis_coeff_j.slice(s![nat,..,..]))
                        .dot(&c_mo_j.slice(s![.., nocc_j..]).t()))
                );
            }
            let cpcis_tdm_i: Array2<f64> = cpcis_tdm_i
                .into_shape([3 * pair.n_atoms, m_i.n_orbs * m_i.n_orbs])
                .unwrap();
            let cpcis_tdm_j: Array2<f64> = cpcis_tdm_j
                .into_shape([3 * pair.n_atoms, m_j.n_orbs * m_j.n_orbs])
                .unwrap();

            let mut cpcis_grad = cpcis_tdm_i.dot(&coulomb_arr_1.dot(&tdm_j_1));
            cpcis_grad = cpcis_grad + tdm_i_1.dot(&coulomb_arr_1).dot(&cpcis_tdm_j.t());
            println!("cpcis_grad {}",cpcis_grad.slice(s![0..10]));
            // add the cpcis term to the gradient
            // gradient = gradient + cpcis_grad;

            // remove data from the RAM after the calculation
            pair.properties.reset_gradient();
        } else {
            println!("ESD Pair");
            // calculate only the coulomb contribution of the gradient
            // calculate F[tdm_j]
            let pair_index: usize = self
                .properties
                .index_of_esd_pair(i.monomer_index, j.monomer_index);
            // get correct pair from pairs vector
            let esd_pair: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // get pair atoms
            let esd_pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                self.monomers[esd_pair.i].slice.atom_as_range(),
                self.monomers[esd_pair.j].slice.atom_as_range(),
            );

            let atoms_i = self.monomers[esd_pair.i].n_atoms;
            // set necessary arrays for the U matrix calculations
            let monomers: &mut Vec<Monomer> = &mut self.monomers;
            let monomer: &mut Monomer = &mut monomers[esd_pair.i];
            monomer.prepare_u_matrix(&esd_pair_atoms[..atoms_i]);
            let monomer: &mut Monomer = &mut monomers[esd_pair.j];
            monomer.prepare_u_matrix(&esd_pair_atoms[atoms_i..]);
            drop(monomer);
            drop(monomers);

            // monomers
            let m_i: &Monomer = &self.monomers[esd_pair.i];
            let m_j: &Monomer = &self.monomers[esd_pair.j];
            let n_orbs_i: usize = m_i.n_orbs;
            let n_orbs_j: usize = m_j.n_orbs;

            // transform the CI coefficients of the monomers to the AO basis
            let nocc_i = m_i.properties.n_occ().unwrap();
            let nvirt_i = m_i.properties.n_virt().unwrap();
            let cis_c_i: ArrayView2<f64> = m_i
                .properties
                .ci_coefficient(state_i)
                .unwrap()
                .into_shape([nocc_i, nvirt_i])
                .unwrap();
            let occs_i = m_i.properties.orbs_slice(0, Some(homo_i + 1)).unwrap();
            let virts_i = m_i.properties.orbs_slice(homo_i + 1, None).unwrap();

            let nocc_j = m_j.properties.n_occ().unwrap();
            let nvirt_j = m_j.properties.n_virt().unwrap();
            let cis_c_j: ArrayView2<f64> = m_j
                .properties
                .ci_coefficient(state_j)
                .unwrap()
                .into_shape([nocc_j, nvirt_j])
                .unwrap();
            let occs_j = m_j.properties.orbs_slice(0, Some(homo_j + 1)).unwrap();
            let virts_j = m_j.properties.orbs_slice(homo_j + 1, None).unwrap();

            let tdm_i: Array2<f64> = occs_i.dot(&cis_c_i.dot(&virts_i.t()));
            let tdm_j: Array2<f64> = occs_j.dot(&cis_c_j.dot(&virts_j.t()));

            // do a scc routine and prepare matrices for the gradient calculation
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

            // calculate the cphf correction
            // calculate the U matrix of both monomers using the CPHF equations
            let u_mat_i: Array3<f64> = m_i.calculate_u_matrix(&esd_pair_atoms[..m_i.n_atoms]);
            let u_mat_j: Array3<f64> = m_j.calculate_u_matrix(&esd_pair_atoms[m_i.n_atoms..]);

            // calculate gradients of the MO coefficients for ALL occupied and virtual orbitals
            // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
            let mut dc_mo_i: Array3<f64> =
                Array3::zeros([3 * esd_pair.n_atoms, m_i.n_orbs, m_i.n_orbs]);
            let mut dc_mo_j: Array3<f64> =
                Array3::zeros([3 * esd_pair.n_atoms, m_i.n_orbs, m_j.n_orbs]);

            // reference to the mo coefficients of fragment I
            let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
            // reference to the mo coefficients of fragment J
            let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

            // iterate over gradient dimensions of both monomers
            for nat in 0..3 * m_i.n_atoms {
                for orb in 0..m_i.n_orbs {
                    dc_mo_i
                        .slice_mut(s![nat, orb, ..])
                        .assign(&u_mat_i.slice(s![nat, .., orb]).dot(&c_mo_i.t()));
                }
            }
            for nat in 0..3 * m_j.n_atoms {
                for orb in 0..m_j.n_orbs {
                    dc_mo_j
                        .slice_mut(s![3 * m_i.n_atoms + nat, orb, ..])
                        .assign(&u_mat_j.slice(s![nat, .., orb]).dot(&c_mo_j.t()));
                }
            }

            let mut cphf_tdm_i: Array3<f64> =
                Array3::zeros([3 * esd_pair.n_atoms, m_i.n_orbs, m_i.n_orbs]);
            let mut cphf_tdm_j: Array3<f64> =
                Array3::zeros([3 * esd_pair.n_atoms, m_j.n_orbs, m_j.n_orbs]);
            for nat in 0..3 * m_i.n_atoms {
                cphf_tdm_i.slice_mut(s![nat, .., ..]).assign(
                    &((dc_mo_i
                        .slice(s![nat, ..nocc_i, ..])
                        .t()
                        .dot(&cis_c_i)
                        .dot(&c_mo_i.slice(s![.., nocc_i..]).t()))
                        + (c_mo_i
                            .slice(s![.., ..nocc_i])
                            .dot(&cis_c_i)
                            .dot(&dc_mo_i.slice(s![nat, nocc_i.., ..])))),
                );
            }
            for nat in 0..3 * m_j.n_atoms {
                cphf_tdm_j
                    .slice_mut(s![3 * m_i.n_atoms + nat, .., ..])
                    .assign(
                        &((dc_mo_j
                            .slice(s![3 * m_i.n_atoms + nat, ..nocc_j, ..])
                            .t()
                            .dot(&cis_c_j)
                            .dot(&c_mo_j.slice(s![.., nocc_j..]).t()))
                            + (c_mo_j
                                .slice(s![.., ..nocc_j])
                                .dot(&cis_c_j)
                                .dot(&dc_mo_j.slice(s![3 * m_i.n_atoms + nat, nocc_j.., ..])))),
                    );
            }
            // transform cphf tdm matrices to the shape [grad,norb*norb]
            let cphf_tdm_i: Array2<f64> = cphf_tdm_i
                .into_shape([3 * esd_pair.n_atoms, m_i.n_orbs * m_i.n_orbs])
                .unwrap();
            let cphf_tdm_j: Array2<f64> = cphf_tdm_j
                .into_shape([3 * esd_pair.n_atoms, m_j.n_orbs * m_j.n_orbs])
                .unwrap();
            let tdm_i: Array1<f64> = tdm_i.into_shape([m_i.n_orbs * m_i.n_orbs]).unwrap();
            let tdm_j: Array1<f64> = tdm_j.into_shape([m_j.n_orbs * m_j.n_orbs]).unwrap();

            // calculate coulomb and exchange integrals in AO basis
            let mut coulomb_arr: Array4<f64> = coulomb_integral_loop_ao(
                m_i.properties.s().unwrap(),
                m_j.properties.s().unwrap(),
                esd_pair.properties.gamma_ao().unwrap(),
                m_i.n_orbs,
                m_j.n_orbs,
            );
            coulomb_arr = 2.0 * coulomb_arr;
            let coulomb_arr: Array2<f64> = coulomb_arr
                .into_shape([m_i.n_orbs * m_i.n_orbs, m_j.n_orbs * m_j.n_orbs])
                .unwrap();

            // (d/dr tdm_i * tdm_j + tdm_i * d/dr tdm_j) * [2*exchange - coulomb]
            let term_1: Array1<f64> = cphf_tdm_i.dot(&coulomb_arr.dot(&tdm_j));
            let term_2: Array1<f64> = tdm_i.dot(&coulomb_arr).dot(&cphf_tdm_j.t());

            gradient = gradient + term_1 + term_2;

            // // Testing the gradient
            // let qov_i: ArrayView2<f64> = m_i.properties.q_ov().unwrap();
            // let qov_j: ArrayView2<f64> = m_j.properties.q_ov().unwrap();
            // let qtrans_i:Array1<f64> = qov_i.dot(&cis_c_i.into_shape([nocc_i*nvirt_i]).unwrap());
            // let qtrans_j:Array1<f64> = qov_j.dot(&cis_c_j.into_shape([nocc_j*nvirt_j]).unwrap());
            // let grad_gamma_pair:ArrayView3<f64> = esd_pair.properties.grad_gamma().unwrap();
            //
            // for nc in 0..3*esd_pair.n_atoms{
            //     gradient[nc] = qtrans_i.dot(&grad_gamma_pair.slice(s![nc,..m_i.n_atoms,m_i.n_atoms..]).dot(&qtrans_j));
            // }

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
        let cis_c: Array2<f64> = mol_i
            .properties
            .ci_coefficient(i.state_index)
            .unwrap()
            .into_shape([nocc, nvirt])
            .unwrap()
            .to_owned();
        let occs = mol_i.properties.orbs_slice(0, Some(i.homo + 1)).unwrap();
        let virts = mol_i.properties.orbs_slice(i.homo + 1, None).unwrap();
        let tdm: Array2<f64> = occs.dot(&cis_c.dot(&virts.t()));

        // initialize return value
        let mut return_gradient: Array1<f64> = Array1::zeros(3 * mol_i.n_atoms);
        drop(mol_i);
        // < LE I | H | CT J_j -> I_b>
        if i.monomer_index == j.electron.m_index {
            // Check if the pair IK is close, so that the overlap is non-zero.
            if type_ij == PairType::Pair {
                // get the index of the pair
                let pair_index: usize = self
                    .properties
                    .index_of_pair(i.monomer_index, j.hole.m_index);
                // get the pair from pairs vector
                let pair: &mut Pair = &mut self.pairs[pair_index];
                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    self.monomers[pair.i].slice.atom_as_range(),
                    self.monomers[pair.j].slice.atom_as_range(),
                );

                let atoms_i = self.monomers[pair.i].n_atoms;
                // set necessary arrays for the U matrix calculations
                let monomers: &mut Vec<Monomer> = &mut self.monomers;
                let monomer: &mut Monomer = &mut monomers[pair.i];
                monomer.prepare_u_matrix(&pair_atoms[..atoms_i]);
                let monomer: &mut Monomer = &mut monomers[pair.j];
                monomer.prepare_u_matrix(&pair_atoms[atoms_i..]);
                drop(monomer);
                drop(monomers);

                // monomers
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];
                let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                let n_orbs_i: usize = m_i.n_orbs;
                let n_orbs_j: usize = m_j.n_orbs;
                // calculate S,dS, gamma_AO and dgamma_AO of the pair
                pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
                // calculate the U matrix of both monomers using the CPHF equations
                let u_mat_i: Array3<f64> = m_i.calculate_u_matrix(&pair_atoms[..m_i.n_atoms]);
                let u_mat_j: Array3<f64> = m_j.calculate_u_matrix(&pair_atoms[m_i.n_atoms..]);

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

                    let mut gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;

                    // calculate the cphf correction
                    // calculate gradients of the MO coefficients for ALL occupied and virtual orbitals
                    // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
                    let mut dc_mo_i: Array3<f64> =
                        Array3::zeros([3 * pair.n_atoms, m_i.n_orbs, m_i.n_orbs]);
                    let mut dc_mo_j: Array2<f64> = Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);

                    // reference to the mo coefficients of fragment I
                    let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
                    // reference to the mo coefficients of fragment J
                    let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

                    // iterate over gradient dimensions of both monomers
                    for nat in 0..3 * m_i.n_atoms {
                        for orb in 0..m_i.n_orbs {
                            dc_mo_i
                                .slice_mut(s![nat, orb, ..])
                                .assign(&u_mat_i.slice(s![nat, .., orb]).dot(&c_mo_i.t()));
                        }
                    }
                    for nat in 0..3 * m_j.n_atoms {
                        dc_mo_j
                            .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                            .assign(&u_mat_j.slice(s![nat, .., j.hole.mo.index]).dot(&c_mo_j.t()));
                    }
                    let mut cphf_tdm_i: Array3<f64> =
                        Array3::zeros([3 * pair.n_atoms, m_i.n_orbs, m_i.n_orbs]);
                    for nat in 0..3 * m_i.n_atoms {
                        cphf_tdm_i.slice_mut(s![nat, .., ..]).assign(
                            &((dc_mo_i
                                .slice(s![nat, ..nocc, ..])
                                .t()
                                .dot(&cis_c)
                                .dot(&c_mo_i.slice(s![.., nocc..]).t()))
                                + (c_mo_i
                                    .slice(s![.., ..nocc])
                                    .dot(&cis_c)
                                    .dot(&dc_mo_i.slice(s![nat, nocc.., ..])))),
                        );
                    }
                    // transform cphf tdm matrices to the shape [grad,norb*norb]
                    let cphf_tdm_i: Array2<f64> = cphf_tdm_i
                        .into_shape([3 * pair.n_atoms, m_i.n_orbs * m_i.n_orbs])
                        .unwrap();
                    let tdm_i: Array1<f64> = tdm.into_shape([m_i.n_orbs * m_i.n_orbs]).unwrap();

                    let mut coulomb_integral: Array4<f64> = coulomb_integral_loop_ao_lect(
                        pair.properties.s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    );
                    let exchange_integral: Array4<f64> = exchange_integral_loop_ao_lect(
                        pair.properties.s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    );
                    coulomb_integral = 2.0 * coulomb_integral - exchange_integral;
                    let coulomb_integral: Array2<f64> = coulomb_integral
                        .into_shape([n_orbs_i * n_orbs_i, n_orbs_i * n_orbs_j])
                        .unwrap();

                    let c_i_ind: ArrayView1<f64> = c_mo_i.slice(s![.., j.electron.mo.index]);
                    let c_j_ind: ArrayView1<f64> = c_mo_j.slice(s![.., j.hole.mo.index]);
                    let derivative_c_i: ArrayView2<f64> =
                        dc_mo_i.slice(s![.., j.electron.mo.index, ..]);

                    // calculate (d/dr tdm_i * C^J_hole * C^I_elec + tdm_i * d/dr C^J_hole * C^I_elec + tdm_i * C^J_hole * d/dr C^I_elec)
                    let term_1: Array1<f64> = cphf_tdm_i.dot(
                        &coulomb_integral.dot(
                            &(into_col(c_i_ind.to_owned()).dot(&into_row(c_j_ind.to_owned())))
                                .into_shape([n_orbs_i * n_orbs_j])
                                .unwrap(),
                        ),
                    );
                    let term_2: Array1<f64> = derivative_c_i.dot(
                        &tdm_i
                            .dot(&coulomb_integral)
                            .into_shape([n_orbs_i, n_orbs_j])
                            .unwrap()
                            .dot(&c_j_ind),
                    );
                    let term_3: Array1<f64> = c_i_ind
                        .dot(
                            &tdm_i
                                .dot(&coulomb_integral)
                                .into_shape([n_orbs_i, n_orbs_j])
                                .unwrap(),
                        )
                        .dot(&dc_mo_j.t());

                    gradient = gradient + term_1 + term_2 + term_3;

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
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_j])
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
                    .into_shape([3 * n_atoms * n_orbs_j, n_orbs_i])
                    .unwrap()
                    .dot(&j.hole.mo.c)
                    .into_shape([3 * n_atoms, n_orbs_i])
                    .unwrap()
                    .dot(&j.electron.mo.c);

                    let mut gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;

                    // calculate the cphf correction
                    // calculate gradients of the MO coefficients for ALL occupied and virtual orbitals
                    // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
                    let mut dc_mo_j: Array3<f64> =
                        Array3::zeros([3 * pair.n_atoms, m_j.n_orbs, m_j.n_orbs]);
                    let mut dc_mo_i: Array2<f64> = Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);

                    // reference to the mo coefficients of fragment I
                    let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
                    // reference to the mo coefficients of fragment J
                    let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

                    // iterate over gradient dimensions of both monomers
                    for nat in 0..3 * m_j.n_atoms {
                        for orb in 0..m_j.n_orbs {
                            dc_mo_j
                                .slice_mut(s![3 * m_i.n_atoms + nat, orb, ..])
                                .assign(&u_mat_j.slice(s![nat, .., orb]).dot(&c_mo_j.t()));
                        }
                    }
                    for nat in 0..3 * m_i.n_atoms {
                        dc_mo_i
                            .slice_mut(s![nat, ..])
                            .assign(&u_mat_j.slice(s![nat, .., j.hole.mo.index]).dot(&c_mo_i.t()));
                    }
                    let mut cphf_tdm_j: Array3<f64> =
                        Array3::zeros([3 * pair.n_atoms, m_j.n_orbs, m_j.n_orbs]);
                    for nat in 0..3 * m_j.n_atoms {
                        cphf_tdm_j.slice_mut(s![nat, .., ..]).assign(
                            &((dc_mo_j
                                .slice(s![nat, ..nocc, ..])
                                .t()
                                .dot(&cis_c)
                                .dot(&c_mo_j.slice(s![.., nocc..]).t()))
                                + (c_mo_j
                                    .slice(s![.., ..nocc])
                                    .dot(&cis_c)
                                    .dot(&dc_mo_j.slice(s![nat, nocc.., ..])))),
                        );
                    }
                    // transform cphf tdm matrices to the shape [grad,norb*norb]
                    let cphf_tdm_j: Array2<f64> = cphf_tdm_j
                        .into_shape([3 * pair.n_atoms, m_j.n_orbs * m_j.n_orbs])
                        .unwrap();
                    let tdm_j: Array1<f64> = tdm.into_shape([m_j.n_orbs * m_j.n_orbs]).unwrap();

                    let mut coulomb_integral: Array4<f64> = coulomb_integral_loop_ao_lect(
                        pair.properties.s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    );
                    let exchange_integral: Array4<f64> = exchange_integral_loop_ao_lect(
                        pair.properties.s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    );
                    coulomb_integral = 2.0 * coulomb_integral - exchange_integral;
                    let coulomb_integral: Array2<f64> = coulomb_integral
                        .into_shape([n_orbs_j * n_orbs_j, n_orbs_j * n_orbs_i])
                        .unwrap();

                    let c_j_ind: ArrayView1<f64> = c_mo_i.slice(s![.., j.electron.mo.index]);
                    let c_i_ind: ArrayView1<f64> = c_mo_j.slice(s![.., j.hole.mo.index]);
                    let derivative_c_j: ArrayView2<f64> =
                        dc_mo_j.slice(s![.., j.electron.mo.index, ..]);

                    // calculate (d/dr tdm_i * C^J_hole * C^I_elec + tdm_i * d/dr C^J_hole * C^I_elec + tdm_i * C^J_hole * d/dr C^I_elec)
                    let term_1: Array1<f64> = cphf_tdm_j.dot(
                        &coulomb_integral.dot(
                            &(into_col(c_j_ind.to_owned()).dot(&into_row(c_i_ind.to_owned())))
                                .into_shape([n_orbs_j * n_orbs_i])
                                .unwrap(),
                        ),
                    );
                    let term_2: Array1<f64> = derivative_c_j.dot(
                        &tdm_j
                            .dot(&coulomb_integral)
                            .into_shape([n_orbs_j, n_orbs_i])
                            .unwrap()
                            .dot(&c_i_ind),
                    );
                    let term_3: Array1<f64> = c_j_ind
                        .dot(
                            &tdm_j
                                .dot(&coulomb_integral)
                                .into_shape([n_orbs_j, n_orbs_i])
                                .unwrap(),
                        )
                        .dot(&dc_mo_i.t());

                    gradient = gradient + term_1 + term_2 + term_3;

                    gradient
                };

                pair.properties.reset_gradient();
                return_gradient = grad;
            }
        } else if i.monomer_index == j.hole.m_index {
            // Check if the pair IJ is close, so that the overlap is non-zero.
            if type_ik == PairType::Pair {
                // get the index of the pair
                let pair_index: usize = self
                    .properties
                    .index_of_pair(i.monomer_index, j.electron.m_index);
                // get the pair from pairs vector
                let pair: &mut Pair = &mut self.pairs[pair_index];
                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    self.monomers[pair.i].slice.atom_as_range(),
                    self.monomers[pair.j].slice.atom_as_range(),
                );

                let atoms_i = self.monomers[pair.i].n_atoms;
                // set necessary arrays for the U matrix calculations
                let monomers: &mut Vec<Monomer> = &mut self.monomers;
                let monomer: &mut Monomer = &mut monomers[pair.i];
                monomer.prepare_u_matrix(&pair_atoms[..atoms_i]);
                let monomer: &mut Monomer = &mut monomers[pair.j];
                monomer.prepare_u_matrix(&pair_atoms[atoms_i..]);
                drop(monomer);
                drop(monomers);

                // monomers
                let m_i: &Monomer = &self.monomers[pair.i];
                let m_j: &Monomer = &self.monomers[pair.j];
                let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                let n_orbs_i: usize = m_i.n_orbs;
                let n_orbs_j: usize = m_j.n_orbs;

                // calculate S,dS, gamma_AO and dgamma_AO of the pair
                pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
                // calculate the U matrix of both monomers using the CPHF equations
                let u_mat_i: Array3<f64> = m_i.calculate_u_matrix(&pair_atoms[..m_i.n_atoms]);
                let u_mat_j: Array3<f64> = m_j.calculate_u_matrix(&pair_atoms[m_i.n_atoms..]);

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

                    let coulomb_integral: Array5<f64> = f_le_ct_coulomb_loop(
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                    );
                    let coulomb_grad: Array1<f64> = coulomb_integral
                        .view()
                        .into_shape([3 * n_atoms * n_orbs_i * n_orbs_i * n_orbs_i, n_orbs_j])
                        .unwrap()
                        .dot(&j.electron.mo.c)
                        .into_shape([3 * n_atoms * n_orbs_i * n_orbs_i, n_orbs_i])
                        .unwrap()
                        .dot(&j.hole.mo.c)
                        .into_shape([3 * n_atoms, n_orbs_i * n_orbs_i])
                        .unwrap()
                        .dot(&tdm.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

                    // println!("coulomb gradient: {}",coulomb_gradient.slice(s![0..10]));
                    // println!("coulomb grad loop: {}",coulomb_grad.slice(s![0..10]));
                    assert!(
                        coulomb_gradient.abs_diff_eq(&coulomb_grad, 1e-14),
                        "LE-LE coulomb gradient is wrong!"
                    );

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

                    let exchange_integral: Array5<f64> = f_le_ct_exchange_loop(
                        pair.properties.s().unwrap(),
                        pair.properties.grad_s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        pair.properties.grad_gamma_lr_ao().unwrap(),
                        n_atoms,
                        n_orbs_i,
                        n_orbs_j,
                    );
                    let exchange_grad: Array1<f64> = exchange_integral
                        .view()
                        .into_shape([3 * n_atoms * n_orbs_i * n_orbs_i * n_orbs_i, n_orbs_j])
                        .unwrap()
                        .dot(&j.electron.mo.c)
                        .into_shape([3 * n_atoms * n_orbs_i * n_orbs_i, n_orbs_i])
                        .unwrap()
                        .dot(&j.hole.mo.c)
                        .into_shape([3 * n_atoms, n_orbs_i * n_orbs_i])
                        .unwrap()
                        .dot(&tdm.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

                    println!(" ");
                    // println!("exchange gradient: {}",exchange_gradient.slice(s![0..10]));
                    // println!("exchange grad loop: {}",exchange_grad.slice(s![0..10]));
                    // assert!(exchange_gradient.abs_diff_eq(&exchange_grad,1e-14),"LE-LE exchange gradient is wrong!");

                    // let mut gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;
                    let mut gradient: Array1<f64> = 2.0 * coulomb_gradient;
                    // let mut gradient: Array1<f64> = - exchange_grad;

                    // calculate the cphf correction
                    // calculate gradients of the MO coefficients for ALL occupied and virtual orbitals
                    // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
                    let mut dc_mo_i: Array3<f64> =
                        Array3::zeros([3 * pair.n_atoms, m_i.n_orbs, m_i.n_orbs]);
                    let mut dc_mo_j: Array2<f64> = Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);

                    // reference to the mo coefficients of fragment I
                    let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
                    // reference to the mo coefficients of fragment J
                    let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

                    // iterate over gradient dimensions of both monomers
                    for nat in 0..3 * m_i.n_atoms {
                        for orb in 0..m_i.n_orbs {
                            dc_mo_i
                                .slice_mut(s![nat, orb, ..])
                                .assign(&u_mat_i.slice(s![nat, .., orb]).dot(&c_mo_i.t()));
                        }
                    }
                    for nat in 0..3 * m_j.n_atoms {
                        dc_mo_j.slice_mut(s![3 * m_i.n_atoms + nat, ..]).assign(
                            &u_mat_j
                                .slice(s![nat, .., j.electron.mo.index])
                                .dot(&c_mo_j.t()),
                        );
                    }
                    let mut cphf_tdm_i: Array3<f64> =
                        Array3::zeros([3 * pair.n_atoms, m_i.n_orbs, m_i.n_orbs]);
                    for nat in 0..3 * m_i.n_atoms {
                        cphf_tdm_i.slice_mut(s![nat, .., ..]).assign(
                            &((dc_mo_i
                                .slice(s![nat, ..nocc, ..])
                                .t()
                                .dot(&cis_c)
                                .dot(&c_mo_i.slice(s![.., nocc..]).t()))
                                + (c_mo_i
                                    .slice(s![.., ..nocc])
                                    .dot(&cis_c)
                                    .dot(&dc_mo_i.slice(s![nat, nocc.., ..])))),
                        );
                    }
                    // transform cphf tdm matrices to the shape [grad,norb*norb]
                    let cphf_tdm_i: Array2<f64> = cphf_tdm_i
                        .into_shape([3 * pair.n_atoms, m_i.n_orbs * m_i.n_orbs])
                        .unwrap();
                    let tdm_i: Array1<f64> = tdm.into_shape([m_i.n_orbs * m_i.n_orbs]).unwrap();

                    let mut coulomb_integral: Array4<f64> = coulomb_integral_loop_ao_lect(
                        pair.properties.s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    );
                    let exchange_integral: Array4<f64> = exchange_integral_loop_ao_lect(
                        pair.properties.s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        n_orbs_i,
                        n_orbs_j,
                        true,
                    );
                    // coulomb_integral = 2.0 * coulomb_integral - exchange_integral;
                    coulomb_integral = 2.0 * coulomb_integral;
                    // coulomb_integral = - exchange_integral;

                    let coulomb_integral: Array2<f64> = coulomb_integral
                        .into_shape([n_orbs_i * n_orbs_i, n_orbs_i * n_orbs_j])
                        .unwrap();

                    let c_i_ind: ArrayView1<f64> = c_mo_i.slice(s![.., j.hole.mo.index]);
                    let c_j_ind: ArrayView1<f64> = c_mo_j.slice(s![.., j.electron.mo.index]);
                    let derivative_c_i: ArrayView2<f64> =
                        dc_mo_i.slice(s![.., j.hole.mo.index, ..]);

                    // calculate (d/dr tdm_i * C^J_hole * C^I_elec + tdm_i * d/dr C^J_hole * C^I_elec + tdm_i * C^J_hole * d/dr C^I_elec)
                    let term_1: Array1<f64> = cphf_tdm_i.dot(
                        &coulomb_integral.dot(
                            &(into_col(c_i_ind.to_owned()).dot(&into_row(c_j_ind.to_owned())))
                                .into_shape([n_orbs_i * n_orbs_j])
                                .unwrap(),
                        ),
                    );
                    let term_2: Array1<f64> = derivative_c_i.dot(
                        &tdm_i
                            .dot(&coulomb_integral)
                            .into_shape([n_orbs_i, n_orbs_j])
                            .unwrap()
                            .dot(&c_j_ind),
                    );
                    let term_3: Array1<f64> = c_i_ind
                        .dot(
                            &tdm_i
                                .dot(&coulomb_integral)
                                .into_shape([n_orbs_i, n_orbs_j])
                                .unwrap(),
                        )
                        .dot(&dc_mo_j.t());

                    gradient = gradient + term_1 + term_2 + term_3;

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

                    let mut gradient: Array1<f64> = 2.0 * coulomb_gradient - exchange_gradient;

                    // calculate the cphf correction
                    // calculate gradients of the MO coefficients for ALL occupied and virtual orbitals
                    // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
                    let mut dc_mo_j: Array3<f64> =
                        Array3::zeros([3 * pair.n_atoms, m_j.n_orbs, m_j.n_orbs]);
                    let mut dc_mo_i: Array2<f64> = Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);

                    // reference to the mo coefficients of fragment I
                    let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
                    // reference to the mo coefficients of fragment J
                    let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

                    // iterate over gradient dimensions of both monomers
                    for nat in 0..3 * m_j.n_atoms {
                        for orb in 0..m_j.n_orbs {
                            dc_mo_j
                                .slice_mut(s![3 * m_i.n_atoms + nat, orb, ..])
                                .assign(&u_mat_j.slice(s![nat, .., orb]).dot(&c_mo_j.t()));
                        }
                    }
                    for nat in 0..3 * m_i.n_atoms {
                        dc_mo_i.slice_mut(s![nat, ..]).assign(
                            &u_mat_j
                                .slice(s![nat, .., j.electron.mo.index])
                                .dot(&c_mo_i.t()),
                        );
                    }
                    let mut cphf_tdm_j: Array3<f64> =
                        Array3::zeros([3 * pair.n_atoms, m_j.n_orbs, m_j.n_orbs]);
                    for nat in 0..3 * m_j.n_atoms {
                        cphf_tdm_j.slice_mut(s![nat, .., ..]).assign(
                            &((dc_mo_j
                                .slice(s![nat, ..nocc, ..])
                                .t()
                                .dot(&cis_c)
                                .dot(&c_mo_j.slice(s![.., nocc..]).t()))
                                + (c_mo_j
                                    .slice(s![.., ..nocc])
                                    .dot(&cis_c)
                                    .dot(&dc_mo_j.slice(s![nat, nocc.., ..])))),
                        );
                    }
                    // transform cphf tdm matrices to the shape [grad,norb*norb]
                    let cphf_tdm_j: Array2<f64> = cphf_tdm_j
                        .into_shape([3 * pair.n_atoms, m_j.n_orbs * m_j.n_orbs])
                        .unwrap();
                    let tdm_j: Array1<f64> = tdm.into_shape([m_j.n_orbs * m_j.n_orbs]).unwrap();

                    let mut coulomb_integral: Array4<f64> = coulomb_integral_loop_ao_lect(
                        pair.properties.s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    );
                    let exchange_integral: Array4<f64> = exchange_integral_loop_ao_lect(
                        pair.properties.s().unwrap(),
                        pair.properties.gamma_lr_ao().unwrap(),
                        n_orbs_i,
                        n_orbs_j,
                        false,
                    );
                    coulomb_integral = 2.0 * coulomb_integral - exchange_integral;
                    let coulomb_integral: Array2<f64> = coulomb_integral
                        .into_shape([n_orbs_j * n_orbs_j, n_orbs_j * n_orbs_i])
                        .unwrap();

                    let c_j_ind: ArrayView1<f64> = c_mo_i.slice(s![.., j.hole.mo.index]);
                    let c_i_ind: ArrayView1<f64> = c_mo_j.slice(s![.., j.electron.mo.index]);
                    let derivative_c_j: ArrayView2<f64> =
                        dc_mo_j.slice(s![.., j.hole.mo.index, ..]);

                    // calculate (d/dr tdm_i * C^J_hole * C^I_elec + tdm_i * d/dr C^J_hole * C^I_elec + tdm_i * C^J_hole * d/dr C^I_elec)
                    let term_1: Array1<f64> = cphf_tdm_j.dot(
                        &coulomb_integral.dot(
                            &(into_col(c_j_ind.to_owned()).dot(&into_row(c_i_ind.to_owned())))
                                .into_shape([n_orbs_j * n_orbs_i])
                                .unwrap(),
                        ),
                    );
                    let term_2: Array1<f64> = derivative_c_j.dot(
                        &tdm_j
                            .dot(&coulomb_integral)
                            .into_shape([n_orbs_j, n_orbs_i])
                            .unwrap()
                            .dot(&c_i_ind),
                    );
                    let term_3: Array1<f64> = c_j_ind
                        .dot(
                            &tdm_j
                                .dot(&coulomb_integral)
                                .into_shape([n_orbs_j, n_orbs_i])
                                .unwrap(),
                        )
                        .dot(&dc_mo_i.t());

                    gradient = gradient + term_1 + term_2 + term_3;

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

                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        self.monomers[pair.i].slice.atom_as_range(),
                        self.monomers[pair.j].slice.atom_as_range(),
                    );
                    let atoms_i = self.monomers[pair.i].n_atoms;

                    // set necessary arrays for the U matrix calculations
                    let monomers: &mut Vec<Monomer> = &mut self.monomers;
                    let monomer: &mut Monomer = &mut monomers[pair.i];
                    monomer.prepare_u_matrix(&pair_atoms[..atoms_i]);
                    let monomer: &mut Monomer = &mut monomers[pair.j];
                    monomer.prepare_u_matrix(&pair_atoms[atoms_i..]);
                    let monomers: usize;

                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];
                    let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i: usize = m_i.n_orbs;
                    let orbs_j: usize = m_j.n_orbs;

                    // calculate S,dS, gamma_AO and dgamma_AO of the pair
                    pair.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);

                    // MO coefficients of the virtual orbitals of I in 2D
                    let c_mat_virts: Array2<f64> =
                        into_col(i.mo.c.to_owned()).dot(&into_row(k.mo.c.to_owned()));
                    // MO coefficients of the occupied orbitals of J in 2D
                    let c_mat_occs: Array2<f64> =
                        into_col(j.mo.c.to_owned()).dot(&into_row(l.mo.c.to_owned()));

                    // reference to the mo coefficients of fragment I
                    let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
                    // reference to the mo coefficients of fragment J
                    let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
                    // calculate the U matrix of both monomers using the CPHF equations
                    let umat_i: Array3<f64> = m_i.calculate_u_matrix(&pair_atoms[..m_i.n_atoms]);
                    let umat_j: Array3<f64> = m_j.calculate_u_matrix(&pair_atoms[m_i.n_atoms..]);

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

                        let coulomb_integral: Array5<f64> = f_coulomb_loop(
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                        );
                        let coulomb_grad: Array1<f64> = coulomb_integral
                            .into_shape([3 * n_atoms * orbs_i * orbs_i, orbs_j * orbs_j])
                            .unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_j]).unwrap())
                            .into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());

                        // println!("coulomb gradient: {}",coulomb_gradient.slice(s![0..10]));
                        // println!("coulomb grad loop: {}",coulomb_grad.slice(s![0..10]));
                        assert!(
                            coulomb_gradient.abs_diff_eq(&coulomb_grad, 1e-14),
                            "Coulomb gradients are NOT equal!"
                        );

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

                        let exchange_integral: Array5<f64> = f_exchange_loop(
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                        );
                        let exchange_grad: Array1<f64> = exchange_integral
                            .into_shape([3 * n_atoms * orbs_i * orbs_i, orbs_j * orbs_j])
                            .unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_j * orbs_j]).unwrap())
                            .into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_i * orbs_i]).unwrap());
                        // println!("exchange gradient: {}",exchange_gradient.slice(s![0..10]));
                        // println!("exchange grad loop: {}",exchange_grad.slice(s![0..10]));
                        assert!(
                            exchange_gradient.abs_diff_eq(&exchange_grad, 1e-14),
                            "Exchange gradients are NOT equal!"
                        );

                        // Assemble the gradient
                        let mut gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;

                        // calculate gradients of the MO coefficients
                        // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
                        let mut dc_mo_i: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);
                        let mut dc_mo_k: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);
                        let mut dc_mo_j: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);
                        let mut dc_mo_l: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);
                        // iterate over gradient dimensions of both monomers
                        for nat in 0..3 * m_i.n_atoms {
                            dc_mo_i
                                .slice_mut(s![nat, ..])
                                .assign(&umat_i.slice(s![nat, .., i.mo.index]).dot(&c_mo_i.t()));
                            dc_mo_k
                                .slice_mut(s![nat, ..])
                                .assign(&umat_i.slice(s![nat, .., k.mo.index]).dot(&c_mo_i.t()));
                        }
                        for nat in 0..3 * m_j.n_atoms {
                            dc_mo_j
                                .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                                .assign(&umat_j.slice(s![nat, .., j.mo.index]).dot(&c_mo_j.t()));
                            dc_mo_l
                                .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                                .assign(&umat_j.slice(s![nat, .., l.mo.index]).dot(&c_mo_j.t()));
                        }

                        // calculate coulomb and exchange integrals in AO basis
                        let mut coulomb_arr: Array4<f64> = coulomb_integral_loop_ao(
                            m_i.properties.s().unwrap(),
                            m_j.properties.s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            m_i.n_orbs,
                            m_j.n_orbs,
                        );
                        let exchange_arr: Array4<f64> = exchange_integral_loop_ao(
                            pair.properties.s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            m_i.n_orbs,
                            m_j.n_orbs,
                        );
                        coulomb_arr = 2.0 * exchange_arr - coulomb_arr;

                        let mut cphf_grad: Array1<f64> = Array1::zeros(3 * pair.n_atoms);
                        // calculate loop version of cphf coulomb gradient
                        let c_i_ind: ArrayView1<f64> = c_mo_i.slice(s![.., i.mo.index]);
                        let c_k_ind: ArrayView1<f64> = c_mo_i.slice(s![.., k.mo.index]);
                        let c_j_ind: ArrayView1<f64> = c_mo_j.slice(s![.., j.mo.index]);
                        let c_l_ind: ArrayView1<f64> = c_mo_j.slice(s![.., l.mo.index]);
                        // for nat in 0..3*pair.n_atoms{
                        //     for mu in 0..m_i.n_orbs{
                        //         for la in 0..m_i.n_orbs{
                        //             for nu in 0..m_j.n_orbs{
                        //                 for sig in 0..m_j.n_orbs{
                        //                     coulomb_grad[nat] += coulomb_arr[[mu,la,nu,sig]] *
                        //                         (dc_mo_i[[nat,mu]] * c_k_ind[la] * c_j_ind[nu]*c_l_ind[sig]
                        //                             + dc_mo_k[[nat,la]] * c_i_ind[mu] * c_j_ind[nu]*c_l_ind[sig]
                        //                             + dc_mo_j[[nat,nu]] * c_i_ind[mu]*c_k_ind[la]*c_l_ind[sig]
                        //                             + dc_mo_l[[nat,sig]] * c_i_ind[mu]*c_k_ind[la]*c_j_ind[nu]);
                        //                 }
                        //             }
                        //         }
                        //     }
                        // }

                        let coulomb_arr: Array2<f64> = coulomb_arr
                            .into_shape([m_i.n_orbs * m_i.n_orbs, m_j.n_orbs * m_j.n_orbs])
                            .unwrap();
                        let c_mat_jl: Array2<f64> =
                            into_col(c_j_ind.to_owned()).dot(&into_row(c_l_ind.to_owned()));
                        let c_mat_ik: Array2<f64> =
                            into_col(c_i_ind.to_owned()).dot(&into_row(c_k_ind.to_owned()));

                        // let mut coulomb_grad_dot:Array1<f64> = Array1::zeros(3*pair.n_atoms);
                        // calculate dot version of cphf coulomb gradient
                        // iterate over the gradient
                        for nat in 0..3 * pair.n_atoms {
                            // dot product of dc_mu,i/dr c_lambda,i to c_mu,lambda of Fragment I
                            let c_i: Array2<f64> = into_col(dc_mo_i.slice(s![nat, ..]).to_owned())
                                .dot(&into_row(c_k_ind.to_owned()));
                            let c_i_2: Array2<f64> = into_col(c_i_ind.to_owned())
                                .dot(&into_row(dc_mo_k.slice(s![nat, ..]).to_owned()));
                            // dot product of dc_nu,a/dr c_sig,a to c_nu,sig of Fragment J
                            let c_j: Array2<f64> = into_col(dc_mo_j.slice(s![nat, ..]).to_owned())
                                .dot(&into_row(c_l_ind.to_owned()));
                            let c_j_2: Array2<f64> = into_col(c_j_ind.to_owned())
                                .dot(&into_row(dc_mo_l.slice(s![nat, ..]).to_owned()));

                            // calculate dot product of coulomb integral with previously calculated coefficients
                            // in AO basis
                            let term_1a = c_i.into_shape(m_i.n_orbs * m_i.n_orbs).unwrap().dot(
                                &coulomb_arr.dot(
                                    &c_mat_jl.view().into_shape(m_j.n_orbs * m_j.n_orbs).unwrap(),
                                ),
                            );
                            let term_1b = c_i_2.into_shape(m_i.n_orbs * m_i.n_orbs).unwrap().dot(
                                &coulomb_arr.dot(
                                    &c_mat_jl.view().into_shape(m_j.n_orbs * m_j.n_orbs).unwrap(),
                                ),
                            );
                            let term_2a = c_mat_ik
                                .view()
                                .into_shape(m_i.n_orbs * m_i.n_orbs)
                                .unwrap()
                                .dot(
                                    &coulomb_arr
                                        .dot(&c_j.into_shape(m_j.n_orbs * m_j.n_orbs).unwrap()),
                                );
                            let term_2b = c_mat_ik
                                .view()
                                .into_shape(m_i.n_orbs * m_i.n_orbs)
                                .unwrap()
                                .dot(
                                    &coulomb_arr
                                        .dot(&c_j_2.into_shape(m_j.n_orbs * m_j.n_orbs).unwrap()),
                                );

                            cphf_grad[nat] = term_1a + term_1b + term_2a + term_2b;
                        }
                        // assert!(coulomb_grad_dot.abs_diff_eq(&coulomb_grad,1.0e-11));

                        // add the cphf gradient to the gradient
                        gradient = gradient + cphf_grad;

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

                        let coulomb_integral: Array5<f64> = f_coulomb_loop(
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            pair.properties.grad_gamma_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                        );
                        let coulomb_grad: Array1<f64> = coulomb_integral
                            .into_shape([3 * n_atoms * orbs_i * orbs_i, orbs_j * orbs_j])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_j * orbs_j]).unwrap())
                            .into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_i * orbs_i]).unwrap());

                        // println!("coulomb gradient: {}",coulomb_gradient.slice(s![0..10]));
                        // println!("coulomb grad loop: {}",coulomb_grad.slice(s![0..10]));
                        assert!(
                            coulomb_gradient.abs_diff_eq(&coulomb_grad, 1e-14),
                            "Coulomb gradients are NOT equal!"
                        );

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

                        let exchange_integral: Array5<f64> = f_exchange_loop(
                            pair.properties.s().unwrap(),
                            pair.properties.grad_s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            pair.properties.grad_gamma_lr_ao().unwrap(),
                            n_atoms,
                            orbs_i,
                            orbs_j,
                        );
                        let exchange_grad: Array1<f64> = exchange_integral
                            .into_shape([3 * n_atoms * orbs_i * orbs_i, orbs_j * orbs_j])
                            .unwrap()
                            .dot(&c_mat_virts.view().into_shape([orbs_j * orbs_j]).unwrap())
                            .into_shape([3 * n_atoms, orbs_i * orbs_i])
                            .unwrap()
                            .dot(&c_mat_occs.view().into_shape([orbs_i * orbs_i]).unwrap());
                        // println!("exchange gradient: {}",exchange_gradient.slice(s![0..10]));
                        // println!("exchange grad loop: {}",exchange_grad.slice(s![0..10]));
                        assert!(
                            exchange_gradient.abs_diff_eq(&exchange_grad, 1e-14),
                            "Exchange gradients are NOT equal!"
                        );

                        // Assemble the gradient
                        let mut gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;

                        // calculate gradients of the MO coefficients
                        // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
                        let mut dc_mo_i: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);
                        let mut dc_mo_k: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);
                        let mut dc_mo_j: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);
                        let mut dc_mo_l: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);
                        // iterate over gradient dimensions of both monomers
                        for nat in 0..3 * m_j.n_atoms {
                            dc_mo_i
                                .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                                .assign(&umat_i.slice(s![nat, .., i.mo.index]).dot(&c_mo_j.t()));
                            dc_mo_k
                                .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                                .assign(&umat_i.slice(s![nat, .., k.mo.index]).dot(&c_mo_j.t()));
                        }
                        for nat in 0..3 * m_i.n_atoms {
                            dc_mo_j
                                .slice_mut(s![nat, ..])
                                .assign(&umat_j.slice(s![nat, .., j.mo.index]).dot(&c_mo_i.t()));
                            dc_mo_l
                                .slice_mut(s![nat, ..])
                                .assign(&umat_j.slice(s![nat, .., l.mo.index]).dot(&c_mo_i.t()));
                        }

                        // calculate coulomb and exchange integrals in AO basis
                        let mut coulomb_arr: Array4<f64> = coulomb_integral_loop_ao(
                            m_i.properties.s().unwrap(),
                            m_j.properties.s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            m_i.n_orbs,
                            m_j.n_orbs,
                        );
                        let exchange_arr: Array4<f64> = exchange_integral_loop_ao(
                            pair.properties.s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            m_i.n_orbs,
                            m_j.n_orbs,
                        );
                        coulomb_arr = 2.0 * exchange_arr - coulomb_arr;

                        let mut cphf_grad: Array1<f64> = Array1::zeros(3 * pair.n_atoms);
                        // calculate loop version of cphf coulomb gradient
                        let c_i_ind: ArrayView1<f64> = c_mo_i.slice(s![.., i.mo.index]);
                        let c_k_ind: ArrayView1<f64> = c_mo_i.slice(s![.., k.mo.index]);
                        let c_j_ind: ArrayView1<f64> = c_mo_j.slice(s![.., j.mo.index]);
                        let c_l_ind: ArrayView1<f64> = c_mo_j.slice(s![.., l.mo.index]);
                        // for nat in 0..3*pair.n_atoms{
                        //     for mu in 0..m_i.n_orbs{
                        //         for la in 0..m_i.n_orbs{
                        //             for nu in 0..m_j.n_orbs{
                        //                 for sig in 0..m_j.n_orbs{
                        //                     coulomb_grad[nat] += coulomb_arr[[mu,la,nu,sig]] *
                        //                         (dc_mo_i[[nat,mu]] * c_k_ind[la] * c_j_ind[nu]*c_l_ind[sig]
                        //                             + dc_mo_k[[nat,la]] * c_i_ind[mu] * c_j_ind[nu]*c_l_ind[sig]
                        //                             + dc_mo_j[[nat,nu]] * c_i_ind[mu]*c_k_ind[la]*c_l_ind[sig]
                        //                             + dc_mo_l[[nat,sig]] * c_i_ind[mu]*c_k_ind[la]*c_j_ind[nu]);
                        //                 }
                        //             }
                        //         }
                        //     }
                        // }

                        let coulomb_arr: Array2<f64> = coulomb_arr
                            .into_shape([m_i.n_orbs * m_i.n_orbs, m_j.n_orbs * m_j.n_orbs])
                            .unwrap();
                        let c_mat_jl: Array2<f64> =
                            into_col(c_j_ind.to_owned()).dot(&into_row(c_l_ind.to_owned()));
                        let c_mat_ik: Array2<f64> =
                            into_col(c_i_ind.to_owned()).dot(&into_row(c_k_ind.to_owned()));

                        // let mut coulomb_grad_dot:Array1<f64> = Array1::zeros(3*pair.n_atoms);
                        // calculate dot version of cphf coulomb gradient
                        // iterate over the gradient
                        for nat in 0..3 * pair.n_atoms {
                            // dot product of dc_mu,i/dr c_lambda,i to c_mu,lambda of Fragment I
                            let c_i: Array2<f64> = into_col(dc_mo_i.slice(s![nat, ..]).to_owned())
                                .dot(&into_row(c_k_ind.to_owned()));
                            let c_i_2: Array2<f64> = into_col(c_i_ind.to_owned())
                                .dot(&into_row(dc_mo_k.slice(s![nat, ..]).to_owned()));
                            // dot product of dc_nu,a/dr c_sig,a to c_nu,sig of Fragment J
                            let c_j: Array2<f64> = into_col(dc_mo_j.slice(s![nat, ..]).to_owned())
                                .dot(&into_row(c_l_ind.to_owned()));
                            let c_j_2: Array2<f64> = into_col(c_j_ind.to_owned())
                                .dot(&into_row(dc_mo_l.slice(s![nat, ..]).to_owned()));

                            // calculate dot product of coulomb integral with previously calculated coefficients
                            // in AO basis
                            let term_1a =
                                c_mat_jl
                                    .view()
                                    .into_shape(m_i.n_orbs * m_i.n_orbs)
                                    .unwrap()
                                    .dot(&coulomb_arr.dot(
                                        &c_i.view().into_shape(m_j.n_orbs * m_j.n_orbs).unwrap(),
                                    ));
                            let term_1b = c_mat_jl
                                .view()
                                .into_shape(m_i.n_orbs * m_i.n_orbs)
                                .unwrap()
                                .dot(&coulomb_arr.dot(
                                    &c_i_2.view().into_shape(m_i.n_orbs * m_i.n_orbs).unwrap(),
                                ));
                            let term_2a = c_j
                                .view()
                                .into_shape(m_i.n_orbs * m_i.n_orbs)
                                .unwrap()
                                .dot(&coulomb_arr.dot(
                                    &c_mat_ik.view().into_shape(m_j.n_orbs * m_j.n_orbs).unwrap(),
                                ));
                            let term_2b = c_j_2
                                .view()
                                .into_shape(m_i.n_orbs * m_i.n_orbs)
                                .unwrap()
                                .dot(&coulomb_arr.dot(
                                    &c_mat_ik.view().into_shape(m_j.n_orbs * m_j.n_orbs).unwrap(),
                                ));

                            cphf_grad[nat] = term_1a + term_1b + term_2a + term_2b;
                        }
                        // assert!(coulomb_grad_dot.abs_diff_eq(&coulomb_grad,1.0e-11));

                        // add the cphf gradient to the gradient
                        gradient = gradient + cphf_grad;

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

                        -coulomb_gradient
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
                    let pair_atoms: Vec<Atom> = get_pair_slice(
                        &self.atoms,
                        self.monomers[pair.i].slice.atom_as_range(),
                        self.monomers[pair.j].slice.atom_as_range(),
                    );
                    let atoms_i = self.monomers[pair.i].n_atoms;

                    // set necessary arrays for the U matrix calculations
                    let monomers: &mut Vec<Monomer> = &mut self.monomers;
                    let monomer: &mut Monomer = &mut monomers[pair.i];
                    monomer.prepare_u_matrix(&pair_atoms[..atoms_i]);
                    let monomer: &mut Monomer = &mut monomers[pair.j];
                    monomer.prepare_u_matrix(&pair_atoms[atoms_i..]);
                    let monomers: usize;

                    // monomers
                    let m_i: &Monomer = &self.monomers[pair.i];
                    let m_j: &Monomer = &self.monomers[pair.j];
                    let n_atoms: usize = m_i.n_atoms + m_j.n_atoms;
                    let orbs_i: usize = m_i.n_orbs;
                    let orbs_j: usize = m_j.n_orbs;

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

                    // reference to the mo coefficients of fragment I
                    let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
                    // reference to the mo coefficients of fragment J
                    let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();
                    // calculate the U matrix of both monomers using the CPHF equations
                    let umat_i: Array3<f64> = m_i.calculate_u_matrix(&pair_atoms[..m_i.n_atoms]);
                    let umat_j: Array3<f64> = m_j.calculate_u_matrix(&pair_atoms[m_i.n_atoms..]);

                    let grad = if m_i.index == i.m_index {
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

                        // let mut gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;
                        let mut gradient: Array1<f64> = 2.0 * exchange_gradient;

                        println!("i index: {}", i.m_index);
                        println!("j index: {}", j.m_index);
                        println!("k index: {}", k.m_index);
                        println!("l index: {}", l.m_index);

                        // calculate gradients of the MO coefficients
                        // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
                        let mut dc_mo_i: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);
                        let mut dc_mo_j: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);
                        let mut dc_mo_k: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);
                        let mut dc_mo_l: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);
                        // iterate over gradient dimensions of both monomers
                        for nat in 0..3 * m_i.n_atoms {
                            dc_mo_i
                                .slice_mut(s![nat, ..])
                                .assign(&umat_i.slice(s![nat, .., i.mo.index]).dot(&c_mo_i.t()));
                            dc_mo_l
                                .slice_mut(s![nat, ..])
                                .assign(&umat_i.slice(s![nat, .., l.mo.index]).dot(&c_mo_i.t()));
                        }
                        for nat in 0..3 * m_j.n_atoms {
                            dc_mo_j
                                .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                                .assign(&umat_j.slice(s![nat, .., j.mo.index]).dot(&c_mo_j.t()));
                            dc_mo_k
                                .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                                .assign(&umat_j.slice(s![nat, .., k.mo.index]).dot(&c_mo_j.t()));
                        }

                        // calculate coulomb and exchange integrals in AO basis
                        let mut coulomb_arr: Array4<f64> = coulomb_integral_loop_ao_ijji(
                            pair.properties.s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            m_i.n_orbs,
                            m_j.n_orbs,
                            true,
                        );
                        let exchange_arr: Array4<f64> = exchange_integral_loop_ao_ijji(
                            pair.properties.s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            m_i.n_orbs,
                            m_j.n_orbs,
                            true,
                        );
                        // coulomb_arr = 2.0 * exchange_arr - coulomb_arr;
                        coulomb_arr = 2.0 * exchange_arr;

                        let mut cphf_grad: Array1<f64> = Array1::zeros(3 * pair.n_atoms);
                        // calculate loop version of cphf coulomb gradient
                        let c_i_ind: ArrayView1<f64> = c_mo_i.slice(s![.., i.mo.index]);
                        let c_j_ind: ArrayView1<f64> = c_mo_j.slice(s![.., j.mo.index]);
                        let c_k_ind: ArrayView1<f64> = c_mo_j.slice(s![.., k.mo.index]);
                        let c_l_ind: ArrayView1<f64> = c_mo_i.slice(s![.., l.mo.index]);

                        // (mu,la| nu,sig) = (jl,ik) (hole,hole| elec,elec)
                        // mu = j, la = l, nu = i, sig = k
                        for nat in 0..3 * pair.n_atoms {
                            for mu in 0..m_j.n_orbs {
                                for la in 0..m_i.n_orbs {
                                    for nu in 0..m_i.n_orbs {
                                        for sig in 0..m_j.n_orbs {
                                            cphf_grad[nat] += coulomb_arr[[mu, la, nu, sig]]
                                                * (dc_mo_i[[nat, nu]]
                                                    * c_k_ind[sig]
                                                    * c_j_ind[mu]
                                                    * c_l_ind[la]
                                                    + dc_mo_k[[nat, sig]]
                                                        * c_i_ind[nu]
                                                        * c_j_ind[mu]
                                                        * c_l_ind[la]
                                                    + dc_mo_j[[nat, mu]]
                                                        * c_i_ind[nu]
                                                        * c_k_ind[sig]
                                                        * c_l_ind[la]
                                                    + dc_mo_l[[nat, la]]
                                                        * c_i_ind[nu]
                                                        * c_k_ind[sig]
                                                        * c_j_ind[mu]);
                                        }
                                    }
                                }
                            }
                        }

                        // let coulomb_arr:Array2<f64> =  coulomb_arr
                        //     .into_shape([m_i.n_orbs*m_i.n_orbs,m_j.n_orbs*m_j.n_orbs]).unwrap();
                        // let c_mat_il:Array2<f64> = into_col(c_i_ind.to_owned())
                        //     .dot(&into_row(c_l_ind.to_owned()));
                        // let c_mat_jk:Array2<f64> = into_col(c_j_ind.to_owned())
                        //     .dot(&into_row(c_k_ind.to_owned()));
                        //
                        // // iterate over the gradient
                        // for nat in 0..3*pair.n_atoms{
                        //     // dot product of dc_mu,i/dr c_lambda,i to c_mu,lambda of Fragment I
                        //     let c_il:Array2<f64> = into_col(dc_mo_i.slice(s![nat,..]).to_owned())
                        //         .dot(&into_row(c_l_ind.to_owned()));
                        //     let c_il_2:Array2<f64> = into_col(c_i_ind.to_owned())
                        //         .dot(&into_row(dc_mo_l.slice(s![nat,..]).to_owned()));
                        //     // dot product of dc_nu,a/dr c_sig,a to c_nu,sig of Fragment J
                        //     let c_jk:Array2<f64> = into_col(dc_mo_j.slice(s![nat,..]).to_owned())
                        //         .dot(&into_row(c_k_ind.to_owned()));
                        //     let c_jk_2:Array2<f64> = into_col(c_j_ind.to_owned())
                        //         .dot(&into_row(dc_mo_k.slice(s![nat,..]).to_owned()));
                        //
                        //     // calculate dot product of coulomb integral with previously calculated coefficients
                        //     // in AO basis
                        //     let term_1a = c_il.into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                        //         .dot(&coulomb_arr.dot(&c_mat_jk.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                        //     let term_1b = c_il_2.into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                        //         .dot(&coulomb_arr.dot(&c_mat_jk.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                        //     let term_2a = c_mat_il.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                        //         .dot(&coulomb_arr.dot(&c_jk.into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                        //     let term_2b = c_mat_il.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                        //         .dot(&coulomb_arr.dot(&c_jk_2.into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                        //
                        //     cphf_grad[nat] = term_1a + term_1b + term_2a + term_2b;
                        // }

                        // add the cphf gradient to the gradient
                        gradient = gradient + cphf_grad;

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

                        // let mut gradient: Array1<f64> = 2.0 * exchange_gradient - coulomb_gradient;
                        let mut gradient: Array1<f64> = 2.0 * exchange_gradient;

                        // calculate gradients of the MO coefficients
                        // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
                        let mut dc_mo_i: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);
                        let mut dc_mo_j: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);
                        let mut dc_mo_k: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_i.n_orbs]);
                        let mut dc_mo_l: Array2<f64> =
                            Array2::zeros([3 * pair.n_atoms, m_j.n_orbs]);
                        // iterate over gradient dimensions of both monomers
                        for nat in 0..3 * m_j.n_atoms {
                            dc_mo_i
                                .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                                .assign(&umat_j.slice(s![nat, .., i.mo.index]).dot(&c_mo_j.t()));
                            dc_mo_l
                                .slice_mut(s![3 * m_i.n_atoms + nat, ..])
                                .assign(&umat_j.slice(s![nat, .., l.mo.index]).dot(&c_mo_j.t()));
                        }
                        for nat in 0..3 * m_i.n_atoms {
                            dc_mo_j
                                .slice_mut(s![nat, ..])
                                .assign(&umat_i.slice(s![nat, .., j.mo.index]).dot(&c_mo_i.t()));
                            dc_mo_k
                                .slice_mut(s![nat, ..])
                                .assign(&umat_i.slice(s![nat, .., k.mo.index]).dot(&c_mo_i.t()));
                        }

                        // calculate coulomb and exchange integrals in AO basis
                        let mut coulomb_arr: Array4<f64> = coulomb_integral_loop_ao_ijji(
                            pair.properties.s().unwrap(),
                            pair.properties.gamma_ao().unwrap(),
                            m_i.n_orbs,
                            m_j.n_orbs,
                            false,
                        );
                        let exchange_arr: Array4<f64> = exchange_integral_loop_ao_ijji(
                            pair.properties.s().unwrap(),
                            pair.properties.gamma_lr_ao().unwrap(),
                            m_i.n_orbs,
                            m_j.n_orbs,
                            false,
                        );
                        // coulomb_arr = 2.0 * exchange_arr - coulomb_arr;
                        coulomb_arr = 2.0 * exchange_arr;

                        let mut cphf_grad: Array1<f64> = Array1::zeros(3 * pair.n_atoms);
                        // calculate loop version of cphf coulomb gradient
                        let c_i_ind: ArrayView1<f64> = c_mo_j.slice(s![.., i.mo.index]);
                        let c_j_ind: ArrayView1<f64> = c_mo_i.slice(s![.., j.mo.index]);
                        let c_k_ind: ArrayView1<f64> = c_mo_i.slice(s![.., k.mo.index]);
                        let c_l_ind: ArrayView1<f64> = c_mo_j.slice(s![.., l.mo.index]);

                        // (mu,la| nu,sig) = (jl,ik) (hole,hole| elec,elec)
                        // mu = j, la = l, nu = i, sig = k
                        for nat in 0..3 * pair.n_atoms {
                            for mu in 0..m_i.n_orbs {
                                for la in 0..m_j.n_orbs {
                                    for nu in 0..m_j.n_orbs {
                                        for sig in 0..m_i.n_orbs {
                                            cphf_grad[nat] += coulomb_arr[[mu, la, nu, sig]]
                                                * (dc_mo_i[[nat, nu]]
                                                    * c_k_ind[sig]
                                                    * c_j_ind[mu]
                                                    * c_l_ind[la]
                                                    + dc_mo_k[[nat, sig]]
                                                        * c_i_ind[nu]
                                                        * c_j_ind[mu]
                                                        * c_l_ind[la]
                                                    + dc_mo_j[[nat, mu]]
                                                        * c_i_ind[nu]
                                                        * c_k_ind[sig]
                                                        * c_l_ind[la]
                                                    + dc_mo_l[[nat, la]]
                                                        * c_i_ind[nu]
                                                        * c_k_ind[sig]
                                                        * c_j_ind[mu]);
                                        }
                                    }
                                }
                            }
                        }

                        // let coulomb_arr:Array2<f64> =  coulomb_arr
                        //     .into_shape([m_i.n_orbs*m_i.n_orbs,m_j.n_orbs*m_j.n_orbs]).unwrap();
                        // let c_mat_il:Array2<f64> = into_col(c_i_ind.to_owned())
                        //     .dot(&into_row(c_l_ind.to_owned()));
                        // let c_mat_jk:Array2<f64> = into_col(c_j_ind.to_owned())
                        //     .dot(&into_row(c_k_ind.to_owned()));
                        //
                        // // iterate over the gradient
                        // for nat in 0..3*pair.n_atoms{
                        //     // dot product of dc_mu,i/dr c_lambda,i to c_mu,lambda of Fragment I
                        //     let c_il:Array2<f64> = into_col(dc_mo_i.slice(s![nat,..]).to_owned())
                        //         .dot(&into_row(c_l_ind.to_owned()));
                        //     let c_il_2:Array2<f64> = into_col(c_i_ind.to_owned())
                        //         .dot(&into_row(dc_mo_l.slice(s![nat,..]).to_owned()));
                        //     // dot product of dc_nu,a/dr c_sig,a to c_nu,sig of Fragment J
                        //     let c_jk:Array2<f64> = into_col(dc_mo_j.slice(s![nat,..]).to_owned())
                        //         .dot(&into_row(c_k_ind.to_owned()));
                        //     let c_jk_2:Array2<f64> = into_col(c_j_ind.to_owned())
                        //         .dot(&into_row(dc_mo_k.slice(s![nat,..]).to_owned()));
                        //
                        //     // calculate dot product of coulomb integral with previously calculated coefficients
                        //     // in AO basis
                        //     let term_1a = c_mat_jk.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                        //         .dot(&coulomb_arr.dot(&c_il.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                        //     let term_1b = c_mat_jk.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                        //         .dot(&coulomb_arr.dot(&c_il_2.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()));
                        //     let term_2a = c_jk.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                        //         .dot(&coulomb_arr.dot(&c_mat_il.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                        //     let term_2b = c_jk_2.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                        //         .dot(&coulomb_arr.dot(&c_mat_il.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                        //
                        //     cphf_grad[nat] = term_1a + term_1b + term_2a + term_2b;
                        // }

                        // add the cphf gradient to the gradient
                        gradient = gradient + cphf_grad;

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
