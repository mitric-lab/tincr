use crate::excited_states::trans_charges;
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::{Monomer, Pair, SuperSystem, GroundStateGradient, PairType, ESDPair, ExcitedStateMonomerGradient};
use crate::gradients::helpers::{f_lr, f_v};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{
    gamma_ao_wise, gamma_ao_wise_from_gamma_atomwise, gamma_atomwise_ab, gamma_gradients_ao_wise,
};
use crate::scc::h0_and_s::{h0_and_s_ab, h0_and_s_gradients};
use ndarray::prelude::*;
use ndarray_linalg::trace::Trace;
use std::ops::{AddAssign, SubAssign};
use ndarray_linalg::{into_row, into_col, Inverse};
use std::time::Instant;
use ndarray::parallel::prelude::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

impl SuperSystem{
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
        // and the U matrix of the CPHF equations
        let tmp = m_i.calculate_ct_fock_gradient(atoms_i, ct_ind_i, true);
        let mut gradh_i = -1.0* tmp.0;
        let u_mat_i:Array3<f64> = tmp.1;

        let tmp = m_j.calculate_ct_fock_gradient(atoms_j, ct_ind_j, false);
        let gradh_j = tmp.0;
        let u_mat_j:Array3<f64> = tmp.1;
        println!("grad i {}",gradh_i.slice(s![0..4]));
        println!("grad j {}",gradh_j.slice(s![0..4]));
        // assert!(1==2);

        // append the fock matrix gradients
        gradh_i.append(Axis(0), gradh_j.view()).unwrap();

        // reference to the mo coefficients of fragment I
        let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
        // reference to the mo coefficients of fragment J
        let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

        let occ_indices_i: &[usize] = m_i.properties.occ_indices().unwrap();
        let virt_indices_j: &[usize] = m_j.properties.virt_indices().unwrap();
        let nocc_i:usize = occ_indices_i.len();
        let orb_ind_i:usize = occ_indices_i[nocc_i -1 -ct_ind_i];
        let orb_ind_j:usize = virt_indices_j[ct_ind_j];
        let c_mat_j:Array2<f64> = into_col(c_mo_j.slice(s![..,orb_ind_j]).to_owned()).dot(&into_row(c_mo_j.slice(s![..,orb_ind_j]).to_owned()));
        let c_mat_i:Array2<f64> = into_col(c_mo_i.slice(s![..,orb_ind_i]).to_owned()).dot(&into_row(c_mo_i.slice(s![..,orb_ind_i]).to_owned()));

        // assert_eq!(c_mat_j,c_mo_j);
        let pair_type:PairType = self.properties.type_of_pair(index_i, index_j);

        // differentiate between pair types
        let return_gradient:Array1<f64> = if pair_type == PairType::Pair{
            // get pair index
            let pair_index:usize = self.properties.index_of_pair(index_i,index_j);
            // get correct pair from pairs vector
            let pair_ij: &mut Pair = &mut self.pairs[pair_index];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            pair_ij.prepare_lcmo_gradient(&pair_atoms,m_i,m_j);
            let pair_grad_s:ArrayView3<f64> = pair_ij.properties.grad_s().unwrap();
            let grad_s_i: ArrayView3<f64> = pair_grad_s.slice(s![.., ..n_orbs_i, ..n_orbs_i]);
            let grad_s_j: ArrayView3<f64> = pair_grad_s.slice(s![.., n_orbs_i.., n_orbs_i..]);

            let coulomb_integral:Array5<f64> = f_v_ct_2(
                m_i.properties.s().unwrap(),
                m_j.properties.s().unwrap(),
                grad_s_i,
                grad_s_j,
                pair_ij.properties.gamma_ao().unwrap(),
                pair_ij.properties.grad_gamma_ao().unwrap(),
                pair_ij.n_atoms,
                n_orbs_i,
                n_orbs_j
            );
            let coulomb_grad:Array1<f64> = coulomb_integral.into_shape([3*pair_ij.n_atoms*n_orbs_i*n_orbs_i,n_orbs_j*n_orbs_j]).unwrap()
                .dot(&c_mat_j.view().into_shape([n_orbs_j * n_orbs_j]).unwrap()).into_shape([3*pair_ij.n_atoms,n_orbs_i*n_orbs_i]).unwrap()
                .dot(&c_mat_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

            let coulomb_gradient: Array1<f64> = f_v_ct(
                c_mat_j.view(),
                m_i.properties.s().unwrap(),
                m_j.properties.s().unwrap(),
                grad_s_i,
                grad_s_j,
                pair_ij.properties.gamma_ao().unwrap(),
                pair_ij.properties.grad_gamma_ao().unwrap(),
                pair_ij.n_atoms,
                n_orbs_i,
            )
                .into_shape([3 * pair_ij.n_atoms, n_orbs_i * n_orbs_i])
                .unwrap()
                .dot(&c_mat_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());
            // .dot(&c_mo_i.into_shape([n_orbs_i * n_orbs_i]).unwrap());

            assert!(coulomb_grad.abs_diff_eq(&coulomb_gradient,1.0e-12));

            let exchange_gradient: Array1<f64> = f_lr_ct(
                c_mat_j.t(),
                pair_ij.properties.s().unwrap(),
                pair_ij.properties.grad_s().unwrap(),
                pair_ij.properties.gamma_lr_ao().unwrap(),
                pair_ij.properties.grad_gamma_lr_ao().unwrap(),
                m_i.n_atoms,
                m_j.n_atoms,
                n_orbs_i,
            )
                .into_shape([3 * pair_ij.n_atoms, n_orbs_i * n_orbs_i])
                .unwrap()
                .dot(&c_mat_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());
            // .dot(&c_mo_i.into_shape([n_orbs_i * n_orbs_i]).unwrap());

            // calculate gradients of the MO coefficients
            // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
            let mut dc_mo_i:Array2<f64> = Array2::zeros([3*pair_ij.n_atoms,m_i.n_orbs]);
            let mut dc_mo_j:Array2<f64> = Array2::zeros([3*pair_ij.n_atoms,m_j.n_orbs]);

            // iterate over gradient dimensions of both monomers
            for nat in 0..3*m_i.n_atoms{
                dc_mo_i.slice_mut(s![nat,..]).assign(&u_mat_i.slice(s![nat,..,orb_ind_i]).dot(&c_mo_i.t()));
            }
            for nat in 0..3*m_j.n_atoms{
                dc_mo_j.slice_mut(s![3*m_i.n_atoms+nat,..]).assign(&u_mat_j.slice(s![nat,..,orb_ind_j]).dot(&c_mo_j.t()));
            }

            // calculate coulomb and exchange integrals in AO basis
            let coulomb_arr:Array4<f64> = f_v_coulomb_loop(
                m_i.properties.s().unwrap(),
                m_j.properties.s().unwrap(),
                pair_ij.properties.gamma_ao().unwrap(),
                m_i.n_orbs,
                m_j.n_orbs
            );//.into_shape([m_i.n_orbs*m_i.n_orbs,m_j.n_orbs*m_j.n_orbs]).unwrap();

            let mut coulomb_grad:Array1<f64> = Array1::zeros(3*pair_ij.n_atoms);
            // calculate loop version of cphf coulomb gradient
            let c_i_ind:ArrayView1<f64> = c_mo_i.slice(s![..,orb_ind_i]);
            let c_j_ind:ArrayView1<f64> = c_mo_j.slice(s![..,orb_ind_j]);
            for nat in 0..3*pair_ij.n_atoms{
                for mu in 0..m_i.n_orbs{
                    for la in 0..m_i.n_orbs{
                        for nu in 0..m_j.n_orbs{
                            for sig in 0..m_j.n_orbs{
                                coulomb_grad[nat] += coulomb_arr[[mu,la,nu,sig]] *
                                    (dc_mo_i[[nat,mu]] * c_i_ind[la] * c_j_ind[nu]*c_j_ind[sig]
                                        + dc_mo_i[[nat,la]] * c_i_ind[mu] * c_j_ind[nu]*c_j_ind[sig]
                                        + dc_mo_j[[nat,nu]] * c_i_ind[mu]*c_i_ind[la]*c_j_ind[sig]
                                        + dc_mo_j[[nat,sig]] * c_i_ind[mu]*c_i_ind[la]*c_j_ind[nu]);
                            }
                        }
                    }
                }
            }

            let coulomb_arr:Array2<f64> =  coulomb_arr.into_shape([m_i.n_orbs*m_i.n_orbs,m_j.n_orbs*m_j.n_orbs]).unwrap();

            // let mut coulomb_grad_2:Array1<f64> = Array1::zeros(3*pair_ij.n_atoms);
            // // calculate dot version of cphf coulomb gradient
            // // iterate over the gradient
            // for nat in 0..3*pair_ij.n_atoms{
            //     // dot product of dc_mu,i/dr c_lambda,i to c_mu,lambda of Fragment I
            //     let c_i:Array2<f64> = into_col(dc_mo_i.slice(s![nat,..]).to_owned())
            //         .dot(&into_row(c_mo_i.slice(s![..,orb_ind_i]).to_owned()));
            //     let c_i_2:Array2<f64> = into_col(c_mo_i.slice(s![..,orb_ind_i]).to_owned())
            //         .dot(&into_row(dc_mo_i.slice(s![nat,..]).to_owned()));
            //     // dot product of dc_nu,a/dr c_sig,a to c_nu,sig of Fragment J
            //     let c_j:Array2<f64> = into_col(dc_mo_j.slice(s![nat,..]).to_owned())
            //         .dot(&into_row(c_mo_j.slice(s![..,orb_ind_j]).to_owned()));
            //     let c_j_2:Array2<f64> = into_col(c_mo_j.slice(s![..,orb_ind_j]).to_owned())
            //         .dot(&into_row(dc_mo_j.slice(s![nat,..]).to_owned()));
            //
            //     // calculate dot product of coulomb integral with previously calculated coefficients
            //     // in AO basis
            //     let term_1a = c_i.into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
            //         .dot(&coulomb_arr.dot(&c_mat_j.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
            //     let term_1b = c_i_2.into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
            //         .dot(&coulomb_arr.dot(&c_mat_j.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
            //     let term_2a = c_mat_i.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
            //         .dot(&coulomb_arr.dot(&c_j.into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
            //     let term_2b = c_mat_i.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
            //         .dot(&coulomb_arr.dot(&c_j_2.into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
            //
            //     coulomb_grad_2[nat] = term_1a + term_1b + term_2a + term_2b;
            // }
            // assert!(coulomb_grad_2.abs_diff_eq(&coulomb_grad,1.0e-12));
            // println!("cphf coulomb grad {}",coulomb_grad.slice(s![0..5]));
            //
            // // calculate the exchange and coumlob integral and multiply those by the
            // // MO coefficient derivatives
            //
            // // assemble the gradient
            // // println!("gradient of orbital energies {}",gradh_i);
            // println!("exchange gradient {}",exchange_gradient.slice(s![0..5]));
            // println!("coulomb gradient {}",coulomb_gradient.slice(s![0..5]));
            // println!("coulomb grad v2 {}",coulomb_grad.slice(s![0..5]));
            // let gradient: Array1<f64> = gradh_i + 2.0 * exchange_gradient - 1.0 * coulomb_gradient;
            // let gradient: Array1<f64> = gradh_i - 1.0 * coulomb_gradient;
            // let gradient: Array1<f64> = - 1.0 * (coulomb_gradient - coulomb_grad);
            let gradient:Array1<f64> = -coulomb_grad;
            // let gradient:Array1<f64> = gradh_i;

            gradient
        } else {
            // get pair index
            let pair_index:usize = self.properties.index_of_esd_pair(index_i,index_j);
            // get correct pair from pairs vector
            let pair_ij: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            let (grad_s_pair, grad_h0_pair) =
                h0_and_s_gradients(&pair_atoms, pair_ij.n_orbs, &self.pairs[0].slako);
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
                    &self.pairs[0].gammafunction,
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
            }

            // calculate the gamma matrix in AO basis
            let g0_ao: Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
                pair_ij.properties.gamma().unwrap(),
                &pair_atoms,
                pair_ij.n_orbs,
            );
            // calculate the gamma gradient matrix
            let (g1, g1_ao): (Array3<f64>, Array3<f64>) = gamma_gradients_ao_wise(
                &self.pairs[0].gammafunction,
                &pair_atoms,
                pair_ij.n_atoms,
                pair_ij.n_orbs,
            );

            let coulomb_gradient: Array1<f64> = f_v_ct(
                c_mat_j.view(),
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
                .dot(&c_mat_i.view().into_shape([n_orbs_i * n_orbs_i]).unwrap());

            // assemble the gradient
            let gradient: Array1<f64> = gradh_i - 1.0 * coulomb_gradient;

            gradient
        };

        return return_gradient;
    }
}

impl Monomer {
    pub fn prepare_u_matrix(&mut self,atoms:&[Atom]){
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
    }

    pub fn calculate_u_matrix(
        &self,
        atoms: &[Atom],
    )->Array3<f64>{
        let timer: Instant = Instant::now();
        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&atoms, self.n_orbs, &self.slako);

        // get necessary arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g0:ArrayView2<f64> = self.properties.gamma().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let orbe:ArrayView1<f64> = self.properties.orbe().unwrap();

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
        }

        // get MO coefficient for the occupied or virtual orbital
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();
        let nocc:usize = occ_indices.len();
        let nvirt:usize = virt_indices.len();
        let mut orbs_occ: Array2<f64> = Array::zeros((self.n_orbs, nocc));
        for (i, index) in occ_indices.iter().enumerate() {
            orbs_occ.slice_mut(s![.., i]).assign(&orbs.column(*index));
        }

        // calculate transition charges
        let (qov,qoo,qvv) = trans_charges(self.n_atoms,atoms,orbs,s,occ_indices,virt_indices);
        // virtual-occupied transition charges
        let qvo:Array2<f64> = qov.clone().into_shape([self.n_atoms,nocc,nvirt]).unwrap()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned().into_shape([self.n_atoms,nvirt*nocc]).unwrap();

        let g0_lr:ArrayView2<f64> = self.properties.gamma_lr().unwrap();
        // calculate B matrix B_ij with i = nvirt, j = nocc
        // for the CPHF iterations
        let mut b_mat:Array3<f64> = Array3::zeros([3*self.n_atoms,self.n_orbs,self.n_orbs]);

        // create orbital energy matrix
        let mut orbe_matrix:Array2<f64> = Array2::zeros((self.n_orbs,self.n_orbs));
        for mu in 0..self.n_orbs{
            for nu in 0..self.n_orbs{
                orbe_matrix[[mu,nu]] = orbe[nu];
            }
        }

        // Calculate integrals partwise before iteration over gradient
        // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
        let integral_vo_2d:Array2<f64> = (2.0 * qvo.t().dot(&g0.dot(&qoo)) - qvo.t().dot(&g0_lr.dot(&qoo)).into_shape([nvirt,nocc,nocc,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nocc*nocc]).unwrap());
        // integral (ij|kl) - (ik|jl), i = nocc, j = nocc
        let integral_oo_2d:Array2<f64> = (2.0 * qoo.t().dot(&g0.dot(&qoo)) -qoo.t().dot(&g0_lr.dot(&qoo)).into_shape([nocc,nocc,nocc,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nocc,nocc*nocc]).unwrap());
        // integral (ij|kl) - (ik|jl), i = nvirt, j = nvirt
        let integral_vv_2d:Array2<f64> = (2.0 * qvv.t().dot(&g0.dot(&qoo)) -qvo.t().dot(&g0_lr.dot(&qvo)).into_shape([nvirt,nocc,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nocc*nocc]).unwrap());
        // integral (ij|kl) - (ik|jl), i = nocc, j = nvirt
        let integral_ov_2d:Array2<f64> = (2.0 * qov.t().dot(&g0.dot(&qoo)) -qoo.t().dot(&g0_lr.dot(&qvo)).into_shape([nocc,nocc,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nocc*nocc]).unwrap());

        // Calculate the B matrix
        for nc in 0..3*self.n_atoms{
            let ds_mo:Array1<f64> = orbs_occ.t().dot(&grad_s.slice(s![nc,..,..])
                .dot(&orbs_occ)).into_shape([nocc*nocc]).unwrap();

            // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
            let integral_vo:Array2<f64> = integral_vo_2d.dot(&ds_mo).into_shape([nvirt,nocc]).unwrap();
            // integral (ij|kl) - (ik|jl), i = nocc, j = nocc
            let integral_oo:Array2<f64> = integral_oo_2d.dot(&ds_mo).into_shape([nocc,nocc]).unwrap();
            // integral (ij|kl) - (ik|jl), i = nvirt, j = nvirt
            let integral_vv:Array2<f64> = integral_vv_2d.dot(&ds_mo).into_shape([nvirt,nvirt]).unwrap();
            // integral (ij|kl) - (ik|jl), i = nocc, j = nvirt
            let integral_ov:Array2<f64> = integral_ov_2d.dot(&ds_mo).into_shape([nocc,nvirt]).unwrap();

            let gradh_mo:Array2<f64> = orbs.t().dot(&grad_h.slice(s![nc,..,..]).dot(&orbs));
            let grads_mo:Array2<f64> = orbs.t().dot(&grad_s.slice(s![nc,..,..]).dot(&orbs));

            // b_mat.slice_mut(s![nc,..,..]).assign(&(&gradh_mo - &grads_mo * &orbe_matrix));
            // b_mat.slice_mut(s![nc,..nocc,..nocc]).sub_assign(&integral_oo);
            // b_mat.slice_mut(s![nc,..nocc,nocc..]).sub_assign(&integral_ov);
            // b_mat.slice_mut(s![nc,nocc..,..nocc]).sub_assign(&integral_vo);
            // b_mat.slice_mut(s![nc,nocc..,nocc..]).sub_assign(&integral_vv);

            // loop version
            for i in 0..nvirt{
                for j in 0..nocc{
                    b_mat[[nc,nocc+i,j]] = gradh_mo[[nocc+i,j]] - grads_mo[[nocc+i,j]] * orbe[j] - integral_vo[[i,j]];
                }
                for j in 0..nvirt{
                    b_mat[[nc,nocc+i,nocc+j]] = gradh_mo[[nocc+i,nocc+j]] - grads_mo[[nocc+i,nocc+j]] * orbe[nocc+j] - integral_vv[[i,j]];
                }
            }
            for i in 0..nocc{
                for j in 0..nocc{
                    b_mat[[nc,i,j]] = gradh_mo[[i,j]] - grads_mo[[i,j]] * orbe[j] - integral_oo[[i,j]];
                }
                for j in 0..nvirt{
                    b_mat[[nc,i,nocc+j]] = gradh_mo[[i,nocc+j]] - grads_mo[[i,nocc+j]] * orbe[nocc+j] - integral_ov[[i,j]];
                }
            }
        }
        // assert!(b_mat.abs_diff_eq(&b_mat_2,1.0e-11));

        println!("Elapsed time calculate B matrix: {:>8.6}",timer.elapsed().as_secs_f64());

        // calculate A_matrix ij,kl i = nvirt, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vo:Array2<f64> = Array2::zeros([nvirt*nocc,nvirt*nocc]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_vo = a_mat_vo - qvv.t().dot(&g0_lr.dot(&qoo)).into_shape([nvirt,nvirt,nocc,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nvirt*nocc]).unwrap();
        // integral (il|jk)
        a_mat_vo = a_mat_vo - qvo.t().dot(&g0_lr.dot(&qov)).into_shape([nvirt,nocc,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nvirt*nocc]).unwrap();

        // calculate A_matrix ij,kl i = nocc, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_oo:Array2<f64> = Array2::zeros([nocc*nocc,nvirt*nocc]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_oo = a_mat_oo - qov.t().dot(&g0_lr.dot(&qoo)).into_shape([nocc,nvirt,nocc,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nocc,nvirt*nocc]).unwrap();
        // integral (il|jk)
        a_mat_oo = a_mat_oo - qoo.t().dot(&g0_lr.dot(&qov)).into_shape([nocc,nocc,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nocc,nvirt*nocc]).unwrap();

        // calculate A_matrix ij,kl i = nocc, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_ov:Array2<f64> = Array2::zeros([nocc*nvirt,nvirt*nocc]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_ov = a_mat_ov - qov.t().dot(&g0_lr.dot(&qvo)).into_shape([nocc,nvirt,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nvirt*nocc]).unwrap();
        // integral (il|jk)
        a_mat_ov = a_mat_ov - qoo.t().dot(&g0_lr.dot(&qvv)).into_shape([nocc,nocc,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nvirt*nocc]).unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vv:Array2<f64> = Array2::zeros([nvirt*nvirt,nvirt*nocc]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_vv = a_mat_vv - qvv.t().dot(&g0_lr.dot(&qvo)).into_shape([nvirt,nvirt,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nvirt*nocc]).unwrap();
        // integral (il|jk)
        a_mat_vv = a_mat_vv - qvo.t().dot(&g0_lr.dot(&qvv)).into_shape([nvirt,nocc,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nvirt*nocc]).unwrap();

        let mut a_matrix:Array4<f64> = Array4::zeros([self.n_orbs,self.n_orbs,self.n_orbs,self.n_orbs]);
        a_matrix.slice_mut(s![..nocc,..nocc,nocc..,..nocc]).assign(&a_mat_oo.into_shape([nocc,nocc,nvirt,nocc]).unwrap());
        a_matrix.slice_mut(s![..nocc,nocc..,nocc..,..nocc]).assign(&a_mat_ov.into_shape([nocc,nvirt,nvirt,nocc]).unwrap());
        a_matrix.slice_mut(s![nocc..,..nocc,nocc..,..nocc]).assign(&a_mat_vo.into_shape([nvirt,nocc,nvirt,nocc]).unwrap());
        a_matrix.slice_mut(s![nocc..,nocc..,nocc..,..nocc]).assign(&a_mat_vv.into_shape([nvirt,nvirt,nvirt,nocc]).unwrap());

        // let a_mat:Array2<f64> = a_matrix.into_shape([self.n_orbs*self.n_orbs,nvirt*nocc]).unwrap();

        let mut a_mat_vo:Array2<f64> = Array2::zeros([nvirt*nocc,nocc*nocc]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo.t().dot(&g0.dot(&qoo));
        // integral (ik|jl)
        a_mat_vo = a_mat_vo - qvo.t().dot(&g0_lr.dot(&qoo)).into_shape([nvirt,nocc,nocc,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nocc*nocc]).unwrap();
        // integral (il|jk)
        a_mat_vo = a_mat_vo - qvo.t().dot(&g0_lr.dot(&qoo)).into_shape([nvirt,nocc,nocc,nocc]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nocc*nocc]).unwrap();

        let mut a_mat_oo:Array2<f64> = Array2::zeros([nocc*nocc,nocc*nocc]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo.t().dot(&g0.dot(&qoo));
        // integral (ik|jl)
        a_mat_oo = a_mat_oo - qoo.t().dot(&g0_lr.dot(&qoo)).into_shape([nocc,nocc,nocc,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nocc,nocc*nocc]).unwrap();
        // integral (il|jk)
        a_mat_oo = a_mat_oo - qoo.t().dot(&g0_lr.dot(&qoo)).into_shape([nocc,nocc,nocc,nocc]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nocc,nocc*nocc]).unwrap();

        let mut a_mat_ov:Array2<f64> = Array2::zeros([nocc*nvirt,nocc*nocc]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov.t().dot(&g0.dot(&qoo));
        // integral (ik|jl)
        a_mat_ov = a_mat_ov - qoo.t().dot(&g0_lr.dot(&qvo)).into_shape([nocc,nocc,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nocc*nocc]).unwrap();
        // integral (il|jk)
        a_mat_ov = a_mat_ov - qoo.t().dot(&g0_lr.dot(&qvo)).into_shape([nocc,nocc,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nocc*nocc]).unwrap();

        let mut a_mat_vv:Array2<f64> = Array2::zeros([nvirt*nvirt,nocc*nocc]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv.t().dot(&g0.dot(&qoo));
        // integral (ik|jl)
        a_mat_vv = a_mat_vv - qvo.t().dot(&g0_lr.dot(&qvo)).into_shape([nvirt,nocc,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nocc*nocc]).unwrap();
        // integral (il|jk)
        a_mat_vv = a_mat_vv - qvo.t().dot(&g0_lr.dot(&qvo)).into_shape([nvirt,nocc,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nocc*nocc]).unwrap();

        a_matrix.slice_mut(s![..nocc,..nocc,..nocc,..nocc]).assign(&a_mat_oo.into_shape([nocc,nocc,nocc,nocc]).unwrap());
        a_matrix.slice_mut(s![..nocc,nocc..,..nocc,..nocc]).assign(&a_mat_ov.into_shape([nocc,nvirt,nocc,nocc]).unwrap());
        a_matrix.slice_mut(s![nocc..,..nocc,..nocc,..nocc]).assign(&a_mat_vo.into_shape([nvirt,nocc,nocc,nocc]).unwrap());
        a_matrix.slice_mut(s![nocc..,nocc..,..nocc,..nocc]).assign(&a_mat_vv.into_shape([nvirt,nvirt,nocc,nocc]).unwrap());

        let mut a_mat_vo:Array2<f64> = Array2::zeros([nvirt*nocc,nocc*nvirt]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo.t().dot(&g0.dot(&qov));
        // integral (ik|jl)
        a_mat_vo = a_mat_vo - qvo.t().dot(&g0_lr.dot(&qov)).into_shape([nvirt,nocc,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nocc*nvirt]).unwrap();
        // integral (il|jk)
        a_mat_vo = a_mat_vo - qvv.t().dot(&g0_lr.dot(&qoo)).into_shape([nvirt,nvirt,nocc,nocc]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nocc*nvirt]).unwrap();

        let mut a_mat_oo:Array2<f64> = Array2::zeros([nocc*nocc,nocc*nvirt]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo.t().dot(&g0.dot(&qov));
        // integral (ik|jl)
        a_mat_oo = a_mat_oo - qoo.t().dot(&g0_lr.dot(&qov)).into_shape([nocc,nocc,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nocc,nocc*nvirt]).unwrap();
        // integral (il|jk)
        a_mat_oo = a_mat_oo - qov.t().dot(&g0_lr.dot(&qoo)).into_shape([nocc,nvirt,nocc,nocc]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nocc,nocc*nvirt]).unwrap();

        let mut a_mat_ov:Array2<f64> = Array2::zeros([nocc*nvirt,nocc*nvirt]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov.t().dot(&g0.dot(&qov));
        // integral (ik|jl)
        a_mat_ov = a_mat_ov - qoo.t().dot(&g0_lr.dot(&qvv)).into_shape([nocc,nocc,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nocc*nvirt]).unwrap();
        // integral (il|jk)
        a_mat_ov = a_mat_ov - qov.t().dot(&g0_lr.dot(&qvo)).into_shape([nocc,nvirt,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nocc*nvirt]).unwrap();

        let mut a_mat_vv:Array2<f64> = Array2::zeros([nvirt*nvirt,nocc*nvirt]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv.t().dot(&g0.dot(&qov));
        // integral (ik|jl)
        a_mat_vv = a_mat_vv - qvo.t().dot(&g0_lr.dot(&qvv)).into_shape([nvirt,nocc,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nocc*nvirt]).unwrap();
        // integral (il|jk)
        a_mat_vv = a_mat_vv - qvv.t().dot(&g0_lr.dot(&qvo)).into_shape([nvirt,nvirt,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nocc*nvirt]).unwrap();

        a_matrix.slice_mut(s![..nocc,..nocc,..nocc,nocc..]).assign(&a_mat_oo.into_shape([nocc,nocc,nocc,nvirt]).unwrap());
        a_matrix.slice_mut(s![..nocc,nocc..,..nocc,nocc..]).assign(&a_mat_ov.into_shape([nocc,nvirt,nocc,nvirt]).unwrap());
        a_matrix.slice_mut(s![nocc..,..nocc,..nocc,nocc..]).assign(&a_mat_vo.into_shape([nvirt,nocc,nocc,nvirt]).unwrap());
        a_matrix.slice_mut(s![nocc..,nocc..,..nocc,nocc..]).assign(&a_mat_vv.into_shape([nvirt,nvirt,nocc,nvirt]).unwrap());

        let mut a_mat_vo:Array2<f64> = Array2::zeros([nvirt*nocc,nvirt*nvirt]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo.t().dot(&g0.dot(&qvv));
        // integral (ik|jl)
        a_mat_vo = a_mat_vo - qvv.t().dot(&g0_lr.dot(&qov)).into_shape([nvirt,nvirt,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nvirt*nvirt]).unwrap();
        // integral (il|jk)
        a_mat_vo = a_mat_vo - qvv.t().dot(&g0_lr.dot(&qov)).into_shape([nvirt,nvirt,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nvirt*nvirt]).unwrap();

        let mut a_mat_oo:Array2<f64> = Array2::zeros([nocc*nocc,nvirt*nvirt]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo.t().dot(&g0.dot(&qvv));
        // integral (ik|jl)
        a_mat_oo = a_mat_oo - qov.t().dot(&g0_lr.dot(&qov)).into_shape([nocc,nvirt,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nocc,nvirt*nvirt]).unwrap();
        // integral (il|jk)
        a_mat_oo = a_mat_oo - qov.t().dot(&g0_lr.dot(&qov)).into_shape([nocc,nvirt,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nocc,nvirt*nvirt]).unwrap();

        let mut a_mat_ov:Array2<f64> = Array2::zeros([nocc*nvirt,nvirt*nvirt]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov.t().dot(&g0.dot(&qvv));
        // integral (ik|jl)
        a_mat_ov = a_mat_ov - qov.t().dot(&g0_lr.dot(&qvv)).into_shape([nocc,nvirt,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nvirt*nvirt]).unwrap();
        // integral (il|jk)
        a_mat_ov = a_mat_ov - qov.t().dot(&g0_lr.dot(&qvv)).into_shape([nocc,nvirt,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nvirt*nvirt]).unwrap();

        let mut a_mat_vv:Array2<f64> = Array2::zeros([nvirt*nvirt,nvirt*nvirt]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv.t().dot(&g0.dot(&qvv));
        // integral (ik|jl)
        a_mat_vv = a_mat_vv - qvv.t().dot(&g0_lr.dot(&qvv)).into_shape([nvirt,nvirt,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nvirt*nvirt]).unwrap();
        // integral (il|jk)
        a_mat_vv = a_mat_vv - qvv.t().dot(&g0_lr.dot(&qvv)).into_shape([nvirt,nvirt,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nvirt*nvirt]).unwrap();

        a_matrix.slice_mut(s![..nocc,..nocc,nocc..,nocc..]).assign(&a_mat_oo.into_shape([nocc,nocc,nvirt,nvirt]).unwrap());
        a_matrix.slice_mut(s![..nocc,nocc..,nocc..,nocc..]).assign(&a_mat_ov.into_shape([nocc,nvirt,nvirt,nvirt]).unwrap());
        a_matrix.slice_mut(s![nocc..,..nocc,nocc..,nocc..]).assign(&a_mat_vo.into_shape([nvirt,nocc,nvirt,nvirt]).unwrap());
        a_matrix.slice_mut(s![nocc..,nocc..,nocc..,nocc..]).assign(&a_mat_vv.into_shape([nvirt,nvirt,nvirt,nvirt]).unwrap());

        let a_mat:Array2<f64> = a_matrix.into_shape([self.n_orbs*self.n_orbs,self.n_orbs*self.n_orbs]).unwrap();

        println!("Elapsed time calculate A and B matrices: {:>8.6}",timer.elapsed().as_secs_f64());
        drop(timer);
        println!("Start iterative cphf routine");
        let timer: Instant = Instant::now();
        let u_mat_pople:Array3<f64> = solve_cphf_pople_test(a_mat.view(),b_mat.view(),orbe.view(),nocc,nvirt,self.n_atoms);
        println!("Elapsed time CPHF pople: {:>8.6}",timer.elapsed().as_secs_f64());
        drop(timer);
        let timer: Instant = Instant::now();
        // let u_mat = solve_cphf_new(a_mat.view(),b_mat.view(),orbe.view(),nocc,nvirt,self.n_atoms,grad_s.view(),orbs.view());
        let u_mat = solve_cphf_iterative_new(a_mat.view(),b_mat.view(),orbe.view(),nocc,nvirt,self.n_atoms,grad_s.view(),orbs.view());
        println!("Elapsed time CPHF iterative: {:>8.6}",timer.elapsed().as_secs_f64());
        drop(timer);
        // println!(" ");
        // println!("U mat pople {}",u_mat_pople);
        // println!(" ");
        // println!("U mat{}",u_mat);
        // assert!(u_mat.abs_diff_eq(&u_mat_pople,1.0e-7));

        return u_mat_pople;
    }

    pub fn calculate_ct_fock_gradient(
        &self,
        atoms: &[Atom],
        ct_index: usize,
        hole: bool,
    ) -> (Array1<f64>,Array3<f64>) {
        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&atoms, self.n_orbs, &self.slako);

        // get necessary arrays from properties
        let diff_p: Array2<f64> = &self.properties.p().unwrap() - &self.properties.p_ref().unwrap();
        let g0_ao: ArrayView2<f64> = self.properties.gamma_ao().unwrap();
        let g0:ArrayView2<f64> = self.properties.gamma().unwrap();
        let g1:ArrayView3<f64> = self.properties.grad_gamma().unwrap();
        let g1_ao: ArrayView3<f64> = self.properties.grad_gamma_ao().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let orbs: ArrayView2<f64> = self.properties.orbs().unwrap();
        let orbe:ArrayView1<f64> = self.properties.orbe().unwrap();

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
        }

        // esp atomwise
        let esp_atomwise:Array1<f64> = 0.5*g0.dot(&self.properties.dq().unwrap());
        // get Omega_AB
        let omega_ab:Array2<f64> = atomwise_to_aowise(esp_atomwise.view(),self.n_orbs,atoms);

        // get MO coefficient for the occupied or virtual orbital
        let mut c_mo: Array1<f64> = Array1::zeros(self.n_orbs);
        let mut ind:usize = 0;
        let occ_indices: &[usize] = self.properties.occ_indices().unwrap();
        let virt_indices: &[usize] = self.properties.virt_indices().unwrap();
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
                let g0_lr:ArrayView2<f64> = self.properties.gamma_lr().unwrap();
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
                let g0_lr:ArrayView2<f64> = self.properties.gamma_lr().unwrap();
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
            &(grad_h.view()
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

        let g0_lr:ArrayView2<f64> = self.properties.gamma_lr().unwrap();
        // calculate B matrix B_ij with i = nvirt, j = nocc
        // for the CPHF iterations
        let mut b_mat:Array3<f64> = Array3::zeros([3*self.n_atoms,self.n_orbs,self.n_orbs]);
        // let mut b_matrix:Array3<f64> = Array3::zeros([3*self.n_atoms,nvirt,nocc]);

        for nc in 0..3*self.n_atoms{
            let ds_mo:Array1<f64> = orbs_occ.t().dot(&grad_s.slice(s![nc,..,..])
                .dot(&orbs_occ)).into_shape([nocc*nocc]).unwrap();
            // integral (ij|kl) - (ik|jl), i = nvirt, j = nocc
            let integral_vo:Array2<f64> = (2.0 * qvo.t().dot(&g0.dot(&qoo)) - qvo.t().dot(&g0_lr.dot(&qoo)).into_shape([nvirt,nocc,nocc,nocc]).unwrap()
                .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nocc*nocc]).unwrap())
                .dot(&ds_mo).into_shape([nvirt,nocc]).unwrap();
            // integral (ij|kl) - (ik|jl), i = nocc, j = nocc
            let integral_oo:Array2<f64> = (2.0 * qoo.t().dot(&g0.dot(&qoo)) -qoo.t().dot(&g0_lr.dot(&qoo)).into_shape([nocc,nocc,nocc,nocc]).unwrap()
                .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nocc,nocc*nocc]).unwrap())
                .dot(&ds_mo).into_shape([nocc,nocc]).unwrap();
            // integral (ij|kl) - (ik|jl), i = nvirt, j = nvirt
            let integral_vv:Array2<f64> = (2.0 * qvv.t().dot(&g0.dot(&qoo)) -qvo.t().dot(&g0_lr.dot(&qvo)).into_shape([nvirt,nocc,nvirt,nocc]).unwrap()
                .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nocc*nocc]).unwrap())
                .dot(&ds_mo).into_shape([nvirt,nvirt]).unwrap();
            // integral (ij|kl) - (ik|jl), i = nocc, j = nvirt
            let integral_ov:Array2<f64> = (2.0 * qov.t().dot(&g0.dot(&qoo)) -qoo.t().dot(&g0_lr.dot(&qvo)).into_shape([nocc,nocc,nvirt,nocc]).unwrap()
                .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nocc*nocc]).unwrap())
                .dot(&ds_mo).into_shape([nocc,nvirt]).unwrap();

            let gradh_mo:Array2<f64> = orbs.t().dot(&grad_h.slice(s![nc,..,..]).dot(&orbs));
            let grads_mo:Array2<f64> = orbs.t().dot(&grad_s.slice(s![nc,..,..]).dot(&orbs));

            for i in 0..nvirt{
                for j in 0..nocc{
                    // b_matrix[[nc,i,j]] = gradh_mo[[nocc+i,j]] - grads_mo[[nocc+i,j]] * orbe[j] - integral_vo[[i,j]];
                    b_mat[[nc,nocc+i,j]] = gradh_mo[[nocc+i,j]] - grads_mo[[nocc+i,j]] * orbe[j] - integral_vo[[i,j]];
                }
                for j in 0..nvirt{
                    b_mat[[nc,nocc+i,nocc+j]] = gradh_mo[[nocc+i,nocc+j]] - grads_mo[[nocc+i,nocc+j]] * orbe[nocc+j] - integral_vv[[i,j]];
                }
            }
            for i in 0..nocc{
                for j in 0..nocc{
                    b_mat[[nc,i,j]] = gradh_mo[[i,j]] - grads_mo[[i,j]] * orbe[j] - integral_oo[[i,j]];
                }
                for j in 0..nvirt{
                    b_mat[[nc,i,nocc+j]] = gradh_mo[[i,nocc+j]] - grads_mo[[i,nocc+j]] * orbe[nocc+j] - integral_ov[[i,j]];
                }
            }
        }

        // calculate A_matrix ij,kl i = nvirt, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vo:Array2<f64> = Array2::zeros([nvirt*nocc,nvirt*nocc]);
        // integral (ij|kl)
        a_mat_vo = 4.0 * qvo.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_vo = a_mat_vo - qvv.t().dot(&g0_lr.dot(&qoo)).into_shape([nvirt,nvirt,nocc,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nvirt*nocc]).unwrap();
        // integral (il|jk)
        a_mat_vo = a_mat_vo - qvo.t().dot(&g0_lr.dot(&qov)).into_shape([nvirt,nocc,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nocc,nvirt*nocc]).unwrap();

        // calculate A_matrix ij,kl i = nocc, j = nocc, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_oo:Array2<f64> = Array2::zeros([nocc*nocc,nvirt*nocc]);
        // integral (ij|kl)
        a_mat_oo = 4.0 * qoo.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_oo = a_mat_oo - qov.t().dot(&g0_lr.dot(&qoo)).into_shape([nocc,nvirt,nocc,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nocc,nvirt*nocc]).unwrap();
        // integral (il|jk)
        a_mat_oo = a_mat_oo - qoo.t().dot(&g0_lr.dot(&qov)).into_shape([nocc,nocc,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nocc,nvirt*nocc]).unwrap();

        // calculate A_matrix ij,kl i = nocc, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_ov:Array2<f64> = Array2::zeros([nocc*nvirt,nvirt*nocc]);
        // integral (ij|kl)
        a_mat_ov = 4.0 * qov.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_ov = a_mat_ov - qov.t().dot(&g0_lr.dot(&qvo)).into_shape([nocc,nvirt,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nvirt*nocc]).unwrap();
        // integral (il|jk)
        a_mat_ov = a_mat_ov - qoo.t().dot(&g0_lr.dot(&qvv)).into_shape([nocc,nocc,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nocc*nvirt,nvirt*nocc]).unwrap();

        // calculate A_matrix ij,kl i = nvirt, j = nvirt, k = nvirt, l = nocc
        // for the CPHF iterations
        let mut a_mat_vv:Array2<f64> = Array2::zeros([nvirt*nvirt,nvirt*nocc]);
        // integral (ij|kl)
        a_mat_vv = 4.0 * qvv.t().dot(&g0.dot(&qvo));
        // integral (ik|jl)
        a_mat_vv = a_mat_vv - qvv.t().dot(&g0_lr.dot(&qvo)).into_shape([nvirt,nvirt,nvirt,nocc]).unwrap()
            .permuted_axes([0,2,1,3]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nvirt*nocc]).unwrap();
        // integral (il|jk)
        a_mat_vv = a_mat_vv - qvo.t().dot(&g0_lr.dot(&qvv)).into_shape([nvirt,nocc,nvirt,nvirt]).unwrap()
            .permuted_axes([0,2,3,1]).as_standard_layout().to_owned().into_shape([nvirt*nvirt,nvirt*nocc]).unwrap();

        let mut a_matrix:Array3<f64> = Array3::zeros([self.n_orbs,self.n_orbs,nvirt*nocc]);
        a_matrix.slice_mut(s![..nocc,..nocc,..]).assign(&a_mat_oo.into_shape([nocc,nocc,nvirt*nocc]).unwrap());
        a_matrix.slice_mut(s![..nocc,nocc..,..]).assign(&a_mat_ov.into_shape([nocc,nvirt,nvirt*nocc]).unwrap());
        a_matrix.slice_mut(s![nocc..,..nocc,..]).assign(&a_mat_vo.into_shape([nvirt,nocc,nvirt*nocc]).unwrap());
        a_matrix.slice_mut(s![nocc..,nocc..,..]).assign(&a_mat_vv.into_shape([nvirt,nvirt,nvirt*nocc]).unwrap());
        let a_mat:Array2<f64> = a_matrix.into_shape([self.n_orbs*self.n_orbs,nvirt*nocc]).unwrap();
        let u_mat = solve_cphf_new(a_mat.view(),b_mat.view(),orbe.view(),nocc,nvirt,self.n_atoms,grad_s.view(),orbs.view());
        // println!("umat {}",u_mat);
        //
        // let (u_mat,grad_omega):(Array3<f64>,Array3<f64>) = solve_cphf(
        //     grad_s.view(),
        //     grad_h0.view(),
        //     orbs,
        //     self.properties.p().unwrap(),
        //     s,
        //     atoms,
        //     occ_indices,
        //     self.n_orbs,
        //     self.n_atoms,
        //     orbe,
        //     omega_ab.view(),
        //     g1,
        //     g0,
        //     self.properties.dq().unwrap()
        // );
        // println!("umat {}",u_mat.slice(s![0,nocc..,..nocc]));
        // let g0_lr:ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        let a_mat:Array2<f64> = if hole {
            // calculate a matrix A_ii,kl = 4 * (ii|kl) - (ik|il) - (il|ik)
            // k = virt, l = occ, i = occupied orbital index of ct state
            // integral (ii|kl)
            let term_1 = 4.0* qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap()
                .slice(s![..,ind,ind]).dot(&g0.dot(&qvo)).into_shape([nvirt,nocc]).unwrap();

            // integral (ik|il)
            let term_2 = - qov.view().into_shape([self.n_atoms,nocc,nvirt]).unwrap().slice(s![..,ind,..]).t()
                .dot(&g0_lr.dot(&qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap().slice(s![..,ind,..])));

            // integral (il|ik)
            let term_3 = -1.0* &(qoo.view().into_shape([self.n_atoms,nocc,nocc]).unwrap().slice(s![..,ind,..]).t()
                .dot(&g0_lr.dot(&qov.view().into_shape([self.n_atoms,nocc,nvirt]).unwrap().slice(s![..,ind,..]))).t());

            term_1 + term_2 + term_3
        }
        else{
            let slice_ind:usize = ind - virt_indices[0];
            // calculate a matrix A_ii,kl = 4 * (ii|kl) - (ik|il) - (il|ik)
            // k = virt, l = occ, i = virtual orbital index of ct state
            // integral (ii|kl)
            let term_1 = 4.0* qvv.view().into_shape([self.n_atoms,nvirt,nvirt]).unwrap()
                .slice(s![..,slice_ind,slice_ind]).dot(&g0.dot(&qvo)).into_shape([nvirt,nocc]).unwrap();

            // integral (ik|il)
            let term_2 = - qvv.view().into_shape([self.n_atoms,nvirt,nvirt]).unwrap().slice(s![..,slice_ind,..]).t()
                .dot(&g0_lr.dot(&qvo.view().into_shape([self.n_atoms,nvirt,nocc]).unwrap().slice(s![..,slice_ind,..])));

            // integral (il|ik)
            let term_3 = -1.0* &(qvo.view().into_shape([self.n_atoms,nvirt,nocc]).unwrap().slice(s![..,slice_ind,..]).t()
                .dot(&g0_lr.dot(&qvv.view().into_shape([self.n_atoms,nvirt,nvirt]).unwrap().slice(s![..,slice_ind,..]))).t());

            term_1 + term_2 + term_3
        };

        // calculate gradient term of the umatrix: sum_k sum_l U^a_kl A_ii,kl
        let mut u_term:Array1<f64> = Array1::zeros(3*self.n_atoms);
        for nat in 0..3*self.n_atoms{
            u_term[nat] = u_mat.slice(s![nat,nocc..,..nocc]).to_owned().into_shape([nvirt*nocc])
                .unwrap().dot(&a_mat.view().into_shape([nvirt*nocc]).unwrap());
            // for (virt_ind,virt) in virt_indices.iter().enumerate(){
            //     for (occ_ind,occ) in occ_indices.iter().enumerate(){
            //         u_term[nat] += u_mat[[nat,*virt,*occ]] * a_mat[[virt_ind,occ_ind]];
            //     }
            // }
        }
        let grad = &grad_h_mo - orbe[ind] * &grad_s_mo - &gradient_term + &u_term;
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
        // let dq:ArrayView1<f64> = self.properties.dq().unwrap();
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
        //             term_2[index] += self.properties.p().unwrap().slice(s![mu,..]).dot(&grad_s.slice(s![n_at,mu,..]));
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
        // let p_mat:ArrayView2<f64> = self.properties.p().unwrap();
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

        return (grad,u_mat);
    }
}

fn f_v_ct_2(
    s_i: ArrayView2<f64>,
    s_j: ArrayView2<f64>,
    grad_s_i: ArrayView3<f64>,
    grad_s_j: ArrayView3<f64>,
    g0_pair_ao: ArrayView2<f64>,
    g1_pair_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb_i: usize,
    n_orb_j:usize,
)->Array5<f64>{
    let mut integral:Array5<f64> = Array5::zeros([3*n_atoms,n_orb_i,n_orb_i,n_orb_j,n_orb_j]);
    let g0_ij:ArrayView2<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]);

    for nc in 0..3 * n_atoms {
        let ds_i: ArrayView2<f64> = grad_s_i.slice(s![nc, .., ..]);
        let ds_j: ArrayView2<f64> = grad_s_j.slice(s![nc, .., ..]);
        let dg_ij: ArrayView2<f64> = g1_pair_ao.slice(s![nc, ..n_orb_i,n_orb_i..]);

        for mu in 0..n_orb_i{
            for la in 0..n_orb_i{
                for nu in 0..n_orb_j{
                    for sig in 0..n_orb_j{
                        integral[[nc,mu,la,nu,sig]] = 0.25 * ((ds_i[[mu,la]]*s_j[[nu,sig]] + s_i[[mu,la]] * ds_j[[nu,sig]]) *
                            (g0_ij[[mu,nu]] + g0_ij[[mu,sig]] + g0_ij[[la,nu]] + g0_ij[[la,sig]]) +
                            s_i[[mu,la]] * s_j[[nu,sig]] * (dg_ij[[mu,nu]] + dg_ij[[mu,sig]] + dg_ij[[la,nu]] + dg_ij[[la,sig]]));
                    }
                }
            }
        }
    }
    return integral;
}

fn f_v_coulomb(
    v: ArrayView2<f64>,
    s_i: ArrayView2<f64>,
    s_j: ArrayView2<f64>,
    g0_pair_ao: ArrayView2<f64>,
    n_orb_i: usize,
) -> Array2<f64> {
    // Calculate the coumlom integral in AO basis
    // between two different monomers

    let vp: Array2<f64> = &v + &(v.t());
    let s_j_v: Array1<f64> = (&s_j * &vp).sum_axis(Axis(0));
    let gsv: Array1<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]).dot(&s_j_v);

    let mut d_f: Array2<f64> = Array2::zeros((n_orb_i, n_orb_i));
    for b in 0..n_orb_i {
        for a in 0..n_orb_i {
            d_f[[a, b]] = s_i[[a, b]] * (gsv[a] + gsv[b])
        }
    }
    d_f = d_f * 0.25;

    return d_f;
}

pub fn f_v_coulomb_loop(
    s_i: ArrayView2<f64>,
    s_j: ArrayView2<f64>,
    g0_pair_ao: ArrayView2<f64>,
    n_orb_i: usize,
    n_orb_j:usize,
)->Array4<f64>{
    let mut integral:Array4<f64> = Array4::zeros([n_orb_i,n_orb_i,n_orb_j,n_orb_j]);
    let g0_ij:ArrayView2<f64> = g0_pair_ao.slice(s![..n_orb_i,n_orb_i..]);

    for mu in 0..n_orb_i{
        for la in 0..n_orb_i{
            for nu in 0..n_orb_j{
                for sig in 0..n_orb_j{
                    integral[[mu,la,nu,sig]] = 0.25 * s_j[[nu,sig]] * s_i[[mu,la]] *
                        (g0_ij[[mu,nu]] + g0_ij[[mu,sig]] + g0_ij[[la,nu]] + g0_ij[[la,sig]]);
                }
            }
        }
    }

    return integral;
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

fn solve_cphf_new(
    a_mat:ArrayView2<f64>,
    b_mat:ArrayView3<f64>,
    orbe:ArrayView1<f64>,
    nocc:usize,
    nvirt:usize,
    nat:usize,
    grad_s:ArrayView3<f64>,
    orbs:ArrayView2<f64>,
)->Array3<f64>{
    // let mut u_mat:Array3<f64> = Array3::zeros([3*nat,nvirt,nocc]);
    let n_orbs:usize = nocc + nvirt;
    let mut u_mat:Array3<f64> = Array3::zeros([3*nat,n_orbs,n_orbs]);
    // let mut u_mat:Array3<f64> = Array::random((3*nat,n_orbs,n_orbs), Uniform::new(0., 1.));

    for nat in 0..3*nat{
        u_mat.slice_mut(s![nat,..,..]).assign(&orbs.t().dot(&grad_s.slice(s![nat,..,..]).dot(&orbs)));
    }

    let mut iteration:usize = 0;
    println!("Iterations: ");

    'cphf_loop: for it in 0..500{
        // let u_prev:Array2<f64> = u_mat.clone().into_shape([3*nat,nvirt*nocc]).unwrap();
        let u_prev:Array3<f64> = u_mat.clone();

        // calculate U matrix
        for nc in 0..3*nat{
            // calculate sum_kl Aij,kl U^a_kl
            // let a_term:Array2<f64> = a_mat.dot(&u_prev.slice(s![nc,..])).into_shape([nvirt,nocc]).unwrap();

            let a_term:Array2<f64> = a_mat.dot(&u_prev.slice(s![nc,nocc..,..nocc]).to_owned()
                .into_shape(nvirt*nocc).unwrap()).into_shape([n_orbs,n_orbs]).unwrap();

            // for virt in 0..nvirt{
            //     for occ in 0..nocc{
            //         u_mat[[nc,virt,occ]] = (1.0/(orbe[occ] - orbe[nocc+virt])) * (a_term[[virt,occ]] + b_mat[[nc,virt,occ]]);
            //     }
            // }
            for orb_i in 0..n_orbs{
                for orb_j in 0..n_orbs{
                    if orb_i != orb_j{
                        u_mat[[nc,orb_i,orb_j]] = (1.0/(orbe[orb_j] - orbe[orb_i])) * (a_term[[orb_i,orb_j]] + b_mat[[nc,orb_i,orb_j]]);
                    }
                }
            }
        }

        // let u_vector:Vec<Array2<f64>> = (0..3*nat).into_par_iter().map(|nc|{
        //     // calculate sum_kl Aij,kl U^a_kl
        //     let mut u_2d:Array2<f64> = Array2::zeros((n_orbs,n_orbs));
        //     let a_term:Array2<f64> = a_mat.dot(&u_prev.slice(s![nc,nocc..,..nocc]).to_owned()
        //         .into_shape(nvirt*nocc).unwrap()).into_shape([n_orbs,n_orbs]).unwrap();
        //
        //     for orb_i in 0..n_orbs{
        //         for orb_j in 0..n_orbs{
        //             if orb_i != orb_j{
        //                 u_2d[[orb_i,orb_j]] = (1.0/(orbe[orb_j] - orbe[orb_i])) * (a_term[[orb_i,orb_j]] + b_mat[[nc,orb_i,orb_j]]);
        //             }
        //         }
        //     }
        //     u_2d
        // }).collect();
        // for (nc,arr) in u_vector.iter().enumerate(){
        //     u_mat.slice_mut(s![nc,..,..]).assign(&arr);
        // }

        // check convergence
        // let diff:Array2<f64> = (&u_prev - &u_mat.view().into_shape([3*nat,nvirt*nocc]).unwrap()).map(|val| val.abs());
        let diff:Array3<f64> = (&u_prev - &u_mat.view()).map(|val| val.abs());
        let not_converged:Vec<f64> = diff.iter().filter_map(|&item| if item > 1e-9 {Some(item)} else {None}).collect();

        if not_converged.len() == 0{
            println!("CPHF converged in {} Iterations.",it);
            break 'cphf_loop;
        }
        iteration = it;
        print!("{}",it);
    }
    println!(" ");
    println!("Number of iterations {}",iteration);

    return u_mat;
}

fn solve_cphf_iterative_new(
    a_mat:ArrayView2<f64>,
    b_mat:ArrayView3<f64>,
    orbe:ArrayView1<f64>,
    nocc:usize,
    nvirt:usize,
    nat:usize,
    grad_s:ArrayView3<f64>,
    orbs:ArrayView2<f64>,
)->Array3<f64>{
    // let mut u_mat:Array3<f64> = Array3::zeros([3*nat,nvirt,nocc]);
    let n_orbs:usize = nocc + nvirt;
    let mut u_mat:Array3<f64> = Array3::zeros([3*nat,n_orbs,n_orbs]);
    let a_sliced:Array3<f64> = a_mat.into_shape([n_orbs*n_orbs,n_orbs,n_orbs]).unwrap()
        .slice(s![..,nocc..,..nocc]).to_owned().into_shape([n_orbs,n_orbs,nvirt*nocc]).unwrap();

    for nat in 0..3*nat{
        u_mat.slice_mut(s![nat,..,..]).assign(&orbs.t().dot(&grad_s.slice(s![nat,..,..]).dot(&orbs)));
    }
    let mut iteration:usize = 0;

    let mut amat_new_3d:Array3<f64> = Array3::zeros([n_orbs,n_orbs,nvirt*nocc]);
    let mut bmat_new:Array3<f64> = Array3::zeros(b_mat.raw_dim());
    for i in 0..n_orbs{
        for j in 0..n_orbs{
            if i != j{
                amat_new_3d.slice_mut(s![i,j,..])
                    .assign(&(&a_sliced.slice(s![i,j,..]) * (1.0/(orbe[j] - orbe[i]))));
                bmat_new.slice_mut(s![..,i,j])
                    .assign(&(&b_mat.slice(s![..,i,j])*(1.0/(orbe[j] - orbe[i]))));
            }
        }
    }
    let a_sliced:Array2<f64> = amat_new_3d.into_shape([n_orbs*n_orbs,nvirt*nocc]).unwrap();

    // let timer: Instant = Instant::now();
    // let u_vec:Vec<Array2<f64>> = (0..3*nat).into_par_iter().map(|nc|{
    //     let mut u_2d:Array2<f64> = u_mat.slice(s![nc,..,..]).to_owned();
    //     let b_2d:ArrayView2<f64> = bmat_new.slice(s![nc,..,..]);
    //     'cphf_loop: for it in 0..500{
    //         let u_prev:Array2<f64> = u_2d.clone();
    //         let a_term:Array2<f64> = a_sliced.dot(&u_prev.slice(s![nocc..,..nocc]).to_owned()
    //             .into_shape(nvirt*nocc).unwrap()).into_shape([n_orbs,n_orbs]).unwrap();
    //         let dampened:Array2<f64> = &(0.2 *&u_prev.slice(s![..,..]))
    //             + &(0.8 * (&a_term + &b_2d));
    //         u_2d.slice_mut(s![..,..]).assign(&dampened);
    //         let diff:Array2<f64> = (&u_prev.view() - &u_2d.view()).map(|val| val.abs());
    //         let not_converged:Vec<f64> = diff.iter().filter_map(|&item| if item > 1e-9 {Some(item)} else {None}).collect();
    //         if not_converged.len() == 0{
    //             println!("CPHF converged in {} Iterations.",it);
    //             break 'cphf_loop;
    //         }
    //     }
    //     u_2d
    // }).collect();
    // for (index,arr) in u_vec.iter().enumerate(){
    //     u_mat.slice_mut(s![index,..,..]).assign(arr);
    // }

    // println!("Elapsed time CPHF loops: {:>8.6}",timer.elapsed().as_secs_f64());
    // drop(timer);

    for nc in 0..3*nat{
        let mut u_2d:Array2<f64> = u_mat.slice(s![nc,..,..]).to_owned();
        let b_2d:ArrayView2<f64> = bmat_new.slice(s![nc,..,..]);

        'cphf_loop: for it in 0..500{
            let u_prev:Array2<f64> = u_2d.clone();

            let a_term:Array2<f64> = a_sliced.dot(&u_prev.slice(s![nocc..,..nocc]).to_owned()
                .into_shape(nvirt*nocc).unwrap()).into_shape([n_orbs,n_orbs]).unwrap();
            let dampened:Array2<f64> = &(0.2 *&u_prev.slice(s![..,..]))
                + &(0.8 * (&a_term + &b_2d));

            u_2d.slice_mut(s![..,..]).assign(&dampened);

            let diff:Array2<f64> = (&u_prev.view() - &u_2d.view()).map(|val| val.abs());
            let not_converged:Vec<f64> = diff.iter().filter_map(|&item| if item > 1e-7 {Some(item)} else {None}).collect();

            if not_converged.len() == 0{
                println!("CPHF converged in {} Iterations.",it);
                break 'cphf_loop;
            }
        }
        u_mat.slice_mut(s![nc,..,..]).assign(&u_2d);
    }

    return u_mat;
}

pub fn solve_cphf_pople(
    a_mat:ArrayView2<f64>,
    b_mat:ArrayView3<f64>,
    orbe:ArrayView1<f64>,
    nocc:usize,
    nvirt:usize,
    nat:usize
)->Array3<f64>{
    let n_orbs:usize = nocc + nvirt;
    // create new A matrix
    // A/(orbe[j] - orbe[i])
    let a_mat_3d:ArrayView3<f64> = a_mat.into_shape([n_orbs,n_orbs,nvirt*nocc]).unwrap();
    let mut amat_new_3d:Array3<f64> = Array3::zeros([n_orbs,n_orbs,nvirt*nocc]);
    let mut bmat_new:Array3<f64> = Array3::zeros(b_mat.raw_dim());
    for i in 0..n_orbs{
        for j in 0..n_orbs{
            if i != j{
                amat_new_3d.slice_mut(s![i,j,..])
                    .assign(&(&a_mat_3d.slice(s![i,j,..]) * (1.0/(orbe[j] - orbe[i]))));
                bmat_new.slice_mut(s![..,i,j])
                    .assign(&(&b_mat.slice(s![..,i,j])*(1.0/(orbe[j] - orbe[i]))));
            }
        }
    }
    // let mut amat_new:Array4<f64> = Array4::zeros((n_orbs,n_orbs,n_orbs,n_orbs));
    // amat_new.slice_mut(s![..,..,nocc..,..nocc])
    //     .assign(&amat_new_3d.into_shape([n_orbs,n_orbs,nvirt,nocc]).unwrap());
    // let amat_new:Array2<f64> = amat_new.into_shape([n_orbs*n_orbs,n_orbs*n_orbs]).unwrap();
    let amat_new:Array2<f64> = amat_new_3d.into_shape([n_orbs*n_orbs,nvirt*nocc]).unwrap();

    let mut u_matrix:Array3<f64> = Array3::zeros([3*nat,n_orbs,n_orbs]);
    // Iteration over the gradient
    for nc in 0..3*nat{
        let b_zero:ArrayView1<f64> = bmat_new.slice(s![nc,..,..]).into_shape([n_orbs*n_orbs]).unwrap();
        let mut saved_b:Vec<Array1<f64>> = Vec::new();
        let mut saved_u_dot_a:Vec<Array1<f64>> = Vec::new();
        let mut b_prev:Array1<f64> = b_zero.to_owned();
        let mut u_prev:Array1<f64> = b_zero.to_owned();
        let mut iteration:usize = 0;
        saved_b.push(b_zero.to_owned());

        let mut first_term:Array1<f64> = amat_new.dot(&b_prev.view().into_shape([n_orbs,n_orbs]).unwrap()
            .slice(s![nocc..,..nocc]).to_owned().into_shape([nvirt*nocc]).unwrap());
        // let mut first_term:Array1<f64> = amat_new.dot(&b_prev);
        saved_u_dot_a.push(first_term.clone());

        'cphf_loop: for it in 0..50{
            let mut second_term:Array1<f64> = Array1::zeros(n_orbs*n_orbs);

            // Gram Schmidt Orthogonalization
            for b_arr in saved_b.iter(){
                second_term = second_term + b_arr.dot(&first_term)/(b_arr.dot(b_arr)) * b_arr;
            }

            b_prev = &first_term - &second_term;
            saved_b.push(b_prev.clone());

            first_term = amat_new.dot(&b_prev.view().into_shape([n_orbs,n_orbs]).unwrap()
                .slice(s![nocc..,..nocc]).to_owned().into_shape([nvirt*nocc]).unwrap());
            // first_term = amat_new.dot(&b_prev);
            saved_u_dot_a.push(first_term.clone());

            let mut u_mat_1d:Array1<f64> = Array1::zeros((n_orbs*n_orbs));
            // calcula the factors a_n and the contributions to the u matrix
            for (b_arr,u_dot_a) in saved_b.iter().zip(saved_u_dot_a.iter()){
                let a_factor:f64 = b_arr.dot(&b_zero)/(b_arr.dot(b_arr)-b_arr.dot(u_dot_a));
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
        u_matrix.slice_mut(s![nc,..,..]).assign(&u_prev.into_shape([n_orbs,n_orbs]).unwrap());
    }
    return u_matrix;
}

pub fn solve_cphf_pople_test(
    a_mat:ArrayView2<f64>,
    b_mat:ArrayView3<f64>,
    orbe:ArrayView1<f64>,
    nocc:usize,
    nvirt:usize,
    nat:usize
)->Array3<f64>{
    let n_orbs:usize = nocc + nvirt;
    // create new A matrix
    // A/(orbe[j] - orbe[i])
    let a_mat_3d:ArrayView4<f64> = a_mat.into_shape([n_orbs,n_orbs,n_orbs,n_orbs]).unwrap();
    let a_mat_3d:Array4<f64> = a_mat_3d.slice(s![..,..,nocc..,..nocc]).to_owned();
    let a_mat_3d:Array3<f64> = a_mat_3d.into_shape([n_orbs,n_orbs,nvirt*nocc]).unwrap();
    let mut amat_new_3d:Array3<f64> = Array3::zeros([n_orbs,n_orbs,nvirt*nocc]);
    let mut bmat_new:Array3<f64> = Array3::zeros(b_mat.raw_dim());
    for i in 0..n_orbs{
        for j in 0..n_orbs{
            if i != j{
                amat_new_3d.slice_mut(s![i,j,..])
                    .assign(&(&a_mat_3d.slice(s![i,j,..]) * (1.0/(orbe[j] - orbe[i]))));
                bmat_new.slice_mut(s![..,i,j])
                    .assign(&(&b_mat.slice(s![..,i,j])*(1.0/(orbe[j] - orbe[i]))));
            }
        }
    }
    let amat_new:Array2<f64> = amat_new_3d.into_shape([n_orbs*n_orbs,nvirt*nocc]).unwrap();

    let mut u_matrix:Array3<f64> = Array3::zeros([3*nat,n_orbs,n_orbs]);
    // Iteration over the gradient
    for nc in 0..3*nat{
        let b_zero:ArrayView1<f64> = bmat_new.slice(s![nc,..,..]).into_shape([n_orbs*n_orbs]).unwrap();
        let mut saved_b:Vec<Array1<f64>> = Vec::new();
        let mut saved_a_dot_b:Vec<Array1<f64>> = Vec::new();
        let mut b_prev:Array1<f64> = b_zero.to_owned();
        let mut u_prev:Array1<f64> = b_zero.to_owned();
        saved_b.push(b_zero.to_owned());

        let mut first_term:Array1<f64> = amat_new.dot(&b_prev.view().into_shape([n_orbs,n_orbs]).unwrap()
            .slice(s![nocc..,..nocc]).to_owned().into_shape([nvirt*nocc]).unwrap());
        // let mut first_term:Array1<f64> = amat_new.dot(&b_prev);
        saved_a_dot_b.push(first_term.clone());

        let mut converged:bool = false;
        'cphf_loop: for it in 0..40{
            let mut second_term:Array1<f64> = Array1::zeros(n_orbs*n_orbs);

            // Gram Schmidt Orthogonalization
            for b_arr in saved_b.iter(){
                second_term = second_term + b_arr.dot(&first_term)/(b_arr.dot(b_arr)) * b_arr;
            }

            b_prev = &first_term - &second_term;
            saved_b.push(b_prev.clone());

            first_term = amat_new.dot(&b_prev.view().into_shape([n_orbs,n_orbs]).unwrap()
                .slice(s![nocc..,..nocc]).to_owned().into_shape([nvirt*nocc]).unwrap());
            // first_term = amat_new.dot(&b_prev);
            saved_a_dot_b.push(first_term.clone());

            if it >=2{
                // build matrix A and vector b to solve A x = b -> x = A^-1 b
                let b_length:usize = saved_b.len();
                let mut a_matrix:Array2<f64> = Array2::zeros([b_length,b_length]);
                let mut b_vec:Array1<f64> = Array1::zeros(b_length);

                // fill matrix and vector
                for (n, b_n) in saved_b.iter().enumerate(){
                    b_vec[n] = b_n.dot(&b_zero);
                    for ((m, b_m),AB_m) in saved_b.iter().enumerate().zip(saved_a_dot_b.iter()){
                        if n == m{
                            a_matrix[[n,m]] = b_n.dot(b_m);
                        }
                        else{
                            a_matrix[[n,m]] = - b_n.dot(AB_m);
                        }
                    }
                }

                let a_inv:Array2<f64> = a_matrix.inv().unwrap();
                let coefficients:Array1<f64> = a_inv.dot(&b_vec);

                let mut u_mat_1d:Array1<f64> = Array1::zeros((n_orbs*n_orbs));
                for (coeff_n, b_n) in coefficients.iter().zip(saved_b.iter()){
                    u_mat_1d = u_mat_1d + *coeff_n * b_n
                }

                let diff:Array1<f64> = (&u_prev - &u_mat_1d).map(|val| val.abs());
                let not_converged:Vec<f64> = diff.iter().filter_map(|&item| if item > 1e-7 {Some(item)} else {None}).collect();
                u_prev = u_mat_1d;

                if not_converged.len() == 0{
                    converged = true;
                    println!("Pople converged in {} iterations.", it);
                    break 'cphf_loop;
                }
            }
        }
        if converged == false{
            println!("CPHF did NOT converge for nuclear coordinate {} !",nc);
        }

        u_matrix.slice_mut(s![nc,..,..]).assign(&u_prev.into_shape([n_orbs,n_orbs]).unwrap());
    }
    return u_matrix;
}

pub fn solve_cphf_inversion(
    a_mat:ArrayView2<f64>,
    b_mat:ArrayView3<f64>,
    orbe:ArrayView1<f64>,
    nocc:usize,
    nvirt:usize,
    nat:usize
)->Array3<f64>{
    let n_orbs:usize = nocc + nvirt;
    // create new A matrix
    // A/(orbe[j] - orbe[i])
    let a_mat_3d:ArrayView3<f64> = a_mat.into_shape([n_orbs,n_orbs,n_orbs*n_orbs]).unwrap();
    let mut amat_new_3d:Array3<f64> = Array3::zeros([n_orbs,n_orbs,n_orbs*n_orbs]);
    let mut bmat_new:Array3<f64> = Array3::zeros(b_mat.raw_dim());
    for i in 0..n_orbs{
        for j in 0..n_orbs{
            if i != j{
                amat_new_3d.slice_mut(s![i,j,..])
                    .assign(&(&a_mat_3d.slice(s![i,j,..]) * (1.0/(orbe[j] - orbe[i]))));
                bmat_new.slice_mut(s![..,i,j])
                    .assign(&(&b_mat.slice(s![..,i,j])*(1.0/(orbe[j] - orbe[i]))));
            }
        }
    }
    let amat_new:Array2<f64> = amat_new_3d.into_shape([n_orbs*n_orbs,n_orbs*n_orbs]).unwrap();
    let one_minus_amat:Array2<f64> = Array::eye(n_orbs*n_orbs) - amat_new;
    let a_inv:Array2<f64> = one_minus_amat.inv().unwrap();
    let a_sliced:Array2<f64> = a_inv.view().into_shape([n_orbs*n_orbs,n_orbs,n_orbs]).unwrap()
        .slice(s![..,nocc..,..nocc]).to_owned().into_shape([n_orbs*n_orbs,nvirt*nocc]).unwrap();

    let mut u_matrix:Array3<f64> = Array3::zeros([3*nat,n_orbs,n_orbs]);
    for nc in 0..3*nat{
        u_matrix.slice_mut(s![nc,..,..]).assign(&a_sliced
            .dot(&bmat_new.slice(s![nc,nocc..,..nocc]).to_owned()
                .into_shape([nvirt*nocc]).unwrap())
            .into_shape([n_orbs,n_orbs]).unwrap());
    }

    return u_matrix;
}