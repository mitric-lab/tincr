use crate::fmo::helpers::get_pair_slice;
use crate::fmo::{
    BasisState, ChargeTransfer, ESDPair, GroundStateGradient, Monomer, Pair, PairType, Particle,
    SuperSystem,
};
use crate::initialization::{Atom, MO};
use crate::scc::gamma_approximation::{gamma_atomwise_ab, gamma_ao_wise_from_gamma_atomwise};
use crate::scc::h0_and_s::h0_and_s_ab;
use ndarray::prelude::*;
use std::ops::AddAssign;
use crate::fmo::lcmo::ct_gradient_old::f_v_coulomb_loop;
use ndarray_linalg::{into_col, into_row};
use std::time::Instant;

impl SuperSystem {
    pub fn ct_gradient_new(
        &mut self,
        index_i: usize,
        index_j: usize,
        ct_ind_i: usize,
        ct_ind_j: usize,
        ct_energy: f64,
        hole_i: bool,
    ) -> Array1<f64> {
        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        // get pair type
        let pair_type: PairType = self.properties.type_of_pair(index_i, index_j);
        let mut ct_gradient: Array1<f64> = Array1::zeros([3 * (m_i.n_atoms + m_j.n_atoms)]);

        if pair_type == PairType::Pair {
            // get pair index
            let pair_index: usize = self.properties.index_of_pair(index_i, index_j);
            // get correct pair from pairs vector
            let pair_ij: &mut Pair = &mut self.pairs[pair_index];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            pair_ij.prepare_lcmo_gradient(&pair_atoms, m_i, m_j);
            pair_ij.prepare_ct_state(&pair_atoms, m_i, m_j, ct_ind_i, ct_ind_j, ct_energy, hole_i);
            ct_gradient = pair_ij.tda_gradient_lc(0);
            // reset gradient specific properties
            pair_ij.properties.reset_gradient();
        } else {
            // Do something for ESD pairs
            // get pair index
            let pair_index: usize = self.properties.index_of_esd_pair(index_i, index_j);
            // get correct pair from pairs vector
            let pair_ij: &mut ESDPair = &mut self.esd_pairs[pair_index];
            // get pair atoms
            let pair_atoms: Vec<Atom> = get_pair_slice(
                &self.atoms,
                m_i.slice.atom_as_range(),
                m_j.slice.atom_as_range(),
            );

            // do a scc calculation of the ESD pair
            pair_ij.prepare_scc(&pair_atoms, m_i, m_j);
            pair_ij.run_scc(&pair_atoms, self.config.scf);

            pair_ij.prepare_lcmo_gradient(&pair_atoms);
            pair_ij.prepare_ct_state(&pair_atoms, m_i, m_j, ct_ind_i, ct_ind_j, ct_energy, hole_i);
            ct_gradient = pair_ij.tda_gradient_nolc(0);
            pair_ij.properties.reset();
        }

        return ct_gradient;
    }

    pub fn exciton_ct_energy(
        &mut self,
        index_i: usize,
        index_j: usize,
        ct_ind_i: usize,
        ct_ind_j: usize,
        hole_i: bool,
    ) -> f64 {
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];

        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        // get occupied and virtual orbitals
        let mut occs: &[usize];
        let mut virts: &[usize];
        let mut hole: MO;
        let mut elec: MO;

        let state: BasisState = if hole_i {
            // Indices of the occupied orbitals of Monomer J.
            occs = m_i.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_j.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_ind_i];
            let virt: usize = virts[ct_ind_j];

            // create hole and electron
            hole = MO::new(
                m_i.properties.mo_coeff(occ).unwrap(),
                m_i.properties.orbe().unwrap()[occ],
                occ,
                m_i.properties.occupation().unwrap()[occ],
            );
            elec = MO::new(
                m_j.properties.mo_coeff(virt).unwrap(),
                m_j.properties.orbe().unwrap()[virt],
                virt,
                m_j.properties.occupation().unwrap()[virt],
            );

            BasisState::CT(ChargeTransfer {
                // system: &self,
                hole: Particle {
                    idx: m_i.index,
                    atoms: &atoms[m_i.slice.atom_as_range()],
                    monomer: &m_i,
                    mo: hole,
                },
                electron: Particle {
                    idx: m_j.index,
                    atoms: &atoms[m_j.slice.atom_as_range()],
                    monomer: &m_j,
                    mo: elec,
                },
            })
        } else {
            // Indices of the occupied orbitals of Monomer J.
            occs = m_j.properties.occ_indices().unwrap();
            // Indices of the virtual orbitals of Monomer J.
            virts = m_i.properties.virt_indices().unwrap();
            // set ct indices
            let nocc: usize = occs.len();
            let occ: usize = occs[nocc - 1 - ct_ind_j];
            let virt: usize = virts[ct_ind_i];

            // create hole and electron
            hole = MO::new(
                m_j.properties.mo_coeff(occ).unwrap(),
                m_j.properties.orbe().unwrap()[occ],
                occ,
                m_j.properties.occupation().unwrap()[occ],
            );
            elec = MO::new(
                m_i.properties.mo_coeff(virt).unwrap(),
                m_i.properties.orbe().unwrap()[virt],
                virt,
                m_i.properties.occupation().unwrap()[virt],
            );

            BasisState::CT(ChargeTransfer {
                // system: &self,
                hole: Particle {
                    idx: m_j.index,
                    atoms: &atoms[m_j.slice.atom_as_range()],
                    monomer: &m_j,
                    mo: hole,
                },
                electron: Particle {
                    idx: m_i.index,
                    atoms: &atoms[m_i.slice.atom_as_range()],
                    monomer: &m_i,
                    mo: elec,
                },
            })
        };
        let val: f64 = self.exciton_coupling(&state, &state);

        return val;
    }

    pub fn exciton_hamiltonian_ct_test(&mut self) -> f64 {
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let max_iter: usize = 50;
        let tolerance: f64 = 1e-4;
        // Number of LE states per monomer.
        let n_le: usize = self.config.lcmo.n_le;
        // Compute the n_le excited states for each monomer.
        for mol in self.monomers.iter_mut() {
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()]);
            mol.run_tda(&atoms[mol.slice.atom_as_range()], n_le, max_iter, tolerance);
        }

        // Construct the diabatic basis states.
        let states: Vec<BasisState> = self.create_diab_basis();

        let ct_state = &states[2 * n_le];
        let val: f64 = self.exciton_coupling(ct_state, ct_state);

        return val;
    }

    pub fn calculate_cphf_correction(&mut self,index_i:usize,index_j:usize,ct_ind_i:usize,ct_ind_j:usize,hole_i:bool)->Array1<f64>{
        // get monomers and atoms of both monomers
        let monomer: &Monomer = &self.monomers[index_i];
        let atoms_i: &[Atom] = &self.atoms[monomer.slice.atom_as_range()];
        let monomer: &Monomer = &self.monomers[index_j];
        let atoms_j: &[Atom] = &self.atoms[monomer.slice.atom_as_range()];

        // prepare the calculation of the U matrices
        let monomers:&mut Vec<Monomer> = &mut self.monomers;
        let monomer:&mut Monomer = &mut monomers[index_i];
        monomer.prepare_u_matrix(&atoms_i);
        let monomer: &mut Monomer = &mut monomers[index_j];
        monomer.prepare_u_matrix(&atoms_j);
        let monomers:usize;
        let monomer:usize;

        // get monomers
        let m_i: &Monomer = &self.monomers[index_i];
        let m_j: &Monomer = &self.monomers[index_j];

        let timer: Instant = Instant::now();
        // calculate the U matrix of both monomers using the CPHF equations
        let u_mat_i:Array3<f64> = m_i.calculate_u_matrix(&atoms_i);
        let u_mat_j:Array3<f64> = m_j.calculate_u_matrix(&atoms_j);
        println!("Elapsed time Calculation of both U matrices {:>8.6}",timer.elapsed().as_secs_f64());
        drop(timer);

        // reference to the mo coefficients of fragment I
        let c_mo_i: ArrayView2<f64> = m_i.properties.orbs().unwrap();
        // reference to the mo coefficients of fragment J
        let c_mo_j: ArrayView2<f64> = m_j.properties.orbs().unwrap();

        // get pair index
        let pair_index:usize = self.properties.index_of_pair(index_i,index_j);
        // get correct pair from pairs vector
        let pair_ij: &mut Pair = &mut self.pairs[pair_index];
        // length of atoms of the pair
        let n_atoms_pair:usize = m_i.n_atoms + m_j.n_atoms;
        let pair_atoms: Vec<Atom> = get_pair_slice(
            &self.atoms,
            m_i.slice.atom_as_range(),
            m_j.slice.atom_as_range(),
        );
        // prepare gamma AO matrix
        let g0_ao:Array2<f64> = gamma_ao_wise_from_gamma_atomwise(
            pair_ij.properties.gamma().unwrap(),
            &pair_atoms,
            pair_ij.n_orbs
        );

        // calculate gradients of the MO coefficients
        // dc_mu,i/dR = sum_m^all U^R_mi, C_mu,m
        let mut dc_mo_i:Array2<f64> = Array2::zeros([3*n_atoms_pair,m_i.n_orbs]);
        let mut dc_mo_j:Array2<f64> = Array2::zeros([3*n_atoms_pair,m_j.n_orbs]);

        let cphf_gradient:Array1<f64> = if hole_i{
            let occ_indices_i: &[usize] = m_i.properties.occ_indices().unwrap();
            let nocc_i:usize = occ_indices_i.len();
            let virt_indices_j: &[usize] = m_j.properties.virt_indices().unwrap();
            let orb_ind_i:usize = occ_indices_i[nocc_i - 1 - ct_ind_i];
            let orb_ind_j:usize = virt_indices_j[ct_ind_j];

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
                g0_ao.view(),
                m_i.n_orbs,
                m_j.n_orbs
            );

            let mut coulomb_grad:Array1<f64> = Array1::zeros(3*n_atoms_pair);
            // calculate loop version of cphf coulomb gradient
            let c_i_ind:ArrayView1<f64> = c_mo_i.slice(s![..,orb_ind_i]);
            let c_j_ind:ArrayView1<f64> = c_mo_j.slice(s![..,orb_ind_j]);

            let timer: Instant = Instant::now();

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
            println!("Elapsed time loop {:>8.6}",timer.elapsed().as_secs_f64());
            drop(timer);
            let timer: Instant = Instant::now();

            let coulomb_arr:Array2<f64> =  coulomb_arr.into_shape([m_i.n_orbs*m_i.n_orbs,m_j.n_orbs*m_j.n_orbs]).unwrap();

            let c_mat_j:Array2<f64> = into_col(c_mo_j.slice(s![..,orb_ind_j]).to_owned())
                .dot(&into_row(c_mo_j.slice(s![..,orb_ind_j]).to_owned()));
            let c_mat_i:Array2<f64> = into_col(c_mo_i.slice(s![..,orb_ind_i]).to_owned())
                .dot(&into_row(c_mo_i.slice(s![..,orb_ind_i]).to_owned()));

            let mut coulomb_grad_2:Array1<f64> = Array1::zeros(3*pair_ij.n_atoms);
            // calculate dot version of cphf coulomb gradient
            // iterate over the gradient
            for nat in 0..3*pair_ij.n_atoms{
                // dot product of dc_mu,i/dr c_lambda,i to c_mu,lambda of Fragment I
                let c_i:Array2<f64> = into_col(dc_mo_i.slice(s![nat,..]).to_owned())
                    .dot(&into_row(c_mo_i.slice(s![..,orb_ind_i]).to_owned()));
                let c_i_2:Array2<f64> = into_col(c_mo_i.slice(s![..,orb_ind_i]).to_owned())
                    .dot(&into_row(dc_mo_i.slice(s![nat,..]).to_owned()));
                // dot product of dc_nu,a/dr c_sig,a to c_nu,sig of Fragment J
                let c_j:Array2<f64> = into_col(dc_mo_j.slice(s![nat,..]).to_owned())
                    .dot(&into_row(c_mo_j.slice(s![..,orb_ind_j]).to_owned()));
                let c_j_2:Array2<f64> = into_col(c_mo_j.slice(s![..,orb_ind_j]).to_owned())
                    .dot(&into_row(dc_mo_j.slice(s![nat,..]).to_owned()));

                // calculate dot product of coulomb integral with previously calculated coefficients
                // in AO basis
                let term_1a = c_i.into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                    .dot(&coulomb_arr.dot(&c_mat_j.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                let term_1b = c_i_2.into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                    .dot(&coulomb_arr.dot(&c_mat_j.view().into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                let term_2a = c_mat_i.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                    .dot(&coulomb_arr.dot(&c_j.into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));
                let term_2b = c_mat_i.view().into_shape(m_i.n_orbs*m_i.n_orbs).unwrap()
                    .dot(&coulomb_arr.dot(&c_j_2.into_shape(m_j.n_orbs*m_j.n_orbs).unwrap()));

                coulomb_grad_2[nat] = term_1a + term_1b + term_2a + term_2b;
            }
            println!("Elapsed time other version {:>8.6}",timer.elapsed().as_secs_f64());
            drop(timer);
            assert!(coulomb_grad_2.abs_diff_eq(&coulomb_grad,1.0e-11));

            coulomb_grad_2
        }
        else{
            let occ_indices_j: &[usize] = m_j.properties.occ_indices().unwrap();
            let nocc_j:usize = occ_indices_j.len();
            let virt_indices_i: &[usize] = m_i.properties.virt_indices().unwrap();
            let orb_ind_j:usize = occ_indices_j[nocc_j - 1 - ct_ind_j];
            let orb_ind_i:usize = virt_indices_i[ct_ind_i];

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
            );

            let mut coulomb_grad:Array1<f64> = Array1::zeros(3*n_atoms_pair);
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
            coulomb_grad
        };
        return cphf_gradient;
    }
}
