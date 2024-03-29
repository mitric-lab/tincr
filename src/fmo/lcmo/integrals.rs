use crate::fmo::lcmo::lcmo_trans_charges::{q_lele, q_pp, ElecHole, q_le_p};
use crate::fmo::{BasisState, ChargeTransfer, LocallyExcited, Monomer, PairType, Particle, SuperSystem, LRC, ChargeTransferPair};
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{gamma_ao_wise_from_gamma_atomwise, gamma_atomwise_ab};
use crate::scc::h0_and_s::h0_and_s_ab;
use hashbrown::HashSet;
use ndarray::prelude::*;
use ndarray::{concatenate, Slice};
use ndarray_linalg::{into_col, into_row, Trace};
use peroxide::fuga::gamma;
use crate::fmo::helpers::get_pair_slice;
use crate::fmo::PairType::Pair;
use crate::fmo::lcmo::cis_gradient::ReducedParticle;
use crate::fmo::lcmo::helpers::coulomb_integral_loop_lect_2e;

impl SuperSystem {
    pub fn exciton_coupling<'a>(&self, lhs: &'a BasisState<'a>, rhs: &'a BasisState<'a>) -> f64 {
        match (lhs, rhs) {
            // Coupling between two LE states.
            (BasisState::LE(ref a), BasisState::LE(ref b)) => {
                if a == b {
                    a.monomer.properties.ci_eigenvalue(a.n).unwrap()
                } else if a.monomer == b.monomer {
                    0.0
                } else {
                    self.le_le(a, b)
                }
            }
            // Coupling between LE and CT state.
            (BasisState::LE(ref a), BasisState::CT(ref b)) => self.le_ct(a, b),
            // Coupling between CT and LE state.
            (BasisState::CT(ref a), BasisState::LE(ref b)) => self.ct_le(a, b),
            // Coupling between CT and CT
            (BasisState::CT(ref a), BasisState::CT(ref b)) => {
                let one_elec: f64 = if a == b {
                    // // orbital energy difference
                    // a.electron.mo.e - a.hole.mo.e

                    // get lcmo_fock matrix
                    let hamiltonian = self.properties.lcmo_fock().unwrap();
                    // get orbital slice for both monomers
                    let indices_elec = a.electron.monomer.slice.orb;
                    let indices_hole = a.hole.monomer.slice.orb;
                    // get views of the lcmo fock matrix for the orbital slices of the monomers
                    let hamiltonian_elec:ArrayView2<f64> = hamiltonian.slice(s![indices_elec,indices_elec]);
                    let hamiltonian_hole:ArrayView2<f64> = hamiltonian.slice(s![indices_hole,indices_hole]);

                    // get the energy values for the MO indices of the hole and the electron
                    let elec_ind:usize = a.electron.mo.idx;
                    let energy_e:f64 = hamiltonian_elec[[elec_ind,elec_ind]];
                    let hole_ind:usize = a.hole.mo.idx;
                    let energy_h:f64 = hamiltonian_hole[[hole_ind,hole_ind]];

                    energy_e-energy_h
                } else {0.0};
                one_elec + self.ct_ct(a, b)
                // one_elec
                // self.ct_ct(a, b)
            },
            (BasisState::LE(ref a),BasisState::PairCT(ref b)) => {
                self.le_ct_new(a,b)
            },
            (BasisState::PairCT(ref a),BasisState::LE(ref b),) => {
                self.ct_le_new(a,b)
            },
            (BasisState::PairCT(ref a),BasisState::PairCT(ref b)) =>{
                if a == b{
                    a.state_energy
                }
                else if a.m_h == b.m_h && a.m_l == b.m_l{
                    0.0
                }
                else{
                    self.ct_ct_new(a,b)
                }
            }
            _ =>{0.0},
        }
    }

    pub fn le_le<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a LocallyExcited<'a>) -> f64 {
        // Check if the ESD approximation is used or not.
        let type_pair: PairType = self
            .properties
            .type_of_pair(i.monomer.index, j.monomer.index);

        // Slices of atoms of I and J.
        let (atoms_i, atoms_j): (Slice, Slice) = (i.monomer.slice.atom, j.monomer.slice.atom);

        // Get the gamma matrix between both sets of atoms.
        let gamma_ab: ArrayView2<f64> = self.properties.gamma_slice(atoms_i, atoms_j).unwrap();

        // Compute the Coulomb interaction between both LE states.
        let coulomb: f64 = i.q_trans.dot(&gamma_ab.dot(&j.q_trans));

        // For the exchange energy, the transition charges between both sets of orbitals are needed.
        // In the case that the monomers are far apart approximation is used,
        // the Exchange coupling is zero.
        let exchange: f64 = match type_pair {
            PairType::ESD => 0.0,
            PairType::Pair => {
                // Reference to the overlap matrix between both sets of orbitals.
                let s_ab: ArrayView2<f64> = self
                    .properties
                    .s()
                    .unwrap()
                    .slice_move(s![i.monomer.slice.orb, j.monomer.slice.orb]);

                // The transition charges between both sets of occupied orbitals are computed. .
                let q_ij: Array3<f64> = q_lele(&i, &j, ElecHole::Hole, ElecHole::Hole, s_ab.view());

                // Transition charges between both sets of virtual orbitals are computed.
                let q_ab: Array3<f64> =
                    q_lele(&i, &j, ElecHole::Electron, ElecHole::Electron, s_ab.view());

                // Reference to the transition density matrix of I in MO basis.
                let b_ia: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();

                // Reference to the transition density matrix of J in MO basis.
                let b_jb: ArrayView2<f64> = j.monomer.properties.tdm(j.n).unwrap();

                // Some properties that are used specify the shapes.
                let n_atoms: usize = i.monomer.n_atoms + j.monomer.n_atoms;
                // Number of occupied orbitals in both monomers.
                let (n_i, n_j): (usize, usize) = (i.occs.ncols(), j.occs.ncols());
                // Number of virtual orbitals in both monomers.
                let (n_a, n_b): (usize, usize) = (i.virts.ncols(), j.virts.ncols());

                // The lrc-Gamma matrix of the dimer. TODO: WHICH GAMMA IS NEEDED HERE???
                let gamma_lc_ab: Array2<f64> = self.gamma_ab_cd(
                    i.monomer.index,
                    j.monomer.index,
                    i.monomer.index,
                    j.monomer.index,
                    LRC::ON,
                );

                // Contract the product b_ia^I b_jb^J ( i^I j^J | a^I b^J)
                let bia_ij = q_ij
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_shape((n_atoms * n_j, n_i))
                    .unwrap()
                    .dot(&b_ia)
                    .into_shape((n_atoms, n_j, n_a))
                    .unwrap()
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_shape((n_atoms, n_a * n_j))
                    .unwrap()
                    .to_owned();

                let ab_bjb: Array2<f64> = q_ab
                    .into_shape([n_atoms * n_a, n_b])
                    .unwrap()
                    .dot(&b_jb.t())
                    .into_shape([n_atoms, n_a * n_j])
                    .unwrap();

                ab_bjb.dot(&bia_ij.t()).dot(&gamma_lc_ab).trace().unwrap()
            }
            PairType::None => 0.0,
        };

        2.0 * coulomb - exchange
    }

    // The matrix elements between two CT states is computed. < CT J_i -> I_a | H | CT L_j -> K_b >
    // The notation CT J_i -> I_a reads as an excitation from orbital i on monomer J to orbital a
    // on monomer I.
    pub fn ct_ct<'a>(&self, state_1: &'a ChargeTransfer<'a>, state_2: &'a ChargeTransfer<'a>) -> f64 {
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

        // The Gamma matrix for the two electron integral (JL|IK) is computed.
        let gamma_jl_ik: Option<Array2<f64>> = match kind {
            CTCoupling::IJIJ => {
                // Gamma matrix between monomer I and monomer J.
                let gamma_j_i: ArrayView2<f64> = self.gamma_a_b(j.idx, i.idx, LRC::OFF);
                Some(gamma_j_i.to_owned())
            }
            CTCoupling::IJJI => {
                // Gamma matrix between the pair JI and IJ.
                if type_ij == PairType::Pair {
                    let gamma_ji_ij: Array2<f64> =
                        self.gamma_ab_cd(j.idx, i.idx, i.idx, j.idx, LRC::OFF);
                    Some(gamma_ji_ij)
                } else {
                    None
                }
            }
            CTCoupling::IJIK => {
                // Gamma matrix between pair JK and monomer I.
                if type_jk == PairType::Pair {
                    let gamma_jk_i: Array2<f64> = self.gamma_ab_c(j.idx, l.idx, i.idx, LRC::OFF);
                    Some(gamma_jk_i)
                } else {
                    None
                }
            }
            CTCoupling::IJJK => {
                // Gamma matrix between pair JK and pair IJ.
                if type_ij == PairType::Pair && type_jk == PairType::Pair {
                    let gamma_jk_ij: Array2<f64> =
                        self.gamma_ab_cd(j.idx, k.idx, i.idx, j.idx, LRC::OFF);
                    Some(gamma_jk_ij)
                } else {
                    None
                }
            }
            CTCoupling::IJKI => {
                // Gamma matrix between pair JI and IK.
                if type_ij == PairType::Pair && type_ik == PairType::Pair {
                    let gamma_ji_ik: Array2<f64> =
                        self.gamma_ab_cd(j.idx, i.idx, i.idx, k.idx, LRC::OFF);
                    Some(gamma_ji_ik)
                } else {
                    None
                }
            }
            CTCoupling::IJKJ => {
                // Gamma matrix between monomer J and pair IK.
                if type_ik == PairType::Pair {
                    let gamma_j_ik: Array2<f64> = self.gamma_ab_c(i.idx, k.idx, j.idx, LRC::OFF);
                    Some(gamma_j_ik.reversed_axes())
                } else {
                    None
                }
            }
            CTCoupling::IJKL => {
                // Gamma matrix between pair JL and pair IK.
                if type_jl == PairType::Pair && type_ik == PairType::Pair {
                    let gamma_jl_ik: Array2<f64> =
                        self.gamma_ab_cd(j.idx, l.idx, i.idx, k.idx, LRC::OFF);
                    Some(gamma_jl_ik)
                } else {
                    None
                }
            }
        };

        let (s_jl, s_ik): (Option<ArrayView2<f64>>, Option<ArrayView2<f64>>) =
            if gamma_jl_ik.is_some() {
                match kind {
                    CTCoupling::IJIJ => {
                        // SJJ SII
                        (
                            Some(j.monomer.properties.s().unwrap()),
                            Some(i.monomer.properties.s().unwrap()),
                        )
                    }
                    CTCoupling::IJJI => {
                        // SJI SIJ
                        let s_ji: ArrayView2<f64> = self
                            .properties
                            .s_slice(j.monomer.slice.orb, i.monomer.slice.orb)
                            .unwrap();
                        let s_ij: ArrayView2<f64> = self
                            .properties
                            .s_slice(i.monomer.slice.orb, j.monomer.slice.orb)
                            .unwrap();
                        (Some(s_ji), Some(s_ij))
                    }
                    CTCoupling::IJIK => {
                        // SJK SII
                        let s_ii: ArrayView2<f64> = i.monomer.properties.s().unwrap();
                        let s_jk: ArrayView2<f64> = self
                            .properties
                            .s_slice(j.monomer.slice.orb, l.monomer.slice.orb)
                            .unwrap();
                        (Some(s_jk), Some(s_ii))
                    }
                    CTCoupling::IJJK => {
                        // SJK SIJ
                        let s_jk: ArrayView2<f64> = self
                            .properties
                            .s_slice(j.monomer.slice.orb, l.monomer.slice.orb)
                            .unwrap();
                        let s_ij: ArrayView2<f64> = self
                            .properties
                            .s_slice(i.monomer.slice.orb, j.monomer.slice.orb)
                            .unwrap();
                        (Some(s_jk), Some(s_ij))
                    }
                    CTCoupling::IJKI => {
                        // SJI SIK
                        let s_ji: ArrayView2<f64> = self
                            .properties
                            .s_slice(j.monomer.slice.orb, i.monomer.slice.orb)
                            .unwrap();
                        let s_ik: ArrayView2<f64> = self
                            .properties
                            .s_slice(i.monomer.slice.orb, k.monomer.slice.orb)
                            .unwrap();
                        (Some(s_ji), Some(s_ik))
                    }
                    CTCoupling::IJKJ => {
                        // SJJ SIK
                        let s_jj: ArrayView2<f64> = j.monomer.properties.s().unwrap();
                        let s_ik: ArrayView2<f64> = self
                            .properties
                            .s_slice(i.monomer.slice.orb, k.monomer.slice.orb)
                            .unwrap();
                        (Some(s_jj), Some(s_ik))
                    }
                    CTCoupling::IJKL => {
                        // SJL SIK
                        let s_jl: ArrayView2<f64> = self
                            .properties
                            .s_slice(j.monomer.slice.orb, l.monomer.slice.orb)
                            .unwrap();
                        let s_ik: ArrayView2<f64> = self
                            .properties
                            .s_slice(i.monomer.slice.orb, k.monomer.slice.orb)
                            .unwrap();
                        (Some(s_jl), Some(s_ik))
                    }
                }
            } else {
                (None, None)
            };

        let ia_jb: f64 = match (type_ij, type_kl) {
            (PairType::Pair, PairType::Pair) => {
                let s_ij: ArrayView2<f64> = self
                    .properties
                    .s_slice(i.monomer.slice.orb, j.monomer.slice.orb)
                    .unwrap();
                let s_kl: ArrayView2<f64> = self
                    .properties
                    .s_slice(k.monomer.slice.orb, l.monomer.slice.orb)
                    .unwrap();
                let gamma_ij_kl_lc: Array2<f64> =
                    self.gamma_ab_cd(i.idx, j.idx, k.idx, l.idx, LRC::ON);
                let q_ia: Array1<f64> = q_pp(i, j, s_ij.view());
                let q_jb: Array1<f64> = q_pp(k, l, s_kl.view());
                q_ia.dot(&gamma_ij_kl_lc).dot(&q_jb)
            }
            (PairType::ESD, PairType::ESD) => 0.0,
            (p1, p2) => {
                0.0
                //panic!("{} {} {} {} {} {} This is not possible", p1, p2, i.idx, j.idx, k.idx, l.idx)
            }
        };
        let ij_ab: f64 = match (gamma_jl_ik, s_jl, s_ik) {
            (Some(gamma), Some(sjl), Some(sik)) => {
                let q_ij: Array1<f64> = q_pp(j, l, sjl.view());
                let q_ab: Array1<f64> = q_pp(i, k, sik.view());
                q_ij.dot(&gamma).dot(&q_ab)
            }
            _ => 0.0,
        };

        2.0 * ia_jb - ij_ab
        // 2.0 * ia_jb
        // - ij_ab
    }

    pub fn le_ct<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) -> f64 {
        // self.le_ct_1e(i, j)
        // let val = self.le_ct_1e(i, j);
        // println!("lect_1e {}",val* 219474.63);
        // val
        self.le_ct_1e(i, j) + self.le_ct_2e(i, j)
        // self.le_ct_2e(i, j)
    }

    pub fn le_ct_1e<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) -> f64 {
        // < LE I | H | CT K_j -> I_b >
        if i.monomer.index == j.electron.idx {
            // Index of the HOMO.
            let homo: usize = i.monomer.properties.homo().unwrap();

            // Transition Density Matrix of the LE state in MO basis.
            let tdm: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();

            // 1. LCMO Fock matrix between monomer of the LE state, I, and monomer of the hole, K.
            // 2. Slice corresponding to the coupling between all occupied orbitals on I and
            //    the orbital that corresponds to the hole on K.
            let f_ij: ArrayView1<f64> = self.properties.lcmo_fock().unwrap()
                .slice_move(s![i.monomer.slice.orb, j.hole.monomer.slice.orb])
                .slice_move(s![..=homo, j.hole.mo.idx]);

            // The matrix element is computed according to: - sum_i b_ib^In * F_ij and returned.
            -1.0 *  tdm.column(j.electron.mo.idx - (homo + 1)).dot(&f_ij)

        // < LE I | H | CT I_j -> J_b >
        } else if i.monomer.index == j.hole.idx {
            // Index of the LUMO.
            let lumo: usize = i.monomer.properties.lumo().unwrap();

            // Transition Density Matrix of the LE state in MO basis.
            let tdm: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();

            // 1. LCMO Fock matrix between monomer of the LE state, I, and the monomer of electron, J.
            // 2. Slice corresponding to all virtual orbitals of monomer I and the orbital of the
            //    hole at monomer J.
            let f_ab: ArrayView1<f64> = self.properties.lcmo_fock().unwrap()
                .slice_move(s![i.monomer.slice.orb, j.electron.monomer.slice.orb])
                .slice_move(s![lumo.., j.electron.mo.idx]);

            // The matrix element is computed according to: sum_a b_ja F_ab
            tdm.row(j.hole.mo.idx).dot(&f_ab)
        } else {
            0.0
        }
    }


    /// The two electron matrix element of between a LE state (on monomer I) and a CT state from orbital j on
    /// monomer J to orbital b to monomer K is computed.
    /// < LE I | H | CT J_j -> K_b >
    pub fn le_ct_2e<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) -> f64 {
        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.monomer.index, j.hole.idx);
        // The same for I and K
        let type_ik: PairType = self.properties.type_of_pair(i.monomer.index, j.electron.idx);
        // and J K
        let type_jk: PairType = self.properties.type_of_pair(j.electron.idx, j.hole.idx);

        // Transition charges of LE state at monomer I.
        let qtrans: ArrayView1<f64> = i.q_trans.view();

        // < LE I | H | CT J_j -> I_b>
        if i.monomer.index == j.electron.idx {
            // Check if the pair IK is close, so that the overlap is non-zero.
            if type_ij == PairType::Pair {
                // Overlap matrix between monomer I and J.
                let s_ij: ArrayView2<f64> = self.properties.s_slice(j.electron.monomer.slice.orb, j.hole.monomer.slice.orb).unwrap();
                // Gamma matrix between pair IJ and monomer I. TODO: Check LC
                let gamma_ij_i: Array2<f64> = self.gamma_ab_c( j.electron.idx, j.hole.idx,i.monomer.index, LRC::ON);
                // q_bj is computed instead of q_jb to use the same overlap and Gamma matrix.
                let q_bj: Array1<f64> = q_pp(&j.electron, &j.hole, s_ij.view());
                // The two electron integral (ia|jb) is computed.
                let ia_jb: f64 = qtrans.dot(&gamma_ij_i.t()).dot(&q_bj);
                // Transition charges between all orbitals on I and the hole on J.
                let q_ij: Array2<f64> = q_le_p(&i, &j.hole, s_ij, ElecHole::Hole);
                // Overlap integral of monomer I.
                let s_ii: ArrayView2<f64> = i.monomer.properties.s().unwrap();
                // Transition charges betwenn all orbitals on I and the electron on I.
                let q_ab: Array2<f64> = q_le_p(&i, &j.electron, s_ii, ElecHole::Electron);

                // The two electron integral b_ia (ij|ab) is computed.
                let ij_ab: f64 = i.tdm.dot(&q_ij.t().dot(&gamma_ij_i.dot(&q_ab)).into_shape([i.occs.ncols() * i.virts.ncols()]).unwrap());
                2.0 * ia_jb - ij_ab
            } else {0.0} // If overlap IK is zero, the coupling is zero.
        // < LE I | H | CT I -> J >
        } else if i.monomer.index == j.hole.idx {
            // Check if the pair IJ is close, so that the overlap is non-zero.
            if type_ik == PairType::Pair {
                // Overlap matrix between monomer I and J.
                let s_ij: ArrayView2<f64> = self.properties.s_slice(j.hole.monomer.slice.orb, j.electron.monomer.slice.orb).unwrap();

                // Gamma matrix between pair IJ and monomer I. TODO: Check LC
                let gamma_ij_i: Array2<f64> = self.gamma_ab_c( j.hole.idx, j.electron.idx,i.monomer.index, LRC::ON);

                // Transition charge between hole on I and electron on J.
                let q_jb: Array1<f64> = q_pp(&j.hole, &j.electron, s_ij.view());

                // The two electron integral (ia|jb) is computed.
                let ia_jb: f64 = qtrans.dot(&gamma_ij_i.t()).dot(&q_jb);

                // Overlap integral of monomer I.
                let s_ii: ArrayView2<f64> = i.monomer.properties.s().unwrap();

                // Transition charges between all orbitals on I and the hole on I.
                let q_ij: Array2<f64> = q_le_p(&i, &j.hole, s_ii.view(), ElecHole::Hole);

                // Transition charges betwenn all orbitals on I and the electron on I.
                let q_ab: Array2<f64> = q_le_p(&i, &j.electron, s_ij.view(), ElecHole::Electron);

                // The two electron integral b_ia (ij|ab) is computed.
                let ij_ab: f64 = i.tdm.dot(&q_ij.t().dot(&gamma_ij_i.t().dot(&q_ab)).into_shape([i.occs.ncols() * i.virts.ncols()]).unwrap());

                2.0 * ia_jb - ij_ab

            } else {0.0} // If overlap IJ is zero, the coupling is zero.
        // < LE I_ia | H | CT K_j -> J_b >
        } else {
            // The integral (ia|jb) requires that the overlap between K and J is non-zero.
            let ia_jb: f64 = if type_jk == PairType::Pair {
                // Overlap matrix between monomer K and J.
                let s_kj: ArrayView2<f64> = self.properties.s_slice(j.hole.monomer.slice.orb, j.electron.monomer.slice.orb).unwrap();

                // Gamma matrix between pair KJ and monomer I. TODO: Check LC
                let gamma_kj_i: Array2<f64> = self.gamma_ab_c( j.hole.idx, j.electron.idx,i.monomer.index, LRC::ON);

                // Transition charge between hole on J and electron on K.
                let q_jb: Array1<f64> = q_pp(&j.hole, &j.electron, s_kj.view());

                // The two electron integral (ia|jb) is computed.
                qtrans.dot(&gamma_kj_i.t()).dot(&q_jb)
            } else {0.0}; // If overlap JK is zero, the integral is zero.
            let ij_ab: f64 = if type_ik == PairType::Pair && type_ij == PairType::Pair {
                // Overlap matrix between monomer I and K.
                let s_ik: ArrayView2<f64> = self.properties.s_slice(i.monomer.slice.orb, j.hole.monomer.slice.orb).unwrap();

                // Overlap matrix between monomer I and J.
                let s_ij: ArrayView2<f64> = self.properties.s_slice(i.monomer.slice.orb, j.electron.monomer.slice.orb).unwrap();

                // Gamma matrix between pair IK and pair IJ. TODO: Check LC
                let gamma_ik_ij: Array2<f64> = self.gamma_ab_cd(i.monomer.index, j.hole.idx, i.monomer.index, j.electron.idx, LRC::ON);

                // Transition charges between all orbitals on I and the hole on K.
                let q_ij: Array2<f64> = q_le_p(&i, &j.hole, s_ik.view(), ElecHole::Hole);

                // Transition charges betwenn all orbitals on I and the electron on J.
                let q_ab: Array2<f64> = q_le_p(&i, &j.electron, s_ij.view(), ElecHole::Electron);

                // The two electron integral b_ia (ij|ab) is computed.
                i.tdm.dot(&q_ij.t().dot(&gamma_ik_ij.dot(&q_ab)).into_shape([i.occs.ncols() * i.virts.ncols()]).unwrap())
            } else {0.0}; // If overlap IK or IJ is zero, the integral is zero.
            2.0 * ia_jb - ij_ab
        }
    }
    pub fn ct_le<'a>(&self, i: &'a ChargeTransfer<'a>, j: &'a LocallyExcited<'a>) -> f64 {
        self.le_ct(j, i)
    }

    pub fn le_ct_new<'a>(&self, i: &'a LocallyExcited<'a>, j: &ChargeTransferPair) -> f64 {
        self.le_ct_1e_new(i, j) + self.le_ct_2e_new(i, j)
    }

    pub fn le_ct_1e_new<'a>(&self,i: &'a LocallyExcited<'a>,j: &ChargeTransferPair)->f64{
        if i.monomer.index == j.m_l{
            // reference to the Monomer of the CT where the hole is placed
            let m_h:&Monomer = &self.monomers[j.m_h];

            // Transition Density Matrix of the LE state in MO basis.
            let tdm_le: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();

            // Transition Density Matrix of the CT state
            let tdm_ct:&Array2<f64> = &j.eigenvectors;

            // Index of the HOMO of the LE.
            let homo_le: usize = i.monomer.properties.homo().unwrap();

            // Index of the HOMO of the CT
            let occ_indices_ct = m_h.properties.occ_indices().unwrap();
            let homo_ct:usize = occ_indices_ct[occ_indices_ct.len()-1];

            let f_ij: ArrayView2<f64> = self.properties.lcmo_fock().unwrap()
                .slice_move(s![i.monomer.slice.orb, m_h.slice.orb])
                .slice_move(s![..=homo_le,..=homo_ct]);

            let nocc_i:usize = f_ij.dim().0;
            let nocc_j:usize = f_ij.dim().1;

            let t_ij:Array1<f64> = tdm_le.dot(&tdm_ct.t()).into_shape([nocc_i*nocc_j]).unwrap();

            -1.0 * f_ij.as_standard_layout().to_owned().into_shape([nocc_i*nocc_j]).unwrap().dot(&t_ij)
        }
        else if i.monomer.index == j.m_h{
            // reference to the Monomer of the CT where the electron is placed
            let m_l:&Monomer = &self.monomers[j.m_l];

            // Transition Density Matrix of the LE state in MO basis.
            let tdm_le: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();

            // Transition Density Matrix of the CT state
            let tdm_ct:&Array2<f64> = &j.eigenvectors;

            // Index of the LUMO of the LE
            let lumo_le: usize = i.monomer.properties.lumo().unwrap();

            // Index of the LUMO of the CT
            let lumo_ct:usize = m_l.properties.virt_indices().unwrap()[0];

            let f_ab: ArrayView2<f64> = self.properties.lcmo_fock().unwrap()
                .slice_move(s![i.monomer.slice.orb, m_l.slice.orb])
                .slice_move(s![lumo_le.., lumo_ct..]);

            let nvirt_a:usize = f_ab.dim().0;
            let nvirt_b:usize = f_ab.dim().1;

            let t_ab:Array1<f64> = tdm_le.t().dot(tdm_ct).into_shape([nvirt_a*nvirt_b]).unwrap();
            f_ab.as_standard_layout().to_owned().into_shape([nvirt_a*nvirt_b]).unwrap().dot(&t_ab)
        }
        else{
            0.0
        }
    }

    pub fn le_ct_2e_new<'a>(&self, i: &'a LocallyExcited<'a>, j: &ChargeTransferPair) -> f64 {
        // Transition charges of LE state at monomer I.
        let qtrans_le: ArrayView1<f64> = i.q_trans.view();
        let active_le:usize = self.config.lcmo.active_space_le;
        let active_ct:usize = self.config.lcmo.active_space_ct;
        let restrict_space:bool = self.config.lcmo.restrict_active_space;

        // calculate the gamma matrix between the three monomers
        let gamma = self.gamma_ab_c(j.m_h,j.m_l,i.monomer.index,LRC::OFF);

        // calculate the coulomb interaction between both charge densities
        let coulomb:f64 = qtrans_le.dot(&gamma.t().dot(&j.q_tr));

        let type_le_h:PairType = self.properties.type_of_pair(i.monomer.index,j.m_h);
        let type_le_l:PairType = self.properties.type_of_pair(i.monomer.index,j.m_l);

        // let exchange:f64 = if type_le_h == PairType::Pair || type_le_l == PairType::Pair{
        let exchange:f64 = if type_le_h == PairType::Pair || type_le_l == PairType::Pair ||
            type_le_h == PairType::None || type_le_l == PairType::None{
            let q_ij = if i.monomer.index == j.m_h{
                let q_oo = i.monomer.properties.q_oo().unwrap();
                let nocc:usize = i.occs.ncols();
                // let q_oo_3d:Array3<f64> =
                //     q_oo.into_shape([i.monomer.n_atoms,nocc,nocc]).unwrap().to_owned();

                let q_oo_3d = if restrict_space{
                    let q_oo_3d:Array3<f64> =
                        q_oo.into_shape([i.monomer.n_atoms,nocc,nocc]).unwrap().to_owned();
                    q_oo_3d.slice(s![..,nocc-active_le..,nocc-active_ct..]).to_owned()
                }else{
                    q_oo.into_shape([i.monomer.n_atoms,nocc,nocc]).unwrap().to_owned()
                };

                q_oo_3d
            }
            else{
                self.q_lect(i,j,ElecHole::Hole)
            };

            let q_ab = if i.monomer.index == j.m_l{
                let q_vv = i.monomer.properties.q_vv().unwrap();
                let nvirt:usize = i.virts.ncols();
                // let q_vv_3d:Array3<f64> = q_vv.into_shape([i.monomer.n_atoms,nvirt,nvirt]).unwrap().to_owned();
                let q_vv_3d = if restrict_space{
                    let q_vv_3d:Array3<f64> = q_vv.into_shape([i.monomer.n_atoms,nvirt,nvirt]).unwrap().to_owned();
                    q_vv_3d.slice(s![..,..active_le,..active_ct]).to_owned()
                }else{
                    q_vv.into_shape([i.monomer.n_atoms,nvirt,nvirt]).unwrap().to_owned()
                };

                q_vv_3d
            }
            else{
                self.q_lect(i,j,ElecHole::Electron)
            };

            let gamma_lr = if i.monomer.index == j.m_h{
                let gamma = self.gamma_ab_c(i.monomer.index,j.m_l,i.monomer.index,LRC::ON);
                gamma.reversed_axes()
            }
            else if i.monomer.index == j.m_l{
                self.gamma_ab_c(i.monomer.index,j.m_h,i.monomer.index,LRC::ON)
            }
            else{
                self.gamma_ab_cd(i.monomer.index, j.m_h, i.monomer.index, j.m_l, LRC::ON)
            };

            // Reference to the transition density matrix of I in MO basis.
            // let b_ia: ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();
            let b_ia = if restrict_space{
                let eigenvectors = i.monomer.properties.tdm(i.n).unwrap();
                let noccs_le:usize = eigenvectors.dim().0;
                let start:usize = noccs_le - active_le;
                eigenvectors.slice(s![start..,..active_le]).to_owned()
            }else{
                i.monomer.properties.tdm(i.n).unwrap().to_owned()
            };

            // Reference to the transition density matrix of J in MO basis.
            let b_jb = if restrict_space{
                let nocc:usize = self.monomers[j.m_h].properties.occ_indices().unwrap().len();
                let start = nocc - active_ct;
                j.eigenvectors.slice(s![start..,..active_ct])
            }
            else{
                j.eigenvectors.view()
            };

            // Some properties that are used specify the shapes.
            let n_atoms_ij:usize = q_ij.dim().0;
            let n_atoms_ab:usize = q_ab.dim().0;
            // Number of occupied orbitals in both monomers.
            let (n_i, n_j): (usize, usize) = (q_ij.dim().1, q_ij.dim().2);
            // Number of virtual orbitals in both monomers.
            let (n_a, n_b): (usize, usize) = (q_ab.dim().1, q_ab.dim().2);

            // Contract the product b_ia^I b_jb^J ( i^I j^J | a^I b^J)
            let bia_ij = q_ij
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .into_shape((n_atoms_ij * n_j, n_i))
                .unwrap()
                .dot(&b_ia)
                .into_shape((n_atoms_ij, n_j, n_a))
                .unwrap()
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .into_shape((n_atoms_ij, n_a * n_j))
                .unwrap()
                .to_owned();

            let ab_bjb: Array2<f64> = q_ab
                .into_shape([n_atoms_ab * n_a, n_b])
                .unwrap()
                .dot(&b_jb.t())
                .into_shape([n_atoms_ab, n_a * n_j])
                .unwrap();

            ab_bjb.dot(&bia_ij.t()).dot(&gamma_lr).trace().unwrap()
        }
        else{
            0.0
        };

        2.0 * coulomb - exchange
    }

    pub fn ct_le_new<'a>(&self, i: &ChargeTransferPair, j: &'a LocallyExcited<'a>) -> f64 {
        self.le_ct_new(j, i)
    }

    pub fn ct_ct_new<'a>(&self, state_1: &ChargeTransferPair, state_2: &ChargeTransferPair) -> f64 {
        // calculate the gamma matrix between the two pairs
        let gamma_ij_kl: Array2<f64> =
            self.gamma_ab_cd(state_1.m_h, state_1.m_l, state_2.m_h, state_2.m_l, LRC::OFF);

        // calculate the coulomb interaction between both charge densities
        let coulomb: f64 = state_1.q_tr.dot(&gamma_ij_kl.dot(&state_2.q_tr));

        // get all possible pair types of the monomers
        let type_hh: PairType = self
            .properties
            .type_of_pair(state_1.m_h, state_2.m_h);
        let type_ll:PairType = self.properties.type_of_pair(state_1.m_l,state_2.m_l);
        let type_hl: PairType = self
            .properties
            .type_of_pair(state_1.m_h, state_2.m_l);
        let type_lh: PairType = self
            .properties
            .type_of_pair(state_1.m_l, state_2.m_h);

        let restrict_space:bool = self.config.lcmo.restrict_active_space;
        let active_ct:usize = self.config.lcmo.active_space_ct;

        // calculate the exchange like integral in case one of the pair type is a real pair
        // let exchange:f64 = if type_hh == PairType::Pair || type_hl == PairType::Pair || type_lh == PairType::Pair || type_ll == PairType::Pair{
        let exchange:f64 = if (type_hh == PairType::Pair && type_ll == PairType::Pair) ||
            (type_hh == PairType::None && type_ll == PairType::None) ||
            (type_hh == PairType::Pair && type_ll == PairType::None) ||
            (type_hh == PairType::None && type_ll == PairType::Pair){
            let q_ij = if state_1.m_h == state_2.m_h{
                let nocc:usize = self.monomers[state_1.m_h].properties.occ_indices().unwrap().len();
                let q_oo = self.monomers[state_1.m_h].properties.q_oo().unwrap();
                let n_atoms:usize = self.monomers[state_1.m_h].n_atoms;
                // let q_oo_3d:Array3<f64> = q_oo.into_shape([n_atoms,nocc,nocc]).unwrap().to_owned();

                let q_oo_3d = if restrict_space{
                    let q_oo_3d:Array3<f64> = q_oo.into_shape([n_atoms,nocc,nocc]).unwrap().to_owned();
                    let start:usize = nocc-active_ct;
                    q_oo_3d.slice(s![..,start..,start..]).to_owned()
                }
                else{
                    q_oo.into_shape([n_atoms,nocc,nocc]).unwrap().to_owned()
                };

                q_oo_3d
            }
            else{
                self.q_ctct(state_1,state_2,ElecHole::Hole)
            };

            let q_ab:Array3<f64> = if state_1.m_l == state_2.m_l{
                let nvirt:usize = self.monomers[state_1.m_h].properties.virt_indices().unwrap().len();
                let q_vv = self.monomers[state_1.m_h].properties.q_vv().unwrap();
                let n_atoms:usize = self.monomers[state_1.m_h].n_atoms;
                // let q_vv_3d:Array3<f64> = q_vv.into_shape([n_atoms,nvirt,nvirt]).unwrap().to_owned();

                let q_vv_3d= if restrict_space{
                    let q_vv_3d:Array3<f64> = q_vv.into_shape([n_atoms,nvirt,nvirt]).unwrap().to_owned();
                    q_vv_3d.slice(s![..,..active_ct,..active_ct]).to_owned()
                }
                else{
                    q_vv.into_shape([n_atoms,nvirt,nvirt]).unwrap().to_owned()
                };

                q_vv_3d
            }
            else{
                self.q_ctct(state_1,state_2,ElecHole::Electron)
            };

            let gamma_lr = if state_1.m_h == state_2.m_h{
                let gamma = self.gamma_ab_c(state_1.m_l,state_2.m_l,state_1.m_h,LRC::ON);
                gamma.reversed_axes()
            }
            else if state_1.m_l == state_2.m_l{
                self.gamma_ab_c(state_1.m_h,state_2.m_h,state_1.m_l,LRC::ON)
            }
            else{
                self.gamma_ab_cd(state_1.m_h,state_2.m_h, state_1.m_l,state_2.m_l, LRC::ON)
            };

            // // Reference to the transition density matrix of the CT 1
            // let b_ia: &Array2<f64> = &state_1.eigenvectors;
            // // Reference to the transition density matrix of the CT 2.
            // let b_jb: &Array2<f64> = &state_2.eigenvectors;

            // Reference to the transition density matrix of the CT 1
            let b_ia = if restrict_space{
                let nocc:usize = self.monomers[state_1.m_h].properties.occ_indices().unwrap().len();
                let start = nocc - active_ct;
                state_1.eigenvectors.slice(s![start..,..active_ct])
            }
            else{
                state_1.eigenvectors.view()
            };
            // Reference to the transition density matrix of the CT 2.
            let b_jb = if restrict_space{
                let nocc:usize = self.monomers[state_2.m_h].properties.occ_indices().unwrap().len();
                let start = nocc - active_ct;
                state_2.eigenvectors.slice(s![start..,..active_ct])
            }
            else{
                state_2.eigenvectors.view()
            };

            // Some properties that are used specify the shapes.
            let n_atoms_ij:usize = q_ij.dim().0;
            let n_atoms_ab:usize = q_ab.dim().0;
            // Number of occupied orbitals in both monomers.
            let (n_i, n_j): (usize, usize) = (q_ij.dim().1, q_ij.dim().2);
            // Number of virtual orbitals in both monomers.
            let (n_a, n_b): (usize, usize) = (q_ab.dim().1, q_ab.dim().2);

            // Contract the product b_ia^I b_jb^J ( i^I j^J | a^I b^J)
            let bia_ij = q_ij
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .into_shape((n_atoms_ij * n_j, n_i))
                .unwrap()
                .dot(&b_ia)
                .into_shape((n_atoms_ij, n_j, n_a))
                .unwrap()
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .into_shape((n_atoms_ij, n_a * n_j))
                .unwrap()
                .to_owned();

            let ab_bjb: Array2<f64> = q_ab
                .into_shape([n_atoms_ab * n_a, n_b])
                .unwrap()
                .dot(&b_jb.t())
                .into_shape([n_atoms_ab, n_a * n_j])
                .unwrap();

            ab_bjb.dot(&bia_ij.t()).dot(&gamma_lr).trace().unwrap()
        }
        else{
            0.0
        };

        2.0 *coulomb - exchange
    }

    /// The two electron matrix element of between a LE state (on monomer I) and a CT state from orbital j on
    /// monomer J to orbital b to monomer K is computed.
    /// < LE I | H | CT J_j -> K_b >
    pub fn le_ct_2e_test<'a>(&self, i: &'a LocallyExcited<'a>, j: &'a ChargeTransfer<'a>) -> f64 {
        // Check if the pair of monomers I and J is close to each other or not: S_IJ != 0 ?
        let type_ij: PairType = self.properties.type_of_pair(i.monomer.index, j.hole.idx);
        // The same for I and K
        let type_ik: PairType = self.properties.type_of_pair(i.monomer.index, j.electron.idx);

        // Transition charges of LE state at monomer I.
        let qtrans: ArrayView1<f64> = i.q_trans.view();

        // < LE I | H | CT J_j -> I_b>
        if i.monomer.index == j.electron.idx {
            // Check if the pair IK is close, so that the overlap is non-zero.
            if type_ij == PairType::Pair {
                // Overlap matrix between monomer I and J.
                let s_ij: ArrayView2<f64> = self.properties.s_slice(j.electron.monomer.slice.orb, j.hole.monomer.slice.orb).unwrap();
                let s_i:ArrayView2<f64> = self.properties.s_slice(i.monomer.slice.orb,i.monomer.slice.orb).unwrap();
                let g_i:ArrayView2<f64> = self.properties.gamma_lr_slice(i.monomer.slice.atom,i.monomer.slice.atom).unwrap();
                let g_ij_outer: ArrayView2<f64> = self.properties.gamma_lr_slice(j.electron.monomer.slice.atom, j.hole.monomer.slice.atom).unwrap();

                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    i.monomer.slice.atom_as_range(),
                    j.hole.monomer.slice.atom_as_range(),
                );

                let pair_orbs:usize = i.monomer.n_orbs+j.hole.monomer.n_orbs;
                let mut g_ij:Array2<f64> = Array2::zeros([pair_atoms.len(),pair_atoms.len()]);
                g_ij.slice_mut(s![0..i.monomer.n_atoms,0..i.monomer.n_atoms]).assign(&g_i);
                g_ij.slice_mut(s![i.monomer.n_atoms..,i.monomer.n_atoms..])
                    .assign(&self.properties.gamma_lr_slice(j.hole.monomer.slice.atom,j.hole.monomer.slice.atom).unwrap());
                g_ij.slice_mut(s![0..i.monomer.n_atoms,i.monomer.n_atoms..]).assign(&g_ij_outer);
                g_ij.slice_mut(s![i.monomer.n_atoms..,..i.monomer.n_atoms]).assign(&g_ij_outer.t());

                let g_i_ao:Array2<f64> = gamma_ao_wise_from_gamma_atomwise(g_i,i.atoms,i.monomer.n_orbs);
                let g_ij_ao:Array2<f64> = gamma_ao_wise_from_gamma_atomwise(g_ij.view(),&pair_atoms,pair_orbs);

                let integral:Array4<f64> =
                    coulomb_integral_loop_lect_2e(s_i,s_ij,g_i_ao.view(),g_ij_ao.view(),i.monomer.n_orbs,j.hole.monomer.n_orbs,false);
                let norbs_i:usize = i.monomer.n_orbs;
                let norbs_j:usize = j.hole.monomer.n_orbs;

                let tdm:ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();
                let t_mat:Array2<f64> = i.occs.dot(&tdm.dot(&i.virts.t()));

                let ct_arr:Array2<f64> = into_col(j.hole.mo.c.to_owned()).dot(&into_row(j.electron.mo.c.to_owned()));
                let integral_2d:Array2<f64> = integral.into_shape([norbs_i*norbs_i,norbs_j*norbs_i]).unwrap();
                let val:f64 = t_mat.into_shape([norbs_i*norbs_i]).unwrap()
                    .dot(&integral_2d.dot(&ct_arr.into_shape([norbs_j*norbs_i]).unwrap()));

                // println!("lect {}",val * 219474.63);
                val

                // // Gamma matrix between pair IJ and monomer I. TODO: Check LC
                // let gamma_ij_i_lr: Array2<f64> = self.gamma_ab_c( j.electron.idx, j.hole.idx,i.monomer.index, LRC::ON);
                // let gamma_ij_i: Array2<f64> = self.gamma_ab_c( j.electron.idx, j.hole.idx,i.monomer.index, LRC::OFF);
                // // q_bj is computed instead of q_jb to use the same overlap and Gamma matrix.
                // let q_bj: Array1<f64> = q_pp(&j.electron, &j.hole, s_ij.view());
                // // The two electron integral (ia|jb) is computed.
                // let ia_jb: f64 = qtrans.dot(&gamma_ij_i.t()).dot(&q_bj);
                // // Transition charges between all orbitals on I and the hole on J.
                // let q_ij: Array2<f64> = q_le_p(&i, &j.hole, s_ij, ElecHole::Hole);
                // // Overlap integral of monomer I.
                // let s_ii: ArrayView2<f64> = i.monomer.properties.s().unwrap();
                // // Transition charges betwenn all orbitals on I and the electron on I.
                // let q_ab: Array2<f64> = q_le_p(&i, &j.electron, s_ii, ElecHole::Electron);
                //
                // // The two electron integral b_ia (ij|ab) is computed.
                // let ij_ab: f64 = i.tdm.dot(&q_ij.t().dot(&gamma_ij_i_lr.dot(&q_ab)).into_shape([i.occs.ncols() * i.virts.ncols()]).unwrap());
                // 2.0 * ia_jb - ij_ab
            } else {0.0} // If overlap IK is zero, the coupling is zero.
            // < LE I | H | CT I -> J >
        } else if i.monomer.index == j.hole.idx {
            // Check if the pair IJ is close, so that the overlap is non-zero.
            if type_ik == PairType::Pair {
                let s_ij: ArrayView2<f64> = self.properties.s_slice(j.hole.monomer.slice.orb, j.electron.monomer.slice.orb).unwrap();
                let s_i:ArrayView2<f64> = self.properties.s_slice(i.monomer.slice.orb,i.monomer.slice.orb).unwrap();
                let g_i:ArrayView2<f64> = self.properties.gamma_lr_slice(i.monomer.slice.atom,i.monomer.slice.atom).unwrap();
                let g_ij_outer: ArrayView2<f64> = self.properties.gamma_lr_slice(j.hole.monomer.slice.atom, j.electron.monomer.slice.atom).unwrap();

                let pair_atoms: Vec<Atom> = get_pair_slice(
                    &self.atoms,
                    i.monomer.slice.atom_as_range(),
                    j.electron.monomer.slice.atom_as_range(),
                );

                let pair_orbs:usize = i.monomer.n_orbs+j.electron.monomer.n_orbs;
                let mut g_ij:Array2<f64> = Array2::zeros([pair_atoms.len(),pair_atoms.len()]);
                g_ij.slice_mut(s![0..i.monomer.n_atoms,0..i.monomer.n_atoms]).assign(&g_i);
                g_ij.slice_mut(s![i.monomer.n_atoms..,i.monomer.n_atoms..])
                    .assign(&self.properties.gamma_lr_slice(j.electron.monomer.slice.atom,j.electron.monomer.slice.atom).unwrap());
                g_ij.slice_mut(s![0..i.monomer.n_atoms,i.monomer.n_atoms..]).assign(&g_ij_outer);
                g_ij.slice_mut(s![i.monomer.n_atoms..,..i.monomer.n_atoms]).assign(&g_ij_outer.t());

                let g_i_ao:Array2<f64> = gamma_ao_wise_from_gamma_atomwise(g_i,i.atoms,i.monomer.n_orbs);
                let g_ij_ao:Array2<f64> = gamma_ao_wise_from_gamma_atomwise(g_ij.view(),&pair_atoms,pair_orbs);

                let integral:Array4<f64> =
                    coulomb_integral_loop_lect_2e(s_i,s_ij,g_i_ao.view(),g_ij_ao.view(),i.monomer.n_orbs,j.electron.monomer.n_orbs,true);
                let norbs_i:usize = i.monomer.n_orbs;
                let norbs_j:usize = j.electron.monomer.n_orbs;

                let tdm:ArrayView2<f64> = i.monomer.properties.tdm(i.n).unwrap();
                let t_mat:Array2<f64> = i.occs.dot(&tdm.dot(&i.virts.t()));

                let ct_arr:Array2<f64> = into_col(j.hole.mo.c.to_owned()).dot(&into_row(j.electron.mo.c.to_owned()));
                let integral_2d:Array2<f64> = integral.into_shape([norbs_i*norbs_i,norbs_i*norbs_j]).unwrap();
                let val:f64 = t_mat.into_shape([norbs_i*norbs_i]).unwrap()
                    .dot(&integral_2d.dot(&ct_arr.into_shape([norbs_i*norbs_j]).unwrap()));

                // println!("lect {}",val* 219474.63);
                val

                // // Overlap matrix between monomer I and J.
                // let s_ij: ArrayView2<f64> = self.properties.s_slice(j.hole.monomer.slice.orb, j.electron.monomer.slice.orb).unwrap();
                //
                // // Gamma matrix between pair IJ and monomer I. TODO: Check LC
                // let gamma_ij_i_lr: Array2<f64> = self.gamma_ab_c( j.hole.idx, j.electron.idx,i.monomer.index, LRC::ON);
                // let gamma_ij_i: Array2<f64> = self.gamma_ab_c( j.hole.idx, j.electron.idx,i.monomer.index, LRC::OFF);
                //
                // // Transition charge between hole on I and electron on J.
                // let q_jb: Array1<f64> = q_pp(&j.hole, &j.electron, s_ij.view());
                //
                // // The two electron integral (ia|jb) is computed.
                // let ia_jb: f64 = qtrans.dot(&gamma_ij_i.t()).dot(&q_jb);
                //
                // // Overlap integral of monomer I.
                // let s_ii: ArrayView2<f64> = i.monomer.properties.s().unwrap();
                //
                // // Transition charges between all orbitals on I and the hole on I.
                // let q_ij: Array2<f64> = q_le_p(&i, &j.hole, s_ii.view(), ElecHole::Hole);
                //
                // // Transition charges betwenn all orbitals on I and the electron on I.
                // let q_ab: Array2<f64> = q_le_p(&i, &j.electron, s_ij.view(), ElecHole::Electron);
                //
                // // The two electron integral b_ia (ij|ab) is computed.
                // let ij_ab: f64 = i.tdm.dot(&q_ij.t().dot(&gamma_ij_i_lr.t().dot(&q_ab)).into_shape([i.occs.ncols() * i.virts.ncols()]).unwrap());
                //
                // 2.0 * ia_jb - ij_ab

            } else {0.0} // If overlap IJ is zero, the coupling is zero.
            // < LE I_ia | H | CT K_j -> J_b >
        } else {
            0.0
        }
    }
}

/// Type that specifies what kind of CT-CT coupling is calculated. The letters I,J,K,L indicate the
/// monomers. The naming follows: IJKL -> < CT J -> I | H | CT L -> K >
pub enum CTCoupling {
    IJIJ,
    IJJI,
    IJIK,
    IJJK,
    IJKI,
    IJKJ,
    IJKL,
}

impl<'a> From<(&Particle<'a>, &Particle<'a>, &Particle<'a>, &Particle<'a>)> for CTCoupling {
    fn from(
        (i, j, k, l): (&Particle<'a>, &Particle<'a>, &Particle<'a>, &Particle<'a>),
    ) -> Self {
        // The indices that provide the index of the corresponding monomers.
        let (a, b): (usize, usize) = (i.idx, j.idx);
        let (c, d): (usize, usize) = (k.idx, l.idx);
        // IJ | IJ
        if a == c && b == d {
            // println!("Coupling kind: IJIJ");
            Self::IJIJ
        }
        // IJ | JI
        else if a == d && b == c {
            // println!("Coupling kind: IJJI");
            Self::IJJI
        }
        // IJ | IK
        else if a == c {
            // println!("Coupling kind: IJIK");
            Self::IJIK
        }
        // IJ | JK
        else if b == c {
            // println!("Coupling kind: IJJK");
            Self::IJJK
        }
        // IJ | KI
        else if a == d {
            // println!("Coupling kind: IJKI");
            Self::IJKI
        }
        // IJ | KJ
        else if b == d {
            // println!("Coupling kind: IJKJ");
            Self::IJKJ
        }
        // IJ | KL
        else {
            // println!("Coupling kind: IJKL");
            Self::IJKL
        }
    }
}

impl From<(&ReducedParticle, &ReducedParticle, &ReducedParticle, &ReducedParticle)> for CTCoupling {
    fn from(
        (i, j, k, l): (&ReducedParticle, &ReducedParticle, &ReducedParticle, &ReducedParticle),
    ) -> Self {
        // The indices that provide the index of the corresponding monomers.
        let (a, b): (usize, usize) = (i.m_index, j.m_index);
        let (c, d): (usize, usize) = (k.m_index, l.m_index);
        // IJ | IJ
        if a == c && b == d {
            // println!("Coupling kind: IJIJ");
            Self::IJIJ
        }
        // IJ | JI
        else if a == d && b == c {
            // println!("Coupling kind: IJJI");
            Self::IJJI
        }
        // IJ | IK
        else if a == c {
            // println!("Coupling kind: IJIK");
            Self::IJIK
        }
        // IJ | JK
        else if b == c {
            // println!("Coupling kind: IJJK");
            Self::IJJK
        }
        // IJ | KI
        else if a == d {
            // println!("Coupling kind: IJKI");
            Self::IJKI
        }
        // IJ | KJ
        else if b == d {
            // println!("Coupling kind: IJKJ");
            Self::IJKJ
        }
        // IJ | KL
        else {
            // println!("Coupling kind: IJKL");
            Self::IJKL
        }
    }
}
