use crate::initialization::Atom;
use ndarray::prelude::*;
use ndarray_linalg::Trace;

pub fn le_i_ct_one_electron_ij(
    occ_indices_i: &[usize],
    virt_indices_j: &[usize],
    exc_coeff_i: ArrayView3<f64>,
    lcmo_h: ArrayView2<f64>,
    n_le: usize,
    n_ct: usize,
) -> Array2<f64> {
    let mut coupling_matrix: Array2<f64> = Array2::zeros((n_le, n_ct));

    for state in 0..n_le {
        let exc_coeff_state_i: ArrayView2<f64> = exc_coeff_i.slice(s![state, .., ..]);

        let mut ct_counter: usize = 0;
        for occ in occ_indices_i.iter() {
            for virt in virt_indices_j.iter() {
                coupling_matrix[[state, ct_counter]] = exc_coeff_state_i
                    .slice(s![*occ, ..])
                    .dot(&lcmo_h.slice(s![.., *virt]));
                ct_counter += 1;
            }
        }
    }
    return coupling_matrix;
}

pub fn le_i_ct_one_electron_ji(
    occ_indices_j: &[usize],
    virt_indices_i: &[usize],
    exc_coeff_i: ArrayView3<f64>,
    lcmo_h: ArrayView2<f64>,
    n_le: usize,
    n_ct: usize,
) -> Array2<f64> {
    let mut coupling_matrix: Array2<f64> = Array2::zeros((n_le, n_ct));

    for state in 0..n_le {
        let exc_coeff_state_i: ArrayView2<f64> = exc_coeff_i.slice(s![state, .., ..]);

        let mut ct_counter: usize = 0;
        for occ in occ_indices_j.iter() {
            for virt in virt_indices_i.iter() {
                coupling_matrix[[state, ct_counter]] = -1.0
                    * exc_coeff_state_i
                        .slice(s![.., *virt])
                        .dot(&lcmo_h.slice(s![.., *occ]));
                ct_counter += 1;
            }
        }
    }
    return coupling_matrix;
}

pub fn le_le_two_electron(
    n_le: usize,
    exc_coeff_i: ArrayView3<f64>,
    exc_coeff_j: ArrayView3<f64>,
    g0_pair: ArrayView2<f64>,
    g0_ij: ArrayView2<f64>,
    q_ov_i: ArrayView2<f64>,
    q_ov_j: ArrayView2<f64>,
    q_oo_inter_frag: ArrayView2<f64>,
    q_vv_inter_frag: ArrayView2<f64>,
) -> Array2<f64> {
    let nat: usize = g0_pair.dim().0;
    let nocc_i: usize = exc_coeff_i.dim().1;
    let nocc_j: usize = exc_coeff_j.dim().1;
    let nvirt_i: usize = exc_coeff_i.dim().2;
    let nvirt_j: usize = exc_coeff_j.dim().2;

    let mut coupling_matrix: Array2<f64> = Array2::zeros((n_le, n_le));
    let qvv_shape: ArrayView2<f64> = q_vv_inter_frag
        .into_shape((nat * nvirt_i, nvirt_j))
        .unwrap();

    for state_i in 0..n_le {
        // coulomb part
        let qov_b_i: Array1<f64> = q_ov_i.dot(
            &exc_coeff_i
                .slice(s![state_i, .., ..])
                .into_shape(nocc_i * nvirt_i)
                .unwrap(),
        );
        // exchange part
        // calculate einsum of g0 and q_oo_inter_frag: aa,aik -> aik
        // and reshape into [nocc_j * nat, nocc_i]
        let g0_qoo: Array2<f64> = g0_pair
            .dot(&q_oo_inter_frag) // nat,nocc_i*nocc_j
            .into_shape([nat, nocc_i, nocc_j])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .into_shape([nocc_j * nat, nocc_i])
            .unwrap()
            .to_owned();

        for state_j in 0..n_le {
            // coulomb part
            let qov_b_j: Array1<f64> = q_ov_j.dot(
                &exc_coeff_j
                    .slice(s![state_j, .., ..])
                    .into_shape(nocc_j * nvirt_j)
                    .unwrap(),
            );
            // exchange part
            // einsum of b_j and reshaped q_vv_inter_frag: kl,laj -> kaj
            let bj_qvv: Array2<f64> = exc_coeff_j
                .slice(s![state_j, .., ..])
                .dot(&qvv_shape.t())
                .into_shape([nocc_j * nat, nvirt_i])
                .unwrap();
            // dot of [nvirt_i,nocc_j*nat] and [nocc_j * nat, nocc_i]
            let temp: Array2<f64> = bj_qvv.t().dot(&g0_qoo); // nvirt_i ,nocc_i
                                                             // Dot [nocc_i,nvirt_i] [nvirt_i,nocc_i] and trace
            let exchange: f64 = (temp.dot(&exc_coeff_i.slice(s![state_i, .., ..])))
                .trace()
                .unwrap();

            coupling_matrix[[state_i, state_j]] =
                2.0 * qov_b_i.dot(&g0_ij.dot(&qov_b_j)) - exchange;
        }
    }
    return coupling_matrix;
}

pub fn le_le_two_electron_esd(
    n_le: usize,
    exc_coeff_i: ArrayView3<f64>,
    exc_coeff_j: ArrayView3<f64>,
    g0_pair: ArrayView2<f64>,
    g0_ij: ArrayView2<f64>,
    q_ov_i: ArrayView2<f64>,
    q_ov_j: ArrayView2<f64>,
) -> Array2<f64> {
    let nocc_i: usize = exc_coeff_i.dim().1;
    let nocc_j: usize = exc_coeff_j.dim().1;
    let nvirt_i: usize = exc_coeff_i.dim().2;
    let nvirt_j: usize = exc_coeff_j.dim().2;

    let mut coupling_matrix: Array2<f64> = Array2::zeros((n_le, n_le));

    for state_i in 0..n_le {
        // coulomb part
        let qov_b_i: Array1<f64> = q_ov_i.dot(
            &exc_coeff_i
                .slice(s![state_i, .., ..])
                .into_shape(nocc_i * nvirt_i)
                .unwrap(),
        );

        for state_j in 0..n_le {
            // coulomb part
            let qov_b_j: Array1<f64> = q_ov_j.dot(
                &exc_coeff_j
                    .slice(s![state_j, .., ..])
                    .into_shape(nocc_j * nvirt_j)
                    .unwrap(),
            );

            coupling_matrix[[state_i, state_j]] = 2.0 * qov_b_i.dot(&g0_ij.dot(&qov_b_j));
        }
    }
    return coupling_matrix;
}

pub fn inter_fragment_trans_charge_ct_ov(
    atoms_i: &[Atom],
    atoms_j: &[Atom],
    orbs_i: ArrayView2<f64>,
    orbs_j: ArrayView2<f64>,
    s_ij: ArrayView2<f64>,
    occ_indices_i: &[usize],
    virt_indices_j: &[usize],
    order:&str,
) -> Array2<f64> {
    // set n_atoms
    let n_atoms_i: usize = atoms_i.len();
    let n_atoms_j: usize = atoms_j.len();
    // set the number of active orbitals
    let n_occ_m: usize = occ_indices_i.len();
    let n_virt_m: usize = virt_indices_j.len();

    let virt_index_start: usize = virt_indices_j[0];
    let virt_index_end: usize = virt_indices_j[n_virt_m - 1];
    let occ_index_start: usize = occ_indices_i[0];
    let occ_index_end: usize = occ_indices_i[n_occ_m - 1];

    // calculate s_mu,nu * c_nu,a
    let s_c_j: Array2<f64> = s_ij.dot(&orbs_j.slice(s![.., virt_index_start..virt_index_end]));
    // calculate c_mu,i.T * s_mu,nu
    let c_i_s: Array2<f64> = orbs_i
        .slice(s![.., occ_index_start..occ_index_end])
        .t()
        .dot(&s_ij);
    // define separate arrays for transition charges for atoms on I and atoms on J
    // the arrays have to be appended after the calculation
    let mut qov_i: Array3<f64> = Array3::zeros([n_atoms_i, n_occ_m, n_virt_m]);
    let mut qov_j: Array3<f64> = Array3::zeros([n_atoms_j, n_occ_m, n_virt_m]);

    let mut mu: usize = 0;
    // calculate sum_mu(on atom A of I) sum_nu(on J) S_mu,nu * orbs_i_mu,i * orbs_j_nu,a
    for (n_i, atom_i) in atoms_i.iter().enumerate() {
        for _ in 0..(atom_i.n_orbs) {
            for (i, occi) in occ_indices_i.iter().enumerate() {
                for (a, virta) in virt_indices_j.iter().enumerate() {
                    qov_i[[n_i, i, a]] += 0.5 * orbs_i[[mu, *occi]] * s_c_j[[mu, a]];
                }
            }
            mu += 1;
        }
    }

    let mut nu: usize = 0;
    // calculate sum_nu(on atom A of J) sum_mu(on I) S_mu,nu * orbs_i_mu,i * orbs_j_nu,a
    for (n_j, atom_j) in atoms_j.iter().enumerate() {
        for _ in 0..(atom_j.n_orbs) {
            for (i, occi) in occ_indices_i.iter().enumerate() {
                for (a, virta) in virt_indices_j.iter().enumerate() {
                    qov_j[[n_j, i, a]] += 0.5 * orbs_j[[nu, *virta]] * c_i_s[[i, nu]];
                }
            }
            nu += 1;
        }
    }
    // append both arrays
    let mut qtrans_result: Array2<f64>
        = Array2::zeros([n_atoms_i + n_atoms_j, n_occ_m * n_virt_m]);
    if order == "ij"{
        qov_i.append(Axis(0), qov_j.view()).unwrap();
        qtrans_result = qov_i
            .into_shape([n_atoms_i + n_atoms_j, n_occ_m * n_virt_m])
            .unwrap();
    }
    else if order == "ji"{
        qov_j.append(Axis(0), qov_i.view()).unwrap();
        qtrans_result = qov_j
            .into_shape([n_atoms_i + n_atoms_j, n_occ_m * n_virt_m])
            .unwrap();
    }

    return qtrans_result;
}

pub fn inter_fragment_trans_charges_oovv(
    atoms_i: &[Atom],
    atoms_j: &[Atom],
    orbs_i: ArrayView2<f64>,
    orbs_j: ArrayView2<f64>,
    s_ij: ArrayView2<f64>,
    occ_indices_i: &[usize],
    occ_indices_j: &[usize],
    virt_indices_i: &[usize],
    virt_indices_j: &[usize],
) -> (Array2<f64>, Array2<f64>) {
    // set n_atoms
    let n_atoms_i: usize = atoms_i.len();
    let n_atoms_j: usize = atoms_j.len();

    // set indices for slicing mo coefficients
    let virt_index_start_i: usize = virt_indices_i[0];
    let virt_index_start_j: usize = virt_indices_j[0];
    let occ_index_start_i: usize = occ_indices_i[0];
    let occ_index_start_j: usize = occ_indices_j[0];
    let occ_index_end_i: usize = virt_indices_i[0] - 1;
    let occ_index_end_j: usize = virt_indices_j[0] - 1;
    let nocc_i: usize = virt_index_start_i;
    let nocc_j: usize = virt_index_start_j;
    let nvirt_i: usize = virt_indices_i.len();
    let nvirt_j: usize = virt_indices_j.len();
    let virt_index_end_i: usize = virt_indices_i[nvirt_i - 1];
    let virt_index_end_j: usize = virt_indices_j[nvirt_j - 1];

    let mut qoo_i: Array3<f64> = Array3::zeros([n_atoms_i, nocc_i, nocc_j]);
    let mut qoo_j: Array3<f64> = Array3::zeros([n_atoms_i, nocc_i, nocc_j]);
    let mut qvv_i: Array3<f64> = Array3::zeros([n_atoms_i, nvirt_i, nvirt_j]);
    let mut qvv_j: Array3<f64> = Array3::zeros([n_atoms_i, nvirt_i, nvirt_j]);

    let sc_mu_i: Array2<f64> = s_ij.dot(&orbs_j.slice(s![.., occ_index_start_j..occ_index_end_j]));
    let cs_i_lambda: Array2<f64> = orbs_i
        .slice(s![.., occ_index_start_i..occ_index_end_i])
        .t()
        .dot(&s_ij);
    let sc_nu_a: Array2<f64> =
        s_ij.dot(&orbs_j.slice(s![.., virt_index_start_j..virt_index_end_j]));
    let cs_a_sigma: Array2<f64> = orbs_i
        .slice(s![.., virt_index_start_i..virt_index_end_i])
        .t()
        .dot(&s_ij);

    let mut mu: usize = 0;
    let mut nu: usize = 0;
    for (n_i, atom_i) in atoms_i.iter().enumerate() {
        for _ in 0..(atom_i.n_orbs) {
            for (i, occi) in occ_indices_i.iter().enumerate() {
                for (j, occj) in occ_indices_j.iter().enumerate() {
                    qoo_i[[n_i, i, j]] += orbs_i[[mu, *occi]] * sc_mu_i[[mu, *occj]];
                }
            }
            for (i, virti) in virt_indices_i.iter().enumerate() {
                for (j, virtj) in virt_indices_j.iter().enumerate() {
                    qoo_i[[n_i, i, j]] += orbs_i[[nu, *virti]] * sc_nu_a[[nu, j]];
                }
            }
            mu += 1;
            nu += 1;
        }
    }
    let mut lambda: usize = 0;
    let mut sigma: usize = 0;
    for (n_j, atom_j) in atoms_j.iter().enumerate() {
        for _ in 0..(atom_j.n_orbs) {
            for (i, occi) in occ_indices_i.iter().enumerate() {
                for (j, occj) in occ_indices_j.iter().enumerate() {
                    qoo_j[[n_j, i, j]] += orbs_j[[lambda, *occj]] * cs_i_lambda[[i, lambda]];
                }
            }
            for (i, virti) in virt_indices_i.iter().enumerate() {
                for (j, virtj) in virt_indices_j.iter().enumerate() {
                    qvv_j[[n_j, i, j]] += orbs_j[[sigma, *virtj]] * cs_a_sigma[[i, sigma]];
                }
            }
            lambda += 1;
            sigma += 1;
        }
    }
    qoo_i.append(Axis(0), qoo_j.view()).unwrap();
    qvv_i.append(Axis(0), qvv_j.view()).unwrap();
    let qoo_result: Array2<f64> = qoo_i
        .into_shape([n_atoms_i + n_atoms_j, nocc_i * nocc_j])
        .unwrap();
    let qvv_result: Array2<f64> = qvv_i
        .into_shape([n_atoms_i + n_atoms_j, nvirt_i * nvirt_j])
        .unwrap();

    return (qoo_result, qvv_result);
}

pub fn le_ct_two_electron(
    n_le: usize,
    n_ct: usize,
    exc_coeff_i: ArrayView3<f64>,
    g0_pair: ArrayView2<f64>,
    g0_ij: ArrayView2<f64>,
    q_ov_i: ArrayView2<f64>,
    q_ov_j: ArrayView2<f64>,
    q_oo_inter_frag: ArrayView2<f64>,
    q_vv_inter_frag: ArrayView2<f64>,
    nocc_j: usize,
    nvirt_j: usize,
) -> Array2<f64> {
    let nocc_i: usize = exc_coeff_i.dim().1;
    let nvirt_i: usize = exc_coeff_i.dim().2;
    let n_atoms: usize = g0_pair.dim().0;
    let mut coupling_matrix: Array2<f64> = Array2::zeros([n_le, n_ct]);

    // calculate the dot product between qoo and g0
    // shape: nocc_i * nocc_j, n_atoms
    let qoo_g0: Array2<f64> = q_oo_inter_frag
        .t()
        .dot(&g0_pair)
        .into_shape([nocc_i, nocc_j * n_atoms])
        .unwrap();
    let qvv_reshaped: ArrayView2<f64> = q_vv_inter_frag
        .into_shape([n_atoms * nvirt_i, nvirt_j])
        .unwrap();

    for state in 0..n_le {
        // Coulomb term
        // excited state matrix * qov -> result: natoms_i
        let qov_exc: Array1<f64> = q_ov_i.dot(
            &exc_coeff_i
                .slice(s![state, .., ..])
                .into_shape([nocc_i * nvirt_i])
                .unwrap(),
        );
        // qov_exc * g0_ij -> natoms_i X [natoms_i,natoms_j] -> result: natoms_j
        let qov_exc_g0: Array1<f64> = qov_exc.dot(&g0_ij);
        // qov_exc_g0 * q_ov_j: result -> nocc_j * nvirt_j = n_ct
        let coulomb: Array1<f64> = 2.0 * qov_exc_g0.dot(&q_ov_j);

        // Exchange term
        let bi_qoo_g0: Array2<f64> = exc_coeff_i
            .slice(s![state, .., ..])
            .t()
            .dot(&qoo_g0)
            .into_shape([nvirt_i, nocc_j, n_atoms])
            .unwrap()
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape([nocc_j, nvirt_i * n_atoms])
            .unwrap()
            .to_owned();
        let exchange: Array1<f64> = bi_qoo_g0
            .dot(&qvv_reshaped)
            .into_shape(nocc_j * nvirt_j)
            .unwrap();

        // reorder the matrix elements of the LE-CT coupling
        let matrix_elements: Array2<f64> =
            (coulomb - exchange).into_shape([nocc_j, nvirt_j]).unwrap();
        let mut ordered_vec: Array1<f64> = Array1::zeros(nocc_j * nvirt_j);
        let mut it: usize = 0;
        for ct_i in 0..nocc_j {
            for ct_j in 0..nvirt_j {
                ordered_vec[it] = matrix_elements[[ct_i, ct_j]];
                it += 1;
            }
        }

        coupling_matrix
            .slice_mut(s![state, ..])
            .assign(&(ordered_vec));
    }
    return coupling_matrix;
}

pub struct CtHandler {
    nocc: usize,
    nvirt: usize,
    n_ct: usize,
    esd_pair:bool,
    index_mat_half: Vec<Vec<(usize, usize, usize, usize)>>,
    index_mat_full: Vec<Vec<(usize, usize, usize, usize)>>,
}

impl CtHandler {
    pub fn new(nocc: usize, nvirt: usize,esd_pair:bool) -> CtHandler {
        let n_ct: usize = nocc * nvirt;
        let index_mat: Vec<Vec<(usize, usize, usize, usize)>> = Vec::new();

        CtHandler {
            nocc: nocc,
            nvirt: nvirt,
            n_ct: n_ct,
            esd_pair:esd_pair,
            index_mat_half: index_mat.clone(),
            index_mat_full: index_mat,
        }
    }

    pub fn create_index_mat(&mut self) {
        // create two matrices with the required indices for the
        // calculation of CT-CT cis matrix elements

        let mut basis: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.nocc {
            for j in 0..self.nvirt {
                basis.push((i, j));
            }
        }

        let mut index_mat: Vec<Vec<(usize, usize, usize, usize)>> = Vec::new();
        let mut index_mat_full: Vec<Vec<(usize, usize, usize, usize)>> = Vec::new();

        for (i, base_i) in basis.iter().enumerate() {
            let mut vec: Vec<(usize, usize, usize, usize)> = Vec::new();
            let mut vec_full: Vec<(usize, usize, usize, usize)> = Vec::new();

            for (j, base_j) in basis.iter().enumerate() {
                if i <= j {
                    vec.push((base_i.0, base_i.1, base_j.0, base_j.1));
                }
                vec_full.push((base_i.0, base_i.1, base_j.0, base_j.1));
            }
            index_mat.push(vec);
            index_mat_full.push(vec_full);
        }
        self.index_mat_half = index_mat;
        self.index_mat_full = index_mat_full;
    }

    pub fn compute_ct_ct(
        &self,
        qoo_i: ArrayView2<f64>,
        qvv_j: ArrayView2<f64>,
        qov_ij: Option<ArrayView2<f64>>,
        gamma: ArrayView2<f64>,
        gamma_offdiag: ArrayView2<f64>,
    ) -> Array2<f64> {
        let mut coupling_mat: Array2<f64> = Array2::zeros((self.n_ct, self.n_ct));
        // set n_atoms
        let n_at_i: usize = qoo_i.dim().0;
        let n_at_j: usize = qvv_j.dim().0;
        let n_at_ij: usize = n_at_i+n_at_j;
        // convert transition charges from 2d to 3d arrays
        let qov_3d: ArrayView3<f64> = qov_ij.unwrap().into_shape([n_at_ij, self.nocc, self.nvirt]).unwrap();
        let qoo_3d: ArrayView3<f64> = qoo_i.into_shape([n_at_i, self.nocc, self.nocc]).unwrap();
        let qvv_3d: ArrayView3<f64> = qvv_j.into_shape([n_at_j, self.nvirt, self.nvirt]).unwrap();

        for (ind_i, vec) in self.index_mat_half.iter().enumerate() {
            for (ind_j, tuple) in vec.iter().enumerate() {
                let index_j: usize = ind_j + ind_i;
                let (i, a, j, b): (usize, usize, usize, usize) = *tuple;

                // coulomb integral (ij|ab)
                let coulomb: f64 = qoo_3d
                    .slice(s![.., i, j])
                    .dot(&gamma_offdiag.dot(&qvv_3d.slice(s![.., a, b])));

                if self.esd_pair == false{
                    // exchange integral is (ia|jb)
                    let exchange: f64 =
                        qov_3d
                            .slice(s![.., i, a])
                            .dot(&gamma.dot(&qov_3d.slice(s![.., j, b])));

                    coupling_mat[[ind_i, index_j]] = 2.0 * exchange - coulomb;
                }
                else{
                    coupling_mat[[ind_i, index_j]] = -1.0* coulomb;
                }
            }
        }
        return coupling_mat;
    }

    pub fn compute_ct_ct_off_diag(
        &self,
        qov_ij: ArrayView2<f64>,
        qov_ji: ArrayView2<f64>,
        qoo_ij: ArrayView2<f64>,
        qvv_ji: ArrayView2<f64>,
        gamma: ArrayView2<f64>,
    ) ->Array2<f64>{
        let mut coupling_mat: Array2<f64> = Array2::zeros((self.n_ct, self.n_ct));
        // set n_atoms
        let n_at: usize = qov_ij.dim().0;
        // convert transition charges from 2d to 3d arrays
        let qov_ij_3d: ArrayView3<f64> = qov_ij.into_shape([n_at, self.nocc, self.nvirt]).unwrap();
        let qov_ji_3d: ArrayView3<f64> = qov_ji.into_shape([n_at, self.nocc, self.nvirt]).unwrap();
        let qoo_3d: ArrayView3<f64> = qoo_ij.into_shape([n_at, self.nocc, self.nocc]).unwrap();
        let qvv_3d: ArrayView3<f64> = qvv_ji.into_shape([n_at, self.nvirt, self.nvirt]).unwrap();

        for (ind_i, vec) in self.index_mat_full.iter().enumerate() {
            for (ind_j, tuple) in vec.iter().enumerate() {
                let (i, a, j, b): (usize, usize, usize, usize) = *tuple;

                // exchange integral is (ia|jb)
                let exchange: f64 = qov_ij_3d
                    .slice(s![.., i, a])
                    .dot(&gamma.dot(&qov_ji_3d.slice(s![.., j, b])));

                // coulomb integral (ij|ab)
                let coulomb: f64 =
                    qoo_3d
                        .slice(s![.., i, j])
                        .dot(&gamma.dot(&qvv_3d.slice(s![.., a, b])));

                coupling_mat[[ind_i, ind_j]] = 2.0 * exchange - coulomb;
            }
        }
        return coupling_mat;
    }
}