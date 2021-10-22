use ndarray::prelude::*;

use derive_builder::*;
use crate::io::settings::LC;

// Types to specify if the transformations are done for occupied-occupied (OO), occupied-virtual (OV)
// or virtual-virtual (VV) orbitals.
#[derive(Copy, Clone)]
pub enum MOSpace {
    OO,
    VV,
    OV,
}

type Hpq = MOSpace;
type Vrs = MOSpace;

/// Linear operators H+ and H- as defined in 8.51 and 8.52 in
/// A. Humeniuk (Dissertation) 2018
/// Methods for Simulating Light-Induced Dynamics in Large Molecular Systems
/// https://refubium.fu-berlin.de/handle/fub188/23194
///
/// An instance of `HPlusMinus` can be created by calling the associated Builder type.
/// An example is shown below. Note, it is optional to provide the gamma lr matrix. But if it is
/// not provided, there will only be reasonable results if lc is set to `LC::OFF`.
/// ```
/// HPlusMinusBuilder::default()
///       .q_oo(q_trans_oo.view())
///       .q_vv(q_trans_vv.view())
///       .q_ov(q_trans_ov.view())
///       .gamma(gamma.view())
///       .gamma_lr(gamma_lr.view()) // optional
///       .lc(LC::ON)
///       .build()
///       .unwrap();
/// ```
#[derive(Builder)]
pub struct HPlusMinus<'a>
{
    /// Transition charges between occupied and occupied MOs.
    q_oo: ArrayView2<'a, f64>,
    /// Transition charges between virtual and virtual MOs.
    q_vv: ArrayView2<'a, f64>,
    /// Transition charges between occupied and virtual MOs.
    q_ov: ArrayView2<'a, f64>,
    /// Transition charges between virtual and occupied MOs.
    #[builder(default = "self.default_q_vo()?")]
    q_vo: Array2<f64>,
    /// Gamma matrix.
    gamma: ArrayView2<'a, f64>,
    /// Screened gamma matrix.
    gamma_lr: Option<ArrayView2<'a, f64>>,
    /// Whether the long-range correction is used or not.
    lc: LC,
    /// Number of atoms.
    #[builder(default = "self.default_n_atoms()?")]
    n_atoms: usize,
    /// Number of occupied orbitals.
    #[builder(default = "self.default_n_occ()?")]
    n_occ: usize,
    /// Number of virtual orbitals.
    #[builder(default = "self.default_n_virt()?")]
    n_virt: usize
}

impl HPlusMinusBuilder<'_> {
    // Private helper method with access to the builder struct.
    fn default_n_atoms(&self) -> Result<usize, String> {
        match self.q_oo {
            Some(q)  => Ok(q.dim().0),
            _ => Err("q_oo is missing".to_string()),
        }
    }

    fn default_n_occ(&self) -> Result<usize, String> {
        match self.q_oo {
            Some(q)  => Ok((q.dim().1 as f64).sqrt() as usize),
            _ => Err("q_oo is missing".to_string()),
        }
    }

    fn default_n_virt(&self) -> Result<usize, String> {
        match self.q_vv {
            Some(q)  => Ok((q.dim().1 as f64).sqrt() as usize),
            _ => Err("q_vv is missing".to_string()),
        }
    }

    fn default_q_vo(&self) -> Result<Array2<f64>, String> {
        match self.q_ov {
            Some(q)  => {
                // Number of occupied orbitals.
                let n_occ: usize = self.default_n_occ().unwrap();
                // Number of virtual orbitals.
                let n_virt: usize = self.default_n_virt().unwrap();
                // Number of atoms.
                let n_atoms: usize = self.default_n_atoms().unwrap();
                // Create q ov from q vo.
                let q_vo: Array2<f64> = q
                    .into_shape([n_atoms, n_occ, n_virt])
                    .unwrap()
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_shape([n_atoms, n_virt * n_occ])
                    .unwrap()
                    .to_owned();
                Ok(q_vo)},
            _ => Err("q_ov is missing".to_string()),
        }
    }
}


impl<'a> HPlusMinus<'a>
{
    /// The H+ operator is applied on the vector `v`.
    pub fn comp_plus(&mut self, v: ArrayView2<f64>, pq: Hpq, rs: Vrs) -> Array2<f64> {
        let t1: Array1<f64> = self.term_one(v.view(), pq, rs);
        let result: Array1<f64> = match self.lc {
            LC::OFF => t1,
            LC::ON => t1 - (self.term_two(v.view(), pq, rs) + self.term_three(v.view(), pq, rs)),
        };
        // Reshape the result into a 2D array.
        let (p_dim, q_dim): (usize, usize) = self.get_dim(pq);
        result.into_shape([p_dim, q_dim]).unwrap().to_owned()
    }

    /// The H- operator is applied on the vector `v`.
    pub fn comp_minus(&mut self, v: ArrayView2<f64>, pq: Hpq, rs: Vrs) -> Array2<f64> {
        let result: Array1<f64> = match self.lc {
            LC::OFF => panic!("This can not be possible, H minus is only needed if LC is on"),
            LC::ON => self.term_two(v.view(), pq, rs) - self.term_three(v.view(), pq, rs),
        };
        // Reshape the result into a 2D array.
        let (p_dim, q_dim): (usize, usize) = self.get_dim(pq);
        result.into_shape([p_dim, q_dim]).unwrap().to_owned()
    }

    /// The first term is computed:
    /// T_pq = sum_{rsAB} 2 x q_pq x gamma_AB x q_rs x V_rs
    /// where p,q,r,s are MO indices and A,B are atom indices.
    fn term_one(&self, v: ArrayView2<f64>, pq: Hpq, rs: Vrs) -> Array1<f64> {
        let q_a_pq_gamma_ab: Array2<f64> = match pq {
            Hpq::OO => self.gamma.dot(&self.q_oo),
            Hpq::OV => self.gamma.dot(&self.q_ov),
            Hpq::VV => self.gamma.dot(&self.q_vv),
        };
        let q_rs_v_rs: Array1<f64> = match rs {
            Vrs::OO => self.q_oo.dot(&flatten(v)),
            Vrs::OV => self.q_ov.dot(&flatten(v)),
            Vrs::VV => self.q_vv.dot(&flatten(v)),
        };
        2.0 * q_rs_v_rs.dot(&q_a_pq_gamma_ab)
    }

    /// The second term is computed (only if the LC correction is used, cx = 1):
    /// sum_{rsAB} cx x q_ps x gamma_lr_AB x q_rq x V_rs
    /// where p,q,r,s are MO indices and A,B are atom indices.
    fn term_two(&mut self, v: ArrayView2<f64>, pq: Hpq, rs: Vrs) -> Array1<f64> {
        // The necessary transition charges are defined based upon the indices of H and V.
        let (q_ps, q_rq): (ArrayView2<f64>, ArrayView2<f64>) = match (pq, rs) {
            (Hpq::OO, Vrs::OO) => (self.q_oo.view(), self.q_oo.view()),
            (Hpq::OO, Vrs::OV) => (self.q_ov.view(), self.q_oo.view()),
            (Hpq::OO, Vrs::VV) => (self.q_ov.view(), self.q_vo.view()),
            (Hpq::OV, Vrs::OO) => (self.q_oo.view(), self.q_ov.view()),
            (Hpq::OV, Vrs::OV) => (self.q_ov.view(), self.q_ov.view()),
            (Hpq::OV, Vrs::VV) => (self.q_ov.view(), self.q_vv.view()),
            (Hpq::VV, Vrs::OO) => (self.q_vo.view(), self.q_ov.view()),
            (Hpq::VV, Vrs::OV) => (self.q_vv.view(), self.q_ov.view()),
            (Hpq::VV, Vrs::VV) => (self.q_vv.view(), self.q_vv.view()),
        };
        // Reference to the long-range corrected gamma matrix.
        let gamma_lr: &ArrayView2<f64> = self.gamma_lr.as_ref().unwrap();
        // The number of atoms.
        let n_atoms: usize = self.n_atoms;
        // The dimension of the sum over p and q.
        let (p_dim, q_dim): (usize, usize) = self.get_dim(pq);
        // The dimension of the sum over r and s.
        let (r_dim, s_dim): (usize, usize) = self.get_dim(rs);

        // Product: gamma_lr_AB [A, B] . q_ps [A, p x s] -> [B, p x s], notice gamma_lr is symmetric
        let gamma_ab_q_ps: Array2<f64> = gamma_lr.dot(&q_ps);

        // The array is reshaped from [B, p x s] into [s x B, p]
        let gamma_ab_q_ps = gamma_ab_q_ps
            .into_shape([n_atoms, p_dim, s_dim])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .into_shape([s_dim * n_atoms, p_dim])
            .unwrap()
            .to_owned();

        // The transition charges are reshaped from [B, r x q] into [r, B x q]
        let q_rq = q_rq
            .into_shape([n_atoms, r_dim, q_dim])
            .unwrap()
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape([r_dim, n_atoms * q_dim])
            .unwrap()
            .to_owned();

        // V_rs.T [s, r] . q_rq [r, B x q] -> [s, B x q]
        let v_q_rq: Array2<f64> = v.t().dot(&q_rq);

        // The array is reordered into the shape [s x B, q]
        let v_q_rq = v_q_rq.into_shape([s_dim * n_atoms, q_dim]).unwrap();

        // [p, s x B] . [s x B, q]
        Array::from_iter(gamma_ab_q_ps.t().dot(&v_q_rq).into_iter())
    }

    /// The third term is computed (also only if the LC correction is used, cx = 1).
    /// sum_{rsAB} cx x q_pr x gamma_lr_AB x q_sq x V_rs
    fn term_three(&mut self, v: ArrayView2<f64>, pq: Hpq, rs: Vrs) -> Array1<f64> {
        // The necessary transition charges are defined based upon the indices of H and V.
        let (q_pr, q_sq): (ArrayView2<f64>, ArrayView2<f64>) = match (pq, rs) {
            (Hpq::OO, Vrs::OO) => (self.q_oo.view(), self.q_oo.view()),
            (Hpq::OO, Vrs::OV) => (self.q_oo.view(), self.q_vo.view()),
            (Hpq::OO, Vrs::VV) => (self.q_ov.view(), self.q_vo.view()),
            (Hpq::OV, Vrs::OO) => (self.q_oo.view(), self.q_ov.view()),
            (Hpq::OV, Vrs::OV) => (self.q_oo.view(), self.q_vv.view()),
            (Hpq::OV, Vrs::VV) => (self.q_ov.view(), self.q_vv.view()),
            (Hpq::VV, Vrs::OO) => (self.q_vo.view(), self.q_ov.view()),
            (Hpq::VV, Vrs::OV) => (self.q_vo.view(), self.q_vv.view()),
            (Hpq::VV, Vrs::VV) => (self.q_vv.view(), self.q_vv.view()),
        };
        // Reference to the long-range corrected gamma matrix.
        let gamma_lr: &ArrayView2<f64> = self.gamma_lr.as_ref().unwrap();
        // The number of atoms.
        let n_atoms: usize = self.n_atoms;
        // The dimension of the sum over p and q.
        let (p_dim, q_dim): (usize, usize) = self.get_dim(pq);
        // The dimension of the sum over r and s.
        let (r_dim, s_dim): (usize, usize) = self.get_dim(rs);

        // Product: gamma_lr_AB [A, B] . q_ps [A, p x r] -> [B, p x r], notice gamma_lr is symmetric
        let gamma_ab_q_pr: Array2<f64> = gamma_lr.dot(&q_pr);

        // The array is reshaped from [B, p x r] into [r x B, p]
        let gamma_ab_q_pr = gamma_ab_q_pr
            .into_shape([n_atoms, p_dim, r_dim])
            .unwrap()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .into_shape([r_dim * n_atoms, p_dim])
            .unwrap()
            .to_owned();

        // The transition charges are reshaped from [B, s x q] into [s, B x q]
        let q_sq = q_sq
            .into_shape([n_atoms, s_dim, q_dim])
            .unwrap()
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .into_shape([s_dim, n_atoms * q_dim])
            .unwrap()
            .to_owned();

        // V_rs [r, s] . q_sq [s, B x q] -> [r, B x q]
        let v_q_sq: Array2<f64> = v.dot(&q_sq);

        // The array is reordered into the shape [r x B, q]
        let v_q_sq = v_q_sq.into_shape([s_dim * n_atoms, q_dim]).unwrap();

        // [p, r x B] . [r x B, q]
        Array::from_iter(gamma_ab_q_pr.t().dot(&v_q_sq).into_iter())
    }

    /// Returns the dimension of either p and q or r and s.
    fn get_dim(&self, mn: MOSpace) -> (usize, usize) {
        // Number of occupied orbitals.
        let n_occ: usize = self.n_occ;
        // Number of virtual orbitals.
        let n_virt: usize = self.n_virt;
        // The tuple of dimension is returned.
        match mn {
            MOSpace::OO => (n_occ, n_occ),
            MOSpace::OV => (n_occ, n_virt),
            MOSpace::VV => (n_virt, n_virt),
        }
    }

}

fn flatten(array: ArrayView2<f64>) -> ArrayView1<f64> {
    array.into_shape([array.dim().0 * array.dim().1]).unwrap()
}