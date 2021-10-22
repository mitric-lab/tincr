use ndarray::prelude::*;
use derive_builder::*;
use core::::gradients::h_operators::LC;

/// Matrix products for iterative solvers.
///
/// An instance of `AplusB` can be created by calling the associated Builder type.
/// An example is shown below. Note, it is optional to provide the gamma lr matrix. But if it is
/// not provided, there will only be reasonable results if lc is set to `LC::OFF`.
/// ```
/// AplusBBuilder::default()
///       .omega(omega.view())
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
pub struct AplusB<'a> {
    // The orbital energy differences.
    omega: ArrayView1<'a, f64>,
    /// Transition charges between occupied and occupied MOs.
    q_oo: ArrayView2<'a, f64>,
    /// Transition charges between virtual and virtual MOs.
    q_vv: ArrayView2<'a, f64>,
    /// Transition charges between occupied and virtual MOs.
    q_ov: ArrayView2<'a, f64>,
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
    n_virt: usize,
}

impl AplusBBuilder<'_> {
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
}


impl AplusB<'_> {
    /// Compute the matrix product for the iterative Z-vector solver.
    /// u_ia = sum_jb (A+B)_{ia,jb} v_{jb}
    fn times(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let mut u_ia: Array1<f64> = &self.omega * &v;
        u_ia += &self.coulomb(v.view());
        match self.lc {
            LC::ON => {
                u_ia += &self.exchange1(v.view());
                u_ia += &self.exchange2(v.view());
            },
            LC::OFF => {},
        }
        u_ia
    }

    /// Coulomb contribution:
    /// u_ia = sum_{A,B,j,b} q_{A,i,a} γ_{A,B} q_{B,j,b} V_{j,b}
    fn coulomb(&self, v: ArrayView1<f64>) -> Array1<f64> {
        self.q_ov.dot(&v).dot(&self.gamma).dot(&self.q_ov)
    }

    /// Exchange contribution:
    /// u_ia = sum_{A,B,j,b} q_{A,i,j} γ_{A,B} q_{B,a,b} V_{j,b}
    fn exchange1(&self, v: ArrayView1<f64>) -> Array1<f64> {
        self.gamma_lr.unwrap() // [natoms, natoms]
            .dot(&self.q_oo) // [natoms, nocc * nocc]
            .into_shape([self.atoms.len() * self.n_occ, self.n_occ]) // => [natoms * nocc, nocc]
            .unwrap()
            .t() // T -> [nocc, natoms * nocc]
            .dot(
                &self.q_vv // [natoms, nvirt * nvirt]
                    .into_shape([self.atoms.len() * self.n_virt, self.n_virt])
                    .unwrap() // => [natoms * nvirt, nvirt]
                    .dot(&v.into_shape([self.n_occ, self.n_virt]).unwrap().t()) // v: [nocc, nvirt]
                    .into_shape([self.atoms.len(), self.n_virt, self.n_occ]) // => [natoms, nvirt, nocc]
                    .unwrap()
                    .permuted_axes([0, 2, 1]) // => [natoms, nocc, nvirt]
                    .as_standard_layout()
                    .into_shape([self.atoms.len() * self.n_occ, self.n_virt]) // [natoms * nocc, nvirt]
                    .unwrap(), // => [nocc, natoms * nocc] . [natoms * nocc, nvirt]
            ) // => [nocc, nvirt]
            .into_shape([self.n_occ * self.n_virt])
            .unwrap()
    }

    /// Exchange contribution:
    /// u_ia = sum_{A,B,j,b} q_{A,j,a} γ_{A,B} q_{B,i,b} V_{j,b}
    fn exchange2(&self, v: ArrayView1<f64>) -> Array1<f64> {
        self.q_ov // [natoms, nocc * nvirt]
            .into_shape([self.atoms.len() * self.n_occ, self.n_virt])
            .unwrap() // => [natoms * nocc, nvirt]
            .dot(&v.into_shape([self.n_occ, self.n_virt]).unwrap().t()) // v.T : [nvirt, nocc]
            .t() // => [nocc, natoms * nocc]
            .dot(
                &self.gamma_lr.unwrap()// [natoms, natoms]
                    .dot(&self.q_ov) // => [natoms, nocc * nvirt]
                    .into_shape([self.atoms.len() * self.n_occ, self.n_virt]) // [natoms * nocc, nvirt]
                    .unwrap()
            ) // => [nocc, nvirt]
            .into_shape([self.n_occ * self.n_virt])
            .unwrap()
    }
}
