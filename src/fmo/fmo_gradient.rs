use crate::fmo::{atomvec_to_aomat, Monomer, Pair, SuperSystem};
use crate::initialization::parameters::RepulsivePotential;
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{
    gamma_ao_wise, gamma_atomwise, gamma_atomwise_ab, gamma_gradients_atomwise,
};
use crate::scc::h0_and_s::{h0_and_s, h0_and_s_ab, h0_and_s_gradients};
use crate::scc::mixer::{BroydenMixer, Mixer};
use crate::scc::mulliken::mulliken;
use crate::scc::scc_routine::{RestrictedSCC, SCCError};
use crate::scc::{
    construct_h1, density_matrix, density_matrix_ref, get_electronic_energy, get_repulsive_energy,
    lc_exact_exchange,
};
use crate::utils::Timer;
use approx::AbsDiffEq;
use log::info;
use nalgebra::Vector3;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::prelude::*;
use ndarray::stack;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use ndarray_stats::QuantileExt;
use nshare::ToNdarray1;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use std::iter::FromIterator;
use std::ops::{SubAssign, AddAssign};

pub trait GroundStateGradient {
    fn scc_gradient(&mut self) -> Array1<f64>;
}

impl GroundStateGradient for Monomer {
    fn scc_gradient(&mut self) -> Array1<f64> {
        // for the evaluation of the gradient it is necessary to compute the derivatives
        // of: - H0
        //     - S
        //     - Gamma
        //     - Repulsive Potential
        // the first three properties are calculated here at the beginning and the gradient that
        // originates from the repulsive potential is added at the end to total gradient

        // derivative of H0 and S
        let (grad_s, grad_h0) = h0_and_s_gradients(&self.atoms, self.n_orbs, &self.slako);

        // and reshape them into a 2D array. the last two dimension (number of orbitals) are compressed
        // into one dimension to be able to just matrix-matrix products for the computation of the gradient
        let grad_s: Array2<f64> = grad_s
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();
        let grad_h0: Array2<f64> = grad_h0
            .into_shape([3 * self.n_atoms, self.n_orbs * self.n_orbs])
            .unwrap();

        // derivative of the gamma matrix and transform it in the same way to a 2D array
        let grad_gamma: Array2<f64> =
            gamma_gradients_atomwise(&self.gammafunction, &self.atoms, self.n_atoms)
                .into_shape([3 * self.n_atoms, self.n_atoms * self.n_atoms])
                .unwrap();

        // take references/views to the necessary properties from the scc calculation
        let gamma: ArrayView2<f64> = self.properties.gamma().unwrap();
        let p: ArrayView2<f64> = self.properties.p().unwrap();
        let h0: ArrayView2<f64> = self.properties.h0().unwrap();
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();
        let s: ArrayView2<f64> = self.properties.s().unwrap();

        // transform the expression Sum_c_in_X (gamma_AC + gamma_aC) * dq_C
        // into matrix of the dimension (norb, norb) to do an element wise multiplication with P
        let mut esp_mat: Array2<f64> =
            atomvec_to_aomat(gamma.dot(&dq).view(), self.n_orbs, &self.atoms) * 0.5;
        let esp_x_p: Array1<f64> = (&p * &esp_mat)
            .into_shape([self.n_orbs * self.n_orbs])
            .unwrap();
        let p_flat: ArrayView1<f64> = p.into_shape([self.n_orbs * self.n_orbs]).unwrap();

        // the gradient part which involves the gradient of the gamma matrix is given by:
        // 1/2 * dq . dGamma / dR . dq
        // the dq's are element wise multiplied into a 2D array and reshaped into a flat one, that
        // has the length of natoms^2. this allows to do only a single matrix vector product of
        // 'grad_gamma' with 'dq_x_dq' and avoids to reshape dGamma multiple times
        let dq_column: ArrayView2<f64> = dq.clone().insert_axis(Axis(1));
        let dq_x_dq: Array1<f64> = (&dq_column.broadcast((self.n_atoms, self.n_atoms)).unwrap()
            * &dq)
            .into_shape([self.n_atoms * self.n_atoms])
            .unwrap();

        // compute the energy weighted density matrix: W = 1/2 * D . (H + H_Coul) . D
        let w: Array1<f64> = 0.5
            * (p.dot(&(&h0 + &(&esp_mat * &s))).dot(&p))
                .into_shape([self.n_orbs * self.n_orbs])
                .unwrap();

        // calculation of the gradient
        // 1st part:  dH0 / dR . P
        let mut gradient: Array1<f64> = grad_h0.dot(&p_flat);

        // 2nd part: dS / dR . W
        gradient -= &grad_s.dot(&w);

        // 3rd part: 1/2 * dS / dR * sum_c_in_X (gamma_ac + gamma_bc) * dq
        gradient += &grad_s.dot(&esp_x_p);

        // 4th part: 1/2 * dq . dGamma / dR . dq
        gradient += &(grad_gamma.dot(&dq_x_dq) * 0.5);

        // last part: dV_rep / dR
        gradient = gradient + gradient_v_rep(&self.atoms, &self.vrep);

        return gradient;
    }
}

//  Compute the gradient of the repulsive potential
//  Parameters:
//  ===========
//  atomlist: list of tuples (Zi, [xi,yi,zi]) for each atom
//  distances: matrix with distances between atoms, distance[i,j]
//    is the distance between atoms i and j
//  directions: directions[i,j,:] is the unit vector pointing from
//    atom j to atom i
//  VREP: dictionary, VREP[(Zi,Zj)] has to be an instance of RepulsivePotential
//    for the atom pair Zi-Zj
fn gradient_v_rep(atoms: &[Atom], v_rep: &RepulsivePotential) -> Array1<f64> {
    let n_atoms: usize = atoms.len();
    let mut grad: Array1<f64> = Array1::zeros([3 * n_atoms]);
    for (i, atomi) in atoms.iter().enumerate() {
        let mut grad_i: Array1<f64> = Array::zeros([3]);
        for (j, atomj) in atoms.iter().enumerate() {
            if i != j {
                let mut r: Vector3<f64> = atomi - atomj;
                let r_ij: f64 = r.norm();
                r /= r_ij;
                let v_ij_deriv: f64 = v_rep.get(atomi.kind, atomj.kind).spline_deriv(r_ij);
                r *= v_ij_deriv;
                // let v: ArrayView1<f64> = unsafe {
                //     ArrayView1::from_shape_ptr(
                //         (r.shape().0, ).strides((r.strides().0, )),
                //         r.as_ptr(),
                //     )
                // };
                let v = Array1::from_iter(r.iter());
                grad_i = &grad_i + &v;
            }
        }
        grad.slice_mut(s![i * 3..i * 3 + 3]).assign(&grad_i);
    }
    return grad;
}

impl Monomer {
    /// Compute the derivative of the partial charges according to equation 24 and 26 in
    /// https://pubs.acs.org/doi/pdf/10.1021/ct500489d.
    fn get_grad_dq(&mut self, grad_s: ArrayView3<f64>, d: ArrayView2<f64>) -> Array2<f64> {
        // get the shape of the derivative of S, it should be [f, n_orb, n_orb], where f = 3 * natoms
        let (f, n_orb, _): (usize, usize, usize) = grad_s.dim();
        // reshape S' so that the last dimension can be contracted with the density matrix
        let grad_s: ArrayView2<f64> = grad_s.into_shape([f * n_orb, n_orb]).unwrap();
        // compute W according to eq. 26 in the reference stated above in matrix fashion
        // S'[f * rho, sigma] . D[sigma, nu] -> X1[f * rho, nu]
        // X1.T[nu, f * rho]  -reshape-> X2[nu * f, rho]
        // X2[nu * f, rho] . D[mu, rho] -> X3 [nu * f, mu];  since D is symmetric -> D = D.T
        // X3[nu * f, mu] -reshape-> X3[nu, f * mu]
        // X3.T[f * mu, nu] -reshape-> W[f, mu, nu]
        let w_a_mu: Array3<f64> = -0.5 * grad_s
            .dot(&d)
            .reversed_axes()
            .into_shape([norb * f, n_orb])
            .unwrap()
            .dot(&d)
            .into_shape([norb, f * n_orb])
            .unwrap()
            .reversed_axes()
            .into_shape([f, norb, norb]);

        let mut grad_dq:  Array2<f64> = Array2::zeros([f, self.n_atoms]);
        let mut mu: usize = 0;
        for (idx, atom) in self.atoms.iter().enumerate() {
            for _ in atom.valorbs.iter() {
                grad_dq.slice_mut(slice![.., idx]).add_assign(w_a_mu.slice(s![.., mu]));
                mu += 1;
            }
        }
        return grad_dq;
    }
}

