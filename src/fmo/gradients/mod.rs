use crate::fmo::*;
use crate::initialization::*;
use crate::scc::*;
use crate::utils::Timer;
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::iter::FromIterator;
use std::ops::AddAssign;

mod embedding;
mod monomer;
mod pair;

pub use monomer::*;
pub use pair::*;

pub trait GroundStateGradient {
    fn get_grad_dq(&self, grad_s: ArrayView3<f64>, p: ArrayView2<f64>) -> Array2<f64>;
    fn scc_gradient(&mut self) -> Array1<f64>;
}

impl SuperSystem {
    pub fn ground_state_gradient(&mut self) -> Array1<f64> {

        let monomer_gradient: Array1<f64> = self.monomer_gradients();
        let pair_gradient: Array1<f64> = self.pair_gradients(monomer_gradient.view());
        let embedding_gradient: Array1<f64> = self.embedding_gradient();

        return monomer_gradient;
    }


    fn monomer_gradients(&mut self) -> Array1<f64> {
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);
        for mol in self.monomers.iter_mut() {
            gradient
                .slice_mut(s![mol.grad_slice])
                .assign(&mol.scc_gradient());
        }
        return gradient;
    }

    fn pair_gradients(&mut self, monomer_gradient: ArrayView1<f64>) -> Array1<f64> {
        let mut gradient: Array1<f64> = Array1::zeros([3 * self.atoms.len()]);
        for pair in self.pairs.iter_mut() {
            // get references to the corresponding monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];

            // compute the gradient of the pair
            let pair_grad: Array1<f64> = pair.scc_gradient();
            // subtract the monomer contributions and assemble it into the gradient
            gradient.slice_mut(s![m_i.grad_slice]).add_assign(
                &(&pair_grad.slice(s![0..m_i.n_atoms]) - &monomer_gradient.slice(s![m_i.grad_slice])),
            );
            gradient.slice_mut(s![m_j.grad_slice]).add_assign(
                &(&pair_grad.slice(s![m_i.n_atoms..]) - &monomer_gradient.slice(s![m_j.grad_slice])),
            );
        }
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
