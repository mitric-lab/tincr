use crate::fmo::*;
use crate::initialization::*;
use crate::scc::gamma_approximation::gamma_gradients_atomwise_2d;
use crate::scc::*;
use crate::utils::Timer;
use nalgebra::Vector3;
use ndarray::prelude::*;
use std::iter::FromIterator;
use std::ops::AddAssign;

impl SuperSystem {
    pub fn embedding_gradient(&mut self) -> bool {
        // initialize the gradient of the embedding energy array with zeros
        let mut gradient: Array1<f64> = Array1::zeros([self.atoms.len()]);
        let dq: ArrayView1<f64> = self.properties.dq().unwrap();

        // TODO: it is not neccessary to calculate the derivative of gamma two times. this should be
        // improved! it is already computed in the gradient of the monomer/pair
        let grad_gamma_sparse: Array2<f64> =
            gamma_gradients_atomwise_2d(&self.gammafunction, &self.atoms, self.atoms.len());
        let mut grad_gamma_dot_dq: Array1<f64> = grad_gamma_sparse.dot(&dq);
        // PARALLEL!

        // compute the gradient of the embedding energy for each pair, that is not treaded within
        // the ES-dimer approximation
        for pair in self.pairs.iter() {
            // get references to the corresponding monomers
            let m_i: &Monomer = &self.monomers[pair.i];
            let m_j: &Monomer = &self.monomers[pair.j];
            // the part of the gradients that corresponds to monomer i
            //let mut grad_i: ArrayViewMut1<f64> = gradient.slice_mut(s![m_i.grad_slice]);

            // if the derivative is w.r.t to an atom that is within this pair:
            // the first part of the equation reads:
            // dDeltaE_IJ^V/dR_a x = DDq_a^IJ sum_(K!=I,J)^(N) sum_(C in K) Dq_C^K dgamma_(a C)/dR_(a x)
            let delta_dq: ArrayView1<f64> = pair.properties.delta_dq().unwrap();
            // broadcast Delta dq into the shape of the gradients
            let delta_dq_shaped: ArrayView1<f64> = delta_dq
                .broadcast([3, pair.n_atoms])
                .unwrap()
                .reversed_axes()
                .into_shape([3 * pair.n_atoms])
                .unwrap();

            gradient.slice_mut(s![m_i.grad_slice]).add_assign(
                &(&delta_dq_shaped.slice(s![..3*m_i.n_atoms])
                    * &(&grad_gamma_dot_dq.slice(s![m_i.grad_slice])
                        - &grad_gamma_sparse
                            .slice(s![m_i.grad_slice, m_i.atom_slice])
                            .dot(&dq.slice(s![m_i.atom_slice])))),
            );

            gradient.slice_mut(s![m_j.grad_slice]).add_assign(
                &(&delta_dq_shaped.slice(s![3*m_i.n_atoms..])
                    * &(&grad_gamma_dot_dq.slice(s![m_j.grad_slice])
                    - &grad_gamma_sparse
                    .slice(s![m_j.grad_slice, m_j.atom_slice])
                    .dot(&dq.slice(s![m_j.atom_slice])))),
            );

            let grad_delta_dq: Array1<f64> = get_grad_delta_dq(pair, m_i, m_j);
            let mut esp_ij: Array1<f64> = Array1::zeros([pair.n_atoms]);
            esp_ij
                .slice_mut(s![..m_i.n_atoms])
                .assign(&m_i.properties.esp_q().unwrap());
            esp_ij
                .slice_mut(s![m_i.n_atoms..])
                .assign(&m_j.properties.esp_q().unwrap());
            esp_ij
                .broadcast([3, pair.n_atoms])
                .unwrap()
                .reversed_axes()
                .into_shape([3 * pair.n_atoms])
                .unwrap();
            let gddq_esp: Array1<f64> = &grad_delta_dq * &esp_ij;
            gradient.slice_mut(s![m_i.grad_slice]).add_assign(gddq_esp.slice(s![..3*m_i.n_atoms]));
            gradient.slice_mut(s![m_j.grad_slice]).add_assign(gddq_esp.slice(s![3*m_i.n_atoms..]));

            // if the derivative is w.r.t to an atom that is not in this pair, a -> K where K != I,J

        }

        return true;
    }
}

fn get_grad_delta_dq(pair: &Pair, m_i: &Monomer, m_j: &Monomer) -> Array1<f64> {
    // get the derivatives of the charge differences w.r.t to the dof
    let grad_dq: ArrayView2<f64> = pair.properties.grad_dq().unwrap();
    let grad_dq_i: ArrayView2<f64> = m_i.properties.grad_dq().unwrap();
    let grad_dq_j: ArrayView2<f64> = m_j.properties.grad_dq().unwrap();

    // compute the difference between dimers and monomers and take the diagonal values
    let mut grad_delta_dq_2d: Array2<f64> = Array2::zeros([3 * pair.n_atoms, pair.n_atoms]);

    // difference for monomer i
    grad_delta_dq_2d
        .slice_mut(s![..(3 * m_i.n_atoms), ..m_i.n_atoms])
        .assign(&(&grad_dq.slice(s![..(3 * m_i.n_atoms), ..m_i.n_atoms]) - &grad_dq_i));

    // difference for monomer j
    grad_delta_dq_2d
        .slice_mut(s![(3 * m_i.n_atoms).., m_i.n_atoms..])
        .assign(&(&grad_dq.slice(s![(3 * m_i.n_atoms).., m_i.n_atoms..]) - &grad_dq_j));
    let grad_delta_dq_3d: Array3<f64> = grad_delta_dq_2d
        .into_shape([3, pair.n_atoms, pair.n_atoms])
        .unwrap();

    // create a temporary array to store the values it will be flattened afterwards
    let mut grad_delta_dq: Array2<f64> = Array2::zeros([3, pair.n_atoms]);

    // take the diagonal of each of the three dimensions
    for i in 0..3 {
        grad_delta_dq
            .slice_mut(s![i, ..])
            .assign(&grad_delta_dq_3d.slice(s![i, .., ..]).diag());
    }
    // reshape it into a one dimensional array
    grad_delta_dq.into_shape([3 * pair.n_atoms]).unwrap()
}
