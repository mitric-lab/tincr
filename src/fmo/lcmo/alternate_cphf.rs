use crate::initialization::Atom;
use crate::fmo::{Monomer, SuperSystem, ExcitedStateMonomerGradient};
use ndarray::prelude::*;
use ndarray_linalg::Inverse;

impl Monomer{
    pub fn mo_derivatives_maurice(&mut self,atoms:&[Atom])->Array3<f64>{
        // calculate angular derivatives
        let angular_derivatives:Array3<f64> = self.solve_angular_derivatives(atoms);
        // derivatives of overlap matrix
        let ds:ArrayView3<f64> = self.properties.grad_s().unwrap();
        // MO coefficients
        let orbs:ArrayView2<f64> = self.properties.orbs().unwrap();

        let nocc:usize = self.properties.occ_indices().unwrap().len();
        let orbs_occ:ArrayView2<f64> = orbs.slice(s![..,..nocc]);
        let orbs_virt:ArrayView2<f64> = orbs.slice(s![..,nocc..]);
        let pmat:Array2<f64> = orbs.dot(&orbs.t());

        let mut dc_mo:Array3<f64> = Array3::zeros([3*self.n_atoms,self.n_orbs,self.n_orbs]);
        for nc in 0..3*self.n_atoms{
            for orb_i in 0..self.n_orbs{
                // contribution of the overlap derivatives
                let s_term:Array1<f64> = -0.5 * pmat.t().dot(&ds.slice(s![nc,..,..])
                    .dot(&orbs.slice(s![..,orb_i])));

                // contribution of the angular derivatives
                let mut angular_term:Array1<f64> = Array1::zeros(self.n_orbs);
                if orb_i < nocc{
                    //angular_term = -1.0 * orbs_virt.dot(&angular_derivatives.slice(s![nc,..,orb_i]));
                    angular_term = -1.0*orbs_virt.dot(&angular_derivatives.slice(s![nc,..,orb_i]));
                }
                else{
                    angular_term = orbs_occ.dot(&angular_derivatives.slice(s![nc,orb_i-nocc,..]));
                }

                dc_mo.slice_mut(s![nc,..,orb_i]).assign(&(s_term+angular_term));
            }
        }

        dc_mo
    }

    pub fn solve_angular_derivatives(&mut self,atoms:&[Atom])->Array3<f64>{
        // calculate necessary matrices
        self.prepare_excited_gradient(atoms);
        // calculate matrix_aick on the left hand side
        let lhs_matrix:Array2<f64> = self.matrix_lhs_terms();
        // calculate the matrix on the right hand side
        let rhs_matrix:Array2<f64> = self.matrices_rhs_terms();
        // invert the lhs matrix
        let lhs_inv:Array2<f64> = lhs_matrix.inv().unwrap();

        // calculate the angular derivative matrix
        let angular_derivatives:Array3<f64> = -1.0 *
            self.angular_derivatives_matmul(lhs_inv.view(),rhs_matrix.view());

        return angular_derivatives;
    }

    pub fn angular_derivatives_matmul(&self, lhs_inv:ArrayView2<f64>,rhs_mat:ArrayView2<f64>)
    ->Array3<f64>
    {
        let nocc:usize = self.properties.occ_indices().unwrap().len();
        let nvirt:usize = self.properties.virt_indices().unwrap().len();

        let mut angular_derivatives:Array2<f64> = Array2::zeros([3*self.n_atoms,nvirt*nocc]);

        for nc in 0..3*self.n_atoms{
            angular_derivatives.slice_mut(s![nc,..])
                .assign(&rhs_mat.slice(s![nc,..]).dot(&lhs_inv));
        }
        angular_derivatives.into_shape([3*self.n_atoms,nvirt,nocc]).unwrap()
    }

    pub fn matrices_rhs_terms(&self)->Array2<f64>{
        let nocc:usize = self.properties.occ_indices().unwrap().len();
        let nvirt:usize = self.properties.virt_indices().unwrap().len();
        let norbs:usize = nocc + nvirt;

        let orbs:ArrayView2<f64> = self.properties.orbs().unwrap();
        let orbs_occ:ArrayView2<f64> = orbs.slice(s![..,..nocc]);
        let orbs_virt:ArrayView2<f64> = orbs.slice(s![..,nocc..]);

        // calculate derivative term depending on H
        let mut h_term:Array4<f64> = Array4::zeros([nvirt,nocc,norbs,norbs]);
        for a in 0..nvirt{
            for i in 0..nocc{
                for mu in 0..norbs{
                    for nu in 0..norbs{
                        h_term[[a,i,mu,nu]] = -2.0 * (orbs_virt[[mu,a]] * orbs_occ[[nu,i]]
                            + orbs_occ[[mu,i]] * orbs_virt[[nu,a]]);
                    }
                }
            }
        }
        let h_term:Array2<f64> = h_term.into_shape([nvirt*nocc,norbs*norbs]).unwrap();

        // calculate term depending on PI_mu,nu,sig,la
        let mut pi_term:Array6<f64> = Array6::zeros([nvirt,nocc,norbs,norbs,norbs,norbs]);

        let pmat_occ:Array2<f64> = orbs_occ.dot(&orbs_occ.t());
        for a in 0..nvirt{
            for i in 0..nocc{
                for mu in 0..norbs{
                    for nu in 0..norbs{
                        for sig in 0..norbs{
                            for la in 0..norbs{
                                let p_term:f64 = pmat_occ[[nu,la]];
                                pi_term[[a,i,mu,nu,sig,la]] = -2.0 * p_term *
                                    (orbs_virt[[mu,a]] * orbs_occ[[sig,i]] +  orbs_occ[[mu,i]] * orbs_virt[[sig,a]]);
                            }
                        }
                    }
                }
            }
        }
        let pi_term:Array2<f64> = pi_term.into_shape([nvirt*nocc,norbs*norbs*norbs*norbs]).unwrap();

        // calculate derivative term depending on S
        let fock_mat:ArrayView2<f64> = self.properties.h_coul_x().unwrap();
        let mut s_term:Array4<f64> = Array4::zeros([nvirt,nocc,norbs,norbs]);

        // terms for the first part of the equation
        let pmat_full:Array2<f64> = orbs.dot(&orbs.t());
        let fock_matmul_1:Array2<f64> = pmat_full.t().dot(&fock_mat.dot(&orbs_occ));
        let fock_matmul_2:Array2<f64> = pmat_full.t().dot(&fock_mat.dot(&orbs_virt));
        // terms for the second part of the equation
        let integrals:Array4<f64> = coulomb_exchange_integral(
            self.properties.s().unwrap(),
            self.properties.gamma_ao().unwrap(),
            self.properties.gamma_lr_ao().unwrap(),
            self.n_orbs,
        );
        // integral 1 has shape : [nvirt, norbs, nocc, norbs] = [a, alpha, i, beta]
        // integral 2 has shaoe: [nocc, norbs, nvirt, norbs] = [i, alpha, a, beta ]
        let (integrals_1,integrals_2):(Array4<f64>,Array4<f64>)
            = integral_terms_s_contribution(
            orbs_occ,
            orbs_virt,
            integrals.view(),
            pmat_full.view(),
            pmat_occ.view(),
            nocc,
            nvirt,
            norbs,
        );

        for a in 0..nvirt{
            for i in 0..nocc{
                for alpha in 0..norbs{
                    for beta in 0..norbs{
                        let term_1:f64 = 2.0 * (fock_matmul_1[[alpha,i]] * orbs_virt[[beta,a]]
                            + fock_matmul_2[[alpha,a]] * orbs_virt[[beta,i]]);
                        let term_2:f64 = 2.0 * (integrals_1[[a,alpha,i,beta]] + integrals_2[[i,alpha,a,beta]]);
                        s_term[[a,i,alpha,beta]] = term_1 + term_2;
                    }
                }
            }
        }
        let s_term:Array2<f64> = s_term.into_shape([nvirt*nocc,norbs*norbs]).unwrap();

        let h0_derivative:ArrayView3<f64> = self.properties.grad_h0().unwrap();
        let h0_derivative:ArrayView2<f64> = h0_derivative.into_shape([3*self.n_atoms,norbs*norbs]).unwrap();
        let s_derivative:ArrayView3<f64> = self.properties.grad_s().unwrap();
        let s_derivative:ArrayView2<f64> = s_derivative.into_shape([3*self.n_atoms,norbs*norbs]).unwrap();
        let mut h0_term:Array2<f64> = Array2::zeros([3*self.n_atoms,nvirt*nocc]);
        let mut grad_s_term:Array2<f64> = Array2::zeros([3*self.n_atoms,nvirt*nocc]);
        let integral_derivative:Array5<f64> =
        f_monomer_coulomb_exchange_loop(
            self.properties.s().unwrap(),
            self.properties.grad_s().unwrap(),
            self.properties.gamma_ao().unwrap(),
            self.properties.gamma_lr_ao().unwrap(),
            self.properties.grad_gamma_ao().unwrap(),
            self.properties.grad_gamma_lr_ao().unwrap(),
            self.n_atoms,
            self.n_orbs,
        );
        let integral_derivative:Array2<f64> = integral_derivative
            .into_shape([3*self.n_atoms,norbs*norbs*norbs*norbs]).unwrap();
        let mut integral_terms:Array2<f64> = Array2::zeros([3*self.n_atoms,nvirt*nocc]);

        for nc in 0..3*self.n_atoms{
            h0_term.slice_mut(s![nc,..]).assign(&h_term.dot(&h0_derivative.slice(s![nc,..])));
            grad_s_term.slice_mut(s![nc,..]).assign(&s_term.dot(&s_derivative.slice(s![nc,..])));
            integral_terms.slice_mut(s![nc,..]).assign(&pi_term.dot(&integral_derivative.slice(s![nc,..])));
        }

        let rhs_term:Array2<f64> = -1.0 *(h0_term + grad_s_term + integral_terms);
        rhs_term
    }

    pub fn matrix_lhs_terms(&self)->Array2<f64>{
        // calculate terms involving two electron integrals
        let integrals:Array2<f64> = self.angular_two_electron_terms();

        // calculate fock matrix terms
        let nocc:usize = self.properties.occ_indices().unwrap().len();
        let nvirt:usize = self.properties.virt_indices().unwrap().len();

        let fock_matrix:ArrayView2<f64> = self.properties.h_coul_x().unwrap();
        let orbs:ArrayView2<f64> = self.properties.orbs().unwrap();
        let fock_mo:Array2<f64> = orbs.t().dot(&fock_matrix.dot(&orbs));
        let mut fock_terms:Array4<f64> = Array4::zeros([nvirt,nocc,nvirt,nocc]);
        for a in 0..nvirt{
            for i in 0..nocc{
                for c in 0..nvirt{
                    for k in 0..nocc{
                        let mut f_ac:f64 = 0.0;
                        let mut f_ik:f64 = 0.0;
                        if i==k{
                            f_ac = fock_mo[[a,c]];
                        }
                        if a==c{
                            f_ik = fock_mo[[i,k]];
                        }
                        fock_terms[[a,i,c,k]] = f_ac - f_ik;
                    }
                }
            }
        }
        let fock_terms:Array2<f64> = fock_terms.into_shape([nvirt*nocc,nvirt*nocc]).unwrap();

        let matrix:Array2<f64> = 4.0 * (integrals + fock_terms);
        matrix
    }

    pub fn angular_two_electron_terms(&self)->Array2<f64>{
        let nocc:usize = self.properties.occ_indices().unwrap().len();
        let nvirt:usize = self.properties.virt_indices().unwrap().len();

        // get transition charges
        let qov:ArrayView2<f64> = self.properties.q_ov().unwrap();
        let qvv:ArrayView2<f64> = self.properties.q_vv().unwrap();
        let qoo:ArrayView2<f64> = self.properties.q_oo().unwrap();
        let qvo:Array2<f64> = qov.view().into_shape([self.n_atoms,nocc,nvirt]).unwrap()
            .to_owned()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned().into_shape([self.n_atoms,nvirt*nocc]).unwrap();

        let g0:ArrayView2<f64> = self.properties.gamma().unwrap();
        let g0_lr:ArrayView2<f64> = self.properties.gamma_lr().unwrap();

        // pi_pqrs = 2.0 *(pr|qs) - (ps|qr)
        // calculate pi_acik = 2.0 * (ai|ck) - (ak|ic)
        // equals 2.0 * (vo|vo) - (vo|ov)
        let coul:Array2<f64> = qvo.t().dot(&g0.dot(&qvo));
        let exch:Array2<f64> = qvo.t().dot(&g0_lr.dot(&qov))
            .into_shape([nvirt,nocc,nocc,nvirt]).unwrap()
            .permuted_axes([0,2,3,1])
            .as_standard_layout()
            .to_owned().into_shape([nvirt*nocc,nvirt*nocc]).unwrap();
        let integral_1:Array2<f64> = 2.0 * coul - exch;

        // calculate pi_akic = 2.0 * (ai|kc) - (ac|ki)
        // equals 2.0 * (vo|ov) - (vv|oo)
        let coul:Array2<f64> = qvo.t().dot(&g0.dot(&qov))
            .into_shape([nvirt,nocc,nocc,nvirt]).unwrap()
            .permuted_axes([0,1,3,2])
            .as_standard_layout()
            .to_owned().into_shape([nvirt*nocc,nvirt*nocc]).unwrap();
        let exch:Array2<f64> = qvv.t().dot(&g0_lr.dot(&qoo))
            .into_shape([nvirt,nvirt,nocc,nocc]).unwrap()
            .permuted_axes([0,3,1,2])
            .as_standard_layout()
            .to_owned().into_shape([nvirt*nocc,nvirt*nocc]).unwrap();
        let integral_2:Array2<f64> = 2.0 * coul - exch;

        integral_1 + integral_2
    }

}

fn integral_terms_s_contribution(
    orbs_occ:ArrayView2<f64>,
    orbs_virt:ArrayView2<f64>,
    integrals:ArrayView4<f64>,
    pmat_full:ArrayView2<f64>,
    pmat_occ:ArrayView2<f64>,
    nocc:usize,
    nvirt:usize,
    norbs:usize,
)->(Array4<f64>,Array4<f64>)
{
    let term_1:Array4<f64> = integrals.into_shape([norbs*norbs*norbs,norbs]).unwrap()
        .dot(&pmat_occ).into_shape([norbs,norbs,norbs,norbs]).unwrap()
        .permuted_axes([3,0,1,2]).as_standard_layout()
        .into_shape([norbs*norbs*norbs,norbs]).unwrap()
        .dot(&orbs_occ).into_shape([norbs,norbs,norbs,nocc]).unwrap()
        .permuted_axes([3,0,1,2]).as_standard_layout()
        .into_shape([nocc*norbs*norbs,norbs]).unwrap()
        .dot(&pmat_full).into_shape([nocc,norbs,norbs,norbs]).unwrap()
        .permuted_axes([3,0,1,2]).as_standard_layout()
        .into_shape([norbs*nocc*norbs,norbs]).unwrap()
        .dot(&orbs_virt).into_shape([norbs,nocc,norbs,nvirt]).unwrap()
        .permuted_axes([3,0,1,2]).as_standard_layout() // [nvirt, norbs, nocc, norbs]
        .to_owned();

    let term_2:Array4<f64> = integrals.into_shape([norbs*norbs*norbs,norbs]).unwrap()
        .dot(&pmat_occ).into_shape([norbs,norbs,norbs,norbs]).unwrap()
        .permuted_axes([3,0,1,2]).as_standard_layout()
        .into_shape([norbs*norbs*norbs,norbs]).unwrap()
        .dot(&orbs_virt).into_shape([norbs,norbs,norbs,nvirt]).unwrap()
        .permuted_axes([3,0,1,2]).as_standard_layout()
        .into_shape([nvirt*norbs*norbs,norbs]).unwrap()
        .dot(&pmat_full).into_shape([nvirt,norbs,norbs,norbs]).unwrap()
        .permuted_axes([3,0,1,2]).as_standard_layout()
        .into_shape([norbs*nvirt*norbs,norbs]).unwrap()
        .dot(&orbs_occ).into_shape([norbs,nvirt,norbs,nocc]).unwrap()
        .permuted_axes([3,0,1,2]).as_standard_layout() // [nocc, norbs, nvirt, norbs]
        .to_owned();

    return (term_1,term_2)
}

fn coulomb_exchange_integral(
    s: ArrayView2<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    n_orbs: usize,
) -> Array4<f64> {
    let mut coulomb_integral: Array4<f64> = Array4::zeros((n_orbs, n_orbs, n_orbs, n_orbs));
    let mut exchange_integral: Array4<f64> = Array4::zeros((n_orbs, n_orbs, n_orbs, n_orbs));
    for mu in 0..n_orbs {
        for nu in 0..n_orbs {
            for la in 0..n_orbs {
                for sig in 0..n_orbs {
                    coulomb_integral[[mu, nu, la, sig]] += 0.25
                        * s[[mu, nu]]
                        * s[[la, sig]]
                        * (g0[[mu, la]] + g0[[mu, sig]] + g0[[nu, la]] + g0[[nu, sig]]);

                    exchange_integral[[mu, nu, la, sig]] += 0.25
                        * s[[mu, la]]
                        * s[[nu, sig]]
                        * (g0_lr[[mu, nu]] + g0_lr[[mu, sig]] + g0_lr[[la, nu]] + g0_lr[[la, sig]]);
                }
            }
        }
    }
    let result: Array4<f64> = 2.0 * coulomb_integral - exchange_integral;
    result
}

fn f_monomer_coulomb_exchange_loop(
    s: ArrayView2<f64>,
    ds: ArrayView3<f64>,
    g0: ArrayView2<f64>,
    g0_lr: ArrayView2<f64>,
    dg: ArrayView3<f64>,
    dg_lr: ArrayView3<f64>,
    n_atoms: usize,
    n_orbs: usize,
) -> Array5<f64> {
    let mut coulomb_integral: Array5<f64> =
        Array5::zeros([3 * n_atoms, n_orbs, n_orbs, n_orbs, n_orbs]);
    let mut exchange_integral: Array5<f64> =
        Array5::zeros([3 * n_atoms, n_orbs, n_orbs, n_orbs, n_orbs]);

    for nc in 0..3 * n_atoms {
        for mu in 0..n_orbs {
            for nu in 0..n_orbs {
                for la in 0..n_orbs {
                    for sig in 0..n_orbs {
                        exchange_integral[[nc, mu, nu, la, sig]] += 0.25
                            * ((ds[[nc, mu, la]] * s[[nu, sig]] + s[[mu, la]] * ds[[nc, nu, sig]])
                            * (g0_lr[[mu, nu]]
                            + g0_lr[[mu, sig]]
                            + g0_lr[[la, nu]]
                            + g0_lr[[la, sig]])
                            + s[[mu, la]]
                            * s[[nu, sig]]
                            * (dg_lr[[nc, mu, nu]]
                            + dg_lr[[nc, mu, sig]]
                            + dg_lr[[nc, la, nu]]
                            + dg_lr[[nc, la, sig]]));

                        coulomb_integral[[nc, mu, nu, la, sig]] += 0.25
                            * ((ds[[nc, mu, nu]] * s[[la, sig]] + s[[mu, nu]] * ds[[nc, la, sig]])
                            * (g0[[mu, la]] + g0[[mu, sig]] + g0[[nu, la]] + g0[[nu, sig]])
                            + s[[mu, nu]]
                            * s[[la, sig]]
                            * (dg[[nc, mu, nu]]
                            + dg[[nc, mu, sig]]
                            + dg[[nc, nu, la]]
                            + dg[[nc, nu, sig]]));
                    }
                }
            }
        }
    }
    return 2.0 * coulomb_integral - exchange_integral;
}