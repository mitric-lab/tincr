mod h_operators;
mod zvector;
mod vector_products;

use ndarray::prelude::*;



#[derive(Copy, Clone)]
pub enum TDA {
    On,
    Off,
}

enum VTypes {}

pub trait ExcitedStateGradient {
    fn tda_or_tddft(&self) -> TDA;

    fn get_overlap(&self) -> ArrayView2<f64>;

    fn get_h0_s_deriv(&self) -> (ArrayView3<f64>, ArrayView3<f64>);

    fn get_excitation_energy(&self) -> f64;

    fn get_occ_coeffs(&self) -> ArrayView2<f64>;

    fn get_virt_coeffs(&self) -> ArrayView2<f64>;

    fn get_occ_energies(&self) -> ArrayView2<f64>;

    fn get_virt_energies(&self) -> ArrayView2<f64>;

    fn get_transition_charges(&self) -> (ArrayView2<f64>, ArrayView2<f64>, ArrayView2<f64>);

    fn get_density_matrix(&self) -> ArrayView2<f64>;

    fn get_gamma(&self) -> ArrayView2<f64>;

    fn get_gamma_lr(&self) -> ArrayView2<f64>;

    fn get_gamma_ao(&self) -> ArrayView2<f64>;

    fn get_gamma_ao_deriv(&self) -> ArrayView3<f64>;

    fn get_gamma_ao_lr(&self) -> ArrayView2<f64>;

    fn get_gamma_ao_lr_deriv(&self) -> ArrayView3<f64>;

    fn get_x_plus_y(&self) -> ArrayView2<f64>;

    fn get_x_minus_y(&self) -> ArrayView2<f64>;

    fn gradient(&self) {
        let x_plus_y: ArrayView2<f64> = self.get_x_plus_y();
        let x_minus_y: ArrayView2<f64> = self.get_x_minus_y();
        let e_i: ArrayView2<f64> = self.get_occ_energies();
        let e_a: ArrayView2<f64> = self.get_virt_energies();
        let tda: TDA = self.tda_or_tddft();

        // The U vectors are computed.
        let _u_ab: Array2<f64> = compute_u(x_plus_y.view(), x_minus_y.view(), tda);
        let _u_ij: Array2<f64> = compute_u(x_plus_y.t(), x_minus_y.t(), tda);

        // The V vectors are computed.
        let _v_ab: Array2<f64> = compute_v(x_plus_y.t(), x_minus_y.t(), e_i.view(), tda);
        let _v_ij: Array2<f64> = compute_v(x_plus_y.view(), x_minus_y.view(), e_a.view(), tda);

        // The T vectors are computed.
        let _t_ab: Array2<f64> = compute_u(x_plus_y.view(), x_minus_y.view(), tda);
        let _t_ij: Array2<f64> = compute_u(x_plus_y.t(), x_minus_y.t(), tda);
    }
}


fn compute_u(x_plus_y: ArrayView2<f64>, x_minus_y: ArrayView2<f64>, tda: TDA) -> Array2<f64> {
    match tda {
        TDA::On => 2.0 * x_plus_y.t().dot(&x_plus_y),
        TDA::Off => x_plus_y.t().dot(&x_minus_y) + x_minus_y.t().dot(&x_plus_y),
    }
}

fn compute_v(
    x_plus_y: ArrayView2<f64>,
    x_minus_y: ArrayView2<f64>,
    e: ArrayView2<f64>,
    tda: TDA,
) -> Array2<f64> {
    match tda {
        TDA::On => 2.0 * x_plus_y.dot(&e).dot(&x_plus_y.t()),
        TDA::Off => x_plus_y.dot(&e).dot(&x_plus_y.t()) + x_minus_y.dot(&e).dot(&x_minus_y.t()),
    }
}

fn compute_t(x_plus_y: ArrayView2<f64>, x_minus_y: ArrayView2<f64>, tda: TDA) -> Array2<f64> {
    match tda {
        TDA::On => x_plus_y.t().dot(&x_plus_y),
        TDA::Off => x_plus_y.t().dot(&x_plus_y) + x_minus_y.t().dot(&x_minus_y),
    }
}
