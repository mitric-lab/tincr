use ndarray::prelude::*;
use crate::excited_states::ProductCache;
use crate::data::{Storage, ExcitedStateData};
use nalgebra::Vector3;
use std::ops::AddAssign;


impl<'a> Storage<'a> {
    pub fn omega(&self) -> ArrayView1<f64> {
        match &self.excited_states.omega {
            Some(value) => value.view(),
            None => panic!("ExcitedStatesData::omega; The omega vector is not set!"),
        }
    }

    pub fn cis_eigenvalues(&self) -> ArrayView1<f64> {
        match &self.excited_states.cis_eigenvalues {
            Some(value) => value.view(),
            None => panic!("ExcitedStatesData::cis_eigenvalues; The cis_eigenvalues vector is not set!"),
        }
    }

    /// Returns the state energies for all states.
    pub fn state_energies(&self) -> Array1<f64> {
        // Reference to the excitation energies.
        let ci_eig: ArrayView1<f64> = self.cis_eigenvalues();
        // The total ground state energy.
        let total_energy: f64 = self.total_energy();
        // An array is created with the total ground state energy for each state.
        let mut energies: Array1<f64> = Array1::from_elem((ci_eig.len() + 1), total_energy);
        // and the excitation energies are added to the states.
        energies.slice_mut(s![1..]).add_assign(&ci_eig);
        energies
    }

    /// Returns the excitation energy of an excited state.
    /// The first excited state has the index 0.
    pub fn cis_eigenvalue(&self, idx:usize) -> f64 {
        match &self.excited_states.cis_eigenvalues {
            Some(value) => value[idx],
            None => panic!("ExcitedStatesData::cis_eigenvalues; The cis_eigenvalues vector is not set!"),
        }
    }

    pub fn cis_coefficients(&self) -> ArrayView2<f64> {
        match &self.excited_states.x_plus_y {
            Some(value) => value.view(),
            None => panic!("ExcitedStatesData::x_plus_y; The cis_coefficients are not set!"),
        }
    }

    /// Returns the CI coefficients (in MO basis) for a specific excited state.
    /// The first excited state has the index 0.
    pub fn cis_coefficient(&self, idx:usize) -> ArrayView1<f64> {
        match &self.excited_states.x_plus_y {
            Some(value) => value.column(idx),
            None => panic!("ExcitedStatesData::cx_plus_y; The cis_coefficients are not set!"),
        }
    }

    /// Returns the 1-particle transition density matrix (in MO basis) for an excited state.
    /// The first excited state has the index 0.
    pub fn tdm(&self, idx:usize) -> ArrayView2<f64> {
        let n_occ: usize = self.n_occ();
        let n_virt: usize = self.n_virt();
        match &self.excited_states.x_plus_y {
            Some(value) => value.column(idx).into_shape([n_occ, n_virt]).unwrap(),
            None => panic!("ExcitedStatesData::x_plus_y; The cis_coefficients are not set!"),
        }
    }

    pub fn x_plus_ys(&self) -> ArrayView2<f64> {
        match &self.excited_states.x_plus_y {
            Some(value) => value.view(),
            None => panic!("ExcitedStatesData::x_plus_y; X + Ys are not set!"),
        }
    }

    /// Returns the X+Y coefficients (in MO basis) for a specific excited state.
    /// The first excited state has the index 0.
    pub fn x_plus_y(&self, idx: usize) -> ArrayView1<f64> {
        match &self.excited_states.x_plus_y {
            Some(value) => value.column(idx),
            None => panic!("ExcitedStatesData::x_plus_y; X + Ys are not set!"),
        }
    }

    pub fn x_minus_ys(&self) -> ArrayView2<f64> {
        match &self.excited_states.x_minus_y {
            Some(value) => value.view(),
            None => panic!("ExcitedStatesData::x_minus_y; X - Ys are not set!"),
        }
    }

    /// Returns the X-Y coefficients (in MO basis) for a specific excited state.
    /// The first excited state has the index 0.
    pub fn x_minus_y(&self, idx: usize) -> ArrayView1<f64> {
        match &self.excited_states.x_minus_y {
            Some(value) => value.column(idx),
            None => panic!("ExcitedStatesData::x_minus_y; X - Ys are not set!"),
        }
    }

    /// Returns the X+Y coefficients (in MO basis) for a specific excited state as 2D matrix.
    /// The first excited state has the index 0.
    pub fn x_plus_y_matrix(&self, idx: usize) -> ArrayView2<f64> {
        match &self.excited_states.x_plus_y {
            Some(value) => value.column(idx).into_shape([self.n_occ(), self.n_virt()]).unwrap(),
            None => panic!("ExcitedStatesData::x_plus_y; X + Ys are not set!"),
        }
    }

    /// Returns the X-Y coefficients (in MO basis) for a specific excited state as 2D matrix.
    /// The first excited state has the index 0.
    pub fn x_minus_y_matrix(&self, idx: usize) -> ArrayView2<f64> {
        match &self.excited_states.x_minus_y {
            Some(value) => value.column(idx).into_shape([self.n_occ(), self.n_virt()]).unwrap(),
            None => panic!("ExcitedStatesData::x_minus_y; X - Ys are not set!"),
        }
    }

    pub fn tr_dipoles(&self) -> ArrayView2<f64> {
        match &self.excited_states.tr_dipoles {
            Some(value) => value.view(),
            None => panic!("ExcitedStatesData::tr_dipoles; The tr_dipoles are not set!"),
        }
    }

    /// Returns the transition dipole moment for a specific excited state.
    pub fn tr_dipole(&self, idx:usize) -> Vector3<f64> {
        match &self.excited_states.tr_dipoles {
            Some(value) => {
                let dip = value.column(idx);
                Vector3::new( dip[0], dip[1], dip[2])
            },
            None => panic!("ExcitedStatesData::tr_dipoles; The tr_dipoles are not set!")
        }
    }

    pub fn osc_strengths(&self) -> ArrayView1<f64> {
        match &self.excited_states.osc_strengths {
            Some(value) => value.view(),
            None => panic!("ExcitedStatesData::osc_strengths; The osc_strengths are not set!"),
        }
    }

    pub fn cache(&self) -> &ProductCache {
        match &self.excited_states.cache {
            Some(value) => value,
            None => panic!("ExcitedStatesData::cache; The cache is not set!"),
        }
    }

    pub fn z_vector(&self) -> ArrayView1<f64> {
        match &self.excited_states.z_vector {
            Some(value) => value.view(),
            None => panic!("ExcitedStatesData::z_vector; The z_vector is not set!"),
        }
    }
}


