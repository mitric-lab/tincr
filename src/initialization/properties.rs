use std::collections::HashMap;
use crate::initialization::property::Property;
use ndarray::prelude::*;

pub struct Properties {
    map: HashMap<&'static str, Property>
}

impl Properties {
    pub fn new() -> Self {
        Properties{map: HashMap::new()}
    }

    pub fn get(&self, name: &'static str) -> Option<&Property> {
        self.map.get(name)
    }

    /// Returns the Property without a reference and removes it from the dict
    pub fn take(&mut self, name: &'static str) -> Option<Property> {
        self.map.remove(name)
    }

    pub fn set(&mut self, name: &'static str, value: Property) {
        self.map.insert(name, value);
    }

    pub fn contains_key(&self, name: &'static str) -> bool {
        self.map.contains_key(name)
    }

    /// Takes the atomic numbers
    pub fn take_atomic_numbers(&mut self) -> Result<Vec<u8>, Property> {
        match self.take("atomic_numbers") {
            Some(value) => value.into_vec_u8(),
            _=> Err(Property::default())
        }
    }

    /// Returns the reference density matrix
    pub fn take_p_ref(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("ref_density_matrix") {
            Some(value) => value.into_array2(),
            _=> Err(Property::default())
        }
    }

    /// Returns the H0 matrix in AO basis.
    pub fn take_h0(&mut self) -> Result<Array2<f64>, Property>{
        match self.take("H0") {
            Some(value) => value.into_array2(),
            _=> Err(Property::default())
        }
    }

    /// Returns the overlap matrix in AO basis.
    pub fn take_s(&mut self) -> Result<Array2<f64>, Property>{
        match self.take("S") {
            Some(value) => value.into_array2(),
            _=> Err(Property::default())
        }
    }

    /// Returns a reference to the gradient of the H0 matrix in AO basis.
    pub fn take_grad_h0(&mut self) -> Result<Array3<f64>, Property>{
        match self.take("gradH0") {
            Some(value) => value.into_array3(),
            _=> Err(Property::default())
        }
    }

    /// Returns a reference to the gradient of the overlap matrix in AO basis.
    pub fn take_grad_s(&mut self) -> Result<Array3<f64>, Property>{
        match self.take("gradS") {
            Some(value) => value.into_array3(),
            _=> Err(Property::default())
        }
    }

    /// Returns the charge differences per atom.
    pub fn take_dq(&mut self) -> Result<Array1<f64>, Property>{
        match self.take("dq") {
            Some(value) => value.into_array1(),
            _=> Err(Property::default())
        }
    }

    /// Returns the density matrix in AO basis.
    pub fn take_p(&mut self) -> Result<Array2<f64>, Property>{
        match self.take("P") {
            Some(value) => value.into_array2(),
            _=> Err(Property::default())
        }
    }

    /// Returns the gamma matrix in atomic basis.
    pub fn take_gamma(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_atom_wise") {
            Some(value) => value.into_array2(),
            _=> Err(Property::default())
        }
    }

    /// Returns the gamma matrix in AO basis.
    pub fn take_gamma_ao(&mut self) -> Result<Array2<f64>, Property>{
        match self.take("gamma_ao_wise") {
            Some(value) => value.into_array2(),
            _=> Err(Property::default())
        }
    }

    /// Returns the long-range corrected gamma matrix in atomic basis.
    pub fn take_gamma_lr(&mut self) -> Result<Array2<f64>, Property>{
        match self.take("gamma_lr_atom_wise") {
            Some(value) => value.into_array2(),
            _=> Err(Property::default())
        }
    }

    /// Returns the long-range corrected gamma matrix in AO basis.
    pub fn take_gamma_lr_ao(&mut self) -> Result<Array2<f64>, Property>{
        match self.take("gamma_lr_ao_wise") {
            Some(value) => value.into_array2(),
            _=> Err(Property::default())
        }
    }

    /// Returns the gradient of the gamma matrix in atomic basis.
    pub fn take_grad_gamma(&mut self) -> Result<Array3<f64>, Property>{
        match self.take("gamma_atom_wise_gradient") {
            Some(value) => value.into_array3(),
            _=> Err(Property::default())
        }
    }

    /// Returns the gradient of the gamma matrix in AO basis.
    pub fn take_grad_gamma_ao(&mut self) -> Result<Array3<f64>, Property>{
        match self.take("gamma_ao_wise_gradient") {
            Some(value) => value.into_array3(),
            _=> Err(Property::default())
        }
    }

    /// Returns the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn take_grad_gamma_lr(&mut self) -> Result<Array3<f64>, Property>{
        match self.take("gamma_lr_atom_wise_gradient"){
            Some(value) => value.into_array3(),
            _=> Err(Property::default())
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn take_grad_gamma_lr_ao(&mut self) -> Result<Array3<f64>, Property>{
       match self.take("gamma_lr_ao_wise_gradient"){
            Some(value) => value.into_array3(),
           _=> Err(Property::default())
        }
    }

    /// Returns a reference the atomic numbers
    pub fn atomic_numbers(&self) -> Option<&[u8]> {
        match self.get("atomic_numbers") {
            Some(value) => Some(value.as_vec_u8().unwrap()),
            _=> None
        }
    }

    /// Returns a reference to the reference density matrix
    pub fn p_ref(&self) -> Option<ArrayView2<f64>> {
        match self.get("ref_density_matrix") {
            Some(value) =>Some(value.as_array2().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the H0 matrix in AO basis.
    pub fn h0(&self) -> Option<ArrayView2<f64>>{
        match self.get("H0") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the overlap matrix in AO basis.
    pub fn s(&self) -> Option<ArrayView2<f64>>{
        match self.get("S") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the gradient of the H0 matrix in AO basis.
    pub fn grad_h0(&self) -> Option<ArrayView3<f64>>{
        match self.get("gradH0") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the gradient of the overlap matrix in AO basis.
    pub fn grad_s(&self) -> Option<ArrayView3<f64>>{
        match self.get("gradS") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the charge differences per atom.
    pub fn dq(&self) -> Option<ArrayView1<f64>>{
        match self.get("dq") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the density matrix in AO basis.
    pub fn p(&self) -> Option<ArrayView2<f64>>{
        match self.get("P") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the gamma matrix in atomic basis.
    pub fn gamma(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_atom_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the gamma matrix in AO basis.
    pub fn gamma_ao(&self) -> Option<ArrayView2<f64>>{
        match self.get("gamma_ao_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the long-range corrected gamma matrix in atomic basis.
    pub fn gamma_lr(&self) -> Option<ArrayView2<f64>>{
        match self.get("gamma_lr_atom_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the long-range corrected gamma matrix in AO basis.
    pub fn gamma_lr_ao(&self) -> Option<ArrayView2<f64>>{
        match self.get("gamma_lr_ao_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the gradient of the gamma matrix in atomic basis.
    pub fn grad_gamma(&self) -> Option<ArrayView3<f64>>{
        match self.get("gamma_atom_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the gradient of the gamma matrix in AO basis.
    pub fn grad_gamma_ao(&self) -> Option<ArrayView3<f64>>{
        match self.get("gamma_ao_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn grad_gamma_lr(&self) -> Option<ArrayView3<f64>>{
        match self.get("gamma_lr_atom_wise_gradient"){
            Some(value) => Some(value.as_array3().unwrap().view()),
            _=> None
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn grad_gamma_lr_ao(&self) -> Option<ArrayView3<f64>>{
        match self.get("gamma_lr_ao_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _=> None
        }
    }

    /// Set the atomic numbers
    pub fn set_atomic_numbers(&mut self, atomic_numbers: Vec<u8>) {self.set("atomic_numbers", Property::from(atomic_numbers))}

    /// Set the reference density matrix
    pub fn set_p_ref(&mut self, ref_p: Array2<f64>) {self.set("ref_density_matrix", Property::from(ref_p))}

    /// Set the H0 matrix in AO basis.
    pub fn set_h0(&mut self, h0: Array2<f64>) {
        self.set("H0", Property::from(h0));
    }

    /// Set the overlap matrix in AO basis.
    pub fn set_s(&mut self, s: Array2<f64>) {
        self.set("S", Property::from(s));
    }

    /// Set the gradient of the H0 matrix in AO basis.
    pub fn set_grad_h0(&mut self, grad_h0: Array3<f64>) {
        self.set("gradH0", Property::from(grad_h0));
    }

    /// Set the gradient of the overlap matrix in AO basis.
    pub fn set_grad_s(&mut self, grad_s: Array3<f64>) {
        self.set("gradS", Property::from(grad_s));
    }

    /// Set the charge differences per atom.
    pub fn set_dq(&mut self, dq: Array1<f64>){
        self.set("dq", Property::from(dq));
    }

    /// Set the density matrix in AO basis.
    pub fn set_p(&mut self, p: Array2<f64>) {
        self.set("P", Property::from(p));
    }

    /// Set the gamma matrix in atomic basis.
    pub fn set_gamma(&mut self, gamma: Array2<f64>) {
        self.set("gamma_atom_wise", Property::from(gamma));
    }

    /// Set the gamma matrix in AO basis.
    pub fn set_gamma_ao(&mut self, gamma_ao: Array2<f64>){
        self.set("gamma_ao_wise", Property::from(gamma_ao));
    }

    /// Set the long-range corrected gamma matrix in atomic basis.
    pub fn set_gamma_lr(&mut self, gamma_lr: Array2<f64>) {
        self.set("gamma_lr_atom_wise", Property::from(gamma_lr));
    }

    /// Set the long-range corrected gamma matrix in AO basis.
    pub fn set_gamma_lr_ao(&mut self, gamma_lr_ao: Array2<f64>){
        self.set("gamma_lr_ao_wise", Property::from(gamma_lr_ao));
    }

    /// Set the gradient of the gamma matrix in atomic basis.
    pub fn set_grad_gamma(&mut self, grad_gamma: Array3<f64>){
        self.set("gamma_atom_wise_gradient", Property::from(grad_gamma));
    }

    /// Set the gradient of the gamma matrix in AO basis.
    pub fn set_grad_gamma_ao(&mut self, grad_gamma_ao: Array3<f64>){
        self.set("gamma_ao_wise_gradient", Property::from(grad_gamma_ao));
    }

    /// Set the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn set_grad_gamma_lr(&mut self, grad_gamma_lr: Array3<f64>) {
        self.set("gamma_lr_atom_wise_gradient", Property::from(grad_gamma_lr));
    }

    /// Set the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn set_grad_gamma_lr_ao(&mut self, grad_gamma_lr_ao: Array3<f64>) {
        self.set("gamma_lr_ao_wise_gradient", Property::from(grad_gamma_lr_ao));
    }
}

