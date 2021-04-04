use std::collections::HashMap;
use crate::initialization::property::Property;
use ndarray::prelude::*;
use crate::h0_and_s::h0_and_s;

pub struct Properties {
    map: HashMap<&'static str, Property>
}

impl Properties {
    pub fn new() -> Self {
        Properties{map: HashMap::new()}
    }

    pub fn get(&self, name: &'static str) -> &Property {
        match self.map.get(name) {
            Some(value) => value,
            _ => None
        }
    }

    pub fn set(&mut self, name: &'static str, value: Property) {
        self.map.insert(name, value);
    }

    /// Returns the H0 matrix in AO basis.
    pub fn h0(&self) -> Option<ArrayView2<f64>>{
        self.get("H0").as_array2()
    }

    /// Returns the overlap matrix in AO basis.
    pub fn s(&self) -> Option<ArrayView2<f64>>{
        self.get("S").as_array2()
    }

    /// Returns the gradient of the H0 matrix in AO basis.
    pub fn grad_h0(&self) -> Option<ArrayView3<f64>>{
        self.get("gradH0").as_array3()
    }

    /// Returns the gradient of the overlap matrix in AO basis.
    pub fn grad_s(&self) -> Option<ArrayView3<f64>>{
        self.get("gradS").as_array3()
    }

    /// Returns the charge differences per atom.
    pub fn dq(&self) -> Option<ArrayView1<f64>>{
        self.get("dq").as_array2()
    }

    /// Returns the density matrix in AO basis.
    pub fn p(&self) -> Option<ArrayView2<f64>>{
        self.get("P").as_array2()
    }

    /// Returns the gamma matrix in atomic basis.
    pub fn gamma(&self) -> Option<ArrayView2<f64>> {
        self.get("gamma_atom_wise").as_array2()
    }

    /// Returns the gamma matrix in AO basis.
    pub fn gamma_ao(&self) -> Option<ArrayView2<f64>>{
        self.get("gamma_ao_wise").as_array2()
    }

    /// Returns the long-range corrected gamma matrix in atomic basis.
    pub fn gamma_lr(&self) -> Option<ArrayView2<f64>>{
        self.get("gamma_lr_atom_wise").as_array2()
    }

    /// Returns the long-range corrected gamma matrix in AO basis.
    pub fn gamma_lr_ao(&self) -> Option<ArrayView2<f64>>{
        self.get("gamma_lr_ao_wise").as_array2()
    }

    /// Returns the gradient of the gamma matrix in atomic basis.
    pub fn grad_gamma(&self) -> Option<ArrayView3<f64>>{
        self.get("gamma_atom_wise_gradient").as_array3()
    }

    /// Returns the gradient of the gamma matrix in AO basis.
    pub fn grad_gamma_ao(&self) -> Option<ArrayView3<f64>>{
        self.get("gamma_ao_wise_gradient").as_array3()
    }

    /// Returns the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn grad_gamma_lr(&self) -> Option<ArrayView3<f64>>{
        self.get("gamma_lr_atom_wise_gradient").as_array3()
    }

    /// Returns the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn grad_gamma_lr_ao(&self) -> Option<ArrayView3<f64>>{
       self.get("gamma_lr_ao_wise_gradient").as_array3()
    }


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
    pub fn set_gamma(&mut self, gamma: Array2<f64> {
        self.set("gamma_atom_wise", Property::from(gamma));
    }

    /// Set the gamma matrix in AO basis.
    pub fn set_gamma_ao(&mut self, gamma_ao: Array2<f64>){
        self.set("gamma_ao_wise", Property::from(gamma_ao));
    }

    /// Set the long-range corrected gamma matrix in atomic basis.
    pub fn set_gamma_lr(&mut self, gamma_lr: Array2<f64> {
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
    pub fn set_grad_gamma_lr(&mut self, grad_gamma_lr: Array3<f64> {
        self.set("gamma_lr_atom_wise_gradient", Property::from(grad_gamma_lr));
    }

    /// Set the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn set_grad_gamma_lr_ao(&mut self, grad_gamma_lr_ao: Array3<f64>{
        self.set("gamma_lr_ao_wise_gradient", Property::from(grad_gamma_lr_ao));
    }
}

