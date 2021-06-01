use crate::initialization::property::Property;
use crate::scc::mixer::BroydenMixer;
use hashbrown::HashMap;
use ndarray::prelude::*;

pub struct Properties {
    map: HashMap<&'static str, Property>,
}

impl Properties {
    pub fn new() -> Self {
        Properties {
            map: HashMap::new(),
        }
    }

    /// Removes all multi dimensional arrays from the HashMap to free the memory
    pub fn reset(&mut self) {
        let multi_dim_data = [
            "H0",
            "S",
            "X",
            //"P",
            "gradH0",
            "gradS",
            "gamma_atom_wise",
            "gamma_ao_wise",
            "gamma_lr_atom_wise",
            "gamma_lr_ao_wise",
            "gamma_atom_wise_gradient",
            "gamma_ao_wise_gradient",
            "gamma_lr_atom_wise_gradient",
            "gamma_lr_ao_wise_gradient",
        ];
        for data_name in multi_dim_data.iter() {
            self.map.remove(*data_name);
        }
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

    /// Takes the scc mixer
    pub fn take_mixer(&mut self) -> Result<BroydenMixer, Property> {
        match self.take("mixer") {
            Some(value) => value.into_mixer(),
            _ => Err(Property::default()),
        }
    }

    /// Takes the atomic numbers
    pub fn take_atomic_numbers(&mut self) -> Result<Vec<u8>, Property> {
        match self.take("atomic_numbers") {
            Some(value) => value.into_vec_u8(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the reference density matrix
    pub fn take_p_ref(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("ref_density_matrix") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the H0 matrix in AO basis.
    pub fn take_h0(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("H0") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the overlap matrix in AO basis.
    pub fn take_s(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("S") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the S^-1/2 in AO basis
    pub fn take_x(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("X") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns a reference to the gradient of the H0 matrix in AO basis.
    pub fn take_grad_h0(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gradH0") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns a reference to the gradient of the overlap matrix in AO basis.
    pub fn take_grad_s(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gradS") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the charge differences per atom.
    pub fn take_dq(&mut self) -> Result<Array1<f64>, Property> {
        match self.take("dq") {
            Some(value) => value.into_array1(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the density matrix in AO basis.
    pub fn take_p(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("P") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gamma matrix in atomic basis.
    pub fn take_gamma(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_atom_wise") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gamma matrix in AO basis.
    pub fn take_gamma_ao(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_ao_wise") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the long-range corrected gamma matrix in atomic basis.
    pub fn take_gamma_lr(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_lr_atom_wise") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the long-range corrected gamma matrix in AO basis.
    pub fn take_gamma_lr_ao(&mut self) -> Result<Array2<f64>, Property> {
        match self.take("gamma_lr_ao_wise") {
            Some(value) => value.into_array2(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gradient of the gamma matrix in atomic basis.
    pub fn take_grad_gamma(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gamma_atom_wise_gradient") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gradient of the gamma matrix in AO basis.
    pub fn take_grad_gamma_ao(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gamma_ao_wise_gradient") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn take_grad_gamma_lr(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gamma_lr_atom_wise_gradient") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn take_grad_gamma_lr_ao(&mut self) -> Result<Array3<f64>, Property> {
        match self.take("gamma_lr_ao_wise_gradient") {
            Some(value) => value.into_array3(),
            _ => Err(Property::default()),
        }
    }

    /// Returns a reference the atomic numbers
    pub fn atomic_numbers(&self) -> Option<&[u8]> {
        match self.get("atomic_numbers") {
            Some(value) => Some(value.as_vec_u8().unwrap()),
            _ => None,
        }
    }

    /// Returns the energy of the last scc iteration
    pub fn last_energy(&self) -> Option<f64> {
        match self.get("last_energy") {
            Some(value) => Some(*value.as_double().unwrap()),
            _ => Some(0.0),
        }
    }

    /// Returns the energy of the last scc iteration
    pub fn occupation(&self) -> Option<&[f64]> {
        match self.get("occupation") {
            Some(value) => Some(value.as_vec_f64().unwrap()),
            _ => None,
        }
    }

    /// Returns a reference to the reference density matrix
    pub fn p_ref(&self) -> Option<ArrayView2<f64>> {
        match self.get("ref_density_matrix") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the H0 matrix in AO basis.
    pub fn h0(&self) -> Option<ArrayView2<f64>> {
        match self.get("H0") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the MO coefficients.
    pub fn orbs(&self) -> Option<ArrayView2<f64>> {
        match self.get("orbs") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the MO energies.
    pub fn orbe(&self) -> Option<ArrayView1<f64>> {
        match self.get("orbe") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the overlap matrix in AO basis.
    pub fn s(&self) -> Option<ArrayView2<f64>> {
        match self.get("S") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to S^-1/2 in AO basis.
    pub fn x(&self) -> Option<ArrayView2<f64>> {
        match self.get("X") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the H0 matrix in AO basis.
    pub fn grad_h0(&self) -> Option<ArrayView3<f64>> {
        match self.get("gradH0") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the overlap matrix in AO basis.
    pub fn grad_s(&self) -> Option<ArrayView3<f64>> {
        match self.get("gradS") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the charge differences per atom.
    pub fn dq(&self) -> Option<ArrayView1<f64>> {
        match self.get("dq") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the derivative of the charge (differences) w.r.t. to the degrees
    /// of freedom per atom. The first dimension is dof and the second one is the atom where the charge
    /// resides.
    pub fn grad_dq(&self) -> Option<ArrayView2<f64>> {
        match self.get("grad_dq") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _=> None,
        }
    }

    /// Returns a reference to the diagonal of the derivative of charge (differences) w.r.t. to the
    /// degrees of freedom per atom. The first dimension is dof and the second one is the atom
    /// where the charge resides.
    pub fn grad_dq_diag(&self) -> Option<ArrayView1<f64>> {
        match self.get("grad_dq_diag") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _=> None,
        }
    }

    /// Returns a reference to the differences of charges differences between the pair and the
    /// corresponding monomers per atom.
    pub fn delta_dq(&self) -> Option<ArrayView1<f64>> {
        match self.get("delta_dq") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between occupied and virtual orbitaös
    pub fn q_ov(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_ov") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between virtual and occupied orbitals
    pub fn q_vo(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_vo") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between occupied orbitals
    pub fn q_oo(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_oo") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the transition charges between virtual orbitals
    pub fn q_vv(&self) -> Option<ArrayView2<f64>> {
        match self.get("q_vv") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the converged Z-vector from the FMO grad. response term.
    pub fn z_vector(&self) -> Option<ArrayView1<f64>> {
        match self.get("z_vector") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the esp charges per atom
    pub fn esp_q(&self) -> Option<ArrayView1<f64>> {
        match self.get("esp_charges") {
            Some(value) => Some(value.as_array1().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the indices of the occupied orbitals.
    pub fn occ_indices(&self) -> Option<&[usize]> {
        match self.get("occ_indices") {
            Some(value) => Some(value.as_vec_usize().unwrap()),
            _ => None,
        }
    }

    /// Returns a reference to the indices of the virtual orbitals.
    pub fn virt_indices(&self) -> Option<&[usize]> {
        match self.get("virt_indices") {
            Some(value) => Some(value.as_vec_usize().unwrap()),
            _ => None,
        }
    }

    /// Returns a reference to the electrostatic potential in AO basis.
    pub fn v(&self) -> Option<ArrayView2<f64>> {
        match self.get("V") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the density matrix in AO basis.
    pub fn p(&self) -> Option<ArrayView2<f64>> {
        match self.get("P") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gamma matrix in atomic basis.
    pub fn gamma(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_atom_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gamma matrix in AO basis.
    pub fn gamma_ao(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_ao_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the long-range corrected gamma matrix in atomic basis.
    pub fn gamma_lr(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_lr_atom_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the long-range corrected gamma matrix in AO basis.
    pub fn gamma_lr_ao(&self) -> Option<ArrayView2<f64>> {
        match self.get("gamma_lr_ao_wise") {
            Some(value) => Some(value.as_array2().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the gamma matrix in atomic basis.
    pub fn grad_gamma(&self) -> Option<ArrayView3<f64>> {
        match self.get("gamma_atom_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the gamma matrix in AO basis.
    pub fn grad_gamma_ao(&self) -> Option<ArrayView3<f64>> {
        match self.get("gamma_ao_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn grad_gamma_lr(&self) -> Option<ArrayView3<f64>> {
        match self.get("gamma_lr_atom_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Returns a reference to the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn grad_gamma_lr_ao(&self) -> Option<ArrayView3<f64>> {
        match self.get("gamma_lr_ao_wise_gradient") {
            Some(value) => Some(value.as_array3().unwrap().view()),
            _ => None,
        }
    }

    /// Set the energy of the last scc iteration
    pub fn set_occupation(&mut self, f: Vec<f64>) {
        self.set("occupation", Property::VecF64(f))
    }

    /// Set the energy of the last scc iteration
    pub fn set_last_energy(&mut self, energy: f64) {
        self.set("last_energy", Property::Double(energy))
    }

    /// Set the scc mixer
    pub fn set_mixer(&mut self, mixer: BroydenMixer) {
        self.set("mixer", Property::from(mixer))
    }

    /// Set the atomic numbers
    pub fn set_atomic_numbers(&mut self, atomic_numbers: Vec<u8>) {
        self.set("atomic_numbers", Property::from(atomic_numbers))
    }

    /// Set the reference density matrix
    pub fn set_p_ref(&mut self, ref_p: Array2<f64>) {
        self.set("ref_density_matrix", Property::from(ref_p))
    }

    /// Set the H0 matrix in AO basis.
    pub fn set_h0(&mut self, h0: Array2<f64>) {
        self.set("H0", Property::from(h0));
    }

    /// Set the overlap matrix in AO basis.
    pub fn set_s(&mut self, s: Array2<f64>) {
        self.set("S", Property::from(s));
    }

    /// Set the MO coefficients from the SCC calculation.
    pub fn set_orbs(&mut self, orbs: Array2<f64>) {
        self.set("orbs", Property::from(orbs));
    }

    /// Set the MO energies from the SCC calculation.
    pub fn set_orbe(&mut self, orbe: Array1<f64>) {
        self.set("orbe", Property::from(orbe));
    }

    /// Set the S^-1/2 in AO basis.
    pub fn set_x(&mut self, x: Array2<f64>) {
        self.set("X", Property::from(x));
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
    pub fn set_dq(&mut self, dq: Array1<f64>) {
        self.set("dq", Property::from(dq));
    }

    /// Set the derivative of the charge (differences) w.r.t. to degrees of freedom per atom
    pub fn set_grad_dq(&mut self, grad_dq: Array2<f64>) { self.set("grad_dq", Property::from(grad_dq))}

    /// Set the diagonal of the derivative of the charge (differences) w.r.t. to degrees of freedom per atom
    pub fn set_grad_dq_diag(&mut self, grad_dq_diag: Array1<f64>) { self.set("grad_dq_diag", Property::from(grad_dq_diag))}

    /// Set the difference of charge differences between the pair dq and the dq's from the
    /// corresponding monomers per atom.
    pub fn set_delta_dq(&mut self, delta_dq: Array1<f64>) {
        self.set("delta_dq", Property::from(delta_dq));
    }

    /// Set the transition charges between occupied and virtual orbitals
    pub fn set_q_ov(&mut self, q_ov: Array2<f64>) {
        self.set("q_ov", Property::from(q_ov));
    }

    /// Set the transition charges between virtual and occupied orbitals
    pub fn set_q_vo(&mut self, q_vo: Array2<f64>) {
        self.set("q_vo", Property::from(q_vo));
    }

    /// Set the transition charges between occupied orbitals
    pub fn set_q_oo(&mut self, q_oo: Array2<f64>) {
        self.set("q_oo", Property::from(q_oo));
    }

    /// Set the transition charges between virtual orbitals
    pub fn set_q_vv(&mut self, q_vv: Array2<f64>) {
        self.set("q_vv", Property::from(q_vv));
    }

    /// Set the converged Z-vector from the FMO gradient response term.
    pub fn set_z_vector(&mut self, z_vector: Array1<f64>) {
        self.set("z_vector", Property::from(z_vector));
    }

    /// Set the indices of the occupied orbitals, starting at 0.
    pub fn set_occ_indices(&mut self, occ_indices: Vec<usize>) {
        self.set("occ_indices", Property::from(occ_indices));
    }

    /// Set the indices of the virtual orbitals.
    pub fn set_virt_indices(&mut self, virt_indices: Vec<usize>) {
        self.set("virt_indices", Property::from(virt_indices));
    }

    /// Set the esp charges per atom
    pub fn set_esp_q(&mut self, esp_q: Array1<f64>) {
        self.set("esp_charges", Property::from(esp_q));
    }

    /// Set the density matrix in AO basis.
    pub fn set_p(&mut self, p: Array2<f64>) {
        self.set("P", Property::from(p));
    }

    /// Set the electrostatic potential in AO basis.
    pub fn set_v(&mut self, v: Array2<f64>) {
        self.set("V", Property::from(v));
    }

    /// Set the gamma matrix in atomic basis.
    pub fn set_gamma(&mut self, gamma: Array2<f64>) {
        self.set("gamma_atom_wise", Property::from(gamma));
    }

    /// Set the gamma matrix in AO basis.
    pub fn set_gamma_ao(&mut self, gamma_ao: Array2<f64>) {
        self.set("gamma_ao_wise", Property::from(gamma_ao));
    }

    /// Set the long-range corrected gamma matrix in atomic basis.
    pub fn set_gamma_lr(&mut self, gamma_lr: Array2<f64>) {
        self.set("gamma_lr_atom_wise", Property::from(gamma_lr));
    }

    /// Set the long-range corrected gamma matrix in AO basis.
    pub fn set_gamma_lr_ao(&mut self, gamma_lr_ao: Array2<f64>) {
        self.set("gamma_lr_ao_wise", Property::from(gamma_lr_ao));
    }

    /// Set the gradient of the gamma matrix in atomic basis.
    pub fn set_grad_gamma(&mut self, grad_gamma: Array3<f64>) {
        self.set("gamma_atom_wise_gradient", Property::from(grad_gamma));
    }

    /// Set the gradient of the gamma matrix in AO basis.
    pub fn set_grad_gamma_ao(&mut self, grad_gamma_ao: Array3<f64>) {
        self.set("gamma_ao_wise_gradient", Property::from(grad_gamma_ao));
    }

    /// Set the gradient of the long-range corrected gamma matrix in atomic basis.
    pub fn set_grad_gamma_lr(&mut self, grad_gamma_lr: Array3<f64>) {
        self.set("gamma_lr_atom_wise_gradient", Property::from(grad_gamma_lr));
    }

    /// Set the gradient of the long-range corrected gamma matrix in AO basis.
    pub fn set_grad_gamma_lr_ao(&mut self, grad_gamma_lr_ao: Array3<f64>) {
        self.set(
            "gamma_lr_ao_wise_gradient",
            Property::from(grad_gamma_lr_ao),
        );
    }
}
