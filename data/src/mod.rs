mod storage;
mod getter;
mod setter;
mod taker;
mod charges;
mod excited_states;
mod gamma_matrices;
mod other;
mod orbitals;
mod gradients;
mod energies;
mod parametrization;

pub use orbitals::OrbType;
use crate::fmo::{BasisState, PairType};
use ndarray::prelude::*;
use crate::scc::mixer::BroydenMixer;
use crate::excited_states::ProductCache;
use crate::scc::gamma_approximation::GammaFunction;
use hashbrown::HashMap;
use crate::param::slako::SlaterKoster;
use crate::param::reppot::RepulsivePotential;

pub struct Storage<'a> {
    charges: ChargeData,
    energies: EnergyData,
    excited_states: ExcitedStateData,
    gammas: GammaData,
    gradients: GradData,
    orbitals: SpatialOrbitals,
    other: OtherData,
    parametrization:Parametrization<'a>,
}

pub struct EnergyData {
    /// The total energy of the system in atomic units.
    pub total_energy: Option<f64>,
    /// IDK
    pub last_energy: Option<f64>,
}

#[derive(Copy, Clone)]
pub struct Parametrization<'a> {
    /// Fock matrix that contains only the one-electron integrals in AO basis.
    pub h0: ArrayView2<'a, f64>,
    /// Overlap integral matrix in AO basis.
    pub s: ArrayView2<'a, f64>,
    /// The unscreened Gamma matrix in atom basis.
    pub gamma: ArrayView2<'a, f64>,
    /// Screened Gamma matrix in atom basis.
    pub gamma_lrc: Option<ArrayView2<'a, f64>>,
}


/// Type that contains the data that is connected to the molecular orbitals.
pub struct SpatialOrbitals {
    /// MO coefficients.
    pub orbs: Option<Array2<f64>>,
    /// MO energies.
    pub orbe: Option<Array1<f64>>,
    /// Number of electrons.
    pub n_elec: usize,
    /// MO occupation numbers.
    pub occupation: Option<Vec<f64>>,
    /// Indices of occupied MOs.
    pub occ_indices: Option<Vec<usize>>,
    /// Indices of virtual MOs.
    pub virt_indices: Option<Vec<usize>>,
    /// Active occupied orbitals (might be used for CIS in truncated active space).
    pub active_occ: Option<Vec<usize>>,
    /// Active virtual orbitals.
    pub active_virt: Option<Vec<usize>>,
    /// Density matrix in AO basis.
    pub p: Option<Array2<f64>>,
    /// Reference density matrix in AO basis.
    pub p_ref: Option<Array2<f64>>,
    /// Difference between the density matrix and the reference density matrix.
    pub delta_p: Option<Array2<f64>>,
    /// Density matrix of alpha electrons in AO basis.
    pub p_alpha: Option<Array2<f64>>,
    /// Density matrix of beta electrons in AO basis.
    pub p_beta: Option<Array2<f64>>,
}


pub struct SpinOrbitals {
    alphas: SpatialOrbitals,
    betas: SpatialOrbitals,
}

pub struct GammaData {
    /// Unscreened Gamma matrix in AO basis.
    pub gamma_ao: Option<Array2<f64>>,
    /// Screened Gamma matrix in AO basis.
    pub gamma_lr_ao: Option<Array2<f64>>,
}

/// Type that contains the data which is connected to atom-centered charges.
pub struct ChargeData {
    /// Partial charges.
    pub dq: Option<Array1<f64>>,
    /// Charge differences between charges of a molecular dimer and the corresponding monomers.
    pub delta_dq: Option<Array1<f64>>,
    /// Partial charges in the basis of the AOs.
    pub q_ao: Option<Array1<f64>>,
    /// Transition charges between occupied and virtual orbitals.
    pub q_ov: Option<Array2<f64>>,
    /// Transition charges between occupied and occupied orbitals.
    pub q_oo: Option<Array2<f64>>,
    /// Transition charges between virtual and virtual orbitals.
    pub q_vv: Option<Array2<f64>>,
    /// Transition charges between virtual and occupied orbitals.
    pub q_vo: Option<Array2<f64>>,
    /// Transition charges for excited states.
    pub q_trans: Option<Array2<f64>>,
    /// Partial charges from the electrostatic potential of other monomers.
    pub esp_q: Option<Array1<f64>>,
    /// Partial charges corresponding to alpha electrons.
    pub dq_alpha: Option<Array1<f64>>,
    /// Partial charges corresponding to beta electrons.
    pub dq_beta: Option<Array1<f64>>,
}

/// Type that contains the data that specifies the excited states.
pub struct ExcitedStateData {
    /// Energy differences between all virtual and occupied MOs.
    pub omega: Option<Array1<f64>>,
    /// Eigenvalues of the excited states (= excitation energies) in a.u.
    pub cis_eigenvalues: Option<Array1<f64>>,
    /// X + Y for TDDFT, in case of CIS these are the eigenvectors of the excited states.
    pub x_plus_y: Option<Array2<f64>>,
    /// X - Y for TDDFT, not used for CIS
    pub x_minus_y: Option<Array2<f64>>,
    /// Transition dipole moments between the electronic ground state and the excited states.
    pub tr_dipoles: Option<Array2<f64>>,
    /// Oscillator strength between the electronic ground state and the excited states.
    pub osc_strengths: Option<Array1<f64>>,
    /// Product cache of the Davidson solver.
    pub cache: Option<ProductCache>,
    /// IDK?
    pub z_vector: Option<Array1<f64>>,
}

pub struct GradData {
    /// Gradient of the overlap matrix.
    pub s: Option<Array3<f64>>,
    /// Gradient of the one-electron integrals in the Fock matrix.
    pub h0: Option<Array3<f64>>,
    /// Gradient of the partial charges.
    pub dq: Option<Array2<f64>>,
    /// Diagonal elements of the gradient of the partial charges.
    pub dq_diag: Option<Array1<f64>>,
    /// Gradient of the unscreened gamma matrix.
    pub gamma: Option<Array3<f64>>,
    /// Gradient of the screened gamma matrix.
    pub gamma_lr: Option<Array3<f64>>,
    /// Gradient of the unscreened gamma matrix in AO basis.
    pub gamma_ao: Option<Array3<f64>>,
    /// Gradient of the screened gamma matrix in AO basis.
    pub gamma_lr_ao: Option<Array3<f64>>
}


pub struct OtherData {
    /// The S^{-1/2} that is needed to transform the general eigenvalue problem to a specific one.
    pub x: Option<Array2<f64>>,
    /// The Broyden mixer, that is needed for the SCF/SCC iterations.
    pub broyden_mixer: Option<BroydenMixer>,
    /// The atomic numbers for the atoms of a given system.
    pub atomic_numbers: Option<Vec<u8>>,
    /// The electrostatic potential that is created by the ensemble of monomers.
    pub v: Option<Array2<f64>>,
    /// The Fock matrix.
    pub fock: Option<Array2<f64>>,
    /// HashMap that maps two monomers to a type of a pair.
    pub pair_types: Option<HashMap<(usize, usize), PairType>>,
    /// HashMap that maps two monomers to the index of the pair.
    pub pair_indices: Option<HashMap<(usize, usize), usize>>,
    /// LCMO-FMO Fock matrix
    pub lcmo_fock: Option<Array2<f64>>,
    /// The gradient of the long-range corrected two electron integrals
    pub flr_dmd0: Option<Array3<f64>>,
}