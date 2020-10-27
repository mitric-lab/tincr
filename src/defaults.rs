
// MOLECULE SPECIFICATION
// charge of the molecule in a.u.
pub const CHARGE: i8 = 0;
// spin multiplicity 2S + 1
pub const MULTIPLICITY: u8 = 1;
// occupation of orbitals is smeared out by Fermi
// distribution with temperature T in Kelvin
pub const TEMPERATURE: f64 = 0.0;

pub const LONG_RANGE_RADIUS: f64 = 3.03;
pub const PROXIMITY_CUTOFF: f64 = 2.00;
pub const LONG_RANGE_CORRECTION: bool = true;



// PARAMETERS
// scaling of hubbard parameters by this factor
pub const HUBBARD_SCALING: f64 = 1.0;
// scaling of repulsive potentials by this factor
pub const REPPOT_SCALING: f64 = 1.0;

// SCF ITERATION
// stop SCF calculation after maxiter iterations
pub const MAX_ITER: usize = 250;
// convergence threshold for relative change in SCF-calculation
pub const SCF_CONV: f64 = 1.0e-7;
// if the relative change drops below this value density mixing is used
pub const MIXING_THRESHOLD: f64 = 1.0e-3;
// shift virtual orbitals up in energy, this shift parameter is gradually
// reduced to zero as the density matrix converges
pub const LEVEL_SHIFT: f64 = 0.1;
// level shifting is turned on, as soon as the HOMO-LUMO gap
// drops below this value
pub const HOMO_LUMO_TOL: f64 = 0.05;
// is no density mixer object is used (density_mixer=None) the next
// guess for the density matrix is constructed as P_next = a*P + (1-a)*P_last
pub const LINEAR_MIXING_COEFFICIENT: f64 = 0.33;