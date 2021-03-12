// MOLECULE SPECIFICATION
// charge of the molecule in a.u.
pub const CHARGE: i8 = 0;
// spin multiplicity 2S + 1
pub const MULTIPLICITY: u8 = 1;
// jobtype
pub const JOBTYPE: &str = "sp";
// config file
pub const CONFIG_FILE_NAME: &str = "tincr.toml";
// occupation of orbitals is smeared out by Fermi
// distribution with temperature T in Kelvin
pub const TEMPERATURE: f64 = 0.0;

pub const LONG_RANGE_RADIUS: f64 = 3.03;
pub const PROXIMITY_CUTOFF: f64 = 30.00;
pub const LONG_RANGE_CORRECTION: bool = true;
pub const DISPERSION_CORRECTION: bool = true;

// PARAMETERS
// scaling of hubbard parameters by this factor
pub const HUBBARD_SCALING: f64 = 1.0;
// scaling of repulsive potentials by this factor
pub const REPPOT_SCALING: f64 = 1.0;

// SCF ITERATION
// stop SCF calculation after maxiter iterations
pub const MAX_ITER: usize = 250;
// convergence threshold for relative change in SCF-calculation
pub const SCF_CHARGE_CONV: f64 = 1.0e-5;
pub const SCF_ENERGY_CONV: f64 = 1.0e-5;

pub const DENSITY_CONV: f64 = 1.0e-3;
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

pub const DIIS_LIMIT: usize = 8;

// Broyden Mixer
pub const BROYDEN_OMEGA0: f64 = 0.01;
pub const BROYDEN_MIN_WEIGHT: f64 = 1.0;
pub const BROYDEN_MAX_WEIGHT: f64 = 1.0e5;
pub const BROYDEN_WEIGHT_FACTOR: f64 = 1.0e-2;
pub const BROYDEN_MIXING_PARAMETER: f64 = 0.3;


pub const SOURCE_DIR_VARIABLE: &str = "TINCR_SRC_DIR";

// Number of active orbitals
pub const ACTIVE_ORBITALS: (usize, usize) = (1000, 1000);
// Numver of excited states
pub const EXCITED_STATES: usize = 1000;
// DO RPA or Casida
pub const RPA: bool = true;
pub const RESTRICTED_ACTIVE_SPACE: bool = true;

// Optimization
pub const GEOM_OPT_MAX_CYCLES: usize = 500;
pub const GEOM_OPT_TOL_DISPLACEMENT: f64 = 0.0;
pub const GEOM_OPT_TOL_GRADIENT: f64 = 0.0;
pub const GEOM_OPT_TOL_ENERGY: f64 = 0.0;
