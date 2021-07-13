#![allow(dead_code)]
#![allow(warnings)]
#[macro_use]
use clap::crate_version;
use crate::constants::BOHR_TO_ANGS;
use crate::defaults::*;
use chemfiles::{Frame, Trajectory};
use clap::App;
use log::{debug, error, info, trace, warn};
use ndarray::*;
use ndarray_linalg::*;
use serde::{Deserialize, Serialize};
use std::{env, fs};
use std::ffi::OsStr;
use std::path::Path;
use std::ptr::eq;
use std::collections::HashMap;
use crate::constants;

fn default_charge() -> i8 {
    CHARGE
}
fn default_multiplicity() -> u8 {
    MULTIPLICITY
}
fn default_jobtype() -> String {
    String::from(JOBTYPE)
}
fn default_use_fmo() -> bool {false}
fn default_long_range_correction() -> bool {
    LONG_RANGE_CORRECTION
}
fn default_long_range_radius() -> f64 {
    LONG_RANGE_RADIUS
}
fn default_verbose() -> i8 {
    0
}
fn default_dispersion_correction() -> bool {
    DISPERSION_CORRECTION
}
fn default_scf_max_cycles() -> usize {
    MAX_ITER
}
fn default_scf_charge_conv() -> f64 {
    SCF_CHARGE_CONV
}
fn default_scf_energy_conv() -> f64 {
    SCF_ENERGY_CONV
}
fn default_temperature() -> f64 {
    TEMPERATURE
}
fn default_geom_opt_max_cycles() -> usize {
    GEOM_OPT_MAX_CYCLES
}
fn default_geom_opt_tol_displacement() -> f64 {
    GEOM_OPT_TOL_DISPLACEMENT
}
fn default_geom_opt_tol_gradient() -> f64 {
    GEOM_OPT_TOL_GRADIENT
}
fn default_geom_opt_tol_energy() -> f64 {
    GEOM_OPT_TOL_ENERGY
}
fn default_nr_active_occ() -> usize {
    ACTIVE_ORBITALS.0
}
fn default_nr_active_virt() -> usize {
    ACTIVE_ORBITALS.1
}
fn default_rpa() -> bool {
    RPA
}
fn default_restricted_active_space() -> bool {
    RESTRICTED_ACTIVE_SPACE
}
fn default_nstates() -> usize {
    EXCITED_STATES
}
fn default_use_mio() -> bool { USE_MIO }
fn default_mio_directory() ->String { String::from( MIO_DIR ) }
fn default_mol_config() -> MoleculeConfig {
    let mol_config: MoleculeConfig = toml::from_str("").unwrap();
    return mol_config;
}
fn default_scc_config() -> SccConfig {
    let scc_config: SccConfig = toml::from_str("").unwrap();
    return scc_config;
}
fn default_opt_config() -> OptConfig {
    let opt_config: OptConfig = toml::from_str("").unwrap();
    return opt_config;
}
fn default_lc_config() -> LCConfig {
    let lc_config: LCConfig = toml::from_str("").unwrap();
    return lc_config;
}
fn default_excited_state_config() -> ExcitedStatesConfig {
    let excited_config: ExcitedStatesConfig = toml::from_str("").unwrap();
    return excited_config;
}
fn default_slater_koster_config()->SlaterKosterConfig{
    let slako_config:SlaterKosterConfig = toml::from_str("").unwrap();
    return slako_config;
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Configuration {
    #[serde(default = "default_jobtype")]
    pub jobtype: String,
    #[serde(default = "default_use_fmo")]
    pub fmo: bool,
    #[serde(default = "default_dispersion_correction")]
    pub dispersion_correction: bool,
    #[serde(default = "default_verbose")]
    pub verbose: i8,
    #[serde(default = "default_mol_config")]
    pub mol: MoleculeConfig,
    #[serde(default = "default_scc_config")]
    pub scf: SccConfig,
    #[serde(default = "default_opt_config")]
    pub opt: OptConfig,
    #[serde(default = "default_lc_config")]
    pub lc: LCConfig,
    #[serde(default = "default_excited_state_config")]
    pub excited: ExcitedStatesConfig,
    #[serde(default = "default_slater_koster_config")]
    pub slater_koster:SlaterKosterConfig
}

impl Configuration {
    pub fn new() -> Self {
        // read tincr configuration file, if it does not exist in the directory
        // the program initializes the default settings and writes an configuration file
        // to the directory
        let config_file_path: &Path = Path::new(CONFIG_FILE_NAME);
        let mut config_string: String = if config_file_path.exists() {
            fs::read_to_string(config_file_path).expect("Unable to read config file")
        } else {
            String::from("")
        };
        // load the configration settings
        let config: Self = toml::from_str(&config_string).unwrap();
        // save the configuration file if it does not exist already
        if config_file_path.exists() == false {
            config_string = toml::to_string(&config).unwrap();
            fs::write(config_file_path, config_string).expect("Unable to write config file");
        }
        return config;
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct MoleculeConfig {
    #[serde(default = "default_charge")]
    pub charge: i8,
    #[serde(default = "default_multiplicity")]
    pub multiplicity: u8,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct SccConfig {
    #[serde(default = "default_scf_max_cycles")]
    pub scf_max_cycles: usize,
    #[serde(default = "default_scf_charge_conv")]
    pub scf_charge_conv: f64,
    #[serde(default = "default_scf_energy_conv")]
    pub scf_energy_conv: f64,
    #[serde(default = "default_temperature")]
    pub electronic_temperature: f64,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct OptConfig {
    #[serde(default = "default_geom_opt_max_cycles")]
    pub geom_opt_max_cycles: usize,
    #[serde(default = "default_geom_opt_tol_displacement")]
    pub geom_opt_tol_displacement: f64,
    #[serde(default = "default_geom_opt_tol_gradient")]
    pub geom_opt_tol_gradient: f64,
    #[serde(default = "default_geom_opt_tol_energy")]
    pub geom_opt_tol_energy: f64,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct LCConfig {
    #[serde(default = "default_long_range_correction")]
    pub long_range_correction: bool,
    #[serde(default = "default_long_range_radius")]
    pub long_range_radius: f64,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct ExcitedStatesConfig {
    #[serde(default = "default_nstates")]
    pub nstates: usize,
    #[serde(default = "default_rpa")]
    pub rpa: bool,
    #[serde(default = "default_restricted_active_space")]
    pub restricted_active_space: bool,
    #[serde(default = "default_nr_active_occ")]
    pub nr_active_occ: usize,
    #[serde(default = "default_nr_active_virt")]
    pub nr_active_virt: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SlaterKosterConfig{
    #[serde(default = "default_use_mio")]
    pub use_mio:bool,
    #[serde(default = "default_mio_directory")]
    pub mio_directory:String,
}