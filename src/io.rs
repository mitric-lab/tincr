#![allow(dead_code)]
#![allow(warnings)]
#[macro_use]

use clap::crate_version;
use crate::constants::BOHR_TO_ANGS;
use crate::defaults::*;
use crate::gradients::*;
use crate::molecule::Molecule;
use chemfiles::{Frame, Trajectory};
use clap::App;
use ndarray::*;
use ndarray_linalg::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::ffi::OsStr;
use std::path::Path;
use std::ptr::eq;
use log::{debug, error, info, trace, warn};


fn default_charge() -> i8 {
    CHARGE
}
fn default_multiplicity() -> u8 {
    MULTIPLICITY
}
fn default_jobtype() -> String {
    String::from(JOBTYPE)
}
fn default_long_range_correction() -> bool {
    LONG_RANGE_CORRECTION
}
fn default_long_range_radius() -> f64 {
    LONG_RANGE_RADIUS
}
fn default_verbose() -> i8{
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

#[derive(Serialize, Deserialize, Clone)]
pub struct GeneralConfig {
    #[serde(default = "default_jobtype")]
    pub jobtype: String,
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
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MoleculeConfig {
    #[serde(default = "default_charge")]
    pub charge: i8,
    #[serde(default = "default_multiplicity")]
    pub multiplicity: u8,
}

#[derive(Serialize, Deserialize, Clone)]
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

#[derive(Serialize, Deserialize, Clone)]
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

#[derive(Serialize, Deserialize, Clone)]
pub struct LCConfig {
    #[serde(default = "default_long_range_correction")]
    pub long_range_correction: bool,
    #[serde(default = "default_long_range_radius")]
    pub long_range_radius: f64,
}

#[derive(Serialize, Deserialize, Clone)]
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

fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename).extension().and_then(OsStr::to_str)
}

pub fn get_coordinates(filename: &str) -> (Vec<u8>, Array2<f64>) {
    // read the geometry file
    let mut trajectory = Trajectory::open(filename, 'r').unwrap();
    let mut frame = Frame::new();
    // if multiple geometries are contained in the file, we will only use the first one
    trajectory.read(&mut frame).unwrap();
    let mut positions: Array2<f64> = Array2::from_shape_vec(
        (frame.size() as usize, 3),
        frame
            .positions()
            .iter()
            .flat_map(|array| array.iter())
            .cloned()
            .collect(),
    )
    .unwrap();
    // transform the coordinates from angstrom to bohr
    positions = positions / BOHR_TO_ANGS;
    // read the atomic number of each coordinate
    let atomic_numbers: Vec<u8> = (0..frame.size())
        .map(|i| frame.atom(i as u64).atomic_number() as u8)
        .collect();

    return (atomic_numbers, positions);
}

pub fn write_header() {
    info!("{: ^80}", "-----------------");
    info!("{: ^80}", "TINCR");
    info!("{: ^80}", "-----------------");
    let mut version_string: String = "version: ".to_owned();
    version_string.push_str(crate_version!());
    info!("{: ^80}", version_string);
    info!("{: ^80}", "");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "::           Roland Mitric          ::");
    info!("{: ^80}", "::  Chair of theoretical chemistry  ::");
    info!("{: ^80}", "::      University of Wuerzburg     ::");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "");
}