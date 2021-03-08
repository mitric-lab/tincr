#![allow(dead_code)]
#![allow(warnings)]
mod broyden;
mod calculator;
mod constants;
mod defaults;
mod diis;
mod fermi_occupation;
mod gamma_approximation;
mod gradients;
mod graph;
mod h0_and_s;
mod internal_coordinates;
mod molecule;
mod mulliken;
mod optimization;
mod parameters;
mod scc_routine;
mod scc_routine_unrestricted;
mod slako_transformations;
mod solver;
mod optimization;
mod internal_coordinates;
mod graph;
mod step;
//mod transition_charges;
//mod solver;
//mod scc_routine_unrestricted;

use crate::gradients::*;
use crate::molecule::Molecule;
use ndarray::*;
use ndarray_linalg::*;
use std::{env, fs};
use std::ptr::eq;
use std::time::{Duration, Instant};
#[macro_use]
extern crate clap;
use clap::{App, Arg};
use crate::io::{get_coordinates, GeneralConfig};
use toml;
use std::path::Path;
use crate::defaults::CONFIG_FILE_NAME;

fn main() {
    let matches = App::new(crate_name!())
        .version(crate_version!())
        .about("software package for tight-binding DFT calculations")
        .arg(Arg::new("xyz-File")
            .about("Sets the xyz file to use")
            .required(true)
            .index(1))
        .get_matches();

    // the file containing the cartesian coordinates is the only mandatory file to
    // start a calculation.
    let geometry_file = matches.value_of("xyz-File").unwrap();
    let (atomic_numbers, positions): (Vec<u8>, Array2<f64>) = get_coordinates(geometry_file);

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
    let config: GeneralConfig = toml::from_str(&config_string).unwrap();
    // save the configuration file if it does not exist already
    if config_file_path.exists() == false {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }

}

