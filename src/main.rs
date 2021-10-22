#![allow(dead_code)]
#![allow(warnings)]

use core::::*;
use core::::{initial_subspace, orbe_differences, ProductCache, trans_charges};
use core::::davidson::Davidson;
use core::::ExcitedState;
use core::::gamma_approximation::gamma_atomwise;
use core::::scc_routine::RestrictedSCC;
use core::::scc_routine_unrestricted::UnrestrictedSCC;
use core::::tda::*;
use core::defaults::CONFIG_FILE_NAME;
use std::{env, fs};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process;
use std::time::{Duration, Instant};

use chemfiles::Frame;
use chrono::offset::LocalResult::Single;
use clap::{App, Arg};
use env_logger::{Builder, init};
use log::info;
use log::LevelFilter;
use ndarray::{Array1, Array2};
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use petgraph::stable_graph::*;
use toml;

use crate::data::Parametrization;
use crate::driver::JobType::Force;
use crate::fmo::{SuperSystem, SuperSystemSetup};
use crate::fmo::gradients::GroundStateGradient;
use crate::io::{frame_to_coordinates, MoldenExporterBuilder};
use crate::io::{Configuration, MoldenExporter, read_file_to_frame, read_input, write_header};
use crate::param::slako::ParamFiles;
use crate::utils::Timer;

mod io;
mod utils;
mod fmo;
mod param;
mod gradients;
mod optimization;
mod data;
mod driver;

#[macro_use]
extern crate clap;

fn main() {

    // Input.
    let matches = App::new(crate_name!())
        .version(crate_version!())
        .about("software package for tight-binding DFT calculations")
        .arg(
            Arg::new("xyz-File")
                .about("Sets the xyz file to use")
                .required(true)
                .index(1),
        )
        .get_matches();

    // The total wall-time timer is started.
    let timer: Timer = Timer::start();

    // The file containing the cartesian coordinates is the only mandatory file to
    // start a calculation.
    let geometry_file = matches.value_of("xyz-File").unwrap();
    let (frame, config): (Frame, Configuration) = read_input(geometry_file);

    // Multithreading.
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallelization.number_of_cores)
        .build_global()
        .unwrap();

    // Logging.
    // The log level is set.
    let log_level: LevelFilter = match config.verbose {
        2 => LevelFilter::Trace,
        1 => LevelFilter::Debug,
        0 => LevelFilter::Info,
        -1 => LevelFilter::Warn,
        -2 => LevelFilter::Error,
        _ => LevelFilter::Info,
    };
    // and the logger is build.
    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, log_level)
        .init();

    // The program header is written to the command line.
    write_header();


    // The computations start at this point.
    // ................................................................

    // Atomic numbers and xyz-coordinates are extracted from the Frame object.
    let (at_numbers, coords): (Vec<u8>, Array2<f64>) = frame_to_coordinates(frame.clone());

    // Two vectors of Atom's are created. The first one with the unique elements and the second
    // with all atoms from the given geometry.
    let (u_atoms, atoms): (AtomVec, AtomVec) = create_atoms(&at_numbers, coords.view());

    // Get the parametrization from the corresponding files.
    let params = Parameters::new(&config, u_atoms.as_slice());

    // Compute the Gamma, H0 and S matrix for the current geometry.
    let param_data = params.compute_matrices(&config, atoms.as_slice());

    let setup = SuperSystemSetup::new(atoms.as_slice());



    // let path = Path::new("/Users/hochej/Downloads/test.molden");
    // molden_exp.write_to(path);

    // ................................................................

    // Finished.
    // The total wall-time is printed together with the end statement.
    info!("{}", timer);
    info!("{: ^80}", "");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "::    Thank you for using TINCR     ::");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "");
    process::exit(1);
}
