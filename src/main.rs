#![allow(dead_code)]
// #![allow(warnings)]

use std::{env, fs};
use std::io::Write;
use std::path::Path;
use std::process;
use std::time::{Duration, Instant};

use clap::{App, Arg};
use env_logger::Builder;
use log::{info};
use log::LevelFilter;
use petgraph::stable_graph::*;
use toml;

use crate::defaults::CONFIG_FILE_NAME;
use crate::io::{Configuration, write_header, read_file_to_frame};
use chemfiles::Frame;
use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;

mod constants;
mod defaults;
mod io;
//mod optimization;
mod initialization;
mod scc;
mod utils;
//mod gradients;
mod param;
mod excited_states;

#[macro_use]
extern crate clap;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

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

    let log_level: LevelFilter = match 0 {
        2 => LevelFilter::Trace,
        1 => LevelFilter::Debug,
        0 => LevelFilter::Info,
        -1 => LevelFilter::Warn,
        -2 => LevelFilter::Error,
        _ => LevelFilter::Info,
    };

    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, log_level)
        .init();

    write_header();

    // the file containing the cartesian coordinates is the only mandatory file to
    // start a calculation.
    let geometry_file = matches.value_of("xyz-File").unwrap();
    let frame: Frame = read_file_to_frame(geometry_file);

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
    let config: Configuration = toml::from_str(&config_string).unwrap();
    // save the configuration file if it does not exist already
    if config_file_path.exists() == false {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }

    let mut system: System = System::from((frame, config));
    system.prepare_scc();
    system.run_scc();

    info!("{: ^80}", "");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "::    Thank you for using TINCR     ::");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "");
    process::exit(1);
}
