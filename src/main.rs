#![allow(dead_code)]
#![allow(warnings)]

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
use crate::io::{MoldenExporterBuilder, frame_to_coordinates};
use ndarray::prelude::*;
use crate::defaults::CONFIG_FILE_NAME;
use crate::io::{Configuration, write_header, read_file_to_frame, read_input, MoldenExporter};
use chemfiles::Frame;
use crate::initialization::{System, create_atoms, Atom, get_parametrization, initialize_gamma_function};
use crate::scc::scc_routine::RestrictedSCC;
use crate::scc::gamma_approximation::gamma_atomwise;
use crate::excited_states::ExcitedState;

use crate::utils::Timer;
use ndarray::{Array2, Array1};
use crate::scc::scc_routine_unrestricted::UnrestrictedSCC;


use crate::excited_states::davidson::Davidson;
use crate::excited_states::tda::*;
use crate::excited_states::{orbe_differences, trans_charges, initial_subspace, ProductCache};

use crate::fmo::gradients::GroundStateGradient;
use crate::fmo::SuperSystem;
use std::fs::File;
use ndarray_npy::write_npy;
use crate::param::slako::ParamFiles;


mod constants;
mod defaults;
mod io;
//mod optimization;
mod initialization;
mod scc;
mod utils;
mod fmo;
mod param;
mod excited_states;
mod gradients;
mod optimization;
mod data;

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
    // and the total wall-time timer is started.
    let timer: Timer = Timer::start();


    // Computations.
    // ................................................................

    // Check if SKF files should be used as for the parametrization or own files.
    let params: ParamFiles = if config.slater_koster.use_skf_files {
        ParamFiles::SKF(&config.slater_koster.path_to_skf)
    } else {
        ParamFiles::OWN
    };

    let (at_numbers, coords): (Vec<u8>, Array2<f64>) = frame_to_coordinates(frame.clone());

    let (u_atoms, atoms): (Vec<Atom>, Vec<Atom>) = create_atoms(&at_numbers, coords.view());
    let (slako, vrep) = get_parametrization(&u_atoms, params);
    // Count the number of orbitals
    let n_orbs: usize = atoms.iter().fold(0, |n, atom| n + atom.n_orbs);
    let gf = initialize_gamma_function(&u_atoms, 0.0);
    let gf_lr = initialize_gamma_function(&u_atoms, config.lc.long_range_radius);
    let gamma: Array2<f64> = gamma_atomwise(&gf, &atoms);
    let gamma_lr: Option<Array2<f64>> = if config.lc.long_range_correction {
        Some(gamma_atomwise(&gf_lr, &atoms))
    } else {
        None
    };
    let (s, h0): (Array2<f64>, Array2<f64>) = slako.h0_and_s(n_orbs, &atoms);



    if config.jobtype == "sp" {
        let mut system = System::from((frame, config.clone()));
        system.prepare_scc();
        system.run_scc();
        // system.prepare_tda();
        // system.run_tda(config.excited.nstates, 50, 1e-4);
    } else if config.jobtype == "fmo" {
        let mut system = SuperSystem::from((frame, config.clone()));
        //gamma_atomwise(&system.gammafunction, &system.atoms, system.atoms.len());
        system.prepare_scc();
        system.run_scc();
        // let molden_exp: MoldenExporter = MoldenExporterBuilder::default()
        //     .atoms(&system.atoms)
        //     .orbs(system.data.orbs()())
        //     .orbe(system.data.orbe())
        //     .f(system.data.occupation().to_vec())
        //     .build()
        //     .unwrap();

        let hamiltonian = system.create_exciton_hamiltonian();
    }

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
