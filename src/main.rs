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
mod io;
mod molecule;
mod mulliken;
mod optimization;
mod parameters;
mod scc_routine;
mod scc_routine_unrestricted;
mod slako_transformations;
mod solver;
mod step;
mod transition_charges;
mod zbrent;
mod test;
mod fmo;
//mod transition_charges;
//mod solver;
//mod scc_routine_unrestricted;

use crate::gradients::*;
use crate::molecule::Molecule;
use crate::solver::get_exc_energies;
use ndarray::*;
use ndarray_linalg::*;
use std::ptr::eq;
use std::time::{Duration, Instant};
use std::{env, fs};
#[macro_use]
extern crate clap;
use crate::defaults::CONFIG_FILE_NAME;
use crate::io::{get_coordinates, GeneralConfig, write_header};
use clap::{App, Arg};
use log::{error, warn, info, debug, trace, Level};
use std::path::Path;
use std::process;
use toml;
use std::io::Write;
use env_logger::Builder;
use log::LevelFilter;
use crate::optimization::optimize_geometry_ic;
use ron::error::ErrorCode::TrailingCharacters;

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

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

    let log_level: LevelFilter = match config.verbose {
         2 => LevelFilter::Trace,
         1 => LevelFilter::Debug,
         0 => LevelFilter::Info,
        -1 => LevelFilter::Warn,
        -2 => LevelFilter::Error,
         _ => LevelFilter::Info,
    };

    Builder::new()
        .format(|buf, record| {
            writeln!(buf,
                     "{}",
                     record.args()
            )
        })
        .filter(None, log_level)
        .init();

    write_header();
    let molecule_timer: Instant = Instant::now();
    info!("{: ^80}", "Initializing Molecule");
    info!("{:-^80}", "");
    info!("{: <25} {}", "geometry filename:", geometry_file);
    info!("{: <25} {}", "number of atoms:", atomic_numbers.len());

    let exit_code: i32 = match &config.jobtype[..] {
        "sp" => {
            let mut mol: Molecule = Molecule::new(
                atomic_numbers,
                positions,
                Some(config.mol.charge),
                Some(config.mol.multiplicity),
                None,
                None,
                config,
            );
            info!("{:>68} {:>8.2} s", "elapsed time:", molecule_timer.elapsed().as_secs_f32());
            drop(molecule_timer);
            info!("{:^80}", "");
            let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
                scc_routine::run_scc(&mol);

            mol.calculator.set_active_orbitals(f.to_vec());
            let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
                get_exc_energies(&f, &mol, Some(4), &s, &orbe, &orbs, false, None);

            0
        }
        "opt" => {
            let mut mol: Molecule = Molecule::new(
                atomic_numbers,
                positions,
                Some(config.mol.charge),
                Some(config.mol.multiplicity),
                None,
                None,
                config,
            );

            let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
                scc_routine::run_scc(&mol);
            mol.calculator.set_active_orbitals(f.to_vec());

            let tmp: (f64, Array1<f64>, Array1<f64>) = optimize_geometry_ic(&mut mol,Some(1));
            let new_energy: f64 = tmp.0;
            let new_gradient: Array1<f64> = tmp.1;
            let new_coords: Array1<f64> = tmp.2;

            let coords_3d: Array2<f64> = new_coords
                .clone()
                .into_shape((new_coords.len() / 3, 3))
                .unwrap();
            0
        }
        _ => {
            error!(
                "ERROR: The specified jobtype {} is not implemented.",
                config.jobtype
            );
            1
        }
    };
    info!("{: ^80}", "");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "::    Thank you for using TINCR     ::");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "");
    process::exit(exit_code);
}
