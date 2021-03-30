#![allow(dead_code)]
#![allow(warnings)]

use std::{env, fs};
use std::io::Write;
use std::path::Path;
use std::process;
use std::ptr::eq;
use std::time::{Duration, Instant};

use clap::{App, Arg};
use env_logger::Builder;
use log::{debug, error, info, Level, trace, warn};
use log::LevelFilter;
use ndarray::*;
use ndarray_linalg::*;
use petgraph::stable_graph::*;
use ron::error::ErrorCode::TrailingCharacters;
use toml;

use crate::defaults::CONFIG_FILE_NAME;
use crate::io::{Configuration, write_header};

mod constants;
mod defaults;
mod h0_and_s;
mod io;
//mod optimization;
mod initialization;
mod scc;
mod utils;
//mod gradients;
mod param;
// mod excited_states;

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
    // let molecule_timer: Instant = Instant::now();
    // info!("{: ^80}", "Initializing Molecule");
    // info!("{:-^80}", "");
    // info!("{: <25} {}", "geometry filename:", geometry_file);
    // info!("{: <25} {}", "number of atoms:", atomic_numbers.len());

    // let exit_code: i32 = match &config.jobtype[..] {
    //     "sp" => {
    //         let mut mol: Molecule = Molecule::new(
    //             atomic_numbers,
    //             positions,
    //             Some(config.mol.charge),
    //             Some(config.mol.multiplicity),
    //             Some(0.0),
    //             None,
    //             config,
    //             None,
    //         );
    //         info!(
    //             "{:>68} {:>8.2} s",
    //             "elapsed time:",
    //             molecule_timer.elapsed().as_secs_f32()
    //         );
    //         drop(molecule_timer);
    //         info!("{:^80}", "");
    //         let molecule_timer: Instant = Instant::now();
    //         let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
    //             scc_routine::run_scc(&mut mol);
    //         info!(
    //             "{:>68} {:>8.2} s",
    //             "elapsed time calculate energy:",
    //             molecule_timer.elapsed().as_secs_f32()
    //         );
    //         drop(molecule_timer);
    //
    //         //mol.calculator.set_active_orbitals(f.to_vec());
    //         //let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
    //         //    get_exc_energies(&f, &mol, Some(4), &s, &orbe, &orbs, false, None);
    //
    //         0
    //     }
    //     "opt" => {
    //         let mut mol: Molecule = Molecule::new(
    //             atomic_numbers,
    //             positions,
    //             Some(config.mol.charge),
    //             Some(config.mol.multiplicity),
    //             None,
    //             None,
    //             config,
    //             None,
    //         );
    //         info!(
    //             "{:>68} {:>8.2} s",
    //             "elapsed time:",
    //             molecule_timer.elapsed().as_secs_f32()
    //         );
    //         drop(molecule_timer);
    //
    //         let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
    //             scc_routine::run_scc(&mut mol);
    //         mol.calculator.set_active_orbitals(f.to_vec());
    //
    //         let tmp: (f64, Array1<f64>, Array1<f64>) = optimize_geometry_ic(&mut mol, Some(1));
    //         let new_energy: f64 = tmp.0;
    //         let new_gradient: Array1<f64> = tmp.1;
    //         let new_coords: Array1<f64> = tmp.2;
    //
    //         let coords_3d: Array2<f64> = new_coords
    //             .clone()
    //             .into_shape((new_coords.len() / 3, 3))
    //             .unwrap();
    //         0
    //     }
    //     "fmo" => {
    //         let (graph, subgraph): (StableUnGraph<u8, f64>, Vec<StableUnGraph<u8, f64>>) =
    //             create_fmo_graph(atomic_numbers.clone(), positions.clone());
    //         // let mut mol: Molecule = Molecule::new(
    //         //     atomic_numbers.clone(),
    //         //     positions.clone(),
    //         //     Some(config.mol.charge),
    //         //     Some(config.mol.multiplicity),
    //         //     Some(0.0),
    //         //     None,
    //         //     config.clone(),
    //         // );
    //         info!(
    //             "{:>68} {:>8.2} s",
    //             "elapsed time graph:",
    //             molecule_timer.elapsed().as_secs_f32()
    //         );
    //         drop(molecule_timer);
    //         let molecule_timer: Instant = Instant::now();
    //         let mut fragments: Vec<Molecule> = create_fragment_molecules(
    //             subgraph,
    //             config.clone(),
    //             atomic_numbers.clone(),
    //             positions.clone(),
    //         );
    //         let (indices_frags, gamma_total, prox_matrix): (Vec<usize>, Array2<f64>, Array2<bool>) =
    //             reorder_molecule(&fragments, config.clone(), positions.raw_dim());
    //         info!(
    //             "{:>68} {:>8.2} s",
    //             "elapsed time create fragment mols:",
    //             molecule_timer.elapsed().as_secs_f32()
    //         );
    //         drop(molecule_timer);
    //
    //         let molecule_timer: Instant = Instant::now();
    //         let fragments_data: cluster_frag_result = fmo_calculate_fragments(&mut fragments);
    //         let (h0, pairs_data): (Array2<f64>, Vec<pair_result>) =
    //             fmo_calculate_pairwise_par(&fragments, &fragments_data, config.clone());
    //         let energy: f64 = fmo_gs_energy(
    //             &fragments,
    //             &fragments_data,
    //             &pairs_data,
    //             &indices_frags,
    //             gamma_total,
    //             prox_matrix,
    //         );
    //         info!(
    //             "{:>68} {:>8.2} s",
    //             "elapsed time calculate energy:",
    //             molecule_timer.elapsed().as_secs_f32()
    //         );
    //         drop(molecule_timer);
    //
    //         println!("FMO Energy {}", energy);
    //         0
    //     }
    //     _ => {
    //         error!(
    //             "ERROR: The specified jobtype {} is not implemented.",
    //             config.jobtype
    //         );
    //         1
    //     }
    // };
    info!("{: ^80}", "");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "::    Thank you for using TINCR     ::");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "");
    process::exit(1);
}
