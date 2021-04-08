#![allow(dead_code)]
#![allow(warnings)]
mod broyden;
mod calculator;
mod constants;
mod defaults;
mod diis;
mod fermi_occupation;
mod fmo;
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
mod test;
mod transition_charges;
mod zbrent;
mod fmo_gradients;
//mod transition_charges;
//mod solver;
//mod scc_routine_unrestricted;

use crate::fmo::*;
use crate::gradients::*;
use crate::fmo_gradients::*;
use crate::molecule::Molecule;
use crate::solver::get_exc_energies;
use ndarray::*;
use ndarray_linalg::*;
use petgraph::stable_graph::*;
use std::ptr::eq;
use std::time::{Duration, Instant};
use std::{env, fs};
#[macro_use]
extern crate clap;
use crate::defaults::CONFIG_FILE_NAME;
use crate::io::{get_coordinates, write_header, GeneralConfig};
use crate::optimization::optimize_geometry_ic;
use clap::{App, Arg};
use env_logger::Builder;
use log::LevelFilter;
use log::{debug, error, info, trace, warn, Level};
use ron::error::ErrorCode::TrailingCharacters;
use std::io::Write;
use std::path::Path;
use std::process;
use toml;
use approx::AbsDiffEq;

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
        .format(|buf, record| writeln!(buf, "{}", record.args()))
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
                Some(0.0),
                None,
                config,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None
            );
            info!(
                "{:>68} {:>8.2} s",
                "elapsed time:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            info!("{:^80}", "");
            let molecule_timer: Instant = Instant::now();
            let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
                scc_routine::run_scc(&mut mol);
            info!(
                "{:>68} {:>8.2} s",
                "elapsed time calculate energy:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);

            mol.calculator.set_active_orbitals(f.to_vec());
            let (grad_e0, grad_vrep, grad_exc, empty_z_vec): (
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array3<f64>,
            ) = get_gradients(&orbe, &orbs, &s, &mol, &None, &None, None, &None, None);
            println!("gradE0 {}",grad_e0);
            println!("");
            println!("gradtotal {}",&grad_e0+ &grad_vrep);
            println!("");
            let num_grad:Array1<f64> = dftb_numerical_gradients(&mut mol);

            println!("numerical dftb gradient {}",num_grad);
            println!("");
            println!("Norm of difference: {}", (grad_e0+ grad_vrep-num_grad).norm());
            //mol.calculator.set_active_orbitals(f.to_vec());
            //let tmp: (Array1<f64>, Array3<f64>, Array3<f64>, Array3<f64>) =
            //    get_exc_energies(&f, &mol, Some(4), &s, &orbe, &orbs, false, None);

            0
        }
        "opt" => {
            let mut mol: Molecule = Molecule::new(
                atomic_numbers,
                positions,
                Some(config.mol.charge),
                Some(config.mol.multiplicity),
                Some(0.0),
                None,
                config,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None
            );
            info!(
                "{:>68} {:>8.2} s",
                "elapsed time:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);

            let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
                scc_routine::run_scc(&mut mol);
            mol.calculator.set_active_orbitals(f.to_vec());

            let tmp: (f64, Array1<f64>, Array1<f64>) = optimize_geometry_ic(&mut mol, Some(0));
            let new_energy: f64 = tmp.0;
            let new_gradient: Array1<f64> = tmp.1;
            let new_coords: Array1<f64> = tmp.2;

            let coords_3d: Array2<f64> = new_coords
                .clone()
                .into_shape((new_coords.len() / 3, 3))
                .unwrap();
            0
        }
        "fmo_grad" => {
            let (graph,graph_indexes, subgraph,connectivity_mat,dist_matrix, dir_matrix, prox_matrix): (StableUnGraph<u8, f64>, Vec<NodeIndex>, Vec<StableUnGraph<u8, f64>>,Array2<bool>,Array2<f64>, Array3<f64>, Array2<bool>) =
                create_fmo_graph(atomic_numbers.clone(), positions.clone());
            // let mut mol: Molecule = Molecule::new(
            //     atomic_numbers.clone(),
            //     positions.clone(),
            //     Some(config.mol.charge),
            //     Some(config.mol.multiplicity),
            //     Some(0.0),
            //     None,
            //     config.clone(),
            // );
            println!(
                "{:>68} {:>8.2} s",
                "elapsed time create_fmo_graph:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            let molecule_timer: Instant = Instant::now();
            let mut fragments: Vec<Molecule> = create_fragment_molecules(
                subgraph,
                config.clone(),
                atomic_numbers.clone(),
                positions.clone(),
            );
            println!(
                "{:>68} {:>8.2} s",
                "elapsed time create fragment mols:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            let molecule_timer: Instant = Instant::now();

            let (indices_frags, gamma_total,g1_total,prox_mat,dist_mat,direct_mat): (Vec<usize>, Array2<f64>,Array3<f64>, Array2<bool>,Array2<f64>, Array3<f64>) =
                reorder_molecule_gradients(&fragments, config.clone(), positions.raw_dim());

            println!(
                "{:>68} {:>8.2} s",
                "elapsed time reorder mol gradients:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            let molecule_timer: Instant = Instant::now();

            let fragments_data: Vec<frag_grad_result> = fmo_calculate_fragment_gradients(&mut fragments);

            println!(
                "{:>68} {:>8.2} s",
                "elapsed time calculate gradients monomers",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            let molecule_timer: Instant = Instant::now();

            let (pairs_data): (Vec<pair_grad_result>) =
                fmo_calculate_pairwise_gradients(&fragments, &fragments_data, config.clone(),&dist_mat,&direct_mat,&prox_mat,&indices_frags,&gamma_total);

            println!(
                "{:>68} {:>8.2} s",
                "elapsed time calculate gradients dimers",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            let molecule_timer: Instant = Instant::now();

            let gradients:Array1<f64> = fmo_gs_gradients(&fragments,&fragments_data,&pairs_data,&indices_frags,gamma_total,g1_total,prox_matrix);

            println!(
                "{:>68} {:>8.2} s",
                "elapsed time calculate total and embedding:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);

            let mut mol: Molecule = Molecule::new(
                atomic_numbers.clone(),
                positions,
                Some(config.mol.charge),
                Some(config.mol.multiplicity),
                Some(0.0),
                None,
                config.clone(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None
            );

            println!(" ");

            let (energy, orbs, orbe, s, f): (f64, Array2<f64>, Array1<f64>, Array2<f64>, Vec<f64>) =
                scc_routine::run_scc(&mut mol);
            mol.calculator.set_active_orbitals(f.to_vec());

            println!("SCC Energy {}",energy);

            let coords: Array1<f64> = mol.positions.clone().into_shape(3 * mol.n_atoms).unwrap();
            let numerical_gradient:Array1<f64> = fmo_numerical_gradient(&atomic_numbers,&coords,config.clone());
            let numerical_gradient_ridders:Array1<f64> = fmo_numerical_gradient_new(&atomic_numbers,&coords,config.clone());
            println!("Numerical gradient fmo ridders method {}",numerical_gradient_ridders);
            println!(" ");

            println!("FMO gradients {}", gradients);
            println!(" ");

            // println!("Difference between FMO and ridders:");
            let diff:Array1<f64> = (&gradients - &numerical_gradient_ridders).mapv(|val|val.abs());
            // println!("{}",diff);
            let max_gradient: f64 = diff.clone()
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
                .unwrap();
            println!("Max deviation: {}",max_gradient);
            println!("Norm of difference {}",(&gradients - &numerical_gradient_ridders).norm());
            println!(" ");

            // println!("Numerical gradient fmo {}",numerical_gradient);
            // println!("Norm of difference {}",(&gradients - &numerical_gradient).norm());
            // println!(" ");

            let (grad_e0, grad_vrep, grad_exc, empty_z_vec): (
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array3<f64>,
            ) = get_gradients(&orbe, &orbs, &s, &mol, &None, &None, None, &None, None);

            //let (en, grad): (f64, Array1<f64>) = optimization::get_energy_and_gradient_s0(&coords, &mut mol);

            println!("DFTB Gradient {}",&grad_e0 + &grad_vrep);
            // let num_grad:Array1<f64> = dftb_numerical_gradients(&mut mol);
            // println!("");
            // println!("numerical dftb gradient {}",num_grad);
            // println!("");
            // println!("Norm of difference: {}", (&(&grad_e0+ &grad_vrep)-&num_grad).norm());
            // println!("");
            // println!("Difference num grad fmo vs dftb {}",numerical_gradient-num_grad);
            // println!("");
            // println!("Difference dftb grad vs fmo grad {}",&gradients-&(&grad_e0+ &grad_vrep));
            0
        }
        "fmo" => {
            let (graph, graph_indexes, subgraph, connectivity_mat, dist_matrix, dir_matrix, prox_matrix): (StableUnGraph<u8, f64>, Vec<NodeIndex>, Vec<StableUnGraph<u8, f64>>, Array2<bool>, Array2<f64>, Array3<f64>, Array2<bool>) =
                create_fmo_graph(atomic_numbers.clone(), positions.clone());
            // let mut mol: Molecule = Molecule::new(
            //     atomic_numbers.clone(),
            //     positions.clone(),
            //     Some(config.mol.charge),
            //     Some(config.mol.multiplicity),
            //     Some(0.0),
            //     None,
            //     config.clone(),
            // );
            println!(
                "{:>68} {:>8.2} s",
                "elapsed time create_fmo_graph:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            let molecule_timer: Instant = Instant::now();
            let mut fragments: Vec<Molecule> = create_fragment_molecules(
                subgraph,
                config.clone(),
                atomic_numbers.clone(),
                positions.clone(),
            );
            let (indices_frags, gamma_total, prox_mat, dist_mat, direct_mat): (Vec<usize>, Array2<f64>, Array2<bool>, Array2<f64>, Array3<f64>) =
                reorder_molecule(&fragments, config.clone(), positions.raw_dim());
            println!(
                "{:>68} {:>8.2} s",
                "elapsed time create fragment mols:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);

            let molecule_timer: Instant = Instant::now();
            let fragments_data: cluster_frag_result = fmo_calculate_fragments(&mut fragments);

            println!(
                "{:>68} {:>8.2} s",
                "elapsed time calculate monomers",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            let molecule_timer: Instant = Instant::now();

            let (h0, pairs_data): (Array2<f64>, Vec<pair_result>) =
                fmo_calculate_pairwise_single(&fragments, &fragments_data, config.clone(), &dist_mat, &direct_mat, &prox_mat, &indices_frags, &gamma_total);

            println!(
                "{:>68} {:>8.2} s",
                "elapsed time calculate dimers",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);
            let molecule_timer: Instant = Instant::now();
            let energy: f64 = fmo_gs_energy(
                &fragments,
                &fragments_data,
                &pairs_data,
                &indices_frags,
                gamma_total,
                prox_mat,
            );
            println!(
                "{:>68} {:>8.2} s",
                "elapsed time calculate total and embedding:",
                molecule_timer.elapsed().as_secs_f32()
            );
            drop(molecule_timer);

            println!("FMO Energy {}", energy);

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
