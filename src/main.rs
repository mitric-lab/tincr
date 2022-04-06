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
use crate::io::MoldenExporterBuilder;
use ndarray::prelude::*;
use crate::defaults::CONFIG_FILE_NAME;
use crate::io::{Configuration, write_header, read_file_to_frame, read_input, MoldenExporter,read_dynamic_input,create_dynamics_data,read_dynamic_input_ehrenfest};
use chemfiles::Frame;
use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use crate::scc::gamma_approximation::gamma_atomwise;
use crate::excited_states::ExcitedState;

use crate::utils::Timer;
use ndarray::{Array2, Array1};
use crate::scc::scc_routine_unrestricted::UnrestrictedSCC;
use rusty_fish::initialization::{DynamicConfiguration, SystemData, Simulation};

use crate::excited_states::davidson::Davidson;
use crate::excited_states::tda::*;
use crate::excited_states::{orbe_differences, trans_charges, initial_subspace, ProductCache};

use crate::fmo::gradients::GroundStateGradient;
use crate::fmo::SuperSystem;
use std::fs::File;
use ndarray_npy::{write_npy, NpzWriter};
use crate::fmo::cis_gradient::ReducedBasisState;


mod constants;
mod defaults;
mod io;
mod dynamics;
//mod optimization;
mod initialization;
mod scc;
mod utils;
mod fmo;
mod param;
mod excited_states;
mod gradients;
mod properties;
mod optimization;
mod coupling;

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

    // create config for the dynamic simulation
    // let dynamics_config:DynamicConfiguration = read_dynamic_input(&config);

    // Computations.
    // ................................................................
    if config.jobtype == "sp" {
        let mut system = System::from((frame, config.clone()));
        // let dynamics_data:SystemData = create_dynamics_data(&system.atoms,dynamics_config);
        // create the struct which starts the dynamics
        // let mut dynamic: Simulation = Simulation::new(&dynamics_data);
        // dynamic.verlet_dynamics(&mut system);
        system.prepare_scc();
        system.run_scc();
        // system.test_tda_lc_gradient();
        // system.optimize_cartesian(Some(0));w
        system.prepare_tda();
        system.run_tda(config.excited.nstates, 150, 1e-4);
        // system.ground_state_gradient(true);
        // system.prepare_excited_grad();
        // let grad_exc:Array1<f64> = system.tda_gradient_lc(0);
        // write_npy("dimer_gradient.npy",&grad_exc);

    } else if config.jobtype == "fmo" {
        let mut system = SuperSystem::from((frame, config.clone()));
        let n_monomer:usize = system.monomers.len();
        let dynamics_config:DynamicConfiguration = read_dynamic_input_ehrenfest(&config,n_monomer);
        let dynamics_data:SystemData = create_dynamics_data(&system.atoms,dynamics_config);
        let mut dynamic: Simulation = Simulation::new(&dynamics_data);
        // dynamic.ehrenfest_dynamics(&mut system);

        let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
        let mut npz_c = NpzWriter::new(File::create("cis_arrays.npz").unwrap());
        let mut npz_q = NpzWriter::new(File::create("qtrans_arrays.npz").unwrap());
        let mut npz_mo = NpzWriter::new(File::create("mo_arrays.npz").unwrap());
        let mut npz_h = NpzWriter::new(File::create("h_arrays.npz").unwrap());
        let mut npz_x = NpzWriter::new(File::create("x_arrays.npz").unwrap());
        let mut npz_sign = NpzWriter::new(File::create("signs_arrays.npz").unwrap());
        let mut npz_sc = NpzWriter::new(File::create("sc_arrays.npz").unwrap());
        dynamic.initialize_ehrenfest(
            &mut system,
            &mut npz,
            0,
            &mut npz_c,
            &mut npz_q,
            &mut npz_mo,
            &mut npz_h,
            &mut npz_x,
            &mut npz_sign,
            &mut npz_sc,
        );
        let old_system = system.clone();
        system.properties.set_old_supersystem(old_system);
        for step in 1..dynamic.config.nstep{
            dynamic.ehrenfest_step(
                &mut system,
                &mut npz,
                step,
                &mut npz_c,
                &mut npz_q,
                &mut npz_mo,
                &mut npz_h,
                &mut npz_x,
                &mut npz_sign,
                &mut npz_sc,
            );
            let old_system = system.clone();
            system.properties.set_old_supersystem(old_system);
        }
        npz.finish();
        npz_c.finish();
        npz_q.finish();
        npz_mo.finish();
        npz_h.finish();
        npz_x.finish();
        npz_sign.finish();
        npz_sc.finish();

        // system.prepare_scc();
        // system.run_scc();
        // println!("Test");
        // let mo_coeff:Array2<f64> = system.monomers[0].properties.orbs().unwrap().to_owned();
        // let h_mat:Array2<f64> = system.monomers[0].properties.h_coul_transformed().unwrap().to_owned();
        //
        // for i in 0..30{
        //    for monomer in system.monomers.iter_mut(){
        //        monomer.properties.set_n_virt(i);
        //    }
        //    println!("Iterataion {}",i);
        //    println!(" ");
        //    for monomer in system.monomers.iter_mut() {
        //        monomer.properties.reset();
        //    }
        //    for pair in system.pairs.iter_mut() {
        //        pair.properties.reset();
        //    }
        //    for esd_pair in system.esd_pairs.iter_mut() {
        //        esd_pair.properties.reset();
        //    }
        //    system.properties.reset();
        //
        //    system.prepare_scc();
        //    system.run_scc();
        //
        //    let mo:Array2<f64> = system.monomers[0].properties.orbs().unwrap().to_owned();
        //    let h:Array2<f64> = system.monomers[0].properties.h_coul_transformed().unwrap().to_owned();
        //    let diff:Array2<f64> = &mo_coeff - &mo;
        //    for elem in diff.iter(){
        //        if elem.abs() > 2.0e-12{
        //            println!("MO coefficients deviate! {}",elem);
        //        }
        //    }
        //    let diff_h:Array2<f64> = &h_mat - &h;
        //    for elem in diff_h.iter(){
        //        if elem.abs() > 2.6e-12{
        //            println!("Hamiltonian deviates! {}",elem);
        //        }
        //    }
        // }
        // println!("{}",mo_coeff);
        // println!("{}",h_mat);

        // system.prepare_scc();
        // system.run_scc();
        // let (diabatic_hamiltonian,states):(Array2<f64>,Vec<ReducedBasisState>) = system.create_diabatic_hamiltonian();
        // for (idx,state) in states.iter().enumerate(){
        //     println!("State {}: ",idx);
        //     println!("{} \n",state);
        // }
        // for atom in system.atoms.iter(){
        //     print!("{} \t",atom.name);
        //     for data in atom.xyz.iter(){
        //         print!("{} \t",data * constants::BOHR_TO_ANGS);
        //     }
        //     println!(" ");
        // }
        // assert!(1==2);
        // system.test_diabatic_overlap();

        // system.create_exciton_hamiltonian();
        // system.test_le_gradient();
        // system.test_ct_gradient();
        // system.test_numerical_ci_coeff_gradient(0,6);
        // system.test_le_le_coupling_gradient();
        // system.test_le_ct_coupling_gradient();
        // system.test_ct_ct_coupling_gradient();
        // create the struct which starts the dynamics
        // let dynamics_data:SystemData = create_dynamics_data(&system.atoms,dynamics_config);
        // let mut dynamic: Simulation = Simulation::new(&dynamics_data,&mut system);
        // dynamic.verlet_dynamics();
        //gamma_atomwise(&system.gammafunction, &system.atoms, system.atoms.len());
        // system.prepare_scc();
        // system.run_scc();
        // system.test_le_gradient();
        // system.test_ct_gradient();
        // system.optimize_cartesian(Some(1));
        // system.test_ct_gradient();
        // system.test_orbital_energy_derivative();
        // let molden_exp: MoldenExporter = MoldenExporterBuilder::default()
        //     .atoms(&system.atoms)
        //     .orbs(system.properties.orbs().unwrap())
        //     .orbe(system.properties.orbe().unwrap())
        //     .f(system.properties.occupation().unwrap().to_vec())
        //     .build()
        //     .unwrap();

        // println!("norbs occ {:?}",system.monomers[0].properties.occ_indices());
        // println!("norbs virt {:?}",system.monomers[0].properties.virt_indices());
        // write_npy("/home/einseler/Downloads/s_matrix.npy", &system.properties.s().unwrap());
        // write_npy("/home/einseler/Downloads/gamma_matrix.npy", &system.properties.gamma().unwrap());
        // write_npy("/home/einseler/Downloads/gamma_lr_matrix.npy", &system.properties.gamma_lr().unwrap());
        // write_npy("/home/einseler/Downloads/coeff_a.npy", &system.monomers[0].properties.orbs().unwrap());
        // write_npy("/home/einseler/Downloads/coeff_b.npy", &system.monomers[1].properties.orbs().unwrap());
        // system.test_orbital_energy_derivative();

        // let hamiltonian = system.create_exciton_hamiltonian();
        // println!("integral {}",hamiltonian);
        // let orbital_vec:Vec<usize> = system.atoms.iter().map(|atom| atom.n_orbs).collect();
        // println!("orbital vec {:?}",orbital_vec);
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
