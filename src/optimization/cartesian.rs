use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use crate::optimization::helpers::*;
use crate::constants;
use ndarray::prelude::*;
use ndarray_linalg::Norm;
use log::{debug, info, log_enabled, trace, warn, Level};
use crate::fmo::SuperSystem;

impl System{
    pub fn optimize_cartesian(&mut self, state:Option<usize>){
        // solve the following optimization problem:
        // minimize f(x) subject to  c_i(x) > 0   for  i=1,...,m
        // where f(x) is a scalar function, x is a real vector of size n
        // References
        // ----------
        // [1] J. Nocedal, S. Wright, 'Numerical Optimization', Springer, 2006

        // set the electronic state
        let state:usize = state.unwrap_or(0);

        // start the optimization
        let (coordinates, gradient) = self.cartesian_optimization_loop(state);

        let new_coords:Array2<f64> = constants::BOHR_TO_ANGS *
            coordinates.into_shape((self.n_atoms,3)).unwrap();
        if log_enabled!(Level::Warn) {
            warn!("final coordinates after the optimization:");
            for (ind,atom) in self.atoms.iter().enumerate(){
                warn!(
                    "{: >5} {:>18.10} {:>18.10} {:>18.10}",
                    atom.name,
                    new_coords[[ind,0]],
                    new_coords[[ind,1]],
                    new_coords[[ind,2]]
                );
            }
            warn!("");
        }
    }

    pub fn calculate_energy_and_gradient(&mut self, state:usize)->(f64,Array1<f64>){
        let mut energy: f64 = 0.0;
        let mut gradient: Array1<f64> = Array::zeros(3 * self.n_atoms);

        if state == 0{
            // ground state energy and gradient
            self.prepare_scc();
            energy = self.run_scc().unwrap();
            gradient = self.ground_state_gradient(false);
        }
        else{
            // excited state calculation
            let excited_state:usize = state -1;
            self.prepare_scc();
            energy = self.run_scc().unwrap();

            // calculate excited states
            self.prepare_tda();
            self.run_tda(self.config.excited.nstates, 100, 1e-4);
            energy += self.properties.ci_eigenvalue(excited_state).unwrap();

            gradient = self.ground_state_gradient(true);
            self.prepare_excited_grad();

            if self.config.lc.long_range_correction{
                gradient = gradient + self.tda_gradient_lc(excited_state);
            }
            else{
                gradient = gradient + self.tda_gradient_nolc(excited_state);
            }
        }
        self.properties.reset();

        return (energy,gradient);
    }

    pub fn calculate_energy_line_search(&mut self, state:usize)->f64{
        let mut energy: f64 = 0.0;

        if state == 0{
            // ground state energy and gradient
            self.prepare_scc();
            energy = self.run_scc().unwrap();
        }
        else{
            // excited state calculation
            let excited_state:usize = state -1;
            self.prepare_scc();
            energy = self.run_scc().unwrap();

            // calculate excited states
            self.prepare_tda();
            self.run_tda(self.config.excited.nstates, 100, 1e-4);
            energy += self.properties.ci_eigenvalue(excited_state).unwrap();
        }
        self.properties.reset();

        return (energy);
    }

    fn cartesian_optimization_loop(&mut self,state:usize)->(Array1<f64>,Array1<f64>){
        // get coordinates
        let coords:Array1<f64> = self.get_xyz();

        // set defaults
        let maxiter: usize = 5000;
        let gtol: f64 = 1.0e-5;
        let ftol: f64 = 1.0e-7;
        let method: String = String::from("BFGS");
        let line_search: String = String::from("largest");

        let n: usize = coords.len();
        let mut x_old: Array1<f64> = coords.clone();

        // variables for the storage of the energy and gradient
        let mut fk: f64 = 0.0;
        let mut grad_fk: Array1<f64> = Array::zeros(n);

        // calculate energy and gradient
        let tmp: (f64, Array1<f64>) = self.calculate_energy_and_gradient(state);
        fk = tmp.0;
        grad_fk = tmp.1;

        let mut pk: Array1<f64> = Array::zeros(n);
        let mut x_kp1: Array1<f64> = Array::zeros(n);
        let mut sk: Array1<f64> = Array::zeros(n);
        let mut yk: Array1<f64> = Array::zeros(n);
        let mut inv_hk: Array2<f64> = Array::eye(n);
        let mut iterations:usize = 0;

        // vector of atom names
        let atom_names:Vec<String> = self.atoms.iter().map(|atom| String::from(atom.name)).collect();
        let first_coords:Array2<f64> = constants::BOHR_TO_ANGS* &coords.view().into_shape([self.n_atoms,3]).unwrap();
        let xyz_out:XYZ_Output =
            XYZ_Output::new(
                atom_names.clone(),
                first_coords);

        write_xyz_custom(&xyz_out);

        'optimization_loop:for k in 0..maxiter {
            println!("iteration {}", k);

            if method == "BFGS" {
                if k > 0 {
                    if yk.dot(&sk) <= 0.0 {
                        println!("yk {}", yk);
                        println!("sk {}", sk);
                        println!("Warning: positive definiteness of Hessian approximation lost in BFGS update, since yk.sk <= 0!")
                    }

                    inv_hk = bfgs_update(inv_hk.view(), sk.view(), yk.view(), k);
                }
                pk = inv_hk.dot(&(-&grad_fk));

            } else if method == "Steepest Descent" {
                pk = -grad_fk.clone();
            }

            if line_search == "Armijo" {
                x_kp1 = self.armijo_line_search(
                    x_old.view(), fk, grad_fk.view(), pk.view(),state);
            } else if line_search == "largest" {
                let amax = 1.0;
                x_kp1 = &x_old + &(amax * &pk);
            }
            let mut f_kp1: f64 = 0.0;
            let mut grad_f_kp1: Array1<f64> = Array::zeros(n);

            // update coordinates
            self.update_xyz(x_kp1.clone());
            // calculate new energy and gradient
            let tmp: (f64, Array1<f64>) = self.calculate_energy_and_gradient(state);
            f_kp1 = tmp.0;
            grad_f_kp1 = tmp.1;

            // check convergence
            let f_change: f64 = (f_kp1 - fk).abs();
            let gnorm: f64 = grad_f_kp1.norm();
            if f_change < ftol && gnorm < gtol {
                // set the last coordinates and gradient
                sk = &x_kp1 - &x_old;
                x_old = x_kp1;
                grad_fk = grad_f_kp1;
                fk = f_kp1;
                iterations = k;

                println!("k = {}, f(x) = {1:.8}, |x(k+1)-x(k)| = {2:.8}, |f(k+1)-f(k)| = {3:.8}, |df/dx| = {4:.8}",k,fk,sk.norm(),f_change,gnorm);
                break 'optimization_loop;
            }

            // step vector
            sk = &x_kp1 - &x_old;
            // gradient difference vector
            yk = &grad_f_kp1 - &grad_fk;
            // new variables for step k become old ones for step k+1
            x_old = x_kp1.clone();
            fk = f_kp1;
            grad_fk = grad_f_kp1;

            // print convergence criteria
            println!("k = {}, f(x) = {1:.8}, |x(k+1)-x(k)| = {2:.8}, |f(k+1)-f(k)| = {3:.8}, |df/dx| = {4:.8}",k,fk,sk.norm(),f_change,gnorm);

            let new_coords:Array2<f64> = constants::BOHR_TO_ANGS * &x_old.view().into_shape((self.n_atoms,3)).unwrap();
            let xyz_out:XYZ_Output =
                XYZ_Output::new(
                    atom_names.clone(),
                    new_coords.clone().into_shape([self.n_atoms,3]).unwrap());
            write_xyz_custom(&xyz_out);
            let opt_energy:Opt_Energy_Output = Opt_Energy_Output::new(k,fk);
            write_opt_energy(&opt_energy);
        }
        let new_coords:Array2<f64> = constants::BOHR_TO_ANGS * &x_old.view().into_shape((self.n_atoms,3)).unwrap();
        let xyz_out:XYZ_Output =
            XYZ_Output::new(
                atom_names.clone(),
                new_coords.clone().into_shape([self.n_atoms,3]).unwrap());
        write_xyz_custom(&xyz_out);
        let opt_energy:Opt_Energy_Output = Opt_Energy_Output::new(iterations,fk);
        write_opt_energy(&opt_energy);

        return (x_old,grad_fk);
    }
}

impl SuperSystem{
    pub fn optimize_cartesian(&mut self, state:Option<usize>){
        // solve the following optimization problem:
        // minimize f(x) subject to  c_i(x) > 0   for  i=1,...,m
        // where f(x) is a scalar function, x is a real vector of size n
        // References
        // ----------
        // [1] J. Nocedal, S. Wright, 'Numerical Optimization', Springer, 2006

        // set the electronic state
        let state:usize = state.unwrap_or(0);

        // start the optimization
        let (coordinates, gradient) = self.cartesian_optimization_loop(state);
        let n_atoms:usize = self.atoms.len();

        let new_coords:Array2<f64> = constants::BOHR_TO_ANGS *
            coordinates.into_shape((n_atoms,3)).unwrap();
        if log_enabled!(Level::Warn) {
            warn!("final coordinates after the optimization:");
            for (ind,atom) in self.atoms.iter().enumerate(){
                warn!(
                    "{: >5} {:>18.10} {:>18.10} {:>18.10}",
                    atom.name,
                    new_coords[[ind,0]],
                    new_coords[[ind,1]],
                    new_coords[[ind,2]]
                );
            }
            warn!("");
        }
    }

    pub fn calculate_energy_and_gradient(&mut self, state:usize)->(f64,Array1<f64>){
        let mut energy: f64 = 0.0;
        let n_atoms:usize = self.atoms.len();
        let mut gradient: Array1<f64> = Array::zeros(3 * n_atoms);

        if state == 0{
            // ground state energy and gradient
            self.prepare_scc();
            energy = self.run_scc().unwrap();
            gradient = self.ground_state_gradient();
        }
        else{
            // excited state calculation
            let excited_state:usize = state -1;
            self.prepare_scc();
            energy = self.run_scc().unwrap();
            gradient = self.ground_state_gradient();

            // calculate excited states
            self.create_exciton_hamiltonian();
            energy += self.properties.ci_eigenvalue(excited_state).unwrap();
            gradient = gradient + self.fmo_cis_gradient(excited_state);

        }
        for monomer in self.monomers.iter_mut(){
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.properties.reset();

        return (energy,gradient);
    }

    pub fn calculate_energy_line_search(&mut self, state:usize)->f64{
        let mut energy: f64 = 0.0;

        if state == 0{
            // ground state energy and gradient
            self.prepare_scc();
            energy = self.run_scc().unwrap();
        }
        else{
            // excited state calculation
            let excited_state:usize = state -1;
            self.prepare_scc();
            energy = self.run_scc().unwrap();

            // calculate excited states
            self.create_exciton_hamiltonian();
            energy += self.properties.ci_eigenvalue(excited_state).unwrap();
        }
        for monomer in self.monomers.iter_mut(){
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.properties.reset();

        return (energy);
    }

    fn cartesian_optimization_loop(&mut self,state:usize)->(Array1<f64>,Array1<f64>){
        // get coordinates
        let coords:Array1<f64> = self.get_xyz();
        let n_atoms:usize = self.atoms.len();

        // set defaults
        let maxiter: usize = 5000;
        let gtol: f64 = 1.0e-6;
        let ftol: f64 = 1.0e-8;
        let method: String = String::from("Steepest Descent");
        let line_search: String = String::from("largest");

        let n: usize = coords.len();
        let mut x_old: Array1<f64> = coords.clone();

        // variables for the storage of the energy and gradient
        let mut fk: f64 = 0.0;
        let mut grad_fk: Array1<f64> = Array::zeros(n);

        // calculate energy and gradient
        let tmp: (f64, Array1<f64>) = self.calculate_energy_and_gradient(state);
        fk = tmp.0;
        grad_fk = tmp.1;

        let mut pk: Array1<f64> = Array::zeros(n);
        let mut x_kp1: Array1<f64> = Array::zeros(n);
        let mut sk: Array1<f64> = Array::zeros(n);
        let mut yk: Array1<f64> = Array::zeros(n);
        let mut inv_hk: Array2<f64> = Array::eye(n);

        // vector of atom names
        let atom_names:Vec<String> = self.atoms.iter().map(|atom| String::from(atom.name)).collect();
        let first_coords:Array2<f64> = constants::BOHR_TO_ANGS* &coords.view().into_shape([n_atoms,3]).unwrap();
        let xyz_out:XYZ_Output =
            XYZ_Output::new(
                atom_names.clone(),
                first_coords);

        write_xyz_custom(&xyz_out);

        'optimization_loop:for k in 0..maxiter {
            println!("iteration {}", k);

            if method == "BFGS" {
                if k > 0 {
                    if yk.dot(&sk) <= 0.0 {
                        println!("yk {}", yk);
                        println!("sk {}", sk);
                        println!("Warning: positive definiteness of Hessian approximation lost in BFGS update, since yk.sk <= 0!")
                    }

                    inv_hk = bfgs_update(inv_hk.view(), sk.view(), yk.view(), k);
                }
                pk = inv_hk.dot(&(-&grad_fk));

            } else if method == "Steepest Descent" {
                pk = -grad_fk.clone();
            }

            if line_search == "Armijo" {
                x_kp1 = self.armijo_line_search(
                    x_old.view(), fk, grad_fk.view(), pk.view(),state);
            } else if line_search == "largest" {
                let amax = 1.0;
                x_kp1 = &x_old + &(amax * &pk);
            }
            let mut f_kp1: f64 = 0.0;
            let mut grad_f_kp1: Array1<f64> = Array::zeros(n);

            // update coordinates and the overlap matrix
            self.update_xyz(x_kp1.clone());
            self.update_s();
            // calculate new energy and gradient
            let tmp: (f64, Array1<f64>) = self.calculate_energy_and_gradient(state);
            f_kp1 = tmp.0;
            grad_f_kp1 = tmp.1;

            // check convergence
            let f_change: f64 = (f_kp1 - fk).abs();
            let gnorm: f64 = grad_f_kp1.norm();
            if f_change < ftol && gnorm < gtol {
                // set the last coordinates and gradient
                x_old = x_kp1;
                grad_fk = grad_f_kp1;
                break 'optimization_loop;
            }

            // step vector
            sk = &x_kp1 - &x_old;
            // gradient difference vector
            yk = &grad_f_kp1 - &grad_fk;
            // new variables for step k become old ones for step k+1
            x_old = x_kp1.clone();
            fk = f_kp1;
            grad_fk = grad_f_kp1;

            // print convergence criteria
            println!("k = {}, f(x) = {1:.8}, |x(k+1)-x(k)| = {2:.8}, |f(k+1)-f(k)| = {3:.8}, |df/dx| = {4:.8}",k,fk,sk.norm(),f_change,gnorm);

            let new_coords:Array2<f64> = constants::BOHR_TO_ANGS * &x_old.view().into_shape((n_atoms,3)).unwrap();
            let xyz_out:XYZ_Output =
                XYZ_Output::new(
                    atom_names.clone(),
                    new_coords.clone().into_shape([n_atoms,3]).unwrap());
            write_xyz_custom(&xyz_out);

            let opt_energy:Opt_Energy_Output = Opt_Energy_Output::new(k,fk);
            write_opt_energy(&opt_energy);
        }
        return (x_old,grad_fk);
    }
}