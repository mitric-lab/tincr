use ndarray::prelude::*;
use crate::initialization::Atom;

pub fn inter_fragment_exchange_integral(
    atoms_i: &[Atom],
    atoms_j: &[Atom],
    orbs_i: ArrayView2<f64>,
    orbs_j: ArrayView2<f64>,
    s_ij: ArrayView2<f64>,
    occ_indices_i: &[usize],
    virt_indices_j: &[usize],
    g0_pair:ArrayView2<f64>,
)->Array1<f64>{
    // set the number of active orbitals
    let n_occ_m: usize = occ_indices_i.len();
    let n_virt_m: usize = virt_indices_j.len();
    // build mutable array for storing the transition charges
    let mut exchange:Array2<f64> = Array2::zeros([n_occ_m,n_virt_m]);

    let mut mu: usize;
    let mut lambda:usize;
    let mut nu:usize;
    let mut sigma:usize;

    for occ in 0..(n_occ_m){
        let occ_i:usize = occ_indices_i[occ];
        for virt in 0..(n_virt_m){
            let virt_j:usize = virt_indices_j[virt];
            mu = 0;
            for atom_ii in atoms_i.iter(){
                for _ in 0..(atom_ii.n_orbs) {
                    lambda = 0;

                    for atom_ij in atoms_i.iter(){
                        for _ in 0..(atom_ij.n_orbs) {
                            nu = 0;

                            for atom_ji in atoms_j.iter(){
                                for _ in 0..(atom_ji.n_orbs){
                                    sigma = 0;

                                    for atom_jj in atoms_j.iter(){
                                        for _ in 0..(atom_jj.n_orbs){
                                            exchange[[occ,virt]] += orbs_i[[mu,occ_i]] * orbs_i[[lambda,occ_i]] * orbs_j[[nu,virt_j]] * orbs_j[[sigma,virt_j]]
                                                * s_ij[[mu,nu]] * s_ij[[lambda,sigma]] * (g0_pair[[mu,lambda]] + g0_pair[[mu,sigma]] + g0_pair[[nu,lambda]] + g0_pair[[nu,sigma]]);

                                            sigma += 1;
                                        }
                                    }
                                    nu +=1;
                                }
                            }
                            lambda += 1;
                        }
                    }
                    mu += 1;
                }
            }
        }
    }
    let exchange:Array1<f64> = exchange.into_shape(n_occ_m*n_virt_m).unwrap();
    return exchange;
}

pub fn inter_fragment_trans_charges(
    atoms_i: &[Atom],
    atoms_j: &[Atom],
    orbs_i: ArrayView2<f64>,
    orbs_j: ArrayView2<f64>,
    s_ij: ArrayView2<f64>,
    occ_indices_i: &[usize],
    virt_indices_j: &[usize],
)->Array2<f64>{
    // set n_atoms
    let n_atoms_i: usize = atoms_i.len();
    let n_atoms_j: usize = atoms_j.len();
    // set the number of active orbitals
    let n_occ_m: usize = occ_indices_i.len();
    let n_virt_m: usize = virt_indices_j.len();

    let virt_index_start:usize = virt_indices_j[0];
    let virt_index_end:usize = virt_indices_j[n_virt_m-1];
    let occ_index_start:usize = occ_indices_i[0];
    let occ_index_end:usize = occ_indices_i[n_occ_m-1];

    // calculate s_mu,nu * c_nu,a
    let s_c_j:Array2<f64> = s_ij.dot(&orbs_j.slice(s![..,virt_index_start..virt_index_end]));
    // calculate c_mu,i.T * s_mu,nu
    let c_i_s:Array2<f64> = orbs_i.slice(s![..,occ_index_start..occ_index_end]).t().dot(&s_ij);
    // define separate arrays for transition charges for atoms on I and atoms on J
    // the arrays have to be appended after the calculation
    let mut qtrans_i:Array3<f64> = Array3::zeros([n_atoms_i,n_occ_m,n_virt_m]);
    let mut qtrans_j:Array3<f64> = Array3::zeros([n_atoms_j,n_occ_m,n_virt_m]);

    let mut mu: usize = 0;
    // calculate sum_mu(on atom A of I) sum_nu(on J) S_mu,nu * orbs_i_mu,i * orbs_j_nu,a
    for (n_i, atom_i) in atoms_i.iter().enumerate() {
        for _ in 0..(atom_i.n_orbs) {
            for (i, occi) in occ_indices_i.iter().enumerate() {
                for (a, virta) in virt_indices_j.iter().enumerate() {
                    qtrans_i[[n_i,i,a]] += orbs_i[[mu, *occi]] * s_c_j[[mu, *virta]];
                }
            }
            mu += 1;
        }
    }

    let mut nu:usize = 0;
    // calculate sum_nu(on atom A of J) sum_mu(on I) S_mu,nu * orbs_i_mu,i * orbs_j_nu,a
    for (n_j, atom_j) in atoms_j.iter().enumerate() {
        for _ in 0..(atom_j.n_orbs) {
            for (i, occi) in occ_indices_i.iter().enumerate() {
                for (a, virta) in virt_indices_j.iter().enumerate() {
                    qtrans_j[[n_j,i,a]] += orbs_j[[nu, *virta]] * c_i_s[[*occi, nu]];
                }
            }
            nu += 1;
        }
    }
    qtrans_i.append(Axis(0),qtrans_j.view()).unwrap();
    let qtrans_result:Array2<f64> = qtrans_i.into_shape([n_atoms_i+n_atoms_j,n_occ_m*n_virt_m]).unwrap();

    return qtrans_result;
}