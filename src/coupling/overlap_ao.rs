use crate::initialization::{Atom,System};
use ndarray::prelude::*;
use crate::param::slako_transformations::{slako_transformation, directional_cosines};
use ndarray_linalg::Determinant;
use ndarray_stats::QuantileExt;
use phf::Slice;
use crate::fmo::basis::LocallyExcited;

pub fn overlap_between_timesteps(le_i: &LocallyExcited, le_j: &LocallyExcited)->Array2<f64>{
    /// compute overlap matrix elements between two sets of atoms using
    /// Slater-Koster rules
    ///
    let atoms_old_j = le_j.monomer.properties.old_atoms().unwrap();
    let mut s:Array2<f64> = Array2::zeros([le_i.monomer.n_orbs, le_j.monomer.n_orbs]);
    let mut mu:usize = 0;
    // iterate over the atoms of the system
    for (i, atom_i) in le_i.atoms.iter().enumerate(){
        // iterate over the orbitals on atom I
        for orbi in atom_i.valorbs.iter(){
            // iterate over the atoms of the old geometry
            let mut nu:usize = 0;
            for (j, atom_j) in atoms_old_j.iter().enumerate(){
                // iterate over the orbitals on atom J
                for orbj in atom_j.valorbs.iter(){
                    if atom_i <= atom_j {
                        let (r, x, y, z): (f64, f64, f64, f64) =
                            directional_cosines(&atom_i.xyz, &atom_j.xyz);
                        s[[mu, nu]] = slako_transformation(
                            r,
                            x,
                            y,
                            z,
                            &le_i.monomer.slako.get(atom_i.kind, atom_j.kind).s_spline,
                            orbi.l,
                            orbi.m,
                            orbj.l,
                            orbj.m,
                        );
                    }
                    else{
                        let (r, x, y, z): (f64, f64, f64, f64) =
                            directional_cosines(&atom_j.xyz, &atom_i.xyz);
                        s[[mu, nu]] = slako_transformation(
                            r,
                            x,
                            y,
                            z,
                            &le_i.monomer.slako.get(atom_j.kind, atom_i.kind).s_spline,
                            orbj.l,
                            orbj.m,
                            orbi.l,
                            orbi.m,
                        );
                    }
                    nu += 1;
                }
            }
            nu += 1;
        }
    }
    return s;
}

pub fn ci_overlap(le_i: &LocallyExcited, le_j: &LocallyExcited, n_states:usize)->Array2<f64>{
    /// Compute CI overlap between TD-DFT 'wavefunctions'
    /// Excitations i->a with coefficients |C_ia| < threshold will be neglected
    /// n_states: Includes the ground state
    let threshold:f64 = 0.01;

    // calculate the overlap between the new and old geometry
    let s_ao:Array2<f64> = overlap_between_timesteps(le_i, le_j);

    let old_orbs_j = le_j.monomer.properties.old_orbs().unwrap();

    let orbs_i:ArrayView2<f64> = le_i.monomer.properties.orbs().unwrap();
    // calculate the overlap between the molecular orbitals
    let s_mo:Array2<f64> = orbs_i.t().dot(&s_ao.dot(&old_orbs_j));

    // get occupied and virtual orbitals
    let occ_indices_i = le_i.monomer.properties.occ_indices().unwrap();
    let virt_indices_i = le_i.monomer.properties.virt_indices().unwrap();
    let n_occ_i:usize = occ_indices_i.len();
    let n_virt_i: usize = virt_indices_i.len();

    let occ_indices_j = le_j.monomer.properties.occ_indices().unwrap();
    let virt_indices_j = le_j.monomer.properties.virt_indices().unwrap();
    let n_occ_j:usize = occ_indices_j.len();
    let n_virt_j: usize = virt_indices_j.len();

    // slice s_mo to get the occupied part and calculate the determinant
    let s_ij:ArrayView2<f64> = s_mo.slice(s![..n_occ_i,..n_occ_j]);
    let det_ij = s_ij.det().unwrap();

    let n_roots: usize = 10;
    // scalar coupling array
    let mut s_ci:Array2<f64> = Array2::zeros((n_states,n_states));
    // get ci coefficients from properties
    // let n_roots:usize = self.config.excited.nstates;
    let ci_coeff:ArrayView2<f64> = le_i.monomer.properties.ci_coefficients().unwrap();
    let ci_coeff:ArrayView3<f64> = ci_coeff.t().into_shape([n_roots,n_occ_i,n_virt_i]).unwrap();
    let old_ci_coeff:ArrayView3<f64> = le_j.monomer.properties.old_ci_coeffs().unwrap().into_shape([n_roots,n_occ_j,n_virt_j]).unwrap();

    // overlap between ground states <Psi0|Psi0'>
    s_ci[[0,0]] = det_ij;

    // calculate the overlap between the excited states
    // iterate over the old CI coefficients
    for i in occ_indices_j.iter() {
        for a in virt_indices_j.iter() {
            // slice old CI coefficients at the indicies i and a
            let coeffs_i = old_ci_coeff.slice(s![..,*i,*a]);
            let max_coeff_i = coeffs_i.map(|val| val.abs()).max().unwrap().to_owned();

            // slice new CI coefficients at the indicies i and a
            let coeffs_new = ci_coeff.slice(s![..,*i,*a]);
            let max_coeff_new = coeffs_new.map(|val| val.abs()).max().unwrap().to_owned();

            // if the value of the coefficient is smaller than the threshold,
            // exclude the excited state
            if max_coeff_new > threshold{
                let mut s_ia: Array2<f64> = s_ij.to_owned();
                // overlap <Psi0|PsiJ'>
                s_ia.slice_mut(s![..,*i]).assign(&s_mo.slice(s![..n_occ_j,*a]));
                let det_ia:f64 = s_ia.det().unwrap();

                // overlaps between ground state <Psi0|PsiJ'> and excited states
                for state_j in (1..n_states){
                    let c0:f64 = coeffs_new[state_j-1];
                    s_ci[[0,state_j]] += c0 * 2.0_f64.sqrt() * (det_ia * det_ij);
                }

            }
            // if the value of the coefficient is smaller than the threshold,
            // exclude the excited state
            if max_coeff_i > threshold{
                let mut s_aj:Array2<f64> = s_ij.to_owned();
                // occupied orbitals in the configuration state function |Psi_ia>
                // oveerlap <1,...,a,...|1,...,j,...>
                s_aj.slice_mut(s![*i,..]).assign(&s_mo.slice(s![*a,..n_occ_i]));
                let det_aj:f64 = s_aj.det().unwrap();

                // overlaps between ground state <PsiI|Psi0'> and excited states
                for state_i in (1..n_states){
                    let c0:f64 = coeffs_i[state_i-1];
                    s_ci[[state_i,0]] += c0 * 2.0_f64.sqrt() * (det_aj * det_ij);
                }

                // iterate over the new CI coefficients
                for j in occ_indices_i.iter(){
                    for b in virt_indices_i.iter(){
                        // slice the new CI coefficients at the indicies j and b
                        let coeffs_j = ci_coeff.slice(s![..,*j,*b]);
                        let max_coeff_j = coeffs_j.map(|val| val.abs()).max().unwrap().to_owned();
                        // if the value of the coefficient is smaller than the threshold,
                        // exclude the excited state
                        if max_coeff_j > threshold{
                            let mut s_ab:Array2<f64> = s_ij.to_owned();
                            // select part of overlap matrix for orbitals
                            // in |Psi_ia> and |Psi_jb>
                            // <1,...,a,...|1,...,b,...>
                            s_ab.slice_mut(s![*i,..]).assign(&s_mo.slice(s![*a,..n_occ_i]));
                            s_ab.slice_mut(s![..,*j]).assign(&s_mo.slice(s![..n_occ_j,*b]));
                            s_ab[[*i,*j]] = s_mo[[*a,*b]];
                            let det_ab:f64 = s_ab.det().unwrap();

                            let mut s_ib:Array2<f64> = s_ij.to_owned();
                            // <1,...,i,...|1,...,b,...>
                            s_ib.slice_mut(s![..,*j]).assign(&s_mo.slice(s![..n_occ_j,*b]));
                            let det_ib:f64 = s_ib.det().unwrap();

                            // loop over excited states
                            for state_i in (1..n_states){
                                for state_j in (1..n_states){
                                    let cc:f64 = coeffs_i[state_i-1] * coeffs_j[state_j-1];
                                    // see eqn. (9.39) in A. Humeniuk, PhD thesis (2018)
                                    s_ci[[state_i,state_j]] += cc * (det_ab * det_ij + det_aj * det_ib);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return s_ci;
}


pub fn get_sign_of_array(arr:ArrayView1<f64>)->Array1<f64>{
    let mut sign:Array1<f64> = Array1::zeros(arr.len());
    arr.iter().enumerate().for_each(|(idx,val)|
        if val.is_sign_positive(){
            sign[idx] = 1.0;
        }else{
            sign[idx] = -1.0;
        }
    );
    return sign;
}