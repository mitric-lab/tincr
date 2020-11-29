use crate::constants::*;
use crate::defaults;
use crate::diis::*;
use crate::fermi_occupation;
use crate::h0_and_s::h0_and_s_ab;
use crate::molecule::*;
use crate::mulliken::*;
use crate::broyden::*;
use approx::AbsDiffEq;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use std::cmp::max;
use std::iter::FromIterator;


// INCOMPLETE
pub fn run_scc(
    molecule: &Molecule,
    max_iter: Option<usize>,
    scf_conv: Option<f64>,
    temperature: Option<f64>,
) -> f64 {
    let max_iter: usize = max_iter.unwrap_or(defaults::MAX_ITER);
    let scf_conv: f64 = scf_conv.unwrap_or(defaults::SCF_CONV);
    let temperature: f64 = temperature.unwrap_or(defaults::TEMPERATURE);

    // construct reference density matrix
    let p0: Array2<f64> = density_matrix_ref(&molecule);
    let mut p_last: Array2<f64> = p0.clone();
    let mut p: Array2<f64> = Array2::zeros(p0.raw_dim());
    // charge guess
    let mut dq: Array1<f64> = Array1::zeros([molecule.n_atoms]);
    let mut q: Array1<f64> = Array::from_iter(molecule.q0.iter().cloned());
    let mut energy_old: f64 = 0.0;
    let mut scf_energy: f64 = 0.0;
    let mut diis_count: usize = 0;
    let (s, h0): (Array2<f64>, Array2<f64>) = h0_and_s_ab(&molecule, &molecule);
    let (gm, gm_a0): (Array2<f64>, Array2<f64>) = get_gamma_matrix(&molecule, Some(0.0));

    let mut fock_error: Vec<Array1<f64>> = Vec::new();
    let mut fock_list: Vec<Array2<f64>> = Vec::new();
    let mut density_mixer: Pulay80 = Pulay80::new();
    let mut fock_mixer: Pulay82 = Pulay82::new();

    let mut broyden_mixer: BroydenMixer = BroydenMixer::new(molecule.n_atoms);

    let mut mixing_flag: bool = false;

    //  compute A = S^(-1/2)
    // 1. diagonalize S
    let (w, v): (Array1<f64>, Array2<f64>) = s.eigh(UPLO::Upper).unwrap();
    // 2. compute inverse square root of the eigenvalues
    let w12: Array2<f64> = Array2::from_diag(&w.map(|x| x.pow(-0.5)));
    // 3. and transform back
    let a: Array2<f64> = v.dot(&w12.dot(&v.t()));

    // add nuclear energy to the total scf energy
    let rep_energy: f64 = get_repulsive_energy(&molecule);
    'scf_loop: for i in 0..max_iter {
        let h1: Array2<f64> = construct_h1(&molecule, gm.view(), dq.view());
        let h_coul: Array2<f64> = h1 * s.view();
        let mut h: Array2<f64> = h_coul + h0.view();
        // convert generalized eigenvalue problem H.C = S.C.e into eigenvalue problem H'.C' = C'.e
        // by Loewdin orthogonalization, H' = X^T.H.X, where X = S^(-1/2)
        let x: Array2<f64> = s.ssqrt(UPLO::Upper).unwrap().inv().unwrap();
        // H' = X^t.H.X
        //println!("H {}", h);
        // if i > 0 {
        //     h = fock_mixer.next(h);
        // }
        let hp: Array2<f64> = x.t().dot(&h).dot(&x);
        let (orbe, cp): (Array1<f64>, Array2<f64>) = hp.eigh(UPLO::Upper).unwrap();
        // C = X.C'
        //println!("ORBE {}", orbe);

        let orbs: Array2<f64> = x.dot(&cp);
        // construct density matrix
        let tmp: (f64, Vec<f64>) = fermi_occupation::fermi_occupation(
            orbe.view(),
            molecule.q0.iter().sum::<f64>() as usize - molecule.charge as usize,
            molecule.nr_unpaired_electrons,
            temperature,
        );
        let mu: f64 = tmp.0;
        let f: Vec<f64> = tmp.1;
        // calculate the density matrix
        p = density_matrix(orbs.view(), &f[..]);
        //println!("P orig {}", p);
        // if mixing_flag {
        //     p = density_mixer.next(p);
        // } else {
        //     // this is only temporary for testing
        //     let mut next_p: Array2<f64> = p.map(|x| 0.33 * *x) + p_last.map(|x| 0.67 * *x);
        //     next_p *= (&p * &s).sum() / (&next_p * &s).sum();
        //     p = next_p;
        // }
        //println!("P diis {}", p);

        //println!("P0 {}", p0);
        // update partial charges using Mulliken analysis
        let (new_q, new_dq): (Array1<f64>, Array1<f64>) = mulliken(
            p.view(),
            p0.view(),
            s.view(),
            &molecule.orbs_per_atom,
            molecule.n_atoms,
        );
        println!("Q BEFORE {}", q);
        q = broyden_mixer.next(new_q, new_dq.clone());
        println!("Q AFTER {}", q);
        dq = new_dq;

        //println!("Q: {}, dq {}", q, dq);

        // compute electronic energy
        scf_energy = get_electronic_energy(p.view(), h0.view(), dq.view(), gm.view());

        // does the density matrix commute with the KS Hamiltonian?
        // diis_error = H * D * S - S * D * H
        if i > 0 {
            let mut diis_e: Array2<f64> = h.dot(&p.dot(&s)) - &s.dot(&p).dot(&h);
            // transform error vector to orthogonal basis
            let mut diis_e: Array1<f64> = Array1::from_iter(a.t().dot(&diis_e.dot(&a)).iter().cloned());
            //println!("DIIS E {}", diis_e);
            fock_mixer.add_error_vector(diis_e);
        }

        if ((scf_energy - energy_old).abs() < scf_conv)  {
            break 'scf_loop;
        }

        if (&p_last - &p).norm()/&p.norm() < 1.0e-4 {
            mixing_flag = true;
        }

        energy_old = scf_energy;
        println!("Iteration {} => SCF-Energy = {:.8} hartree", i, scf_energy + rep_energy);
        assert_ne!(i + 1, 50, "SCF not converged");
        p_last = p;
    }
    println!("SCF Converged!");
    return scf_energy+rep_energy;
}

/// Compute energy due to core electrons and nuclear repulsion
fn get_repulsive_energy(molecule: &Molecule) -> f64 {
    let mut e_nuc: f64 = 0.0;
    for (i, (z_i, posi)) in molecule.atomic_numbers[1..molecule.n_atoms]
        .iter()
        .zip(
            molecule
                .positions
                .slice(s![1..molecule.n_atoms, ..])
                .outer_iter(),
        )
        .enumerate()
    {
        for (z_j, posj) in molecule.atomic_numbers[0..i + 1]
            .iter()
            .zip(molecule.positions.slice(s![0..i + 1, ..]).outer_iter())
        {
            let z_1: u8;
            let z_2: u8;
            if z_i > z_j {
                z_1 = *z_j;
                z_2 = *z_i;
            } else {
                z_1 = *z_i;
                z_2 = *z_j;
            }
            let r: f64 = (&posi - &posj).norm();
            // nucleus-nucleus and core-electron repulsion
            e_nuc += &molecule.v_rep[&(z_1, z_2)].spline_eval(r);
        }
    }
    return e_nuc;
}

/// the repulsive potential, the dispersion correction and only depend on the nuclear
/// geometry and do not change during the SCF cycle
fn get_nuclear_energy() {}

/// Compute electronic energies
fn get_electronic_energy(
    p: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
) -> f64 {
    //println!("P {}", p);
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));
    // electronic energy as sum of band structure energy and Coulomb energy
    //println!("E BS {} E COUL {} dQ {}", e_band_structure, e_coulomb, dq);
    let e_elec: f64 = e_band_structure + e_coulomb;
    // long-range Hartree-Fock exchange
    // if ....Iteration {} =>
    //println!("               E_bs = {:.7}  E_coulomb = {:.7}", e_band_structure, e_coulomb);
    return e_elec;
}

/// Construct the density matrix
/// P_mn = sum_a f_a C_ma* C_na
fn density_matrix(orbs: ArrayView2<f64>, f: &[f64]) -> Array2<f64> {
    let occ_indx: Vec<usize> = f.iter().positions(|&x| x > 0.0).collect();
    let occ_orbs: Array2<f64> = orbs.select(Axis(1), &occ_indx);
    let f_occ: Vec<f64> = f.iter().filter(|&&x| x > 0.0).cloned().collect();
    // THIS IS NOT AN EFFICIENT WAY TO BUILD THE LEFT HAND SIDE
    let mut f_occ_mat: Vec<f64> = Vec::new();
    for i in 0..occ_orbs.nrows() {
        for val in f_occ.iter() {
            f_occ_mat.push(*val);
        }
    }
    let f_occ_mat: Array2<f64> = Array2::from_shape_vec(occ_orbs.raw_dim(), f_occ_mat).unwrap();
    let p: Array2<f64> = (f_occ_mat * &occ_orbs).dot(&occ_orbs.t());
    return p;
}

/// Construct reference density matrix
/// all atoms should be neutral
fn density_matrix_ref(mol: &Molecule) -> Array2<f64> {
    let mut p0: Array2<f64> = Array2::zeros((mol.n_orbs, mol.n_orbs));
    // iterate over orbitals on center i
    let mut idx: usize = 0;
    for (_i, (zi, _posi)) in mol.iter_atomlist().enumerate() {
        // how many electrons are put into the nl-shell
        for (iv, (_ni, _li, _mi)) in mol.valorbs[zi].iter().enumerate() {
            p0[[idx, idx]] = mol.valorbs_occupation[zi][iv] as f64;
            idx += 1;
        }
    }
    return p0;
}

fn construct_h1(mol: &Molecule, gamma: ArrayView2<f64>, dq: ArrayView1<f64>) -> Array2<f64> {
    let e_stat_pot: Array1<f64> = gamma.dot(&dq);
    let mut h1: Array2<f64> = Array2::zeros([mol.n_orbs, mol.n_orbs]);

    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, (z_i, pos_i)) in mol.iter_atomlist().enumerate() {
        for (n_i, l_i, m_i) in &mol.valorbs[z_i] {
            nu = 0;
            for (j, (z_j, pos_j)) in mol.iter_atomlist().enumerate() {
                for (n_j, l_j, m_j) in &mol.valorbs[z_j] {
                    h1[[mu, nu]] = 0.5 * (e_stat_pot[i] + e_stat_pot[j]);
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h1;
}

#[test]
fn h1_construction() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers, positions, charge, multiplicity);
    let (gm, _gm_a0): (Array2<f64>, Array2<f64>) = get_gamma_matrix(&mol, Some(0.0));
    let dq: Array1<f64> = array![0.4900936727759634, -0.2450466365939161, -0.2450470361820512];
    let h1: Array2<f64> = construct_h1(&mol, gm.view(), dq.view());
    let h1_ref: Array2<f64> = array![
        [
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0138472664342115,
            0.0138473229910027
        ],
        [
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0138472664342115,
            0.0138473229910027
        ],
        [
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0138472664342115,
            0.0138473229910027
        ],
        [
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0296041126328175,
            0.0138472664342115,
            0.0138473229910027
        ],
        [
            0.0138472664342115,
            0.0138472664342115,
            0.0138472664342115,
            0.0138472664342115,
            -0.0019095797643945,
            -0.0019095232076034
        ],
        [
            0.0138473229910027,
            0.0138473229910027,
            0.0138473229910027,
            0.0138473229910027,
            -0.0019095232076034,
            -0.0019094666508122
        ]
    ];
    assert!(h1.abs_diff_eq(&h1_ref, 1e-06));
}

#[test]
fn density_matrix_test() {
    let orbs: Array2<f64> = array![
        [
            8.7633819729731810e-01,
            -7.3282344651120093e-07,
            -2.5626947507277165e-01,
            5.9002324562689818e-16,
            4.4638746598694098e-05,
            6.5169204671842440e-01
        ],
        [
            1.5609816246174135e-02,
            -1.9781345423283686e-01,
            -3.5949495734350723e-01,
            -8.4834397825097219e-01,
            2.8325036729124353e-01,
            -2.9051012618890415e-01
        ],
        [
            2.5012007142380756e-02,
            -3.1696154856040176e-01,
            -5.7602794926745904e-01,
            5.2944545948125210e-01,
            4.5385929584577422e-01,
            -4.6549179289356957e-01
        ],
        [
            2.0847639428373754e-02,
            5.2838141513310655e-01,
            -4.8012912372328725e-01,
            6.2706942441176916e-16,
            -7.5667689316910480e-01,
            -3.8791258902430070e-01
        ],
        [
            1.6641902124514810e-01,
            -3.7146607333368298e-01,
            2.5136104623642469e-01,
            -4.0589133922668506e-16,
            -7.2004448997899451e-01,
            -7.6949321948982752e-01
        ],
        [
            1.6641959153799185e-01,
            3.7146559134907692e-01,
            2.5135994443678955e-01,
            -1.0279311111336347e-15,
            7.1993588930877628e-01,
            -7.6959837655952745e-01
        ]
    ];
    let f: Vec<f64> = vec![2., 2., 2., 2., 0., 0.];
    let p: Array2<f64> = density_matrix(orbs.view(), &f[..]);
    let p_ref: Array2<f64> = array![
        [
            1.6672853597938484,
            0.2116144144027609,
            0.3390751794256264,
            0.282623268095985,
            0.162846907840508,
            0.1628473832190555
        ],
        [
            0.2116144144027609,
            1.776595917657724,
            -0.357966065395007,
            0.1368169475860144,
            -0.0285685423132669,
            -0.3224914900258041
        ],
        [
            0.3390751794256264,
            -0.357966065395007,
            1.426421833339374,
            0.2192252885141318,
            -0.0457761047995666,
            -0.5167363487612707
        ],
        [
            0.282623268095985,
            0.1368169475860144,
            0.2192252885141318,
            1.020291038750183,
            -0.6269841692414225,
            0.1581194812138258
        ],
        [
            0.162846907840508,
            -0.0285685423132669,
            -0.0457761047995666,
            -0.6269841692414225,
            0.4577294196704165,
            -0.0942187608833704
        ],
        [
            0.1628473832190555,
            -0.3224914900258041,
            -0.5167363487612707,
            0.1581194812138258,
            -0.0942187608833704,
            0.4577279753425148
        ]
    ];
    assert!(p.abs_diff_eq(&p_ref, 1e-16));
}

#[test]
fn reference_density_matrix() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];
    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers, positions, charge, multiplicity);
    let p0: Array2<f64> = density_matrix_ref(&mol);
    let p0_ref: Array2<f64> = array![
        [2., 0., 0., 0., 0., 0.],
        [0., 1.3333333333333333, 0., 0., 0., 0.],
        [0., 0., 1.3333333333333333, 0., 0., 0.],
        [0., 0., 0., 1.3333333333333333, 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]
    ];
    assert!(p0.abs_diff_eq(&p0_ref, 1e-16));
}

#[test]
fn self_consistent_charge_routine() {
    let atomic_numbers: Vec<u8> = vec![8, 1, 1];
    let mut positions: Array2<f64> = array![
        [0.34215, 1.17577, 0.00000],
        [1.31215, 1.17577, 0.00000],
        [0.01882, 1.65996, 0.77583]
    ];

    // transform coordinates in au
    positions = positions / 0.529177249;
    let charge: Option<i8> = Some(0);
    let multiplicity: Option<u8> = Some(1);
    let mol: Molecule = Molecule::new(atomic_numbers, positions, charge, multiplicity);
    let energy: f64 = run_scc(&mol, None, None, None);
    println!("ENERGY: {}", energy);
    assert_eq!(1, 2);
}
