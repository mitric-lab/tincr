use std::HashMap;
mod parameters;
use ndarray::{ArrayView1, Array};

/// has to be fixed
fn count_orbitals(atomlist:f64, valorbs: f64) -> u64 {
    let norb: u64= 0;

    return norb;
}


pub fn h0_and_s_ab(
    atomlist_a: Vec<Atom>,
    atomlist_b: Vec<Atom>,
    skt: HashMap<(u8, u8), SlaterKosterTable>,
    orbital_energies: HashMap<u8, HashMap<(u8, u8), f64>>,
    m_proximity: ArrayView1,
) -> (Array<f64, Ix2>, Array<f64, Ix2>) {
    ///
    /// compute Hamiltonian and overlap matrix elements between two sets of atoms. If the sets
    /// A and B contain exactly the same structure AisB should be set to True to ensure that
    /// the diagonal elements of the Hamiltonian are replaced by the correct on-site energies.
    ///
    /// Parameters:
    /// ===========
    /// dim_a: number of orbitals of all the atoms in set A
    /// dim_b:  ''                                 in set B
    /// atomlist_a, atomlist_b: list of (Zi,(xi,yi,zi)) for each atom
    ///
    ///
    let mut h0: Array<f64, Ix2> = Array::zeros((dim_a, dim_b));
    let mut s: Array<f64, Ix2> = Array::zeros((dim_a, dim_b));
    // iterate over atoms
    let mu = 0;
    for (i, atom_a) in atomlist_a.iter().enumerate() {
        // iterate over orbitals on center i
        for _ in atom_a.valorbs.iter() {
            // iterate over atoms
            let nu = 0;
            for (j, atom_b) in atomlist_b.iter().enumerate() {
                // iterate over orbitals on center j
                for _ in atom_b.valorbs.iter() {
                    if mu == nu && a_is_b == true {
                        assert_eq!(atom_a.number, atom_b.number);
                        // use the true single particle orbitals energies
                        h0[[mu, nu]] = orbital_energies[Zi][(ni,li)];
                        // orbitals are normalized to 1
                        s[[mu, nu]] = 1.0;
                    } else {
                        // initialize matrix elements of S and H0
                        let mut s_mu_nu: f64 = 0.0;
                        let mut h0_mu_nu: f64 = 0.0;
                        if atom_a.number <= atom_b.number {
                            // the first atom given to getHamiltonian() or getOverlap()
                            // has to be always the one with lower atomic number
                            if i == j && a_is_b == true {
                                assert!(mu != nu);
                                s_mu_nu = 0.0;
                                h0_mu_nu = 0.0;
                            } else {
                                // let s = SKT[(Zi,Zj)].getOverlap(li,mi,posi, lj,mj,posj);
                                // let h0 = SKT[(Zi,Zj)].getHamiltonian0(li,mi,posi, lj,mj,posj);
                                s_mu_nu = 0.0;
                                h0_mu_nu = 0.0;
                            }
                        } else {
                            // swap atoms if Zj > Zi
                            // let s  = SKT[(Zj,Zi)].getOverlap(lj,mj,posj, li,mi,posi);
                            // let h0 = SKT[(Zj,Zi)].getHamiltonian0(lj,mj,posj, li,mi,posi);
                            s_mu_nu = 0.0;
                            h0_mu_nu = 0.0;
                        }
                        h0[[mu, nu]] = h0_mu_nu;
                        s[[mu, nu]] = s_mu_nu;
                    }
                    let nu = nu + 1;
                }
            }
            let mu = mu + 1;
        }
    }
    return (s, h0);
}