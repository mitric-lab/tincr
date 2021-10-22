use crate::fmo::*;
use crate::fmo::helpers::{MolIndices, MolIncrements, MolecularSlice};
use crate::data::{Storage, Parametrization, SpatialOrbitals, OrbType};
use core::::{Atom, AtomVec, AtomSlice};
use rayon::prelude::*;
use ndarray::{indices, Data};
use itertools::izip;
use hashbrown::HashMap;
use crate::param::reppot::RepulsivePotential;
use crate::param::slako::SlaterKoster;
use crate::io::Configuration;

pub struct SuperSystemSetup {
    /// Atomic indices of each monomer.
    pub indices: Option<Vec<Vec<usize>>>,
    pub slices: Option<Vec<MolecularSlice>>,
    pub sorted_atoms: Option<AtomVec>,
    pub n_atoms: Option<Vec<usize>>,
    pub n_orbs: Option<Vec<usize>>,
    pub n_occs: Option<Vec<usize>>,
    pub n_virts: Option<Vec<usize>>,
    pub n_elecs: Option<Vec<usize>>,
    pub pairs: Option<Vec<(usize, usize)>>,
    pub esd_pairs: Option<Vec<(usize, usize)>>,
    pub pair_indices: Option<HashMap<(usize, usize),usize>>,
    pub pair_types: Option<HashMap<(usize, usize), PairType>>,
    pub count: usize,
    pub update_interval: usize,
}

#[derive(Copy, Clone)]
pub struct DataSetup<'a> {
    pub atoms: AtomSlice<'a>,
    pub parameters: Parametrization<'a>,
    pub vrep: &'a RepulsivePotential,
    pub slako: &'a SlaterKoster,
}


impl SuperSystemSetup {

    pub fn new(atoms: AtomSlice) -> Self {
        Self::default()
    }

    fn compute_indices(atoms: AtomSlice) -> Vec<Vec<usize>> {
        // Build a connectivity graph to distinguish the individual monomers from each other
        let graph: Graph = build_graph(atoms);
        // The molecular system is fragmented into individual non-connected molecules.
        fragmentation(&graph)
    }

    fn set_slices(&mut self) {
        // The [Monomer]s are initialized
        let mut mol_indices: MolIndices = MolIndices::new();

        let mut slices: Vec<MolecularSlice> = Vec::with_capacity(self.indices.len());

        for (atom, orbs, occs, virts) in izip!(self.atoms.len().unwrap(), self.n_orbs().unwrap(), self.n_occs.unwrap(), self.n_virts.unwrap()) {

            let increments: MolIncrements = MolIncrements { atom, orbs, occs, virts };

            // Create the slices for the atoms, grads and orbitals
            slices.push(MolecularSlice::new(mol_indices, increments));

            // Increment the indices..
            mol_indices.add(increments);

        }
        self.slices = Some(slices);
    }

    fn set_pairs(&mut self, m_atoms: &[AtomVec]) {
        // Initialize the close pairs and the ones that are treated within the ES-dimer approx
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut esd_pairs: Vec<(usize, usize)> = Vec::new();

        // Create a HashMap that maps the Monomers to the type of Pair. To identify if a pair of
        // monomers are considered a real pair or should be treated with the ESD approx.
        let mut pair_iter: usize = 0;
        let mut pair_indices: HashMap<(usize, usize),usize> = HashMap::new();
        let mut pair_types: HashMap<(usize, usize), PairType> = HashMap::new();

        // The construction of the [Pair]s requires that the [Atom]s are ordered after
        // each monomer
        // TODO: Read the vdw scaling parameter from the input file instead of setting hard to 2.0
        // TODO: PARALLEL: this loop should be parallelized
        for (i, atoms_i) in m_atoms.iter().enumerate() {
            for (j, atoms_j) in m_atoms[(i + 1)..].iter().enumerate() {
                match get_pair_type(
                    &atoms_i,
                    &atoms_j,
                    2.0,
                ) {
                    PairType::Pair => {
                        pairs.push((i, j));
                        pair_types.insert((i, j), PairType::Pair);
                        pair_indices.insert((i, j),pair_iter);
                    },
                    PairType::ESD => {
                        esd_pairs.push((i, j));
                        pair_types.insert((i, j), PairType::ESD);
                        pair_indices.insert((i, j), pair_iter);
                    },
                    _ => {}
                }
                pair_iter += 1;
            }
        }
        
        self.pairs = Some(pairs);
        self.esd_pairs = Some(esd_pairs);
        self.pair_indices = Some(pair_indices);
        self.pair_types = Some(pair_types);
    }

    fn update(&mut self, atoms: AtomSlice) {
        // Update of the indices that correspond to a monomer.
        self.indices = Some(SuperSystemSetup::compute_indices(atoms));

        // Clone the atoms that belong to this monomer, they will be stored in the sorted list
        let m_atoms: Vec<AtomVec> = self.indices.iter().map(
            |indices| indices.iter().map(|i| atoms[i].clone()).collect())
            .collect();

        // Number of atoms per monomer.
        self.atoms.len() = Some(m_atoms.iter().map(|atoms| atoms.len()).collect());

        // Count the number of orbitals
        self.n_orbs() = Some(m_atoms.iter().map(|atoms| atoms.n_orbs().iter().sum()).collect());

        // Count the number of electrons.
        self.n_elecs = Some(m_atoms.iter().map(|atoms| atoms.n_elec.iter().sum()).collect());

        // Number of occupied orbitals.
        self.n_occs = Some(self.n_elecs.as_ref().unwrap().iter().map(|n| n / 2).collect());

        // Number of virtual orbitals.
        self.n_virts = Some(self.n_orbs().as_ref().unwrap().iter().zip(self.n_occs.iter()).map(|(orbs, occ)| orbs - occ).collect());

        self.set_slices();

        self.set_pairs(&m_atoms);

        // Get all [Atom]s of the SuperSystem in an order that corresponds to the order of
        // the monomers
        self.sorted_atoms = Some(m_atoms.into_iter().flatten().collect());
    }

    fn create_monomers<'a>(&self, data: DataSetup<'a>) -> Vec<Monomer<'a>> {

        let mut monomers: Vec<Monomer> = Vec::with_capacity(self.indices.len());

        for (indices, slice, orbs, elecs) in izip!(self.indices.unwrap(), self.slices.unwrap(), self.n_orbs().unwrap(), self.n_elecs.unwrap()) {
            // Slice of the atoms corresponding to this monomer.
            let atoms = data.atoms.cl[indices[0]..indices[indices.len()-1]];
            // A slice of the parametrized matrices is taken.
            let params = data.params.slice(slice);
            // Construct the orbitals.
            let orbitals = SpatialOrbitals::new(orbs, elecs, OrbType::Restricted);
            // Data storage for this monomer is created.
            let storage = Storage::new_with_orbitals(params, orbitals);

            let index: usize = monomers.len();

            monomers.push(Monomer {
                atoms,
                index,
                slice: slice.clone(),
                data: storage,
                vrep: data.vrep,
                slako: data.slako,
            });
        }
        monomers
    }

    fn create_pairs<'a, 'b>(&self, monomers: &'b[Monomer<'a>], data: DataSetup<'a>) -> Vec<Pair<'a>> {
        let pair_indices: &[(usize, usize)] = self.pairs.as_ref().unwrap();
        let mut pairs: Vec<Pair> = Vec::with_capacity(pair_indices.len());
        for ij in pair_indices.iter() {
            let atoms_i = data.atoms[monomers[ij.0].slice.atom_as_range()];
            let atoms_j = data.atoms[monomers[ij.1].slice.atom_as_range()];
            pairs.push(Pair::new(&monomers[ij.0], &monomers[ij.1], (atoms_i, atoms_j),data.parameters));
        }
        pairs
    }

    fn create_esd_pairs<'a, 'b>(&self, monomers: &'b[Monomer<'a>], data: DataSetup<'a>) -> Vec<ESDPair<'a>> {
        let pair_indices: &[(usize, usize)] = self.esd_pairs.as_ref().unwrap();
        let mut esd_pairs: Vec<ESDPair> = Vec::with_capacity(pair_indices.len());
        for ij in pair_indices.iter() {
            esd_pairs.push(ESDPair::new(&monomers[ij.0], &monomers[ij.1], data.parameters));
        }
        esd_pairs
    }


    pub fn create<'a>(&mut self, config: &'a Configuration, data: DataSetup<'a>) -> SuperSystem<'a> {

        if self.count % self.update_interval == 0 {
            self.update(data.atoms);
            self.count += 1;
        }

        let monomers: Vec<Monomer> = self.create_monomers(data);
        let pairs: Vec<Pair> = self.create_pairs(&monomers, data);
        let esd_pairs: Vec<ESDPair> = self.create_esd_pairs(&monomers, data);

        // Number of orbitals.
        let n_orbs: usize = self.n_orbs().as_ref().unwrap().iter().sum();
        // Number of electrons.
        let n_elec: usize = self.n_elecs.as_ref().unwrap().iter().sum();

        let orbitals = SpatialOrbitals::new(n_orbs, n_elec, OrbType::Restricted);
        let storage = Storage::new_with_orbitals(data.parameters, orbitals);

        SuperSystem{
            config,
            atoms: data.atoms,
            monomers,
            pairs,
            esd_pairs,
            data: storage,
            vrep: data.vrep,
            slako: data.slako,
        }
    }



}

impl Default for SuperSystemSetup {
    fn default() -> Self {
        Self {
            indices: None,
            slices: None,
            sorted_atoms: None,
            n_atoms: None,
            n_orbs: None,
            n_occs: None,
            n_virts: None,
            n_elecs: None,
            pairs: None,
            esd_pairs: None,
            pair_indices: None,
            pair_types: None,
            count: 0,
            update_interval: 1,
        }
    }
}







