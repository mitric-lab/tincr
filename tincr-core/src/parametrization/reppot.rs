use crate::parametrization::reppot_spline::RepulsivePotentialTable;
use crate::{AtomSlice, Element};
use hashbrown::HashMap;
use itertools::Itertools;
use soa_derive::soa_zip;

/// Type that holds the mapping between element pairs and their [RepulsivePotentialTable].
/// This is basically a struct that allows to get the [RepulsivePotentialTable] without a s
/// order of the [Element] tuple.
#[derive(Clone)]
pub struct RepulsivePotential {
    pub map: HashMap<(Element, Element), RepulsivePotentialTable>,
}

impl RepulsivePotential {
    /// Create a new RepulsivePotential, to map the [Element] pairs to a [RepulsivePotentialTable]
    pub fn new() -> Self {
        RepulsivePotential {
            map: HashMap::new(),
        }
    }

    /// Add a [RepulsivePotentialTable] from a pair of two [Element]s
    pub fn add(&mut self, kind1: Element, kind2: Element) {
        self.map
            .insert((kind1, kind2), RepulsivePotentialTable::new(kind1, kind2));
    }

    /// Return the [RepulsivePotentialTable] for the tuple of two [Element]s. The order of
    /// the tuple does not play a role.
    pub fn get(&self, kind1: Element, kind2: Element) -> &RepulsivePotentialTable {
        self.map
            .get(&(kind1, kind2))
            .unwrap_or_else(|| self.map.get(&(kind2, kind1)).unwrap())
    }

    /// Compute energy due to core electrons and nuclear repulsion.
    pub fn get_repulsive_energy(&self, atoms: AtomSlice) -> f64 {
        soa_zip!(atoms, [xyz, kind])
            .combinations_with_replacement(2)
            .map(|ij| {
                self.get(*ij[0].1, *ij[1].1)
                    .spline_eval((ij[0].0 - ij[1].0).norm())
            })
            .sum()
    }
}
