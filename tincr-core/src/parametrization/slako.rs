use crate::parametrization::skf_handler::SkfHandler;
use crate::parametrization::slako_spline::SlaterKosterTable;
use crate::Element;
use hashbrown::HashMap;

fn get_nan_vec() -> Vec<f64> {
    vec![f64::NAN]
}

fn get_nan_value() -> f64 {
    f64::NAN
}

fn init_none() -> Option<(Vec<f64>, Vec<f64>, usize)> {
    None
}

fn get_inf_value() -> f64 {
    f64::INFINITY
}

fn init_hashmap() -> HashMap<u8, (Vec<f64>, Vec<f64>, usize)> {
    HashMap::new()
}

pub enum ParamFiles<'a> {
    SKF(&'a str),
    OWN,
}

/// Type that holds the mapping between element pairs and their [SlaterKosterTable].
/// This is basically a struct that allows to get the [SlaterKosterTable] without a strict
/// order of the [Element] tuple.
#[derive(Clone)]
pub struct SlaterKoster {
    pub map: HashMap<(Element, Element), SlaterKosterTable>,
}

impl SlaterKoster {
    /// Create a new [SlaterKoster] type, that maps a tuple of [Element] s to a [SlaterKosterTable].
    pub fn new() -> Self {
        SlaterKoster {
            map: HashMap::new(),
        }
    }

    /// Add a new [SlaterKosterTable] from a tuple of two [Element]s. THe
    pub fn add(&mut self, kind1: Element, kind2: Element) {
        self.map
            .insert((kind1, kind2), SlaterKosterTable::new(kind1, kind2));
    }

    pub fn add_from_handler(
        &mut self,
        kind1: Element,
        kind2: Element,
        handler: SkfHandler,
        optional_table: Option<SlaterKosterTable>,
        order: &str,
    ) {
        self.map.insert(
            (kind1, kind2),
            SlaterKosterTable::from((&handler, optional_table, order)),
        );
    }

    pub fn get(&self, kind1: Element, kind2: Element) -> &SlaterKosterTable {
        self.map
            .get(&(kind1, kind2))
            .unwrap_or_else(|| self.map.get(&(kind2, kind1)).unwrap())
    }
}
