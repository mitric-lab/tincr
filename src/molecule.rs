use ndarray::prelude::*;

struct Molecule {
    atomic_numbers: Vec<u8>,
    positions: Array2<f64>,
    charge: i8,
    multiplicity: u8,
}

impl Molecule {
    fn iter_atomlist(
        &self,
    ) -> std::iter::Zip<
        std::slice::Iter<'_, u8>,
        ndarray::iter::AxisIter<'_, f64, ndarray::Dim<[usize; 1]>>,
    > {
        self.atomic_numbers.iter().zip(self.positions.outer_iter())
    }
}
