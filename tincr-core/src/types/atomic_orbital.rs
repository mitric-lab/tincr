use std::cmp::Ordering;

/// Type that specifies an atomic orbital by its three quantum numbers and holds its energy
#[derive(Copy, Clone)]
pub struct AtomicOrbital {
    pub n: i8,
    pub l: i8,
    pub m: i8,
    pub energy: f64,
}

impl From<(i8, i8, i8)> for AtomicOrbital {
    fn from(qnumber: (i8, i8, i8)) -> Self {
        Self {
            n: qnumber.0,
            l: qnumber.1,
            m: qnumber.2,
            energy: 0.0,
        }
    }
}

impl From<((i8, i8, i8), f64)> for AtomicOrbital {
    fn from(numbers_energy: ((i8, i8, i8), f64)) -> Self {
        Self {
            n: numbers_energy.0 .0,
            l: numbers_energy.0 .1,
            m: numbers_energy.0 .2,
            energy: numbers_energy.1,
        }
    }
}

impl PartialOrd for AtomicOrbital {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.ord_idx().partial_cmp(&other.ord_idx())
    }
}

impl AtomicOrbital {
    /// Compute an index, that orders the AtomicOrbital in Cartesian order.
    /// 1 s  => 100 |
    /// 2 s  => 200 | 2 px  => 210 | 2 py  => 211 | 2  pz => 212 |
    /// 3 s  => 300 | 3 px  => 310 | 3 py  => 311 | 3  pz => 312 |
    ///             | 3 dz2 => 320 | 3 dzx => 321 | 3 dyz => 322 | 3 dx2y2 => 323 | 3 dxy => 324
    pub fn ord_idx(&self) -> usize {
        let mut index: usize = (self.n * 100) as usize;
        index += (self.l * 10) as usize;
        index += match self.l {
            0 => 0, // s-orbitals
            1 => { // p-orbitals
                match self.m {
                    1 => 0, // px
                    -1 => 1,// py
                    0 => 2, // pz
                    _ => 3,
                }
            },
            2 => { // d-orbitals
                match self.m {
                    0 => 0,  // dz2
                    1 => 1,  // dzx
                    -1 => 2, // dyz
                    -2 => 3, // dx2y2
                    2 => 4,  // dxy
                    _ => 5,
                }
            },
            _ => 6,
        };
        index
    }
}

impl PartialEq for AtomicOrbital {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.m == other.m && self.l == other.l
    }
}

impl Eq for AtomicOrbital {}