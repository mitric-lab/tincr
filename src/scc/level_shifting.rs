use ndarray::prelude::*;
use crate::defaults;

pub struct LevelShifter {
    shift_value: f64,
    pub(crate) weight: f64,
    vv_block: Array2<f64>,
    activated: bool,
}

impl LevelShifter {
    pub fn empty() -> LevelShifter {
        let empty_2d_array: Array2<f64> = Array2::zeros([1, 1]);
        LevelShifter {
            shift_value: 0.0,
            weight: 0.0,
            vv_block: empty_2d_array,
            activated: false,
        }
    }

    pub fn new(n_orb: usize, lumo_idx: usize) -> LevelShifter {
        let mut vv_block: Array2<f64> = Array2::zeros([n_orb, n_orb]);
        let n_virts = n_orb - lumo_idx;
        let v_ones: Array2<f64> = Array2::eye(n_virts);
        vv_block
            .slice_mut(s![lumo_idx.., lumo_idx..])
            .assign(&v_ones);
        LevelShifter {
            shift_value: defaults::HOMO_LUMO_SHIFT,
            weight: 1.0,
            vv_block: vv_block,
            activated: true,
        }
    }

    fn shift(&mut self, orbs: ArrayView2<f64>) -> Array2<f64> {
        orbs.dot(&(self.weight * self.shift_value * &self.vv_block).dot(&orbs.t()))
    }

    fn reduce_weight(&mut self) {
        self.weight *= 0.5;
    }

    fn turn_off(&mut self) {
        self.weight = 0.0;
        self.activated = false;
    }

    fn is_empty(&self) -> bool {
        self.shift_value == 0.0
    }
}