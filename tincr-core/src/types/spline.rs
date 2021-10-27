use rusty_fitpack::{splder_uniform, splev_uniform, splrep};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Spline {
    t: Vec<f64>,
    c: Vec<f64>,
    k: usize,
}

#[derive(Copy, Clone)]
pub enum SorH {
    S,
    H0,
}

impl Spline {
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        let (t, c, k) = splrep(
            x.to_vec(),
            y.to_vec(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        Self { t, c, k }
    }

    pub fn eval(&self, x: f64) -> f64 {
        splev_uniform(&self.t, &self.c, self.k, x)
    }

    pub fn deriv(&self, x: f64) -> f64 {
        splder_uniform(&self.t, &self.c, self.k, x, 1)
    }
}
