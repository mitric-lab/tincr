use crate::fmo::{atomvec_to_aomat, Monomer, Pair, SuperSystem};
use crate::initialization::parameters::RepulsivePotential;
use crate::initialization::Atom;
use crate::scc::gamma_approximation::{
    gamma_ao_wise, gamma_atomwise, gamma_atomwise_ab, gamma_gradients_atomwise,
};
use crate::scc::h0_and_s::{h0_and_s, h0_and_s_ab, h0_and_s_gradients};
use crate::scc::mixer::{BroydenMixer, Mixer};
use crate::scc::mulliken::mulliken;
use crate::scc::scc_routine::{RestrictedSCC, SCCError};
use crate::scc::{
    construct_h1, density_matrix, density_matrix_ref, get_electronic_energy, get_repulsive_energy,
    lc_exact_exchange,
};
use crate::utils::Timer;
use approx::AbsDiffEq;
use log::info;
use nalgebra::Vector3;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::prelude::*;
use ndarray::stack;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt, UPLO};
use ndarray_stats::QuantileExt;
use nshare::ToNdarray1;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use std::iter::FromIterator;
use std::ops::{AddAssign, SubAssign};

