use serde::{Serialize, Deserialize};
use crate::io::{Configuration, frame_to_coordinates};
use chemfiles::Frame;
use crate::param::slako::ParamFiles;
use ndarray::prelude::*;
use core::::{Atom, create_atoms, get_parametrization, initialize_gamma_function};
use core::::gamma_approximation::gamma_atomwise;
use crate::data::{Parametrization};
use core::::scc_routine::RestrictedSCC;

