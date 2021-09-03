use crate::properties::property::Property;
use crate::scc::mixer::BroydenMixer;
use hashbrown::HashMap;
use ndarray::prelude::*;
use crate::excited_states::ProductCache;
use std::ops::AddAssign;
use ndarray::Slice;
use crate::fmo::PairType;

