mod zbrent;
mod array_helper;

pub use array_helper::ToOwnedF;
pub use zbrent::zbrent;
use std::time::Instant;
use std::fmt;


pub enum Calculation {
    Converged,
    NotConverged,
}

/// A simple timer based on std::time::Instant, to implement the std::fmt::Display trait on
pub struct Timer {
    time: Instant,
}

impl Timer {
    pub fn start() -> Self {
        Timer{time: Instant::now()}
    }
}

// Implement `Display` for Instant.
impl fmt::Display for Timer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use `self.number` to refer to each positional data point.
        write!(f,
            "{:>68} {:>8.2} s",
            "elapsed time:",
            self.time.elapsed().as_secs_f32()
        )
    }
}
