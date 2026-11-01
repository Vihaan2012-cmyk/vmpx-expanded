//! SPIKE: concolic (directed) dynamic-assist loop over a recovered VM function.
//! Observed tier only; never sound output.
pub mod ctx;
pub mod engine;
pub mod harness;
pub mod lower;
pub mod pretty;
pub mod recover;
pub mod solve;
pub mod symbolic;
