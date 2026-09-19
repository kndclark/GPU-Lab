//! sm-differ -- the cross-architecture regression differ for this lab.
//!
//! The library half exists so the integration tests can reach the domain types
//! without going through the binary. `main.rs` stays thin on purpose: argument
//! parsing and printing, nothing that deserves a test of its own.
//!
//! Only the modules that are actually written live here. The Prometheus
//! client, the JUnit reader and the Docker driver are designed and their
//! dependencies are chosen (see Cargo.toml) but not yet implemented -- and an
//! empty module that compiles is a claim that something exists, so there are
//! none.

pub mod measure;
pub mod prom;
pub mod store;
pub mod verdict;
