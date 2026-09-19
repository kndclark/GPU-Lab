//! One integration binary, with the suites as modules.
//!
//! Every file placed directly under tests/ compiles to its own binary and
//! re-links the whole crate, so a suite split across files pays that cost once
//! per file. Collapsing them here keeps it to one link. It also makes sharing
//! helpers ordinary -- no `mod common;` repeated in every file.

mod cli;
mod prometheus;
mod properties;
mod storage;
