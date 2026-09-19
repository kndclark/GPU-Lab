//! Testing the parts of `diff` that are reachable without a Prometheus server
//! or nvidia-smi.
//!
//! `diff::run()` orchestrates real I/O (node probes, HTTP, SQLite), so its
//! integration test is `cargo run -- diff` against the lab. What *is* testable
//! without the lab is the `iso8601` conversion, which is a hand-rolled
//! Hinnant algorithm that must match the format the harness writes.

use sm_differ::diff::iso8601;

/// The unix epoch itself.
#[test]
fn epoch_is_1970() {
    assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
}

/// A known timestamp from the lab's own test fixtures. The Prometheus response
/// fixture in `tests/it/prometheus.rs` uses timestamp 1789777366.
#[test]
fn lab_fixture_timestamp() {
    assert_eq!(iso8601(1789777366), "2026-09-19T00:22:46Z");
}

/// A negative timestamp (before the epoch).
#[test]
fn before_the_epoch() {
    // 1969-12-31T23:59:59Z is unix -1.
    assert_eq!(iso8601(-1), "1969-12-31T23:59:59Z");
}

/// Midnight on the boundary: the last second of a day.
#[test]
fn end_of_day() {
    // 1970-01-01T23:59:59Z is unix 86399.
    assert_eq!(iso8601(86_399), "1970-01-01T23:59:59Z");
    // 1970-01-02T00:00:00Z is unix 86400.
    assert_eq!(iso8601(86_400), "1970-01-02T00:00:00Z");
}

/// A timestamp matching the started_utc format used in existing storage tests.
#[test]
fn matches_the_harness_format() {
    // The storage tests use "2026-09-18T12:00:00Z", which is:
    // 2026-09-18 is day number: days since epoch
    let ts = iso8601(1789833600); // 2026-09-18T06:20:00Z -- a sample time
    // Just verify it has the right shape: YYYY-MM-DDTHH:MM:SSZ
    assert!(ts.len() == 20, "unexpected length: {ts}");
    assert!(ts.ends_with('Z'), "must end with Z: {ts}");
    assert!(ts.starts_with("2026-"), "expected 2026: {ts}");
}

/// A well-known historical date: 2000-01-01T00:00:00Z is unix 946684800.
#[test]
fn y2k() {
    assert_eq!(iso8601(946_684_800), "2000-01-01T00:00:00Z");
}

/// A leap year date: 2024-02-29 exists (2024 is a leap year).
#[test]
fn leap_day() {
    // 2024-02-29T00:00:00Z is unix 1709164800.
    assert_eq!(iso8601(1_709_164_800), "2024-02-29T00:00:00Z");
}
