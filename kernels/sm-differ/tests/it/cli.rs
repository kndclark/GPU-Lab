//! The binary answers, says which verdict it reached, and carries it in the
//! exit code.

use assert_cmd::Command;
use predicates::prelude::*;

fn sm_differ() -> Command {
    Command::cargo_bin("sm-differ").expect("binary should build")
}

#[test]
fn reports_a_regression_and_exits_zero() {
    sm_differ()
        .args([
            "verdict",
            "--baseline",
            "175.9",
            "176.1",
            "175.7",
            "--current",
            "140.2",
            "140.0",
            "139.8",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("VERDICT: REGRESSED"));
}

#[test]
fn lower_is_better_inverts_the_direction() {
    sm_differ()
        .args([
            "verdict",
            "--polarity",
            "lower",
            "--baseline",
            "0.031",
            "0.030",
            "0.032",
            "--current",
            "0.016",
            "0.015",
            "0.017",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("VERDICT: IMPROVED"));
}

/// A broken comparison exits 3, not 2: clap uses 2 for a usage error, and a
/// caller has to be able to tell those apart.
#[test]
fn a_nan_is_refused_and_exits_three() {
    sm_differ()
        .args([
            "verdict",
            "--baseline",
            "NaN",
            "1.0",
            "--current",
            "1.0",
            "1.0",
        ])
        .assert()
        .code(3)
        .stdout(predicate::str::contains("BROKEN"));
}

#[test]
fn a_usage_error_still_exits_two() {
    sm_differ()
        .args(["verdict", "--baseline", "1.0"])
        .assert()
        .code(2);
}

#[test]
fn a_sub_one_percent_change_is_not_a_direction() {
    sm_differ()
        .args([
            "verdict",
            "--baseline",
            "100.0",
            "100.1",
            "99.9",
            "--current",
            "100.5",
            "100.6",
            "100.4",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("VERDICT: unchanged"));
}

/// The quantised-timer case: a coarse instrument must not buy confidence.
#[test]
fn flat_readings_refuse_rather_than_assert_certainty() {
    sm_differ()
        .args([
            "verdict",
            "--polarity",
            "lower",
            "--baseline",
            "12",
            "12",
            "12",
            "--current",
            "13",
            "13",
            "13",
        ])
        .assert()
        .code(3)
        .stdout(predicate::str::contains("BROKEN"));
}

/// Gates that cannot judge are refused before any measurement is read, and the
/// failure is loud rather than a silent green.
#[test]
fn unjudgeable_gates_are_refused() {
    sm_differ()
        .args([
            "verdict",
            "--significance",
            "nan",
            "--baseline",
            "175.9",
            "176.1",
            "--current",
            "140.2",
            "140.0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("refusing to run"));
}

/// Negative readings are expressible -- a delta metric is a legitimate input.
#[test]
fn negative_readings_are_accepted() {
    sm_differ()
        .args([
            "verdict",
            "--polarity",
            "lower",
            "--baseline",
            "-10.2",
            "-9.8",
            "-10.0",
            "--current",
            "-20.1",
            "-19.9",
            "-20.0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("VERDICT: IMPROVED"));
}
