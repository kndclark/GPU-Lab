//! Parsing nvidia-smi output without nvidia-smi.
//!
//! The I/O half (`Node::probe`) shells out to nvidia-smi or ssh, so it can
//! only be tested on the lab hardware. The parsing half is exercised here with
//! the same captured-fixture approach `prometheus.rs` uses.

use sm_differ::node::{NodeInfo, parse_nvidia_smi_output};

/// Real output from the desktop (sm_86, RTX 3090).
#[test]
fn parses_the_desktop_card() {
    let out = "595.91.07, NVIDIA GeForce RTX 3090, 8.6, 42\n";
    let info = parse_nvidia_smi_output(out).expect("valid desktop output");
    assert_eq!(
        info,
        NodeInfo {
            arch: "sm_86".into(),
            driver: "595.91.07".into(),
            gpu: "NVIDIA GeForce RTX 3090".into(),
            node: "desktop".into(),
            temperature_c: 42.0,
        }
    );
}

/// Real output from the laptop (sm_120, RTX 5090).
#[test]
fn parses_the_laptop_card() {
    let out = "575.51.03, NVIDIA GeForce RTX 5090 Laptop GPU, 12.0, 55\n";
    let info = parse_nvidia_smi_output(out).expect("valid laptop output");
    assert_eq!(info.arch, "sm_120");
    assert_eq!(info.node, "laptop");
    assert_eq!(info.driver, "575.51.03");
    assert_eq!(info.temperature_c, 55.0);
}

/// An unknown compute capability is a hard error naming the two known cards,
/// not a guess. The lab is built on two specific cards, and an unrecognised
/// one means something has changed that the tool cannot handle silently.
#[test]
fn unknown_compute_cap_is_refused_by_name() {
    let out = "595.91.07, Some GPU, 9.0, 40\n";
    let err = parse_nvidia_smi_output(out).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("90"), "should name the cap: {msg}");
    assert!(msg.contains("8.6"), "should name the known cards: {msg}");
}

/// Fewer than four fields is not parseable.
#[test]
fn too_few_fields_is_an_error() {
    let out = "595.91.07, NVIDIA GeForce RTX 3090\n";
    assert!(parse_nvidia_smi_output(out).is_err());
}

/// Empty output (nvidia-smi returned nothing).
#[test]
fn empty_output_is_an_error() {
    assert!(parse_nvidia_smi_output("").is_err());
    assert!(parse_nvidia_smi_output("\n").is_err());
    assert!(parse_nvidia_smi_output("  \n").is_err());
}

/// A temperature that is not a number is an error, not a default.
#[test]
fn non_numeric_temperature_is_an_error() {
    let out = "595.91.07, NVIDIA GeForce RTX 3090, 8.6, N/A\n";
    let err = parse_nvidia_smi_output(out).unwrap_err();
    assert!(
        format!("{err:#}").contains("N/A"),
        "should name the bad value"
    );
}

/// Extra fields beyond the expected four are tolerated — nvidia-smi may add
/// columns, and refusing them would break the tool on a version that does.
#[test]
fn extra_fields_are_ignored() {
    let out = "595.91.07, NVIDIA GeForce RTX 3090, 8.6, 42, 300\n";
    let info = parse_nvidia_smi_output(out).expect("extra fields should not cause failure");
    assert_eq!(info.arch, "sm_86");
    assert_eq!(info.temperature_c, 42.0);
}

/// Whitespace around fields is trimmed, matching the CSV format nvidia-smi uses.
#[test]
fn whitespace_around_fields_is_trimmed() {
    let out = "  595.91.07 ,  NVIDIA GeForce RTX 3090 ,  8.6 ,  42  \n";
    let info = parse_nvidia_smi_output(out).expect("should handle whitespace");
    assert_eq!(info.driver, "595.91.07");
    assert_eq!(info.temperature_c, 42.0);
}
