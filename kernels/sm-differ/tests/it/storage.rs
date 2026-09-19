//! The database must preserve absences, and must not let a stored row
//! re-enter the system unchecked.

use sm_differ::measure::{Measurement, Provenance, Refusal, Sample};
use sm_differ::store::Store;
use sm_differ::verdict::{Broken, Gates, Polarity, Verdict, compare_recorded};

fn prov(arch: &str, driver: &str) -> Provenance {
    Provenance {
        git_ref: "abc1234".into(),
        arch: arch.into(),
        node: if arch == "sm_86" { "desktop" } else { "laptop" }.into(),
        driver: driver.into(),
        image_id: "sha256:97e2441087b4".into(),
    }
}

fn sample(v: Vec<f64>) -> Measurement {
    let n = v.len();
    Measurement::Measured(Sample::new(v, n).unwrap())
}

/// A refusal is a row, not a gap. If the database dropped refusals, the next
/// reader would see a missing row and have to guess what it meant.
#[test]
fn a_refusal_round_trips_as_a_refusal() {
    let s = Store::in_memory().unwrap();
    let p = prov("sm_120", "595.91.07");
    let run = s.begin_run(&p, "2026-09-18T12:00:00Z").unwrap();

    let too_hot = Measurement::Refused(Refusal::TooHot {
        observed_c: 88.5,
        threshold_c: 80.0,
    });
    s.record(run, "decode", "tok_per_s", &too_hot).unwrap();

    let back = s.latest("sm_120", "decode", "tok_per_s").unwrap().unwrap();
    assert_eq!(back.measurement, too_hot);
    assert!(
        back.measurement.sample().is_none(),
        "a refusal must not yield a sample"
    );
    assert_eq!(back.provenance, p);
}

#[test]
fn not_run_round_trips_and_stays_absent() {
    let s = Store::in_memory().unwrap();
    let run = s
        .begin_run(&prov("sm_86", "595.91.07"), "2026-09-18T12:00:00Z")
        .unwrap();
    s.record(run, "prefill", "ttft_s", &Measurement::NotRun)
        .unwrap();
    let back = s.latest("sm_86", "prefill", "ttft_s").unwrap().unwrap();
    assert_eq!(back.measurement, Measurement::NotRun);
}

#[test]
fn readings_round_trip_exactly() {
    let s = Store::in_memory().unwrap();
    let run = s
        .begin_run(&prov("sm_86", "595.91.07"), "2026-09-18T12:00:00Z")
        .unwrap();
    let m = sample(vec![175.9, 176.1, 175.7]);
    s.record(run, "decode", "tok_per_s", &m).unwrap();
    let back = s.latest("sm_86", "decode", "tok_per_s").unwrap().unwrap();
    assert_eq!(back.measurement, m);
}

/// Re-recording replaces rather than accumulating. Two rows for one metric in
/// one run would be averaged or arbitrarily picked by some later query, and
/// both are wrong.
#[test]
fn re_recording_a_metric_replaces_it() {
    let s = Store::in_memory().unwrap();
    let run = s
        .begin_run(&prov("sm_86", "595.91.07"), "2026-09-18T12:00:00Z")
        .unwrap();
    s.record(run, "decode", "tok_per_s", &sample(vec![1.0, 2.0]))
        .unwrap();
    s.record(run, "decode", "tok_per_s", &sample(vec![10.0, 20.0]))
        .unwrap();
    let all = s.history("sm_86", "decode", "tok_per_s").unwrap();
    assert_eq!(
        all.len(),
        1,
        "a re-run must not leave two rows for one metric"
    );
    assert_eq!(all[0].measurement, sample(vec![10.0, 20.0]));
}

/// The newest run wins, and history comes back newest first.
#[test]
fn latest_is_the_newest_run() {
    let s = Store::in_memory().unwrap();
    let p = prov("sm_120", "595.91.07");
    let older = s.begin_run(&p, "2026-09-17T09:00:00Z").unwrap();
    let newer = s.begin_run(&p, "2026-09-18T09:00:00Z").unwrap();
    s.record(older, "decode", "tok_per_s", &sample(vec![100.0, 101.0]))
        .unwrap();
    s.record(newer, "decode", "tok_per_s", &sample(vec![200.0, 201.0]))
        .unwrap();

    let latest = s.latest("sm_120", "decode", "tok_per_s").unwrap().unwrap();
    assert_eq!(latest.measurement, sample(vec![200.0, 201.0]));
    let hist = s.history("sm_120", "decode", "tok_per_s").unwrap();
    assert_eq!(hist.len(), 2);
    assert_eq!(hist[0].started_utc, "2026-09-18T09:00:00Z");
}

/// Architectures do not bleed into each other. The whole point of the tool is
/// that sm_86 and sm_120 are separate series.
#[test]
fn architectures_are_kept_apart() {
    let s = Store::in_memory().unwrap();
    let a = s
        .begin_run(&prov("sm_86", "595.91.07"), "2026-09-18T09:00:00Z")
        .unwrap();
    let b = s
        .begin_run(&prov("sm_120", "595.91.07"), "2026-09-18T09:00:00Z")
        .unwrap();
    s.record(a, "decode", "tok_per_s", &sample(vec![175.9, 176.1]))
        .unwrap();
    s.record(b, "decode", "tok_per_s", &sample(vec![140.2, 140.0]))
        .unwrap();

    assert_eq!(
        s.latest("sm_86", "decode", "tok_per_s")
            .unwrap()
            .unwrap()
            .measurement,
        sample(vec![175.9, 176.1])
    );
    assert!(
        s.latest("sm_120", "decode", "other_metric")
            .unwrap()
            .is_none()
    );
}

/// A corrupt row is an error, not a degraded measurement. This is the
/// constructor-bypass defect again, one layer out: the database is exactly the
/// place a row written by an older version would come back from.
#[test]
fn a_corrupt_row_is_an_error_not_a_zero() {
    let s = Store::in_memory().unwrap();
    let run = s
        .begin_run(&prov("sm_86", "595.91.07"), "2026-09-18T09:00:00Z")
        .unwrap();
    s.record(run, "decode", "tok_per_s", &sample(vec![1.0, 2.0]))
        .unwrap();
    // Simulate a row from a version that did not enforce the floor.
    s.overwrite_raw_for_tests(
        run,
        "decode",
        "tok_per_s",
        r#"{"Measured":{"values":[42.0]}}"#,
    )
    .unwrap();
    let err = s.latest("sm_86", "decode", "tok_per_s").unwrap_err();
    assert!(
        format!("{err:#}").contains("not readable"),
        "expected a readable error, got {err:#}"
    );
}

/// Diffing across a driver mismatch is refused -- the runbook's kill list.
#[test]
fn a_driver_mismatch_refuses_to_diff() {
    let base = (
        prov("sm_120", "595.91.07"),
        sample(vec![175.9, 176.1, 175.7]),
    );
    let curr = (
        prov("sm_120", "596.21.00"),
        sample(vec![140.2, 140.0, 139.8]),
    );
    let got = compare_recorded(
        (&base.0, &base.1),
        (&curr.0, &curr.1),
        Polarity::HigherIsBetter,
        Gates::default(),
    );
    assert!(
        matches!(got, Verdict::Broken(Broken::DriverMismatch { .. })),
        "a 20% drop across different drivers must refuse, got {got:?}"
    );

    // Same driver, same numbers: now it is answerable.
    let ok = compare_recorded(
        (&base.0, &base.1),
        (&base.0, &curr.1),
        Polarity::HigherIsBetter,
        Gates::default(),
    );
    assert!(matches!(ok, Verdict::Regressed { .. }), "got {ok:?}");
}

/// The runbook's sharpest Phase 3 exit criterion, in miniature: two runs of
/// the same thing must report Unchanged, or the harness is the noise source.
#[test]
fn two_recordings_of_the_same_ref_report_unchanged() {
    let s = Store::in_memory().unwrap();
    let p = prov("sm_86", "595.91.07");
    let r1 = s.begin_run(&p, "2026-09-18T09:00:00Z").unwrap();
    let r2 = s.begin_run(&p, "2026-09-18T10:00:00Z").unwrap();
    let readings = vec![175.9, 176.1, 175.7, 176.0, 175.8];
    s.record(r1, "decode", "tok_per_s", &sample(readings.clone()))
        .unwrap();
    s.record(r2, "decode", "tok_per_s", &sample(readings))
        .unwrap();

    let hist = s.history("sm_86", "decode", "tok_per_s").unwrap();
    let got = compare_recorded(
        (&hist[1].provenance, &hist[1].measurement),
        (&hist[0].provenance, &hist[0].measurement),
        Polarity::HigherIsBetter,
        Gates::default(),
    );
    assert!(matches!(got, Verdict::Unchanged { .. }), "got {got:?}");
}
