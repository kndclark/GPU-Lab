//! Properties of the verdict function.
//!
//! A comparison has algebraic structure, and structure is what property tests
//! are for. These assert over generated inputs rather than a handful of chosen
//! ones, which matters most for the property this whole crate exists to
//! protect: that a missing measurement can never be mistaken for a zero.
//! A hand-written example suite under-covers exactly that case, because the
//! author picks the examples and the author is the one with the blind spot.
//!
//! Several tests below carry the name of a defect that was found by attacking
//! this crate rather than by writing it. Those are regression tests, and the
//! comment on each says what the code did before.

use proptest::prelude::*;
use sm_differ::measure::{Measurement, Refusal, Sample};
use sm_differ::verdict::{Broken, GateError, Gates, Polarity, Verdict, compare};

/// Readings that look like benchmark data: a centre with a little jitter.
/// Real runs cluster; they do not scatter across four orders of magnitude.
fn clustered() -> impl Strategy<Value = Vec<f64>> {
    (1.0f64..1000.0, 0.001f64..0.05, 2usize..12).prop_map(|(centre, spread, n)| {
        (0..n)
            .map(|i| {
                // Deterministic jitter, so a shrunk counterexample stays
                // reproducible: alternate above and below the centre.
                let step = (i as f64 / n as f64) - 0.5;
                centre * (1.0 + spread * step)
            })
            .collect()
    })
}

/// Anything finite and positive, including wildly-spread samples. Used where
/// the property must hold for ugly data too.
fn any_readings() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(0.001f64..10_000.0, 2..12)
}

fn measured(v: Vec<f64>) -> Measurement {
    let n = v.len();
    Measurement::Measured(Sample::new(v, n).expect("generator should only produce valid samples"))
}

proptest! {
    /// Comparing a run against itself must never report a direction. If this
    /// ever fails, the tool is manufacturing differences out of nothing.
    ///
    /// Stated as "never a direction" rather than "always Unchanged" because a
    /// sample whose spread exceeds its own mean is legitimately refused -- see
    /// `a_baseline_lost_in_its_own_noise_is_refused`.
    #[test]
    fn comparing_a_run_to_itself_is_never_a_direction(v in any_readings()) {
        let m = measured(v);
        let got = compare(&m, &m, Polarity::HigherIsBetter, Gates::default());
        prop_assert!(
            !matches!(got, Verdict::Improved { .. } | Verdict::Regressed { .. }),
            "comparing a sample with itself claimed a direction: {got:?}"
        );
    }

    /// For well-behaved data the stronger form holds: self-comparison is
    /// exactly Unchanged.
    #[test]
    fn self_comparison_of_clean_data_is_unchanged(v in clustered()) {
        let m = measured(v);
        let got = compare(&m, &m, Polarity::HigherIsBetter, Gates::default());
        prop_assert!(matches!(got, Verdict::Unchanged { .. }), "got {got:?}");
    }

    /// Swapping baseline and current must flip the direction, never preserve it.
    #[test]
    fn swapping_arguments_flips_direction(a in clustered(), b in clustered()) {
        let (ma, mb) = (measured(a), measured(b));
        let forward = compare(&ma, &mb, Polarity::HigherIsBetter, Gates::default());
        let backward = compare(&mb, &ma, Polarity::HigherIsBetter, Gates::default());

        match (&forward, &backward) {
            (Verdict::Improved { .. }, Verdict::Regressed { .. })
            | (Verdict::Regressed { .. }, Verdict::Improved { .. })
            | (Verdict::Unchanged { .. }, Verdict::Unchanged { .. })
            | (Verdict::Broken(_), Verdict::Broken(_)) => {}
            // Asymmetry is legitimate in one narrow band only. The percentage
            // is relative to whichever side is the baseline, so for a forward
            // change of f the backward change is -f/(1+f); the two straddle the
            // 1% floor only for |change| in roughly 0.990%..1.010%. The old
            // version of this test allowed a 6% window -- 300x too wide, and it
            // would have passed against a substantially broken implementation.
            (Verdict::Improved { change_pct, .. }, Verdict::Unchanged { .. })
            | (Verdict::Regressed { change_pct, .. }, Verdict::Unchanged { .. })
            | (Verdict::Unchanged { .. }, Verdict::Improved { change_pct, .. })
            | (Verdict::Unchanged { .. }, Verdict::Regressed { change_pct, .. }) => {
                prop_assert!(
                    (0.98..=1.02).contains(&change_pct.abs()),
                    "direction disagreed at {change_pct:.4}%, outside the 1% straddle band: \
                     {forward:?} vs {backward:?}"
                );
            }
            _ => prop_assert!(false, "{forward:?} then {backward:?}"),
        }
    }

    /// Reversing polarity must swap Improved and Regressed and touch nothing
    /// else. This is what stops a latency metric being read as throughput.
    #[test]
    fn polarity_only_renames_the_direction(a in clustered(), b in clustered()) {
        let (ma, mb) = (measured(a), measured(b));
        let higher = compare(&ma, &mb, Polarity::HigherIsBetter, Gates::default());
        let lower = compare(&ma, &mb, Polarity::LowerIsBetter, Gates::default());
        match (&higher, &lower) {
            (Verdict::Improved { .. }, Verdict::Regressed { .. })
            | (Verdict::Regressed { .. }, Verdict::Improved { .. })
            | (Verdict::Unchanged { .. }, Verdict::Unchanged { .. })
            | (Verdict::Broken(_), Verdict::Broken(_)) => {}
            _ => prop_assert!(false, "polarity changed more than direction: {higher:?} / {lower:?}"),
        }
    }

    /// THE PROPERTY THIS CRATE EXISTS FOR. A missing measurement on either
    /// side absorbs: the answer is Broken, whatever the other side holds. No
    /// arithmetic is performed and no number is invented.
    #[test]
    fn a_missing_side_absorbs(v in clustered()) {
        let present = measured(v);
        let absent = [
            Measurement::NotRun,
            Measurement::Refused(Refusal::TooHot { observed_c: 88.0, threshold_c: 80.0 }),
            Measurement::Refused(Refusal::NotFinite),
            Measurement::Refused(Refusal::NoVariance { got: 1 }),
            Measurement::Refused(Refusal::TooFewSamples { got: 12, wanted: 50 }),
        ];
        for m in &absent {
            for (l, r) in [(m, &present), (&present, m)] {
                let got = compare(l, r, Polarity::HigherIsBetter, Gates::default());
                prop_assert!(
                    matches!(got, Verdict::Broken(_)),
                    "a missing side produced {got:?} instead of Broken"
                );
            }
        }
    }

    /// A change strictly inside the practical-significance floor is never
    /// reported as a direction, however statistically clean it is. This is
    /// "do not report thermal weather as a kernel regression", stated as a
    /// property rather than as a hope.
    #[test]
    fn changes_under_the_noise_floor_are_never_a_direction(
        base in 10.0f64..1000.0,
        frac in -0.009f64..0.009,
    ) {
        // Real jitter, deliberately: with perfectly flat samples the noise
        // cannot be estimated at all and the answer is Broken, which is a
        // different test (see quantised_readings_are_not_certainty).
        let jitter = [0.999, 1.0, 1.001, 0.9995, 1.0005, 1.0];
        let b = measured(jitter.iter().map(|j| base * j).collect());
        let c = measured(jitter.iter().map(|j| base * (1.0 + frac) * j).collect());
        let got = compare(&b, &c, Polarity::HigherIsBetter, Gates::default());
        prop_assert!(
            matches!(got, Verdict::Unchanged { .. }),
            "a {:.3}% change cleared the 1% floor: {got:?}", frac * 100.0
        );
    }
}

// ---------------------------------------------------------------------------
// Regression tests. Each one is a defect that reached a verdict before it was
// found by attacking this crate.
// ---------------------------------------------------------------------------

/// A coarser instrument must not buy more confidence.
///
/// Before this was fixed, two flat samples with different means returned
/// `p = 0.0` -- maximal certainty -- so gate 1 became inert and the tool
/// degenerated to a bare percentage threshold, which its own documentation
/// calls wrong. Latencies of 12,12,12 against 13,13,13 from a millisecond
/// timer were announced `REGRESSED (p = 0.0000)`, while the *same* latencies
/// measured finely came back `unchanged, p = 0.058`. Integer MiB from
/// nvidia-smi and millisecond timers make this ordinary lab data.
#[test]
fn quantised_readings_are_not_certainty() {
    let coarse_b = measured(vec![12.0, 12.0, 12.0]);
    let coarse_c = measured(vec![13.0, 13.0, 13.0]);
    let got = compare(
        &coarse_b,
        &coarse_c,
        Polarity::LowerIsBetter,
        Gates::default(),
    );
    assert!(
        matches!(got, Verdict::Broken(Broken::NoiseNotEstimable { .. })),
        "flat samples with different means must refuse, not assert certainty: {got:?}"
    );

    // The same physical latencies, measured with a finer instrument, are
    // answerable -- and the answer is that they cannot be told apart.
    let fine_b = measured(vec![12.3, 11.8, 12.4]);
    let fine_c = measured(vec![12.6, 13.4, 12.9]);
    let fine = compare(&fine_b, &fine_c, Polarity::LowerIsBetter, Gates::default());
    assert!(matches!(fine, Verdict::Unchanged { .. }), "got {fine:?}");
}

/// Flat *and equal* is answerable: there is no difference to explain.
#[test]
fn flat_and_equal_is_unchanged_not_broken() {
    let m = measured(vec![7000.0, 7000.0, 7000.0]);
    let got = compare(&m, &m, Polarity::HigherIsBetter, Gates::default());
    assert!(matches!(got, Verdict::Unchanged { .. }), "got {got:?}");
}

/// A baseline smaller than its own spread cannot anchor a percentage.
///
/// Before this, only an exact zero was refused, so a delta metric hovering
/// near zero produced changes like +2,000,099,900% that cleared any floor --
/// gate 2 switched off by plausible data, exactly as gate 1 was by the defect
/// above.
#[test]
fn a_baseline_lost_in_its_own_noise_is_refused() {
    let b = measured(vec![-0.00000005, 0.00000015]);
    let c = measured(vec![1.0, 1.0001]);
    let got = compare(&b, &c, Polarity::HigherIsBetter, Gates::default());
    assert!(
        matches!(
            got,
            Verdict::Broken(Broken::BaselineIndistinguishableFromZero { .. })
        ),
        "got {got:?}"
    );
}

/// Zero is a measurement. It is not a missing measurement, and the two must
/// reach different answers.
///
/// This diverges from the obvious expectation that 0-vs-0 is "unchanged". It
/// cannot be: a relative change against a zero baseline is not defined, and
/// answering "unchanged" would be the tool inventing a conclusion its
/// arithmetic does not support. It returns Broken -- but a *different* Broken
/// than a missing side does, which is the distinction that matters.
#[test]
fn zero_is_not_missing() {
    let zero = measured(vec![0.0, 0.0, 0.0]);
    let got = compare(&zero, &zero, Polarity::HigherIsBetter, Gates::default());
    assert_eq!(got, Verdict::Broken(Broken::BaselineIsZero));

    let missing = compare(
        &Measurement::NotRun,
        &zero,
        Polarity::HigherIsBetter,
        Gates::default(),
    );
    assert!(matches!(
        missing,
        Verdict::Broken(Broken::BaselineMissing(_))
    ));

    assert_ne!(
        got, missing,
        "a zero baseline and a missing baseline must not be the same verdict"
    );
}

/// A non-finite reading never becomes a Sample.
#[test]
fn nan_is_refused_at_ingest() {
    assert!(matches!(
        Sample::new(vec![1.0, f64::NAN], 2),
        Err(Refusal::NotFinite)
    ));
    assert!(matches!(
        Sample::new(vec![1.0, f64::INFINITY], 2),
        Err(Refusal::NotFinite)
    ));
}

/// Every reading finite, the mean not.
///
/// `1e308 + 1e308` overflows. Before this was caught at ingest, the resulting
/// `change_pct` was NaN, which compares false against every gate, so it landed
/// in a *conclusive* `Unchanged` printing "NaN% observed" and exiting 0.
#[test]
fn a_finite_sample_with_an_infinite_mean_is_refused() {
    assert!(matches!(
        Sample::new(vec![1e308, 1e308], 2),
        Err(Refusal::NotFinite)
    ));
}

/// One reading has no variance, so it cannot support a verdict -- and it must
/// say so without inventing a claim about what the run asked for.
///
/// This previously reported `TooFewSamples { got: 1, wanted: 1 }`, which
/// rendered as "only 1 of 1 samples completed" -- self-contradictory, and a
/// fabricated fact in a record the crate insists is durable evidence.
#[test]
fn a_single_reading_is_refused_without_inventing_a_target() {
    match Sample::new(vec![42.0], 1) {
        Err(Refusal::NoVariance { got: 1 }) => {}
        other => panic!("expected NoVariance, got {other:?}"),
    }
    // A genuine shortfall still reports as one.
    match Sample::new(vec![1.0, 2.0, 3.0], 50) {
        Err(Refusal::TooFewSamples { got: 3, wanted: 50 }) => {}
        other => panic!("expected TooFewSamples, got {other:?}"),
    }
}

/// A stored row must pass the same gate a live reading does.
///
/// The derived `Deserialize` built the private field directly, so
/// `{"Measured":{"values":[]}}` produced a Sample that had never been checked.
/// It rendered as "NaN (n=0)", satisfied `sample().is_some()`, and reached a
/// conclusive `Unchanged` with exit 0. A 50% throughput collapse stored as
/// one-reading rows came back "unchanged" the same way.
#[test]
fn a_stored_row_cannot_bypass_the_constructor() {
    for bad in [
        r#"{"Measured":{"values":[]}}"#,
        r#"{"Measured":{"values":[42.0]}}"#,
        r#"{"Measured":{"values":[1e308,1e308]}}"#,
        // Unknown fields are refused too: a row carrying both readings and a
        // refusal must not quietly read as if it only had readings.
        r#"{"Measured":{"values":[1.0,2.0],"refusal":{"NotFinite":null}}}"#,
    ] {
        let parsed: Result<Measurement, _> = serde_json::from_str(bad);
        assert!(parsed.is_err(), "{bad} deserialized into {:?}", parsed.ok());
    }

    // A well-formed row still round-trips, and absences stay absent.
    for good in [
        Measurement::Measured(Sample::new(vec![1.0, 2.0, 3.0], 3).unwrap()),
        Measurement::NotRun,
        Measurement::Refused(Refusal::TooHot {
            observed_c: 90.0,
            threshold_c: 80.0,
        }),
    ] {
        let json = serde_json::to_string(&good).unwrap();
        let back: Measurement = serde_json::from_str(&json).unwrap();
        assert_eq!(good, back, "round trip changed {json}");
    }
}

/// The thresholds that judge measurements must themselves be judged.
///
/// `--significance nan` made every comparison fail gate 1 silently, so a real
/// 20% regression printed "unchanged" and exited 0. `--noise=-1` manufactured
/// an IMPROVED verdict out of a +0.00% change.
#[test]
fn gates_that_cannot_judge_are_refused() {
    for bad in [f64::NAN, 0.0, 1.0, 1.5, -0.1, f64::INFINITY] {
        assert!(
            matches!(Gates::new(bad, 0.01), Err(GateError::Significance(_))),
            "significance {bad} was accepted"
        );
    }
    for bad in [f64::NAN, -1.0, f64::INFINITY] {
        assert!(
            matches!(Gates::new(0.05, bad), Err(GateError::Noise(_))),
            "noise threshold {bad} was accepted"
        );
    }
    assert!(Gates::new(0.05, 0.01).is_ok());
    // A zero noise floor is legitimate -- it means "report any real change".
    assert!(Gates::new(0.05, 0.0).is_ok());
}

/// With a zero noise floor, an exactly-zero change must still not be a
/// direction. It used to fall through to `0.0 > 0.0 == false` and be announced
/// as a regression.
#[test]
fn no_change_is_never_a_regression() {
    let m = measured(vec![100.0, 101.0, 99.0]);
    let gates = Gates::new(0.5, 0.0).unwrap();
    let got = compare(&m, &m, Polarity::HigherIsBetter, gates);
    assert!(matches!(got, Verdict::Unchanged { .. }), "got {got:?}");
}

/// A real regression, well clear of both gates, must be reported as one --
/// the tool has to be capable of saying yes, not only of withholding.
#[test]
fn a_real_regression_is_reported() {
    let b = measured(vec![175.9, 176.1, 175.7, 176.0, 175.8]);
    let c = measured(vec![140.2, 140.0, 139.8, 140.4, 140.1]);
    match compare(&b, &c, Polarity::HigherIsBetter, Gates::default()) {
        Verdict::Regressed {
            change_pct,
            p_value,
        } => {
            assert!(
                change_pct < -15.0,
                "expected a large drop, got {change_pct}"
            );
            assert!(p_value < 0.01, "expected high confidence, got p={p_value}");
        }
        other => panic!("a 20% throughput drop was not reported as a regression: {other:?}"),
    }
}

/// A negative baseline must not invert the direction. Verified here because
/// the CLI could not express negative readings at all until recently.
#[test]
fn a_negative_baseline_keeps_its_direction() {
    let b = measured(vec![-10.2, -9.8, -10.0]);
    let c = measured(vec![-20.1, -19.9, -20.0]);
    assert!(matches!(
        compare(&b, &c, Polarity::LowerIsBetter, Gates::default()),
        Verdict::Improved { .. }
    ));
    assert!(matches!(
        compare(&b, &c, Polarity::HigherIsBetter, Gates::default()),
        Verdict::Regressed { .. }
    ));
}

/// The twin of `a_stored_row_cannot_bypass_the_constructor`.
///
/// `Gates` had the same shape as `Sample` did -- a validating constructor and
/// a derived `Deserialize` that ignored it. Found by searching for the pattern
/// after fixing the first one, rather than by waiting for it to cause harm.
#[test]
fn stored_gates_cannot_bypass_validation() {
    for bad in [
        r#"{"significance_level":1.5,"noise_threshold":0.01}"#,
        r#"{"significance_level":0.0,"noise_threshold":0.01}"#,
        r#"{"significance_level":0.05,"noise_threshold":-1.0}"#,
        r#"{"significance_level":0.05,"noise_threshold":0.01,"extra":1}"#,
    ] {
        let parsed: Result<Gates, _> = serde_json::from_str(bad);
        assert!(parsed.is_err(), "{bad} deserialized into {:?}", parsed.ok());
    }
    let good: Gates = serde_json::from_str(r#"{"significance_level":0.05,"noise_threshold":0.01}"#)
        .expect("valid gates should parse");
    assert_eq!(good, Gates::default());
}
