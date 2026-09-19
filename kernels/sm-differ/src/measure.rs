//! What a measurement is, and -- more to the point -- what it is when there
//! isn't one.
//!
//! This module exists because of a single failure mode that has cost this lab
//! more time than every architectural difference it has found: a run producing
//! a confident, well-formatted, wrong number and raising no error. Three
//! instances, each checked against the repo rather than recalled:
//!
//! - **A cascade counted as findings.** One overnight run reported **1939**
//!   moe failures on sm_120. Under one-pytest-process-per-file the real figure
//!   was **97 failures and 22 errors** -- a single device-side assert had
//!   destroyed the CUDA context and every later test in the process reported
//!   an indistinguishable launch failure. See `docs/phase2-dev-plane.md`.
//! - **Two bugs counted as one.** The 405 sm_120 failures in
//!   `test_cutlass_scaled_mm.py` were first written up as "all 400 INT8 tests
//!   fail with the dispatch rejection". They are **189** of that kind and
//!   **216** of a different one, failing further down a path with no capability
//!   check at all. A confident count concealed the more serious defect. See
//!   `docs/contributions.md`, C3.
//! - **A verifier that certified itself.** `training/verify.py` "used to print
//!   ADAPTER EFFECTIVE on keyword recall alone" -- an answer calling the 3090
//!   a laptop GPU scored as a win because it contained the token "3090".
//!
//! What the three share is not that a number was missing. It is that a number
//! was *present and unearned*. So the rule this module enforces is: **a
//! measurement that did not happen must never be representable as a number.**
//! Not as 0.0, not as NaN, not as a default. The three ways a measurement can
//! be absent are three variants the compiler will not let a caller ignore.
//!
//! So the rule this module enforces is: **a missing measurement must never be
//! representable as a number.** Not as 0.0, not as NaN, not as a default. The
//! three ways a measurement can be absent are three variants the compiler will
//! not let a caller ignore.

use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Why a benchmark produced no number.
///
/// These are *data*, not errors in the `Result` sense. A GPU that was too hot
/// is a durable fact about that run which belongs in the database and in the
/// report; it is not a failure of the function that read it. `Result` is for
/// "this call did not work", and it is used at the recording boundary only.
// No Eq: two variants carry f64, and f64 is PartialEq but not Eq -- which is
// the type system stating the fact this whole crate is organised around.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Error)]
pub enum Refusal {
    /// The node was above its thermal gate when the benchmark would have run.
    /// §07 of the runbook bans single-sample benchmarks and requires refusing
    /// to record from a throttled node; this is that refusal, recorded.
    #[error("node was at {observed_c} C, above the {threshold_c} C gate")]
    TooHot { observed_c: f64, threshold_c: f64 },

    /// Fewer samples came back than the run asked for.
    #[error("only {got} of {wanted} samples completed")]
    TooFewSamples { got: usize, wanted: usize },

    /// Fewer than two readings, so there is no variance and no way to tell a
    /// regression from noise -- which is the whole job. Distinct from
    /// TooFewSamples because it is a hard floor, not a shortfall against a
    /// request: reporting "only 1 of 1 samples completed" would be both
    /// self-contradictory and a fabricated claim about what the run asked for.
    #[error("{got} reading(s); at least 2 are needed to bound noise")]
    NoVariance { got: usize },

    /// The benchmark emitted a value that is not a number. Recorded rather
    /// than silently dropped, because a NaN is evidence that something
    /// upstream divided by zero or read an empty result -- it is a finding
    /// about the harness, not an absence of data.
    #[error("benchmark produced a non-finite value")]
    NotFinite,

    /// The driver, image or node did not match what the run was pinned to.
    /// Refusing to diff across a driver mismatch is on the runbook's kill list.
    #[error("environment did not match the pinned run: {0}")]
    EnvironmentMismatch(String),
}

/// A set of repeated readings of one metric, and nothing else.
///
/// Construction is fallible on purpose. A non-finite reading is rejected where
/// it enters the system, with a named refusal, instead of travelling
/// three layers inward to a comparison that will silently answer `false` to
/// every question asked of it. Verified on this toolchain: `nan < 1.0`,
/// `nan > 1.0` and `nan == nan` are all `false`, so a naive
/// `if new < old { Improved }` classifies a NaN as "not improved" and says
/// nothing about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SampleWire", into = "SampleWire")]
pub struct Sample {
    values: Vec<NotNan<f64>>,
}

/// The on-the-wire shape of a `Sample`, and the only way one can be
/// deserialized.
///
/// This type exists because of a real defect found by attacking the crate:
/// `#[derive(Deserialize)]` directly on `Sample` built the private `values`
/// field without consulting `Sample::new`, so `{"Measured":{"values":[]}}`
/// produced a Sample that had never passed a single check. It rendered as
/// "NaN (n=0)", satisfied `sample().is_some()`, and reached a *conclusive*
/// `Unchanged` verdict with exit 0 -- a measurement that never happened,
/// presented as a comparison that found nothing wrong. A 50% throughput
/// collapse recorded as two one-reading rows came back "unchanged" the same
/// way. That is precisely the failure this module was written to make
/// impossible, arriving through the one door nobody had checked.
///
/// `deny_unknown_fields` is part of the same fix: a stored row carrying both
/// readings and a refusal must not quietly read as if it only had readings.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SampleWire {
    values: Vec<f64>,
}

impl TryFrom<SampleWire> for Sample {
    type Error = Refusal;
    fn try_from(w: SampleWire) -> Result<Self, Self::Error> {
        // The hard floor only. A stored row cannot know how many readings the
        // original run asked for, so it must not invent a `wanted`.
        Sample::checked(w.values)
    }
}

impl From<Sample> for SampleWire {
    fn from(s: Sample) -> Self {
        SampleWire {
            values: s.values.iter().map(|v| v.into_inner()).collect(),
        }
    }
}

impl Sample {
    /// Build a sample from raw readings, refusing the ones that are not
    /// numbers and refusing a run that returned fewer readings than it asked
    /// for.
    pub fn new(raw: impl IntoIterator<Item = f64>, wanted: usize) -> Result<Self, Refusal> {
        let s = Self::checked(raw.into_iter().collect::<Vec<_>>())?;
        if s.values.len() < wanted {
            return Err(Refusal::TooFewSamples {
                got: s.values.len(),
                wanted,
            });
        }
        Ok(s)
    }

    /// Everything a Sample must satisfy regardless of where it came from: the
    /// readings are finite, there are at least two of them, and the statistics
    /// derived from them are finite too.
    ///
    /// That last check is not redundant, and it was found by attacking this
    /// crate rather than by reasoning about it. Every reading can be finite
    /// while the mean is not: `1e308 + 1e308` overflows to infinity, and the
    /// resulting `change_pct` is NaN, which compares false against every gate
    /// and lands in a *conclusive* `Unchanged` reading "NaN% observed". The
    /// check belongs here rather than in `compare` -- a sample whose mean is
    /// not a number was never a sample.
    ///
    /// This is also the only path a deserialized Sample can take, so a stored
    /// row cannot reintroduce anything this rejects.
    fn checked(raw: Vec<f64>) -> Result<Self, Refusal> {
        let mut values = Vec::with_capacity(raw.len());
        for v in raw {
            // NotNan rejects NaN but NOT infinity, so is_finite() is the wider
            // gate and has to come first.
            if !v.is_finite() {
                return Err(Refusal::NotFinite);
            }
            values.push(NotNan::new(v).map_err(|_| Refusal::NotFinite)?);
        }
        if values.len() < 2 {
            return Err(Refusal::NoVariance { got: values.len() });
        }
        let s = Self { values };
        if !s.mean().is_finite() || !s.variance().is_finite() {
            return Err(Refusal::NotFinite);
        }
        Ok(s)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn mean(&self) -> f64 {
        self.values.iter().map(|v| v.into_inner()).sum::<f64>() / self.values.len() as f64
    }

    /// Sample variance, Bessel-corrected (n-1). These are samples drawn from a
    /// noisy process, not a whole population, so n-1 is the right denominator
    /// and the difference matters at the small n this tool runs.
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let n = self.values.len() as f64;
        self.values
            .iter()
            .map(|v| {
                let d = v.into_inner() - m;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0)
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Coefficient of variation, the lab's usual way of saying how settled a
    /// number is -- the Phase 1 decode baseline was recorded as 50 samples at
    /// 0.5% CV. Returns `None` when the mean is zero, because a relative
    /// spread around zero is not defined and 0.0 would be a lie.
    pub fn coefficient_of_variation(&self) -> Option<f64> {
        let m = self.mean();
        if m == 0.0 {
            None
        } else {
            Some(self.std_dev() / m.abs())
        }
    }

    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        self.values.iter().map(|v| v.into_inner())
    }
}

/// One metric on one node for one run: measured, refused, or never attempted.
///
/// A flat enum rather than `Option<Sample>` because there are three distinct
/// absences here and `Option` has room for one. Collapsing them forces
/// `Option<Option<..>>` the first time the report needs to say *why* a cell is
/// empty, and "the benchmark never ran" and "the card was too hot to trust"
/// are different facts that a reader will want told apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Measurement {
    Measured(Sample),
    Refused(Refusal),
    /// The benchmark was not attempted -- not scheduled, or the node was down.
    /// Distinct from `Refused`, which means it was attempted and rejected.
    NotRun,
}

impl Measurement {
    /// The sample, if there is one. Deliberately not called `unwrap_or_default`
    /// or anything that could hand back a zero: there is no sensible default
    /// for "we did not measure this".
    pub fn sample(&self) -> Option<&Sample> {
        match self {
            Measurement::Measured(s) => Some(s),
            Measurement::Refused(_) | Measurement::NotRun => None,
        }
    }

    /// A short phrase for a report cell. Every variant answers; there is no
    /// wildcard arm, so a new variant will fail to compile here rather than
    /// quietly print as something else.
    pub fn describe(&self) -> String {
        match self {
            Measurement::Measured(s) => format!("{:.4} (n={})", s.mean(), s.len()),
            Measurement::Refused(r) => format!("refused: {r}"),
            Measurement::NotRun => "not run".to_string(),
        }
    }
}

/// The conditions a measurement was taken under.
///
/// Recorded on every row, not because it is interesting but because a
/// cross-architecture diff is only readable if you can show the two runs
/// differed in architecture and in nothing else. The same argument the
/// kernel harness makes by writing a sidecar `.env.json` next to every result
/// file -- this is that sidecar, as a type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The upstream ref the run was made against.
    pub git_ref: String,
    /// `sm_86` or `sm_120`. The axis the whole lab is built on.
    pub arch: String,
    /// `desktop` or `laptop`.
    pub node: String,
    /// Driver version. Diffing across a mismatch is on the runbook's kill
    /// list, so this is the field that enforces it.
    pub driver: String,
    /// The container image the benchmark ran in.
    pub image_id: String,
}
