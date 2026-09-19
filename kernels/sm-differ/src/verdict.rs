//! Turning two measurements into one of four answers.
//!
//! The entire product of this tool is a `Verdict`. Everything else -- building
//! images, driving nodes, storing rows -- is plumbing that exists to produce
//! one honestly.
//!
//! **Two gates, not a threshold.** A difference is reported only when it is
//! both statistically distinguishable from noise *and* large enough to care
//! about. This is Criterion's structure and it is adopted rather than invented,
//! because "never report a regression that is actually noise" is exactly the
//! problem Criterion already solved: it gates on a significance test at
//! p < 0.05 and, separately, on a practical-significance floor of 1%. Either
//! gate alone is wrong. Significance alone will flag a 0.1% change as a
//! regression once the sample count is high enough; a bare percentage floor
//! alone will flag thermal weather on a 175 W laptop as a kernel change.
//!
//! **A missing measurement is never a number.** If either side is absent the
//! answer is `Broken` and no arithmetic happens at all -- see `compare`.

use crate::measure::{Measurement, Provenance};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Which direction is good for a given metric.
///
/// Not inferable from the numbers, and getting it backwards would invert every
/// verdict silently. Decode throughput in tok/s is better when it rises; TTFT
/// and p95 latency are better when they fall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarity {
    HigherIsBetter,
    LowerIsBetter,
}

/// The thresholds a comparison is judged against, together in one place so a
/// report can print the rules it applied alongside the answer it reached.
///
/// `try_from` is not decoration. `Gates` has the same shape that produced the
/// worst defect in this crate -- a fallible constructor alongside a derived
/// `Deserialize` that walks straight past it. Found by searching for the twin
/// after fixing it on `Sample`: without this, a stored or piped config of
/// `{"significance_level": 1.5, "noise_threshold": -1.0}` would rebuild
/// exactly the gates `Gates::new` exists to reject, and every comparison made
/// with them would be nonsense reported as fact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "GatesWire")]
pub struct Gates {
    /// Significance level for the t-test. Criterion's default, and the
    /// conventional 0.05: about 5% of identical benchmarks will register as
    /// different by chance alone.
    pub significance_level: f64,
    /// Practical-significance floor as a fraction. Criterion's default 0.01
    /// means relative changes under 1% are ignored even when statistically
    /// significant.
    pub noise_threshold: f64,
}

impl Default for Gates {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            noise_threshold: 0.01,
        }
    }
}

/// The on-the-wire shape of `Gates`, and the only way one can be deserialized.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GatesWire {
    significance_level: f64,
    noise_threshold: f64,
}

impl TryFrom<GatesWire> for Gates {
    type Error = GateError;
    fn try_from(w: GatesWire) -> Result<Self, Self::Error> {
        Gates::new(w.significance_level, w.noise_threshold)
    }
}

/// Why a set of gates was rejected.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GateError {
    #[error("significance level must be between 0 and 1, exclusive; got {0}")]
    Significance(f64),
    #[error("noise threshold must be finite and >= 0; got {0}")]
    Noise(f64),
}

impl Gates {
    /// The only way to build gates from untrusted numbers.
    ///
    /// The measurements in this crate are guarded by `NotNan`; until this
    /// existed the *thresholds that judge them* were raw f64 taken straight
    /// from the command line. Found by attacking the crate, and the
    /// consequences were worse than the hole it closes elsewhere:
    /// `--significance nan` made every comparison fail the significance gate
    /// silently, so a real 20% throughput regression printed "unchanged" and
    /// exited 0 -- a green build for a broken kernel. `--noise=-1` did the
    /// opposite, manufacturing an `IMPROVED` verdict out of a +0.00% change.
    pub fn new(significance_level: f64, noise_threshold: f64) -> Result<Self, GateError> {
        if !(significance_level.is_finite() && significance_level > 0.0 && significance_level < 1.0)
        {
            return Err(GateError::Significance(significance_level));
        }
        if !(noise_threshold.is_finite() && noise_threshold >= 0.0) {
            return Err(GateError::Noise(noise_threshold));
        }
        Ok(Self {
            significance_level,
            noise_threshold,
        })
    }
}

/// Why no comparison was possible. Kept separate from `Refusal` because these
/// are facts about the *pair*, not about either measurement on its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Broken {
    BaselineMissing(String),
    CurrentMissing(String),
    BothMissing,
    /// The baseline mean is zero, so a relative change is not defined. Reported
    /// rather than papered over with a zero or an infinity.
    BaselineIsZero,

    /// The baseline mean is not distinguishable from zero given its own
    /// spread, so a *relative* change is meaningless even though it is
    /// arithmetically computable.
    ///
    /// Without this, the practical-significance gate quietly switches off near
    /// zero: a baseline of 5e-8 against a current of 1.0 yields a change of
    /// +2,000,099,900%, which clears any percentage floor ever set. Refusing a
    /// single exact zero and admitting everything either side of it was a
    /// discontinuity with nothing behind it.
    BaselineIndistinguishableFromZero {
        mean: f64,
        std_dev: f64,
    },

    /// Neither sample varied, so there is no way to estimate noise, and
    /// without that a difference cannot be told from an artefact of the
    /// instrument.
    ///
    /// This is the fix for the worst defect found while attacking this crate.
    /// The code previously answered `p = 0.0` here -- maximal confidence --
    /// which inverted the tool's purpose: a *coarser* instrument made it
    /// *more* certain. Latencies of 12,12,12 against 13,13,13 from a
    /// millisecond timer were reported `REGRESSED (p = 0.0000)`, while the
    /// same physical latencies measured finely (12.3,11.8,12.4 against
    /// 12.6,13.4,12.9) came back `unchanged, p = 0.058`. Throwing away jitter
    /// information must never buy confidence. Integer MiB from nvidia-smi and
    /// millisecond timers make this ordinary data, not an edge case.
    NoiseNotEstimable {
        flat_at: f64,
        versus: f64,
    },

    /// The two runs were taken under different drivers, so any difference
    /// between them is unattributable.
    ///
    /// "Diffing across a driver mismatch" is on the runbook's kill list, and
    /// the reason is in this lab's own history: the two nodes were once 20
    /// driver releases apart, and the container layer cannot pin the host.
    /// Refusing is the only honest answer -- both numbers are real, but
    /// nothing can be concluded from their difference.
    DriverMismatch {
        baseline: String,
        current: String,
    },
}

/// The answer. Four variants, matched exhaustively everywhere, and
/// deliberately NOT `#[non_exhaustive]`: that attribute has no effect inside
/// the defining crate and, across a crate boundary, would force callers to
/// write a wildcard arm -- destroying the compile error that is the whole
/// reason this is an enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    Improved {
        change_pct: f64,
        p_value: f64,
    },
    Regressed {
        change_pct: f64,
        p_value: f64,
    },
    /// Includes the numbers that *failed* to clear the gates, so a reader can
    /// see how close a call it was instead of being told only "nothing here".
    Unchanged {
        change_pct: f64,
        p_value: f64,
    },
    Broken(Broken),
}

impl Verdict {
    /// One word for a table cell. No wildcard arm, on purpose.
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::Improved { .. } => "IMPROVED",
            Verdict::Regressed { .. } => "REGRESSED",
            Verdict::Unchanged { .. } => "unchanged",
            Verdict::Broken(_) => "BROKEN",
        }
    }

    /// True only for a verdict backed by two real measurements. `Broken` is
    /// not a result about the code under test -- it is a result about the run.
    pub fn is_conclusive(&self) -> bool {
        !matches!(self, Verdict::Broken(_))
    }
}

/// Welch's t-test for two samples of unequal variance, returning the
/// two-sided p-value.
///
/// Welch rather than Student's because the two nodes are a 400 W desktop and a
/// 175 W laptop: assuming equal variance between them is exactly the
/// assumption this lab has no right to make.
///
/// Criterion does this by bootstrap resampling instead, which makes no
/// distributional assumption at all. This is the analytic approximation, and
/// the trade is stated rather than hidden: it is simpler and deterministic,
/// but it assumes roughly normal sample means, which at n >= 20 is usually
/// reasonable and at n = 2 is not. Revisit if runs settle on small n.
pub(crate) enum Significance {
    P(f64),
    /// The observed spread is zero on both sides, so noise cannot be
    /// estimated. Not "certainly different" and not "certainly the same" --
    /// unknowable from this data.
    Unestimable,
}

fn welch_p_value(a: &crate::measure::Sample, b: &crate::measure::Sample) -> Significance {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (va, vb) = (a.variance() / na, b.variance() / nb);
    let se = (va + vb).sqrt();

    // Guard before dividing: t would be 0/0 or x/0 here. Note the condition is
    // `se == 0.0`, not "both samples are flat" -- squaring deviations can
    // underflow to zero on data that genuinely varies, and this arm has to be
    // correct for that case too.
    if se == 0.0 {
        // Equal means and no spread is the one honest answer available: there
        // is no difference to explain, so nothing is significant.
        return if a.mean() == b.mean() {
            Significance::P(1.0)
        } else {
            Significance::Unestimable
        };
    }

    let t = (a.mean() - b.mean()) / se;

    // Welch-Satterthwaite degrees of freedom.
    let df = (va + vb).powi(2) / (va.powi(2) / (na - 1.0) + vb.powi(2) / (nb - 1.0));
    if !df.is_finite() || df <= 0.0 {
        return Significance::Unestimable;
    }

    match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => Significance::P(2.0 * (1.0 - dist.cdf(t.abs()))),
        // Refusing to guess. An unconstructable distribution means the inputs
        // were degenerate, and saying so beats inventing a number.
        Err(_) => Significance::Unestimable,
    }
}

/// Compare a baseline measurement against a current one.
///
/// Order matters: `baseline` is the earlier run, `current` the newer one, and
/// `change_pct` is signed relative to the baseline.
pub fn compare(
    baseline: &Measurement,
    current: &Measurement,
    polarity: Polarity,
    gates: Gates,
) -> Verdict {
    // Gate zero, before any arithmetic: both sides must actually exist. This
    // is the branch the whole crate is built around -- there is no path here
    // that substitutes a default for a missing measurement.
    let (b, c) = match (baseline.sample(), current.sample()) {
        (Some(b), Some(c)) => (b, c),
        (None, None) => return Verdict::Broken(Broken::BothMissing),
        (None, Some(_)) => {
            return Verdict::Broken(Broken::BaselineMissing(baseline.describe()));
        }
        (Some(_), None) => {
            return Verdict::Broken(Broken::CurrentMissing(current.describe()));
        }
    };

    let bm = b.mean();
    if bm == 0.0 {
        return Verdict::Broken(Broken::BaselineIsZero);
    }
    // A relative change only means something if the baseline is itself
    // distinguishable from zero. One sigma is the test: a baseline whose mean
    // is smaller than its own spread cannot anchor a percentage. Refusing an
    // exact zero and admitting everything either side of it was a
    // discontinuity with nothing behind it -- a baseline of 5e-8 against a
    // current of 1.0 produced +2,000,099,900%, which clears any floor.
    let b_sd = b.std_dev();
    if bm.abs() <= b_sd {
        return Verdict::Broken(Broken::BaselineIndistinguishableFromZero {
            mean: bm,
            std_dev: b_sd,
        });
    }

    let change_pct = (c.mean() - bm) / bm.abs() * 100.0;

    // Gate 1: is the difference distinguishable from noise? An unestimable
    // significance is not a failed gate -- it means the question cannot be
    // answered from this data at all, which is a third outcome and is reported
    // as one rather than rounded into "no change".
    let p_value = match welch_p_value(b, c) {
        Significance::P(p) => p,
        Significance::Unestimable => {
            return Verdict::Broken(Broken::NoiseNotEstimable {
                flat_at: bm,
                versus: c.mean(),
            });
        }
    };
    let significant = p_value < gates.significance_level;
    // Gate 2: is it big enough to care about?
    let practical = (change_pct / 100.0).abs() > gates.noise_threshold;

    // Exactly zero movement is never a direction, whatever the gates say.
    // Without this, a noise_threshold of 0 lets change_pct == 0.0 fall through
    // to `better = 0.0 > 0.0 == false` and be announced as a regression -- a
    // finding manufactured out of no change at all.
    if !(significant && practical) || change_pct == 0.0 {
        return Verdict::Unchanged {
            change_pct,
            p_value,
        };
    }

    let better = match polarity {
        Polarity::HigherIsBetter => change_pct > 0.0,
        Polarity::LowerIsBetter => change_pct < 0.0,
    };
    if better {
        Verdict::Improved {
            change_pct,
            p_value,
        }
    } else {
        Verdict::Regressed {
            change_pct,
            p_value,
        }
    }
}

/// Compare two measurements that carry the conditions they were taken under.
///
/// This is what anything reading from the database should call. `compare`
/// will happily compare two numbers from different drivers because it cannot
/// see the drivers; this refuses first, then delegates.
pub fn compare_recorded(
    baseline: (&Provenance, &Measurement),
    current: (&Provenance, &Measurement),
    polarity: Polarity,
    gates: Gates,
) -> Verdict {
    let (bp, bm) = baseline;
    let (cp, cm) = current;
    if bp.driver != cp.driver {
        return Verdict::Broken(Broken::DriverMismatch {
            baseline: bp.driver.clone(),
            current: cp.driver.clone(),
        });
    }
    compare(bm, cm, polarity, gates)
}
