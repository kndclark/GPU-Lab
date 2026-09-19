//! The product: two architectures, one metric, one verdict.
//!
//! Everything else in this crate is a part. This is the thing that uses them,
//! and the order matters more than it looks:
//!
//! 1. **Ask each node what it is**, because a driver version is not in
//!    Prometheus and a row without provenance cannot be refused later.
//! 2. **Check the thermal gate before looking at the metric.** The runbook
//!    bans recording from a throttled node, so a node that got too hot during
//!    the window produces a refusal rather than a number -- and the number is
//!    not even fetched, because having it in hand is how it ends up being used
//!    "just this once".
//! 3. **Record both sides, including the refusals.** A refusal is a durable
//!    fact about a run, not an absence of one.
//! 4. **Then compare**, through the path that refuses a driver mismatch.

use crate::measure::{Measurement, Provenance, Refusal, Sample};
use crate::node::Node;
use crate::prom::Prometheus;
use crate::store::Store;
use crate::verdict::{Gates, Polarity, Verdict, compare_recorded};
use anyhow::{Context, Result, bail};

/// What one side of the comparison needs.
pub struct Side {
    pub provenance: Provenance,
    pub measurement: Measurement,
}

pub struct Request<'a> {
    /// PromQL for the metric under test. It must carry the `arch` label, which
    /// every series in this lab already does.
    pub query: &'a str,
    pub bench: &'a str,
    pub metric: &'a str,
    pub baseline_arch: &'a str,
    pub current_arch: &'a str,
    pub window_s: i64,
    pub step_s: u32,
    pub polarity: Polarity,
    pub gates: Gates,
    pub thermal_limit_c: f64,
    pub git_ref: &'a str,
}

/// The temperature series every node in this lab already publishes. Hard-coded
/// rather than a flag: the thermal gate is a rule, not a preference, and a
/// caller who could point it at a different metric could point it at one that
/// never trips.
const TEMPERATURE_QUERY: &str = "llamaswap_gpu_temperature_celsius";

pub struct Outcome {
    pub baseline: Side,
    pub current: Side,
    pub verdict: Verdict,
}

/// Gather both sides, record them, and compare.
pub fn run(
    prom: &Prometheus,
    store: &Store,
    nodes: &[Node],
    req: &Request<'_>,
    now: i64,
) -> Result<Outcome> {
    let start = now - req.window_s;

    // 1. identity, from the nodes themselves
    let mut infos = Vec::new();
    for n in nodes {
        infos.push(
            n.probe()
                .context("probing a node for its driver and arch")?,
        );
    }
    let find = |arch: &str| {
        infos
            .iter()
            .find(|i| i.arch == arch)
            .with_context(|| format!("no reachable node reports arch {arch}"))
    };

    // 2. the thermal gate, over the same window the metric covers
    let temps = prom
        .query_range(TEMPERATURE_QUERY, start, now, req.step_s)
        .context("reading the thermal gate")?;
    let hottest = |arch: &str| -> Option<f64> {
        temps
            .iter()
            .filter(|s| s.arch() == Some(arch))
            .flat_map(|s| s.samples.iter().map(|(_, v)| *v))
            // f64 has no Ord; total_cmp gives a total order over the samples.
            // It is deliberately not mixed with any `<` comparison elsewhere,
            // because the two are documented as inconsistent.
            .max_by(|a, b| a.total_cmp(b))
    };

    // 3. the metric itself
    let series = prom
        .query_range(req.query, start, now, req.step_s)
        .context("reading the metric under test")?;

    let side = |arch: &str| -> Result<Side> {
        let info = find(arch)?;
        let provenance = Provenance {
            git_ref: req.git_ref.to_string(),
            arch: info.arch.clone(),
            node: info.node.clone(),
            driver: info.driver.clone(),
            image_id: String::new(),
        };

        // The gate first, and the metric is not consulted if it trips.
        let measurement = match hottest(arch) {
            Some(peak) if peak > req.thermal_limit_c => Measurement::Refused(Refusal::TooHot {
                observed_c: peak,
                threshold_c: req.thermal_limit_c,
            }),
            // No temperature reading at all is not permission to proceed. The
            // gate exists to be passed, not skipped when it is quiet.
            None => Measurement::Refused(Refusal::EnvironmentMismatch(format!(
                "no {TEMPERATURE_QUERY} samples for {arch} in the window; \
                 the thermal gate could not be checked"
            ))),
            Some(_) => match series.iter().find(|s| s.arch() == Some(arch)) {
                None => Measurement::NotRun,
                Some(s) => {
                    let values: Vec<f64> = s.samples.iter().map(|(_, v)| *v).collect();
                    // `wanted` is the number of scrapes the window should have
                    // produced, so a series with holes in it is reported as the
                    // shortfall it is rather than quietly averaged.
                    let wanted = (req.window_s / req.step_s.max(1) as i64).max(2) as usize;
                    match Sample::new(values, wanted) {
                        Ok(sample) => Measurement::Measured(sample),
                        Err(refusal) => Measurement::Refused(refusal),
                    }
                }
            },
        };
        Ok(Side {
            provenance,
            measurement,
        })
    };

    let baseline = side(req.baseline_arch)?;
    let current = side(req.current_arch)?;
    if baseline.provenance.arch == current.provenance.arch {
        bail!(
            "baseline and current are both {} -- there is nothing to compare",
            baseline.provenance.arch
        );
    }

    // 4. record, refusals included
    let started = iso8601(now);
    for s in [&baseline, &current] {
        let run_id = store.begin_run(&s.provenance, &started)?;
        store.record(run_id, req.bench, req.metric, &s.measurement)?;
    }

    let verdict = compare_recorded(
        (&baseline.provenance, &baseline.measurement),
        (&current.provenance, &current.measurement),
        req.polarity,
        req.gates,
    );
    Ok(Outcome {
        baseline,
        current,
        verdict,
    })
}

/// Unix seconds to an ISO-8601 UTC stamp, matching the `started_utc` the
/// kernel harness writes beside every result file.
pub fn iso8601(unix: i64) -> String {
    // Civil-from-days, Howard Hinnant's algorithm. A date library would be a
    // dependency earned by one call site.
    let (days, secs) = (unix.div_euclid(86_400), unix.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = era * 400 + yoe + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}
