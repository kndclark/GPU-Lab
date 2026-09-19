//! Reading benchmark metrics out of the Prometheus that Phase 1 already runs.
//!
//! The runbook originally specified scraping vLLM's `/metrics` directly, "no
//! Prometheus server in the loop". That was reversed on 2026-09-18: the
//! monitoring stack is already up, already scraping both nodes, and already
//! labels every series with `arch` and `node` -- the exact dimension this tool
//! is built around. Querying it removes an exposition-format parser from the
//! design and leaves the scrape history available afterwards, so a suspicious
//! number can be re-examined rather than only re-measured.
//!
//! **Two things about this API decide the shape of everything below.**
//!
//! 1. **Sample values arrive as JSON strings, not numbers** -- `[1789777366,
//!    "1"]`. Prometheus does this deliberately, because it has to transmit
//!    `NaN` and `+Inf`, which JSON numbers cannot represent. So every value
//!    passes through a string parse, and that parse is the exact place a
//!    careless `unwrap_or(0.0)` would reintroduce the silent zero this crate
//!    exists to prevent. It does not appear anywhere in this file.
//! 2. **Errors come back as HTTP 400 with a JSON body**, verified against the
//!    lab's own server: a malformed query returns
//!    `{"status":"error","errorType":"bad_data","error":"..."}` with status
//!    400. ureq treats 4xx as an `Err` by default, which would throw that
//!    message away and report a bare status code, so status-as-error is turned
//!    off and the body is read either way.

use crate::measure::{Refusal, Sample};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::BTreeMap;

/// A client for one Prometheus server.
pub struct Prometheus {
    agent: ureq::Agent,
    base: String,
}

/// One time series: its labels, and the samples that came back for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub labels: BTreeMap<String, String>,
    /// (unix seconds, value). Values are already parsed, and a value that
    /// could not be parsed never reaches here -- see `Raw::parse`.
    pub samples: Vec<(f64, f64)>,
}

impl Series {
    /// The label Prometheus is already tagging every lab series with.
    pub fn arch(&self) -> Option<&str> {
        self.labels.get("arch").map(String::as_str)
    }

    pub fn node(&self) -> Option<&str> {
        self.labels.get("node").map(String::as_str)
    }

    /// Turn the samples into a `Sample`, or say why not.
    ///
    /// This goes through `Sample::new`, which is the whole point: a series of
    /// `NaN`s -- which is what Prometheus returns for a metric that existed
    /// but had no data -- becomes `Refusal::NotFinite`, not a column of zeros.
    pub fn into_sample(self, wanted: usize) -> Result<Sample, Refusal> {
        Sample::new(self.samples.into_iter().map(|(_, v)| v), wanted)
    }
}

// --- the wire format -------------------------------------------------------

#[derive(Deserialize)]
#[serde(tag = "status")]
enum Envelope {
    #[serde(rename = "success")]
    Success { data: Data },
    #[serde(rename = "error")]
    Error {
        #[serde(rename = "errorType")]
        error_type: String,
        error: String,
    },
}

#[derive(Deserialize)]
struct Data {
    #[serde(rename = "resultType")]
    result_type: String,
    result: Vec<RawSeries>,
}

#[derive(Deserialize)]
struct RawSeries {
    #[serde(default)]
    metric: BTreeMap<String, String>,
    #[serde(default)]
    values: Vec<Raw>,
}

/// One sample as Prometheus sends it: a numeric timestamp and a *string* value.
#[derive(Deserialize)]
struct Raw(f64, String);

impl Raw {
    /// Parse the value, refusing to invent one.
    ///
    /// A value that is not a number at all is a protocol violation and is an
    /// error. A value that parses to NaN or an infinity is *not* -- those are
    /// legitimate Prometheus values, and they are passed through as the
    /// numbers they are so that `Sample::new` can refuse them by name further
    /// in. Neither case is ever silently replaced.
    fn parse(&self) -> Result<(f64, f64)> {
        let v: f64 = self
            .1
            .parse()
            .with_context(|| format!("Prometheus returned {:?}, which is not a number", self.1))?;
        Ok((self.0, v))
    }
}

// --- the client ------------------------------------------------------------

impl Prometheus {
    /// `base` is the server root, e.g. `http://10.10.0.1:9090`.
    pub fn new(base: &str) -> Self {
        let config = ureq::Agent::config_builder()
            // Read the body on a 4xx: Prometheus puts its diagnostic there and
            // a bare "400" tells the operator nothing they can act on.
            .http_status_as_error(false)
            .build();
        Self {
            agent: config.new_agent(),
            base: base.trim_end_matches('/').to_string(),
        }
    }

    /// `/api/v1/query_range`, the endpoint that returns a series over time.
    ///
    /// `start` and `end` are unix seconds; `step` is the resolution in
    /// seconds. The step bounds what this can see: asking for a finer step
    /// than the scrape interval does not produce more information.
    pub fn query_range(&self, query: &str, start: i64, end: i64, step: u32) -> Result<Vec<Series>> {
        let url = format!("{}/api/v1/query_range", self.base);
        let mut resp = self
            .agent
            .get(&url)
            .query("query", query)
            .query("start", start.to_string())
            .query("end", end.to_string())
            .query("step", step.to_string())
            .call()
            .with_context(|| format!("querying {url} for {query:?}"))?;

        let body = resp
            .body_mut()
            .read_to_string()
            .context("reading the Prometheus response body")?;
        parse_range_response(&body)
            .with_context(|| format!("Prometheus rejected or malformed the query {query:?}"))
    }
}

/// Split out from the client so it can be tested against recorded fixtures
/// without a server. Every shape asserted here was captured from the lab's own
/// Prometheus rather than written from the documentation.
pub fn parse_range_response(body: &str) -> Result<Vec<Series>> {
    let env: Envelope =
        serde_json::from_str(body).context("Prometheus response was not the expected JSON")?;
    let data = match env {
        Envelope::Success { data } => data,
        Envelope::Error { error_type, error } => bail!("Prometheus error [{error_type}]: {error}"),
    };
    if data.result_type != "matrix" {
        // query_range always returns a matrix. Anything else means the caller
        // reached the wrong endpoint, and guessing at the shape would turn a
        // clear mistake into a confusing one.
        bail!(
            "expected resultType \"matrix\" from query_range, got {:?}",
            data.result_type
        );
    }
    data.result
        .into_iter()
        .map(|s| {
            Ok(Series {
                labels: s.metric,
                samples: s
                    .values
                    .iter()
                    .map(Raw::parse)
                    .collect::<Result<Vec<_>>>()?,
            })
        })
        .collect()
}
