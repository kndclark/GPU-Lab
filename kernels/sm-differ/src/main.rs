//! The CLI entry point.
//!
//! Why there is no `--build` subcommand yet, and why there will not be a
//! bollard dependency when there is: the Docker CLI has defaulted to BuildKit
//! since Docker 23.0, while the Engine API's /build endpoint that library
//! clients call is the classic builder, which is deprecated. Driving the API
//! programmatically therefore means this tool's builds can differ from the
//! builds a human gets by running the same Dockerfile by hand. For a tool
//! whose entire product is a trustworthy comparison, a build path that
//! silently diverges from the one everyone else uses is a measurement bug, not
//! an inconvenience -- and this lab has already been bitten by exactly that
//! shape once, when `nvcc -arch` quietly embedded PTX and let mistargeted
//! images JIT instead of failing. So: shell out to `docker`, and get the same
//! builder everyone else gets.

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use sm_differ::diff::{self, Request};
use sm_differ::measure::{Measurement, Sample};
use sm_differ::node::Node;
use sm_differ::prom::Prometheus;
use sm_differ::store::Store;
use sm_differ::verdict::{Gates, Polarity, Verdict, compare};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[derive(Parser)]
#[command(
    name = "sm-differ",
    version,
    about = "Cross-architecture regression differ for the sm_86 / sm_120 lab",
    // Without this clap reads "-10.2" as a flag, so a metric that can go
    // negative -- any delta -- is unreachable from the binary even though the
    // library handles it correctly.
    allow_negative_numbers = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compare one metric across the two architectures and print the verdict.
    ///
    /// This is the tool doing its actual job: it asks each node what driver it
    /// is on, checks the thermal gate over the same window the metric covers,
    /// records both sides -- refusals included -- and only then compares.
    Diff(DiffArgs),

    /// Compare two sets of readings and print the verdict.
    ///
    /// allow_negative_numbers is set here and not only on the parent: in clap 4
    /// the setting does not propagate to subcommands, so without it "-10.2" is
    /// parsed as a flag and a delta metric cannot be expressed at all.
    #[command(allow_negative_numbers = true)]
    ///
    /// Takes literal numbers so the comparison can be exercised without a
    /// database, a node or a benchmark -- the same reason the C4 repro next
    /// door takes a mode argument instead of a fixture.
    Verdict {
        /// Baseline readings, repeated: --baseline 175.9 --baseline 176.1
        #[arg(long = "baseline", required = true, num_args = 1..)]
        baseline: Vec<f64>,
        /// Current readings, repeated.
        #[arg(long = "current", required = true, num_args = 1..)]
        current: Vec<f64>,
        /// Which direction counts as better for this metric.
        #[arg(long, value_enum, default_value_t = PolarityArg::Higher)]
        polarity: PolarityArg,
        /// Significance level for the t-test.
        #[arg(long, default_value_t = 0.05)]
        significance: f64,
        /// Practical-significance floor, as a fraction. 0.01 = ignore changes under 1%.
        #[arg(long, default_value_t = 0.01)]
        noise: f64,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum PolarityArg {
    /// Throughput: tok/s, samples/s.
    Higher,
    /// Latency: TTFT, p95.
    Lower,
}

impl From<PolarityArg> for Polarity {
    fn from(p: PolarityArg) -> Self {
        match p {
            PolarityArg::Higher => Polarity::HigherIsBetter,
            PolarityArg::Lower => Polarity::LowerIsBetter,
        }
    }
}

fn main() -> Result<()> {
    // try_from_default_env rather than from_default_env: the latter is silent
    // when RUST_LOG is unset, which is the common case here.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // Logs to stderr, always. Anything on stdout is a result, so that a verdict
    // can be piped without log lines contaminating it.
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stderr))
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Diff(args) => cmd_diff(args),
        Command::Verdict {
            baseline,
            current,
            polarity,
            significance,
            noise,
        } => {
            // Refusing before a single measurement is read: gates that cannot
            // judge must not be allowed to judge. Until this existed the
            // measurements were guarded by NotNan while the thresholds judging
            // them came straight off the command line unchecked, so
            // `--significance nan` turned a real 20% regression into
            // "unchanged", exit 0 -- a green build for a broken kernel.
            let gates = Gates::new(significance, noise)
                .context("refusing to run with gates that cannot judge")?;
            let n_base = baseline.len();
            let n_curr = current.len();

            // A refusal is a result, not a crash: build the Measurement either
            // way and let compare() reach Broken through the normal path.
            let b = match Sample::new(baseline, n_base) {
                Ok(s) => Measurement::Measured(s),
                Err(r) => Measurement::Refused(r),
            };
            let c = match Sample::new(current, n_curr) {
                Ok(s) => Measurement::Measured(s),
                Err(r) => Measurement::Refused(r),
            };

            let v = compare(&b, &c, polarity.into(), gates);
            print_verdict(&b, &c, &v, gates);

            // Exit code carries the verdict, following the house pattern from
            // the C4 repro. Four distinct codes, because a caller has to be
            // able to tell these apart and only the first is a pass:
            //   0  a conclusive verdict -- Improved, Regressed or Unchanged
            //   1  the gates were refused; nothing was measured
            //   2  a usage error (clap's own)
            //   3  the comparison is Broken -- this run cannot support a
            //      conclusion about the code under test
            // 3 rather than 2 specifically so "you invoked me wrong" and "the
            // data cannot answer" are not the same signal.
            if v.is_conclusive() {
                Ok(())
            } else {
                std::process::exit(3)
            }
        }
    }
}

#[derive(Args)]
#[command(allow_negative_numbers = true)]
struct DiffArgs {
    /// PromQL for the metric under test. It must carry the `arch` label,
    /// which every series in this lab already does.
    #[arg(long)]
    query: String,
    /// Names this measurement is stored under.
    #[arg(long, default_value = "adhoc")]
    bench: String,
    #[arg(long, default_value = "value")]
    metric: String,
    #[arg(long, default_value = "sm_86")]
    baseline: String,
    #[arg(long, default_value = "sm_120")]
    current: String,
    /// How far back to look, in seconds.
    #[arg(long, default_value_t = 600)]
    window: i64,
    /// Resolution in seconds. Asking for a finer step than the scrape
    /// interval does not produce more information.
    #[arg(long, default_value_t = 60)]
    step: u32,
    #[arg(long, value_enum, default_value_t = PolarityArg::Higher)]
    polarity: PolarityArg,
    /// Refuse to record from a node that went above this during the window.
    #[arg(long, default_value_t = 80.0)]
    thermal_limit: f64,
    #[arg(long, default_value = "http://10.10.0.1:9090")]
    prometheus: String,
    /// The other node, reached over ssh. This node is probed locally.
    #[arg(long, default_value = "10.10.0.1")]
    peer: String,
    /// Where results are recorded.
    #[arg(long, default_value = "sm-differ.db")]
    db: std::path::PathBuf,
    /// The upstream ref these numbers belong to.
    #[arg(long, default_value = "unknown")]
    git_ref: String,
    #[arg(long, default_value_t = 0.05)]
    significance: f64,
    #[arg(long, default_value_t = 0.01)]
    noise: f64,
}

fn cmd_diff(args: DiffArgs) -> Result<()> {
    let gates = Gates::new(args.significance, args.noise)
        .context("refusing to run with gates that cannot judge")?;
    let prom = Prometheus::new(&args.prometheus);
    let store = Store::open(&args.db)?;
    let nodes = [Node::local(), Node::ssh(&args.peer)];
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("reading the clock")?
        .as_secs() as i64;

    let req = Request {
        query: &args.query,
        bench: &args.bench,
        metric: &args.metric,
        baseline_arch: &args.baseline,
        current_arch: &args.current,
        window_s: args.window,
        step_s: args.step,
        polarity: args.polarity.into(),
        gates,
        thermal_limit_c: args.thermal_limit,
        git_ref: &args.git_ref,
    };
    let out = diff::run(&prom, &store, &nodes, &req, now)?;

    println!("  query    : {}", args.query);
    println!(
        "  window   : {}s ending {}",
        args.window,
        diff::iso8601(now)
    );
    println!("  gate     : refuse above {:.0} C", args.thermal_limit);
    for (label, s) in [("baseline", &out.baseline), ("current ", &out.current)] {
        println!(
            "  {label} : {:<7} {:<8} driver {}  {}",
            s.provenance.arch,
            s.provenance.node,
            s.provenance.driver,
            s.measurement.describe()
        );
    }
    println!("  recorded : {}", args.db.display());
    println!();
    print_only_verdict(&out.verdict, gates);
    if out.verdict.is_conclusive() {
        Ok(())
    } else {
        std::process::exit(3)
    }
}

fn print_verdict(b: &Measurement, c: &Measurement, v: &Verdict, gates: Gates) {
    print_inputs(b, c, gates);
    print_only_verdict(v, gates);
}

fn print_inputs(b: &Measurement, c: &Measurement, gates: Gates) {
    println!("  baseline : {}", b.describe());
    println!("  current  : {}", c.describe());
    if let (Some(bs), Some(cs)) = (b.sample(), c.sample()) {
        let cv = |s: &Sample| {
            s.coefficient_of_variation()
                .map(|x| format!("{:.2}%", x * 100.0))
                .unwrap_or_else(|| "undefined (mean is zero)".to_string())
        };
        println!("  spread   : baseline CV {}, current CV {}", cv(bs), cv(cs));
    }
    println!(
        "  gates    : p < {}, and |change| > {:.1}%",
        gates.significance_level,
        gates.noise_threshold * 100.0
    );
    println!();
}

/// The verdict half, on its own, so `diff` can print its own inputs and still
/// end the same way `verdict` does.
fn print_only_verdict(v: &Verdict, _gates: Gates) {
    // Exhaustive: a fifth variant would fail to compile here rather than
    // silently print as something else.
    match v {
        Verdict::Improved {
            change_pct,
            p_value,
        } => {
            println!("VERDICT: IMPROVED  {change_pct:+.2}%  (p = {p_value:.4})");
            println!("         Both gates cleared -- the change is larger than the noise floor");
            println!("         and unlikely to be noise.");
        }
        Verdict::Regressed {
            change_pct,
            p_value,
        } => {
            println!("VERDICT: REGRESSED  {change_pct:+.2}%  (p = {p_value:.4})");
            println!("         Both gates cleared -- the change is larger than the noise floor");
            println!("         and unlikely to be noise.");
        }
        Verdict::Unchanged {
            change_pct,
            p_value,
        } => {
            println!("VERDICT: unchanged  ({change_pct:+.2}% observed, p = {p_value:.4})");
            println!("         The observed difference did not clear both gates. This is not a");
            println!("         claim that the two are identical -- only that this run cannot");
            println!("         tell them apart.");
        }
        Verdict::Broken(reason) => {
            println!("VERDICT: BROKEN -- no comparison was possible.");
            println!("         {reason:?}");
            println!("         This is a result about the run, not about the code under test.");
        }
    }
}
