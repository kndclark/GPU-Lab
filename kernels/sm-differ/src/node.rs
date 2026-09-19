//! Asking a node what it is.
//!
//! Provenance is not decoration. The runbook requires a driver version on
//! every row and refuses to diff across a mismatch, and Prometheus does not
//! carry one -- checked: there is no `driver` label anywhere in the lab's
//! series. So the node has to be asked directly, and that is what this does.
//!
//! **Compute capability is the identity, not the hostname.** This follows
//! `bin/lab`, which says why: it is the axis the whole lab is built on, it
//! cannot drift, and it still answers when the direct link is down. An
//! unrecognised capability is a hard error naming the two cards this lab
//! knows, rather than a guess.

use anyhow::{Context, Result, bail};
use std::process::Command;

/// How to reach a node. Local means this machine; remote shells out to ssh,
/// which is what two known hosts need rather than a protocol.
#[derive(Debug, Clone)]
pub enum Reach {
    Local,
    Ssh { host: String },
}

#[derive(Debug, Clone)]
pub struct Node {
    pub reach: Reach,
}

/// What a node reports about itself, at the moment it was asked.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeInfo {
    /// `sm_86` or `sm_120`, derived from compute capability.
    pub arch: String,
    /// `desktop` or `laptop`, derived the same way -- see the module comment.
    pub node: String,
    pub driver: String,
    pub gpu: String,
    pub temperature_c: f64,
}

impl Node {
    pub fn local() -> Self {
        Self {
            reach: Reach::Local,
        }
    }

    pub fn ssh(host: &str) -> Self {
        Self {
            reach: Reach::Ssh {
                host: host.to_string(),
            },
        }
    }

    /// Ask the node what it is. One nvidia-smi call, one line of CSV.
    pub fn probe(&self) -> Result<NodeInfo> {
        const QUERY: &str = "nvidia-smi --query-gpu=driver_version,name,compute_cap,temperature.gpu \
             --format=csv,noheader";

        let out = match &self.reach {
            Reach::Local => run(Command::new("sh").arg("-c").arg(QUERY))?,
            Reach::Ssh { host } => run(Command::new("ssh")
                // BatchMode so a missing key fails immediately instead of
                // hanging a benchmark run on a password prompt nobody sees.
                .args([
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    host,
                    QUERY,
                ]))?,
        };

        parse_nvidia_smi_output(&out)
    }
}

/// Parse the CSV output from `nvidia-smi --query-gpu=driver_version,name,
/// compute_cap,temperature.gpu --format=csv,noheader`.
///
/// Separated from `Node::probe` so the parsing can be tested without shelling
/// out to nvidia-smi or ssh.
pub fn parse_nvidia_smi_output(out: &str) -> Result<NodeInfo> {
    let line = out
        .lines()
        .next()
        .filter(|l| !l.trim().is_empty())
        .context("nvidia-smi returned nothing")?;
    let f: Vec<&str> = line.split(',').map(str::trim).collect();
    if f.len() < 4 {
        bail!("unrecognised nvidia-smi output: {line:?}");
    }

    // Match bin/lab: capability decides, and the dot is stripped so 8.6
    // becomes 86. An unknown card is refused by name.
    let cap = f[2].replace('.', "");
    let (arch, node) = match cap.as_str() {
        "86" => ("sm_86", "desktop"),
        "120" => ("sm_120", "laptop"),
        other => bail!(
            "unrecognised compute cap {other:?} -- this harness knows only \
             the lab's two cards (8.6 desktop, 12.0 laptop)"
        ),
    };

    Ok(NodeInfo {
        arch: arch.to_string(),
        node: node.to_string(),
        driver: f[0].to_string(),
        gpu: f[1].to_string(),
        temperature_c: f[3]
            .parse()
            .with_context(|| format!("temperature {:?} is not a number", f[3]))?,
    })
}

/// Run a command and insist on a clean exit. A non-zero status carries
/// stderr, because "the probe failed" without the reason is the kind of
/// message that sends an operator to go and run it by hand.
fn run(cmd: &mut Command) -> Result<String> {
    let out = cmd.output().context("spawning the probe")?;
    if !out.status.success() {
        bail!(
            "probe exited {}: {}",
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}
