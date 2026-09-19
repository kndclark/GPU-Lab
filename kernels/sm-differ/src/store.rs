//! Where runs are recorded, and read back.
//!
//! One SQLite file, two tables: a `run` is a (ref, arch, node, driver) the
//! benchmarks were taken under, and a `measurement` is one metric from one
//! benchmark within it.
//!
//! **A measurement is stored as JSON, not as a column of numbers.** That is
//! the load-bearing decision here. A column-per-field schema has to invent a
//! value for "the card was too hot" -- usually NULL, which the next reader
//! coerces to zero, which is the failure this crate exists to prevent. Storing
//! the `Measurement` enum whole means an absence is stored as an absence, and
//! it comes back through the same validating path a live reading takes, so a
//! row written by an older version cannot re-enter the system unchecked.

use crate::measure::{Measurement, Provenance};
use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use rusqlite_migration::{M, Migrations};

/// Schema history. Append only -- never edit a migration that has shipped,
/// because `user_version` records only how many ran, not which.
const MIGRATIONS_SLICE: &[M<'_>] = &[M::up(
    "CREATE TABLE run (
         id          INTEGER PRIMARY KEY,
         git_ref     TEXT NOT NULL,
         arch        TEXT NOT NULL,
         node        TEXT NOT NULL,
         driver      TEXT NOT NULL,
         image_id    TEXT NOT NULL,
         started_utc TEXT NOT NULL
     );
     CREATE TABLE measurement (
         id      INTEGER PRIMARY KEY,
         run_id  INTEGER NOT NULL REFERENCES run(id),
         bench   TEXT NOT NULL,
         metric  TEXT NOT NULL,
         -- the Measurement enum as JSON; see the module comment
         value   TEXT NOT NULL,
         UNIQUE(run_id, bench, metric)
     );
     CREATE INDEX measurement_lookup ON measurement(bench, metric);",
)];

const MIGRATIONS: Migrations<'_> = Migrations::from_slice(MIGRATIONS_SLICE);

pub struct Store {
    conn: Connection,
}

/// A measurement with the conditions it was taken under. Nothing in this crate
/// compares two measurements without their provenance, because two numbers
/// from different drivers are not comparable and the type should say so.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorded {
    pub provenance: Provenance,
    pub started_utc: String,
    pub measurement: Measurement,
}

impl Store {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("opening the results database at {}", path.display()))?;
        Self::prepare(conn)
    }

    /// For tests, and for a dry run that should leave nothing behind.
    pub fn in_memory() -> Result<Self> {
        Self::prepare(Connection::open_in_memory()?)
    }

    fn prepare(mut conn: Connection) -> Result<Self> {
        // Foreign keys are off by default in SQLite, which would make the
        // REFERENCES clause above decorative.
        conn.pragma_update(None, "foreign_keys", "ON")?;
        MIGRATIONS
            .to_latest(&mut conn)
            .context("applying the schema migrations")?;
        Ok(Self { conn })
    }

    /// Record the conditions of a run and return its id.
    pub fn begin_run(&self, p: &Provenance, started_utc: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO run (git_ref, arch, node, driver, image_id, started_utc)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![p.git_ref, p.arch, p.node, p.driver, p.image_id, started_utc],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Record one metric. A refusal is recorded exactly like a reading is --
    /// that is the point. Re-recording the same metric for the same run
    /// replaces it, so a re-run does not silently accumulate duplicates that a
    /// later query would average.
    pub fn record(&self, run_id: i64, bench: &str, metric: &str, m: &Measurement) -> Result<()> {
        let json = serde_json::to_string(m).context("serialising a measurement")?;
        self.conn.execute(
            "INSERT INTO measurement (run_id, bench, metric, value) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(run_id, bench, metric) DO UPDATE SET value = excluded.value",
            params![run_id, bench, metric, json],
        )?;
        Ok(())
    }

    /// The most recent recording of one metric on one architecture, with the
    /// conditions it was taken under.
    pub fn latest(&self, arch: &str, bench: &str, metric: &str) -> Result<Option<Recorded>> {
        self.conn
            .query_row(
                "SELECT r.git_ref, r.arch, r.node, r.driver, r.image_id, r.started_utc, m.value
                   FROM measurement m JOIN run r ON r.id = m.run_id
                  WHERE r.arch = ?1 AND m.bench = ?2 AND m.metric = ?3
                  ORDER BY r.started_utc DESC, r.id DESC
                  LIMIT 1",
                params![arch, bench, metric],
                |row| {
                    Ok((
                        Provenance {
                            git_ref: row.get(0)?,
                            arch: row.get(1)?,
                            node: row.get(2)?,
                            driver: row.get(3)?,
                            image_id: row.get(4)?,
                        },
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                },
            )
            .optional()?
            .map(|(provenance, started_utc, json)| {
                // Through the validating path, deliberately. A row written by
                // an older version of this tool gets exactly the scrutiny a
                // live reading gets, and a row that cannot pass it is an error
                // rather than a silently degraded measurement.
                let measurement: Measurement = serde_json::from_str(&json).with_context(|| {
                    format!("stored measurement for {bench}/{metric} on {arch} is not readable")
                })?;
                Ok(Recorded {
                    provenance,
                    started_utc,
                    measurement,
                })
            })
            .transpose()
    }

    /// Write a raw JSON value, bypassing serialisation.
    ///
    /// This exists so the tests can plant the kind of row an older version of
    /// this tool would have written, and prove the read path still refuses it.
    /// It is named for what it does rather than something reassuring, because
    /// the whole argument for the JSON column is that reads are validated --
    /// and a helper that quietly undid that on the write side would hollow it
    /// out.
    #[doc(hidden)]
    pub fn overwrite_raw_for_tests(
        &self,
        run_id: i64,
        bench: &str,
        metric: &str,
        json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO measurement (run_id, bench, metric, value) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(run_id, bench, metric) DO UPDATE SET value = excluded.value",
            params![run_id, bench, metric, json],
        )?;
        Ok(())
    }

    /// Every recording of one metric on one architecture, newest first.
    pub fn history(&self, arch: &str, bench: &str, metric: &str) -> Result<Vec<Recorded>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.git_ref, r.arch, r.node, r.driver, r.image_id, r.started_utc, m.value
               FROM measurement m JOIN run r ON r.id = m.run_id
              WHERE r.arch = ?1 AND m.bench = ?2 AND m.metric = ?3
              ORDER BY r.started_utc DESC, r.id DESC",
        )?;
        let rows = stmt
            .query_map(params![arch, bench, metric], |row| {
                Ok((
                    Provenance {
                        git_ref: row.get(0)?,
                        arch: row.get(1)?,
                        node: row.get(2)?,
                        driver: row.get(3)?,
                        image_id: row.get(4)?,
                    },
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        rows.into_iter()
            .map(|(provenance, started_utc, json)| {
                let measurement: Measurement =
                    serde_json::from_str(&json).context("reading a stored measurement")?;
                Ok(Recorded {
                    provenance,
                    started_utc,
                    measurement,
                })
            })
            .collect()
    }
}
