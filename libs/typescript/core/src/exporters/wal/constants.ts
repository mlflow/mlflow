/**
 * Tunable defaults for the WAL exporter and uploader daemon.
 *
 * All values can be overridden by environment variables. Env vars are read
 * at module load time, so changes take effect the next time the process
 * that imports this module starts (e.g. the next Stop hook invocation, or
 * the next daemon respawn after an idle exit).
 */

function readPositiveInt(envName: string, fallback: number): number {
  const raw = process.env[envName];
  if (raw === undefined || raw === '') {
    return fallback;
  }
  // `Number` (vs `Number.parseInt`) refuses trailing garbage and unit
  // suffixes — e.g. "15000abc", "15s", and "10.9" all become NaN or a
  // non-integer, so the `isInteger` check below rejects them cleanly.
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    console.warn(
      `[mlflow][wal] Ignoring invalid ${envName}=${JSON.stringify(raw)}; ` +
        `expected a positive integer. Falling back to default ${fallback}.`,
    );
    return fallback;
  }
  return parsed;
}

/**
 * Like {@link readPositiveInt} but accepts `0` as a valid value. Used for
 * knobs where `0` carries the semantic of "disabled" (e.g. retention sweep
 * off) rather than "invalid".
 */
function readNonNegativeInt(envName: string, fallback: number): number {
  const raw = process.env[envName];
  if (raw === undefined || raw === '') {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) {
    console.warn(
      `[mlflow][wal] Ignoring invalid ${envName}=${JSON.stringify(raw)}; ` +
        `expected a non-negative integer. Falling back to default ${fallback}.`,
    );
    return fallback;
  }
  return parsed;
}

/**
 * Wall-clock budget for retrying a single record before the daemon moves it
 * to the dead-letter file (`failed.log`), in milliseconds.
 *
 * Reuses Python's `MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT` (seconds, default
 * 500) and multiplies by 1000 to keep the daemon math in ms alongside
 * {@link backoff}. The anchor for "elapsed" is the daemon's first attempt
 * on the record (`WalRecord.firstAttemptAt`), not the record's creation
 * time, so a record that sat in the WAL while the daemon was idle still
 * gets a fresh budget once a daemon picks it up.
 *
 * On every failure the daemon checks `elapsed + next_backoff_delay`; if
 * that would exceed the budget, the record is dead-lettered instead of
 * re-appended. This mirrors Python's
 * `_retry_databricks_sdk_call_with_exponential_backoff`.
 *
 * `0` is a legitimate value, not "invalid → fall back to default": it
 * means "dead-letter on the first failure" and matches Python exactly.
 */
export const RETRY_TIMEOUT_MS: number =
  readNonNegativeInt('MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT', 500) * 1_000;

/**
 * Interval between daemon batch loop iterations, in milliseconds.
 *
 * Reuses `MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS` — the same env var
 * the Python `BatchSpanProcessor` and `SpanBatcher` consult — so an operator
 * sets one knob to control "max latency between a trace landing and being
 * shipped to the backend" across both runtimes.
 *
 * Note that the cost shape is different in the two layers: Python's variable
 * gates an in-process span buffer flush (effectively free per tick), while
 * the TS variable gates an out-of-process daemon poll that reads the on-disk
 * WAL each tick. Setting this very aggressively (e.g. <1000 ms) saves only
 * a few seconds of trace-export lag and pays the disk-scan cost continuously
 * — usually not worth it.
 *
 * The default is intentionally higher than the Python side's 5000 ms because
 * the disk-scan cost makes the trade-off lean toward fewer ticks. Operators
 * who want the runtimes to behave identically can simply set this env var.
 */
export const BATCH_INTERVAL_MS: number = readPositiveInt(
  'MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS',
  15_000,
);

/**
 * Threshold for daemon idle shutdown, in milliseconds. After this much
 * time with the WAL empty and no new records appended, the daemon flushes,
 * releases its lock, and exits.
 *
 * Override via `MLFLOW_TRACE_DAEMON_IDLE_MS`.
 */
export const IDLE_MS: number = readPositiveInt('MLFLOW_TRACE_DAEMON_IDLE_MS', 120_000);

/**
 * Retention window (in UTC days) for the daily-rotated `failed.log.<date>`
 * and `daemon.log.<date>` files under the WAL directory. Files older than
 * `today_utc − LOG_RETENTION_DAYS` are unlinked once per daemon startup;
 * a value of `0` disables the sweep entirely.
 *
 * This is the only resource the daemon owns whose disk footprint grows
 * monotonically — `queue.log` is bounded by compaction, and the lock
 * socket is managed by the lock lifecycle. `MLFLOW_TRACE_LOG_RETENTION_DAYS`
 * is the operator knob; the default of 14 days is enough to cover a
 * multi-day backend outage's dead-letter trail while still being a
 * meaningful upper bound.
 */
export const LOG_RETENTION_DAYS: number = readNonNegativeInt('MLFLOW_TRACE_LOG_RETENTION_DAYS', 14);
