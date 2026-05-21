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
 * Maximum number of upload attempts per record before the daemon moves the
 * record to the dead-letter file (`failed.log`).
 *
 * Override via `MLFLOW_TRACE_MAX_RETRY_ATTEMPTS`.
 */
export const MAX_ATTEMPTS: number = readPositiveInt('MLFLOW_TRACE_MAX_RETRY_ATTEMPTS', 10);

/**
 * Interval between daemon batch loop iterations, in milliseconds.
 *
 * Override via `MLFLOW_TRACE_BATCH_INTERVAL_MS`.
 */
export const BATCH_INTERVAL_MS: number = readPositiveInt('MLFLOW_TRACE_BATCH_INTERVAL_MS', 15_000);

/**
 * Threshold for daemon idle shutdown, in milliseconds. After this much
 * time with the WAL empty and no new records appended, the daemon flushes,
 * releases its lock, and exits.
 *
 * Override via `MLFLOW_TRACE_DAEMON_IDLE_MS`.
 */
export const IDLE_MS: number = readPositiveInt('MLFLOW_TRACE_DAEMON_IDLE_MS', 120_000);
