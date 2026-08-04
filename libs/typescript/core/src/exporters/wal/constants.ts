/**
 * Tunable defaults for the WAL exporter and uploader daemon.
 */

import { readNonNegativeInt, readPositiveInt } from '../../core/utils/env';

/**
 * Wall-clock budget for retrying a single record before the daemon moves it
 * to the dead-letter file (`failed.log`), in milliseconds.
 */
export const RETRY_TIMEOUT_MS: number =
  readNonNegativeInt('MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT', 500) * 1_000;

/**
 * Interval between daemon batch loop iterations, in milliseconds.
 */
export const BATCH_INTERVAL_MS: number = readPositiveInt(
  'MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS',
  15_000,
);

/**
 * Threshold for daemon idle shutdown, in milliseconds. After this much
 * time with the WAL empty and no new records appended, the daemon flushes,
 * releases its lock, and exits.
 */
export const IDLE_MS: number = readPositiveInt('MLFLOW_TRACE_DAEMON_IDLE_MS', 120_000);

/**
 * Retention window (in UTC days) for the daily-rotated `failed.log.<date>`
 * and `daemon.log.<date>` files under the WAL directory. Files older than
 * `today_utc − LOG_RETENTION_DAYS` are unlinked once per daemon startup;
 * a value of `0` disables the sweep entirely.
 */
export const LOG_RETENTION_DAYS: number = readNonNegativeInt('MLFLOW_TRACE_LOG_RETENTION_DAYS', 14);
