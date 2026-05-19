/**
 * Filesystem paths for the per-user MLflow trace WAL (Write-Ahead Log) spool.
 *
 * The spool is shared across all Claude Code sessions (and other MLflow TS
 * integrations) for a single OS user. All paths derive from the spool root,
 * which can be overridden by setting `MLFLOW_WAL_DIR`.
 */

import { homedir, platform, userInfo } from 'node:os';
import { join } from 'node:path';

/**
 * Root directory of the per-user spool. Defaults to `~/.mlflow/wal/`.
 *
 * Empty-string values for `MLFLOW_WAL_DIR` (e.g. `export MLFLOW_WAL_DIR=""`)
 * are treated the same as unset to avoid silently producing relative paths.
 */
export function getWalDir(): string {
  const override = process.env.MLFLOW_WAL_DIR;
  if (override !== undefined && override !== '') {
    return override;
  }
  return join(homedir(), '.mlflow', 'wal');
}

/**
 * Append-only Write-Ahead Log of pending trace uploads.
 */
export function getWalPath(): string {
  return join(getWalDir(), 'queue.log');
}

/**
 * Append-only dead-letter file containing records that have exhausted
 * `MLFLOW_TRACE_MAX_RETRY_ATTEMPTS` attempts. Records here are durable and
 * inspectable / replayable by operators; the daemon never re-reads this file.
 */
export function getDeadLetterPath(): string {
  return join(getWalDir(), 'failed.log');
}

/**
 * Daemon log file with diagnostic lines (retries, DLQ entries, fatal errors).
 */
export function getDaemonLogPath(): string {
  return join(getWalDir(), 'daemon.log');
}

/**
 * Path used as the daemon's singleton liveness lock.
 *
 * - POSIX: a UNIX domain socket file at `<wal_dir>/daemon.sock`. The daemon
 *   binds it with `net.createServer().listen()` for its whole lifetime;
 *   the kernel's exclusive-bind semantic on that path is what gives us the
 *   at-most-one-daemon guarantee.
 * - Windows: a Named Pipe at `\\?\pipe\mlflow-trace-daemon-<user>`. Same
 *   `net.createServer().listen()` API, different kernel object. Named pipes
 *   are kernel-refcounted and disappear automatically on process exit, so
 *   no stale-cleanup logic is needed there.
 *
 * The username is sanitized into the pipe name to keep it a valid Windows
 * path component and to scope the pipe per OS user.
 */
export function getLockSocketPath(): string {
  if (platform() === 'win32') {
    const sanitizedUser = userInfo().username.replace(/[^a-zA-Z0-9_-]/g, '_');
    return `\\\\?\\pipe\\mlflow-trace-daemon-${sanitizedUser}`;
  }
  return join(getWalDir(), 'daemon.sock');
}
