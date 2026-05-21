/**
 * Filesystem paths for the MLflow trace WAL (Write-Ahead Log) spool.
 *
 * A spool is shared across all Claude Code sessions
 *
 * The daemon's singleton lock is scoped per `(user, spool root)` on both
 * platforms; its exact location depends on the platform and the resolved
 * spool root's length:
 * - POSIX, common case: `<spool>/daemon.sock` next to `queue.log`.
 * - POSIX, long-path fallback: `<tmpdir>/mlflow-<scoped_id>.sock` when the
 *   natural path would exceed `sun_path`.
 * - Windows: a Named Pipe whose name embeds a hash of the spool root.
 */

import { createHash } from 'node:crypto';
import { homedir, platform, tmpdir, userInfo } from 'node:os';
import { join, resolve } from 'node:path';

/**
 * Maximum allowed length, in bytes, for a UNIX domain socket path.
 */
const POSIX_SOCKET_PATH_MAX_BYTES = 103;

/**
 * Root directory of the per-user spool. Defaults to `~/.mlflow/wal/`.
 */
export function getWalDir(): string {
  const override = process.env.MLFLOW_WAL_DIR;
  const raw =
    override !== undefined && override !== '' ? override : join(homedir(), '.mlflow', 'wal');
  return resolve(raw);
}

/**
 * Append-only Write-Ahead Log of pending trace uploads.
 */
export function getWalPath(): string {
  return join(getWalDir(), 'queue.log');
}

function dateSuffix(d: Date = new Date()): string {
  const yyyy = d.getUTCFullYear().toString().padStart(4, '0');
  const mm = (d.getUTCMonth() + 1).toString().padStart(2, '0');
  const dd = d.getUTCDate().toString().padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

/**
 * Append-only dead-letter file containing records whose retry-budget window
 * (`MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT`) has elapsed. Records here are
 * durable and inspectable / replayable by operators.
 */
export function getDeadLetterPath(now: Date = new Date()): string {
  return join(getWalDir(), `failed.log.${dateSuffix(now)}`);
}

/**
 * Daemon log file with diagnostic lines (retries, DLQ entries, fatal errors).
 */
export function getDaemonLogPath(now: Date = new Date()): string {
  return join(getWalDir(), `daemon.log.${dateSuffix(now)}`);
}

/**
 * Stable per (user, wal_dir) identifier used to scope the daemon lock
 */
function getScopedLockId(): string {
  const sanitizedUser = userInfo().username.replace(/[^a-zA-Z0-9_-]/g, '_');
  const walDirHash = createHash('sha256').update(getWalDir()).digest('hex').slice(0, 12);
  return `${sanitizedUser}-${walDirHash}`;
}

/**
 * Path used as the daemon's singleton liveness lock.
 */
export function getLockSocketPath(): string {
  const scopedId = getScopedLockId();

  if (platform() === 'win32') {
    return `\\\\?\\pipe\\mlflow-trace-daemon-${scopedId}`;
  }

  const naturalPath = join(getWalDir(), 'daemon.sock');
  if (Buffer.byteLength(naturalPath, 'utf8') <= POSIX_SOCKET_PATH_MAX_BYTES) {
    return naturalPath;
  }

  // Short basename so the fallback itself stays under sun_path even on
  // macOS, whose tmpdir is already ~49 chars (`/var/folders/<2>/<30>/T/`).
  const fallback = join(tmpdir(), `mlflow-${scopedId}.sock`);
  console.warn(
    `[mlflow][wal] WAL dir produces a socket path longer than ` +
      `${POSIX_SOCKET_PATH_MAX_BYTES} bytes (${Buffer.byteLength(naturalPath, 'utf8')}); ` +
      `using ${fallback} for the daemon lock instead.`,
  );
  return fallback;
}
