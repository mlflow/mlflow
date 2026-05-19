/**
 * Filesystem paths for the MLflow trace WAL (Write-Ahead Log) spool.
 *
 * A spool is shared across all Claude Code sessions (and other MLflow TS
 * integrations) that resolve to the same `MLFLOW_WAL_DIR` for a single OS
 * user. The WAL data files (`queue.log`, `failed.log`, `daemon.log`)
 * always live inside the resolved spool root.
 *
 * The daemon's singleton lock is scoped per `(user, spool root)` on both
 * platforms; its exact location depends on the platform and the resolved
 * spool root's length:
 * - POSIX, common case: `<spool>/daemon.sock` next to `queue.log`.
 * - POSIX, long-path fallback: `<tmpdir>/mlflow-<scoped_id>.sock` when the
 *   natural path would exceed `sun_path`.
 * - Windows: a Named Pipe whose name embeds a hash of the spool root.
 *
 * See {@link getLockSocketPath} for the full rationale.
 */

import { createHash } from 'node:crypto';
import { homedir, platform, tmpdir, userInfo } from 'node:os';
import { join, resolve } from 'node:path';

/**
 * Maximum allowed length, in bytes, for a UNIX domain socket path.
 *
 * `sun_path` is fixed-size in `struct sockaddr_un`: 108 bytes on Linux,
 * 104 on macOS/BSD. Each is one byte of NUL terminator, so the actual
 * string max is 107 / 103 respectively. We use the tighter macOS/BSD
 * limit (103) for the check so the same code works on both kernels
 * without per-platform branching.
 */
const POSIX_SOCKET_PATH_MAX_BYTES = 103;

/**
 * Root directory of the per-user spool. Defaults to `~/.mlflow/wal/`.
 *
 * The result is always an absolute, resolved path so that callers comparing
 * or hashing the WAL dir cannot accidentally treat `/tmp/foo` and
 * `/tmp/foo/` as two different roots.
 *
 * Empty-string values for `MLFLOW_WAL_DIR` (e.g. `export MLFLOW_WAL_DIR=""`)
 * are treated the same as unset to avoid silently producing relative paths.
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
 * Stable per-`(user, wal_dir)` identifier used to scope the daemon lock when
 * the lock cannot simply live inside the WAL dir (Windows pipe namespace,
 * POSIX `tmpdir` fallback). 12 hex chars (~48 bits) of SHA-256 is plenty
 * for the handful of distinct WAL dirs a single user would realistically
 * use, and the sanitized username keeps multi-user hosts isolated.
 */
function getScopedLockId(): string {
  const sanitizedUser = userInfo().username.replace(/[^a-zA-Z0-9_-]/g, '_');
  const walDirHash = createHash('sha256').update(getWalDir()).digest('hex').slice(0, 12);
  return `${sanitizedUser}-${walDirHash}`;
}

/**
 * Path used as the daemon's singleton liveness lock.
 *
 * The lock is scoped per `(user, wal_dir)` on both platforms so that a user
 * with two terminals overriding `MLFLOW_WAL_DIR` to different values gets
 * one independent daemon per WAL dir (each daemon only reads its own
 * `queue.log`). A single shared lock identifier would otherwise let one
 * daemon claim the lock and leave the other WAL stranded.
 *
 * - **POSIX, common case**: a UNIX domain socket file at
 *   `<wal_dir>/daemon.sock`. The daemon binds it with
 *   `net.createServer().listen()` for its whole lifetime; the kernel's
 *   exclusive-bind semantic on that path is what gives us the
 *   at-most-one-daemon guarantee. Scoping falls out naturally from the
 *   socket file living inside the WAL dir.
 * - **POSIX, long-path fallback**: `<wal_dir>/daemon.sock` can exceed the
 *   fixed `sun_path` size (104 bytes on macOS/BSD, 108 on Linux), e.g.
 *   if `MLFLOW_WAL_DIR` resolves to a sandboxed container dir or a deep
 *   user folder. When the natural path would exceed
 *   {@link POSIX_SOCKET_PATH_MAX_BYTES}, the socket falls back to
 *   `<tmpdir>/mlflow-<scoped_id>.sock`. A `console.warn` makes the
 *   deviation discoverable. Per-(user, wal_dir) scoping is preserved by
 *   the `<scoped_id>` suffix.
 * - **Windows**: a Named Pipe at `\\?\pipe\mlflow-trace-daemon-<scoped_id>`.
 *   Named pipes share a single flat namespace per machine, so the pipe
 *   name has to encode both the user (to keep multi-user hosts isolated)
 *   and a hash of the resolved WAL dir (to match POSIX's per-WAL-dir
 *   scoping). Named pipes are kernel-refcounted and disappear
 *   automatically on process exit, so no stale-cleanup logic is needed.
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
