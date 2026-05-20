/**
 * Filesystem paths for the MLflow trace WAL (Write-Ahead Log) spool.
 *
 * A spool is shared across all Claude Code sessions (and other MLflow TS
 * integrations) that resolve to the same `MLFLOW_WAL_DIR` for a single OS
 * user. The WAL files always live inside the resolved spool root:
 *
 * - `queue.log` — single file, compacted on the fly. The hot work queue.
 * - `failed.log.<YYYY-MM-DD>` — daily-rotated dead-letter file. Append-only
 *   forever; never compacted. The suffix comes from the *UTC* date so the
 *   day boundary is stable across timezones and matches the `…Z` ISO
 *   timestamps written inside `daemon.log`.
 * - `daemon.log.<YYYY-MM-DD>` — daily-rotated diagnostic stream. Same UTC
 *   convention as `failed.log`.
 *
 * "Rotation" is stateless: every write resolves its path from the current
 * date, so a writer that crosses midnight UTC lands subsequent lines in
 * the next-day file with no rename, no detection, no coordination.
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
 * `YYYY-MM-DD` in UTC. Used as the suffix on `failed.log` and `daemon.log`
 * so the two files rotate daily without any explicit rename. UTC keeps the
 * boundary stable across the timezones a multi-region rollout will hit and
 * matches the `…Z` ISO timestamps the daemon writes inside the files.
 *
 * Exposed as an optional argument purely so tests can pin the date.
 */
function dateSuffix(d: Date = new Date()): string {
  const yyyy = d.getUTCFullYear().toString().padStart(4, '0');
  const mm = (d.getUTCMonth() + 1).toString().padStart(2, '0');
  const dd = d.getUTCDate().toString().padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

/**
 * Append-only dead-letter file containing records whose retry-budget window
 * (`MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT`) has elapsed. Records here are
 * durable and inspectable / replayable by operators; the daemon never
 * re-reads this file.
 *
 * Rotated daily by the UTC date in the filename suffix. Day boundaries that
 * occur mid-write naturally split lines into the next-day file because the
 * path is resolved from the clock at every `appendDeadLetter` call.
 */
export function getDeadLetterPath(now: Date = new Date()): string {
  return join(getWalDir(), `failed.log.${dateSuffix(now)}`);
}

/**
 * Daemon log file with diagnostic lines (retries, DLQ entries, fatal errors).
 *
 * Same daily-by-UTC rotation as {@link getDeadLetterPath}: the daemon's
 * `log()` helper resolves this on every line, so a daemon that spans
 * midnight writes earlier lines to the `…<today>` file and later lines to
 * the `…<tomorrow>` file with no explicit handoff.
 */
export function getDaemonLogPath(now: Date = new Date()): string {
  return join(getWalDir(), `daemon.log.${dateSuffix(now)}`);
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
