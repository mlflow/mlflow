/**
 * The trace-uploader daemon — long-lived background process that drains
 * the WAL on behalf of all hooks running for this OS user.
 */

import { appendFileSync } from 'node:fs';
import { mkdir, readdir, unlink } from 'node:fs/promises';
import { createServer, Server, Socket } from 'node:net';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { MlflowClient } from '../../clients/client';
import { MlflowHttpError } from '../../clients/utils';
import { SerializedTraceData, TraceData } from '../../core/entities/trace_data';
import { SerializedTraceInfo, TraceInfo } from '../../core/entities/trace_info';
import { BatchingWriter } from './batching_writer';
import { clearClientCache, clearClientForUri, clientForUri } from './clients';
import { BATCH_INTERVAL_MS, IDLE_MS, LOG_RETENTION_DAYS, RETRY_TIMEOUT_MS } from './constants';
import { createIpcConnectionHandler } from './ipc';
import { getDaemonLogPath, getLockSocketPath, getWalDir } from './paths';
import { PidLock, tryAcquirePidLock } from './pid_lock';
import { appendDeadLetter, compact, readPending, walSize } from './storage';
import { isDaemonAlive } from './supervisor';
import { WalRecord } from './types';

const MAX_BACKOFF_MS = 5 * 60_000;

const COMPACTION_THRESHOLD_BYTES = 100_000;

function isErrnoException(err: unknown): err is NodeJS.ErrnoException {
  return typeof err === 'object' && err != null && 'code' in err;
}

function formatError(err: unknown): string {
  if (err instanceof Error) {
    return err.stack ?? err.message;
  }
  return String(err);
}

/**
 * Append a single timestamped line to the daemon log file. Uses sync
 * I/O so two concurrent log calls cannot interleave;
 */
export function log(line: string): void {
  const timestamp = new Date().toISOString();
  const fullLine = `[${timestamp}] [pid=${process.pid}] ${line}\n`;
  try {
    appendFileSync(getDaemonLogPath(), fullLine);
  } catch {
    // Failures are swallowed so a broken disk never crashes the daemon mid-batch.
  }
}

/** Awaits `ms` real milliseconds. Used for the batch loop's pacing. */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Exponential backoff with a fixed cap.
 */
export function backoff(attempt: number): number {
  return Math.min(2_000 * 2 ** (attempt - 1), MAX_BACKOFF_MS);
}

/**
 * Default connection handler used when {@link acquireLock} is invoked
 * without one (e.g. by tests that only care about the lock semantic).
 */
function defaultConnectionHandler(socket: Socket): void {
  socket.end();
}

/**
 * Result of a single `listen()` attempt on the lock socket. Distinguishes
 * the three outcomes acquireLock has to dispatch on: bound (we won),
 * EADDRINUSE (someone else holds the path — live daemon or stale file),
 * or any other error (unexpected).
 */
type BindResult =
  | { kind: 'bound'; server: Server }
  | { kind: 'eaddrinuse' }
  | { kind: 'error'; err: Error };

/**
 * Try to bind a fresh server to `socketPath`. Never throws — every
 * outcome is encoded in the returned discriminated union so callers can
 * keep their control flow linear instead of try/catching around a
 * Promise that resolves three different ways.
 */
function tryBind(socketPath: string, onConnection: (socket: Socket) => void): Promise<BindResult> {
  return new Promise<BindResult>((resolve) => {
    const server = createServer(onConnection);
    server.once('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE') {
        resolve({ kind: 'eaddrinuse' });
        return;
      }
      resolve({ kind: 'error', err });
    });
    server.listen(socketPath, () => resolve({ kind: 'bound', server }));
  });
}

/**
 * Handle returned by {@link acquireLock} on success. Pairs the bound
 * server with the optional cross-process PID lock so {@link releaseLock}
 * can tear both down together. `pidLock` is `null` on Windows because
 * named pipes already provide kernel-refcounted cross-process
 * exclusion; the file-based lock would be pure overhead there.
 */
export interface DaemonLock {
  server: Server;
  pidLock: PidLock | null;
}

/**
 * Acquire the singleton daemon lock.
 *
 * Returns the {@link DaemonLock} on success, or `null` when another
 * daemon is already running (the caller should `process.exit(0)`
 * cleanly).
 *
 * ## How exclusion is enforced
 *
 * On POSIX, exclusion is enforced by the atomic PID lock acquired in
 * {@link tryAcquirePidLock} *before* the socket bind. The PID lock uses
 * `link()`, a single kernel-atomic syscall — only one process can win,
 * the rest see `EEXIST` and concede. With the lock held, no sibling
 * daemon is past this point at the same time as us, so the subsequent
 * `bind()` runs uncontested.
 *
 * On Windows, named pipes are kernel-refcounted: two daemons cannot
 * bind the same pipe name simultaneously regardless of the order of
 * operations. We skip the PID lock there and rely on `listen()`'s
 * native exclusion — the loser of a concurrent-bind race sees
 * `EADDRINUSE` from {@link bindUnderPidLock}, which returns `null` so
 * we concede here the same way the PID-lock-lost path does on POSIX.
 */
export async function acquireLock(
  onConnection: (socket: Socket) => void = defaultConnectionHandler,
): Promise<DaemonLock | null> {
  if (await isDaemonAlive()) {
    return null;
  }

  // Cross-process exclusion gate. On POSIX this is the only mechanism
  // that prevents two daemons from racing through stale-socket
  // recovery and ending up both bound;
  const pidLock = process.platform === 'win32' ? null : await tryAcquirePidLock();
  if (process.platform !== 'win32' && pidLock == null) {
    return null;
  }

  try {
    const socketPath = getLockSocketPath();
    const server = await bindUnderPidLock(socketPath, onConnection);
    if (server == null) {
      // Windows-only path: a sibling daemon won the named-pipe race.
      // No PID lock to release (Windows path; `pidLock` is always
      // null here). Concede the same way the POSIX PID-lock-lost case
      // does so the caller can `process.exit(0)` cleanly.
      return null;
    }
    return { server, pidLock };
  } catch (err) {
    // Bind path failed after we acquired the PID lock — release it so
    // the next spawn isn't blocked by our orphaned lockfile.
    await pidLock?.release();
    throw err;
  }
}

/**
 * Bind the lock socket, with recovery semantics that depend on the
 * cross-process exclusion guarantee held by the caller.
 *
 * Returns:
 *   - the bound `Server` on success.
 *   - `null` on Windows when a sibling daemon won the named-pipe bind
 *     race. POSIX handles the equivalent "sibling won" case via the
 *     PID lock acquisition earlier in {@link acquireLock}, so this
 *     return value only fires on Windows.
 *
 * The asymmetric handling reflects platform reality: Unix domain
 * sockets leave a filesystem entry that survives the owning process,
 * so POSIX needs an unlink + rebind dance for orphan recovery. Windows
 * named pipes are kernel-refcounted — when the owning process dies
 * the pipe disappears from the kernel namespace automatically, so
 * there is no orphan recovery to do; `EADDRINUSE` there can only mean
 * a live sibling.
 */
async function bindUnderPidLock(
  socketPath: string,
  onConnection: (socket: Socket) => void,
): Promise<Server | null> {
  const first = await tryBind(socketPath, onConnection);
  if (first.kind === 'bound') {
    return first.server;
  }
  if (first.kind === 'error') {
    throw first.err;
  }

  // EADDRINUSE on the first attempt. The right action depends on
  // platform:
  //   - Windows: no PID lock is held (named pipes provide their own
  //     kernel-level exclusion), and there is no orphan pipe file to
  //     unlink. EADDRINUSE here means a live sibling daemon won the
  //     race; concede by returning null and let `acquireLock`
  //     propagate that as a clean "another daemon is running" signal.
  //   - POSIX: we hold the PID lock, so no sibling daemon is racing
  //     us. EADDRINUSE must be a leftover socket file from a daemon
  //     that crashed without cleanup. Unlink the orphan and retry.
  if (process.platform === 'win32') {
    return null;
  }

  try {
    await unlink(socketPath);
  } catch (err) {
    if (isErrnoException(err) && err.code !== 'ENOENT') {
      throw err;
    }
  }

  const second = await tryBind(socketPath, onConnection);
  if (second.kind === 'bound') {
    return second.server;
  }
  if (second.kind === 'eaddrinuse') {
    throw new Error(
      `bind failed with EADDRINUSE on ${socketPath} while holding the PID lock — ` +
        `another process is squatting on the daemon's socket path`,
    );
  }
  throw second.err;
}

/**
 * Release the lock acquired by {@link acquireLock}. Closes the server,
 * unlinks the POSIX socket file, and (when present) releases the PID
 * lock so the next daemon's `acquireLock` finds a clean slate.
 */
export async function releaseLock(server: Server, pidLock: PidLock | null = null): Promise<void> {
  await new Promise<void>((resolve) => server.close(() => resolve()));
  if (process.platform !== 'win32') {
    try {
      await unlink(getLockSocketPath());
    } catch (err) {
      if (isErrnoException(err) && err.code !== 'ENOENT') {
        log(`releaseLock: socket unlink failed: ${formatError(err)}`);
      }
    }
  }
  if (pidLock != null) {
    try {
      await pidLock.release();
    } catch (err) {
      log(`releaseLock: pid lock release failed: ${formatError(err)}`);
    }
  }
}

const ROTATED_LOG_REGEX = /^(failed|daemon)\.log\.(\d{4})-(\d{2})-(\d{2})$/;

const MS_PER_DAY = 24 * 60 * 60 * 1_000;

/**
 * One-shot retention sweep for the daily-rotated `failed.log.<date>` and
 * `daemon.log.<date>` files in the WAL dir.
 *
 * Called once per daemon startup, after the liveness lock is held and
 * before the batch loop starts.
 *
 * The sweep is best-effort: any failure (missing dir, EPERM on a file the
 * user hand-edited, malformed filenames) is logged at debug and
 * swallowed. Retention housekeeping must never abort daemon startup.
 */
export async function pruneOldLogs(
  opts: { now?: Date; retentionDays?: number } = {},
): Promise<void> {
  const retentionDays = opts.retentionDays ?? LOG_RETENTION_DAYS;
  if (retentionDays <= 0) {
    return;
  }

  const now = opts.now ?? new Date();

  const todayUtcMs = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const cutoffMs = todayUtcMs - retentionDays * MS_PER_DAY;

  const walDir = getWalDir();
  let entries: string[];
  try {
    entries = await readdir(walDir);
  } catch (err) {
    log(`pruneOldLogs: readdir failed for ${walDir}: ${formatError(err)}`);
    return;
  }

  for (const name of entries) {
    const match = ROTATED_LOG_REGEX.exec(name);
    if (match == null) {
      continue;
    }
    const [, , yyyy, mm, dd] = match;
    const year = Number(yyyy);
    const month = Number(mm);
    const day = Number(dd);
    const fileDate = new Date(Date.UTC(year, month - 1, day));

    if (
      fileDate.getUTCFullYear() !== year ||
      fileDate.getUTCMonth() !== month - 1 ||
      fileDate.getUTCDate() !== day
    ) {
      continue;
    }
    if (fileDate.getTime() >= cutoffMs) {
      continue;
    }

    const fullPath = join(walDir, name);
    try {
      await unlink(fullPath);
      log(`pruneOldLogs: removed ${name} (older than ${retentionDays} day(s))`);
    } catch (err) {
      log(`pruneOldLogs: failed to unlink ${name}: ${formatError(err)}`);
    }
  }
}

/**
 * HTTP statuses that {@link uploadOne} treats as "the cached
 * credentials are stale; rebuild and retry once."
 */
const AUTH_REFRESH_STATUSES: ReadonlySet<number> = new Set([401, 403]);

function isAuthRefreshableError(err: unknown): boolean {
  return err instanceof MlflowHttpError && AUTH_REFRESH_STATUSES.has(err.status);
}

/**
 * Perform the create-trace + upload-trace-data sequence.
 */
async function performUpload(client: MlflowClient, record: WalRecord): Promise<void> {
  const traceInfo = TraceInfo.fromJson(record.traceInfo as unknown as SerializedTraceInfo);
  const traceData = TraceData.fromJson(record.traceData as SerializedTraceData);
  const resolvedInfo = await client.createTrace(traceInfo);
  await client.uploadTraceData(resolvedInfo, traceData);
}

/**
 * Process a single WAL record: deserialize, push to the backend,
 * tombstone on success or schedule retry / dead-letter on failure.
 */
export async function uploadOne(
  record: WalRecord,
  client: MlflowClient,
  writer: BatchingWriter,
  factory: ClientFactory = clientForUri,
): Promise<void> {
  // Anchor the retry-budget window the first time the daemon touches
  // the record.
  if (record.firstAttemptAt === undefined) {
    record.firstAttemptAt = Date.now();
  }
  const traceId = (record.traceInfo as { trace_id?: string }).trace_id ?? '<unknown>';

  try {
    await performUpload(client, record);
    await writer.submitTombstone(record.id);
    log(`Uploaded record ${record.id} (trace ${traceId})`);
    return;
  } catch (err) {
    if (!isAuthRefreshableError(err)) {
      await handleUploadFailure(record, err, writer);
      return;
    }

    log(
      `Auth failure on record ${record.id} (trace ${traceId}) for ${record.trackingUri}; ` +
        `evicting cached client and retrying once: ${formatError(err)}`,
    );
    clearClientForUri(record.trackingUri);

    let refreshed: MlflowClient;
    try {
      refreshed = factory(record.trackingUri);
    } catch (factoryErr) {
      log(
        `Failed to rebuild client for ${record.trackingUri} after auth error: ${formatError(factoryErr)}`,
      );

      await handleUploadFailure(record, factoryErr, writer);
      return;
    }

    try {
      await performUpload(refreshed, record);
      await writer.submitTombstone(record.id);
      log(`Uploaded record ${record.id} (trace ${traceId}) after credential refresh`);
    } catch (retryErr) {
      // The credential-refresh retry also failed — fall back to the
      // normal failure machinery (backoff / dead-letter).
      await handleUploadFailure(record, retryErr, writer);
    }
  }
}

/**
 * Branch the failure path: either dead-letter the record (if the next
 * retry's backoff would push the total elapsed time past
 * `RETRY_TIMEOUT_MS`) or re-append it with bumped `attempts` and a
 * future `nextAttemptAt`. Either way the original row is tombstoned so
 * the next batch doesn't see it again.
 *
 * Retries use a fresh row `id` so per-attempt rows are independently
 * addressable in the log; the MLflow trace id inside `traceInfo` and
 * the original `firstAttemptAt` stay stable across attempts.
 */
async function handleUploadFailure(
  record: WalRecord,
  err: unknown,
  writer: BatchingWriter,
): Promise<void> {
  const nextAttempts = record.attempts + 1;
  const traceId = (record.traceInfo as { trace_id?: string }).trace_id ?? '<unknown>';
  const delayMs = backoff(nextAttempts);
  const anchor = record.firstAttemptAt ?? record.createdAt;
  const now = Date.now();
  const elapsedMs = now - anchor;

  if (elapsedMs + delayMs >= RETRY_TIMEOUT_MS) {
    log(
      `Dead-lettering record ${record.id} (trace ${traceId}) after ${record.attempts} prior ` +
        `attempts, elapsed=${elapsedMs}ms; next retry would push past ${RETRY_TIMEOUT_MS}ms ` +
        `budget: ${formatError(err)}`,
    );
    await appendDeadLetter(record);
    await writer.submitTombstone(record.id);
    return;
  }

  const retry: WalRecord = {
    ...record,
    id: randomUUID(),
    attempts: nextAttempts,
    nextAttemptAt: now + delayMs,
    firstAttemptAt: anchor,
  };
  log(
    `Retry record ${record.id} → ${retry.id} (trace ${traceId}, attempts=${nextAttempts}, ` +
      `delay=${delayMs}ms, elapsed=${elapsedMs}ms): ${formatError(err)}`,
  );
  // Both submissions ride the same `BatchingWriter` tick: one `writev`,
  // one `fsync`, atomic. Either both lines are durable or neither is.
  await Promise.all([writer.submit(retry), writer.submitTombstone(record.id)]);
}

export type ClientFactory = (trackingUri: string) => MlflowClient;

/**
 * Group `records` by `trackingUri` and upload each group in parallel.
 */
export async function processBatch(
  records: WalRecord[],
  writer: BatchingWriter,
  factory: ClientFactory = clientForUri,
): Promise<void> {
  if (records.length === 0) {
    return;
  }

  const groups = new Map<string, WalRecord[]>();
  for (const record of records) {
    const list = groups.get(record.trackingUri);
    if (list === undefined) {
      groups.set(record.trackingUri, [record]);
    } else {
      list.push(record);
    }
  }

  await Promise.all(
    [...groups.entries()].map(async ([trackingUri, group]) => {
      let client: MlflowClient;
      try {
        client = factory(trackingUri);
      } catch (err) {
        log(`Failed to build client for ${trackingUri}: ${formatError(err)}`);

        await Promise.all(group.map((record) => handleUploadFailure(record, err, writer)));
        return;
      }
      await Promise.all(group.map((record) => uploadOne(record, client, writer, factory)));
    }),
  );
}

/**
 * Wire SIGTERM / SIGINT handlers to abort the supplied controller so
 * the daemon finishes its current batch and exits cleanly rather than
 * being torn down mid-write. The abort is consulted at the top of each
 * {@link runBatchLoop} iteration.
 */
function installSignalHandlers(controller: AbortController): void {
  for (const sig of ['SIGTERM', 'SIGINT'] as const) {
    process.on(sig, () => {
      log(`Received ${sig}; will shut down after current batch.`);
      controller.abort();
    });
  }
}

/**
 * The batch loop. Broken out from {@link main} so the dependencies
 * (client factory, writer, timings, shutdown signal) can be injected
 * in tests without standing up the full daemon lifecycle.
 *
 * Pass an `AbortSignal` to request a graceful shutdown — the loop
 * finishes its current iteration and returns. Without a signal the
 * loop only exits via the `idleMs` timeout path (or by throwing).
 */
export async function runBatchLoop({
  factory = clientForUri,
  writer = new BatchingWriter(),
  batchIntervalMs = BATCH_INTERVAL_MS,
  idleMs = IDLE_MS,
  signal,
}: {
  factory?: ClientFactory;
  writer?: BatchingWriter;
  batchIntervalMs?: number;
  idleMs?: number;
  signal?: AbortSignal;
} = {}): Promise<void> {
  let lastNonEmpty = Date.now();

  while (signal?.aborted !== true) {
    try {
      const pending = await readPending();
      const now = Date.now();
      const due = pending.filter((r) => r.nextAttemptAt <= now);
      
      if (due.length > 0) {
        lastNonEmpty = now;
        await processBatch(due, writer, factory);
      } else if (now - lastNonEmpty >= idleMs) {
        log('WAL idle threshold reached; shutting down.');
        return;
      }

      if ((await walSize()) > COMPACTION_THRESHOLD_BYTES) {
        await compact().catch((err) => log(`Periodic compact failed: ${formatError(err)}`));
      }
    } catch (err) {
      // Don't let a transient FS / parse error tear down the daemon.
      log(`Batch iteration error: ${formatError(err)}`);
    }

    await sleep(batchIntervalMs);
  }
}

/**
 * Daemon entry point.
 */
export async function main(): Promise<void> {
  await mkdir(getWalDir(), { recursive: true });

  const ipcWriter = new BatchingWriter();
  const handle = await acquireLock(createIpcConnectionHandler(ipcWriter));
  if (handle == null) {
    log('Another daemon is already running; exiting cleanly.');
    process.exit(0);
    return;
  }

  log(`Daemon started; pid=${process.pid}, walDir=${getWalDir()}`);
  const shutdown = new AbortController();
  installSignalHandlers(shutdown);

  let exitCode = 0;
  try {
    await pruneOldLogs().catch((err) => log(`Retention sweep failed: ${formatError(err)}`));
    await compact().catch((err) => log(`Startup compact failed: ${formatError(err)}`));
    await runBatchLoop({ writer: ipcWriter, signal: shutdown.signal });
    await compact().catch((err) => log(`Final compact failed: ${formatError(err)}`));
  } catch (err) {
    log(`Daemon main loop crashed: ${formatError(err)}`);
    exitCode = 1;
  } finally {
    await releaseLock(handle.server, handle.pidLock).catch((err) =>
      log(`releaseLock failed: ${formatError(err)}`),
    );
    clearClientCache();
    log(`Daemon exited with code ${exitCode}.`);
  }

  process.exit(exitCode);
}

// Run main() only when invoked directly (i.e. `node bundle/daemon.cjs`),
// not when this module is imported by tests.
if (require.main === module) {
  main().catch((err) => {
    console.error('[mlflow][wal-daemon] fatal:', err);
    process.exit(1);
  });
}
