import { spawn } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { BatchingWriter } from '../../../src/exporters/wal/batching_writer';
import {
  acquireLock,
  backoff,
  log,
  processBatch,
  pruneOldLogs,
  releaseLock,
  runBatchLoop,
  uploadOne,
} from '../../../src/exporters/wal/daemon';
import {
  getDaemonLogPath,
  getDeadLetterPath,
  getLockSocketPath,
  getPidLockPath,
  getWalPath,
} from '../../../src/exporters/wal/paths';
import { appendRecord, readPending } from '../../../src/exporters/wal/storage';
import type { WalRecord } from '../../../src/exporters/wal/types';
import type { MlflowClient } from '../../../src/clients/client';
import { MlflowHttpError } from '../../../src/clients/utils';
import type { TraceInfo } from '../../../src/core/entities/trace_info';
import type { TraceData } from '../../../src/core/entities/trace_data';

/**
 * Spawn a trivial child, wait for it to exit, and return its now-freed
 * pid. Used so the integration tests can seed a stale PID lockfile
 * with a pid that is guaranteed to fail `process.kill(pid, 0)` with
 * ESRCH and trigger the stale-recovery branch in `tryAcquirePidLock`.
 */
function spawnAndReapPid(): Promise<number> {
  return new Promise<number>((resolve, reject) => {
    const child = spawn(process.execPath, ['-e', 'process.exit(0)']);
    const pid = child.pid;
    if (pid == null) {
      reject(new Error('child spawn produced no pid'));
      return;
    }
    child.once('exit', () => resolve(pid));
    child.once('error', reject);
  });
}

function makeRecord(overrides: Partial<WalRecord> = {}): WalRecord {
  // Fields shaped so TraceInfo.fromJson / TraceData.fromJson succeed:
  // execution_duration is the "Ns" string form expected by the parser
  // (a raw number throws because of `.replace`), spans is an empty array.
  const base: WalRecord = {
    id: 'row-1',
    trackingUri: 'http://localhost:5000',
    experimentId: '0',
    traceInfo: {
      trace_id: 'tr-1',
      trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: '0' } },
      request_time: 0,
      execution_duration: '0s',
      state: 'OK',
      request_preview: '',
      response_preview: '',
      client_request_id: '',
      tags: {},
      trace_metadata: {},
      assessments: [],
    },
    traceData: { spans: [] },
    attempts: 0,
    nextAttemptAt: 0,
    createdAt: Date.now(),
  };
  return { ...base, ...overrides };
}

function makeClient(opts: {
  createTrace?: jest.Mock;
  uploadTraceData?: jest.Mock;
  exportOtlpSpans?: jest.Mock;
}): MlflowClient {
  return {
    createTrace:
      opts.createTrace ?? jest.fn().mockImplementation((info: TraceInfo) => Promise.resolve(info)),
    uploadTraceData:
      opts.uploadTraceData ??
      jest.fn().mockImplementation((_info: TraceInfo, _data: TraceData) => Promise.resolve()),
    exportOtlpSpans:
      opts.exportOtlpSpans ??
      jest
        .fn()
        .mockImplementation((_experimentId: string, _bytes: Uint8Array) => Promise.resolve()),
    getHost: () => 'http://localhost:5000',
  } as unknown as MlflowClient;
}

describe('wal/daemon', () => {
  let walDir: string;
  // Capture the developer's pre-test value so we restore (not just unset)
  // in afterEach. Mirrors the pattern in paths.test.ts so running
  // `MLFLOW_WAL_DIR=/some/dir jest` doesn't lose the override for later
  // tests in the same worker.
  const originalWalDir = process.env.MLFLOW_WAL_DIR;

  beforeEach(async () => {
    walDir = await mkdtemp(join(tmpdir(), 'mlflow-daemon-'));
    process.env.MLFLOW_WAL_DIR = walDir;
  });

  afterEach(async () => {
    if (originalWalDir === undefined) {
      delete process.env.MLFLOW_WAL_DIR;
    } else {
      process.env.MLFLOW_WAL_DIR = originalWalDir;
    }
    await rm(walDir, { recursive: true, force: true });
  });

  describe('backoff', () => {
    it('doubles on each attempt until the cap', () => {
      expect(backoff(1)).toBe(2_000);
      expect(backoff(2)).toBe(4_000);
      expect(backoff(3)).toBe(8_000);
      expect(backoff(8)).toBe(256_000);
    });

    it('caps at 5 minutes', () => {
      expect(backoff(9)).toBe(300_000);
      expect(backoff(20)).toBe(300_000);
    });
  });

  describe('log', () => {
    it('appends a timestamped line to the daemon log file', () => {
      log('hello');
      log('world');
      const path = getDaemonLogPath();
      // daemon.log is daily-rotated; assert the suffix is present so a
      // regression to a single unbounded file is caught here too.
      expect(path).toMatch(/[/\\]daemon\.log\.\d{4}-\d{2}-\d{2}$/);
      const contents = readFileSync(path, 'utf8');
      const lines = contents.trim().split('\n');
      expect(lines).toHaveLength(2);
      expect(lines[0]).toMatch(/^\[\d{4}-\d{2}-\d{2}T.+Z\] \[pid=\d+\] hello$/);
      expect(lines[1]).toMatch(/world$/);
    });

    it('swallows write errors silently', () => {
      // Point the log path at a directory that doesn't exist by deleting
      // the WAL dir mid-test. appendFileSync will throw internally; log()
      // must not propagate.
      process.env.MLFLOW_WAL_DIR = join(walDir, 'nope', 'deeper');
      expect(() => log('ignored')).not.toThrow();
    });
  });

  describe('acquireLock / releaseLock', () => {
    it('binds the lock socket and releases cleanly', async () => {
      const handle = await acquireLock();
      expect(handle).not.toBeNull();
      expect(existsSync(getLockSocketPath())).toBe(true);
      await releaseLock(handle!.server, handle!.pidLock);
      expect(existsSync(getLockSocketPath())).toBe(false);
    });

    it('returns null when another daemon already holds the lock', async () => {
      const first = await acquireLock();
      expect(first).not.toBeNull();
      try {
        const second = await acquireLock();
        expect(second).toBeNull();
      } finally {
        await releaseLock(first!.server, first!.pidLock);
      }
    });

    it('unlinks a stale POSIX socket file before binding', async () => {
      if (process.platform === 'win32') {
        return; // Named pipes have no stale-file concept.
      }
      // Simulate a previously-crashed daemon by writing a regular file
      // at the socket path. acquireLock should unlink it on its way to
      // a fresh bind.
      await writeFile(getLockSocketPath(), 'stale');
      const handle = await acquireLock();
      try {
        expect(handle).not.toBeNull();
      } finally {
        await releaseLock(handle!.server, handle!.pidLock);
      }
    });
  });

  // POSIX-only PID-lock integration. On Windows the named pipe is
  // kernel-refcounted and acquireLock returns `pidLock: null`, so
  // there's nothing for these tests to assert. Use a describe-level
  // skip rather than per-`it` guards so the suite reports honestly on
  // CI ("skipped" instead of "passed in 0ms").
  const describePosix = process.platform === 'win32' ? describe.skip : describe;

  describePosix('acquireLock / releaseLock — PID lock integration', () => {
    it('creates the pid lockfile with our pid and removes it on release', async () => {
      const handle = await acquireLock();
      try {
        expect(handle).not.toBeNull();
        expect(handle!.pidLock).not.toBeNull();
        // Lockfile must exist alongside the socket and record our pid
        // exactly — content-stamped release relies on the exact match.
        expect(existsSync(getPidLockPath())).toBe(true);
        const content = await readFile(getPidLockPath(), 'utf8');
        expect(content).toBe(`${process.pid}\n`);
      } finally {
        await releaseLock(handle!.server, handle!.pidLock);
      }
      expect(existsSync(getPidLockPath())).toBe(false);
    });

    it('recovers from a stale pid lockfile left behind by a crashed daemon', async () => {
      // A previously-crashed daemon's pid file with a now-dead pid.
      // tryAcquirePidLock's liveness probe (`process.kill(pid, 0)`)
      // should return ESRCH, the content-stamped unlink should clear
      // the file, and acquireLock should proceed to bind normally.
      const deadPid = await spawnAndReapPid();
      await writeFile(getPidLockPath(), `${deadPid}\n`);
      const handle = await acquireLock();
      try {
        expect(handle).not.toBeNull();
        expect(handle!.pidLock).not.toBeNull();
        const content = await readFile(getPidLockPath(), 'utf8');
        expect(content).toBe(`${process.pid}\n`);
      } finally {
        await releaseLock(handle!.server, handle!.pidLock);
      }
    });

    it('concedes when an alive process already holds the pid lockfile', async () => {
      await writeFile(getPidLockPath(), `${process.pid}\n`);
      const handle = await acquireLock();
      expect(handle).toBeNull();
      // Lockfile must remain untouched when we concede.
      const content = await readFile(getPidLockPath(), 'utf8');
      expect(content).toBe(`${process.pid}\n`);
      // And the socket must not have been bound.
      expect(existsSync(getLockSocketPath())).toBe(false);
    });

    it('serializes concurrent acquireLock calls to a single winner', async () => {
      // Both calls run inside the same Node event loop, but the
      // `link()` syscall inside tryAcquirePidLock still goes through
      // libuv's fs threadpool and the kernel's inode lock — so exactly
      // one wins the atomic create, and the other reads our (alive)
      // pid and concedes via the live-holder branch.
      const [first, second] = await Promise.all([acquireLock(), acquireLock()]);
      try {
        const acquired = [first, second].filter((h) => h != null);
        const conceded = [first, second].filter((h) => h == null);
        expect(acquired).toHaveLength(1);
        expect(conceded).toHaveLength(1);
        // The winner owns the socket; the loser never bound anything.
        expect(existsSync(getLockSocketPath())).toBe(true);
        // The winner's pid is recorded in the lockfile.
        const content = await readFile(getPidLockPath(), 'utf8');
        expect(content).toBe(`${process.pid}\n`);
      } finally {
        if (first != null) {
          await releaseLock(first.server, first.pidLock);
        }
        if (second != null) {
          await releaseLock(second.server, second.pidLock);
        }
      }
    });

    it('releases the pid lockfile when both socket and pid lock are unlinked together', async () => {
      // Verifies the teardown order is symmetric to acquisition:
      // releaseLock must remove the socket file AND the pid lockfile
      // so a subsequent acquireLock starts from a fully-clean slate.
      const handle = await acquireLock();
      expect(handle).not.toBeNull();
      expect(existsSync(getLockSocketPath())).toBe(true);
      expect(existsSync(getPidLockPath())).toBe(true);
      await releaseLock(handle!.server, handle!.pidLock);
      expect(existsSync(getLockSocketPath())).toBe(false);
      expect(existsSync(getPidLockPath())).toBe(false);
    });

    it('allows a fresh acquire after a clean release', async () => {
      // End-to-end smoke for the acquire → release → reacquire cycle.
      // Catches regressions where releaseLock leaves stale state that
      // the next acquireLock cannot recover from.
      const first = await acquireLock();
      expect(first).not.toBeNull();
      await releaseLock(first!.server, first!.pidLock);
      const second = await acquireLock();
      try {
        expect(second).not.toBeNull();
      } finally {
        await releaseLock(second!.server, second!.pidLock);
      }
    });
  });

  describe('uploadOne', () => {
    it('uploads the trace and tombstones the WAL row on success', async () => {
      const createTrace = jest.fn().mockImplementation((info: TraceInfo) => Promise.resolve(info));
      const uploadTraceData = jest
        .fn()
        .mockImplementation((_info: TraceInfo, _data: TraceData) => Promise.resolve());
      const client = makeClient({ createTrace, uploadTraceData });

      const record = makeRecord({ id: 'row-success' });
      await uploadOne(record, client, new BatchingWriter());

      expect(createTrace).toHaveBeenCalledTimes(1);
      expect(uploadTraceData).toHaveBeenCalledTimes(1);

      // After tombstone the record should not be in the live set.
      const pending = await readPending();
      expect(pending.find((r) => r.id === 'row-success')).toBeUndefined();
    });

    it('logs captured OTLP spans to the DB when record.otlpSpans is present', async () => {
      const exportOtlpSpans = jest
        .fn()
        .mockImplementation((_experimentId: string, _bytes: Uint8Array) => Promise.resolve());
      const client = makeClient({ exportOtlpSpans });

      const otlpBytes = Buffer.from([0x0a, 0x01, 0x02, 0x03]);
      const record = makeRecord({
        id: 'row-otlp',
        experimentId: 'exp-7',
        otlpSpans: otlpBytes.toString('base64'),
      });
      await uploadOne(record, client, new BatchingWriter());

      expect(exportOtlpSpans).toHaveBeenCalledTimes(1);
      const [experimentId, bytes] = exportOtlpSpans.mock.calls[0] as [string, Uint8Array];
      expect(experimentId).toBe('exp-7');
      expect(Buffer.from(bytes).equals(otlpBytes)).toBe(true);

      const pending = await readPending();
      expect(pending.find((r) => r.id === 'row-otlp')).toBeUndefined();
    });

    it('skips span logging when record.otlpSpans is absent', async () => {
      const exportOtlpSpans = jest.fn();
      const client = makeClient({ exportOtlpSpans });

      const record = makeRecord({ id: 'row-no-otlp' });
      await uploadOne(record, client, new BatchingWriter());

      expect(exportOtlpSpans).not.toHaveBeenCalled();
    });

    it('tombstones the record even when span logging fails (non-fatal)', async () => {
      const exportOtlpSpans = jest.fn().mockRejectedValue(new Error('501 not supported'));
      const createTrace = jest.fn().mockImplementation((info: TraceInfo) => Promise.resolve(info));
      const uploadTraceData = jest
        .fn()
        .mockImplementation((_info: TraceInfo, _data: TraceData) => Promise.resolve());
      const client = makeClient({ createTrace, uploadTraceData, exportOtlpSpans });

      const record = makeRecord({
        id: 'row-otlp-fail',
        otlpSpans: Buffer.from([0x0a, 0x01]).toString('base64'),
      });
      await uploadOne(record, client, new BatchingWriter());

      expect(createTrace).toHaveBeenCalledTimes(1);
      expect(uploadTraceData).toHaveBeenCalledTimes(1);
      expect(exportOtlpSpans).toHaveBeenCalledTimes(1);

      // The trace info + artifact already uploaded, so a span-log failure must
      // not retry or dead-letter: the row is tombstoned and nothing re-appended.
      const pending = await readPending();
      expect(pending).toHaveLength(0);
    });

    it('re-appends a fresh row with bumped attempts on transient failure', async () => {
      const createTrace = jest.fn().mockRejectedValue(new Error('5xx'));
      const client = makeClient({ createTrace });

      const record = makeRecord({ id: 'row-retry', attempts: 0 });
      const before = Date.now();
      await uploadOne(record, client, new BatchingWriter());
      const after = Date.now();

      const pending = await readPending();
      const retry = pending.find((r) => r.id !== 'row-retry');
      expect(pending.find((r) => r.id === 'row-retry')).toBeUndefined();
      expect(retry).toBeDefined();
      expect(retry!.attempts).toBe(1);
      // backoff(1) = 2_000 ms; the new schedule must be in [before+2s, after+2s].
      expect(retry!.nextAttemptAt).toBeGreaterThanOrEqual(before + 2_000);
      expect(retry!.nextAttemptAt).toBeLessThanOrEqual(after + 2_000);
      // firstAttemptAt is anchored on the first daemon touch and pinned
      // onto the retry row so the wall-clock retry budget is measured
      // from a stable point even across multiple retries / restarts.
      expect(retry!.firstAttemptAt).toBeGreaterThanOrEqual(before);
      expect(retry!.firstAttemptAt).toBeLessThanOrEqual(after);
    });

    it('preserves firstAttemptAt across retries (anchor survives daemon respawn)', async () => {
      const createTrace = jest.fn().mockRejectedValue(new Error('5xx'));
      const client = makeClient({ createTrace });

      // A record a previous daemon already tried twice. A new daemon
      // picking it up must preserve the original anchor, not reset it
      // — otherwise a record could indefinitely outlive the retry
      // budget by piggybacking on each daemon respawn's fresh clock.
      const originalAnchor = Date.now() - 50_000;
      const record = makeRecord({
        id: 'row-restart',
        attempts: 2,
        firstAttemptAt: originalAnchor,
      });
      await uploadOne(record, client, new BatchingWriter());

      const pending = await readPending();
      const retry = pending.find((r) => r.id !== 'row-restart')!;
      expect(retry.firstAttemptAt).toBe(originalAnchor);
      expect(retry.attempts).toBe(3);
    });

    it('dead-letters the record when elapsed retry time exceeds RETRY_TIMEOUT_MS', async () => {
      const createTrace = jest.fn().mockRejectedValue(new Error('still broken'));
      const client = makeClient({ createTrace });

      // Backdate firstAttemptAt by more than the default 500s budget,
      // so the next backoff (~2s) plus the elapsed wall clock blows
      // through the deadline and the record dead-letters immediately
      // regardless of the attempts counter.
      const longAgo = Date.now() - 600_000;
      const record = makeRecord({
        id: 'row-dlq',
        attempts: 3,
        firstAttemptAt: longAgo,
      });
      await uploadOne(record, client, new BatchingWriter());

      const pending = await readPending();
      expect(pending).toHaveLength(0);

      const failedContents = await readFile(getDeadLetterPath(), 'utf8');
      const failedLines = failedContents.trim().split('\n');
      expect(failedLines).toHaveLength(1);
      const parsed = JSON.parse(failedLines[0]) as {
        type: 'append';
        record: WalRecord;
      };
      expect(parsed.type).toBe('append');
      expect(parsed.record.id).toBe('row-dlq');
    });

    // The credential-refresh-on-401/403 retry path. The cached
    // MlflowClient holds an AuthProvider snapshot from daemon start; a
    // token rotation must surface as a transient failure that the daemon
    // recovers from automatically rather than dead-letters forever.
    describe('auth-refresh retry', () => {
      it.each([
        [401, 'Unauthorized'],
        [403, 'Forbidden'],
      ] as const)(
        'evicts the cached client and retries once after a %i, succeeding the second time',
        async (status, statusText) => {
          const staleCreateTrace = jest
            .fn()
            .mockRejectedValue(new MlflowHttpError(status, statusText, ''));
          const freshCreateTrace = jest
            .fn()
            .mockImplementation((info: TraceInfo) => Promise.resolve(info));
          const freshUploadTraceData = jest.fn().mockResolvedValue(undefined);
          const stale = makeClient({ createTrace: staleCreateTrace });
          const fresh = makeClient({
            createTrace: freshCreateTrace,
            uploadTraceData: freshUploadTraceData,
          });
          const factory = jest.fn(() => fresh);

          const record = makeRecord({ id: `row-${status}-retry` });
          await uploadOne(record, stale, new BatchingWriter(), factory);

          // First attempt used the stale client and got the auth error;
          // the auth-refresh branch then asked the factory for a fresh
          // client (exactly once, for this URI) and the retry pushed
          // through successfully.
          expect(staleCreateTrace).toHaveBeenCalledTimes(1);
          expect(factory).toHaveBeenCalledTimes(1);
          expect(factory).toHaveBeenCalledWith(record.trackingUri);
          expect(freshCreateTrace).toHaveBeenCalledTimes(1);
          expect(freshUploadTraceData).toHaveBeenCalledTimes(1);

          // Successful retry: tombstoned, no live row, no dead-letter.
          const pending = await readPending();
          expect(pending.find((r) => r.id === record.id)).toBeUndefined();
        },
      );

      it('falls through to handleUploadFailure when both attempts hit 401', async () => {
        const staleCreateTrace = jest
          .fn()
          .mockRejectedValue(new MlflowHttpError(401, 'Unauthorized', ''));
        const freshCreateTrace = jest
          .fn()
          .mockRejectedValue(new MlflowHttpError(401, 'Unauthorized', ''));
        const stale = makeClient({ createTrace: staleCreateTrace });
        const fresh = makeClient({ createTrace: freshCreateTrace });
        const factory = jest.fn(() => fresh);

        // Backdate firstAttemptAt past the 500s budget so the post-retry
        // failure dead-letters immediately rather than re-appending yet
        // another retry row we'd have to sit on.
        const longAgo = Date.now() - 600_000;
        const record = makeRecord({
          id: 'row-401-persistent',
          firstAttemptAt: longAgo,
        });
        await uploadOne(record, stale, new BatchingWriter(), factory);

        // Both attempts ran; factory was consulted exactly once for the
        // refresh; the second failure routed to dead-letter (not a
        // perpetual invalidate-retry loop).
        expect(staleCreateTrace).toHaveBeenCalledTimes(1);
        expect(freshCreateTrace).toHaveBeenCalledTimes(1);
        expect(factory).toHaveBeenCalledTimes(1);

        const pending = await readPending();
        expect(pending).toHaveLength(0);
        const failedContents = await readFile(getDeadLetterPath(), 'utf8');
        expect(failedContents).toContain('"id":"row-401-persistent"');
      });

      it('dead-letters with the factory error when refresh-time factory throws', async () => {
        const staleCreateTrace = jest
          .fn()
          .mockRejectedValue(new MlflowHttpError(401, 'Unauthorized', ''));
        const stale = makeClient({ createTrace: staleCreateTrace });
        const factory = jest.fn(() => {
          throw new Error('credential provider blew up');
        });

        const longAgo = Date.now() - 600_000;
        const record = makeRecord({
          id: 'row-refresh-factory-throws',
          firstAttemptAt: longAgo,
        });
        await uploadOne(record, stale, new BatchingWriter(), factory);

        // Stale client was tried once and got the auth error; the
        // refresh asked the factory once and that throw aborted the
        // retry before any fresh client call could happen.
        expect(staleCreateTrace).toHaveBeenCalledTimes(1);
        expect(factory).toHaveBeenCalledTimes(1);
        expect(factory).toHaveBeenCalledWith(record.trackingUri);

        const pending = await readPending();
        expect(pending).toHaveLength(0);
        const failedContents = await readFile(getDeadLetterPath(), 'utf8');
        expect(failedContents).toContain('"id":"row-refresh-factory-throws"');
      });

      it('does not refresh the client on non-auth HTTP errors', async () => {
        const createTrace = jest
          .fn()
          .mockRejectedValue(new MlflowHttpError(500, 'Internal Server Error', ''));
        const client = makeClient({ createTrace });
        const factory = jest.fn();

        const record = makeRecord({ id: 'row-5xx' });
        await uploadOne(record, client, new BatchingWriter(), factory);

        // 5xx is not a credential problem — single attempt, no factory
        // consultation, regular retry row appended by handleUploadFailure.
        expect(createTrace).toHaveBeenCalledTimes(1);
        expect(factory).not.toHaveBeenCalled();

        const pending = await readPending();
        const retry = pending.find((r) => r.id !== 'row-5xx');
        expect(pending.find((r) => r.id === 'row-5xx')).toBeUndefined();
        expect(retry).toBeDefined();
        expect(retry!.attempts).toBe(1);
      });

      it('does not refresh the client on transport-level errors', async () => {
        // A non-MlflowHttpError (e.g. fetch DNS / ECONNRESET / abort
        // wrapped by makeRequest into `new Error('API request failed: ...')`)
        // must not be confused with an auth failure — the `instanceof`
        // gate in isAuthRefreshableError is what protects us here.
        const createTrace = jest.fn().mockRejectedValue(new Error('ECONNRESET'));
        const client = makeClient({ createTrace });
        const factory = jest.fn();

        const record = makeRecord({ id: 'row-transport' });
        await uploadOne(record, client, new BatchingWriter(), factory);

        expect(factory).not.toHaveBeenCalled();
        const pending = await readPending();
        const retry = pending.find((r) => r.id !== 'row-transport');
        expect(pending.find((r) => r.id === 'row-transport')).toBeUndefined();
        expect(retry).toBeDefined();
        expect(retry!.attempts).toBe(1);
      });
    });
  });

  describe('processBatch', () => {
    it('groups records by trackingUri and calls the factory once per URI', async () => {
      const clientA = makeClient({});
      const clientB = makeClient({});
      const factory = jest.fn((uri: string) => (uri.endsWith(':5001') ? clientB : clientA));

      const records = [
        makeRecord({ id: 'r1', trackingUri: 'http://localhost:5000' }),
        makeRecord({ id: 'r2', trackingUri: 'http://localhost:5000' }),
        makeRecord({ id: 'r3', trackingUri: 'http://localhost:5001' }),
      ];

      await processBatch(records, new BatchingWriter(), factory);

      expect(factory).toHaveBeenCalledTimes(2);
      expect(factory).toHaveBeenCalledWith('http://localhost:5000');
      expect(factory).toHaveBeenCalledWith('http://localhost:5001');
      const pending = await readPending();
      // All three should have been tombstoned by their respective uploads.
      expect(pending).toHaveLength(0);
    });

    it('no-ops on an empty input', async () => {
      const factory = jest.fn();
      await processBatch([], new BatchingWriter(), factory);
      expect(factory).not.toHaveBeenCalled();
    });

    it('dead-letters records when their client factory throws', async () => {
      const factory = jest.fn(() => {
        throw new Error('auth provider blew up');
      });

      // Backdate firstAttemptAt past the default 500s retry budget so
      // the synthetic factory failure dead-letters immediately instead
      // of just bumping the retry counter.
      const longAgo = Date.now() - 600_000;
      const records = [makeRecord({ id: 'r-bad', firstAttemptAt: longAgo })];
      await processBatch(records, new BatchingWriter(), factory);

      const pending = await readPending();
      expect(pending).toHaveLength(0);

      const failedContents = await readFile(getDeadLetterPath(), 'utf8');
      expect(failedContents).toContain('"id":"r-bad"');
    });
  });

  describe('queue.log layout after retry', () => {
    it('atomically batches the retry append + original tombstone in one fsync', async () => {
      const createTrace = jest.fn().mockRejectedValueOnce(new Error('first attempt fails'));
      const client = makeClient({ createTrace });

      const record = makeRecord({ id: 'row-order', attempts: 0 });
      await uploadOne(record, client, new BatchingWriter());

      const contents = await readFile(getWalPath(), 'utf8');
      const lines = contents.trim().split('\n');
      // `handleUploadFailure`'s retry branch enqueues both the new
      // retry row and the tombstone for the original id on the same
      // `BatchingWriter` tick, so `drainAndFlush` packs them into a
      // single `writev` + `fsync`. The on-disk order matches the
      // enqueue order (append first, then tombstone) and the pair is
      // atomic — either both lines are durable or neither is, so a
      // crash between them is structurally impossible.
      expect(lines).toHaveLength(2);
      const first = JSON.parse(lines[0]) as { type: string; record?: WalRecord };
      const second = JSON.parse(lines[1]) as { type: string; id?: string };
      expect(first.type).toBe('append');
      expect(first.record!.attempts).toBe(1);
      expect(second.type).toBe('tombstone');
      expect(second.id).toBe('row-order');
    });
  });

  describe('pruneOldLogs', () => {
    // All retention tests pin `now` to a fixed UTC date so the cutoff
    // arithmetic stays stable regardless of when CI happens to run.
    const NOW = new Date('2026-05-20T12:00:00.000Z');

    async function seed(names: string[]): Promise<void> {
      for (const name of names) {
        await writeFile(join(walDir, name), 'x');
      }
    }

    function existsInWal(name: string): boolean {
      return existsSync(join(walDir, name));
    }

    // Per-file existence checks are used throughout this block instead of
    // full directory equality because `pruneOldLogs` itself calls `log()`
    // when it unlinks a file, which writes to today's `daemon.log.<date>`
    // and would show up in any whole-listing comparison. We assert the
    // narrower property the function actually owns: which seeded files
    // survive vs disappear.

    it('removes files older than the retention window and keeps newer ones', async () => {
      // retentionDays=14 with NOW=2026-05-20 → keep dates >= 2026-05-06.
      const kept = ['failed.log.2026-05-19', 'failed.log.2026-05-06', 'daemon.log.2026-05-19'];
      const removed = ['failed.log.2026-05-05', 'failed.log.2026-04-01', 'daemon.log.2026-05-05'];
      await seed([...kept, ...removed]);

      await pruneOldLogs({ now: NOW, retentionDays: 14 });

      for (const name of kept) {
        expect(existsInWal(name)).toBe(true);
      }
      for (const name of removed) {
        expect(existsInWal(name)).toBe(false);
      }
    });

    it('is a no-op when retention is 0 (disabled)', async () => {
      const seeded = ['failed.log.2020-01-01', 'daemon.log.2020-01-01'];
      await seed(seeded);

      await pruneOldLogs({ now: NOW, retentionDays: 0 });

      for (const name of seeded) {
        expect(existsInWal(name)).toBe(true);
      }
    });

    it('ignores files that do not match the rotated-log pattern', async () => {
      // Including `queue.log`, the lock socket, files without a date
      // suffix, malformed dates, and unrelated artifacts the operator
      // may have dropped into the WAL dir.
      const seeded = [
        'queue.log',
        'failed.log',
        'daemon.log',
        'failed.log.bogus',
        'failed.log.2026-13-99',
        'failed.log.2026-02-30',
        'README.md',
        'failed.log.2026-05-19.bak',
      ];
      await seed(seeded);

      await pruneOldLogs({ now: NOW, retentionDays: 14 });

      // Every seeded entry must still be there — the regex must not
      // strip-and-reparse, and Date.UTC's overflow-tolerance must not
      // leak into the cutoff comparison.
      for (const name of seeded) {
        expect(existsInWal(name)).toBe(true);
      }
    });

    it('preserves the active queue.log even when the WAL dir is full of expired logs', async () => {
      await seed(['queue.log', 'failed.log.2020-01-01', 'daemon.log.2020-01-01']);

      await pruneOldLogs({ now: NOW, retentionDays: 14 });

      expect(existsInWal('queue.log')).toBe(true);
      expect(existsInWal('failed.log.2020-01-01')).toBe(false);
      expect(existsInWal('daemon.log.2020-01-01')).toBe(false);
    });

    it('does not throw when the WAL directory is missing', async () => {
      // Simulate "daemon started, then someone rm -rf'd the dir before
      // the sweep ran". The sweep must log-and-continue, never throw.
      process.env.MLFLOW_WAL_DIR = join(walDir, 'does-not-exist');
      await expect(pruneOldLogs({ now: NOW, retentionDays: 14 })).resolves.toBeUndefined();
    });
  });

  describe('runBatchLoop idle semantics', () => {
    it('shuts down via idleMs when the WAL is empty', async () => {
      // Baseline: no records seeded → due.length === 0 every
      // iteration → idle-shutdown branch fires on the first iteration
      // after `idleMs` elapses.
      const factory = jest.fn();
      const start = Date.now();
      await runBatchLoop({
        factory,
        writer: new BatchingWriter(),
        batchIntervalMs: 10,
        idleMs: 50,
      });
      const elapsed = Date.now() - start;
      expect(elapsed).toBeGreaterThanOrEqual(50);
      // Generous upper bound — guards against an accidental infinite
      // loop while tolerating CI scheduler jitter.
      expect(elapsed).toBeLessThan(2_000);
      expect(factory).not.toHaveBeenCalled();
    });

    it('shuts down via idleMs when all pending records are future-dated', async () => {
      // Regression test for the bug where `pending.length > 0` short-
      // circuited the idle-shutdown check, pinning the daemon alive
      // until each retry matured even though nothing was actually
      // being worked on. The fix gates the idle check on
      // `due.length === 0`, so a WAL populated only by future-dated
      // retries is treated as not-actively-working and the loop
      // winds down through the normal `idleMs` path.
      await appendRecord(makeRecord({ id: 'row-future', nextAttemptAt: Date.now() + 60_000 }));
      const factory = jest.fn();
      const start = Date.now();
      await runBatchLoop({
        factory,
        writer: new BatchingWriter(),
        batchIntervalMs: 10,
        idleMs: 50,
      });
      const elapsed = Date.now() - start;
      expect(elapsed).toBeGreaterThanOrEqual(50);
      expect(elapsed).toBeLessThan(2_000);
      expect(factory).not.toHaveBeenCalled();
      // The record is still durably parked in the WAL: the supervisor's
      // next respawn (driven by a fresh IPC submission) will pick it
      // up and either retry or dead-letter against the original
      // `firstAttemptAt` budget — at-least-once delivery preserved.
      expect(await readPending()).toHaveLength(1);
    });
  });
});
