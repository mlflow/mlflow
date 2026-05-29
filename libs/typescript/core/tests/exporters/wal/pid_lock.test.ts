import { spawn } from 'child_process';
import { existsSync } from 'fs';
import { mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { readLockHolderPid, tryAcquirePidLock } from '../../../src/exporters/wal/pid_lock';
import { getPidLockPath } from '../../../src/exporters/wal/paths';

// The lock is a POSIX-only mechanism — Windows daemons rely on
// kernel-refcounted named pipes for cross-process exclusion, so the
// entire suite is a no-op there. Wrap with a describe variant that
// skips on Windows to keep the test report honest about coverage.
const describePosix = process.platform === 'win32' ? describe.skip : describe;

/**
 * Spawn a trivial child, wait for it to exit, and return its now-freed
 * pid. Used to obtain a pid that is guaranteed to fail
 * `process.kill(pid, 0)` with ESRCH so the stale-recovery path can be
 * exercised deterministically. The PID could in principle be recycled
 * by the OS between the child exiting and our liveness check, but in a
 * single-digit-millisecond test window the probability is negligible
 * — and even if it happened, the test would falsely fail rather than
 * silently pass, which is the safe failure mode.
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

describePosix('wal/pid_lock', () => {
  let walDir: string;
  // Mirror daemon.test.ts: capture the developer's pre-test value so
  // running `MLFLOW_WAL_DIR=/some/dir jest` doesn't lose the override
  // for later tests in the same worker.
  const originalWalDir = process.env.MLFLOW_WAL_DIR;

  beforeEach(async () => {
    walDir = await mkdtemp(join(tmpdir(), 'mlflow-pidlock-'));
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

  describe('tryAcquirePidLock', () => {
    it('claims the lock and writes our pid when no lockfile exists', async () => {
      const lock = await tryAcquirePidLock();
      try {
        expect(lock).not.toBeNull();
        const content = await readFile(getPidLockPath(), 'utf8');
        expect(content).toBe(`${process.pid}\n`);
      } finally {
        await lock?.release();
      }
    });

    it('returns null when an alive process already holds the lock', async () => {
      // Seed the lockfile with our own pid — guaranteed alive — so the
      // liveness probe inside tryAcquirePidLock has to detect us and
      // concede rather than steal the lock from ourselves.
      await writeFile(getPidLockPath(), `${process.pid}\n`);
      const lock = await tryAcquirePidLock();
      expect(lock).toBeNull();
      // Lockfile must remain untouched after a conceding acquisition.
      const content = await readFile(getPidLockPath(), 'utf8');
      expect(content).toBe(`${process.pid}\n`);
    });

    it('recovers from a stale lockfile whose pid is dead', async () => {
      const deadPid = await spawnAndReapPid();
      await writeFile(getPidLockPath(), `${deadPid}\n`);
      const lock = await tryAcquirePidLock();
      try {
        expect(lock).not.toBeNull();
        const content = await readFile(getPidLockPath(), 'utf8');
        expect(content).toBe(`${process.pid}\n`);
      } finally {
        await lock?.release();
      }
    });

    it('recovers from a malformed lockfile (non-numeric content)', async () => {
      await writeFile(getPidLockPath(), 'this is not a pid\n');
      const lock = await tryAcquirePidLock();
      try {
        expect(lock).not.toBeNull();
        const content = await readFile(getPidLockPath(), 'utf8');
        expect(content).toBe(`${process.pid}\n`);
      } finally {
        await lock?.release();
      }
    });

    it('recovers from an empty lockfile', async () => {
      await writeFile(getPidLockPath(), '');
      const lock = await tryAcquirePidLock();
      try {
        expect(lock).not.toBeNull();
      } finally {
        await lock?.release();
      }
    });

    it('rejects a lockfile that records pid 0 (process-group signal target)', async () => {
      // pid 0 would make process.kill(0, 0) signal our entire process
      // group; parsePid must screen it out before the liveness probe
      // ever runs. Recovery should treat it as malformed and reclaim.
      await writeFile(getPidLockPath(), '0\n');
      const lock = await tryAcquirePidLock();
      try {
        expect(lock).not.toBeNull();
        const content = await readFile(getPidLockPath(), 'utf8');
        expect(content).toBe(`${process.pid}\n`);
      } finally {
        await lock?.release();
      }
    });

    it('rejects a lockfile that records a negative pid (process-group signal target)', async () => {
      await writeFile(getPidLockPath(), '-12345\n');
      const lock = await tryAcquirePidLock();
      try {
        expect(lock).not.toBeNull();
      } finally {
        await lock?.release();
      }
    });

    it('serializes concurrent acquirers in the same event loop to a single winner', async () => {
      // Both acquirers run in the same Node process so they see the
      // same `process.pid`. The link() atomicity is what arbitrates;
      // the loser observes EEXIST, reads our own (alive) pid, and
      // concedes via the live-holder branch.
      const [first, second] = await Promise.all([tryAcquirePidLock(), tryAcquirePidLock()]);
      try {
        const acquired = [first, second].filter((l) => l != null);
        const conceded = [first, second].filter((l) => l == null);
        expect(acquired).toHaveLength(1);
        expect(conceded).toHaveLength(1);
      } finally {
        await first?.release();
        await second?.release();
      }
    });

    it('cleans up the temp file on a successful acquisition', async () => {
      const lock = await tryAcquirePidLock();
      try {
        // No `daemon.pid.<pid>.<uuid>.tmp` should survive a successful
        // acquire — the implementation unlinks it after the link()
        // succeeds so the WAL dir doesn't accumulate per-spawn garbage.
        const { readdir } = await import('fs/promises');
        const entries = await readdir(walDir);
        const tmps = entries.filter((e) => e.startsWith('daemon.pid.') && e.endsWith('.tmp'));
        expect(tmps).toEqual([]);
      } finally {
        await lock?.release();
      }
    });
  });

  describe('PidLock.release', () => {
    it('unlinks the lockfile when content still matches our pid', async () => {
      const lock = await tryAcquirePidLock();
      expect(lock).not.toBeNull();
      await lock!.release();
      expect(existsSync(getPidLockPath())).toBe(false);
    });

    it('leaves the lockfile intact when content has been hijacked', async () => {
      const lock = await tryAcquirePidLock();
      expect(lock).not.toBeNull();
      // Simulate an external rewrite (operator intervention, or a
      // successor that erroneously stole the path while we were still
      // running). release() must not blind-unlink the file in this
      // case — that would erase the successor's claim.
      await writeFile(getPidLockPath(), '99999\n');
      await lock!.release();
      expect(existsSync(getPidLockPath())).toBe(true);
      const content = await readFile(getPidLockPath(), 'utf8');
      expect(content).toBe('99999\n');
    });

    it('is idempotent', async () => {
      const lock = await tryAcquirePidLock();
      expect(lock).not.toBeNull();
      await lock!.release();
      // Second release: lockfile already gone, must be a no-op.
      await expect(lock!.release()).resolves.toBeUndefined();
      expect(existsSync(getPidLockPath())).toBe(false);
    });
  });

  describe('readLockHolderPid', () => {
    it('returns the recorded pid when a lockfile exists', async () => {
      await writeFile(getPidLockPath(), '4242\n');
      await expect(readLockHolderPid()).resolves.toBe(4242);
    });

    it('returns null when no lockfile exists', async () => {
      await expect(readLockHolderPid()).resolves.toBeNull();
    });

    it('returns null when the lockfile is malformed', async () => {
      await writeFile(getPidLockPath(), 'garbage\n');
      await expect(readLockHolderPid()).resolves.toBeNull();
    });
  });
});
