/**
 * Cross-process exclusion lock for the daemon singleton.
 *
 * The lock is POSIX-only; on Windows the daemon's named-pipe path is
 * kernel-refcounted and provides cross-process exclusion natively, so
 * `acquireLock` skips this module on `process.platform === 'win32'`.
 */

import { randomUUID } from 'node:crypto';
import { link, readFile, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { getPidLockPath, getWalDir } from './paths';

const MAX_ACQUIRE_ATTEMPTS = 5;

function isErrnoException(err: unknown): err is NodeJS.ErrnoException {
  return typeof err === 'object' && err != null && 'code' in err;
}

export interface PidLock {
  release(): Promise<void>;
}

/**
 * Attempt to claim the daemon's cross-process PID lock.
 *
 * Returns a {@link PidLock} on success, or `null` when another live
 * process already holds it. Throws on unrecoverable I/O errors
 * (`EACCES`, `ENOSPC`, missing WAL dir, etc.) — callers should treat
 * a throw as "daemon cannot start" rather than "concede to sibling."
 */
export async function tryAcquirePidLock(): Promise<PidLock | null> {
  const lockPath = getPidLockPath();
  const expectedContent = `${process.pid}\n`;

  for (let attempt = 0; attempt < MAX_ACQUIRE_ATTEMPTS; attempt++) {
    
    const tmpPath = join(getWalDir(), `daemon.pid.${process.pid}.${randomUUID()}.tmp`);

    await writeFile(tmpPath, expectedContent, { flag: 'wx' });

    try {
      await link(tmpPath, lockPath);
      await unlink(tmpPath).catch(() => {});
      return { release: () => releaseLockFile(lockPath, expectedContent) };
    } catch (err) {
      await unlink(tmpPath).catch(() => {});
      if (!isErrnoException(err) || err.code !== 'EEXIST') {
        throw err;
      }
    }

    // EEXIST path. Inspect the holder to decide between conceding and
    // recovering from a stale lockfile left by a crashed daemon.
    let content: string;
    try {
      content = await readFile(lockPath, 'utf8');
    } catch (err) {
      if (isErrnoException(err) && err.code === 'ENOENT') {
        // Holder released between our link attempt and our read; the
        // path is free again, loop and reattempt the link.
        continue;
      }
      throw err;
    }

    const holderPid = parsePid(content);
    if (holderPid != null && isProcessAlive(holderPid)) {
      return null;
    }

    // Stale or malformed. Unlink the lockfile, but only if its content
    // is still the same bytes we just read - otherwise a successor has
    // already taken over and we would nuke their fresh lockfile.
    const fresh = await readFile(lockPath, 'utf8').catch(() => null);
    if (fresh === content) {
      await unlink(lockPath).catch(() => {});
    }
  }

  throw new Error(
    `tryAcquirePidLock: gave up after ${MAX_ACQUIRE_ATTEMPTS} contention attempts on ${lockPath}`,
  );
}

/**
 * Read the pid recorded in the daemon's lockfile, or `null` if the
 * file is missing or unparseable. Provided as a diagnostic helper for
 * callers (e.g. supervisor) that want to log "who owns the lock right
 * now"; not used by acquisition itself.
 */
export async function readLockHolderPid(): Promise<number | null> {
  try {
    const content = await readFile(getPidLockPath(), 'utf8');
    return parsePid(content);
  } catch (err) {
    if (isErrnoException(err) && err.code === 'ENOENT') {
      return null;
    }
    throw err;
  }
}

function parsePid(content: string): number | null {
  const trimmed = content.trim();
  if (trimmed === '') {
    return null;
  }
  const n = Number.parseInt(trimmed, 10);
  // PIDs are positive integers; reject 0 (which `process.kill` would
  // interpret as "signal every process in our process group") and
  // negatives (which would target a process group).
  return Number.isInteger(n) && n > 0 ? n : null;
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    // `EPERM` means the process exists but we lack permission to
    // signal it (different uid). Treat as alive — the lockfile's
    // owner is alive enough that we should not unlink their lock.
    if (isErrnoException(err) && err.code === 'EPERM') {
      return true;
    }
    return false;
  }
}

async function releaseLockFile(lockPath: string, expectedContent: string): Promise<void> {
  const fresh = await readFile(lockPath, 'utf8').catch(() => null);
  if (fresh === expectedContent) {
    await unlink(lockPath).catch(() => {});
  }
}
