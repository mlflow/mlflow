import { existsSync } from 'fs';
import { mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import { createServer, Server } from 'net';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  ensureDaemon,
  isDaemonAlive,
  resolveDaemonEntry,
} from '../../../src/exporters/wal/supervisor';
import { getLockSocketPath } from '../../../src/exporters/wal/paths';

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(
  predicate: () => boolean | Promise<boolean>,
  timeoutMs = 2000,
): Promise<void> {
  const start = Date.now();
  for (;;) {
    if (await predicate()) {
      return;
    }
    if (Date.now() - start > timeoutMs) {
      throw new Error(`waitFor: predicate did not become true within ${timeoutMs}ms`);
    }
    await sleep(20);
  }
}

describe('wal/supervisor', () => {
  let walDir: string;

  const originalWalDir = process.env.MLFLOW_WAL_DIR;
  const originalDaemonExecutable = process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE;

  beforeEach(async () => {
    walDir = await mkdtemp(join(tmpdir(), 'mlflow-supervisor-'));
    process.env.MLFLOW_WAL_DIR = walDir;
  });

  afterEach(async () => {
    if (originalWalDir === undefined) {
      delete process.env.MLFLOW_WAL_DIR;
    } else {
      process.env.MLFLOW_WAL_DIR = originalWalDir;
    }
    if (originalDaemonExecutable === undefined) {
      delete process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE;
    } else {
      process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE = originalDaemonExecutable;
    }
    await rm(walDir, { recursive: true, force: true });
  });

  describe('isDaemonAlive', () => {
    it('returns false when no socket is bound', async () => {
      expect(await isDaemonAlive()).toBe(false);
    });

    it('returns true when a server is bound to the lock socket', async () => {
      const server: Server = createServer((s) => s.end());
      await new Promise<void>((resolve) => server.listen(getLockSocketPath(), resolve));
      try {
        expect(await isDaemonAlive()).toBe(true);
      } finally {
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    });
  });

  describe('resolveDaemonEntry', () => {
    it('returns the value of MLFLOW_TRACE_DAEMON_EXECUTABLE when set', () => {
      process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE = '/custom/path/to/daemon.cjs';
      expect(resolveDaemonEntry()).toBe('/custom/path/to/daemon.cjs');
    });

    it('treats an empty MLFLOW_TRACE_DAEMON_EXECUTABLE as unset', () => {
      process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE = '';
      // With the env var empty, the resolver falls through to package
      // resolution. We can't predict the absolute path, but it must end
      // in `bundle/daemon.cjs` per the documented contract.
      const resolved = resolveDaemonEntry();
      expect(resolved.endsWith(join('bundle', 'daemon.cjs'))).toBe(true);
    });
  });

  describe('ensureDaemon', () => {
    let stubScript: string;
    let counterFile: string;
    let pidFile: string;

    beforeEach(async () => {
      stubScript = join(walDir, 'stub-daemon.js');
      counterFile = join(walDir, 'spawns.log');
      pidFile = join(walDir, 'pids.log');

      // Minimal CJS script: record that we spawned, bind the lock
      // socket, and self-terminate after a short window so a failing
      // test cannot leak indefinitely. fs writes happen *before* the
      // listen() call so the counter / pid record is durable even if
      // the parent has already moved on by the time we bind.
      const lockPath = getLockSocketPath();
      const stubSource = [
        "const net = require('net');",
        "const fs = require('fs');",
        `fs.appendFileSync(${JSON.stringify(counterFile)}, '+');`,
        `fs.appendFileSync(${JSON.stringify(pidFile)}, process.pid + '\\n');`,
        'const server = net.createServer((s) => s.end());',
        `server.listen(${JSON.stringify(lockPath)}, () => {`,
        '  setTimeout(() => process.exit(0), 3000);',
        '});',
      ].join('\n');
      await writeFile(stubScript, stubSource);
      process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE = stubScript;
    });

    afterEach(async () => {
      // Kill any stub daemons that haven't self-destructed yet so
      // subsequent tests don't see stale processes / sockets.
      if (existsSync(pidFile)) {
        const pids = (await readFile(pidFile, 'utf8')).split('\n').filter((l) => l.length > 0);
        for (const pid of pids) {
          try {
            process.kill(Number.parseInt(pid, 10), 'SIGKILL');
          } catch {
            // Already exited.
          }
        }
      }
    });

    it('spawns a daemon when none is alive', async () => {
      await ensureDaemon();
      await waitFor(() => existsSync(counterFile));
      expect(await readFile(counterFile, 'utf8')).toBe('+');
    });

    it('does not spawn a second daemon when one is already alive', async () => {
      await ensureDaemon();
      await waitFor(() => existsSync(counterFile));
      // Wait until the first stub has actually bound the socket; only
      // after that will the second probe see "alive".
      await waitFor(() => isDaemonAlive(), 3000);

      await ensureDaemon();
      // Give any (errant) second spawn time to write to the counter.
      await sleep(200);

      expect(await readFile(counterFile, 'utf8')).toBe('+');
    });

    it('logs and swallows errors instead of throwing', async () => {
      // Point the resolver at a nonexistent file so spawn fails (the
      // child can't load it). ensureDaemon must still resolve cleanly.
      process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE = join(walDir, 'does-not-exist.js');
      const errSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      try {
        await expect(ensureDaemon()).resolves.toBeUndefined();
      } finally {
        errSpy.mockRestore();
      }
    });
  });
});
