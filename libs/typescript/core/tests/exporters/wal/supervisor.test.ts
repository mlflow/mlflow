import * as childProcess from 'node:child_process';
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

jest.mock('node:child_process', () => {
  const actual = jest.requireActual<typeof import('node:child_process')>('node:child_process');
  return {
    ...actual,
    spawn: jest.fn((...args: Parameters<typeof actual.spawn>) => actual.spawn(...args)),
  };
});

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

    it('swallows broken-entry failures and surfaces a diagnostic to stderr', async () => {
      // Broken-install failures surface asynchronously, *after*
      // `ensureDaemon` has already resolved (the spawn itself is
      // fire-and-forget by design). The listener in `spawnDaemon` must:
      //   1. let `ensureDaemon` resolve cleanly (no upstream crash);
      //   2. log a breadcrumb mentioning the broken entry so a broken
      //      install (missing `bundle/daemon.cjs`, bad
      //      `MLFLOW_TRACE_DAEMON_EXECUTABLE`) doesn't degrade into
      //      ~3.85 s of opaque ECONNREFUSED retries in the WAL exporter
      //      with zero indication of cause.
      //
      // Two child events can carry the diagnostic and the listener
      // hooks both: `'error'` fires when libuv can't start the binary
      // at all (we can't easily trigger this here because `spawnDaemon`
      // runs `spawn(process.execPath, [entry], ...)` and the node
      // binary always exists), and `'exit'` fires with a non-zero code
      // when the binary starts but the entry script fails to load —
      // which is what pointing `MLFLOW_TRACE_DAEMON_EXECUTABLE` at a
      // nonexistent `.js` file actually triggers. The test accepts
      // either: both equally serve the reviewer-visible "broken
      // install" purpose.
      const missing = join(walDir, 'does-not-exist.js');
      process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE = missing;
      const errSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      try {
        await expect(ensureDaemon()).resolves.toBeUndefined();
        // Generous timeout: node cold-start + script-resolution failure
        // + exit can comfortably exceed 1 s on a busy CI box.
        await waitFor(
          () =>
            errSpy.mock.calls.some(
              ([msg]) =>
                typeof msg === 'string' &&
                msg.includes(missing) &&
                /daemon (spawn failed|exited unexpectedly)/.test(msg),
            ),
          5000,
        );
        const match = errSpy.mock.calls.find(
          ([msg]) =>
            typeof msg === 'string' &&
            msg.includes(missing) &&
            /daemon (spawn failed|exited unexpectedly)/.test(msg),
        );
        expect(match).toBeDefined();
        expect(match![0]).toContain(missing);
      } finally {
        errSpy.mockRestore();
      }
    });

    it('catches synchronous spawn failures via the outer ensureDaemon catch', async () => {
      // The per-child `'error'` / `'exit'` listeners installed by
      // `spawnDaemon` only fire for *async* failures (the binary started
      // but the entry script blew up). Synchronous failures from
      // `spawn()` itself — libuv refusing the args, or
      // `resolveDaemonEntry()` throwing on a broken package-resolution
      // path — rely on the outer try/catch in `ensureDaemon`'s IIFE.
      // Without that catch, `spawnInFlight` would settle as a rejected
      // promise and hooks would crash on cold spawn failures the
      // per-child listener can't help with.
      //
      // The top-of-file `jest.mock('node:child_process', ...)` installs
      // a passthrough mock; here we override the next `spawn` call
      // (and only that one) with a synchronous throw. The override
      // is scoped to this test via `mockImplementationOnce`, so the
      // adjacent coalescing test below sees real spawn behavior again.
      const spawnMock = childProcess.spawn as jest.MockedFunction<typeof childProcess.spawn>;
      spawnMock.mockImplementationOnce(() => {
        throw new Error('synthetic synchronous spawn failure');
      });
      const errSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      try {
        // The outer catch must convert the synchronous throw into a
        // clean resolution + a diagnostic log line. Anything else
        // (rejection, unhandled crash, no log) is a regression in the
        // defense-in-depth path.
        await expect(ensureDaemon()).resolves.toBeUndefined();
        expect(spawnMock).toHaveBeenCalled();
        expect(
          errSpy.mock.calls.some(
            ([msg]) => typeof msg === 'string' && msg.includes('ensureDaemon failed'),
          ),
        ).toBe(true);
      } finally {
        errSpy.mockRestore();
      }
    });

    it('coalesces concurrent calls into a single spawn', async () => {
      // The realistic trigger is multiple `submitRecord` calls in one
      // hook all hitting ECONNREFUSED during a daemon cold-start. Each
      // would otherwise spawn its own child; the in-flight latch should
      // dedupe them down to a single `spawn()` invocation.
      const concurrent = 5;
      await Promise.all(Array.from({ length: concurrent }, () => ensureDaemon()));
      await waitFor(() => existsSync(counterFile));
      // Give any (errant) extra spawns time to write to the counter.
      await sleep(200);

      const counterContents = await readFile(counterFile, 'utf8');
      expect(counterContents).toBe('+');
    });
  });
});
