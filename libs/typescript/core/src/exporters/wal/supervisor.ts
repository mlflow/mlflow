/**
 * Daemon supervisor - handles the "is the uploader daemon alive? if not,
 * spawn one" check invoked by `WalSpanExporter` on each WAL append.
 */

import { spawn } from 'node:child_process';
import { createConnection } from 'node:net';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { getLockSocketPath } from './paths';

const PROBE_TIMEOUT_MS = 1000;

/**
 * Probe the daemon's liveness lock with a short timeout.
 *
 * Returns:
 * - `true` if the socket/pipe at `getLockSocketPath()` is currently
 *   accepting connections (i.e. a daemon has bound it).
 * - `false` on any error (ENOENT — no socket file; ECONNREFUSED — a
 *   stale socket file with no listener) or on probe timeout.
 */
export function isDaemonAlive(): Promise<boolean> {
  return new Promise((resolve) => {
    const socketPath = getLockSocketPath();
    const socket = createConnection(socketPath);

    let settled = false;
    const finish = (alive: boolean): void => {
      if (settled) {
        return;
      }
      settled = true;
      socket.destroy();
      resolve(alive);
    };

    const timer = setTimeout(() => finish(false), PROBE_TIMEOUT_MS);

    socket.once('connect', () => {
      clearTimeout(timer);
      finish(true);
    });
    socket.once('error', () => {
      clearTimeout(timer);
      finish(false);
    });
  });
}

/**
 * Resolve the absolute path of the daemon entry point.
 *
 * Precedence:
 *   1. `MLFLOW_TRACE_DAEMON_EXECUTABLE` env var — used by tests and
 *      advanced operators who want to override the bundled daemon.
 *   2. The bundled daemon at `<@mlflow/core install dir>/bundle/daemon.cjs`.
 */
export function resolveDaemonEntry(): string {
  const override = process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE;
  if (override !== undefined && override !== '') {
    return override;
  }
  // DO NOT collapse this into a plain string literal. The concatenation
  // hides the specifier from esbuild's static scanner (which walks every
  // literal passed to `require.resolve(...)`) so the lookup stays
  // runtime-only. The design requires the daemon to run in its own
  // long-lived process - separate from any individual hook invocation -
  // so the hook can exit immediately after appending to the WAL while
  // the daemon handles uploads, retries, and group-commit batching
  // across many concurrent hooks.
  //
  // If a literal specifier appeared here, a consumer re-bundling
  // @mlflow/core into their own hook (e.g. claude-code's
  // `bundle/stop.cjs`) would suffer two bundler-induced regressions:
  //   (a) esbuild inlines @mlflow/core's package.json contents into the
  //       consumer bundle, so `dirname(pkgJsonPath)` resolves to the
  //       *consumer's* bundle dir instead of @mlflow/core's install
  //       dir, and `join(..., 'bundle', 'daemon.cjs')` points at a
  //       file that does not exist.
  //   (b) esbuild follows the path forward and inlines the entire
  //       `bundle/daemon.cjs` source into the hook bundle - bloating
  //       it by many MB (OTel SDKs, Databricks SDK, HTTP stack, retry
  //       logic) and making it tempting to `require()` the daemon
  //       in-process, which would re-couple hook latency to backend
  //       latency and break the "one daemon per host, N hooks" model.
  // Both outcomes are bad; the string-concat prevents both.
  const pkgJsonModule = '@mlflow' + '/core' + '/package.json';
  const requireFromHere = createRequire(__filename);
  const pkgJsonPath = requireFromHere.resolve(pkgJsonModule);
  return join(dirname(pkgJsonPath), 'bundle', 'daemon.cjs');
}

/**
 * Fork a detached daemon process. Assumes no daemon is currently bound —
 * the caller is responsible for guarding with {@link isDaemonAlive}.
 */
export function spawnDaemon(): void {
  const entry = resolveDaemonEntry();
  const child = spawn(process.execPath, [entry], {
    detached: true,
    stdio: 'ignore',
    env: process.env,
  });

  child.once('error', (err) => {
    console.error(`[mlflow][wal] daemon spawn failed (entry=${entry}):`, err);
  });

  child.once('exit', (code, signal) => {
    if (code != null && code !== 0) {
      console.error(
        `[mlflow][wal] daemon exited unexpectedly: entry=${entry} code=${code} signal=${signal ?? 'none'}`,
      );
    }
  });
  child.unref();
}

let spawnInFlight: Promise<void> | null = null;

/**
 * Ensure a daemon is alive; spawn one if not.
 */
export function ensureDaemon(): Promise<void> {
  if (spawnInFlight != null) {
    return spawnInFlight;
  }
  const inFlight = (async () => {
    try {
      if (await isDaemonAlive()) {
        return;
      }
      spawnDaemon();
    } catch (err) {
      console.error('[mlflow][wal] ensureDaemon failed:', err);
    } finally {
      spawnInFlight = null;
    }
  })();
  spawnInFlight = inFlight;
  return inFlight;
}
