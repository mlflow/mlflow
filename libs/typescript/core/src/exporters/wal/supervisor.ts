/**
 * Daemon supervisor - handles the "is the uploader daemon alive? if not,
 * spawn one" check invoked by `MlflowWalSpanExporter` on each WAL append.
 */

import { spawn } from 'node:child_process';
import { createConnection } from 'node:net';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { getLockSocketPath } from './paths';

const PROBE_TIMEOUT_MS = 1000;

/**
 * Upper bound on how long {@link ensureDaemon} waits for a freshly
 * spawned daemon to bind its lock socket.
 */
const SPAWN_BIND_WAIT_TIMEOUT_MS = 2000;
const SPAWN_BIND_WAIT_POLL_MS = 50;

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
  // DO NOT collapse this into a plain string literal. esbuild
  // evaluates `require.resolve(literal)` at build time and bakes the
  // resolved (developer-machine) path directly into the shipped
  // bundle. On end users' machines that path does not exist, so
  // `spawn` fails with ENOENT and traces silently stop uploading
  // while the WAL grows unbounded. The string-concat hides the
  // specifier from esbuild's AST scanner so the lookup runs at
  // runtime against the user's own node_modules.
  //
  // This defense lives in the supervisor (not in consumers' esbuild
  // configs) because every plugin-marketplace consumer must bundle
  // @mlflow/core into their hook to ship as a self-contained file.
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
 * Poll {@link isDaemonAlive} until it returns true or `timeoutMs`
 * elapses. Does not throw on timeout — the IPC client's own retry
 * loop will surface any persistent connection failure.
 */
async function waitForDaemonBind(timeoutMs: number, pollIntervalMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await isDaemonAlive()) {
      return;
    }
    await new Promise<void>((resolve) => setTimeout(resolve, pollIntervalMs));
  }
}

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
      await waitForDaemonBind(SPAWN_BIND_WAIT_TIMEOUT_MS, SPAWN_BIND_WAIT_POLL_MS);
    } catch (err) {
      console.error('[mlflow][wal] ensureDaemon failed:', err);
    } finally {
      spawnInFlight = null;
    }
  })();
  spawnInFlight = inFlight;
  return inFlight;
}
