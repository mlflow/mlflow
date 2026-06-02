/**
 * Hook ↔ daemon IPC over the daemon's lock socket / Named Pipe.
 *
 * Client side ({@link submitRecord}):
 *   1. Connect to {@link getLockSocketPath}.
 *   2. Write a single newline-terminated JSON request.
 *   3. Read a single newline-terminated JSON response.
 *   4. Close.
 * The promise resolves only after the daemon has fsynced the record's
 * byte to `queue.log`.
 *
 * Server side ({@link createIpcConnectionHandler}):
 *   - Wait for one line of data on the connection.
 *   - Parse it as an {@link IpcRequest}.
 *   - Dispatch the `append` op into the daemon's {@link BatchingWriter}.
 *   - Once the writer's promise settles, write a one-line
 *     {@link IpcResponse} and close.
 *   - Connections that close before sending a line are liveness probes
 *     (used by {@link isDaemonAlive}); we let them go silently.
 */

import { createConnection, Socket } from 'node:net';
import { JSONBig } from '../../core/utils/json';
import { BatchingWriter } from './batching_writer';
import { getLockSocketPath } from './paths';
import { IpcRequest, IpcResponse } from './protocol';
import { ensureDaemon } from './supervisor';
import { WalRecord } from './types';

/**
 * Upper bound on a single submit round-trip (connect + write + ACK).
 */
const SUBMIT_TIMEOUT_MS = 10_000;

const INITIAL_CONNECT_RETRY_DELAY_MS = 50;
const CONNECT_RETRY_DELAYS_MS = Array.from(
  { length: 6 },
  (_, i) => INITIAL_CONNECT_RETRY_DELAY_MS * 2 ** i,
);

const DAEMON_NO_RESPONSE_CODE = 'EDAEMONNORESPONSE';

const MAX_REQUEST_BYTES = 16 * 1024 * 1024;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isConnectError(err: unknown): boolean {
  if (typeof err !== 'object' || err == null) {
    return false;
  }
  const code = (err as { code?: string }).code;
  return (
    code === 'ECONNREFUSED' ||
    code === 'ENOENT' ||
    code === 'EPIPE' ||
    code === 'ECONNRESET' ||
    code === DAEMON_NO_RESPONSE_CODE
  );
}

/**
 * Send `record` to the running daemon and wait for the post-fsync ACK.
 *
 * Delivery semantics: **at-least-once**.
 *
 * A successful resolution means the record was durably fsynced to
 * `queue.log` at least once. A rejection means the daemon could not be
 * acknowledged after 7 attempts (1 initial + 6 backoffs from
 * `CONNECT_RETRY_DELAYS_MS`), but the daemon may still have persisted
 * the record on one of those attempts before the failure surfaced.
 *
 * The internal retry loop retries on five connect-class codes:
 * `ECONNREFUSED`, `ENOENT`, `EPIPE`, `ECONNRESET`, and
 * `EDAEMONNORESPONSE`. The first two prove the request never reached a
 * daemon and are safe; the last three can fire *after* the daemon has
 * already fsynced (e.g. daemon crashed between fsync and ACK, or the
 * socket reset post-fsync), so a single logical submit can produce
 * more than one physical line in `queue.log` and more than one
 * `createTrace` call to the backend.
 *
 */
export async function submitRecord(record: WalRecord): Promise<void> {
  const payload = JSONBig.stringify({ op: 'append', record } satisfies IpcRequest) + '\n';
  let lastErr: unknown;
  for (let i = 0; i <= CONNECT_RETRY_DELAYS_MS.length; i++) {
    if (i > 0) {
      await sleep(CONNECT_RETRY_DELAYS_MS[i - 1]);
    }
    try {
      const response = await sendRequest(payload);
      if (response.ok) {
        return;
      }
      throw new Error(`Daemon rejected record: ${response.error}`);
    } catch (err) {
      lastErr = err;
      if (!isConnectError(err)) {
        throw err;
      }
      // Fire-and-forget spawn; the next loop iteration's connect will
      // observe the bound socket once the daemon finishes binding.
      ensureDaemon().catch(() => {});
    }
  }

  throw lastErr as Error;
}

/**
 * One IPC round-trip against the daemon's lock socket.
 */
function sendRequest(payload: string): Promise<IpcResponse> {
  return new Promise<IpcResponse>((resolve, reject) => {
    const socket = createConnection(getLockSocketPath());
    const chunks: Buffer[] = [];
    let settled = false;

    const finish = (act: () => void): void => {
      if (settled) {
        return;
      }
      settled = true;
      socket.destroy();
      act();
    };

    const timer = setTimeout(
      () => finish(() => reject(new Error(`IPC submit timed out after ${SUBMIT_TIMEOUT_MS}ms`))),
      SUBMIT_TIMEOUT_MS,
    );

    socket.once('connect', () => {
      socket.write(payload);
    });
    socket.on('data', (chunk) => chunks.push(chunk));
    socket.once('end', () => {
      clearTimeout(timer);
      const text = Buffer.concat(chunks).toString('utf8').trimEnd();
      if (text === '') {
        finish(() =>
          reject(
            Object.assign(new Error('Daemon closed connection without responding'), {
              code: DAEMON_NO_RESPONSE_CODE,
            }),
          ),
        );
        return;
      }
      try {
        const parsed = JSONBig.parse(text) as IpcResponse;
        finish(() => resolve(parsed));
      } catch (err) {
        finish(() => reject(err as Error));
      }
    });
    socket.once('error', (err) => {
      clearTimeout(timer);
      finish(() => reject(err));
    });
  });
}

/**
 * Per-connection state for an in-flight IPC request. One instance per
 * accepted socket.
 */
function handleConnection(writer: BatchingWriter, socket: Socket): void {
  const chunks: Buffer[] = [];
  let bytesReceived = 0;
  let dispatched = false;

  socket.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code !== 'ECONNRESET' && err.code !== 'EPIPE') {
      console.error(
        `[mlflow][wal] ipc connection error: code=${err.code ?? 'none'} message=${err.message}`,
      );
    }
    socket.destroy();
  });

  socket.on('data', (chunk: Buffer) => {
    if (dispatched) {
      // Extra bytes after the first line are not part of the protocol
      // today; ignore them rather than reject
      return;
    }
    bytesReceived += chunk.length;
    if (bytesReceived > MAX_REQUEST_BYTES) {
      dispatched = true;
      socket.pause();
      sendResponse(socket, {
        ok: false,
        error: `request exceeds max size ${MAX_REQUEST_BYTES} bytes without newline terminator`,
      });
      const destroyTimer = setTimeout(() => socket.destroy(), 500);
      destroyTimer.unref();
      socket.once('close', () => clearTimeout(destroyTimer));
      return;
    }
    // Prior chunks contained no newline (otherwise `dispatched` would
    // be true and we'd have returned above), so only this chunk needs
    // scanning. `0x0A` is the byte value of '\n'.
    const newlineInChunk = chunk.indexOf(0x0a);
    if (newlineInChunk < 0) {
      chunks.push(chunk);
      return;
    }
    chunks.push(chunk.subarray(0, newlineInChunk));
    const line = Buffer.concat(chunks).toString('utf8');
    dispatched = true;
    void dispatch(writer, socket, line);
  });
}

async function dispatch(writer: BatchingWriter, socket: Socket, line: string): Promise<void> {
  let request: IpcRequest;
  try {
    request = JSONBig.parse(line) as IpcRequest;
  } catch (err) {
    sendResponse(socket, {
      ok: false,
      error: `malformed request: ${(err as Error).message}`,
    });
    return;
  }

  if (request.op !== 'append') {
    sendResponse(socket, {
      ok: false,
      error: `unknown op: ${(request as { op: string }).op}`,
    });
    return;
  }

  const record = request.record as Partial<WalRecord> | null | undefined;
  if (
    typeof record !== 'object' ||
    record == null ||
    typeof record.id !== 'string' ||
    record.id === ''
  ) {
    sendResponse(socket, {
      ok: false,
      error: 'invalid record: must be an object with a non-empty string id',
    });
    return;
  }

  try {
    await writer.submit(request.record);
    sendResponse(socket, { ok: true });
  } catch (err) {
    sendResponse(socket, { ok: false, error: (err as Error).message });
  }
}

function sendResponse(socket: Socket, response: IpcResponse): void {
  // `socket.end` does not throw synchronously for our inputs (always a
  // string, never null, never on a not-yet-constructed handle). Any
  // transport-level failure during the write surfaces asynchronously
  // via the `'error'` event, which is handled at the top of
  // handleConnection - no defensive try/catch needed here.
  socket.end(JSONBig.stringify(response) + '\n');
}

/**
 * Build the per-connection handler that the daemon installs on its
 * lock-socket server. Bound to a {@link BatchingWriter} so the
 * fsync-coalescing state persists across connections.
 */
export function createIpcConnectionHandler(writer: BatchingWriter): (socket: Socket) => void {
  return (socket: Socket) => handleConnection(writer, socket);
}
