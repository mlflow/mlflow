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

const CONNECT_RETRY_DELAYS_MS = [50, 100, 250, 500, 1000, 2000];

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isConnectError(err: unknown): boolean {
  if (typeof err !== 'object' || err == null) {
    return false;
  }
  const code = (err as { code?: string }).code;
  return code === 'ECONNREFUSED' || code === 'ENOENT' || code === 'EPIPE' || code === 'ECONNRESET';
}

/**
 * Send `record` to the running daemon and wait for the post-fsync ACK.
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

  if (lastErr instanceof Error) {
    throw lastErr;
  }
  const wrapped = new Error(`Failed to submit record after retries: ${String(lastErr)}`);
  if (typeof lastErr === 'object' && lastErr != null && 'code' in lastErr) {
    (wrapped as { code?: string }).code = (lastErr as { code?: string }).code;
  }
  throw wrapped;
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
      const text = Buffer.concat(chunks).toString('utf8').replace(/\n$/, '');
      if (text === '') {
        finish(() =>
          reject(
            Object.assign(new Error('Daemon closed connection without responding'), {
              code: 'ECONNRESET',
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
  let buffer = '';
  let dispatched = false;

  socket.on('error', () => {
    socket.destroy();
  });

  socket.on('data', (chunk: Buffer) => {
    if (dispatched) {
      // Extra bytes after the first line are not part of the protocol
      // today; ignore them rather than reject
      return;
    }
    buffer += chunk.toString('utf8');
    const newlineIdx = buffer.indexOf('\n');
    if (newlineIdx < 0) {
      return;
    }
    const line = buffer.slice(0, newlineIdx);
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

  try {
    await writer.submit(request.record);
    sendResponse(socket, { ok: true });
  } catch (err) {
    sendResponse(socket, { ok: false, error: (err as Error).message });
  }
}

function sendResponse(socket: Socket, response: IpcResponse): void {
  const payload = JSONBig.stringify(response) + '\n';

  try {
    socket.end(payload);
  } catch {
    socket.destroy();
  }
}

/**
 * Build the per-connection handler that the daemon installs on its
 * lock-socket server. Bound to a {@link BatchingWriter} so the
 * fsync-coalescing state persists across connections.
 */
export function createIpcConnectionHandler(writer: BatchingWriter): (socket: Socket) => void {
  return (socket: Socket) => handleConnection(writer, socket);
}
