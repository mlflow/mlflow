import { mkdtemp, readFile, rm } from 'fs/promises';
import { createServer, Server } from 'net';
import { tmpdir } from 'os';
import { join } from 'path';
import { BatchingWriter } from '../../../src/exporters/wal/batching_writer';
import { createIpcConnectionHandler, submitRecord } from '../../../src/exporters/wal/ipc';
import { getLockSocketPath, getWalPath } from '../../../src/exporters/wal/paths';
import type { IpcRequest, IpcResponse } from '../../../src/exporters/wal/protocol';
import type { WalRecord } from '../../../src/exporters/wal/types';

function makeRecord(idSuffix: string, overrides: Partial<WalRecord> = {}): WalRecord {
  return {
    id: `wal-${idSuffix}`,
    trackingUri: 'http://localhost:5000',
    experimentId: '0',
    traceInfo: { trace_id: `t-${idSuffix}` },
    traceData: { spans: [] },
    attempts: 0,
    nextAttemptAt: 0,
    createdAt: Date.now(),
    ...overrides,
  };
}

async function closeServer(server: Server): Promise<void> {
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

describe('wal/ipc submitRecord', () => {
  let walDir: string;

  const originalWalDir = process.env.MLFLOW_WAL_DIR;
  const originalDaemonExe = process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE;

  beforeEach(async () => {
    walDir = await mkdtemp(join(tmpdir(), 'mlflow-ipc-'));
    process.env.MLFLOW_WAL_DIR = walDir;
    process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE = join(walDir, 'noop-daemon.cjs');
  });

  afterEach(async () => {
    if (originalWalDir === undefined) {
      delete process.env.MLFLOW_WAL_DIR;
    } else {
      process.env.MLFLOW_WAL_DIR = originalWalDir;
    }
    if (originalDaemonExe === undefined) {
      delete process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE;
    } else {
      process.env.MLFLOW_TRACE_DAEMON_EXECUTABLE = originalDaemonExe;
    }
    await rm(walDir, { recursive: true, force: true });
  });

  it('round-trips an append request: request → ok ACK', async () => {
    const requestChunks: Buffer[] = [];
    const server = createServer((socket) => {
      let responded = false;
      socket.on('data', (chunk: Buffer) => {
        requestChunks.push(chunk);
        if (responded) {
          return;
        }
        const buffered = Buffer.concat(requestChunks).toString('utf8');
        if (!buffered.includes('\n')) {
          return;
        }
        responded = true;
        const response: IpcResponse = { ok: true };
        socket.end(JSON.stringify(response) + '\n');
      });
    });
    await new Promise<void>((resolve) => server.listen(getLockSocketPath(), resolve));

    try {
      const record = makeRecord('a');
      await expect(submitRecord(record)).resolves.toBeUndefined();
      const request = Buffer.concat(requestChunks).toString('utf8').trim();
      const parsed = JSON.parse(request) as IpcRequest;
      expect(parsed.op).toBe('append');
      expect(parsed.record.id).toBe(record.id);
    } finally {
      await closeServer(server);
    }
  });

  it('throws when the daemon responds with an error', async () => {
    const server = createServer((socket) => {
      socket.on('data', () => {
        const response: IpcResponse = { ok: false, error: 'disk full' };
        socket.end(JSON.stringify(response) + '\n');
      });
    });
    await new Promise<void>((resolve) => server.listen(getLockSocketPath(), resolve));

    try {
      await expect(submitRecord(makeRecord('b'))).rejects.toThrow(/disk full/);
    } finally {
      await closeServer(server);
    }
  });

  it('throws after exhausting retries when no daemon is listening', async () => {
    // No server bound; submitRecord should retry, fail to reach
    // anyone, and surface a connect error to the caller. The full
    // CONNECT_RETRY_DELAYS_MS sleeps (~3.15s) still elapse - we
    // don't currently inject those in tests - but pointing
    // MLFLOW_WAL_DIR at a tmpdir guarantees each connect attempt
    // fails fast with ENOENT instead of stacking on top of the
    // (much longer) per-attempt kernel connect timeout that would
    // fire against an unreachable bound socket. The 10s test
    // timeout accommodates the configured backoff with margin; if
    // retry latency ever becomes a CI bottleneck the fix is to make
    // CONNECT_RETRY_DELAYS_MS injectable rather than to chase it
    // here.
    const r = makeRecord('c');
    await expect(submitRecord(r)).rejects.toMatchObject({
      // The contract we care about is that retry exhaustion surfaces
      // a transport-level error code to the caller. The exact value
      // varies by platform (POSIX UNIX-domain sockets vs. Windows
      // Named Pipes) and by Node/libuv version, so pinning to
      // ECONNREFUSED|ENOENT produced unnecessary Windows-CI
      // fragility; matching any non-empty errno-style string is what
      // this assertion actually wants to express.
      code: expect.stringMatching(/^\S+$/),
    });
  }, 10_000);
});

describe('wal/ipc createIpcConnectionHandler', () => {
  let walDir: string;
  let server: Server | null = null;
  const originalWalDir = process.env.MLFLOW_WAL_DIR;

  beforeEach(async () => {
    walDir = await mkdtemp(join(tmpdir(), 'mlflow-ipc-handler-'));
    process.env.MLFLOW_WAL_DIR = walDir;
  });

  afterEach(async () => {
    if (server) {
      await closeServer(server);
      server = null;
    }
    if (originalWalDir === undefined) {
      delete process.env.MLFLOW_WAL_DIR;
    } else {
      process.env.MLFLOW_WAL_DIR = originalWalDir;
    }
    await rm(walDir, { recursive: true, force: true });
  });

  async function startHandlerServer(writer: BatchingWriter): Promise<Server> {
    const s = createServer(createIpcConnectionHandler(writer));
    await new Promise<void>((resolve) => s.listen(getLockSocketPath(), resolve));
    return s;
  }

  it('persists the submitted record and acks with ok=true', async () => {
    const writer = new BatchingWriter();
    server = await startHandlerServer(writer);

    const record = makeRecord('h1');
    await submitRecord(record);

    const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
    expect(lines).toHaveLength(1);
    const parsed = JSON.parse(lines[0]) as { type: string; record: WalRecord };
    expect(parsed.type).toBe('append');
    expect(parsed.record.id).toBe(record.id);
  });

  it('responds with ok=false for an unknown op', async () => {
    const writer = new BatchingWriter();
    server = await startHandlerServer(writer);

    const { createConnection } = await import('node:net');
    const result = await new Promise<string>((resolve, reject) => {
      const socket = createConnection(getLockSocketPath());
      const chunks: Buffer[] = [];
      socket.on('connect', () => {
        socket.write(JSON.stringify({ op: 'frobnicate', record: makeRecord('x') }) + '\n');
      });
      socket.on('data', (chunk) => chunks.push(chunk));
      socket.on('end', () => resolve(Buffer.concat(chunks).toString('utf8').trim()));
      socket.on('error', reject);
    });
    const parsed = JSON.parse(result) as IpcResponse;
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) {
      expect(parsed.error).toMatch(/unknown op/);
    }
  });

  it('rejects requests whose record is missing or has no string id', async () => {
    // Exercises the structural check in dispatch() that stops bad
    // payloads from being durably persisted to queue.log and then
    // poisoning every subsequent batch loop iteration that tries to
    // upload them. Walks each rejection branch (non-object, null,
    // missing id, non-string id, empty string id) through one IPC
    // round-trip per case so the WAL is never written.
    const writer = new BatchingWriter();
    server = await startHandlerServer(writer);

    const { createConnection } = await import('node:net');
    const send = (record: unknown): Promise<IpcResponse> =>
      new Promise<IpcResponse>((resolve, reject) => {
        const socket = createConnection(getLockSocketPath());
        const chunks: Buffer[] = [];
        socket.on('connect', () => {
          socket.write(JSON.stringify({ op: 'append', record }) + '\n');
        });
        socket.on('data', (chunk) => chunks.push(chunk));
        socket.on('end', () => {
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString('utf8').trim()) as IpcResponse);
          } catch (err) {
            reject(err as Error);
          }
        });
        socket.on('error', reject);
      });

    const badCases: unknown[] = [null, undefined, 42, 'not an object', {}, { id: 123 }, { id: '' }];
    for (const bad of badCases) {
      const response = await send(bad);
      expect(response.ok).toBe(false);
      if (!response.ok) {
        expect(response.error).toMatch(/invalid record/);
      }
    }

    // queue.log must never have been written: no successful submits
    // and no partial lines from rejected ones.
    await expect(readFile(getWalPath(), 'utf8')).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('rejects oversized requests without exhausting memory', async () => {
    // Stream bytes well past the 16 MiB cap without ever sending a
    // newline. The handler should respond with ok=false, pause its
    // read side so the kernel buffer stops growing, and half-close
    // the write side so we observe a clean `'end'` here.
    const writer = new BatchingWriter();
    server = await startHandlerServer(writer);

    const { createConnection } = await import('node:net');
    const result = await new Promise<string>((resolve, reject) => {
      const socket = createConnection(getLockSocketPath());
      const chunks: Buffer[] = [];
      socket.on('connect', () => {
        // 1 MiB of 'A's per chunk × 17 chunks = 17 MiB, no trailing
        // newline. We send chunks back-to-back; once the daemon's
        // cumulative buffer crosses 16 MiB it must short-circuit and
        // respond, after which it ignores any further bytes.
        const oneMib = Buffer.alloc(1024 * 1024, 0x41);
        const pump = (remaining: number): void => {
          if (remaining === 0 || socket.destroyed) {
            return;
          }
          if (!socket.write(oneMib)) {
            socket.once('drain', () => pump(remaining - 1));
            return;
          }
          setImmediate(() => pump(remaining - 1));
        };
        pump(17);
      });
      socket.on('data', (chunk) => chunks.push(chunk));
      socket.on('end', () => resolve(Buffer.concat(chunks).toString('utf8').trim()));
      socket.on('error', (err: NodeJS.ErrnoException) => {
        // EPIPE can surface from the pump's writes once the daemon
        // pauses reading and our kernel send buffer eventually fills.
        // That's expected; the contract we care about is that the
        // daemon replied before that backpressure hit.
        if (err.code === 'EPIPE') {
          return;
        }
        reject(err);
      });
    });
    const parsed = JSON.parse(result) as IpcResponse;
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) {
      expect(parsed.error).toMatch(/exceeds max size/);
    }
  }, 15_000);

  it('measures the cap in wire bytes, not UTF-16 code units', async () => {
    const writer = new BatchingWriter();
    server = await startHandlerServer(writer);

    const { createConnection } = await import('node:net');
    const result = await new Promise<string>((resolve, reject) => {
      const socket = createConnection(getLockSocketPath());
      const chunks: Buffer[] = [];
      socket.on('connect', () => {
        const cjkChunk = Buffer.from('中'.repeat(349_525), 'utf8');
        const pump = (remaining: number): void => {
          if (remaining === 0 || socket.destroyed) {
            return;
          }
          if (!socket.write(cjkChunk)) {
            socket.once('drain', () => pump(remaining - 1));
            return;
          }
          setImmediate(() => pump(remaining - 1));
        };
        pump(17);
      });
      socket.on('data', (chunk) => chunks.push(chunk));
      socket.on('end', () => resolve(Buffer.concat(chunks).toString('utf8').trim()));
      socket.on('error', (err: NodeJS.ErrnoException) => {
        if (err.code === 'EPIPE') {
          return;
        }
        reject(err);
      });
    });
    const parsed = JSON.parse(result) as IpcResponse;
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) {
      expect(parsed.error).toMatch(/exceeds max size/);
    }
  }, 15_000);

  it('handles a multi-byte UTF-8 codepoint split across chunk boundaries', async () => {
    const writer = new BatchingWriter();
    server = await startHandlerServer(writer);

    const record = makeRecord('with-中');
    const payload = Buffer.from(JSON.stringify({ op: 'append', record }) + '\n', 'utf8');
    // First high-bit byte in the payload is the `0xE4` of `中` (the
    // rest of the JSON is ASCII), so split immediately after it to
    // straddle the codepoint.
    const cjkStart = payload.indexOf(0xe4);
    expect(cjkStart).toBeGreaterThan(0);
    const chunkA = payload.subarray(0, cjkStart + 1);
    const chunkB = payload.subarray(cjkStart + 1);

    const { createConnection } = await import('node:net');
    const response = await new Promise<IpcResponse>((resolve, reject) => {
      const socket = createConnection(getLockSocketPath());
      const respChunks: Buffer[] = [];
      socket.on('connect', () => {
        socket.write(chunkA, () => {
          // setTimeout (rather than setImmediate) gives the kernel a
          // tick to deliver chunk A as its own `'data'` event before
          // chunk B arrives, otherwise the writes may coalesce and
          // the test won't actually exercise the split-codepoint
          // path it's meant to guard.
          setTimeout(() => socket.write(chunkB), 10);
        });
      });
      socket.on('data', (chunk) => respChunks.push(chunk));
      socket.on('end', () => {
        try {
          resolve(JSON.parse(Buffer.concat(respChunks).toString('utf8').trim()) as IpcResponse);
        } catch (err) {
          reject(err as Error);
        }
      });
      socket.on('error', reject);
    });
    expect(response.ok).toBe(true);

    const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
    expect(lines).toHaveLength(1);
    const persisted = JSON.parse(lines[0]) as { record: WalRecord };
    expect(persisted.record.id).toBe('wal-with-中');
  });

  it('tolerates probe connections that close without sending data', async () => {
    const writer = new BatchingWriter();
    server = await startHandlerServer(writer);

    const { createConnection } = await import('node:net');
    await new Promise<void>((resolve, reject) => {
      const socket = createConnection(getLockSocketPath());
      socket.on('connect', () => {
        socket.destroy();
        resolve();
      });
      socket.on('error', reject);
    });

    // A subsequent real submit must still succeed: the probe-and-tear
    // pattern must not have left the handler in a bad state.
    await submitRecord(makeRecord('after-probe'));
    const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
    expect(lines).toHaveLength(1);
  });
});
