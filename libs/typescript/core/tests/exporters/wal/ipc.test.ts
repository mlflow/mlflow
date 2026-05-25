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
    const requests: string[] = [];
    const server = createServer((socket) => {
      socket.on('data', (chunk: Buffer) => {
        requests.push(chunk.toString('utf8'));
        const response: IpcResponse = { ok: true };
        socket.end(JSON.stringify(response) + '\n');
      });
    });
    await new Promise<void>((resolve) => server.listen(getLockSocketPath(), resolve));

    try {
      const record = makeRecord('a');
      await expect(submitRecord(record)).resolves.toBeUndefined();
      expect(requests).toHaveLength(1);
      const parsed = JSON.parse(requests[0].trim()) as IpcRequest;
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
    // anyone, and surface a connect error to the caller. We don't
    // want to wait the full ~3.85s backoff in CI, so we override the
    // socket path to something that ENOENTs immediately on every
    // retry (no kernel timeouts) and rely on the bounded retry count
    // to give up quickly.
    const r = makeRecord('c');
    await expect(submitRecord(r)).rejects.toMatchObject({
      code: expect.stringMatching(/ECONNREFUSED|ENOENT/),
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
