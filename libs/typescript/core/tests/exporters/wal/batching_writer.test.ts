import * as fsPromises from 'node:fs/promises';
import { mkdtemp, readFile, rm } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { BatchingWriter } from '../../../src/exporters/wal/batching_writer';
import { getWalPath } from '../../../src/exporters/wal/paths';
import { appendTombstone, readPending } from '../../../src/exporters/wal/storage';
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

async function spyOnFileHandleSync(walDir: string): Promise<jest.SpyInstance> {
  let spy: jest.SpyInstance | undefined;
  try {
    const probeFh = await fsPromises.open(join(walDir, '.probe'), 'w');
    const fdProto = Object.getPrototypeOf(probeFh) as { sync?: () => Promise<void> };
    await probeFh.close();
    expect(typeof fdProto.sync).toBe('function');
    spy = jest.spyOn(fdProto as { sync: () => Promise<void> }, 'sync');
    return spy;
  } catch (err) {
    // `spy` is undefined for any throw before the install, so the
    // optional chain is a no-op; if the install succeeded and a later
    // line throws, this restores the prototype before propagating.
    spy?.mockRestore();
    throw err;
  }
}

describe('wal/batching_writer', () => {
  let walDir: string;
  const originalWalDir = process.env.MLFLOW_WAL_DIR;

  beforeEach(async () => {
    walDir = await mkdtemp(join(tmpdir(), 'mlflow-wal-batch-'));
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

  it('persists a single submitted record after the ack', async () => {
    const writer = new BatchingWriter();
    const record = makeRecord('s1');
    await writer.submit(record);

    const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0])).toEqual({ type: 'append', record });
  });

  it('coalesces concurrent submits into a single fsync (group commit)', async () => {
    const syncSpy = await spyOnFileHandleSync(walDir);
    try {
      const writer = new BatchingWriter();
      const records = Array.from({ length: 25 }, (_, i) =>
        makeRecord(i.toString().padStart(2, '0')),
      );

      const syncsBefore = syncSpy.mock.calls.length;
      // Fire all submits in the same tick before awaiting any of them.
      await Promise.all(records.map((r) => writer.submit(r)));
      const syncsAfter = syncSpy.mock.calls.length;

      // Allow at most one extra fsync above the one we expect, in case
      // a stray tick splits the batch in two; the regression we care
      // about is the "N fsyncs for N submits" shape.
      expect(syncsAfter - syncsBefore).toBeLessThanOrEqual(2);
      expect(syncsAfter - syncsBefore).toBeGreaterThanOrEqual(1);

      const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
      expect(lines).toHaveLength(records.length);
      const ids = lines.map((l) => (JSON.parse(l) as { record: WalRecord }).record.id).sort();
      expect(ids).toEqual(records.map((r) => r.id).sort());
    } finally {
      syncSpy.mockRestore();
    }
  });

  it('resolves submit() only after the byte is durable', async () => {
    // The post-submit file read must observe the record — i.e. the
    // ack-after-fsync invariant. If submit resolved before write+sync
    // completed this read would race and occasionally come up empty.
    const writer = new BatchingWriter();
    const record = makeRecord('durable');
    await writer.submit(record);

    const contents = await readFile(getWalPath(), 'utf8');
    expect(contents).toContain(`"id":"${record.id}"`);
  });

  it('produces well-formed lines when batched submits interleave with direct writers', async () => {
    const writer = new BatchingWriter();
    const r1 = makeRecord('mixed-1');
    const r2 = makeRecord('mixed-2');

    const submitR1 = writer.submit(r1);
    const tombstone = appendTombstone(r1.id);
    const submitR2 = writer.submit(r2);
    await Promise.all([submitR1, tombstone, submitR2]);

    const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
    // Two appends + one tombstone, none truncated, all valid JSON.
    expect(lines).toHaveLength(3);
    for (const line of lines) {
      expect(() => {
        JSON.parse(line);
      }).not.toThrow();
    }

    const pending = await readPending();
    expect(pending.map((r) => r.id)).toContain(r2.id);
  });

  it('persists a single submitted tombstone after the ack', async () => {
    const writer = new BatchingWriter();
    const record = makeRecord('t-single');
    await writer.submit(record);
    await writer.submitTombstone(record.id);

    const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[1])).toEqual({ type: 'tombstone', id: record.id });

    // Tombstone has shadowed the append: nothing should remain pending.
    const pending = await readPending();
    expect(pending.find((r) => r.id === record.id)).toBeUndefined();
  });

  it('resolves submitTombstone() only after the byte is durable', async () => {
    // Symmetric to the submit() ack-after-fsync test: the post-ack file
    // read must observe the tombstone, otherwise tombstones could be
    // ack'd before being durable and a daemon crash mid-batch would
    // resurrect already-uploaded records on the next pass.
    const writer = new BatchingWriter();
    const id = 'wal-t-durable';
    await writer.submitTombstone(id);

    const contents = await readFile(getWalPath(), 'utf8');
    expect(contents).toContain(`"id":"${id}"`);
    expect(contents).toContain('"type":"tombstone"');
  });

  it('coalesces appends and tombstones into a single fsync when submitted in the same tick', async () => {
    // The whole point of routing tombstones through the BatchingWriter:
    // a tombstone storm from the upload loop should batch with concurrent
    // hook submits instead of paying its own fsync per call.
    const syncSpy = await spyOnFileHandleSync(walDir);
    try {
      const writer = new BatchingWriter();
      const records = Array.from({ length: 10 }, (_, i) => makeRecord(`mix-a-${i}`));
      const tombstoneIds = Array.from({ length: 10 }, (_, i) => `mix-t-${i}`);

      const syncsBefore = syncSpy.mock.calls.length;
      await Promise.all([
        ...records.map((r) => writer.submit(r)),
        ...tombstoneIds.map((id) => writer.submitTombstone(id)),
      ]);
      const syncsAfter = syncSpy.mock.calls.length;

      // Same shape guarantee as the append-only group-commit test: a
      // stray tick may split the batch into two, but never N fsyncs for
      // N submits.
      expect(syncsAfter - syncsBefore).toBeLessThanOrEqual(2);
      expect(syncsAfter - syncsBefore).toBeGreaterThanOrEqual(1);

      const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
      expect(lines).toHaveLength(records.length + tombstoneIds.length);

      const tombstoneLines = lines.filter((l) => l.includes('"type":"tombstone"'));
      const appendLines = lines.filter((l) => l.includes('"type":"append"'));
      expect(tombstoneLines).toHaveLength(tombstoneIds.length);
      expect(appendLines).toHaveLength(records.length);
    } finally {
      syncSpy.mockRestore();
    }
  });
});
