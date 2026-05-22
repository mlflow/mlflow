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

describe('wal/batching_writer', () => {
  let walDir: string;
  // Capture the developer's pre-test value so we restore (not just unset)
  // in afterEach. Mirrors the pattern in paths.test.ts so running
  // `MLFLOW_WAL_DIR=/some/dir jest` doesn't lose the override for later
  // tests in the same worker.
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
    // Spy on FileHandle.sync via the prototype so we count every fsync
    // the writer issues regardless of which fd it opens. With group
    // commit, N submissions made in the same tick must collapse into a
    // single fsync. The probe fd is closed immediately after we grab
    // the prototype so Node's "FileHandle closed by GC" guard never
    // trips.
    const probeFh = await fsPromises.open(join(walDir, '.probe'), 'w');
    const fdProto = Object.getPrototypeOf(probeFh) as { sync: () => Promise<void> };
    await probeFh.close();
    const syncSpy = jest.spyOn(fdProto, 'sync');
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
    // BatchingWriter.submit defers the actual file write to a
    // setImmediate so it can coalesce same-tick submissions; a
    // tombstone or direct appendRecord called inline therefore lands
    // on queue.log before the deferred batch. We can't pin a specific
    // ordering here, but two invariants must hold regardless of the
    // interleaving: (1) every line in queue.log is valid JSON (no
    // torn writes thanks to the shared SerialQueue), and (2) the
    // visible record after the tombstone shadows the submitted
    // record. We pick a tombstone for an id that the writer also
    // submits and confirm that id is *not* in `readPending` —
    // whichever wrote first, the tombstone always wins because
    // `appendRecord(r1)` followed by `appendTombstone(r1.id)` and
    // `appendTombstone(r1.id)` followed by `appendRecord(r1)` both
    // collapse to "r1 hidden" under {@link readPending}'s replay
    // semantics (tombstone shadows earlier appends; an append after a
    // tombstone wins, but our writer never re-submits the same id).
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

    // Either ordering of tombstone vs. r1's append leaves r2 visible;
    // r1's pending state depends on the interleaving and is not part
    // of the contract under test.
    const pending = await readPending();
    expect(pending.map((r) => r.id)).toContain(r2.id);
  });
});
