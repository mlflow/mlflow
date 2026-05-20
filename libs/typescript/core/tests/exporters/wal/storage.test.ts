import { mkdtemp, readFile, rm, stat, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  appendDeadLetter,
  appendRecord,
  appendTombstone,
  compact,
  readPending,
  walSize,
} from '../../../src/exporters/wal/storage';
import { getDeadLetterPath, getWalPath } from '../../../src/exporters/wal/paths';
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

describe('wal/storage', () => {
  let walDir: string;

  beforeEach(async () => {
    walDir = await mkdtemp(join(tmpdir(), 'mlflow-wal-'));
    process.env.MLFLOW_WAL_DIR = walDir;
  });

  afterEach(async () => {
    delete process.env.MLFLOW_WAL_DIR;
    await rm(walDir, { recursive: true, force: true });
  });

  it('returns an empty array when no WAL file exists yet', async () => {
    expect(await readPending()).toEqual([]);
    expect(await walSize()).toBe(0);
  });

  it('appends a record and reads it back', async () => {
    const record = makeRecord('a');
    await appendRecord(record);

    const pending = await readPending();
    expect(pending).toHaveLength(1);
    expect(pending[0]).toEqual(record);
    expect(await walSize()).toBeGreaterThan(0);
  });

  it('round-trips bigint fields (regression for SerializedSpan timestamps)', async () => {
    // `SerializedSpan.start_time_unix_nano` is a `bigint`; native
    // `JSON.stringify` throws on those. The storage layer uses JSONBig
    // so spans coming out of `Span.toJson()` survive a WAL round-trip
    // unchanged. The value here is intentionally above
    // `Number.MAX_SAFE_INTEGER` to catch a regression to plain
    // `JSON.parse(JSON.stringify(...))` (which would silently lose
    // precision rather than throw).
    const startNs = 1_750_000_000_123_456_789n;
    const record = makeRecord('bigint', {
      traceData: {
        spans: [
          {
            trace_id: 't-bigint',
            start_time_unix_nano: startNs,
            end_time_unix_nano: startNs + 5_000_000n,
          },
        ],
      },
    });

    await appendRecord(record);
    const pending = await readPending();
    expect(pending).toHaveLength(1);
    const restored = (pending[0].traceData as { spans: Array<Record<string, unknown>> }).spans[0];
    expect(restored.start_time_unix_nano).toBe(startNs);
    expect(restored.end_time_unix_nano).toBe(startNs + 5_000_000n);
  });

  it('hides a record once it has been tombstoned', async () => {
    const r1 = makeRecord('a');
    const r2 = makeRecord('b');
    await appendRecord(r1);
    await appendRecord(r2);
    await appendTombstone(r1.id);

    const pending = await readPending();
    expect(pending.map((r) => r.id)).toEqual(['wal-b']);
  });

  it('treats a tombstone before its corresponding append as a no-op', async () => {
    // Records use fresh ids per attempt, so a tombstone never legitimately
    // precedes its append; if it does (e.g. corrupt log), the later append
    // must still win.
    const r = makeRecord('a');
    await appendTombstone(r.id);
    await appendRecord(r);

    const pending = await readPending();
    expect(pending).toHaveLength(1);
    expect(pending[0]?.id).toBe(r.id);
  });

  it('accumulates entries in the dead-letter file without touching queue.log', async () => {
    const live = makeRecord('live');
    await appendRecord(live);

    const dead1 = makeRecord('dead-1', { attempts: 10 });
    const dead2 = makeRecord('dead-2', { attempts: 10 });
    await appendDeadLetter(dead1);
    await appendDeadLetter(dead2);

    const dlqPath = getDeadLetterPath();
    // failed.log is daily-rotated; the resolved path must include a
    // YYYY-MM-DD suffix so a regression to a single unbounded file is
    // caught in CI.
    expect(dlqPath).toMatch(/[/\\]failed\.log\.\d{4}-\d{2}-\d{2}$/);
    const dlqContents = await readFile(dlqPath, 'utf8');
    const dlqLines = dlqContents.split('\n').filter((l) => l.length > 0);
    expect(dlqLines).toHaveLength(2);
    expect(JSON.parse(dlqLines[0])).toEqual({ type: 'append', record: dead1 });
    expect(JSON.parse(dlqLines[1])).toEqual({ type: 'append', record: dead2 });

    const pending = await readPending();
    expect(pending).toEqual([live]);
  });

  it('preserves live records and shrinks the file when compacting', async () => {
    const r1 = makeRecord('a');
    const r2 = makeRecord('b');
    const r3 = makeRecord('c');
    await appendRecord(r1);
    await appendRecord(r2);
    await appendRecord(r3);
    await appendTombstone(r2.id);

    const sizeBefore = await walSize();
    await compact();
    const sizeAfter = await walSize();

    expect(sizeAfter).toBeLessThan(sizeBefore);

    const pending = await readPending();
    expect(pending.map((r) => r.id).sort()).toEqual(['wal-a', 'wal-c']);

    const lines = (await readFile(getWalPath(), 'utf8')).split('\n').filter((l) => l.length > 0);
    expect(lines).toHaveLength(2);
    for (const line of lines) {
      expect(JSON.parse(line).type).toBe('append');
    }
  });

  it('is a no-op when compacting a non-existent WAL', async () => {
    await expect(compact()).resolves.toBeUndefined();
    await expect(stat(getWalPath())).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('skips malformed lines but keeps surrounding records', async () => {
    const r1 = makeRecord('a');
    const r2 = makeRecord('b');
    await appendRecord(r1);
    await writeFile(getWalPath(), '{not valid json}\n', { flag: 'a' });
    await appendRecord(r2);

    const debugSpy = jest.spyOn(console, 'debug').mockImplementation(() => {});
    try {
      const pending = await readPending();
      expect(pending.map((r) => r.id).sort()).toEqual(['wal-a', 'wal-b']);
      expect(debugSpy).toHaveBeenCalled();
    } finally {
      debugSpy.mockRestore();
    }
  });

  it('survives 50 concurrent appends with no torn or lost lines', async () => {
    const records = Array.from({ length: 50 }, (_, i) => makeRecord(i.toString().padStart(2, '0')));
    await Promise.all(records.map((r) => appendRecord(r)));

    const contents = await readFile(getWalPath(), 'utf8');
    const lines = contents.split('\n').filter((l) => l.length > 0);
    expect(lines).toHaveLength(50);
    for (const line of lines) {
      expect(() => {
        JSON.parse(line);
      }).not.toThrow();
    }

    const pending = await readPending();
    expect(pending).toHaveLength(50);
    const ids = new Set(pending.map((r) => r.id));
    for (const r of records) {
      expect(ids.has(r.id)).toBe(true);
    }
  });

  it('records nonzero walSize after an append', async () => {
    expect(await walSize()).toBe(0);
    await appendRecord(makeRecord('a'));
    const size = await walSize();
    expect(size).toBeGreaterThan(0);

    const fsSize = (await stat(getWalPath())).size;
    expect(size).toBe(fsSize);
  });
});
