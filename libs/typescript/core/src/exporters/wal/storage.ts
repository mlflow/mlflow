/**
 * Append-only JSONL store backing the trace upload WAL.
 */

import { createReadStream, existsSync } from 'node:fs';
import { mkdir, open, rename, stat, unlink } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createInterface } from 'node:readline';
import { JSONBig } from '../../core/utils/json';
import { getDeadLetterPath, getWalPath } from './paths';
import { WalLine, WalRecord } from './types';

/**
 * Promise-chained queue that serializes async tasks one at a time.
 */
class SerialQueue {
  private tail: Promise<void> = Promise.resolve();

  run<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.tail.catch(() => {}).then(fn);
    this.tail = result.then(
      () => {},
      () => {},
    );
    return result;
  }
}

export const queueWriter = new SerialQueue();

const deadLetterWriter = new SerialQueue();

export async function ensureParentDir(filePath: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
}

/**
 * Append one JSONL line to `path`, fsync, and close.
 *
 * Encoded as a single `Buffer` so the write goes out in one `O_APPEND`
 * syscall — line-sized records sit well under `PIPE_BUF`, so concurrent
 * appenders never interleave their bytes.
 */
async function appendJsonLine(path: string, line: WalLine): Promise<void> {
  await ensureParentDir(path);
  const buf = Buffer.from(JSONBig.stringify(line) + '\n', 'utf8');
  const fh = await open(path, 'a');
  try {
    await fh.write(buf);
    await fh.sync();
  } finally {
    await fh.close();
  }
}

/**
 * Append a pending trace upload to the WAL.
 */
export function appendRecord(record: WalRecord): Promise<void> {
  return queueWriter.run(() => appendJsonLine(getWalPath(), { type: 'append', record }));
}

/**
 * Append a tombstone for `id` to the WAL. Logically shadows any earlier
 * `{type:"append"}` line for the same id.
 */
export function appendTombstone(id: string): Promise<void> {
  return queueWriter.run(() => appendJsonLine(getWalPath(), { type: 'tombstone', id }));
}

/**
 * Append `record` to the dead-letter file (`failed.log.<YYYY-MM-DD>`).
 */
export function appendDeadLetter(record: WalRecord): Promise<void> {
  return deadLetterWriter.run(() =>
    appendJsonLine(getDeadLetterPath(), { type: 'append', record }),
  );
}

/**
 * Replay `queue.log` and return the set of records still considered pending.
 */
export async function readPending(): Promise<WalRecord[]> {
  const path = getWalPath();
  if (!existsSync(path)) {
    return [];
  }

  const alive = new Map<string, WalRecord>();
  const stream = createReadStream(path, { encoding: 'utf8' });
  const rl = createInterface({ input: stream, crlfDelay: Infinity });

  for await (const line of rl) {
    if (line === '') {
      continue;
    }
    let parsed: unknown;
    try {
      // JSONBig.parse so bigint fields written by `appendJsonLine` are
      // restored as `bigint` (and not as the lossy `number` JSON.parse
      // would coerce them to once stringified back through `JSONBig`).
      parsed = JSONBig.parse(line);
    } catch {
      console.debug(`[mlflow][wal] Skipping malformed WAL line: ${line.slice(0, 120)}`);
      continue;
    }
    if (!parsed || typeof parsed !== 'object') {
      continue;
    }
    const obj = parsed as { type?: unknown; record?: unknown; id?: unknown };
    if (obj.type === 'append') {
      const record = obj.record as WalRecord | undefined;
      if (record && typeof record.id === 'string') {
        alive.set(record.id, record);
      }
    } else if (obj.type === 'tombstone' && typeof obj.id === 'string') {
      alive.delete(obj.id);
    }
  }

  return [...alive.values()];
}

/**
 * Rewrite `queue.log` to contain only currently-live records.
 */
export function compact(): Promise<void> {
  return queueWriter.run(async () => {
    const path = getWalPath();
    if (!existsSync(path)) {
      return;
    }

    const liveRecords = await readPending();

    const tmpPath = `${path}.tmp.${process.pid}`;
    const tmpFh = await open(tmpPath, 'w');
    try {
      for (const record of liveRecords) {
        const buf = Buffer.from(
          JSONBig.stringify({ type: 'append', record } as WalLine) + '\n',
          'utf8',
        );
        await tmpFh.write(buf);
      }
      await tmpFh.sync();
      await tmpFh.close();
      await rename(tmpPath, path);
    } catch (err) {
      await tmpFh.close().catch(() => {});
      await unlink(tmpPath).catch(() => {});
      throw err;
    }
  });
}

/**
 * Size of `queue.log` in bytes, or 0 if the file does not exist.
 * Used by the daemon's idle-shutdown heuristic.
 */
export async function walSize(): Promise<number> {
  const path = getWalPath();
  if (!existsSync(path)) {
    return 0;
  }
  return (await stat(path)).size;
}
