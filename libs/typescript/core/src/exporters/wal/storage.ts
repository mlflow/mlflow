/**
 * Append-only JSONL store backing the trace upload Write-Ahead Log.
 *
 * Two files live here, both under `getWalDir()`:
 *
 * - `queue.log` — pending uploads. Two line types: `{type:"append",record}`
 *   and `{type:"tombstone",id}`. Readers replay the file into a
 *   `Map<id, WalRecord>` and `tombstone` lines act as `Map.delete(id)`.
 *   Periodically compacted (read live records, rewrite, atomic rename) so
 *   the file does not grow without bound.
 * - `failed.log.<YYYY-MM-DD>` — dead-letter file, rotated daily by UTC date.
 *   Records whose retry-budget window (`RETRY_TIMEOUT_MS`) expired. Only ever appended
 *   to; never tombstoned, never compacted. Inspectable by operators; the
 *   daemon never re-reads it. Rotation is stateless: every
 *   {@link appendDeadLetter} call resolves the path from the current date,
 *   so a record dead-lettered at 23:59:59 UTC and another at 00:00:00 UTC
 *   land in two adjacent dated files automatically.
 *
 * Concurrency model:
 *
 * - In-process: every write to a given file goes through a {@link SerialQueue}
 *   so that open/write/fsync/close phases never interleave. Two separate
 *   queues (one per file) allow append-to-queue.log and append-to-failed.log
 *   to make progress in parallel.
 * - Cross-process: hooks open queue.log with `flag: 'a'` and write a single
 *   line buffer per record, which is atomic on common filesystems for the
 *   sizes we write. The daemon is the only compactor; `compact` snapshots
 *   the file size up-front, writes the compacted output to a tmp file, then
 *   re-reads any tail bytes that arrived during the rewrite and appends them
 *   before the final atomic `rename`. A tiny race window remains between the
 *   tail-read and the rename; any record appended in that ~ms window would
 *   be overwritten. We accept this for v1 because compaction is
 *   daemon-driven (rare) and the only racers are Stop hooks (low rate).
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
 *
 * Errors thrown by an enqueued task surface to that task's caller (the
 * `Promise` returned by `run`), but they do not break the chain — subsequent
 * tasks still run. This matters because a transient `EIO` on one append must
 * not poison every later append in the same process.
 *
 * The `tail` invariant: it is always a fulfilled `Promise<void>`. The
 * `result.then(() => {}, () => {})` rebind below swallows both outcomes,
 * so by induction every later `run` reads a fulfilled `this.tail`. The
 * `.catch(() => {})` before `.then(fn)` is therefore strictly defensive
 * — it costs one extra microtask per enqueue (negligible next to the
 * `fsync` we're already gated on) and keeps the next-task scheduling
 * correct even if a future refactor accidentally lets `this.tail`
 * reject (e.g. by inlining the chain or removing the rebind).
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

const queueWriter = new SerialQueue();
const deadLetterWriter = new SerialQueue();

async function ensureParentDir(filePath: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
}

/**
 * Append a single JSONL line + `\n` to `path`, fsync, and close.
 *
 * Encodes the line into a single `Buffer` so that the write hits the file in
 * one syscall. On the filesystems we target (ext4, APFS, NTFS) writes up to
 * `PIPE_BUF` (~4 KiB) are atomic; line-sized records are well below that,
 * and even multi-KB records are atomic in practice for `flag: 'a'` opens.
 * If a write is ever torn (e.g. host crash mid-write), {@link readPending}
 * tolerates the bad line by parsing-and-skipping.
 *
 * Uses {@link JSONBig} so `SerializedSpan.start_time_unix_nano` (and other
 * `bigint` fields on the trace data) round-trip cleanly. Native
 * `JSON.stringify` throws `TypeError: Do not know how to serialize a BigInt`
 * on those fields, matching the codec the rest of the codebase already uses
 * for outbound trace payloads (`clients/utils.ts`, `clients/artifacts/*`).
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
 *
 * Returns when the line is durable on disk (post-fsync). Callers may safely
 * exit immediately after this resolves.
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
 * Append `record` to the dead-letter file (`failed.log`).
 *
 * Dead-lettering does not tombstone the live queue on its own; callers
 * (the daemon) should pair this with {@link appendTombstone} so that the
 * next batch tick does not retry the same poison record.
 */
export function appendDeadLetter(record: WalRecord): Promise<void> {
  return deadLetterWriter.run(() =>
    appendJsonLine(getDeadLetterPath(), { type: 'append', record }),
  );
}

/**
 * Replay `queue.log` and return the set of records still considered pending.
 *
 * Unknown line types and malformed lines (failed `JSON.parse`) are silently
 * skipped with a `console.debug` so a single torn write cannot block the
 * whole batch. Returns an empty array if the WAL file does not exist yet.
 *
 * The optional `byteLimit` bounds the read to `[0, byteLimit)`. {@link compact}
 * uses this to snapshot the WAL at the same byte offset it captures with
 * `stat`, so cross-process appends that land mid-read do not end up in
 * `liveRecords` AND in the tail-byte copy (which would double-write them).
 * Every other caller (e.g. the daemon's batch loop) omits the option and
 * reads to EOF as before.
 */
export async function readPending(opts: { byteLimit?: number } = {}): Promise<WalRecord[]> {
  const path = getWalPath();
  if (!existsSync(path)) {
    return [];
  }
  // Empty snapshot: don't open the stream at all (createReadStream would
  // reject `end: -1`).
  if (opts.byteLimit !== undefined && opts.byteLimit <= 0) {
    return [];
  }

  const alive = new Map<string, WalRecord>();
  const streamOpts: Parameters<typeof createReadStream>[1] =
    opts.byteLimit !== undefined
      ? { encoding: 'utf8', start: 0, end: opts.byteLimit - 1 }
      : { encoding: 'utf8' };
  const stream = createReadStream(path, streamOpts);
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
 *
 * Procedure:
 *   1. Stat the WAL to snapshot `startSize`.
 *   2. Read pending records.
 *   3. Write each record as a fresh `{type:"append"}` line to
 *      `queue.log.tmp.<pid>`.
 *   4. Stat the WAL again; if it grew since step 1, copy the tail bytes
 *      `[startSize, currentSize)` (representing concurrent appends from
 *      other processes) into the tmp file. These bytes are guaranteed to
 *      consist of full lines because writers fsync after each line.
 *   5. fsync, close, and atomically rename the tmp file onto `queue.log`.
 *
 * Steps 3–5 all live inside the same try / catch so that a failure in
 * close or rename — not just in the write/sync block — still triggers
 * the tmp-file cleanup. Otherwise a flaky FS could orphan one
 * `queue.log.tmp.<pid>` per daemon lifetime.
 *
 * The whole thing runs inside the in-process queue writer so it cannot
 * interleave with in-process appends. Cross-process appends that land
 * between step 4 and step 5 will be lost — see the file header comment.
 */
export function compact(): Promise<void> {
  return queueWriter.run(async () => {
    const path = getWalPath();
    if (!existsSync(path)) {
      return;
    }

    const startSize = (await stat(path)).size;
    const liveRecords = await readPending({ byteLimit: startSize });

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

      const currentSize = (await stat(path)).size;
      if (currentSize > startSize) {
        const tailLength = currentSize - startSize;
        const tail = Buffer.alloc(tailLength);
        const srcFh = await open(path, 'r');
        try {
          await srcFh.read(tail, 0, tailLength, startSize);
        } finally {
          await srcFh.close();
        }
        await tmpFh.write(tail);
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
