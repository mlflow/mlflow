/**
 * Group-commit writer for WAL lines.
 */

import { open, FileHandle } from 'node:fs/promises';
import { JSONBig } from '../../core/utils/json';
import { getWalPath } from './paths';
import { ensureParentDir, queueWriter } from './storage';
import { WalLine, WalRecord } from './types';

interface BatchItem {
  line: WalLine;
  resolve: () => void;
  reject: (err: Error) => void;
}

export class BatchingWriter {
  private pending: BatchItem[] = [];
  private flushScheduled = false;

  /**
   * Enqueue an append for `record`. Resolves once the record (and its
   * batch-mates) have been fsynced to `queue.log`.
   */
  submit(record: WalRecord): Promise<void> {
    return this.enqueue({ type: 'append', record });
  }

  /**
   * Enqueue a tombstone for `id`. Resolves once the tombstone (and its
   * batch-mates) have been fsynced to `queue.log`.
   */
  submitTombstone(id: string): Promise<void> {
    return this.enqueue({ type: 'tombstone', id });
  }

  private enqueue(line: WalLine): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.pending.push({ line, resolve, reject });
      this.scheduleFlush();
    });
  }

  private scheduleFlush(): void {
    if (this.flushScheduled) {
      return;
    }
    this.flushScheduled = true;
    // setImmediate fires at the end of the current event-loop tick,
    // so every `submit` call made within the same tick batches
    // together.
    setImmediate(() => {
      this.flushScheduled = false;
      if (this.pending.length === 0) {
        return;
      }
      void queueWriter.run(() => this.drainAndFlush());
    });
  }

  private async drainAndFlush(): Promise<void> {
    const batch = this.pending;
    this.pending = [];
    if (batch.length === 0) {
      return;
    }

    const path = getWalPath();
    try {
      await ensureParentDir(path);
    } catch (err) {
      const error = err as Error;
      for (const item of batch) {
        item.reject(error);
      }
      return;
    }

    let fh: FileHandle;
    try {
      fh = await open(path, 'a');
    } catch (err) {
      const error = err as Error;
      for (const item of batch) {
        item.reject(error);
      }
      return;
    }

    try {
      for (const item of batch) {
        const buf = Buffer.from(JSONBig.stringify(item.line) + '\n', 'utf8');
        await fh.write(buf);
      }
      await fh.sync();
    } catch (err) {
      const error = err as Error;
      for (const item of batch) {
        item.reject(error);
      }
      await fh.close().catch(() => {});
      return;
    }

    await fh.close().catch(() => {});
    for (const item of batch) {
      item.resolve();
    }
  }
}
