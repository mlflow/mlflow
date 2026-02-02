/**
 * LogQueue for batching and uploading telemetry events
 *
 * Maintains a queue of telemetry records and flushes them every 15 seconds
 * by batching and uploading to the telemetry endpoint.
 */

import { UI_TELEMETRY_ENDPOINT } from './constants';
import { type TelemetryRecord } from './types';

const FLUSH_INTERVAL_MS = 30000; // 30 seconds

export class LogQueue {
  private queue: TelemetryRecord[] = [];
  private flushTimer: number | null = null;

  constructor() {
    this.startFlushTimer();
  }

  public enqueue(record: TelemetryRecord): void {
    // if the queue has been destroyed, don't enqueue any more records
    if (this.flushTimer === null) {
      return;
    }
    this.queue.push(record);
  }

  private startFlushTimer(): void {
    if (this.flushTimer !== null) {
      return; // Loop already running
    }
    this.scheduleNextFlush();
  }

  private scheduleNextFlush(): void {
    // eslint-disable-next-line no-restricted-globals
    this.flushTimer = self.setTimeout(() => {
      this.flush();
      this.scheduleNextFlush(); // Continue the loop
    }, FLUSH_INTERVAL_MS) as unknown as number;
  }

  private stopFlushTimer(): void {
    if (this.flushTimer !== null) {
      // eslint-disable-next-line no-restricted-globals
      self.clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }

    this.flush();
  }

  /**
   * Flush the queue by batching and uploading logs to the server. Re-queues
   * failed records.
   */
  public async flush(): Promise<void> {
    if (this.queue.length === 0 || !navigator.onLine) {
      return;
    }

    const records = [...this.queue];
    this.queue = [];

    try {
      // Send batch to server
      const response = await fetch(UI_TELEMETRY_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ records }),
      });

      if (!response.ok) {
        console.error(`[LogQueue] Failed to upload batch: ${response.status}`);
        this.queue.unshift(...records);
        return;
      }

      const responseJson = await response.json();
      if (responseJson.status === 'disabled') {
        this.destroy();
      }
    } catch (error) {
      console.error('[LogQueue] Error uploading batch:', error);
      this.queue.unshift(...records);
    }
  }

  public clear(): void {
    this.queue = [];
  }

  public destroy(): void {
    this.stopFlushTimer();
    this.queue = [];
  }
}
