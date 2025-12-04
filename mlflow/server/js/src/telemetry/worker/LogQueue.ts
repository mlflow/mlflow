/**
 * LogQueue for batching and uploading telemetry events
 *
 * Maintains a queue of telemetry records and flushes them every 15 seconds
 * by batching and uploading to the telemetry endpoint.
 */

import { TELEMETRY_ENDPOINT } from './constants';
import { type TelemetryRecord } from './types';

const FLUSH_INTERVAL_MS = 15000; // 15 seconds

export class LogQueue {
  private queue: TelemetryRecord[] = [];
  private flushTimer: number | null = null;
  private isFlushing = false;

  constructor() {
    this.startFlushTimer();
  }

  /**
   * Add a telemetry record to the queue
   */
  public enqueue(record: TelemetryRecord): void {
    this.queue.push(record);
  }

  /**
   * Start the periodic flush timer
   */
  private startFlushTimer(): void {
    if (this.flushTimer !== null) {
      return;
    }

    this.flushTimer = self.setInterval(() => {
      this.flush();
    }, FLUSH_INTERVAL_MS) as unknown as number;
  }

  /**
   * Stop the periodic flush timer
   */
  private stopFlushTimer(): void {
    if (this.flushTimer !== null) {
      self.clearInterval(this.flushTimer);
      this.flushTimer = null;
    }

    this.flush();
  }

  /**
   * Flush the queue by batching and uploading logs to the server. Re-queues
   * failed records.
   */
  public async flush(): Promise<void> {
    if (this.isFlushing || this.queue.length === 0 || !navigator.onLine) {
      return;
    }

    this.isFlushing = true;
    const records = [...this.queue];
    this.queue = [];

    try {
      // Send batch to server
      const response = await fetch(TELEMETRY_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ records }),
      });

      if (!response.ok) {
        console.error(`[LogQueue] Failed to upload batch: ${response.status}`);
        this.queue.unshift(...records);
      }
    } catch (error) {
      console.error('[LogQueue] Error uploading batch:', error);
      this.queue.unshift(...records);
    } finally {
      this.isFlushing = false;
    }
  }

  /**
   * Clear the queue without uploading
   */
  public clear(): void {
    this.queue = [];
  }

  /**
   * Destroy the queue and stop the flush timer
   */
  public destroy(): void {
    this.stopFlushTimer();
    this.queue = [];
  }
}
