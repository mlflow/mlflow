/**
 * TelemetryLogger SharedWorker
 *
 * A shared worker that coordinates telemetry logging across multiple browser tabs/windows.
 * This ensures telemetry events are logged once even when the user has multiple tabs open.
 */

import {
  WorkerToClientMessageType,
  type TelemetryRecord,
  type TelemetryConfig,
  ClientToWorkerMessageType,
} from './types';
import { TELEMETRY_ENDPOINT } from './constants';
import { LogQueue } from './LogQueue';

const scope = self as any as SharedWorkerGlobalScope;

/**
 * Fetch telemetry configuration from the server
 */
async function fetchConfig(): Promise<TelemetryConfig | null> {
  try {
    const response = await fetch(TELEMETRY_ENDPOINT, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch config: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('[TelemetryWorker] Failed to fetch config:', error);
    return null;
  }
}

class TelemetryLogger {
  private config: Promise<TelemetryConfig | null> = fetchConfig();
  private sessionId = crypto.randomUUID();
  private logQueue: LogQueue = new LogQueue();

  public async addLogToQueue(record: TelemetryRecord): Promise<void> {
    const config = await this.config;
    if (!config || config.disable_telemetry) {
      return;
    }

    this.logQueue.enqueue({ ...record, session_id: this.sessionId });
  }
}

const logger = new TelemetryLogger();

/**
 * Handle messages from connected ports
 */
function handleMessage(event: MessageEvent, port: MessagePort): void {
  const message = event.data;
  console.log('@@@ handleMessage', message);

  if (message.type !== ClientToWorkerMessageType.LOG_EVENT) {
    return;
  }

  logger.addLogToQueue(message.payload as TelemetryRecord).catch((error) => {
    console.error('[TelemetryWorker] Error logging event:', error);
  });
}

/**
 * SharedWorker entry point
 */
const ports: Set<MessagePort> = new Set();

scope.onconnect = (event: MessageEvent) => {
  const port = event.ports[0];
  ports.add(port);
  port.onmessage = (e) => {
    handleMessage(e, port);
  };
  port.postMessage({ type: WorkerToClientMessageType.READY });
};
