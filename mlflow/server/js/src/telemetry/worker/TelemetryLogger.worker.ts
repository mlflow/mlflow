import {
  WorkerToClientMessageType,
  type TelemetryRecord,
  type TelemetryConfig,
  ClientToWorkerMessageType,
} from './types';
import { UI_TELEMETRY_ENDPOINT } from './constants';
import { LogQueue } from './LogQueue';

// eslint-disable-next-line no-restricted-globals
const scope = self as any as SharedWorkerGlobalScope;

async function fetchConfig(): Promise<TelemetryConfig | null> {
  try {
    const response = await fetch(UI_TELEMETRY_ENDPOINT, {
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
  private samplingValue: number = Math.random() * 100;

  public async addLogToQueue(record: Omit<TelemetryRecord, 'session_id'>): Promise<void> {
    const config = await this.config;

    if (!config || (config.disable_ui_telemetry ?? true)) {
      return;
    }

    // check if the sampling value is less than the rollout percentage
    const isEnabled = this.samplingValue < (config.ui_rollout_percentage ?? 0);
    if (!isEnabled) {
      return;
    }

    const isIgnored = config.disable_ui_events?.includes(record.params?.['componentId'] ?? '');
    if (isIgnored) {
      return;
    }

    this.logQueue.enqueue({ ...record, session_id: this.sessionId });
  }

  public destroy(): void {
    this.logQueue.destroy();
  }
}

const logger = new TelemetryLogger();

function handleMessage(event: MessageEvent): void {
  const message = event.data;

  switch (message.type) {
    case ClientToWorkerMessageType.LOG_EVENT:
      logger.addLogToQueue(message.payload as TelemetryRecord).catch((error) => {
        console.error('[TelemetryWorker] Error logging event:', error);
      });
      break;
    case ClientToWorkerMessageType.SHUTDOWN:
      logger.destroy();
      scope.close();
      break;
  }
}

scope.onconnect = (event: MessageEvent) => {
  const port = event.ports[0];
  port.onmessage = handleMessage;
  // client only starts sending logs after receiving the READY message
  port.postMessage({ type: WorkerToClientMessageType.READY });
};
