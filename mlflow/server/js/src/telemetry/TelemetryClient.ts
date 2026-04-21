/**
 * TelemetryLogger client
 *
 * Wrapper for interacting with the TelemetryLogger SharedWorker.
 * Provides a simple API for logging telemetry events.
 */
import { v4 as uuidv4 } from 'uuid';
import type { TelemetryRecord } from './worker/types';
import {
  isDesignSystemEvent,
  isTelemetryDevLoggingEnabled,
  TELEMETRY_ENABLED_STORAGE_KEY,
  TELEMETRY_ENABLED_STORAGE_VERSION,
} from './utils';
import { WorkerToClientMessageType, ClientToWorkerMessageType } from './worker/types';
import { getLocalStorageItem } from '../shared/web-shared/hooks/useLocalStorage';

const LOCAL_STORAGE_INSTALLATION_ID_KEY = 'mlflow-telemetry-installation-id';

// Components whose onView events should be tracked (intentional impressions, not noise)
const VIEW_EVENT_ALLOWLIST: ReadonlySet<string> = new Set([
  'mlflow.gateway.setup_guide',
  'mlflow.issue-detection.completed',
]);

/**
 * Allowlist of metadata keys that may be attached to custom telemetry events.
 * Every value MUST be a static/enumerated string — never user-generated content.
 * To add a new key, update this type and add a corresponding validator below.
 */
type AllowedTelemetryMetadataKey = 'secretMode' | 'provider' | 'model' | 'usageTracking';

export type AllowedTelemetryMetadata = Partial<Record<AllowedTelemetryMetadataKey, string | null | undefined>>;

/**
 * Runtime validators for metadata values. Keys with validators are checked before logging;
 * values that fail validation are silently dropped. Keys without validators (e.g., 'model')
 * are passed through — callers must ensure they come from a trusted source (e.g., API catalog).
 */
const METADATA_VALIDATORS: Partial<Record<AllowedTelemetryMetadataKey, ReadonlySet<string>>> = {
  secretMode: new Set(['new', 'existing']),
  usageTracking: new Set(['true', 'false']),
  provider: new Set([
    'openai',
    'anthropic',
    'bedrock',
    'gemini',
    'vertex_ai',
    'azure',
    'groq',
    'databricks',
    'xai',
    'cohere',
    'mistral',
    'together_ai',
    'fireworks_ai',
    'replicate',
    'huggingface',
    'ai21',
    'perplexity',
    'deepinfra',
    'nvidia_nim',
    'cerebras',
    'deepseek',
    'openrouter',
    'ollama',
  ]),
};

function validateMetadata(metadata: AllowedTelemetryMetadata): Record<string, string | null | undefined> {
  const validated: Record<string, string | null | undefined> = {};
  for (const [key, value] of Object.entries(metadata)) {
    const allowedValues = METADATA_VALIDATORS[key as AllowedTelemetryMetadataKey];
    if (allowedValues) {
      if (value != null && allowedValues.has(value)) {
        validated[key] = value;
      }
      // silently drop values that fail validation
    } else {
      // no validator for this key — pass through (e.g., 'model')
      validated[key] = value;
    }
  }
  return validated;
}

class TelemetryClient {
  private installationId: string = this.getInstallationId();
  private port: MessagePort | null = null;
  private ready: Promise<boolean> = this.initWorker();

  private getInstallationId(): string {
    // not using `getLocalStorageItem` because this key is not used in react
    // eslint-disable-next-line @databricks/no-direct-storage
    const localStorageInstallationId = localStorage.getItem(LOCAL_STORAGE_INSTALLATION_ID_KEY);

    if (!localStorageInstallationId) {
      const installationId = uuidv4();
      // eslint-disable-next-line @databricks/no-direct-storage
      localStorage.setItem(LOCAL_STORAGE_INSTALLATION_ID_KEY, installationId);
      return installationId;
    } else {
      return localStorageInstallationId;
    }
  }

  private getTelemetryEnabled(): boolean {
    // need to use the function from web-shared because this key is
    // changed using `useLocalStorage` inside the settings page, which
    // appends the version to the key.
    const telemetryEnabled = getLocalStorageItem(
      TELEMETRY_ENABLED_STORAGE_KEY,
      TELEMETRY_ENABLED_STORAGE_VERSION,
      false,
      // default to true as the feature is opt-out
      true,
    );

    return telemetryEnabled;
  }

  private initWorker(): Promise<boolean> {
    return new Promise((resolve) => {
      try {
        // if telemetry is disabled, we don't need to initialize the worker at all
        if (!this.getTelemetryEnabled()) {
          resolve(false);
          return;
        }

        // Create SharedWorker instance
        this.port = new SharedWorker(new URL('./worker/TelemetryLogger.worker.ts', import.meta.url), {
          name: 'telemetry-worker',
        }).port;

        if (!this.port) {
          resolve(false);
          return;
        }

        const handleReadyMessage = (event: MessageEvent): void => {
          if (event.data.type === WorkerToClientMessageType.READY) {
            resolve(true);
          }
        };

        // Listen for the "READY" message from worker
        this.port.onmessage = handleReadyMessage;
      } catch (error) {
        // fail silently
        resolve(false);
      }
    });
  }

  // Log a telemetry event from the Design System event provider
  public async logEvent(record: any): Promise<void> {
    const isReady = await this.ready;
    if (!isReady || !this.port) {
      return;
    }

    if (!isDesignSystemEvent(record)) {
      return;
    }

    // drop view events to reduce noise, except for explicitly tracked impressions
    if (record.eventType === 'onView' && !VIEW_EVENT_ALLOWLIST.has(record.componentId)) {
      return;
    }

    // session_id is generated by the worker
    const payload: Omit<TelemetryRecord, 'session_id'> = {
      installation_id: this.installationId,
      event_name: 'ui_event',
      // convert from ms to ns
      timestamp_ns: Date.now() * 1e6,
      params: {
        componentId: record.componentId,
        componentViewId: record.componentViewId,
        componentType: record.componentType,
        componentSubType: record.componentSubType,
        eventType: record.eventType,
        // Include value for events, this only happens when valueHasNoPii=true
        ...(record.value !== undefined && { value: String(record.value) }),
      },
    };

    if (process.env['NODE_ENV'] === 'development' && isTelemetryDevLoggingEnabled()) {
      // eslint-disable-next-line no-console
      console.log(
        `[TelemetryClient] Event "${record.eventType}" on component "${record.componentId}", payload:`,
        payload,
      );
    }

    this.port?.postMessage({
      type: ClientToWorkerMessageType.LOG_EVENT,
      payload,
    });
  }

  /**
   * Log a telemetry event with custom metadata.
   *
   * By calling this method, you confirm that ALL metadata values are:
   *   - Static/enumerated values (e.g., "new" | "existing"), NOT user-generated strings
   *   - Free of PII, secrets, tokens, or any user-identifiable information
   *
   * Values for keys with validators (secretMode, usageTracking, provider) are checked
   * at runtime and silently dropped if invalid. Keys without validators (e.g., model)
   * are passed through — callers must ensure they come from a trusted source.
   *
   * Adding a new metadata key requires updating `AllowedTelemetryMetadataKey` in this file.
   * Do NOT pass user input, free-text fields, names, or IDs that could identify a user.
   */
  // eslint-disable-next-line @typescript-eslint/naming-convention
  public async logEventWithMetadata_I_CONFIRM_THERE_IS_NO_PII(
    componentId: string,
    eventType: string,
    metadata: AllowedTelemetryMetadata,
  ): Promise<void> {
    const isReady = await this.ready;
    if (!isReady || !this.port) {
      return;
    }

    const validatedMetadata = validateMetadata(metadata);

    const payload: Omit<TelemetryRecord, 'session_id'> = {
      installation_id: this.installationId,
      event_name: 'ui_event',
      timestamp_ns: Date.now() * 1e6,
      params: {
        ...validatedMetadata,
        // Spread metadata first so componentId/eventType cannot be overridden
        componentId,
        eventType,
      },
    };

    if (process.env['NODE_ENV'] === 'development' && isTelemetryDevLoggingEnabled()) {
      // eslint-disable-next-line no-console
      console.log(`[TelemetryClient] Custom event "${eventType}" on "${componentId}", metadata:`, payload);
    }

    this.port?.postMessage({
      type: ClientToWorkerMessageType.LOG_EVENT,
      payload,
    });
  }

  public shutdown(): void {
    this.port?.postMessage({
      type: ClientToWorkerMessageType.SHUTDOWN,
    });
    this.port = null;
  }

  // used for restarting the worker after a shutdown
  public start(): void {
    this.ready = this.initWorker();
  }
}

// Singleton instance
export const telemetryClient: TelemetryClient = new TelemetryClient();
