export enum WorkerToClientMessageType {
  READY = 'READY',
}

export enum ClientToWorkerMessageType {
  LOG_EVENT = 'LOG_EVENT',
  SHUTDOWN = 'SHUTDOWN',
}

export interface TelemetryRecord {
  installation_id: string;
  session_id: string;
  event_name: string;
  timestamp_ns: number;
  params?: Record<string, string | null | undefined>;
  status?: string;
  duration_ms?: number;
}

export interface TelemetryConfig {
  disable_ui_events?: string[];
  disable_ui_telemetry?: boolean;
  ui_rollout_percentage?: number;
}

export interface TelemetryConfigResponse {
  config: TelemetryConfig | null;
}

export interface TelemetryMessage {
  type: ClientToWorkerMessageType;
  payload?: unknown;
  requestId?: string;
}
