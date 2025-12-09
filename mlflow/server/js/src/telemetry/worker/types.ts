export enum WorkerToClientMessageType {
  READY = 'READY',
}

export enum ClientToWorkerMessageType {
  LOG_EVENT = 'LOG_EVENT',
}

export interface TelemetryRecord {
  session_id: string;
  event_name: string;
  timestamp_ns: number;
  params?: Record<string, unknown>;
  status?: string;
  duration_ms?: number;
}

export interface TelemetryConfig {
  disable_ui_events: string[];
  disable_telemetry: boolean;
  mlflow_version: '3.7.1.dev0';
  ui_rollout_percentage: number;
}

export interface TelemetryConfigResponse {
  config: TelemetryConfig | null;
}

export interface TelemetryMessage {
  type: ClientToWorkerMessageType;
  payload?: unknown;
  requestId?: string;
}
