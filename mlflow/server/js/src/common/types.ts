export type AliasMap = { alias: string; version: string }[];

/**
 * Simple key/value
 */
export interface KeyValueEntity {
  key: string;
  value: string;
}

export interface JobProgressPayload {
  phase?: string;
  completed?: number;
  total?: number;
  unit?: string;
}

export interface JobProgressMetadata {
  error_message?: string | null;
  status_message?: string | null;
  progress_payload?: JobProgressPayload | null;
  progress_updated_at?: number | null;
}
