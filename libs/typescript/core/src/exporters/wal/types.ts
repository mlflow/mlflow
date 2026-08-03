/**
 * Type definitions for the per-user trace upload WAL.
 *
 * The WAL is an append-only JSONL file. Each line is one of the variants of
 * {@link WalLine}: either an append (a complete trace record) or a tombstone
 * (a marker that shadows a previously-appended record by id).
 */

/**
 * One queued trace upload, serialized to a single JSONL line.
 */
export interface WalRecord {
  id: string;
  trackingUri: string;
  experimentId: string;
  traceInfo: Record<string, unknown>;
  traceData: unknown;
  attempts: number;
  nextAttemptAt: number;
  createdAt: number;
  firstAttemptAt?: number;
  otlpSpans?: string; // base64 encoded ExportTraceServiceRequest protobuf
}

export type WalLine = { type: 'append'; record: WalRecord } | { type: 'tombstone'; id: string };
