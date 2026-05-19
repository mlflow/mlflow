/**
 * Type definitions for the per-user trace upload WAL.
 *
 * The WAL is an append-only JSONL file. Each line is one of the variants of
 * {@link WalLine}: either an append (a complete trace record) or a tombstone
 * (a marker that shadows a previously-appended record by id).
 */

/**
 * One queued trace upload, serialized to a single JSONL line.
 *
 * Field semantics:
 * - `id`: a fresh uuid v4 per WAL row (not the MLflow trace id). Retries
 *   tombstone the old row and re-append a new row with a fresh `id`, so
 *   per-attempt rows are independently tombstone-addressable. The MLflow
 *   trace id lives inside `traceInfo`.
 * - `trackingUri`: routing key. The daemon groups records by this field
 *   and uses one memoized `MlflowClient` per distinct value.
 * - `experimentId`: captured at hook time so the daemon does not need to
 *   re-resolve experiments at upload time.
 * - `traceInfo` / `traceData`: results of `TraceInfo.toJson()` and
 *   `TraceData.toJson()`. Deserialized via the corresponding `fromJson`
 *   factories inside the daemon's batch loop.
 * - `attempts`, `nextAttemptAt`: retry state. `nextAttemptAt` is the unix
 *   ms epoch before which the daemon skips this record on a batch tick.
 * - `createdAt`: when the record first entered the WAL (unix ms). Used
 *   for diagnostics / ordering only; not part of retry semantics.
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
}

/**
 * Discriminated union for a single line of the WAL file.
 *
 * Readers (`storage.readPending`) replay the file linewise into a
 * `Map<id, WalRecord>`, applying tombstones as deletions. The "current"
 * set of pending records is whatever survives the replay.
 */
export type WalLine =
  | { type: 'append'; record: WalRecord }
  | { type: 'tombstone'; id: string };
