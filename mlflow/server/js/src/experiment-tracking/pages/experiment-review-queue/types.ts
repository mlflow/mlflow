/**
 * Wire-format types for review queues.
 *
 * These mirror the proto definitions in `mlflow/protos/review_queues.proto`
 * (snake_case on the wire; enum values are the uppercase proto NAME per the
 * proto-JSON convention). Server-side validation lives in
 * `mlflow/genai/review_queues/validation.py`.
 */

export type ReviewQueueType = 'USER' | 'CUSTOM';

export type ReviewStatus = 'PENDING' | 'COMPLETE' | 'DECLINED';

export type ReviewTargetType = 'TRACE';

export interface ReviewQueue {
  queue_id: string;
  experiment_id: string;
  name: string;
  queue_type: ReviewQueueType;
  created_by?: string;
  creation_time_ms: number;
  last_update_time_ms: number;
  /** Assigned-user pool (0..N; exactly one == name for a USER queue). */
  users?: string[];
  /** Attached label-schema ids; empty for a USER queue (resolves to all). */
  schema_ids?: string[];
}

export interface ReviewQueueItem {
  queue_id: string;
  target_type: ReviewTargetType;
  target_id: string;
  status: ReviewStatus;
  /** Set only in the COMPLETE / DECLINED terminal states. */
  completed_by?: string;
  completed_time_ms?: number;
  creation_time_ms: number;
  last_update_time_ms: number;
}
