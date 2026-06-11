import type { ReviewQueue } from './types';

/** Case-insensitive identity compare (created_by may be normalized server-side). */
export const sameUser = (a: string | undefined, b: string): boolean =>
  (a ?? '').trim().toLowerCase() === b.trim().toLowerCase();

/** Whether the reviewer owns the queue (its server-stamped `created_by`). */
export const isQueueOwner = (queue: ReviewQueue, reviewer: string): boolean => sameUser(queue.created_by, reviewer);

/** Whether the reviewer is in the queue's assigned-user pool. */
export const isQueueMember = (queue: ReviewQueue, reviewer: string): boolean =>
  (queue.users ?? []).some((u) => sameUser(u, reviewer));

/**
 * Whether the current reviewer may manage a CUSTOM queue — edit its name /
 * members / questions, remove items, and delete it. Allowed to an experiment
 * manager (`canManage`) or the owning experiment EDITor (`canEdit` and they
 * created it). Personal USER queues are never managed here. Ownership amplifies
 * EDIT and never substitutes for it, mirroring the server-side gate.
 */
export const canManageQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  queue.queue_type === 'CUSTOM' && (canManage || (canEdit && isQueueOwner(queue, reviewer)));

/**
 * Whether the current reviewer may delete a queue: a manager deletes any queue
 * (including a personal USER queue), while an owning EDITor deletes only their
 * own CUSTOM queue. This is broader than {@link canManageQueue} — a USER queue
 * has no editable settings but a manager can still delete it. Mirrors the
 * server's `validate_can_delete_review_queue`.
 */
export const canDeleteQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  canManage || (canEdit && isQueueOwner(queue, reviewer) && queue.queue_type === 'CUSTOM');

/**
 * Whether the current reviewer may open a queue and read its items / answers: a
 * manager, the owning EDITor, or an assigned member. Mirrors the server's
 * detail-tier gate (`validate_can_view_review_queue`).
 */
export const canInspectQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  canManage || (canEdit && isQueueOwner(queue, reviewer)) || isQueueMember(queue, reviewer);
