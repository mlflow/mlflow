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
 * Whether the current reviewer may manage a CUSTOM queue's settings — edit its
 * name / members / questions. Allowed to an experiment manager (`canManage`) or
 * the owning experiment EDITor (`canEdit` and they created it). Personal USER
 * queues have no editable settings, so they're never managed here. Ownership
 * amplifies EDIT and never substitutes for it, mirroring the server-side gate.
 */
export const canManageQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  queue.queue_type === 'CUSTOM' && (canManage || (canEdit && isQueueOwner(queue, reviewer)));

/**
 * Shared rule for the destructive owner/manager queue actions — deleting the
 * queue and removing its items (un-assigning traces). A manager may act on any
 * queue; an EDIT owner only on their own CUSTOM queue. A personal USER queue's
 * lifecycle and contents are a manager's responsibility, never its assignee's,
 * so an EDIT user can neither delete their USER queue nor prune its traces.
 * Mirrors the server's `validate_can_delete_review_queue` /
 * `validate_can_remove_items_from_review_queue`.
 */
const canDeleteOrPruneQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  canManage || (canEdit && isQueueOwner(queue, reviewer) && queue.queue_type === 'CUSTOM');

/** Whether the reviewer may delete a queue. See {@link canDeleteOrPruneQueue}. */
export const canDeleteQueue = canDeleteOrPruneQueue;

/**
 * Whether the reviewer may remove items (un-assign traces) from a queue. Same
 * rule as deleting it: a manager may prune any queue, but an EDIT owner only
 * their own CUSTOM queue. See {@link canDeleteOrPruneQueue}.
 */
export const canRemoveQueueItems = canDeleteOrPruneQueue;

/**
 * Whether the current reviewer may open a queue and read its items / answers: a
 * manager, the owning EDITor, or an assigned member. Mirrors the server's
 * detail-tier gate (`validate_can_view_review_queue`).
 */
export const canInspectQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  canManage || (canEdit && isQueueOwner(queue, reviewer)) || isQueueMember(queue, reviewer);
