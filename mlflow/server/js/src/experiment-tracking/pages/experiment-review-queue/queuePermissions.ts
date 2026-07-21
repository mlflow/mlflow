import type { ReviewQueue } from './types';

/**
 * Canonical form of a user identifier. Mirrors the server's `normalize_user`
 * (strip + lowercase): assigned users and user-queue names are stored normalized,
 * while the user roster carries display casing, so compare/key on this form.
 */
export const normalizeUser = (user: string | undefined): string => (user ?? '').trim().toLowerCase();

/** Case-insensitive identity compare (created_by may be normalized server-side). */
export const sameUser = (a: string | undefined, b: string): boolean => normalizeUser(a) === normalizeUser(b);

/** Whether the reviewer owns the queue (its server-stamped `created_by`). */
export const isQueueOwner = (queue: ReviewQueue, reviewer: string): boolean => sameUser(queue.created_by, reviewer);

/** Whether the reviewer is in the queue's assigned-user pool. */
export const isQueueMember = (queue: ReviewQueue, reviewer: string): boolean =>
  (queue.users ?? []).some((u) => sameUser(u, reviewer));

// The helpers below mirror the server-side review-queue gates (`validate_can_*`).
// Ownership (`isQueueOwner`) amplifies EDIT but never substitutes for it.

/** Manage a CUSTOM queue's settings (name / members / questions): manager, or owning editor. */
export const canManageQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  queue.queue_type === 'CUSTOM' && (canManage || (canEdit && isQueueOwner(queue, reviewer)));

/**
 * Shared rule for the destructive actions (delete queue, remove items): a manager
 * acts on any queue, an EDIT owner only on their own CUSTOM queue. A USER queue's
 * lifecycle is a manager's responsibility, never its assignee's.
 */
const canDeleteOrPruneQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  canManage || (canEdit && isQueueOwner(queue, reviewer) && queue.queue_type === 'CUSTOM');

export const canDeleteQueue = canDeleteOrPruneQueue;
export const canRemoveQueueItems = canDeleteOrPruneQueue;

/** Open a queue and read its items / answers: manager, owning editor, or assigned member. */
export const canInspectQueue = (queue: ReviewQueue, reviewer: string, canManage: boolean, canEdit: boolean): boolean =>
  canManage || (canEdit && isQueueOwner(queue, reviewer)) || isQueueMember(queue, reviewer);
