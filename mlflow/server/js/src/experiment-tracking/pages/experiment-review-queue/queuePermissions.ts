import type { ReviewQueue } from './types';

/** Case-insensitive identity compare (created_by may be normalized server-side). */
export const sameUser = (a: string | undefined, b: string): boolean =>
  (a ?? '').trim().toLowerCase() === b.trim().toLowerCase();

/**
 * Whether the current reviewer may manage a queue — edit its assigned members
 * and remove traces from it. True when the caller manages reviews, the queue is
 * CUSTOM (personal USER queues aren't managed here), and either it is the
 * experiment's default queue (everyone's queue; any manager may edit it),
 * there's no auth (everyone is admin), or the caller created it. A queue's
 * questions are fixed at creation, so questions are never editable here.
 */
export const canManageQueue = (
  queue: ReviewQueue,
  reviewer: string,
  authAvailable: boolean,
  canManage: boolean,
): boolean =>
  canManage &&
  queue.queue_type === 'CUSTOM' &&
  (Boolean(queue.is_default) || !authAvailable || sameUser(queue.created_by, reviewer));

/**
 * Whether the current reviewer may delete a queue. Same as managing it, except
 * the experiment's default queue can never be deleted.
 */
export const canDeleteQueue = (
  queue: ReviewQueue,
  reviewer: string,
  authAvailable: boolean,
  canManage: boolean,
): boolean => !queue.is_default && canManageQueue(queue, reviewer, authAvailable, canManage);
