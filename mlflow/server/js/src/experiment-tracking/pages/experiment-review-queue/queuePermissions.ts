import type { ReviewQueue } from './types';

/** Case-insensitive identity compare (created_by may be normalized server-side). */
export const sameUser = (a: string | undefined, b: string): boolean =>
  (a ?? '').trim().toLowerCase() === b.trim().toLowerCase();

/**
 * Whether the current reviewer may manage a queue — delete it, and remove
 * traces from it. True when the caller manages reviews, the queue is CUSTOM
 * (personal USER queues aren't managed here), and either there's no auth
 * (everyone is admin) or the caller created it. A queue's questions are fixed
 * at creation, so there is no post-creation editing.
 */
export const canManageQueue = (
  queue: ReviewQueue,
  reviewer: string,
  authAvailable: boolean,
  canManage: boolean,
): boolean => canManage && queue.queue_type === 'CUSTOM' && (!authAvailable || sameUser(queue.created_by, reviewer));
