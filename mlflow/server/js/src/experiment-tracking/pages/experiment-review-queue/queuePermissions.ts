import type { ReviewQueue } from './types';

/** Case-insensitive identity compare (created_by may be normalized server-side). */
export const sameUser = (a: string | undefined, b: string): boolean =>
  (a ?? '').trim().toLowerCase() === b.trim().toLowerCase();

/**
 * Whether the current reviewer may manage a queue — edit its assigned members
 * and questions, and remove items from it. Management requires experiment
 * MANAGE (folded into `canManage`); only CUSTOM queues are managed here (personal
 * USER queues aren't). Visibility already restricts which queues a non-manager
 * sees, so no per-queue ownership check is needed.
 */
export const canManageQueue = (queue: ReviewQueue, canManage: boolean): boolean =>
  canManage && queue.queue_type === 'CUSTOM';

/**
 * Whether the current reviewer may delete a queue. Currently the same as
 * managing it (a distinct action, kept separate so the two can diverge).
 */
export const canDeleteQueue = (queue: ReviewQueue, canManage: boolean): boolean => canManageQueue(queue, canManage);
