import { describe, it, expect } from '@jest/globals';

import { canManageQueue, sameUser } from './queuePermissions';
import type { ReviewQueue } from './types';

const queue = (overrides: Partial<ReviewQueue> = {}): ReviewQueue => ({
  queue_id: 'rq-1',
  experiment_id: 'exp-1',
  name: 'Relevance review',
  queue_type: 'CUSTOM',
  created_by: 'alice',
  creation_time_ms: 1_780_000_000_000,
  last_update_time_ms: 1_780_000_000_000,
  ...overrides,
});

describe('sameUser', () => {
  it('compares case-insensitively and trims whitespace', () => {
    expect(sameUser('Alice', 'alice')).toBe(true);
    expect(sameUser('  alice ', 'ALICE')).toBe(true);
    expect(sameUser('alice', 'bob')).toBe(false);
  });

  it('treats a missing identity as no match', () => {
    expect(sameUser(undefined, 'alice')).toBe(false);
    expect(sameUser('', 'alice')).toBe(false);
  });
});

describe('canManageQueue', () => {
  it('lets a review manager manage any CUSTOM queue, regardless of owner', () => {
    expect(canManageQueue(queue({ created_by: 'someone-else' }), true)).toBe(true);
    expect(canManageQueue(queue({ created_by: 'alice' }), true)).toBe(true);
  });

  it('never manages USER (personal) queues', () => {
    expect(canManageQueue(queue({ queue_type: 'USER' }), true)).toBe(false);
  });

  it('requires review-management (MANAGE) permission', () => {
    expect(canManageQueue(queue({ created_by: 'alice' }), false)).toBe(false);
  });
});
