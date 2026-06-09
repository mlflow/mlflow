import { describe, it, expect } from '@jest/globals';

import { canDeleteQueue, canManageQueue, sameUser } from './queuePermissions';
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
  it('lets anyone who manages reviews manage CUSTOM queues on a no-auth server', () => {
    expect(canManageQueue(queue({ created_by: 'someone-else' }), 'default', false, true)).toBe(true);
  });

  it('lets the owner manage their CUSTOM queue on an auth server', () => {
    expect(canManageQueue(queue({ created_by: 'alice' }), 'alice', true, true)).toBe(true);
  });

  it('does not let a non-owner manage a CUSTOM queue on an auth server', () => {
    expect(canManageQueue(queue({ created_by: 'alice' }), 'bob', true, true)).toBe(false);
  });

  it('never manages USER (personal) queues', () => {
    expect(canManageQueue(queue({ queue_type: 'USER', created_by: 'alice' }), 'alice', true, true)).toBe(false);
    expect(canManageQueue(queue({ queue_type: 'USER' }), 'default', false, true)).toBe(false);
  });

  it('requires review-management permission', () => {
    expect(canManageQueue(queue({ created_by: 'alice' }), 'alice', true, false)).toBe(false);
  });

  it('lets any manager manage the default queue, even a non-owner on an auth server', () => {
    expect(canManageQueue(queue({ is_default: true, created_by: 'someone-else' }), 'bob', true, true)).toBe(true);
    // still gated on review-management permission
    expect(canManageQueue(queue({ is_default: true, created_by: 'someone-else' }), 'bob', true, false)).toBe(false);
  });
});

describe('canDeleteQueue', () => {
  it('lets a manager delete a custom queue they can manage', () => {
    expect(canDeleteQueue(queue({ created_by: 'alice' }), 'alice', true, true)).toBe(true);
    expect(canDeleteQueue(queue({ created_by: 'someone-else' }), 'default', false, true)).toBe(true);
  });

  it('never deletes the default queue, even for a manager who can otherwise manage it', () => {
    expect(canDeleteQueue(queue({ is_default: true }), 'alice', true, true)).toBe(false);
    expect(canDeleteQueue(queue({ is_default: true, created_by: 'alice' }), 'alice', false, true)).toBe(false);
  });

  it('does not delete a queue the caller cannot manage', () => {
    expect(canDeleteQueue(queue({ created_by: 'alice' }), 'bob', true, true)).toBe(false);
  });
});
