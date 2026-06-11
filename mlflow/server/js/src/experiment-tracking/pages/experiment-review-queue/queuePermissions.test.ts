import { describe, it, expect } from '@jest/globals';

import {
  canInspectQueue,
  canManageQueue,
  isQueueMember,
  isQueueOwner,
  sameUser,
} from './queuePermissions';
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

describe('isQueueOwner / isQueueMember', () => {
  it('matches the owner case-insensitively', () => {
    expect(isQueueOwner(queue({ created_by: 'Alice' }), 'alice')).toBe(true);
    expect(isQueueOwner(queue({ created_by: 'bob' }), 'alice')).toBe(false);
    expect(isQueueOwner(queue({ created_by: undefined }), 'alice')).toBe(false);
  });

  it('matches assigned-pool membership case-insensitively', () => {
    expect(isQueueMember(queue({ users: ['Bob', 'alice'] }), 'ALICE')).toBe(true);
    expect(isQueueMember(queue({ users: ['bob'] }), 'alice')).toBe(false);
    expect(isQueueMember(queue({ users: undefined }), 'alice')).toBe(false);
  });
});

describe('canManageQueue', () => {
  const manager = { canManage: true, canEdit: true };
  const editor = { canManage: false, canEdit: true };
  const reader = { canManage: false, canEdit: false };

  it('lets a manager manage any CUSTOM queue regardless of owner', () => {
    expect(canManageQueue(queue({ created_by: 'someone-else' }), 'alice', true, true)).toBe(true);
  });

  it('lets an EDIT owner manage their own CUSTOM queue', () => {
    expect(canManageQueue(queue({ created_by: 'alice' }), 'alice', editor.canManage, editor.canEdit)).toBe(true);
  });

  it('does not let an EDIT non-owner manage a CUSTOM queue', () => {
    expect(canManageQueue(queue({ created_by: 'bob' }), 'alice', editor.canManage, editor.canEdit)).toBe(false);
  });

  it('does not let a READ owner manage (ownership amplifies EDIT, never substitutes)', () => {
    expect(canManageQueue(queue({ created_by: 'alice' }), 'alice', reader.canManage, reader.canEdit)).toBe(false);
  });

  it('never manages USER (personal) queues, even for a manager', () => {
    expect(canManageQueue(queue({ queue_type: 'USER' }), 'alice', manager.canManage, manager.canEdit)).toBe(false);
  });
});

describe('canInspectQueue', () => {
  it('allows a manager, the EDIT owner, or an assigned member', () => {
    // Manager (not owner, not member).
    expect(canInspectQueue(queue({ created_by: 'bob', users: ['carol'] }), 'alice', true, true)).toBe(true);
    // EDIT owner (not member).
    expect(canInspectQueue(queue({ created_by: 'alice', users: ['carol'] }), 'alice', false, true)).toBe(true);
    // Member with only READ.
    expect(canInspectQueue(queue({ created_by: 'bob', users: ['alice'] }), 'alice', false, false)).toBe(true);
  });

  it('denies an EDIT non-owner non-member', () => {
    expect(canInspectQueue(queue({ created_by: 'bob', users: ['carol'] }), 'alice', false, true)).toBe(false);
  });

  it('denies a READ owner who is not a member (ownership needs EDIT)', () => {
    expect(canInspectQueue(queue({ created_by: 'alice', users: ['carol'] }), 'alice', false, false)).toBe(false);
  });
});
