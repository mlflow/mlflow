import { describe, it, expect } from '@jest/globals';

import { getQueueAssignability, type QueueAssignability } from './queueAssignability';
import type { LabelSchema } from '../../components/label-schemas';
import type { ReviewQueue, ReviewQueueType } from './types';

const schema = (schema_id: string): LabelSchema => ({ schema_id }) as LabelSchema;

const queue = (queue_type: ReviewQueueType, schema_ids?: string[]): ReviewQueue =>
  ({ queue_id: 'q1', queue_type, schema_ids }) as ReviewQueue;

describe('getQueueAssignability', () => {
  it.each<[string, ReviewQueue, LabelSchema[], QueueAssignability]>([
    ['USER queue with experiment schemas', queue('USER'), [schema('a')], { assignable: true }],
    [
      'USER queue with no experiment schemas',
      queue('USER'),
      [],
      { assignable: false, reason: 'no-experiment-schemas' },
    ],
    ['CUSTOM queue with a resolvable schema', queue('CUSTOM', ['a', 'gone']), [schema('a')], { assignable: true }],
    [
      'CUSTOM queue with no attached schemas',
      queue('CUSTOM', []),
      [schema('a')],
      { assignable: false, reason: 'no-resolvable-schemas' },
    ],
    [
      'CUSTOM queue whose every attached schema was deleted',
      queue('CUSTOM', ['gone']),
      [schema('a')],
      { assignable: false, reason: 'no-resolvable-schemas' },
    ],
    [
      'CUSTOM queue with undefined schema_ids',
      queue('CUSTOM', undefined),
      [schema('a')],
      { assignable: false, reason: 'no-resolvable-schemas' },
    ],
  ])('%s', (_label, q, labelSchemas, expected) => {
    expect(getQueueAssignability(q, labelSchemas)).toEqual(expected);
  });
});
