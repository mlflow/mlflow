import type { LabelSchema } from '../../components/label-schemas';
import type { ReviewQueue } from './types';

/**
 * A queue is only worth assigning traces to if it will actually present
 * questions to the reviewer. The rule mirrors how the Review tab resolves a
 * queue's questions (see `ExperimentReviewQueuePage`):
 *
 *  - a USER queue inherits every label schema in the experiment, so it is
 *    assignable iff the experiment defines at least one schema;
 *  - a CUSTOM queue uses its explicit subset, so it is assignable iff at
 *    least one attached `schema_id` still resolves to an existing schema
 *    (an attachment can dangle after its schema is deleted).
 */
export interface QueueAssignability {
  assignable: boolean;
  /** Why the queue can't be assigned to, when `assignable` is false. */
  reason?: 'no-experiment-schemas' | 'no-resolvable-schemas';
}

export const getQueueAssignability = (queue: ReviewQueue, labelSchemas: LabelSchema[]): QueueAssignability => {
  if (queue.queue_type === 'USER') {
    return labelSchemas.length > 0 ? { assignable: true } : { assignable: false, reason: 'no-experiment-schemas' };
  }
  const existingIds = new Set(labelSchemas.map((s) => s.schema_id));
  const hasResolvable = (queue.schema_ids ?? []).some((id) => existingIds.has(id));
  return hasResolvable ? { assignable: true } : { assignable: false, reason: 'no-resolvable-schemas' };
};
