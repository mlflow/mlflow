import { describe, it, expect } from '@jest/globals';

import { describeTraceDeletionOutcome } from './useDeleteTraces';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';

const rejection = (message: string): PromiseRejectedResult => ({
  status: 'rejected',
  reason: new ErrorWrapper(JSON.stringify({ error_code: 'PERMISSION_DENIED', message }), 403),
});

describe('describeTraceDeletionOutcome', () => {
  // Traces delete 100 per request, so a larger selection is several requests and the
  // deletion is not atomic. A condition can refuse one chunk and permit another, and the
  // permitted chunk really deletes. `Promise.all` reported the whole deletion as failed,
  // leaving the user with an error beside a list that had genuinely shrunk.

  it('reports how many were deleted alongside the failure', () => {
    const message = describeTraceDeletionOutcome([rejection('Permission denied.')], 100);
    expect(message).toContain('Could not delete some traces');
    expect(message).toContain('Permission denied');
    expect(message).toContain('100 traces were deleted');
    expect(message).not.toContain('..');
  });

  it('says plainly when nothing was deleted', () => {
    // The hook does not use this branch -- when nothing was deleted it rethrows the
    // original `ErrorWrapper` so downstream consumers can still inspect its status. The
    // wording is kept because the function is exported and the branch is reachable.
    const message = describeTraceDeletionOutcome([rejection('Permission denied.')], 0);
    expect(message).toContain('No traces were deleted');
  });

  it('agrees in number for a single trace', () => {
    expect(describeTraceDeletionOutcome([rejection('Nope.')], 1)).toContain('1 trace was deleted');
  });

  it('states one shared reason once', () => {
    const message = describeTraceDeletionOutcome(
      [rejection('Permission denied.'), rejection('Permission denied.')],
      50,
    );
    expect(message.match(/Permission denied/g)).toHaveLength(1);
  });
});
