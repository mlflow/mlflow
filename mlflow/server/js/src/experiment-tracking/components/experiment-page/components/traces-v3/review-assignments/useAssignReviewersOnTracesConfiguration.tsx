import { useCallback, useMemo, useState } from 'react';

import { BulkAssignReviewersModal } from './BulkAssignReviewersModal';

/**
 * Bundles the bulk "Assign Reviewers" modal with a callback that opens
 * it for a set of trace ids, so the traces table can expose it through
 * the shared ``TraceActions.assignReviewersAction`` slot without owning
 * any review-assignment state itself.
 */
export const useAssignReviewersOnTracesConfiguration = (experimentId: string) => {
  const [traceIds, setTraceIds] = useState<string[] | null>(null);

  const showAssignReviewersModal = useCallback((ids: string[]) => {
    setTraceIds(ids);
  }, []);

  const AssignReviewersModal = useMemo(
    () =>
      traceIds === null ? null : (
        <BulkAssignReviewersModal experimentId={experimentId} traceIds={traceIds} onClose={() => setTraceIds(null)} />
      ),
    [traceIds, experimentId],
  );

  return { showAssignReviewersModal, AssignReviewersModal };
};
