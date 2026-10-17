import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { MlflowService } from '../../../sdk/MlflowService';
import { invalidateMlflowSearchTracesCache } from '@databricks/web-shared/genai-traces-table';

import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';

/** The server's reason for a rejected chunk, with any trailing period trimmed. */
const reasonOf = (reason: unknown): string => {
  const raw =
    reason instanceof ErrorWrapper
      ? reason.getUserVisibleError()?.message
      : reason instanceof Error
        ? reason.message
        : undefined;
  return raw?.trim().replace(/\.+$/, '') || 'the request was rejected';
};

/**
 * Describe a deletion in which at least one chunk was rejected.
 *
 * Traces are deleted 100 per request, so a selection larger than that is several requests
 * and the deletion is not atomic. A condition can refuse one chunk and permit another, and
 * the permitted one really does delete. Reporting only the failure left the user looking at
 * an error beside a list that had genuinely shrunk.
 */
export const describeTraceDeletionOutcome = (failures: PromiseRejectedResult[], deleted: number): string => {
  const reasons = new Set(failures.map(({ reason }) => reasonOf(reason)));
  const why = reasons.size === 1 ? [...reasons][0] : [...reasons].join('; ');
  const head = `Could not delete some traces: ${why}`;
  if (deleted === 0) return `${head}. No traces were deleted.`;
  return `${head}. ${deleted} ${deleted === 1 ? 'trace was' : 'traces were'} deleted.`;
};

export const useDeleteTracesMutation = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation<
    { traces_deleted: number },
    Error,
    {
      experimentId: string;
      traceRequestIds: string[];
    }
  >({
    // prettier-ignore
    mutationFn: async ({
      experimentId,
      traceRequestIds,
    }) => {
      // Chunk the trace IDs into groups of 100
      const chunks = [];
      for (let i = 0; i < traceRequestIds.length; i += 100) {
        chunks.push(traceRequestIds.slice(i, i + 100));
      }


      // Make parallel calls for each chunk. `allSettled`, not `all`: every chunk is in
      // flight already, so a chunk that succeeds deletes its traces whatever the others
      // do. `Promise.all` rejected on the first failure and reported the whole deletion
      // as failed, while the successful chunks had already removed their traces -- the
      // user saw an error and a shorter list.
      const results = await Promise.allSettled(
        chunks.map((chunk) => MlflowService.deleteTracesV3(experimentId, chunk)),
      );

      const deleted = results.reduce(
        (sum, result) => sum + (result.status === 'fulfilled' ? result.value.traces_deleted : 0),
        0,
      );
      const failures = results.filter((result) => result.status === 'rejected') as PromiseRejectedResult[];
      if (failures.length > 0) {
        // Nothing was deleted: rethrow the original rejection untouched. That is the
        // pre-existing path, and consumers downstream may inspect the `ErrorWrapper` (its
        // status, its error code), so this change must not turn it into a plain `Error`.
        // Only a PARTIAL deletion is newly described, and it is a case no consumer could
        // have been reporting correctly before, because the count never reached them.
        if (deleted === 0) throw failures[0].reason;
        throw new Error(describeTraceDeletionOutcome(failures, deleted));
      }

      return { traces_deleted: deleted };
    },
    // `onSettled`, not `onSuccess`: a PARTIAL deletion throws, so it lands on the error
    // path, and the permitted chunks really are gone from the server. Invalidating only on
    // success left the table listing traces that no longer existed, which read to the user
    // as a delete that had silently failed. Refreshing after a total failure costs one
    // search request and is never wrong, so every outcome refreshes.
    onSettled: () => invalidateMlflowSearchTracesCache({ queryClient }),
  });

  return mutation;
};
