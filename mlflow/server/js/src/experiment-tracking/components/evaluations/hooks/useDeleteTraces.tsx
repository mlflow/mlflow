import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { MlflowService } from '../../../sdk/MlflowService';
import { invalidateMlflowSearchTracesCache } from '@databricks/web-shared/genai-traces-table';

export const useDeleteTracesMutation = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation<{ traces_deleted: number }, Error, { experimentId: string; traceRequestIds: string[] }>({
    mutationFn: async ({ experimentId, traceRequestIds }) => {
      // Chunk the trace IDs into groups of 100
      const chunks = [];
      for (let i = 0; i < traceRequestIds.length; i += 100) {
        chunks.push(traceRequestIds.slice(i, i + 100));
      }

      // Make parallel calls for each chunk
      const deletePromises = chunks.map((chunk) => MlflowService.deleteTracesV3(experimentId, chunk));

      const results = await Promise.all(deletePromises);

      // Sum up the total traces deleted
      const totalDeleted = results.reduce((sum, result) => sum + result.traces_deleted, 0);

      return { traces_deleted: totalDeleted };
    },
    onSuccess: () => invalidateMlflowSearchTracesCache({ queryClient }),
  });

  return mutation;
};
