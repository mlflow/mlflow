import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import type { PredefinedError } from '@databricks/web-shared/errors';
import type { ScorerConfig } from '../types';
import type { GetScheduledScorersResponse } from './useGetScheduledScorers';
import { updateScheduledScorersCache } from './scheduledScorersCacheUtils';
import { listScheduledScorers, updateScheduledScorers, deleteScheduledScorers } from '../api';

// Define response types based on monitoring_service.proto
export type DeleteScheduledScorersResponse = {
  experiment_id: string;
  scheduled_scorers?: {
    scorers: ScorerConfig[];
  };
};

export const useDeleteScheduledScorerMutation = () => {
  const queryClient = useQueryClient();

  return useMutation<
    DeleteScheduledScorersResponse, // TData - response type
    PredefinedError, // TError - error type (fetchOrFail throws PredefinedError)
    {
      experimentId: string;
      scorerNames?: string[]; // If provided, delete only specific scorers. If not provided, delete all scorers.
    } // TVariables - input type
  >({
    mutationFn: async ({ experimentId, scorerNames }) => {
      // If no specific scorer names provided, delete all scorers using the DELETE endpoint
      if (!scorerNames || scorerNames.length === 0) {
        return await deleteScheduledScorers(experimentId);
      }

      // For selective deletion, get existing scorers and remove only specified ones using PATCH
      const existingData: GetScheduledScorersResponse = await listScheduledScorers(experimentId);
      const existingScorerConfigs = existingData.scheduled_scorers?.scorers || [];

      // Filter out the scorers to be deleted
      const remainingScorerConfigs = existingScorerConfigs.filter((config) => !scorerNames.includes(config.name));

      // If no scorers remain after filtering, delete all scorers using DELETE endpoint
      if (remainingScorerConfigs.length === 0) {
        return await deleteScheduledScorers(experimentId);
      }

      // Otherwise, update with the remaining scorers using PATCH endpoint
      return await updateScheduledScorers(experimentId, { scorers: remainingScorerConfigs });
    },
    onSuccess: (data, variables) => {
      updateScheduledScorersCache(queryClient, data, variables.experimentId);
    },
  });
};
