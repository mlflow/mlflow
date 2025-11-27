import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import type { PredefinedError } from '@databricks/web-shared/errors';
import type { ScorerConfig } from '../types';
import { removeScheduledScorersFromCache } from './scheduledScorersCacheUtils';
import { deleteScheduledScorers } from '../api';

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
      return await deleteScheduledScorers(experimentId, scorerNames);
    },
    onSuccess: (data, variables) => {
      removeScheduledScorersFromCache(queryClient, variables.experimentId, variables.scorerNames);
    },
  });
};
