import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import type { PredefinedError } from '@databricks/web-shared/errors';
import type { ScheduledScorer, ScorerConfig } from '../types';
import { transformScheduledScorer, convertRegisterScorerResponseToConfig } from '../utils/scorerTransformUtils';
import { updateScheduledScorersCache } from './scheduledScorersCacheUtils';
import { registerScorer, type RegisterScorerResponse } from '../api';

// Define request and response types based on monitoring_service.proto
export type UpdateScheduledScorersRequest = {
  scheduled_scorers: {
    scorers: ScorerConfig[];
  };
  update_mask: {
    paths: string;
  };
};

export type UpdateScheduledScorersResponse = {
  experiment_id: string;
  scheduled_scorers?: {
    scorers: ScorerConfig[];
  };
};

export const useUpdateScheduledScorerMutation = () => {
  const queryClient = useQueryClient();

  return useMutation<
    UpdateScheduledScorersResponse, // TData - response type
    PredefinedError, // TError - error type (fetchOrFail throws PredefinedError)
    {
      experimentId: string;
      scheduledScorers: ScheduledScorer[];
    } // TVariables - input type
  >({
    mutationFn: async ({ experimentId, scheduledScorers }) => {
      // The backend will update if the name already exists
      const updatePromises = scheduledScorers.map(async (scorer) => {
        const scorerConfig = transformScheduledScorer(scorer);
        const registerResponse: RegisterScorerResponse = await registerScorer(experimentId, scorerConfig);
        return convertRegisterScorerResponseToConfig(registerResponse);
      });

      // Wait for all updates to complete and collect the updated configs
      const updatedScorerConfigs = await Promise.all(updatePromises);

      // Return response in expected format
      return {
        experiment_id: experimentId,
        scheduled_scorers: {
          scorers: updatedScorerConfigs,
        },
      };
    },
    onSuccess: (data, variables) => {
      updateScheduledScorersCache(queryClient, data, variables.experimentId, true);
    },
  });
};
