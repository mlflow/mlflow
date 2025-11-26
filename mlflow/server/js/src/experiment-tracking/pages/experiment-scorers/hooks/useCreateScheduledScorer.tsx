import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import type { PredefinedError } from '@databricks/web-shared/errors';
import type { ScheduledScorer, ScorerConfig } from '../types';
import { transformScheduledScorer, convertRegisterScorerResponseToConfig } from '../utils/scorerTransformUtils';
import { updateScheduledScorersCache } from './scheduledScorersCacheUtils';
import { registerScorer, type RegisterScorerResponse } from '../api';

// Define request and response types based on monitoring_service.proto
export type CreateScheduledScorersRequest = {
  scheduled_scorers: {
    scorers: ScorerConfig[];
  };
};

export type CreateScheduledScorersResponse = {
  experiment_id: string;
  scheduled_scorers?: {
    scorers: ScorerConfig[];
  };
};

export const useCreateScheduledScorerMutation = () => {
  const queryClient = useQueryClient();

  return useMutation<
    CreateScheduledScorersResponse, // TData - response type
    PredefinedError, // TError - error type (fetchOrFail throws PredefinedError)
    { experimentId: string; scheduledScorer: ScheduledScorer } // TVariables - input type
  >({
    mutationFn: async ({ experimentId, scheduledScorer }) => {
      // Transform the scorer to backend format
      const scorerConfig = transformScheduledScorer(scheduledScorer);

      // Register the single scorer using the register endpoint
      const registerResponse: RegisterScorerResponse = await registerScorer(experimentId, scorerConfig);

      // Convert the register response to ScorerConfig
      const createdScorerConfig = convertRegisterScorerResponseToConfig(registerResponse);

      return {
        experiment_id: experimentId,
        scheduled_scorers: {
          scorers: [createdScorerConfig],
        },
      };
    },
    onSuccess: (data, variables) => {
      updateScheduledScorersCache(queryClient, data, variables.experimentId, true);
    },
  });
};
