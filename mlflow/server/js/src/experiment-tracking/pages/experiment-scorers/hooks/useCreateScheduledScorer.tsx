import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import type { PredefinedError } from '@databricks/web-shared/errors';
import type { ScheduledScorer, ScorerConfig } from '../types';
import { transformScheduledScorer, convertRegisterScorerResponseToConfig } from '../utils/scorerTransformUtils';
import { updateScheduledScorersCache } from './scheduledScorersCacheUtils';
import { registerScorer, updateOnlineScoringConfig, type RegisterScorerResponse } from '../api';

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

      // Also update the online scoring config (sample_rate, filter_string)
      // Convert from percentage (0-100) to decimal (0-1) for the API
      const sampleRateDecimal = (scheduledScorer.sampleRate ?? 0) / 100;
      await updateOnlineScoringConfig(
        experimentId,
        scheduledScorer.name,
        sampleRateDecimal,
        scheduledScorer.filterString,
      );

      // Convert the register response to ScorerConfig and add online config fields
      const createdScorerConfig = convertRegisterScorerResponseToConfig(registerResponse);
      createdScorerConfig.sample_rate = sampleRateDecimal;
      createdScorerConfig.filter_string = scheduledScorer.filterString;

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
