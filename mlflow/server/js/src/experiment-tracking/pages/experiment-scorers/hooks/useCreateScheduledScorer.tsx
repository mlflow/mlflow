import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import type { PredefinedError } from '@databricks/web-shared/errors';
import { NotFoundError } from '@databricks/web-shared/errors';
import type { ScheduledScorer, ScorerConfig } from '../types';
import { transformScheduledScorer } from '../utils/scorerTransformUtils';
import type { GetScheduledScorersResponse } from './useGetScheduledScorers';
import { updateScheduledScorersCache } from './scheduledScorersCacheUtils';
import { listScheduledScorers, updateScheduledScorers, createScheduledScorers } from '../api';

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
      // Get existing scorers to check for duplicates and build new list
      // Note: Backend returns empty list instead of 404 when no scorers exist
      const existingData: GetScheduledScorersResponse = await listScheduledScorers(experimentId);
      const existingScorerConfigs = existingData.scheduled_scorers?.scorers || [];

      // Check for duplicate names
      const existingNames = existingScorerConfigs.map((config) => config.name);
      if (existingNames.includes(scheduledScorer.name)) {
        throw new Error(
          `A scorer with name '${scheduledScorer.name}' has already been registered. ` +
            'Update the existing scorer or choose a different name.',
        );
      }

      // Add new scorer to existing list
      const newScorerConfig = transformScheduledScorer(scheduledScorer);

      // If no scorers exist, use upsert pattern to handle empty list case
      if (existingScorerConfigs.length === 0) {
        // Use upsert helper: try update first, fallback to create on 404
        try {
          return await updateScheduledScorers(experimentId, { scorers: [newScorerConfig] });
        } catch (updateError) {
          // If update fails with 404, no scorers exist yet, create them
          if (updateError instanceof NotFoundError) {
            return await createScheduledScorers(experimentId, { scorers: [newScorerConfig] });
          }
          // Re-throw other errors
          throw updateError;
        }
      } else {
        // Add new scorer to existing list and update
        const newScorerConfigs = [...existingScorerConfigs, newScorerConfig];

        return await updateScheduledScorers(experimentId, { scorers: newScorerConfigs });
      }
    },
    onSuccess: (data, variables) => {
      updateScheduledScorersCache(queryClient, data, variables.experimentId);
    },
  });
};
