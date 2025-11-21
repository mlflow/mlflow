import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import type { PredefinedError } from '@databricks/web-shared/errors';
import type { ScheduledScorer, ScorerConfig } from '../types';
import { transformScheduledScorer } from '../utils/scorerTransformUtils';
import type { GetScheduledScorersResponse } from './useGetScheduledScorers';
import { updateScheduledScorersCache } from './scheduledScorersCacheUtils';
import { listScheduledScorers, updateScheduledScorers } from '../api';

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
      // First, fetch existing scorers
      const existingData: GetScheduledScorersResponse = await listScheduledScorers(experimentId);

      // Get existing scorers or empty array if none exist
      const existingScorerConfigs = existingData.scheduled_scorers?.scorers || [];

      // Create a map of scheduled scorer names to update
      const updateMap = new Map(scheduledScorers.map((scorer) => [scorer.name, scorer]));

      // Update existing scorers in place, keeping ones not being updated
      const updatedScorerConfigs = existingScorerConfigs.map((existingConfig) => {
        const updatedScorer = updateMap.get(existingConfig.name);
        if (updatedScorer) {
          // Transform and replace with updated scorer
          return transformScheduledScorer(updatedScorer);
        }
        // Keep existing scorer as-is
        return existingConfig;
      });

      // Add any new scorers that didn't exist before
      scheduledScorers.forEach((scorer) => {
        const exists = existingScorerConfigs.some((existing) => existing.name === scorer.name);
        if (!exists) {
          updatedScorerConfigs.push(transformScheduledScorer(scorer));
        }
      });

      return updateScheduledScorers(experimentId, { scorers: updatedScorerConfigs });
    },
    onSuccess: (data, variables) => {
      updateScheduledScorersCache(queryClient, data, variables.experimentId);
    },
  });
};
