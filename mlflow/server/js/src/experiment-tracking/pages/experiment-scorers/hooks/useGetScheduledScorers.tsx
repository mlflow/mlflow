import { useQuery, type UseQueryResult } from '@databricks/web-shared/query-client';
import { UnknownError, type PredefinedError } from '@databricks/web-shared/errors';
import type { ScheduledScorer, ScorerConfig } from '../types';
import { convertMLflowScorerToConfig, transformScorerConfig } from '../utils/scorerTransformUtils';
import { listScheduledScorers, ListScorersResponse } from '../api';

// Define response types
export type GetScheduledScorersResponse = {
  experiment_id: string;
  scheduled_scorers?: {
    scorers: ScorerConfig[];
  };
};

export interface ScheduledScorersResponse {
  experimentId: string;
  scheduledScorers: ScheduledScorer[];
}

export function useGetScheduledScorers(
  experimentId?: string,
): UseQueryResult<ScheduledScorersResponse, PredefinedError> {
  return useQuery<ScheduledScorersResponse, PredefinedError>({
    queryKey: ['mlflow', 'scheduled-scorers', experimentId],
    queryFn: async () => {
      if (!experimentId) {
        throw new UnknownError('Experiment ID is required');
      }
      const response: ListScorersResponse = await listScheduledScorers(experimentId);

      // Transform the response to match ScheduledScorersResponse
      const scheduledScorers = response.scorers?.map(convertMLflowScorerToConfig).map(transformScorerConfig) || [];

      return {
        experimentId: experimentId,
        scheduledScorers,
      };
    },
    enabled: !!experimentId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: true,
  });
}
