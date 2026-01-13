import { useQuery, type UseQueryResult } from '@databricks/web-shared/query-client';
import { UnknownError, type PredefinedError } from '@databricks/web-shared/errors';
import type { ScheduledScorer, ScorerConfig } from '../types';
import { convertMLflowScorerToConfig, transformScorerConfig } from '../utils/scorerTransformUtils';
import { listScheduledScorers, getOnlineScoringConfigs, ListScorersResponse } from '../api';

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

      // Convert to ScorerConfig format first
      const scorerConfigs = response.scorers?.map(convertMLflowScorerToConfig) || [];

      // Get scorer IDs to fetch online configs
      const scorerIds = response.scorers?.map((scorer) => scorer.scorer_id).filter(Boolean) || [];

      // Fetch online scoring configs if there are scorers
      const onlineConfigs: Record<string, { sample_rate: number; filter_string?: string }> = {};
      if (scorerIds.length > 0) {
        try {
          const configsResponse = await getOnlineScoringConfigs(scorerIds);
          // Backend returns an array of configs, convert to object keyed by scorer_id
          const configsArray = configsResponse.configs || [];
          for (const config of configsArray) {
            if (config.scorer_id) {
              onlineConfigs[config.scorer_id] = config;
            }
          }
        } catch {
          // If fetching online configs fails, continue without them
          // The UI will show default values
        }
      }

      // Merge online configs into scorer configs
      const scorerConfigsWithOnlineConfig = scorerConfigs.map((config, index) => {
        const scorerId = response.scorers?.[index]?.scorer_id;
        const onlineConfig = scorerId ? onlineConfigs[scorerId] : undefined;
        if (onlineConfig) {
          return {
            ...config,
            sample_rate: onlineConfig.sample_rate,
            filter_string: onlineConfig.filter_string,
          };
        }
        return config;
      });

      // Transform the response to match ScheduledScorersResponse
      const scheduledScorers = scorerConfigsWithOnlineConfig.map(transformScorerConfig);

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
