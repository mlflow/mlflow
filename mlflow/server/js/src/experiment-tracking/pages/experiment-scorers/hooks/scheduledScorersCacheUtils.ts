import type { QueryClient } from '@databricks/web-shared/query-client';
import { transformScorerConfig } from '../utils/scorerTransformUtils';
import type { CreateScheduledScorersResponse } from './useCreateScheduledScorer';
import type { ScheduledScorersResponse } from './useGetScheduledScorers';

/**
 * Updates the scheduled scorers cache optimistically or falls back to invalidation.
 * Shared utility for both create and update mutations.
 */
export function updateScheduledScorersCache(
  queryClient: QueryClient,
  data: CreateScheduledScorersResponse,
  experimentId: string,
): void {
  // Get the existing cached data to compare experiment IDs
  const existingData: ScheduledScorersResponse | undefined = queryClient.getQueryData([
    'mlflow',
    'scheduled-scorers',
    experimentId,
  ]);

  // Safely update the cache if response structure is valid and experiment ID matches cached data
  if (
    data &&
    data.scheduled_scorers &&
    data.experiment_id &&
    (!existingData?.experimentId || data.experiment_id === existingData.experimentId)
  ) {
    const scheduledScorers = data.scheduled_scorers?.scorers?.map(transformScorerConfig) || [];

    queryClient.setQueryData<ScheduledScorersResponse>(['mlflow', 'scheduled-scorers', experimentId], {
      experimentId: data.experiment_id,
      scheduledScorers,
    });
  } else {
    // Fallback to invalidation if response structure is unexpected or experiment ID mismatch
    queryClient.invalidateQueries(['mlflow', 'scheduled-scorers', experimentId]);
  }
}
