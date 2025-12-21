import type { QueryClient } from '@databricks/web-shared/query-client';
import { transformScorerConfig } from '../utils/scorerTransformUtils';
import type { ScheduledScorer } from '../types';
import type { CreateScheduledScorersResponse } from './useCreateScheduledScorer';
import type { UpdateScheduledScorersResponse } from './useUpdateScheduledScorer';
import type { ScheduledScorersResponse } from './useGetScheduledScorers';

/**
 * Updates the scheduled scorers cache optimistically or falls back to invalidation.
 * Shared utility for both create and update mutations.
 *
 * @param queryClient - The React Query client
 * @param data - Response data containing scheduled scorers
 * @param experimentId - The experiment ID
 * @param merge - If true, merges new scorers with existing ones; if false, replaces entire cache
 */
export function updateScheduledScorersCache(
  queryClient: QueryClient,
  data: CreateScheduledScorersResponse | UpdateScheduledScorersResponse,
  experimentId: string,
  merge: boolean = false,
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
    const newScorers = data.scheduled_scorers?.scorers?.map(transformScorerConfig) || [];

    let finalScorers: ScheduledScorer[];

    if (merge && existingData?.scheduledScorers) {
      // Merge: update existing scorers by name, keep others unchanged
      const updatedScorerMap = new Map(newScorers.map((scorer) => [scorer.name, scorer]));

      finalScorers = existingData.scheduledScorers.map((existingScorer) => {
        return updatedScorerMap.get(existingScorer.name) || existingScorer;
      });

      // Add any new scorers that don't exist in the cache yet
      const existingNames = new Set(existingData.scheduledScorers.map((s) => s.name));
      newScorers.forEach((scorer) => {
        if (!existingNames.has(scorer.name)) {
          finalScorers.push(scorer);
        }
      });
    } else {
      // Fully replace the cache with the new scorers
      finalScorers = newScorers;
    }

    queryClient.setQueryData<ScheduledScorersResponse>(['mlflow', 'scheduled-scorers', experimentId], {
      experimentId: data.experiment_id,
      scheduledScorers: finalScorers,
    });
  } else {
    // Fallback to invalidation if response structure is unexpected or experiment ID mismatch
    queryClient.invalidateQueries(['mlflow', 'scheduled-scorers', experimentId]);
  }
}

/**
 * Removes scheduled scorers from the cache by name or invalidates if needed.
 * Used by the delete mutation.
 *
 * @param queryClient - The React Query client
 * @param experimentId - The experiment ID
 * @param scorerNames - Array of scorer names to remove; if empty/undefined, invalidates cache
 */
export function removeScheduledScorersFromCache(
  queryClient: QueryClient,
  experimentId: string,
  scorerNames?: string[],
): void {
  const existingData = queryClient.getQueryData<ScheduledScorersResponse>([
    'mlflow',
    'scheduled-scorers',
    experimentId,
  ]);

  if (existingData?.scheduledScorers && scorerNames && scorerNames.length > 0) {
    // Remove the deleted scorers from the cache
    const deletedNamesSet = new Set(scorerNames);
    const remainingScorers = existingData.scheduledScorers.filter((scorer) => !deletedNamesSet.has(scorer.name));

    queryClient.setQueryData<ScheduledScorersResponse>(['mlflow', 'scheduled-scorers', experimentId], {
      experimentId: experimentId,
      scheduledScorers: remainingScorers,
    });
  } else {
    // If no scorer names provided (delete all) or cache is empty, invalidate to refetch
    queryClient.invalidateQueries(['mlflow', 'scheduled-scorers', experimentId]);
  }
}
