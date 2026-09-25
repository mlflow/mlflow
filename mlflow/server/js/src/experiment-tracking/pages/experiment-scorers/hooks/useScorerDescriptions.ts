import { useMemo } from 'react';
import { useQuery } from '@databricks/web-shared/query-client';
import { listScheduledScorers } from '../api';

export function useScorerDescriptions(experimentId?: string): Record<string, string> {
  const { data } = useQuery({
    queryKey: ['mlflow', 'scorer-descriptions', experimentId],
    queryFn: () => {
      if (!experimentId) {
        throw new Error('Experiment ID is required');
      }
      return listScheduledScorers(experimentId);
    },
    staleTime: 5 * 60 * 1000,
    enabled: Boolean(experimentId),
  });

  return useMemo(() => {
    const descriptions: Record<string, string> = {};
    for (const scorer of data?.scorers ?? []) {
      try {
        const serializedScorer: unknown = JSON.parse(scorer.serialized_scorer);
        if (
          typeof serializedScorer === 'object' &&
          serializedScorer !== null &&
          'description' in serializedScorer &&
          typeof serializedScorer.description === 'string'
        ) {
          descriptions[scorer.scorer_name] = serializedScorer.description;
        }
      } catch {
        // A malformed scorer should not hide descriptions from the remaining registry entries.
      }
    }
    return descriptions;
  }, [data]);
}
