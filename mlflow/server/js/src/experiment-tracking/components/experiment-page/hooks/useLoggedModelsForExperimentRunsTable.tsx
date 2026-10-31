import { useEffect, useMemo } from 'react';
import { useSearchLoggedModelsQuery } from '../../../hooks/logged-models/useSearchLoggedModelsQuery';
import type { LoggedModelProto } from '../../../types';

export const useLoggedModelsForExperimentRunsTable = ({
  experimentIds,
  runIds,
  enabled = true,
}: {
  experimentIds: string[];
  runIds: string[];
  enabled?: boolean;
}) => {
  const isEnabled = enabled && runIds.length > 0;
  const searchQuery =
    runIds.length > 0
      ? `attributes.source_run_id IN (${[...runIds]
          .sort()
          .map((runId) => `'${runId}'`)
          .join(',')})`
      : undefined;
  const {
    data: loggedModelsData,
    nextPageToken,
    isFetching,
    error,
    loadMoreResults,
  } = useSearchLoggedModelsQuery(
    { experimentIds, searchQuery },
    {
      enabled: isEnabled,
    },
  );

  useEffect(() => {
    if (isEnabled && nextPageToken && !isFetching && !error) {
      void loadMoreResults();
    }
  }, [isEnabled, nextPageToken, isFetching, error, loadMoreResults]);

  const loggedModelsByRunId = useMemo(
    () =>
      loggedModelsData?.reduce<Record<string, LoggedModelProto[]>>((acc, model) => {
        const { source_run_id } = model.info ?? {};
        if (!source_run_id) {
          return acc;
        }
        if (!acc[source_run_id]) {
          acc[source_run_id] = [];
        }
        acc[source_run_id].push(model);
        return acc;
      }, {}),
    [loggedModelsData],
  );

  return loggedModelsByRunId;
};
