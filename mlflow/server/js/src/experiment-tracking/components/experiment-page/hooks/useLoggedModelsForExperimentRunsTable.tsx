import { useEffect, useMemo } from 'react';
import { useSearchLoggedModelsQuery } from '../../../hooks/logged-models/useSearchLoggedModelsQuery';
import type { LoggedModelProto } from '../../../types';

/**
 * Upper bound on the number of pages fetched automatically. `SearchLoggedModels` defaults to 50
 * models per page, so this covers experiments with up to ~1000 logged models while keeping the
 * number of requests bounded for very large experiments.
 */
const MAX_AUTO_FETCHED_PAGES = 20;

export const useLoggedModelsForExperimentRunsTable = ({
  experimentIds,
  enabled = true,
}: {
  experimentIds: string[];
  enabled?: boolean;
}) => {
  const {
    data: loggedModelsData,
    nextPageToken,
    pageCount,
    isFetching,
    loadMoreResults,
  } = useSearchLoggedModelsQuery(
    { experimentIds },
    {
      enabled,
    },
  );

  // This hook feeds the "Models" column for every run in the table, so a single page of results
  // is not enough: the models belonging to a run can sit on any page. Keep draining pages until
  // the backend stops returning a page token, otherwise runs whose models fall outside the first
  // page render an empty cell.
  const shouldFetchNextPage = enabled && Boolean(nextPageToken) && !isFetching && pageCount < MAX_AUTO_FETCHED_PAGES;

  useEffect(() => {
    if (shouldFetchNextPage) {
      loadMoreResults();
    }
  }, [shouldFetchNextPage, loadMoreResults]);

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
