import type { LoggedModelProto, RunEntity } from '../../types';
import { useEffect, useMemo } from 'react';
import { compact, sortBy, uniq } from 'lodash';
import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQueries } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../sdk/MlflowService';
import { useArrayMemo } from '../../../common/hooks/useArrayMemo';

type UseRegisteredModelRelatedRunNamesQueryKey = ['USE_RELATED_RUNS_DATA_FOR_LOGGED_MODELS', { runUuid: string }];

const getQueryKey = (runUuid: string): UseRegisteredModelRelatedRunNamesQueryKey => [
  'USE_RELATED_RUNS_DATA_FOR_LOGGED_MODELS',
  { runUuid },
];

const queryFn = async ({
  queryKey: [, { runUuid }],
}: QueryFunctionContext<UseRegisteredModelRelatedRunNamesQueryKey>): Promise<RunEntity | null> => {
  try {
    const data = await MlflowService.getRun({ run_id: runUuid });
    return data?.run;
  } catch (e) {
    return null;
  }
};

/**
 * Hook used to fetch necessary run data based on metadata found in logged models
 */
export const useRelatedRunsDataForLoggedModels = ({ loggedModels = [] }: { loggedModels?: LoggedModelProto[] }) => {
  const runUuids = useMemo(() => {
    // Extract all run ids found in metrics and source run ids
    const allMetricRunUuids = compact(
      loggedModels?.flatMap((loggedModel) => loggedModel?.data?.metrics?.map((metric) => metric.run_id)),
    );
    const allSourceRunUuids = compact(loggedModels?.map((loggedModel) => loggedModel?.info?.source_run_id));
    const distinctRunUuids = sortBy(uniq([...allMetricRunUuids, ...allSourceRunUuids]));

    return distinctRunUuids;
  }, [loggedModels]);

  const queryResults = useQueries({
    queries: runUuids.map((runUuid) => ({
      queryKey: getQueryKey(runUuid),
      queryFn,
      cacheTime: Infinity,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      retry: false,
    })),
  });

  const loading = queryResults.some(({ isLoading }) => isLoading);
  const error = queryResults.find(({ error }) => error)?.error as Error | undefined;

  const memoizedQueryResults = useArrayMemo(queryResults.map(({ data }) => data));

  const data = useMemo(
    () => memoizedQueryResults.map((data) => data).filter(Boolean) as RunEntity[],
    [memoizedQueryResults],
  );

  return {
    data,
    loading,
    error,
  };
};
