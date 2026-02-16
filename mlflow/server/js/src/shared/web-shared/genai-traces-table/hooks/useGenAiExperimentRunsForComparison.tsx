import type { QueryFunctionContext } from '@databricks/web-shared/query-client';
import { useQuery } from '@databricks/web-shared/query-client';

import { getAjaxUrl, makeRequest } from '../utils/FetchUtils';

type UseExperimentRunsForTraceComparisonQueryKey = ['EXPERIMENT_RUNS_FOR_TRACE_COMPARISON', { experimentId: string }];

const getQueryKey = (experimentId: string): UseExperimentRunsForTraceComparisonQueryKey => [
  'EXPERIMENT_RUNS_FOR_TRACE_COMPARISON',
  { experimentId },
];

type RawSearchRunsResponse = {
  runs: {
    info?: {
      run_uuid?: string;
      run_name?: string;
    };
  }[];
};

const queryFn = async ({
  queryKey: [, { experimentId }],
}: QueryFunctionContext<UseExperimentRunsForTraceComparisonQueryKey>): Promise<RawSearchRunsResponse> => {
  const url = getAjaxUrl('ajax-api/2.0/mlflow/runs/search');
  return makeRequest(url, 'POST', {
    experiment_ids: [experimentId],
  });
};

/**
 * Fetches the runs for the given experiment, used for the "compare to" dropdown in the eval page.
 */
export const useGenAiExperimentRunsForComparison = (experimentId: string, disabled = false) => {
  const { data, error, isLoading, isFetching } = useQuery<
    RawSearchRunsResponse,
    Error,
    RawSearchRunsResponse,
    UseExperimentRunsForTraceComparisonQueryKey
  >(getQueryKey(experimentId), {
    queryFn,
    enabled: !disabled,
    cacheTime: Infinity,
    staleTime: Infinity,
  });

  const runInfos = data?.runs?.map((run) => ({
    runUuid: run.info?.run_uuid,
    runName: run.info?.run_name,
  }));

  return {
    requestError: error,
    isLoading,
    isFetching,
    runInfos,
  };
};
