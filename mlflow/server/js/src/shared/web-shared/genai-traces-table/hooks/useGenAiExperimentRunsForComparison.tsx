import type { QueryFunctionContext } from '../../query-client/queryClient';
import { useQuery } from '../../query-client/queryClient';

import { getAjaxUrl, makeRequest } from '../utils/FetchUtils';
import {
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
} from '@mlflow/mlflow/src/experiment-tracking/constants';

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
      start_time?: number | string | null;
    };
    data?: {
      tags?: Array<{ key: string; value: string }>;
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

// Same exclusion as the evaluation runs list; see `isIssueDetectionRun` in useExperimentEvaluationRunsData.
const isIssueDetectionRun = (run: RawSearchRunsResponse['runs'][0]): boolean =>
  Boolean(
    run.data?.tags?.some(
      ({ key, value }) => key === MLFLOW_RUN_TYPE_TAG && value === MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
    ),
  );

/**
 * Fetches the runs for the given experiment, used for the "compare to" dropdown in the eval page.
 */
export const useGenAiExperimentRunsForComparison = (
  experimentId: string | undefined,
  disabled = false,
): {
  requestError: Error | null;
  isLoading: boolean;
  isFetching: boolean;
  runInfos:
    | {
        runUuid: string | undefined;
        runName: string | undefined;
        startTime?: number;
      }[]
    | undefined;
} => {
  const { data, error, isLoading, isFetching } = useQuery<
    RawSearchRunsResponse,
    Error,
    RawSearchRunsResponse,
    UseExperimentRunsForTraceComparisonQueryKey
  >(getQueryKey(experimentId ?? ''), {
    queryFn,
    enabled: !disabled && Boolean(experimentId),
    cacheTime: Infinity,
    staleTime: Infinity,
  });

  const runInfos = data?.runs
    ?.filter((run) => !isIssueDetectionRun(run))
    .map((run) => {
      const startTime = run.info?.start_time;
      const parsedStartTime =
        startTime === undefined || startTime === null || (typeof startTime === 'string' && startTime.trim() === '')
          ? undefined
          : Number(startTime);
      return {
        runUuid: run.info?.run_uuid,
        runName: run.info?.run_name,
        startTime: parsedStartTime !== undefined && Number.isFinite(parsedStartTime) ? parsedStartTime : undefined,
      };
    });

  return {
    requestError: error,
    isLoading,
    isFetching,
    runInfos,
  };
};
