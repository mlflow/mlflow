import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useMemo } from 'react';
import type { SearchOptimizationJobsResponse } from '../types';
import { isJobRunning } from '../types';
import { PromptOptimizationApi } from '../api';

type OptimizationJobsQueryKey = ['optimization_jobs', { experimentId: string }];

const queryFn = ({ queryKey }: QueryFunctionContext<OptimizationJobsQueryKey>) => {
  const [, { experimentId }] = queryKey;
  return PromptOptimizationApi.searchJobs(experimentId);
};

export const useOptimizationJobsQuery = ({ experimentId }: { experimentId: string }) => {
  const queryResult = useQuery<
    SearchOptimizationJobsResponse,
    Error,
    SearchOptimizationJobsResponse,
    OptimizationJobsQueryKey
  >(['optimization_jobs', { experimentId }], {
    queryFn,
    retry: false,
    staleTime: 30000, // Consider data fresh for 30 seconds
    cacheTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
    refetchOnWindowFocus: false, // Don't refetch when window regains focus
  });

  // Check if any jobs are running to enable polling
  const hasRunningJobs = useMemo(() => {
    return queryResult.data?.jobs?.some((job) => isJobRunning(job.state?.status)) ?? false;
  }, [queryResult.data?.jobs]);

  // Re-fetch with polling when there are running jobs
  useQuery<SearchOptimizationJobsResponse, Error, SearchOptimizationJobsResponse, OptimizationJobsQueryKey>(
    ['optimization_jobs', { experimentId }],
    {
      queryFn,
      retry: false,
      refetchInterval: hasRunningJobs ? 30000 : false, // Poll every 30s if jobs are running
      enabled: hasRunningJobs,
    },
  );

  return {
    data: queryResult.data?.jobs,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
    hasRunningJobs,
  };
};
