import { useMemo } from 'react';
import type { JobProgressMetadata } from '../types';
import { fetchAPI, getAjaxUrl } from '../utils/FetchUtils';
import type { QueryFunctionContext, UseQueryOptions } from '../utils/reactQueryHooks';
import { useQuery } from '../utils/reactQueryHooks';

const GET_JOB_DATA_QUERY_KEY = 'GET_TRACKING_SERVER_JOB_STATUS';

export enum TrackingJobStatus {
  RUNNING = 'RUNNING',
  PENDING = 'PENDING',
  SUCCEEDED = 'SUCCEEDED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED',
  TIMEOUT = 'TIMEOUT',
  NEEDS_RECOVERY = 'NEEDS_RECOVERY',
}

export const isTrackingJobInFlight = (status: TrackingJobStatus | undefined): boolean =>
  status === TrackingJobStatus.PENDING ||
  status === TrackingJobStatus.RUNNING ||
  status === TrackingJobStatus.NEEDS_RECOVERY;

export const isTrackingJobTerminal = (status: TrackingJobStatus | undefined): boolean =>
  status === TrackingJobStatus.SUCCEEDED ||
  status === TrackingJobStatus.FAILED ||
  status === TrackingJobStatus.TIMEOUT ||
  status === TrackingJobStatus.CANCELLED;

export type TrackingJobQueryResult<ResultType> = (
  | {
      status:
        | TrackingJobStatus.RUNNING
        | TrackingJobStatus.PENDING
        | TrackingJobStatus.NEEDS_RECOVERY
        | TrackingJobStatus.CANCELLED
        | TrackingJobStatus.TIMEOUT;
    }
  | {
      status: TrackingJobStatus.SUCCEEDED;
      // Actual result is present in the payload if the job succeeded
      result: ResultType;
    }
  | {
      status: TrackingJobStatus.FAILED;
      // In case of failure, the result is the error message
      result: string;
    }
) &
  JobProgressMetadata & {
    jobId: string;
  };

type QueryKey = [typeof GET_JOB_DATA_QUERY_KEY, string[] | undefined];

const queryFn = async ({ queryKey: [, jobIds] }: QueryFunctionContext<QueryKey>) => {
  const responsesData = await Promise.all(
    (jobIds ?? []).map(async (jobId) => {
      const responseData = await fetchAPI(getAjaxUrl(`ajax-api/3.0/jobs/${jobId}`));
      const { status, result, error_message, status_message, progress_payload, progress_updated_at } = responseData;
      return { jobId, status, result, error_message, status_message, progress_payload, progress_updated_at };
    }),
  );
  return responsesData;
};

/**
 * Gets the current status of a tracking server job.
 */
export const useGetTrackingServerJobStatus = <T = any,>(
  jobIds?: string[],
  options?: UseQueryOptions<TrackingJobQueryResult<T>[], Error, TrackingJobQueryResult<T>[], QueryKey>,
) => {
  const isEnabled = options?.enabled ?? true;
  const queryResult = useQuery<TrackingJobQueryResult<T>[], Error, TrackingJobQueryResult<T>[], QueryKey>({
    queryKey: [GET_JOB_DATA_QUERY_KEY, jobIds],
    queryFn,
    ...options,
  });

  // Determine if any of the jobs are still running
  const areJobsRunning =
    isEnabled &&
    (queryResult.isLoading || queryResult.data?.some((response) => isTrackingJobInFlight(response.status)));

  const jobResults = useMemo(
    () =>
      queryResult.data?.reduce(
        (acc, response) => {
          acc[response.jobId] = response;
          return acc;
        },
        {} as Record<string, TrackingJobQueryResult<T>>,
      ),
    [queryResult.data],
  );

  return { jobResults, areJobsRunning };
};
