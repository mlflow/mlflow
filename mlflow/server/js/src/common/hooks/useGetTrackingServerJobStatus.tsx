import { fetchAPI, getAjaxUrl } from '../utils/FetchUtils';
import { useQuery, UseQueryOptions } from '../utils/reactQueryHooks';

const GET_JOB_DATA_QUERY_KEY = 'GET_TRACKING_SERVER_JOB_STATUS';

export type TrackingJobQueryResult<ResultType> = (
  | {
      status: 'RUNNING' | 'PENDING';
    }
  | {
      status: 'SUCCEEDED';
      result: ResultType;
    }
  | {
      status: 'FAILED';
      result: string;
    }
) & {
  jobId: string;
};

/**
 * Gets the current status of a tracking server job.
 */
export const useGetTrackingServerJobStatus = <T = any,>(
  jobIds?: string[],
  options?: UseQueryOptions<TrackingJobQueryResult<T>[]>,
) => {
  const isEnabled = options?.enabled ?? true;
  const queryResult = useQuery<TrackingJobQueryResult<T>[], Error, TrackingJobQueryResult<T>[], any>({
    queryKey: [GET_JOB_DATA_QUERY_KEY, jobIds],
    queryFn: async () => {
      const responsesData = await Promise.all(
        (jobIds ?? []).map(async (jobId) => {
          const responseData = await fetchAPI(getAjaxUrl(`ajax-api/3.0/jobs/${jobId}`));
          const { status, result } = responseData;
          return { jobId, status, result };
        }),
      );
      return responsesData;
    },
    ...options,
  });

  const areJobsRunning =
    isEnabled &&
    (queryResult.isLoading ||
      queryResult.data?.some((response) => response.status === 'PENDING' || response.status === 'RUNNING'));

  const jobResults = queryResult.data?.reduce((acc, response) => {
    acc[response.jobId] = response;
    return acc;
  }, {} as Record<string, TrackingJobQueryResult<T>>);

  return { jobResults, areJobsRunning };
};
