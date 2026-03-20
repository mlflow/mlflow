import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const FETCH_JOB_STATUS_QUERY_KEY = 'FETCH_JOB_STATUS';

export enum JobStatus {
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  SUCCEEDED = 'SUCCEEDED',
  FAILED = 'FAILED',
  TIMEOUT = 'TIMEOUT',
  CANCELED = 'CANCELED',
}

export interface FetchJobStatusResponse {
  status: JobStatus;
  result?: unknown;
  status_details?: {
    stage?: string;
  };
}

const POLLING_INTERVAL_MS = 3000;

export const isJobComplete = (status: JobStatus | undefined): boolean => {
  return (
    status === JobStatus.SUCCEEDED ||
    status === JobStatus.FAILED ||
    status === JobStatus.TIMEOUT ||
    status === JobStatus.CANCELED
  );
};

export interface UseFetchJobStatusResult {
  status: JobStatus | undefined;
  result: unknown;
  status_details?: {
    stage?: string;
  };
  isLoading: boolean;
  isFetching: boolean;
  refetch: () => void;
  error: Error | null;
}

export const useFetchJobStatus = ({
  jobId,
  enabled = true,
}: {
  jobId: string | undefined;
  enabled?: boolean;
}): UseFetchJobStatusResult => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<FetchJobStatusResponse, Error>({
    queryKey: [FETCH_JOB_STATUS_QUERY_KEY, jobId],
    queryFn: async () => {
      if (!jobId) {
        throw new Error('jobId is required');
      }
      return (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/jobs/${jobId}`))) as FetchJobStatusResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && !!jobId,
    refetchInterval: (data, query) => {
      if (isJobComplete(data?.status) || query.state.error) {
        return false;
      }
      return POLLING_INTERVAL_MS;
    },
  });

  return {
    status: data?.status,
    result: data?.result,
    status_details: data?.status_details,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
