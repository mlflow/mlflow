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
  NEEDS_RECOVERY = 'NEEDS_RECOVERY',
}

export interface FetchJobStatusResponse {
  status: JobStatus;
  result?: unknown;
  error_message?: string | null;
  status_message?: string | null;
  progress_payload?: {
    phase?: string;
    completed?: number;
    total?: number;
    unit?: string;
  } | null;
  progress_updated_at?: number | null;
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
  error_message?: string | null;
  status_message?: string | null;
  progress_payload?: {
    phase?: string;
    completed?: number;
    total?: number;
    unit?: string;
  } | null;
  progress_updated_at?: number | null;
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
    error_message: data?.error_message,
    status_message: data?.status_message,
    progress_payload: data?.progress_payload,
    progress_updated_at: data?.progress_updated_at,
    status_details: data?.status_details,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
