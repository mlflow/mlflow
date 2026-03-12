import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const FETCH_ISSUE_JOB_STATUS_QUERY_KEY = 'FETCH_ISSUE_JOB_STATUS';

export enum IssueJobStatus {
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  SUCCEEDED = 'SUCCEEDED',
  FAILED = 'FAILED',
  TIMEOUT = 'TIMEOUT',
  CANCELED = 'CANCELED',
}

export interface IssueJobResult {
  issues?: number;
  summary?: string;
  total_traces_analyzed?: number;
  total_cost_usd?: number;
}

export interface FetchIssueJobStatusResponse {
  status: IssueJobStatus;
  result?: IssueJobResult;
}

const POLLING_INTERVAL_MS = 3000;

export const isJobComplete = (status: IssueJobStatus | undefined): boolean => {
  return (
    status === IssueJobStatus.SUCCEEDED ||
    status === IssueJobStatus.FAILED ||
    status === IssueJobStatus.TIMEOUT ||
    status === IssueJobStatus.CANCELED
  );
};

export interface UseFetchIssueJobStatusResult {
  status: IssueJobStatus | undefined;
  result: IssueJobResult | undefined;
  isLoading: boolean;
  isFetching: boolean;
  refetch: () => void;
  error: Error | null;
}

export const useFetchIssueJobStatus = ({
  jobId,
  enabled = true,
}: {
  jobId: string | undefined;
  enabled?: boolean;
}): UseFetchIssueJobStatusResult => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<FetchIssueJobStatusResponse, Error>({
    queryKey: [FETCH_ISSUE_JOB_STATUS_QUERY_KEY, jobId],
    queryFn: async () => {
      if (!jobId) {
        throw new Error('jobId is required');
      }
      return (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/issues/job/${jobId}`))) as FetchIssueJobStatusResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && !!jobId,
    refetchInterval: (data) => {
      if (isJobComplete(data?.status)) {
        return false;
      }
      return POLLING_INTERVAL_MS;
    },
  });

  return {
    status: data?.status,
    result: data?.result,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
