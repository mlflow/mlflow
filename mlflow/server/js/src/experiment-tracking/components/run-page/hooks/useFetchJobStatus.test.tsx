import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useFetchJobStatus, JobStatus, isJobComplete } from './useFetchJobStatus';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

describe('useFetchJobStatus', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  describe('isJobComplete', () => {
    test('returns true for SUCCEEDED status', () => {
      expect(isJobComplete(JobStatus.SUCCEEDED)).toBe(true);
    });

    test('returns true for FAILED status', () => {
      expect(isJobComplete(JobStatus.FAILED)).toBe(true);
    });

    test('returns true for TIMEOUT status', () => {
      expect(isJobComplete(JobStatus.TIMEOUT)).toBe(true);
    });

    test('returns true for CANCELED status', () => {
      expect(isJobComplete(JobStatus.CANCELED)).toBe(true);
    });

    test('returns false for PENDING status', () => {
      expect(isJobComplete(JobStatus.PENDING)).toBe(false);
    });

    test('returns false for RUNNING status', () => {
      expect(isJobComplete(JobStatus.RUNNING)).toBe(false);
    });

    test('returns false for undefined status', () => {
      expect(isJobComplete(undefined)).toBe(false);
    });
  });

  describe('useFetchJobStatus hook', () => {
    test('does not fetch when jobId is undefined', () => {
      const { result } = renderHook(() => useFetchJobStatus({ jobId: undefined }), { wrapper });

      expect(result.current.status).toBeUndefined();
      expect(result.current.result).toBeUndefined();
      expect(fetchAPI).not.toHaveBeenCalled();
    });

    test('fetches job status successfully with object result', async () => {
      const mockResponse = {
        status: JobStatus.SUCCEEDED,
        result: { issues: 5, summary: 'Test summary' },
      };
      jest.mocked(fetchAPI).mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useFetchJobStatus({ jobId: 'job-123' }), { wrapper });

      await waitFor(() => {
        expect(result.current.status).toBe(JobStatus.SUCCEEDED);
        expect(result.current.result).toEqual({ issues: 5, summary: 'Test summary' });
        expect(result.current.isLoading).toBe(false);
      });

      expect(fetchAPI).toHaveBeenCalledWith('ajax-api/3.0/mlflow/jobs/job-123');
    });

    test('fetches job status with string result (error message)', async () => {
      const mockResponse = {
        status: JobStatus.FAILED,
        result: 'Job failed due to error',
      };
      jest.mocked(fetchAPI).mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useFetchJobStatus({ jobId: 'job-456' }), { wrapper });

      await waitFor(() => {
        expect(result.current.status).toBe(JobStatus.FAILED);
        expect(result.current.result).toBe('Job failed due to error');
        expect(result.current.isLoading).toBe(false);
      });
    });

    test('fetches pending job status with null result', async () => {
      const mockResponse = {
        status: JobStatus.PENDING,
        result: null,
      };
      jest.mocked(fetchAPI).mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useFetchJobStatus({ jobId: 'job-pending' }), { wrapper });

      await waitFor(() => {
        expect(result.current.status).toBe(JobStatus.PENDING);
        expect(result.current.result).toBeNull();
        expect(result.current.isLoading).toBe(false);
      });
    });

    test('handles fetch error', async () => {
      const mockError = new Error('Network error');
      jest.mocked(fetchAPI).mockRejectedValue(mockError);

      const { result } = renderHook(() => useFetchJobStatus({ jobId: 'job-error' }), { wrapper });

      await waitFor(() => {
        expect(result.current.error).toEqual(mockError);
        expect(result.current.isLoading).toBe(false);
      });
    });

    test('does not fetch when enabled is false', () => {
      renderHook(() => useFetchJobStatus({ jobId: 'job-123', enabled: false }), { wrapper });

      expect(fetchAPI).not.toHaveBeenCalled();
    });

    test('does not fetch when jobId is empty string', () => {
      renderHook(() => useFetchJobStatus({ jobId: '' }), { wrapper });

      expect(fetchAPI).not.toHaveBeenCalled();
    });

    test('refetches when refetch is called', async () => {
      const mockResponse = {
        status: JobStatus.RUNNING,
        result: null,
      };
      jest.mocked(fetchAPI).mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useFetchJobStatus({ jobId: 'job-123' }), { wrapper });

      await waitFor(() => {
        expect(result.current.status).toBe(JobStatus.RUNNING);
      });

      expect(fetchAPI).toHaveBeenCalledTimes(1);

      const updatedResponse = {
        status: JobStatus.SUCCEEDED,
        result: { issues: 3 },
      };
      jest.mocked(fetchAPI).mockResolvedValue(updatedResponse);

      result.current.refetch();

      await waitFor(() => {
        expect(result.current.status).toBe(JobStatus.SUCCEEDED);
        expect(result.current.result).toEqual({ issues: 3 });
      });

      expect(fetchAPI).toHaveBeenCalledTimes(2);
    });
  });
});
