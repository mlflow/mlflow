import { renderHook, act } from '@testing-library/react';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useRetryIssueDetection } from './useRetryIssueDetection';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  createTraceLocationForExperiment: (experimentId: string) => ({
    type: 'MLFLOW_EXPERIMENT',
    mlflow_experiment: { experiment_id: experimentId },
  }),
}));

const mockInvokeIssueDetection = jest.fn();
jest.mock('../../experiment-page/components/traces-v3/hooks/useInvokeIssueDetection', () => ({
  useInvokeIssueDetection: () => ({ mutateAsync: mockInvokeIssueDetection }),
}));

const EXPERIMENT_ID = 'exp-1';
const RUN_UUID = 'run-abc';
const TRACE_IDS = ['trace-1', 'trace-2', 'trace-3'];
const CATEGORIES = ['correctness', 'latency'] as any;

function mockTracesResponse(traceIds: string[], nextPageToken?: string) {
  return {
    traces: traceIds.map((id) => ({ trace_id: id })),
    ...(nextPageToken ? { next_page_token: nextPageToken } : {}),
  };
}

describe('useRetryIssueDetection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockInvokeIssueDetection.mockResolvedValue({ job_id: 'job-new', run_id: 'run-new' });
  });

  describe('retryIssueDetection', () => {
    it('fetches trace IDs and invokes detection with endpoint_name', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce(mockTracesResponse(TRACE_IDS));

      const { result } = renderHook(() => useRetryIssueDetection());

      await act(async () => {
        await result.current.retryIssueDetection({
          experimentId: EXPERIMENT_ID,
          runUuid: RUN_UUID,
          endpointName: 'my-endpoint',
          categories: CATEGORIES,
        });
      });

      expect(fetchAPI).toHaveBeenCalledWith(
        'ajax-api/3.0/mlflow/traces/search',
        expect.objectContaining({
          method: 'POST',
          body: expect.objectContaining({ filter: `run_id = '${RUN_UUID}'` }),
        }),
      );

      expect(mockInvokeIssueDetection).toHaveBeenCalledWith({
        experimentId: EXPERIMENT_ID,
        traceIds: TRACE_IDS,
        categories: CATEGORIES,
        provider: '',
        model: '',
        endpoint_name: 'my-endpoint',
      });
    });

    it('splits provider and model from model tag when no endpoint_name', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce(mockTracesResponse(TRACE_IDS));

      const { result } = renderHook(() => useRetryIssueDetection());

      await act(async () => {
        await result.current.retryIssueDetection({
          experimentId: EXPERIMENT_ID,
          runUuid: RUN_UUID,
          model: 'openai:/gpt-4',
          categories: CATEGORIES,
        });
      });

      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ provider: 'openai', model: 'gpt-4', endpoint_name: undefined }),
      );
    });

    it('paginates until next_page_token is absent', async () => {
      jest
        .mocked(fetchAPI)
        .mockResolvedValueOnce(mockTracesResponse(['trace-1', 'trace-2'], 'token-page-2'))
        .mockResolvedValueOnce(mockTracesResponse(['trace-3']));

      const { result } = renderHook(() => useRetryIssueDetection());

      await act(async () => {
        await result.current.retryIssueDetection({
          experimentId: EXPERIMENT_ID,
          runUuid: RUN_UUID,
          endpointName: 'my-endpoint',
          categories: CATEGORIES,
        });
      });

      expect(fetchAPI).toHaveBeenCalledTimes(2);
      expect(fetchAPI).toHaveBeenNthCalledWith(
        2,
        'ajax-api/3.0/mlflow/traces/search',
        expect.objectContaining({ body: expect.objectContaining({ page_token: 'token-page-2' }) }),
      );
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ traceIds: ['trace-1', 'trace-2', 'trace-3'] }),
      );
    });

    it('throws when no traces are found', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce(mockTracesResponse([]));

      const { result } = renderHook(() => useRetryIssueDetection());

      await expect(
        act(async () => {
          await result.current.retryIssueDetection({
            experimentId: EXPERIMENT_ID,
            runUuid: RUN_UUID,
            endpointName: 'my-endpoint',
            categories: CATEGORIES,
          });
        }),
      ).rejects.toThrow('No traces found for this run');

      expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
    });

    it('throws when trace fetch fails and does not invoke detection', async () => {
      jest.mocked(fetchAPI).mockRejectedValueOnce(new Error('Network error'));

      const { result } = renderHook(() => useRetryIssueDetection());

      await expect(
        act(async () => {
          await result.current.retryIssueDetection({
            experimentId: EXPERIMENT_ID,
            runUuid: RUN_UUID,
            endpointName: 'my-endpoint',
            categories: CATEGORIES,
          });
        }),
      ).rejects.toThrow('Network error');

      expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
    });

    it('resets isRetrying to false after success', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce(mockTracesResponse(TRACE_IDS));

      const { result } = renderHook(() => useRetryIssueDetection());

      expect(result.current.isRetrying).toBe(false);

      await act(async () => {
        await result.current.retryIssueDetection({
          experimentId: EXPERIMENT_ID,
          runUuid: RUN_UUID,
          endpointName: 'my-endpoint',
          categories: CATEGORIES,
        });
      });

      expect(result.current.isRetrying).toBe(false);
    });

    it('resets isRetrying to false after failure', async () => {
      jest.mocked(fetchAPI).mockRejectedValueOnce(new Error('fail'));

      const { result } = renderHook(() => useRetryIssueDetection());

      await act(async () => {
        await result.current
          .retryIssueDetection({
            experimentId: EXPERIMENT_ID,
            runUuid: RUN_UUID,
            endpointName: 'my-endpoint',
            categories: CATEGORIES,
          })
          .catch(() => {});
      });

      expect(result.current.isRetrying).toBe(false);
    });
  });

  describe('resolveProviderParams (via invoke call)', () => {
    it('uses empty provider and model when endpoint_name is set', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce(mockTracesResponse(TRACE_IDS));

      const { result } = renderHook(() => useRetryIssueDetection());

      await act(async () => {
        await result.current.retryIssueDetection({
          experimentId: EXPERIMENT_ID,
          runUuid: RUN_UUID,
          endpointName: 'gateway-ep',
          categories: CATEGORIES,
        });
      });

      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ provider: '', model: '', endpoint_name: 'gateway-ep' }),
      );
    });

    it('handles model tag without :/ separator', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce(mockTracesResponse(TRACE_IDS));

      const { result } = renderHook(() => useRetryIssueDetection());

      await act(async () => {
        await result.current.retryIssueDetection({
          experimentId: EXPERIMENT_ID,
          runUuid: RUN_UUID,
          model: 'somemodel',
          categories: CATEGORIES,
        });
      });

      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ provider: 'somemodel', model: '' }),
      );
    });
  });
});
