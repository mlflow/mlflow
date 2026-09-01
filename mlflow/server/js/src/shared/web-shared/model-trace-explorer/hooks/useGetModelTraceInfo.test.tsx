import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, jest } from '@jest/globals';

import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';
import type { ModelTrace, ModelTraceInfoV3 } from '../ModelTrace.types';
import { fetchTraceInfoV3, getExperimentTraceV3, TracesServiceV3 } from '../api';
import { useGetModelTraceInfo } from './useGetModelTraceInfo';

jest.mock('../FeatureUtils', () => ({
  shouldUseTracesV4API: () => false,
}));

jest.mock('../api', () => {
  const actual = jest.requireActual<typeof import('../api')>('../api');
  return {
    ...actual,
    fetchTraceInfoV3: jest.fn(),
    getExperimentTraceV3: jest.fn(),
    TracesServiceV3: {
      ...actual.TracesServiceV3,
      getTraceV3: jest.fn(),
    },
  };
});

const traceInfo: ModelTraceInfoV3 = {
  trace_id: 'tr-test',
  trace_location: {
    type: 'MLFLOW_EXPERIMENT',
    mlflow_experiment: { experiment_id: '1' },
  },
  request_time: '2026-09-01T00:00:00Z',
  state: 'OK',
  tags: { 'mlflow.trace.spansLocation': 'TRACKING_STORE' },
};

describe('useGetModelTraceInfo', () => {
  const mockFetchTraceInfoV3 = jest.mocked(fetchTraceInfoV3);
  const mockGetExperimentTraceV3 = jest.mocked(getExperimentTraceV3);
  const mockGetTraceV3 = jest.mocked(TracesServiceV3.getTraceV3);
  const setModelTrace = jest.fn();
  const setAssessmentsPaneEnabled = jest.fn();
  let queryClient: QueryClient;

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient = new QueryClient();
    mockFetchTraceInfoV3.mockResolvedValue({ trace: { trace_info: traceInfo } });
    mockGetExperimentTraceV3.mockResolvedValue({
      trace: { trace_info: traceInfo, spans: [{ span_id: 'span-1' }] as any },
    });
  });

  it('refreshes TRACKING_STORE traces without requesting an artifact', async () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
    const { result } = renderHook(
      () => useGetModelTraceInfo({ traceId: 'tr-test', setModelTrace, setAssessmentsPaneEnabled }),
      { wrapper },
    );

    await waitFor(() => expect(mockFetchTraceInfoV3).toHaveBeenCalled());
    await waitFor(() => expect(result.current.refreshTrace).toBeDefined());
    await act(async () => result.current.refreshTrace?.());

    expect(mockGetExperimentTraceV3).toHaveBeenCalledWith({ traceId: 'tr-test' });
    expect(mockGetTraceV3).not.toHaveBeenCalled();

    const updateModelTrace = setModelTrace.mock.calls.at(-1)?.[0] as (trace: ModelTrace) => ModelTrace;
    expect(
      updateModelTrace({
        info: traceInfo,
        data: { spans: [] },
        _paginatedResult: { totalSpanCount: 501, isVirtualized: true },
      }),
    ).toEqual({
      info: traceInfo,
      data: { spans: [{ span_id: 'span-1' }] },
      _paginatedResult: { totalSpanCount: 501, isVirtualized: true },
    });
  });
});
