import { renderHook, cleanup, waitFor } from '@testing-library/react';
import { MlflowService } from '../../../sdk/MlflowService';
import { ModelSpanType, ModelTraceStatus, type ModelTraceData } from '@databricks/web-shared/model-trace-explorer';
import { useExperimentTraceData } from './useExperimentTraceData';
import Utils from '../../../../common/utils/Utils';

const testRequestId = 'tr-trace-1';

const mockTraceData: ModelTraceData = {
  spans: [
    {
      name: 'test-span',
      start_time: 0,
      end_time: 1,
      status: {
        status_code: 1,
        description: 'OK',
      },
      context: {
        span_id: 'span-1',
        trace_id: testRequestId,
      },
      parent_span_id: null,
      span_type: 'test-span-type',
      type: ModelSpanType.FUNCTION,
    },
  ],
};

const mockMangledTraceData: any = {
  spans: { something: ['very', 'wrong'] },
};

describe('useExperimentTraceData', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    cleanup();
  });

  const renderTestHook = (skip?: boolean) => renderHook(() => useExperimentTraceData(testRequestId, skip));
  test('fetches trace data', async () => {
    jest.spyOn(MlflowService, 'getExperimentTraceData').mockImplementation(() => Promise.resolve(mockTraceData));

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook();
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    // Check that the traces are fetched correctly.
    expect(result.current.traceData).toBeDefined();
    expect(result.current.traceData).toEqual(mockTraceData);
  });

  test('returns error when necessary', async () => {
    jest
      .spyOn(MlflowService, 'getExperimentTraceData')
      .mockImplementation(() => Promise.reject(new Error('Some error')));

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook();
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toEqual('Some error');
  });

  test('shows error when data is badly formatted', async () => {
    jest.spyOn(MlflowService, 'getExperimentTraceData').mockImplementation(() => Promise.resolve(mockMangledTraceData));
    jest.spyOn(console, 'error').mockImplementation(() => {});
    jest.spyOn(Utils, 'logErrorAndNotifyUser');

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook();
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    expect(Utils.logErrorAndNotifyUser).toHaveBeenCalled();
  });

  test("doesn't dispatch a network request if skip argument is provided", async () => {
    jest.spyOn(MlflowService, 'getExperimentTraceData');
    renderTestHook(true);
    expect(MlflowService.getExperimentTraceData).not.toHaveBeenCalled();
  });
});
