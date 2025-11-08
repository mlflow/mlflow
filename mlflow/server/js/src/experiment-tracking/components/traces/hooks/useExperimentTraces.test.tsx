import { renderHook, cleanup, waitFor, act } from '@testing-library/react';
import { useExperimentTraces } from './useExperimentTraces';
import { MlflowService } from '../../../sdk/MlflowService';
import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { first, last } from 'lodash';
import type { KeyValueEntity } from '../../../../common/types';

const testExperimentId = 'some-experiment-id';
const testExperimentIds = [testExperimentId];

const generateMockTrace = (uniqueId: string, timestampMs = 100, metadata: KeyValueEntity[] = []): ModelTraceInfo => ({
  request_id: `tr-${uniqueId}`,
  experiment_id: testExperimentId,
  timestamp_ms: 1712134300000 + timestampMs,
  execution_time_ms: timestampMs,
  status: 'OK',
  attributes: {},
  request_metadata: [...metadata],
  tags: [],
});

const pagesCount = 3;

describe('useExperimentTraces', () => {
  afterEach(() => {
    cleanup();
  });

  const renderTestHook = (
    filter = '',
    sorting: {
      id: string;
      desc: boolean;
    }[] = [],
    runUuid?: string,
  ) =>
    renderHook(() =>
      useExperimentTraces({
        experimentIds: testExperimentIds,
        filter,
        sorting,
        runUuid,
      }),
    );
  test('fetches traces and navigates through pages', async () => {
    // Mocking the getExperimentTraces function to return 100 traces per page.
    // We will use simple {"page": 1} token to simulate pagination.
    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation((_, __, token) => {
      const page = token ? JSON.parse(token).page : 1;
      const traces = new Array(100).fill(0).map((_, i) => generateMockTrace(`trace-page${page}-${i + 1}`, i));
      const next_page_token = page < pagesCount ? JSON.stringify({ page: page + 1 }) : undefined;
      return Promise.resolve({ traces, next_page_token });
    });

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook();
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    // Check that the traces are fetched correctly.
    expect(first(result.current.traces)?.request_id).toEqual('tr-trace-page1-1');
    expect(last(result.current.traces)?.request_id).toEqual('tr-trace-page1-100');
    expect(result.current.error).toEqual(undefined);

    // Check that the pagination works correctly.
    expect(result.current.hasPreviousPage).toEqual(false);
    expect(result.current.hasNextPage).toEqual(true);

    // Fetch the next page and check that the traces are updated.
    await act(async () => {
      result.current.fetchNextPage();
    });

    expect(first(result.current.traces)?.request_id).toEqual('tr-trace-page2-1');
    expect(last(result.current.traces)?.request_id).toEqual('tr-trace-page2-100');

    // Fetch the previous page and check that the traces are updated.
    await act(async () => {
      result.current.fetchPrevPage();
    });

    expect(first(result.current.traces)?.request_id).toEqual('tr-trace-page1-1');
    expect(last(result.current.traces)?.request_id).toEqual('tr-trace-page1-100');

    // Move to the last page and check that the pagination is correct.
    await act(async () => {
      result.current.fetchNextPage();
    });
    await act(async () => {
      result.current.fetchNextPage();
    });

    expect(first(result.current.traces)?.request_id).toEqual('tr-trace-page3-1');
    expect(last(result.current.traces)?.request_id).toEqual('tr-trace-page3-100');

    expect(result.current.hasPreviousPage).toEqual(true);
    expect(result.current.hasNextPage).toEqual(false);
  });

  test('navigates through pages maintaining the same sort by value', async () => {
    // Mocking the getExperimentTraces function to return 100 traces per page.
    // We will use simple {"page": 1} token to simulate pagination.
    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation((_, __, token) => {
      const page = token ? JSON.parse(token).page : 1;
      const traces = new Array(100).fill(0).map((_, i) => generateMockTrace(`trace-page${page}-${i + 1}`, i));
      const next_page_token = page < pagesCount ? JSON.stringify({ page: page + 1 }) : undefined;
      return Promise.resolve({ traces, next_page_token });
    });

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook('', [{ id: 'timestamp_ms', desc: false }]);
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    expect(MlflowService.getExperimentTraces).toHaveBeenLastCalledWith(
      ['some-experiment-id'],
      'timestamp_ms ASC',
      undefined,
      '',
    );

    // Fetch the next page and check that the traces are updated.
    await act(async () => {
      result.current.fetchNextPage();
    });

    expect(MlflowService.getExperimentTraces).toHaveBeenLastCalledWith(
      ['some-experiment-id'],
      'timestamp_ms ASC',
      '{"page":2}',
      '',
    );
  });

  test('returns error when necessary', async () => {
    // Mocking the getExperimentTraces function to return 100 traces per page.
    // On the second page, we will return an error.
    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation((_, __, token) => {
      if (token) {
        return Promise.reject(new Error('Some error'));
      }
      const traces = new Array(100).fill(0).map((_, i) => generateMockTrace(`trace-${i + 1}`, i));
      return Promise.resolve({ traces, next_page_token: 'some-token' });
    });

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook();

    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    // Check that the traces are fetched correctly.
    expect(first(result.current.traces)?.request_id).toEqual('tr-trace-1');
    expect(last(result.current.traces)?.request_id).toEqual('tr-trace-100');

    // Check that the pagination works correctly.
    expect(result.current.error).toEqual(undefined);
    expect(result.current.hasPreviousPage).toEqual(false);
    expect(result.current.hasNextPage).toEqual(true);

    // Fetch the next page and check that the error is returned.
    await act(async () => {
      result.current.fetchNextPage();
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toEqual('Some error');
  });

  test('requests for run names', async () => {
    jest.spyOn(MlflowService, 'searchRuns').mockImplementation(
      async () =>
        ({
          runs: [
            {
              info: {
                runUuid: 'run-1',
                runName: 'A run number one',
              },
            },
          ],
        } as any),
    );

    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation((_, __, token) => {
      const traces = [
        // First two traces reference the same run, third trace references another one, last one does not have any run
        generateMockTrace('trace-1', 0, [{ key: 'mlflow.sourceRun', value: 'run-1' }]),
        generateMockTrace('trace-2', 0, [{ key: 'mlflow.sourceRun', value: 'run-1' }]),
        generateMockTrace('trace-3', 0, [{ key: 'mlflow.sourceRun', value: 'run-2' }]),
        generateMockTrace('trace-4', 0),
      ];
      return Promise.resolve({ traces });
    });

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook();

    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    const getTraceById = (id: string) => result.current.traces.find((trace) => trace.request_id === id);

    // First two traces reference to the actual run
    expect(getTraceById('tr-trace-1')?.runName).toEqual('A run number one');
    expect(getTraceById('tr-trace-2')?.runName).toEqual('A run number one');

    // Third trace references to the nonexistent run so the run ID is used instead of the name
    expect(getTraceById('tr-trace-3')?.runName).toEqual('run-2');

    // Last trace does not reference to any run
    expect(getTraceById('tr-trace-4')?.runName).toEqual(undefined);
  });

  test('sends proper filter query', async () => {
    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation(() => Promise.resolve({ traces: [] }));

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook('tags.test_tag="xyz"');
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    // Check that the trace API is called with the correct filter query.
    expect(MlflowService.getExperimentTraces).toHaveBeenCalledWith(
      testExperimentIds,
      expect.anything(),
      undefined,
      'tags.test_tag="xyz"',
    );
  });

  test('does correct run ID filtering', async () => {
    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation(() => Promise.resolve({ traces: [] }));

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook(undefined, undefined, 'test-run-id');
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    // Check that the trace API is called with the correct filter query.
    expect(MlflowService.getExperimentTraces).toHaveBeenLastCalledWith(
      testExperimentIds,
      expect.anything(),
      undefined,
      "request_metadata.`mlflow.sourceRun`='test-run-id'",
    );
  });

  test('sends proper filter query with run ID filtering', async () => {
    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation(() => Promise.resolve({ traces: [] }));

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook('tags.test_tag="xyz"', undefined, 'test-run-id');
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    // Check that the trace API is called with the correct filter query.
    expect(MlflowService.getExperimentTraces).toHaveBeenLastCalledWith(
      testExperimentIds,
      expect.anything(),
      undefined,
      'tags.test_tag="xyz" AND request_metadata.`mlflow.sourceRun`=\'test-run-id\'',
    );
  });

  test('sends proper sorting query', async () => {
    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation(() => Promise.resolve({ traces: [] }));

    // Render the hook and wait for the traces to be fetched.
    const { result } = renderTestHook('tags.test_tag="xyz"', [{ id: 'timestamp_ms', desc: false }]);
    await waitFor(() => {
      expect(result.current.loading).toEqual(false);
    });

    // Check that the trace API is called with the correct filter query.
    expect(MlflowService.getExperimentTraces).toHaveBeenCalledWith(
      testExperimentIds,
      'timestamp_ms ASC',
      undefined,
      'tags.test_tag="xyz"',
    );
  });
});
