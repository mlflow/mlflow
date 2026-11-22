import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { PropsWithChildren } from 'react';
import { MlflowService } from '../../../sdk/MlflowService';
import type { RunEntity } from '../../../types';
import {
  EVAL_RUNS_BASELINE_TAG,
  getEvalRunsBaselineRunUuid,
  isMatchingEvalRunsBaseline,
  useGetEvalRunsBaseline,
  useSetEvalRunsBaseline,
} from './useEvalRunsBaseline';

const wrapper = ({ children }: PropsWithChildren<unknown>) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

describe('evaluation runs baseline storage', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
  });

  test.each([
    { value: 'run-1', expected: 'run-1' },
    { value: '123456789', expected: '123456789' },
    { value: JSON.stringify({ runUuid: 'legacy-run' }), expected: 'legacy-run' },
    { value: JSON.stringify({ unexpected: 'value' }), expected: undefined },
  ])('reads $expected from the baseline tag value', ({ value, expected }) => {
    expect(getEvalRunsBaselineRunUuid([{ key: EVAL_RUNS_BASELINE_TAG, value }])).toBe(expected);
  });

  test('writes the selected run UUID with the existing experiment-tag API', async () => {
    const onSuccess = jest.fn();
    const setExperimentTag = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
    const { result } = renderHook(() => useSetEvalRunsBaseline({ experimentId: 'experiment-1', onSuccess }), {
      wrapper,
    });

    act(() => result.current.setBaseline('run-2'));

    await waitFor(() => {
      expect(setExperimentTag).toHaveBeenCalledWith({
        experiment_id: 'experiment-1',
        key: EVAL_RUNS_BASELINE_TAG,
        value: 'run-2',
      });
      expect(onSuccess).toHaveBeenCalledTimes(1);
    });
  });

  test('fetches the baseline independently from the filtered run results', async () => {
    const baseline: RunEntity = {
      data: { params: [], tags: [], metrics: [] },
      info: {
        artifactUri: '',
        endTime: 0,
        experimentId: 'experiment-1',
        lifecycleStage: 'active',
        runUuid: 'baseline',
        runName: 'Baseline',
        startTime: 0,
        status: 'FINISHED',
      },
    };
    const getRun = jest.spyOn(MlflowService, 'getRun').mockResolvedValue({ run: baseline });
    const { result } = renderHook(() => useGetEvalRunsBaseline('baseline'), { wrapper });

    await waitFor(() => expect(result.current.data).toBe(baseline));
    expect(getRun).toHaveBeenCalledWith({ run_id: 'baseline' });
  });

  test('fetches a missing baseline once instead of retrying it', async () => {
    const getRun = jest.spyOn(MlflowService, 'getRun').mockRejectedValue(new Error('Run not found'));
    const { result } = renderHook(() => useGetEvalRunsBaseline('deleted-run'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(getRun).toHaveBeenCalledTimes(1);
  });

  test('clears the baseline with the existing experiment-tag API', async () => {
    const onSuccess = jest.fn();
    const deleteExperimentTag = jest.spyOn(MlflowService, 'deleteExperimentTag').mockResolvedValue(undefined);
    const { result } = renderHook(() => useSetEvalRunsBaseline({ experimentId: 'experiment-1', onSuccess }), {
      wrapper,
    });

    act(() => result.current.clearBaseline());

    await waitFor(() => {
      expect(deleteExperimentTag).toHaveBeenCalledWith({
        experiment_id: 'experiment-1',
        key: EVAL_RUNS_BASELINE_TAG,
      });
      expect(onSuccess).toHaveBeenCalledTimes(1);
    });
  });

  test('rejects a fetched baseline from another experiment', () => {
    const run = { info: { runUuid: 'run-1', experimentId: 'experiment-2', lifecycleStage: 'active' } };

    expect(isMatchingEvalRunsBaseline(run, 'run-1', 'experiment-1')).toBe(false);
  });

  test('rejects a deleted fetched baseline', () => {
    const run = { info: { runUuid: 'run-1', experimentId: 'experiment-1', lifecycleStage: 'deleted' } };

    expect(isMatchingEvalRunsBaseline(run, 'run-1', 'experiment-1')).toBe(false);
  });

  test.each([undefined, null])('rejects a fetched baseline with lifecycle stage %s', (lifecycleStage) => {
    const run = { info: { runUuid: 'run-1', experimentId: 'experiment-1', lifecycleStage } };

    expect(isMatchingEvalRunsBaseline(run, 'run-1', 'experiment-1')).toBe(false);
  });

  test('accepts an active fetched baseline from the current experiment', () => {
    const run = { info: { runUuid: 'run-1', experimentId: 'experiment-1', lifecycleStage: 'active' } };

    expect(isMatchingEvalRunsBaseline(run, 'run-1', 'experiment-1')).toBe(true);
  });
});
