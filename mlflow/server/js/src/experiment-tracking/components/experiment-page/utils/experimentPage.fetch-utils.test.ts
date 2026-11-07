import { jest, describe, test, expect } from '@jest/globals';
import { EXPERIMENT_LOG_MODEL_HISTORY_TAG } from './experimentPage.common-utils';
import {
  fetchModelVersionsForRuns,
  shouldNestChildrenAndFetchParents,
  sortRunsForNesting,
} from './experimentPage.fetch-utils';
import type { SingleRunData } from './experimentPage.row-utils';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../fixtures/experiment-runs.fixtures';

jest.mock('../../../../model-registry/constants', () => ({
  MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER: 5,
}));

describe('experiment page fetch utils', () => {
  describe('sortRunsForNesting', () => {
    const runs: SingleRunData[] = [2, 10, undefined].map((value, index) => ({
      runInfo: {
        ...EXPERIMENT_RUNS_MOCK_STORE.entities.runInfosByUuid['experiment123456789_run1'],
        runUuid: String(index),
        startTime: value ?? 0,
      },
      metrics: value === undefined ? [] : [{ key: 'score.v1', value, timestamp: 0, step: 0 }],
      params: value === undefined ? [] : [{ key: 'size', value: String(value) }],
      tags: {},
      datasets: [],
    }));

    test.each([
      ['metrics.`score.v1`', true, ['0', '1', '2']],
      ['metrics.`score.v1`', false, ['1', '0', '2']],
      ['params.`size`', true, ['1', '0', '2']],
      ['params.`size`', false, ['0', '1', '2']],
      ['attributes.start_time', true, ['2', '0', '1']],
      ['attributes.start_time', false, ['1', '0', '2']],
    ])('sorts %s (ascending: %s)', (orderByKey, orderByAsc, expected) => {
      expect(sortRunsForNesting(runs, { orderByKey, orderByAsc }).map(({ runInfo }) => runInfo.runUuid)).toEqual(
        expected,
      );
      expect(runs.map(({ runInfo }) => runInfo.runUuid)).toEqual(['0', '1', '2']);
    });
  });

  describe('fetchModelVersionsForRuns', () => {
    // Let's generate 20 runs result
    const runsDataPayload = new Array(20).fill(0).map((_, index) => ({
      info: {
        run_id: `run_${index}`,
      },
      data: {
        // Then put log model history in 10 of them
        tags: index % 2 ? [{ key: EXPERIMENT_LOG_MODEL_HISTORY_TAG, value: 'abc' }] : [],
      },
    })) as any;

    test('it correctly constructs action for fetching model versions basing on search runs API result', () => {
      const actionCreatorMock = jest.fn();
      const dispatchMock = jest.fn();
      // @ts-expect-error Argument of type 'Mock<UnknownFunction>' is not assignable to parameter
      fetchModelVersionsForRuns(runsDataPayload, actionCreatorMock, dispatchMock);

      // We're chunking by 5 runs so we expect 2 calls for sum of 10 runs
      expect(actionCreatorMock).toHaveBeenCalledTimes(2);
      expect(actionCreatorMock.mock.calls[0][0]).toEqual({ run_id: ['run_1', 'run_3', 'run_5', 'run_7', 'run_9'] });
      expect(actionCreatorMock.mock.calls[1][0]).toEqual({
        run_id: ['run_11', 'run_13', 'run_15', 'run_17', 'run_19'],
      });
    });
  });

  describe('shouldNestChildrenAndFetchParents', () => {
    test.each([
      ['tags.`mlflow.user`', ''],
      ['tags.`mlflow.user`', 'metrics.accuracy > 0'],
      ['metrics.`accuracy`', ''],
      ['metrics.`accuracy`', 'metrics.accuracy > 0'],
      ['attributes.start_time', 'metrics.accuracy > 0'],
      ['', ''],
    ])('returns true for orderByKey "%s" and searchFilter "%s"', (orderByKey, searchFilter) => {
      expect(shouldNestChildrenAndFetchParents({ orderByKey, searchFilter })).toBe(true);
    });

    test('returns false for empty orderByKey with a search filter', () => {
      expect(shouldNestChildrenAndFetchParents({ orderByKey: '', searchFilter: 'metrics.accuracy > 0' })).toBe(false);
    });
  });
});
