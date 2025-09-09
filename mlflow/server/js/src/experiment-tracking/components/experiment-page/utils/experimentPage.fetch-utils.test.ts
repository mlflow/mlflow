import { EXPERIMENT_LOG_MODEL_HISTORY_TAG } from './experimentPage.common-utils';
import { fetchModelVersionsForRuns } from './experimentPage.fetch-utils';

jest.mock('../../../../model-registry/constants', () => ({
  MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER: 5,
}));

describe('experiment page fetch utils', () => {
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
      fetchModelVersionsForRuns(runsDataPayload, actionCreatorMock, dispatchMock);

      // We're chunking by 5 runs so we expect 2 calls for sum of 10 runs
      expect(actionCreatorMock).toHaveBeenCalledTimes(2);
      expect(actionCreatorMock.mock.calls[0][0]).toEqual({ run_id: ['run_1', 'run_3', 'run_5', 'run_7', 'run_9'] });
      expect(actionCreatorMock.mock.calls[1][0]).toEqual({
        run_id: ['run_11', 'run_13', 'run_15', 'run_17', 'run_19'],
      });
    });
  });
});
