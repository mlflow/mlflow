import { EXPERIMENT_LOG_MODEL_HISTORY_TAG } from './experimentPage.common-utils';
import { detectSqlSyntaxInSearchQuery, fetchModelVersionsForRuns } from './experimentPage.fetch-utils';

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

  describe('detectSqlSyntaxInSearchQuery', () => {
    test('it detects SQL syntax with regular identifiers', () => {
      expect(detectSqlSyntaxInSearchQuery('metrics.accuracy > 0.5')).toBe(true);
      expect(detectSqlSyntaxInSearchQuery('params.learning_rate = 0.01')).toBe(true);
      expect(detectSqlSyntaxInSearchQuery('attributes.run_name LIKE "test%"')).toBe(true);
    });

    test('it detects SQL syntax with backtick-quoted identifiers containing spaces', () => {
      expect(detectSqlSyntaxInSearchQuery('metrics.`avg model score` > 0.5')).toBe(true);
      expect(detectSqlSyntaxInSearchQuery('params.`learning rate` = 0.01')).toBe(true);
      expect(detectSqlSyntaxInSearchQuery('metrics.`test metric name` < 100')).toBe(true);
    });

    test('it detects SQL syntax with backtick-quoted identifiers containing special characters', () => {
      expect(detectSqlSyntaxInSearchQuery('metrics.`metric-with-dash` > 0')).toBe(true);
      expect(detectSqlSyntaxInSearchQuery('params.`param.with.dots` = "value"')).toBe(true);
    });

    test('it does not detect SQL syntax in plain text', () => {
      expect(detectSqlSyntaxInSearchQuery('just some text')).toBe(false);
      expect(detectSqlSyntaxInSearchQuery('foobar')).toBe(false);
      expect(detectSqlSyntaxInSearchQuery('run name contains spaces')).toBe(false);
    });
  });
});
