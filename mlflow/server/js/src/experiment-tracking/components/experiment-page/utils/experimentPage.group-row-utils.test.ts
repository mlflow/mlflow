import { compact, first, flatMap, get, last } from 'lodash';
import {
  createAggregatedMetricHistory,
  createRunsGroupByKey,
  createValueAggregatedMetricHistory,
  getGroupedRowRenderMetadata,
  getRunGroupDisplayName,
  normalizeRunsGroupByKey,
} from './experimentPage.group-row-utils';
import type { RowGroupRenderMetadata } from './experimentPage.row-types';
import { RunGroupingAggregateFunction, RunGroupingMode } from './experimentPage.row-types';
import type { SingleRunData } from './experimentPage.row-utils';
import { MOCK_RUN_UUIDS_TO_HISTORY_MAP } from '../fixtures/experiment-runs.fixtures';

describe('createRunsGroupByKey', () => {
  it('should return an empty string if mode is undefined', () => {
    expect(createRunsGroupByKey(undefined, 'groupByData', RunGroupingAggregateFunction.Max)).toEqual('');
  });
  it('should return the correct group key when mode is defined', () => {
    expect(createRunsGroupByKey(RunGroupingMode.Param, 'param2', RunGroupingAggregateFunction.Max)).toEqual(
      'param.max.param2',
    );
    expect(createRunsGroupByKey(RunGroupingMode.Tag, 'tagA', RunGroupingAggregateFunction.Min)).toEqual('tag.min.tagA');
    expect(createRunsGroupByKey(RunGroupingMode.Dataset, 'dataset', RunGroupingAggregateFunction.Average)).toEqual(
      'dataset.average.dataset',
    );
  });
});
describe('normalizeRunsGroupByKey', () => {
  it('should return null if groupByKey is undefined', () => {
    const groupByKey = undefined;
    const result = normalizeRunsGroupByKey(groupByKey);
    expect(result).toBeNull();
  });
  it('should return null if groupByKey does not match the expected pattern', () => {
    const groupByKey = 'invalidKey';
    const result = normalizeRunsGroupByKey(groupByKey);
    expect(result).toBeNull();
  });
  it('should return null object if group by config does not match the expected mode', () => {
    expect(normalizeRunsGroupByKey('somemode.min.groupByData')).toBeNull();
    expect(normalizeRunsGroupByKey('param.somefunction.groupByData')).toBeNull();
  });
  it('should return properly parsed group by config', () => {
    expect(normalizeRunsGroupByKey('param.min.groupByData')).toEqual({
      aggregateFunction: RunGroupingAggregateFunction.Min,
      groupByKeys: [
        {
          mode: RunGroupingMode.Param,
          groupByData: 'groupByData',
        },
      ],
    });
    expect(normalizeRunsGroupByKey('tag.max.groupByData')).toEqual({
      aggregateFunction: RunGroupingAggregateFunction.Max,
      groupByKeys: [
        {
          mode: RunGroupingMode.Tag,
          groupByData: 'groupByData',
        },
      ],
    });
    expect(normalizeRunsGroupByKey('dataset.average.dataset')).toEqual({
      aggregateFunction: RunGroupingAggregateFunction.Average,
      groupByKeys: [
        {
          mode: RunGroupingMode.Dataset,
          groupByData: 'dataset',
        },
      ],
    });
  });
});
describe('getGroupedRowRenderMetadata', () => {
  describe('grouping by single tags or single param', () => {
    const testRunData: SingleRunData[] = [
      {
        runInfo: {
          runUuid: 'run1',
        } as any,
        datasets: [],
        metrics: [{ key: 'metric1', value: 2 }] as any,
        params: [
          { key: 'param1', value: 'param_1_value_1' },
          { key: 'param2', value: 'param_2_value_1' },
          { key: 'param3', value: 'param_3_value' },
          { key: 'param_number', value: '-170' },
        ] as any,
        tags: { tag1: { key: 'tag1', value: 'tag_1_value_1' }, tag2: { key: 'tag2', value: 'tag_2_value_1' } } as any,
      },
      {
        runInfo: {
          runUuid: 'run2',
        } as any,
        datasets: [],
        metrics: [{ key: 'metric1', value: 8 }] as any,
        params: [
          { key: 'param1', value: 'param_1_value_1' },
          { key: 'param2', value: 'param_2_value_2' },
          { key: 'param3', value: 'param_3_value' },
          { key: 'param_number', value: '-50' },
        ] as any,
        tags: { tag1: { key: 'tag1', value: 'tag_1_value_1' }, tag2: { key: 'tag2', value: 'tag_2_value_2' } } as any,
      },
      {
        runInfo: {
          runUuid: 'run3',
        } as any,
        datasets: [],
        metrics: [{ key: 'metric1', value: 14 }] as any,
        params: [
          { key: 'param3', value: 'param_3_value' },
          { key: 'param_number', value: '-80' },
        ] as any,
        tags: {},
      },
    ];
    it('grouping by tags with groups containing multiple runs and contracted ungrouped runs', () => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'tag1',
              mode: RunGroupingMode.Tag,
            },
          ],
        },
        groupsExpanded: {},
        runData: testRunData,
      });
      // We expect 4 rows: 2 groups and 2 runs
      expect(groupedRunsMetadata).toHaveLength(4);
      expect(get(groupedRunsMetadata, '0.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '1.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '2.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '3.isGroup')).toEqual(true);
      // First group contains runs relevant to "tag_1_value_1" value of "tag1"
      expect(get(groupedRunsMetadata, '0.groupId')).toEqual('tag.tag1.tag_1_value_1');
      // Second group contains remaining runs without value
      expect(get(groupedRunsMetadata, '3.groupId')).toEqual('tag.tag1');
    });
    it('grouping by tags with groups containing single runs and contracted ungrouped runs', () => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'tag2',
              mode: RunGroupingMode.Tag,
            },
          ],
        },
        groupsExpanded: {},
        runData: testRunData,
      });
      // We expect 5 rows: 3 groups and 2 runs
      expect(groupedRunsMetadata).toHaveLength(5);
      expect(get(groupedRunsMetadata, '0.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '1.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '2.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '3.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '4.isGroup')).toEqual(true);
      // First group contains runs relevant to "tag_2_value_1" value of "tag2"
      expect(get(groupedRunsMetadata, '0.groupId')).toEqual('tag.tag2.tag_2_value_1');
      // Second group contains runs relevant to "tag_2_value_2" value of "tag2"
      expect(get(groupedRunsMetadata, '2.groupId')).toEqual('tag.tag2.tag_2_value_2');
      // Third group contains remaining runs without value
      expect(get(groupedRunsMetadata, '4.groupId')).toEqual('tag.tag2');
    });
    it('grouping by tags with groups containing expanded ungrouped runs', () => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'tag2',
              mode: RunGroupingMode.Tag,
            },
          ],
        },
        // Expand group with remaining runs
        groupsExpanded: { 'tag.tag2': true },
        runData: testRunData,
      });
      // We expect 6 rows: 3 groups and 3 runs
      expect(groupedRunsMetadata).toHaveLength(6);
      expect(get(groupedRunsMetadata, '0.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '1.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '2.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '3.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '4.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '5.isGroup')).toBeUndefined();
      // First two runs contains that are grouped, last run is ungrouped (does not have any matching value)
      expect(get(groupedRunsMetadata, '1.belongsToGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '3.belongsToGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '5.belongsToGroup')).toEqual(false);
    });
    it('grouping using unknown mode returns only ungrouped values', () => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'tag2',
              mode: 'someUnknownMode' as any,
            },
          ],
        },
        groupsExpanded: {},
        runData: testRunData,
      });
      const allRunUuids = testRunData.map((run) => run.runInfo.runUuid);

      expect(get(groupedRunsMetadata, '0.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '0.isRemainingRunsGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '0.runUuids')).toEqual(allRunUuids);
    });

    it('grouping by params with groups containing all groups contracted', () => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'tag2',
              mode: RunGroupingMode.Tag,
            },
          ],
        },
        // Contract all groups
        groupsExpanded: { 'tag.tag2.tag_2_value_1': false, 'tag.tag2.tag_2_value_2': false, 'tag.tag2': false },
        runData: testRunData,
      });
      // We should have only group rows without any runs
      expect(groupedRunsMetadata).toHaveLength(3);
      expect(get(groupedRunsMetadata, '0.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '1.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '2.isGroup')).toEqual(true);
    });
    it.each([
      // Aggregate functions and expected metric values according to the test data
      [RunGroupingAggregateFunction.Average, 8],
      [RunGroupingAggregateFunction.Min, 2],
      [RunGroupingAggregateFunction.Max, 14],
    ])('groups by param and aggregates the metric data properly by %s', (aggregateFunction, result) => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction,
          groupByKeys: [
            {
              groupByData: 'param3',
              mode: RunGroupingMode.Param,
            },
          ],
        },
        groupsExpanded: {},
        runData: testRunData,
      });
      expect(groupedRunsMetadata).toHaveLength(4);
      expect(get(groupedRunsMetadata, '0.aggregatedMetricEntities')).toEqual([
        { key: 'metric1', value: result, maxStep: 0 },
      ]);
    });
    it.each([
      // Aggregate functions and expected param values according to the test data
      [RunGroupingAggregateFunction.Average, -100],
      [RunGroupingAggregateFunction.Min, -170],
      [RunGroupingAggregateFunction.Max, -50],
    ])('groups by param and aggregates the param data properly by %s', (aggregateFunction, result) => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction,
          groupByKeys: [
            {
              groupByData: 'param3',
              mode: RunGroupingMode.Param,
            },
          ],
        },
        groupsExpanded: {},
        runData: testRunData,
      });
      expect(groupedRunsMetadata).toHaveLength(4);
      expect(get(groupedRunsMetadata, '0.aggregatedParamEntities')).toEqual([
        { key: 'param_number', maxStep: 0, value: result },
      ]);
    });
  });
  describe('grouping by datasets', () => {
    const testRunData: SingleRunData[] = [
      {
        runInfo: {
          runUuid: 'run1',
        } as any,
        datasets: [{ dataset: { name: 'dataset_alpha', digest: '1234' } }] as any,
        metrics: [{ key: 'metric1', value: 2 }] as any,
        params: [],
        tags: {},
      },
      {
        runInfo: {
          runUuid: 'run2',
        } as any,
        datasets: [{ dataset: { name: 'dataset_alpha', digest: '1234' } }] as any,
        metrics: [{ key: 'metric1', value: 8 }] as any,
        params: [],
        tags: {},
      },
      {
        runInfo: {
          runUuid: 'run3',
        } as any,
        // Similar dataset but with another digest
        datasets: [{ dataset: { name: 'dataset_alpha', digest: '1a2b3c4d' } }] as any,
        metrics: [{ key: 'metric1', value: 14 }] as any,
        params: [],
        tags: {},
      },
      {
        runInfo: {
          runUuid: 'run4',
        } as any,
        datasets: [{ dataset: { name: 'dataset_beta', digest: '321' } }] as any,
        metrics: [{ key: 'metric1', value: 14 }] as any,
        params: [],
        tags: {},
      },
    ];
    it('grouping by datasets with groups containing multiple runs and contracted ungrouped runs', () => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'dataset',
              mode: RunGroupingMode.Dataset,
            },
          ],
        },
        groupsExpanded: {},
        runData: testRunData,
      });
      // We expect 7 rows: 3 groups and 4 runs
      expect(groupedRunsMetadata).toHaveLength(7);
      expect(get(groupedRunsMetadata, '0.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '1.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '2.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '3.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '4.isGroup')).toBeUndefined();
      expect(get(groupedRunsMetadata, '5.isGroup')).toEqual(true);
      expect(get(groupedRunsMetadata, '6.isGroup')).toBeUndefined();
      // We expect 3 groups - first one contains two runs
      expect(get(groupedRunsMetadata, '0.groupId')).toEqual('dataset.dataset.dataset_alpha.1234');
      expect(get(groupedRunsMetadata, '0.runUuids')).toEqual(['run1', 'run2']);
      // Second group differs only by digest and contains one run
      expect(get(groupedRunsMetadata, '3.groupId')).toEqual('dataset.dataset.dataset_alpha.1a2b3c4d');
      expect(get(groupedRunsMetadata, '3.runUuids')).toEqual(['run3']);
      // Third group contains one run
      expect(get(groupedRunsMetadata, '5.groupId')).toEqual('dataset.dataset.dataset_beta.321');
      expect(get(groupedRunsMetadata, '5.runUuids')).toEqual(['run4']);
    });
  });
  describe('grouping by multiple keys', () => {
    const smallTestRunData: SingleRunData[] = [
      {
        runInfo: {
          runUuid: 'run1',
        } as any,
        datasets: [],
        metrics: [{ key: 'metric1', value: 2 }] as any,
        params: [
          { key: 'param1', value: 'param_1_value_1' },
          { key: 'param2', value: 'param_2_value_1' },
          { key: 'param3', value: 'param_3_value' },
          { key: 'param_number', value: '-170' },
        ] as any,
        tags: { tag1: { key: 'tag1', value: 'tag_1_value_1' }, tag2: { key: 'tag2', value: 'tag_2_value_1' } } as any,
      },
      {
        runInfo: {
          runUuid: 'run2',
        } as any,
        datasets: [{ dataset: { name: 'dataset_alpha', digest: '1234' } }] as any,
        metrics: [{ key: 'metric1', value: 8 }] as any,
        params: [
          { key: 'param1', value: 'param_1_value_1' },
          { key: 'param2', value: 'param_2_value_2' },
          { key: 'param3', value: 'param_3_value' },
          { key: 'param_number', value: '-50' },
        ] as any,
        tags: { tag1: { key: 'tag1', value: 'tag_1_value_1' }, tag2: { key: 'tag2', value: 'tag_2_value_2' } } as any,
      },
      {
        runInfo: {
          runUuid: 'run3',
        } as any,
        datasets: [
          { dataset: { name: 'dataset_alpha', digest: '1234' } },
          { dataset: { name: 'dataset_beta', digest: '321' } },
        ] as any,
        metrics: [{ key: 'metric1', value: 14 }] as any,
        params: [
          { key: 'param3', value: 'param_3_value' },
          { key: 'param_number', value: '-80' },
        ],
        tags: {},
      },
      {
        runInfo: {
          runUuid: 'run4',
        } as any,
        datasets: [] as any,
        metrics: [{ key: 'metric1', value: 14 }] as any,
        params: [
          { key: 'param3', value: 'param_3_value' },
          { key: 'param_number', value: '-80' },
        ],
        tags: {},
      },
    ];

    test('it should properly group by parameter and tag', () => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'param2',
              mode: RunGroupingMode.Param,
            },
            {
              mode: RunGroupingMode.Tag,
              groupByData: 'tag2',
            },
          ],
        },
        groupsExpanded: {},
        runData: smallTestRunData,
      });

      const resultingGroups = groupedRunsMetadata?.filter(({ isGroup }) => isGroup) as RowGroupRenderMetadata[];

      // We expect 3 groups: 2 groups with grouped values and 1 "remaining runs" group
      expect(resultingGroups.map(getRunGroupDisplayName)).toEqual([
        'param2: param_2_value_1, tag2: tag_2_value_1',
        'param2: param_2_value_2, tag2: tag_2_value_2',
        '',
      ]);
      expect(get(last(resultingGroups), 'isRemainingRunsGroup')).toEqual(true);

      expect(get(resultingGroups, '0.runUuids')).toEqual(['run1']);
      expect(get(resultingGroups, '1.runUuids')).toEqual(['run2']);
      expect(get(resultingGroups, '2.runUuids')).toEqual(['run3', 'run4']);
    });
    test('it should properly group by datasets and tag', () => {
      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'dataset',
              mode: RunGroupingMode.Dataset,
            },
            {
              mode: RunGroupingMode.Tag,
              groupByData: 'tag2',
            },
          ],
        },
        groupsExpanded: {},
        runData: smallTestRunData,
      });

      const resultingGroups = groupedRunsMetadata?.filter(({ isGroup }) => isGroup) as RowGroupRenderMetadata[];

      // We expect 5 groups: 4 groups with grouped values and 1 "remaining runs" group
      expect(resultingGroups.map(getRunGroupDisplayName)).toEqual([
        'tag2: tag_2_value_1, dataset: (none)',
        'tag2: tag_2_value_2, dataset: dataset_alpha.1234',
        'tag2: (none), dataset: dataset_alpha.1234',
        'tag2: (none), dataset: dataset_beta.321',
        '',
      ]);
      expect(get(last(resultingGroups), 'isRemainingRunsGroup')).toEqual(true);

      expect(get(resultingGroups, '0.runUuids')).toEqual(['run1']);
      expect(get(resultingGroups, '1.runUuids')).toEqual(['run2']);
      expect(get(resultingGroups, '2.runUuids')).toEqual(['run3']);
      expect(get(resultingGroups, '3.runUuids')).toEqual(['run3']);
      expect(get(resultingGroups, '4.runUuids')).toEqual(['run4']);
    });
    test('it should properly group heavily uncorrelated runs data', () => {
      const heavyRunsData: SingleRunData[] = new Array(50).fill(0).map((_, i) => {
        return {
          datasets: new Array(50).fill(0).map((_, j) => ({
            dataset: {
              digest: `digest_${j}`,
              name: `name_dataset${j}_run_${i}`,
            },
          })),
          metrics: [],
          params: [
            {
              key: 'param_a',
              value: `param_value_${i}`,
            },
          ],
          runInfo: {
            runUuid: `run${i}`,
          },
          tags: {
            tag_a: {
              key: 'tag_a',
              value: `tag_value_${i}`,
            },
          },
        } as any;
      });

      const groupedRunsMetadata = getGroupedRowRenderMetadata({
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              groupByData: 'dataset',
              mode: RunGroupingMode.Dataset,
            },
            {
              mode: RunGroupingMode.Tag,
              groupByData: 'tag_a',
            },
            {
              mode: RunGroupingMode.Param,
              groupByData: 'param_a',
            },
          ],
        },
        groupsExpanded: {},
        runData: heavyRunsData,
      });

      // We expect 50*50 groups and 50*50 runs, since there are
      // 50 distinct datasets in each run with 50 distinct tag and param values
      const resultingGroups = groupedRunsMetadata?.filter(({ isGroup }) => isGroup) as RowGroupRenderMetadata[];
      const resultingRuns = groupedRunsMetadata?.filter(({ isGroup }) => !isGroup) as RowGroupRenderMetadata[];
      expect(resultingGroups).toHaveLength(50 * 50);
      expect(resultingRuns).toHaveLength(50 * 50);
    });
  });
});

describe('metric history aggregation', () => {
  it('correctly generates aggregated metrics', () => {
    // for metric vs metric plotting
    const {
      min: valuesMin,
      max: valuesMax,
      average: valuesAvg,
    } = createValueAggregatedMetricHistory(
      MOCK_RUN_UUIDS_TO_HISTORY_MAP,
      'metric', // metricKey
      'base', // selectedXAxisMetricKey
      false, // ignoreOutliers
    );
    // for metric vs step plotting
    const metricsHistoryInGroup = flatMap(MOCK_RUN_UUIDS_TO_HISTORY_MAP, (obj) => {
      return obj['metric'].metricsHistory;
    });
    const steps = [0, 1, 2, 3, 4];
    const {
      min: stepMin,
      max: stepMax,
      average: stepAvg,
    } = createAggregatedMetricHistory(steps, 'metric', metricsHistoryInGroup);
    /**
     * the mock metric object is constructed such that the min and max
     * are -2 and +2 with respect to the base. however, the base differs
     * from run to run, for example:
     *
     * run1 (metric is +2): base = [1, 2, 3, 4, 5], metric = [3, 4, 5, 6, 7]
     * run2 (metric is -2): base = [5, 4, 3, 2, 1], metric = [3, 2, 1, 0, -1]
     *
     * therefore, when `metric` is plotted with `base` as the x-axis, we should
     * see min/max consistently being +/- 2 vs. average.
     *
     * with `steps` as the x-axis, we should see the range expand from +/- 0 to +/- 8.
     *
     * the asserts below should illustrate the differences between the two
     * aggregators.
     */
    expect(valuesAvg.map((e) => e.value)).toEqual([1, 2, 3, 4, 5]);
    expect(stepAvg.map((e) => e.value)).toEqual([3, 3, 3, 3, 3]);
    expect(valuesMin.map((e) => e.value)).toEqual([-1, 0, 1, 2, 3]);
    expect(stepMin.map((e) => e.value)).toEqual([3, 2, 1, 0, -1]);
    expect(valuesMax.map((e) => e.value)).toEqual([3, 4, 5, 6, 7]);
    expect(stepMax.map((e) => e.value)).toEqual([3, 4, 5, 6, 7]);
  });
});
