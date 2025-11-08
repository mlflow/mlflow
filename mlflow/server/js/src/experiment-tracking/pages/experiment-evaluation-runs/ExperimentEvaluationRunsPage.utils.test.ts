import type { ExperimentEvaluationRunsGroupData } from './ExperimentEvaluationRunsPage.utils';
import { getGroupByRunsData } from './ExperimentEvaluationRunsPage.utils';
import type { RunDatasetWithTags, RunEntity } from '../../types';
import type { KeyValueEntity } from '../../../common/types';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import {
  RunGroupingMode,
  RunGroupingAggregateFunction,
} from '../../components/experiment-page/utils/experimentPage.row-types';

const createMockDataset = (digest: string): RunDatasetWithTags => ({
  dataset: {
    digest,
    name: 'dataset',
    profile: 'profile',
    schema: 'schema',
    source: 'source',
    sourceType: 'code',
  },
  tags: [],
});

const createMockRun = ({
  runUuid,
  datasets,
  params,
  tags,
}: {
  runUuid: string;
  datasets?: RunDatasetWithTags[];
  params?: KeyValueEntity[];
  tags?: KeyValueEntity[];
}): RunEntity => ({
  data: {
    params: params ?? [],
    tags: tags ?? [],
    metrics: [],
  },
  info: {
    artifactUri: '',
    endTime: 0,
    experimentId: 'exp-1',
    lifecycleStage: '',
    runUuid,
    runName: 'Test Run',
    startTime: 0,
    status: 'FINISHED',
  },
  inputs: {
    datasetInputs: datasets ?? [],
    modelInputs: [],
  },
  outputs: {
    modelOutputs: [],
  },
});

const MOCK_RUNS = [
  createMockRun({
    runUuid: 'run-1',
    datasets: [createMockDataset('digest-1')],
    tags: [{ key: 'tag-1', value: 'value-1' }],
  }),
  createMockRun({
    runUuid: 'run-2',
    datasets: [createMockDataset('digest-1')],
    params: [{ key: 'param-1', value: 'value-1' }],
  }),
  createMockRun({
    runUuid: 'run-3',
    datasets: [createMockDataset('digest-2')],
  }),
  createMockRun({
    runUuid: 'run-4',
  }),
];

describe('ExperimentEvaluationRunsPage.utils', () => {
  describe('getGroupByRunsData', () => {
    it('should return runs unchanged if groupBy is null', () => {
      const result = getGroupByRunsData(MOCK_RUNS, null);

      expect(result).toEqual(MOCK_RUNS);
      expect(result).toHaveLength(4);
    });

    it('should group runs by dataset digest', () => {
      const groupBy: RunsGroupByConfig = {
        aggregateFunction: RunGroupingAggregateFunction.Average,
        groupByKeys: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset' }],
      };

      const result = getGroupByRunsData(MOCK_RUNS, groupBy);

      expect(result).toHaveLength(3);
      const group0 = result[0] as ExperimentEvaluationRunsGroupData;
      expect(group0.groupValues).toBeDefined();
      expect(group0.groupValues).toEqual([
        { mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'digest-1' },
      ]);
      expect(group0.subRuns).toHaveLength(2);
      expect(group0.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-1', 'run-2']);

      const group1 = result[1] as ExperimentEvaluationRunsGroupData;
      expect(group1.groupValues).toEqual([
        { mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'digest-2' },
      ]);
      expect(group1.subRuns).toHaveLength(1);
      expect(group1.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-3']);

      const group2 = result[2] as ExperimentEvaluationRunsGroupData;
      expect(group2.groupValues).toEqual([{ mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: null }]);
      expect(group2.subRuns).toHaveLength(1);
      expect(group2.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-4']);
    });

    it('should group runs by parameter value', () => {
      const groupBy: RunsGroupByConfig = {
        aggregateFunction: RunGroupingAggregateFunction.Average,
        groupByKeys: [{ mode: RunGroupingMode.Param, groupByData: 'param-1' }],
      };

      const result = getGroupByRunsData(MOCK_RUNS, groupBy);

      expect(result).toHaveLength(2);
      const group0 = result[0] as ExperimentEvaluationRunsGroupData;
      expect(group0.groupValues).toEqual([{ mode: RunGroupingMode.Param, groupByData: 'param-1', value: null }]);
      expect(group0.subRuns).toHaveLength(3);
      expect(group0.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-1', 'run-3', 'run-4']);

      const group1 = result[1] as ExperimentEvaluationRunsGroupData;
      expect(group1.groupValues).toEqual([{ mode: RunGroupingMode.Param, groupByData: 'param-1', value: 'value-1' }]);
      expect(group1.subRuns).toHaveLength(1);
      expect(group1.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-2']);
    });

    it('should group runs by tag value', () => {
      const groupBy: RunsGroupByConfig = {
        aggregateFunction: RunGroupingAggregateFunction.Average,
        groupByKeys: [{ mode: RunGroupingMode.Tag, groupByData: 'tag-1' }],
      };

      const result = getGroupByRunsData(MOCK_RUNS, groupBy);

      expect(result).toHaveLength(2);
      const group0 = result[0] as ExperimentEvaluationRunsGroupData;
      expect(group0.groupValues).toEqual([{ mode: RunGroupingMode.Tag, groupByData: 'tag-1', value: 'value-1' }]);
      expect(group0.subRuns).toHaveLength(1);
      expect(group0.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-1']);

      const group1 = result[1] as ExperimentEvaluationRunsGroupData;
      expect(group1.groupValues).toEqual([{ mode: RunGroupingMode.Tag, groupByData: 'tag-1', value: null }]);
      expect(group1.subRuns).toHaveLength(3);
      expect(group1.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-2', 'run-3', 'run-4']);
    });

    it('should group runs by combinations of grouping modes', () => {
      const groupBy: RunsGroupByConfig = {
        aggregateFunction: RunGroupingAggregateFunction.Average,
        groupByKeys: [
          { mode: RunGroupingMode.Dataset, groupByData: 'dataset' },
          { mode: RunGroupingMode.Param, groupByData: 'param-1' },
        ],
      };

      const result = getGroupByRunsData(MOCK_RUNS, groupBy);

      // 4 groups:
      // - dataset: digest-1, param-1: value-1
      // - dataset: digest-1, param-1: null
      // - dataset: digest-2, param-1: null
      // - dataset: null, param-1: null
      expect(result).toHaveLength(4);
      const group0 = result[0] as ExperimentEvaluationRunsGroupData;
      expect(group0.groupValues).toEqual([
        { mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'digest-1' },
        { mode: RunGroupingMode.Param, groupByData: 'param-1', value: null },
      ]);
      expect(group0.subRuns).toHaveLength(1);
      expect(group0.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-1']);

      const group1 = result[1] as ExperimentEvaluationRunsGroupData;
      expect(group1.groupValues).toEqual([
        { mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'digest-1' },
        { mode: RunGroupingMode.Param, groupByData: 'param-1', value: 'value-1' },
      ]);
      expect(group1.subRuns).toHaveLength(1);
      expect(group1.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-2']);

      const group2 = result[2] as ExperimentEvaluationRunsGroupData;
      expect(group2.groupValues).toEqual([
        { mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'digest-2' },
        { mode: RunGroupingMode.Param, groupByData: 'param-1', value: null },
      ]);
      expect(group2.subRuns).toHaveLength(1);
      expect(group2.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-3']);

      const group3 = result[3] as ExperimentEvaluationRunsGroupData;
      expect(group3.groupValues).toEqual([
        { mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: null },
        { mode: RunGroupingMode.Param, groupByData: 'param-1', value: null },
      ]);
      expect(group3.subRuns).toHaveLength(1);
      expect(group3.subRuns?.map((r) => r.info.runUuid)).toEqual(['run-4']);
    });

    it('should handle empty runs array', () => {
      const groupBy: RunsGroupByConfig = {
        aggregateFunction: RunGroupingAggregateFunction.Average,
        groupByKeys: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset' }],
      };

      const result = getGroupByRunsData([], groupBy);

      expect(result).toEqual([]);
    });
  });
});
