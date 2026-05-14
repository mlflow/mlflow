import { describe, it, expect } from '@jest/globals';
import type { ExperimentEvaluationRunsGroupData } from './ExperimentEvaluationRunsPage.utils';
import { getGroupByRunsData, reconcileSelectedColumns } from './ExperimentEvaluationRunsPage.utils';
import type { RunDatasetWithTags, RunEntity } from '../../types';
import type { KeyValueEntity } from '../../../common/types';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import {
  RunGroupingMode,
  RunGroupingAggregateFunction,
} from '../../components/experiment-page/utils/experimentPage.row-types';
import {
  EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
  EvalRunsTableKeyedColumnPrefix,
} from './ExperimentEvaluationRunsTable.constants';

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

  describe('reconcileSelectedColumns', () => {
    const metric = (key: string) => `${EvalRunsTableKeyedColumnPrefix.METRIC}.${key}`;
    const param = (key: string) => `${EvalRunsTableKeyedColumnPrefix.PARAM}.${key}`;

    it('initializes defaults when previous state only has base columns', () => {
      const previous = { ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE };
      const uniqueColumns = [metric('a'), metric('b'), metric('c')];

      const result = reconcileSelectedColumns({
        previous,
        uniqueColumns,
        defaultVisibleMetricColumns: 5,
        enableImprovedComparison: true,
      });

      // Base columns still present
      for (const baseKey of Object.keys(EVAL_RUNS_TABLE_BASE_SELECTION_STATE)) {
        expect(result).toHaveProperty(baseKey, EVAL_RUNS_TABLE_BASE_SELECTION_STATE[baseKey]);
      }
      // New metric columns enabled by default
      expect(result[metric('a')]).toBe(true);
      expect(result[metric('b')]).toBe(true);
      expect(result[metric('c')]).toBe(true);
    });

    it("preserves the user's choice for columns that still exist when uniqueColumns changes", () => {
      // User had A, B, C and explicitly disabled B
      const previous = {
        ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
        [metric('a')]: true,
        [metric('b')]: false,
        [metric('c')]: true,
      };
      // Filter narrows to A, B (C disappeared)
      const uniqueColumns = [metric('a'), metric('b')];

      const result = reconcileSelectedColumns({
        previous,
        uniqueColumns,
        defaultVisibleMetricColumns: 5,
        enableImprovedComparison: true,
      });

      // User's disabled choice for B is preserved
      expect(result[metric('b')]).toBe(false);
      // A still enabled
      expect(result[metric('a')]).toBe(true);
      // C dropped because it no longer exists
      expect(metric('c') in result).toBe(false);
    });

    it('drops columns that no longer appear in uniqueColumns', () => {
      const previous = {
        ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
        [metric('a')]: true,
        [metric('b')]: true,
        [param('p1')]: true,
      };
      const uniqueColumns = [metric('a')];

      const result = reconcileSelectedColumns({
        previous,
        uniqueColumns,
        defaultVisibleMetricColumns: 5,
        enableImprovedComparison: true,
      });

      expect(result[metric('a')]).toBe(true);
      expect(metric('b') in result).toBe(false);
      expect(param('p1') in result).toBe(false);
    });

    it('applies default-enabled state for newly-appeared metric columns', () => {
      // User had A enabled, B disabled
      const previous = {
        ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
        [metric('a')]: true,
        [metric('b')]: false,
      };
      // Filter widens to A, B, C, D (C and D are new)
      const uniqueColumns = [metric('a'), metric('b'), metric('c'), metric('d')];

      const result = reconcileSelectedColumns({
        previous,
        uniqueColumns,
        defaultVisibleMetricColumns: 5,
        enableImprovedComparison: true,
      });

      // Existing user choices preserved
      expect(result[metric('a')]).toBe(true);
      expect(result[metric('b')]).toBe(false);
      // Newly-appeared metric columns enabled by default (within visible limit)
      expect(result[metric('c')]).toBe(true);
      expect(result[metric('d')]).toBe(true);
    });

    it('respects the visible metric limit only for newly-appeared columns when flag is on', () => {
      const previous = { ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE };
      const uniqueColumns = [metric('a'), metric('b'), metric('c'), metric('d'), metric('e'), metric('f'), metric('g')];

      const result = reconcileSelectedColumns({
        previous,
        uniqueColumns,
        defaultVisibleMetricColumns: 5,
        enableImprovedComparison: true,
      });

      // First five metrics enabled
      expect(result[metric('a')]).toBe(true);
      expect(result[metric('b')]).toBe(true);
      expect(result[metric('c')]).toBe(true);
      expect(result[metric('d')]).toBe(true);
      expect(result[metric('e')]).toBe(true);
      // Beyond the limit, default to disabled
      expect(result[metric('f')]).toBe(false);
      expect(result[metric('g')]).toBe(false);
    });

    it('enables all metric columns by default when the improved comparison flag is off', () => {
      const previous = { ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE };
      const uniqueColumns = [metric('a'), metric('b'), metric('c'), metric('d'), metric('e'), metric('f'), metric('g')];

      const result = reconcileSelectedColumns({
        previous,
        uniqueColumns,
        defaultVisibleMetricColumns: 5,
        enableImprovedComparison: false,
      });

      for (const col of uniqueColumns) {
        expect(result[col]).toBe(true);
      }
    });

    it('defaults newly-appeared non-metric columns (params, tags) to disabled', () => {
      const previous = { ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE };
      const uniqueColumns = [param('p1'), `${EvalRunsTableKeyedColumnPrefix.TAG}.t1`];

      const result = reconcileSelectedColumns({
        previous,
        uniqueColumns,
        defaultVisibleMetricColumns: 5,
        enableImprovedComparison: true,
      });

      expect(result[param('p1')]).toBe(false);
      expect(result[`${EvalRunsTableKeyedColumnPrefix.TAG}.t1`]).toBe(false);
    });

    it("preserves the user's overrides to base columns", () => {
      const previous = {
        ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
        // User toggled model_version on
        model_version: true,
        // User hid status
        status: false,
      };

      const result = reconcileSelectedColumns({
        previous,
        uniqueColumns: [],
        defaultVisibleMetricColumns: 5,
        enableImprovedComparison: true,
      });

      expect(result['model_version']).toBe(true);
      expect(result['status']).toBe(false);
    });
  });
});
