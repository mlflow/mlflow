import { describe, it, expect } from '@jest/globals';
import type { ExperimentEvaluationRunsGroupData } from './ExperimentEvaluationRunsPage.utils';
import { flattenRunEntityOrGroupData, getGroupByRunsData, getNestedRuns } from './ExperimentEvaluationRunsPage.utils';
import type { RunDatasetWithTags, RunEntity } from '../../types';
import type { KeyValueEntity } from '../../../common/types';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import {
  RunGroupingMode,
  RunGroupingAggregateFunction,
} from '../../components/experiment-page/utils/experimentPage.row-types';
import { getParentRunTagName } from '../../actions';

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
  parentRunId,
}: {
  runUuid: string;
  datasets?: RunDatasetWithTags[];
  params?: KeyValueEntity[];
  tags?: KeyValueEntity[];
  parentRunId?: string;
}): RunEntity => ({
  data: {
    params: params ?? [],
    tags: [...(tags ?? []), ...(parentRunId ? [{ key: getParentRunTagName(), value: parentRunId }] : [])],
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

  describe('getNestedRuns', () => {
    it('should return empty array for empty input', () => {
      const result = getNestedRuns([]);
      expect(result).toEqual([]);
    });

    it('should return runs unchanged when no parent-child relationships exist', () => {
      const runs = [
        createMockRun({ runUuid: 'run1' }),
        createMockRun({ runUuid: 'run2' }),
        createMockRun({ runUuid: 'run3' }),
      ];
      const result = getNestedRuns(runs);

      expect(result).toHaveLength(3);
      expect(result[0].children).toBeUndefined();
      expect(result[1].children).toBeUndefined();
      expect(result[2].children).toBeUndefined();
    });

    it('should nest single child under parent', () => {
      const runs = [
        createMockRun({ runUuid: 'parent1' }),
        createMockRun({ runUuid: 'child1', parentRunId: 'parent1' }),
      ];
      const result = getNestedRuns(runs);

      expect(result).toHaveLength(1);
      expect(result[0].info.runUuid).toBe('parent1');
      expect(result[0].children).toHaveLength(1);
      expect(result[0].children![0].info.runUuid).toBe('child1');
    });

    it('should nest multiple children under same parent', () => {
      const runs = [
        createMockRun({ runUuid: 'parent1' }),
        createMockRun({ runUuid: 'child1', parentRunId: 'parent1' }),
        createMockRun({ runUuid: 'child2', parentRunId: 'parent1' }),
        createMockRun({ runUuid: 'child3', parentRunId: 'parent1' }),
      ];
      const result = getNestedRuns(runs);

      expect(result).toHaveLength(1);
      expect(result[0].info.runUuid).toBe('parent1');
      expect(result[0].children).toHaveLength(3);
      expect(result[0].children!.map((c) => c.info.runUuid)).toEqual(['child1', 'child2', 'child3']);
    });

    it('should handle multi-level hierarchy (grandparent → parent → child)', () => {
      const runs = [
        createMockRun({ runUuid: 'grandparent' }),
        createMockRun({ runUuid: 'parent', parentRunId: 'grandparent' }),
        createMockRun({ runUuid: 'child', parentRunId: 'parent' }),
      ];
      const result = getNestedRuns(runs);

      expect(result).toHaveLength(1);
      expect(result[0].info.runUuid).toBe('grandparent');
      expect(result[0].children).toHaveLength(1);
      expect(result[0].children![0].info.runUuid).toBe('parent');
      expect(result[0].children![0].children).toHaveLength(1);
      expect(result[0].children![0].children![0].info.runUuid).toBe('child');
    });

    it('should omit children whose parents are missing', () => {
      const runs = [
        createMockRun({ runUuid: 'orphan1', parentRunId: 'missing-parent' }),
        createMockRun({ runUuid: 'orphan2', parentRunId: 'another-missing' }),
        createMockRun({ runUuid: 'regular-parent' }),
        createMockRun({ runUuid: 'regular-child', parentRunId: 'regular-parent' }),
      ];
      const result = getNestedRuns(runs);

      expect(result).toHaveLength(1);
      expect(result[0].info.runUuid).toBe('regular-parent');
      expect(result[0].children).toHaveLength(1);
      expect(result[0].children![0].info.runUuid).toBe('regular-child');
    });

    it('should handle multiple independent parent-child families', () => {
      const runs = [
        createMockRun({ runUuid: 'parent1' }),
        createMockRun({ runUuid: 'child1a', parentRunId: 'parent1' }),
        createMockRun({ runUuid: 'child1b', parentRunId: 'parent1' }),
        createMockRun({ runUuid: 'parent2' }),
        createMockRun({ runUuid: 'child2a', parentRunId: 'parent2' }),
        createMockRun({ runUuid: 'standalone' }),
      ];
      const result = getNestedRuns(runs);

      expect(result).toHaveLength(3);

      const family1 = result.find((r) => r.info.runUuid === 'parent1');
      const family2 = result.find((r) => r.info.runUuid === 'parent2');
      const standalone = result.find((r) => r.info.runUuid === 'standalone');

      expect(family1!.children).toHaveLength(2);
      expect(family2!.children).toHaveLength(1);
      expect(standalone!.children).toBeUndefined();
    });

    it('should preserve run order in children array', () => {
      const runs = [
        createMockRun({ runUuid: 'parent' }),
        createMockRun({ runUuid: 'child-first', parentRunId: 'parent' }),
        createMockRun({ runUuid: 'child-second', parentRunId: 'parent' }),
        createMockRun({ runUuid: 'child-third', parentRunId: 'parent' }),
      ];
      const result = getNestedRuns(runs);

      expect(result[0].children!.map((c) => c.info.runUuid)).toEqual(['child-first', 'child-second', 'child-third']);
    });

    it('should not mutate original runs array', () => {
      const runs = [createMockRun({ runUuid: 'parent' }), createMockRun({ runUuid: 'child', parentRunId: 'parent' })];
      const originalRunsClone = JSON.parse(JSON.stringify(runs));

      getNestedRuns(runs);

      expect(runs).toEqual(originalRunsClone);
    });
  });

  describe('flattenRunEntityOrGroupData', () => {
    it('should return empty array for empty input', () => {
      const result = flattenRunEntityOrGroupData([]);
      expect(result).toEqual([]);
    });

    it('should return flat array unchanged (no hierarchy)', () => {
      const runs = [
        createMockRun({ runUuid: 'run1' }),
        createMockRun({ runUuid: 'run2' }),
        createMockRun({ runUuid: 'run3' }),
      ];
      const result = flattenRunEntityOrGroupData(runs);

      expect(result).toHaveLength(3);
      expect(result[0].info.runUuid).toBe('run1');
      expect(result[1].info.runUuid).toBe('run2');
      expect(result[2].info.runUuid).toBe('run3');
    });

    it('should flatten deeply nested hierarchies (3+ levels)', () => {
      const runs = [
        createMockRun({ runUuid: 'grandparent' }),
        createMockRun({ runUuid: 'parent', parentRunId: 'grandparent' }),
        createMockRun({ runUuid: 'child', parentRunId: 'parent' }),
        createMockRun({ runUuid: 'grandchild', parentRunId: 'child' }),
      ];
      const nested = getNestedRuns(runs);
      const result = flattenRunEntityOrGroupData(nested);

      expect(result).toHaveLength(4);
      const uuids = result.map((r) => r.info.runUuid);
      expect(uuids).toContain('grandparent');
      expect(uuids).toContain('parent');
      expect(uuids).toContain('child');
      expect(uuids).toContain('grandchild');
    });

    it('should flatten grouped runs (extract from subRuns)', () => {
      const runs = [
        createMockRun({ runUuid: 'run1', datasets: [createMockDataset('digest-1')] }),
        createMockRun({ runUuid: 'run2', datasets: [createMockDataset('digest-1')] }),
        createMockRun({ runUuid: 'run3', datasets: [createMockDataset('digest-2')] }),
      ];
      const groupBy: RunsGroupByConfig = {
        aggregateFunction: RunGroupingAggregateFunction.Average,
        groupByKeys: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset' }],
      };
      const grouped = getGroupByRunsData(runs, groupBy);
      const result = flattenRunEntityOrGroupData(grouped);

      expect(result).toHaveLength(3);
      const uuids = result.map((r) => r.info.runUuid);
      expect(uuids).toContain('run1');
      expect(uuids).toContain('run2');
      expect(uuids).toContain('run3');
    });

    it('should handle multiple independent parent-child families', () => {
      const runs = [
        createMockRun({ runUuid: 'parent1' }),
        createMockRun({ runUuid: 'child1a', parentRunId: 'parent1' }),
        createMockRun({ runUuid: 'child1b', parentRunId: 'parent1' }),
        createMockRun({ runUuid: 'parent2' }),
        createMockRun({ runUuid: 'child2a', parentRunId: 'parent2' }),
        createMockRun({ runUuid: 'standalone' }),
      ];
      const nested = getNestedRuns(runs);
      const result = flattenRunEntityOrGroupData(nested);

      expect(result).toHaveLength(6);
      const uuids = result.map((r) => r.info.runUuid);
      expect(uuids).toContain('parent1');
      expect(uuids).toContain('child1a');
      expect(uuids).toContain('child1b');
      expect(uuids).toContain('parent2');
      expect(uuids).toContain('child2a');
      expect(uuids).toContain('standalone');
    });

    it('should preserve all run properties during flattening', () => {
      const run = createMockRun({
        runUuid: 'test-run',
        datasets: [createMockDataset('test-digest')],
        params: [{ key: 'param1', value: 'value1' }],
        tags: [{ key: 'tag1', value: 'tagvalue1' }],
      });
      const result = flattenRunEntityOrGroupData([run]);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual(run);
      expect(result[0].data.params).toEqual([{ key: 'param1', value: 'value1' }]);
      expect(result[0].data.tags).toEqual([{ key: 'tag1', value: 'tagvalue1' }]);
      expect(result[0].inputs?.datasetInputs).toHaveLength(1);
    });

    it('should not mutate original input', () => {
      const runs = [createMockRun({ runUuid: 'parent' }), createMockRun({ runUuid: 'child', parentRunId: 'parent' })];
      const nested = getNestedRuns(runs);
      const originalNestedClone = JSON.parse(JSON.stringify(nested));

      flattenRunEntityOrGroupData(nested);

      expect(nested).toEqual(originalNestedClone);
    });

    it('should handle runs with only group headers (no actual runs)', () => {
      // Edge case: group structure exists but no actual runs
      const emptyGroup: ExperimentEvaluationRunsGroupData = {
        groupKey: 'empty-group',
        groupValues: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'test' }],
        subRuns: [],
      };
      const result = flattenRunEntityOrGroupData([emptyGroup]);

      expect(result).toEqual([]);
    });
  });
});
