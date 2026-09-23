import type { RunEntity, RunEntityWithChildren } from '../../types';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import type { RunGroupByGroupingValue } from '../../components/experiment-page/utils/experimentPage.row-types';
import { RunGroupingMode } from '../../components/experiment-page/utils/experimentPage.row-types';
import { getParentRunTagName } from '../../actions';

export type ExperimentEvaluationRunsGroupData = {
  groupKey: string;
  groupValues: RunGroupByGroupingValue[];
  subRuns: RunEntity[];
};

export type RunEntityOrGroupData = RunEntity | RunEntityWithChildren | ExperimentEvaluationRunsGroupData;

// string key for easy access in the map object
const createGroupKey = (groupData: RunGroupByGroupingValue) => {
  if (groupData.mode === RunGroupingMode.Dataset) {
    return `Dataset: ${groupData.value}`;
  } else {
    return `${groupData.groupByData} (${groupData.mode}): ${groupData.value}`;
  }
};

const getGroupValues = (run: RunEntity, groupBy: RunsGroupByConfig): RunGroupByGroupingValue[] => {
  const groupByKeys = groupBy.groupByKeys;

  const values: RunGroupByGroupingValue[] = [];

  for (const groupByKey of groupByKeys) {
    switch (groupByKey.mode) {
      case RunGroupingMode.Dataset:
        values.push({
          mode: RunGroupingMode.Dataset,
          groupByData: 'dataset',
          // in genai evaluate, it's not possible to have multiple dataset inputs,
          // so we can just use the first one. however, this logic will need
          // to be updated if we support multiple dataset inputs in the future
          value: run.inputs?.datasetInputs?.[0]?.dataset?.digest ?? null,
        });
        break;
      case RunGroupingMode.Param:
        const param = run.data?.params?.find((p) => p.key === groupByKey.groupByData);
        values.push({
          mode: RunGroupingMode.Param,
          groupByData: groupByKey.groupByData,
          value: param?.value ?? null,
        });
        break;
      case RunGroupingMode.Tag:
        const tag = run.data?.tags?.find((t) => t.key === groupByKey.groupByData);
        values.push({
          mode: RunGroupingMode.Tag,
          groupByData: groupByKey.groupByData,
          value: tag?.value ?? null,
        });
        break;
      default:
        break;
    }
  }

  return values;
};

export const getGroupByRunsData = (runs: RunEntity[], groupBy: RunsGroupByConfig | null): RunEntityOrGroupData[] => {
  if (!groupBy) {
    return runs;
  }

  const runGroupsMap: Record<
    string,
    {
      groupValues: RunGroupByGroupingValue[];
      subRuns: RunEntity[];
    }
  > = {};

  for (const run of runs) {
    const groupValues = getGroupValues(run, groupBy);
    const groupKey = groupValues.map(createGroupKey).join(', ');
    if (!runGroupsMap[groupKey]) {
      runGroupsMap[groupKey] = {
        groupValues,
        subRuns: [],
      };
    }
    runGroupsMap[groupKey].subRuns.push(run);
  }

  const runsWithGroupValues: RunEntityOrGroupData[] = [];
  Object.entries(runGroupsMap).forEach(([groupKey, { groupValues, subRuns }]) => {
    const groupHeadingRow: RunEntityOrGroupData = {
      groupKey,
      groupValues,
      subRuns,
    };
    runsWithGroupValues.push(groupHeadingRow);
  });

  return runsWithGroupValues;
};

/**
 * Nests child runs under their parent runs.
 * Returns array with only top-level runs, children nested in 'children' property.
 */
export const getNestedRuns = (runs: RunEntity[]): RunEntityWithChildren[] => {
  const parentToChildren: Record<string, RunEntity[]> = {};
  const runToParentId: Record<string, string | null> = {};

  runs.forEach((run) => {
    const runId = run.info.runUuid;
    const parentId = run.data?.tags?.find((t) => t.key === getParentRunTagName())?.value ?? null;
    runToParentId[runId] = parentId;

    if (parentId) {
      if (!parentToChildren[parentId]) {
        parentToChildren[parentId] = [];
      }
      parentToChildren[parentId].push(run);
    }
  });

  const visited = new Set<string>();

  const nestChildren = (run: RunEntity): RunEntityWithChildren => {
    const runUuid = run.info.runUuid;

    if (visited.has(runUuid)) {
      return run;
    }

    visited.add(runUuid);

    const children = parentToChildren[runUuid];
    if (!children) {
      return run;
    }
    return {
      ...run,
      children: children.map((child) => nestChildren(child)),
    };
  };

  // Hide child runs until their parent has been fetched so that they are never rendered as roots.
  const rootRuns = runs.filter((run) => !runToParentId[run.info.runUuid]);

  return rootRuns.map((run) => nestChildren(run));
};

/**
 * Flattens RunEntityOrGroupData[] into a flat array of RunEntity objects.
 * Recursively extracts all actual run entities from groups and nested children.
 */
export const flattenRunEntityOrGroupData = (runs: RunEntityOrGroupData[]): RunEntity[] => {
  const hasHierarchy = runs.some((row) => 'subRuns' in row || 'children' in row);
  if (!hasHierarchy) {
    return runs as RunEntity[];
  }

  const flatRuns: RunEntity[] = [];

  const walkRows = (rows: RunEntityOrGroupData[]) => {
    rows.forEach((row) => {
      // Extract actual run entities (has 'info' property)
      if ('info' in row) {
        flatRuns.push(row);
      }

      // Recurse into subRuns (for grouped data)
      if ('subRuns' in row && row.subRuns) {
        walkRows(row.subRuns);
      }

      // Recurse into children (for nested parent-child runs)
      if ('children' in row && row.children) {
        walkRows(row.children);
      }
    });
  };

  walkRows(runs);
  return flatRuns;
};
