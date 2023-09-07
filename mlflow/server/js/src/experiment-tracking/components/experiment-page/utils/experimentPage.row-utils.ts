import { isNumber } from 'lodash';
import Utils from '../../../../common/utils/Utils';
import type {
  ExperimentEntity,
  KeyValueEntity,
  ModelVersionInfoEntity,
  RunInfoEntity,
  RunDatasetWithTags,
} from '../../../types';
import {
  RunRowDateAndNestInfo,
  RunRowModelsInfo,
  RunRowType,
  RunRowVersionInfo,
} from './experimentPage.row-types';
import { SearchExperimentRunsFacetsState } from '../models/SearchExperimentRunsFacetsState';
import { ExperimentRunsSelectorResult } from './experimentRuns.selector';
import {
  EXPERIMENT_FIELD_PREFIX_METRIC,
  EXPERIMENT_FIELD_PREFIX_PARAM,
  EXPERIMENT_FIELD_PREFIX_TAG,
  EXPERIMENT_PARENT_ID_TAG,
} from './experimentPage.common-utils';

/**
 * A simple tree-like interface used in nested rows calculations.
 */
interface SimpleTreeNode {
  value: string;
  parent?: SimpleTreeNode;
}

/**
 * An intermediate interface representing single row in agGrid (but not necessarily
 * a single run - these might be nested and not expanded). Is created from the data
 * originating from the store, then after enriching with metrics, params, attributed etc.
 * is being transformed to RunRowType which serves as a final agGrid compatible type.
 */
interface RowRenderMetadata {
  index: number;
  isParent?: boolean;
  hasExpander?: boolean;
  expanderOpen?: boolean;
  isPinnable?: boolean;
  runInfo: RunInfoEntity;
  level: number;
  childrenIds?: string[];
  params: KeyValueEntity[];
  metrics: KeyValueEntity[];
  tags: Record<string, KeyValueEntity>;
  datasets: RunDatasetWithTags[];
}

/**
 * For a given run dataset from the store, this function prepares
 * a list of rows metadata discarding any information about the parent/child run hierarchy.
 */
const getFlatRowRenderMetadata = (runData: SingleRunData[]) =>
  runData.map<RowRenderMetadata>(
    ({ runInfo, metrics = [], params = [], tags = {}, datasets = [] }, index) => ({
      index,
      runInfo,
      level: 0, // All runs will be on "0" level here,
      isPinnable: !tags[EXPERIMENT_PARENT_ID_TAG]?.value,
      metrics: metrics,
      params: params,
      tags: tags,
      datasets: datasets,
    }),
  );

/**
 * For a given run dataset from the store, this function prepares
 * a list of rows metadata taking the parent/child run hierarchy into consideration.
 */
const getNestedRowRenderMetadata = ({
  runsExpanded,
  runData,
}: {
  runsExpanded: Record<string, boolean>;
  runData: SingleRunData[];
}) => {
  // First, map run IDs to their indices - will be helpful later on.
  const runIdToIndex: Record<string, number> = {};
  runData.forEach(({ runInfo }, index) => {
    runIdToIndex[runInfo.run_uuid] = index;
  });

  // Create a list of tree nodes for all run infos. Each leaf's value is the run UUID.
  const treeNodes: SimpleTreeNode[] = runData.map(({ runInfo }) => ({ value: runInfo.run_uuid }));

  // We're going to check if any hierarchy is detected in the run set. If not,
  // we will not bother with unnecessary calculations.
  let foundHierarchy = false;

  // Iterate through all the tags and assign proper parent references
  runData.forEach(({ tags }, index) => {
    const parentRunId = tags?.[EXPERIMENT_PARENT_ID_TAG];
    if (parentRunId) {
      const parentRunIndex = runIdToIndex[parentRunId.value];
      if (parentRunIndex !== undefined) {
        foundHierarchy = true;
        treeNodes[index].parent = treeNodes[parentRunIndex];
      }
    }
  });

  // If no parent tags are found, we're not going calculate
  // tree-related stuff and return a flat list instead.
  if (!foundHierarchy) {
    return getFlatRowRenderMetadata(runData);
  }

  // Iterate through the tree and convert it to a flat parent->children mapping array
  const parentIdToChildren: Record<string, number[]> = {};
  const rootIndexes: any[] = [];
  treeNodes.forEach((treeNode, index) => {
    const { parent } = treeNode;
    if (parent !== undefined && parent.value !== treeNode.value) {
      if (parentIdToChildren[parent.value]) {
        parentIdToChildren[parent.value].push(index);
      } else {
        parentIdToChildren[parent.value] = [index];
      }
    } else {
      // If a node has no parent, let's register it as a root index
      rootIndexes.push(index);
    }
  });

  const resultRowsMetadata: RowRenderMetadata[] = [];

  // Create and invoke a simple DFS search with "visited" set so we won't be caught in a cycle
  const visited = new Set();
  const doDfs = (dfsIndex: number, currLevel: number) => {
    if (!visited.has(dfsIndex)) {
      const currentNodeRunInfo = runData[dfsIndex].runInfo;
      const currentNodeRunId = currentNodeRunInfo.run_uuid;

      // Only rows that are top-level parents or those being on the top level are pinnable
      const isPinnable = Boolean(rootIndexes.includes(dfsIndex)) || currLevel === 0;

      const rowMetadata: RowRenderMetadata = {
        index: dfsIndex,
        isParent: false,
        hasExpander: false,
        level: currLevel,
        runInfo: currentNodeRunInfo,
        params: runData[dfsIndex].params || [],
        metrics: runData[dfsIndex].metrics || [],
        tags: runData[dfsIndex].tags || {},
        datasets: runData[dfsIndex].datasets || [],
        isPinnable,
      };
      if (parentIdToChildren[currentNodeRunId]) {
        rowMetadata.isParent = true;
        rowMetadata.hasExpander = true;
        rowMetadata.expanderOpen = Boolean(runsExpanded[currentNodeRunId]);
        rowMetadata.childrenIds = parentIdToChildren[currentNodeRunId].map(
          (cIdx) => runData[cIdx].runInfo.run_uuid,
        );
      }

      resultRowsMetadata.push(rowMetadata);
      visited.add(dfsIndex);

      const childrenIndices = parentIdToChildren[currentNodeRunId];
      // Repeat DFS for children nodes - only if the current node is expanded
      if (childrenIndices) {
        if (runsExpanded[currentNodeRunId]) {
          childrenIndices.forEach((dIdx) => {
            doDfs(dIdx, currLevel + 1);
          });
        }
      }
    }
  };

  // Invoke the DFS for all root indexes
  rootIndexes.forEach((rootNodeIndex) => {
    doDfs(rootNodeIndex, 0);
  });
  return resultRowsMetadata;
};

/**
 * Iterates through all key/value data given for a run and
 * returns mapped dataset in a "PREFIX-NAME" form, e.g. '$$$param$$$-paramname".
 * Fills '-' placeholder in all empty places.
 */
const createKeyValueDataForRunRow = (
  list: { key: string; value: string }[],
  keys: string[],
  prefix: string,
) => {
  if (!list) {
    return [];
  }

  const map: Record<string, string> = {};

  // First, populate all values (cells) with default placeholder: '-'
  for (const key of keys) {
    map[`${prefix}-${key}`] = '-';
  }

  // Then, override with existing value if found
  for (const { key, value } of list) {
    if (value || isNumber(value)) {
      map[`${prefix}-${key}`] = value;
    }
  }

  return map;
};

/**
 * Temporary function that assigns randomized, yet stable color
 * from the static palette basing on an input string. Used for coloring runs.
 *
 * TODO: make a decision on the final color hashing per run
 */
const getStableColorByStringHash = (data: string) => {
  // Taken from Figma design
  const colors = [
    '#077A9D',
    '#8BCAE7',
    '#FFAB00',
    '#FFDB96',
    '#00A972',
    '#99DDB4',
    '#BA7B23',
    '#FF3621',
    '#FCA4A1',
    '#919191',
    '#00875C',
    '#1B5162',
    '#914B9F',
    '#D01F0B',
    '#BD89C7',
    '#AB4057',
    '#5F5F5F',
    '#BF7080',
    '#C2C2C2',
    '#7F1035',
  ];
  let a = 0,
    b = 0;

  // Let's use super simple hashing method
  for (let i = 0; i < data.length; i++) {
    a = (a + data.charCodeAt(i)) % 255;
    b = (b + a) % 255;
  }

  // eslint-disable-next-line no-bitwise
  return colors[(a | (b << 8)) % colors.length];
};

/**
 * Creates ag-grid compatible row dataset for all given runs basing on
 * the data retrieved from the API and from the refux store.
 * Please refer to PrepareRunsGridDataParams type for type reference.
 */
export const prepareRunsGridData = ({
  experiments,
  modelVersionsByRunUuid,
  runsExpanded,
  nestChildren,
  referenceTime,
  paramKeyList,
  metricKeyList,
  tagKeyList,
  runsPinned,
  runsHidden,
  runData,
  runUuidsMatchingFilter,
}: PrepareRunsGridDataParams) => {
  const experimentNameMap = Utils.getExperimentNameMap(
    Utils.sortExperimentsById(experiments),
  ) as Record<string, { name: string; basename: string }>;

  // Let's start with generating intermediate row metadata - either as a nested or a flat list.
  // We need to assemble separate hierarchies for pinned rows and unpinned rows:
  const rowRenderMetadata: RowRenderMetadata[] = nestChildren
    ? getNestedRowRenderMetadata({ runData, runsExpanded })
    : getFlatRowRenderMetadata(runData);

  // We will aggregate children of pinned parent rows here so we will easily pin them as well
  const childrenToPin: string[] = [];

  // Now, enrich the intermediate row metadata with attributes, metrics and params and
  // return it as a grid-consumable "RunRowType" type.
  const runs = rowRenderMetadata.map<RunRowType>((runInfoMetadata) => {
    const {
      runInfo,
      isParent = false,
      hasExpander = false,
      level = 0,
      expanderOpen = false,
      isPinnable = false,
      childrenIds = [],
      tags,
      params,
      metrics,
      datasets,
    } = runInfoMetadata;

    const formattedMetrics = (metrics || []).map(({ key, value }) => ({
      key,
      value: Utils.formatMetric(value),
    }));

    // Extract necessary basic info
    const runUuid = runInfo.run_uuid;
    const { experiment_id: experimentId } = runInfo;
    const experimentName = experimentNameMap[experimentId];
    const user = Utils.getUser(runInfo, tags);
    const duration = Utils.getDuration(runInfo.start_time, runInfo.end_time);
    const runName = Utils.getRunName(runInfo) || runInfo.run_uuid;

    // Extract visible tags (i.e. those not prefixed with "mlflow.")
    const visibleTags = Utils.getVisibleTagValues(tags).map(([key, value]) => ({
      key,
      value,
    }));

    // Prepare a data package to be used by "Start time" cell
    const runDateAndNestInfo: RunRowDateAndNestInfo = {
      startTime: runInfo.start_time,
      referenceTime,
      experimentId,
      runUuid,
      runStatus: runInfo.status,
      isParent,
      hasExpander,
      expanderOpen,
      childrenIds,
      level,
    };

    // Prepare a data package to be used by "Models" cell
    const models: RunRowModelsInfo = {
      registeredModels: modelVersionsByRunUuid[runInfo.run_uuid] || [], // ModelInfoEntity
      // @ts-expect-error TS(2322): Type 'unknown[]' is not assignable to type '{ arti... Remove this comment to see the full error message
      loggedModels: Utils.getLoggedModelsFromTags(tags),
      experimentId: runInfo.experiment_id,
      runUuid: runInfo.run_uuid,
    };

    // Prepare a data package to be used by "Version" cell
    const version: RunRowVersionInfo = {
      version: Utils.getSourceVersion(tags),
      name: Utils.getSourceName(tags),
      type: Utils.getSourceType(tags),
    };

    const isCurrentRowHidden = runsHidden.includes(runUuid);
    const isCurrentRowPinned = isPinnable && runsPinned.includes(runUuid);
    const isParentPinned = childrenToPin.includes(runUuid);

    // If this or a parent row is pinned, pin children as well
    if (isCurrentRowPinned || isParentPinned) {
      childrenToPin.push(...childrenIds);
    }

    // Compile everything into a data object to be consumed by the grid component
    return {
      runUuid,
      runDateAndNestInfo,
      runInfo,
      experimentName,
      experimentId,
      duration,
      user,
      runName,
      tags,
      models,
      params,
      version,
      pinnable: isPinnable,
      color: getStableColorByStringHash(runUuid),
      hidden: isCurrentRowHidden,
      pinned: isCurrentRowPinned || isParentPinned,
      ...createKeyValueDataForRunRow(params, paramKeyList, EXPERIMENT_FIELD_PREFIX_PARAM),
      ...createKeyValueDataForRunRow(
        formattedMetrics,
        metricKeyList,
        EXPERIMENT_FIELD_PREFIX_METRIC,
      ),
      datasets,
      ...createKeyValueDataForRunRow(visibleTags, tagKeyList, EXPERIMENT_FIELD_PREFIX_TAG),
    };
  });

  // If the flat structure is displayed, we can hoist pinned rows to the top
  return [
    // Add pinned rows to the top
    ...runs.filter(({ pinned }) => pinned),

    // Next, add all remaining rows - however, sweep out all runs that don't match the current filter. This
    // will hide all filtered out runs that were pinned before, but were eventually un-pinned.
    ...runs.filter(({ pinned, runUuid }) => !pinned && runUuidsMatchingFilter.includes(runUuid)),
  ];
};

export type SingleRunData = {
  runInfo: RunInfoEntity;
  params: KeyValueEntity[];
  metrics: KeyValueEntity[];
  datasets: RunDatasetWithTags[];
  tags: Record<string, KeyValueEntity>;
};

/**
 * All parameters necessary to calculate run row data.
 */
type PrepareRunsGridDataParams = Pick<
  ExperimentRunsSelectorResult,
  'metricKeyList' | 'paramKeyList' | 'modelVersionsByRunUuid'
> &
  Pick<SearchExperimentRunsFacetsState, 'runsExpanded' | 'runsPinned' | 'runsHidden'> & {
    /**
     * List of experiments containing the runs
     */
    experiments: ExperimentEntity[];

    /**
     * Registered model versions arrays per run uuid
     */
    modelVersionsByRunUuid: Record<string, ModelVersionInfoEntity[]>;

    /**
     * Boolean flag indicating if hierarchical runs should be generated
     */
    nestChildren: boolean;

    /**
     * List of all visible tag keys
     */
    tagKeyList: string[];

    /**
     * A reference time necessary to calculate "xxx minutes ago"-like labels
     */
    referenceTime: Date;

    /**
     * List of simplified run objects containing all relevant data
     */
    runData: SingleRunData[];

    /**
     * List of all runs IDs that match the current filter
     * (this excludes all rows that on the list just because they are pinned)
     */
    runUuidsMatchingFilter: string[];
  };

export const extractRunRowParamFloat = (
  run: RunRowType,
  paramName: string,
  fallback = undefined,
) => {
  const paramEntity = extractRunRowParam(run, paramName);
  if (!paramEntity) {
    return fallback;
  }
  return parseFloat(paramEntity) || fallback;
};

export const extractRunRowParamInteger = (
  run: RunRowType,
  paramName: string,
  fallback = undefined,
) => {
  const paramEntity = extractRunRowParam(run, paramName);
  if (!paramEntity) {
    return fallback;
  }
  return parseInt(paramEntity, 10) || fallback;
};

export const extractRunRowParam = (run: RunRowType, paramName: string, fallback = undefined) => {
  const paramEntity = run.params.find(({ key }) => paramName === key);
  return paramEntity?.value || fallback;
};
