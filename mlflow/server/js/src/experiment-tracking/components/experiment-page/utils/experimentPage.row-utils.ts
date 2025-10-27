import { useMemo } from 'react';
import { isNumber, isString, keyBy, last, sortBy, uniq } from 'lodash';
import Utils from '../../../../common/utils/Utils';
import type {
  ExperimentEntity,
  ModelVersionInfoEntity,
  RunInfoEntity,
  RunDatasetWithTags,
  MetricEntity,
} from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { LoggedModelProto } from '../../../types';
import type {
  RowGroupRenderMetadata,
  RowRenderMetadata,
  RunGroupParentInfo,
  RunRowDateAndNestInfo,
  RunRowModelsInfo,
  RunRowType,
  RunRowVersionInfo,
} from './experimentPage.row-types';
import { RunGroupingAggregateFunction, RunGroupingMode } from './experimentPage.row-types';
import type { ExperimentRunsSelectorResult } from './experimentRuns.selector';
import {
  EXPERIMENT_FIELD_PREFIX_METRIC,
  EXPERIMENT_FIELD_PREFIX_PARAM,
  EXPERIMENT_FIELD_PREFIX_TAG,
  EXPERIMENT_PARENT_ID_TAG,
} from './experimentPage.common-utils';
import { getStableColorForRun } from '../../../utils/RunNameUtils';
import { shouldEnableToggleIndividualRunsInGroups } from '../../../../common/utils/FeatureUtils';
import { getGroupedRowRenderMetadata, type RunsGroupByConfig } from './experimentPage.group-row-utils';
import { type ExperimentPageUIState, RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';
import { determineIfRowIsHidden } from './experimentPage.common-row-utils';
import { type ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';

/**
 * A simple tree-like interface used in nested rows calculations.
 */
interface SimpleTreeNode {
  value: string;
  parent?: SimpleTreeNode;
}

/**
 * For a given run dataset from the store, this function prepares
 * a list of rows metadata discarding any information about the parent/child run hierarchy.
 */
const getFlatRowRenderMetadata = (runData: SingleRunData[]) =>
  runData.map<RowRenderMetadata>(({ runInfo, metrics = [], params = [], tags = {}, datasets = [] }, index) => ({
    index,
    runInfo,
    level: 0, // All runs will be on "0" level here,
    isPinnable: !tags[EXPERIMENT_PARENT_ID_TAG]?.value,
    metrics: metrics,
    params: params,
    tags: tags,
    datasets: datasets,
    rowUuid: runInfo.runUuid,
  }));

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
    runIdToIndex[runInfo.runUuid] = index;
  });

  // Create a list of tree nodes for all run infos. Each leaf's value is the run UUID.
  const treeNodes: SimpleTreeNode[] = runData.map(({ runInfo }) => ({ value: runInfo.runUuid }));

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
      const currentNodeRunId = currentNodeRunInfo.runUuid;

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
        rowUuid: currentNodeRunId,
      };
      if (parentIdToChildren[currentNodeRunId]) {
        rowMetadata.isParent = true;
        rowMetadata.hasExpander = true;
        rowMetadata.expanderOpen = Boolean(runsExpanded[currentNodeRunId]);
        rowMetadata.childrenIds = parentIdToChildren[currentNodeRunId].map((cIdx) => runData[cIdx].runInfo.runUuid);
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
  list: { key: string; value: string | number }[],
  keys: (string | number)[],
  prefix: string,
) => {
  if (!list) {
    return {};
  }

  const map: Record<string, string | number> = {};

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
 * Creates ag-grid compatible row dataset for all given runs basing on
 * the data retrieved from the API and from the refux store.
 * Please refer to PrepareRunsGridDataParams type for type reference.
 */
export const prepareRunsGridData = ({
  experiments,
  modelVersionsByRunUuid,
  loggedModelsV3ByRunUuid = {},
  runsExpanded,
  nestChildren,
  referenceTime,
  paramKeyList,
  metricKeyList,
  tagKeyList,
  runsPinned,
  runsHidden,
  runsHiddenMode = RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
  runsVisibilityMap = {},
  runData,
  runUuidsMatchingFilter,
  groupBy = null,
  groupsExpanded = {},
  useGroupedValuesInCharts,
  searchFacetsState,
}: PrepareRunsGridDataParams) => {
  const experimentNameMap = Utils.getExperimentNameMap(Utils.sortExperimentsById(experiments)) as Record<
    string,
    { name: string; basename: string }
  >;

  // Gate grouping by the feature flag
  const shouldGroupRows = Boolean(groupBy);

  // Early returning function that will return relevant row render metadata depending on a determined mode.
  // It can be either grouped rows, nested rows (parent/child) or flat rows.
  const getRowRenderMetadata = (): (RowRenderMetadata | RowGroupRenderMetadata)[] => {
    // If grouping is enabled and configured, we will return grouped rows
    if (shouldGroupRows) {
      const groupedRows = getGroupedRowRenderMetadata({
        runData,
        groupBy,
        groupsExpanded,
        searchFacetsState,
        runsHidden,
        runsVisibilityMap,
        runsHiddenMode,
        useGroupedValuesInCharts,
      });

      if (groupedRows) {
        return groupedRows;
      }
    }

    // If nesting is enabled, we will return nested rows
    if (nestChildren) {
      return getNestedRowRenderMetadata({
        runData,
        runsExpanded,
      });
    }

    // Otherwise, we will return flat list of rows
    return getFlatRowRenderMetadata(runData);
  };

  // We will aggregate children of pinned parent rows here so we will easily pin them as well
  const childrenToPin: string[] = [];

  // Now, enrich the intermediate row metadata with attributes, metrics and params and
  // return it as a grid-consumable "RunRowType" type.
  const rows = getRowRenderMetadata().map<RunRowType>((runOrGroupInfoMetadata) => {
    // If the row is a group parent, we will create a special row for it
    if (runOrGroupInfoMetadata.isGroup) {
      return createGroupParentRow(runOrGroupInfoMetadata, runsPinned, metricKeyList, paramKeyList);
    }

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
      belongsToGroup,
      rowUuid,
      visibilityControl,
      hidden,
    } = runOrGroupInfoMetadata;

    // Extract necessary basic info
    const runUuid = runInfo.runUuid;
    const { experimentId } = runInfo;
    const experimentName = experimentNameMap[experimentId];
    const user = Utils.getUser(runInfo, tags);
    const duration = Utils.getDuration(runInfo.startTime, runInfo.endTime);
    const runName = Utils.getRunName(runInfo) || runInfo.runUuid;

    // Extract visible tags (i.e. those not prefixed with "mlflow.")
    const visibleTags = Utils.getVisibleTagValues(tags).map(([key, value]) => ({
      key,
      value,
    }));

    // Prepare a data package to be used by "Start time" cell
    const runDateAndNestInfo: RunRowDateAndNestInfo = {
      startTime: runInfo.startTime,
      referenceTime,
      experimentId,
      runUuid,
      runStatus: runInfo.status,
      isParent,
      hasExpander,
      expanderOpen,
      childrenIds,
      level,
      belongsToGroup: Boolean(belongsToGroup),
    };

    // Prepare a data package to be used by "Models" cell
    const models: RunRowModelsInfo = {
      registeredModels: modelVersionsByRunUuid[runInfo.runUuid] || [], // ModelInfoEntity
      loggedModels: Utils.getLoggedModelsFromTags(tags),
      loggedModelsV3: loggedModelsV3ByRunUuid[runInfo.runUuid] || [],
      experimentId: runInfo.experimentId,
      runUuid: runInfo.runUuid,
    };

    // Prepare a data package to be used by "Version" cell
    const version: RunRowVersionInfo = {
      version: Utils.getSourceVersion(tags),
      name: Utils.getSourceName(tags),
      type: Utils.getSourceType(tags),
    };

    const isCurrentRowPinned = isPinnable && runsPinned.includes(runUuid);
    const isParentPinned = childrenToPin.includes(runUuid);

    // If this or a parent row is pinned, pin children as well
    if (isCurrentRowPinned || isParentPinned) {
      childrenToPin.push(...childrenIds);
    }

    // Compile everything into a data object to be consumed by the grid component
    return {
      runUuid,
      rowUuid,
      runDateAndNestInfo,
      runInfo,
      experimentName,
      experimentId,
      duration,
      user,
      runName,
      runStatus: runInfo.status,
      tags,
      models,
      params,
      version,
      pinnable: isPinnable,
      defaultColor: getStableColorForRun(runUuid),
      visibilityControl,
      hidden: hidden ?? false,
      pinned: isCurrentRowPinned || isParentPinned,
      ...createKeyValueDataForRunRow(params, paramKeyList, EXPERIMENT_FIELD_PREFIX_PARAM),
      ...createKeyValueDataForRunRow(metrics, metricKeyList, EXPERIMENT_FIELD_PREFIX_METRIC),
      datasets,
      ...createKeyValueDataForRunRow(visibleTags, tagKeyList, EXPERIMENT_FIELD_PREFIX_TAG),
    };
  });

  // If grouping is enabled, we need to group rows into chunks and hoist pinned rows within them
  if (shouldGroupRows && rows.some((row) => row.groupParentInfo)) {
    const chunks = rows.reduce<RunRowType[][]>((chunkContainer, run) => {
      if (run.groupParentInfo) {
        chunkContainer.push([]);
      }
      last(chunkContainer)?.push(run);
      return chunkContainer;
    }, []);

    const sortedChunks = sortBy(
      chunks,
      (chunk) => chunk[0]?.groupParentInfo && !runsPinned.includes(chunk[0]?.groupParentInfo?.groupId),
    );

    const chunksWithSortedRuns = sortedChunks.map((chunkRuns) => {
      const [groupHeader, ...runs] = chunkRuns;
      return [groupHeader, ...hoistPinnedRuns(runs, runUuidsMatchingFilter)];
    });

    const groupedRuns = chunksWithSortedRuns.flat();

    // If toggling individual runs in groups is enabled, hidden flags
    // should be already contained in the array so we can return it right away
    if (shouldEnableToggleIndividualRunsInGroups()) {
      return groupedRuns;
    }
    return determineVisibleRuns(groupedRuns, runsHidden, runsHiddenMode, runsVisibilityMap);
  }

  // If the flat structure is displayed, we can hoist pinned rows to the top
  return determineVisibleRuns(
    hoistPinnedRuns(rows, runUuidsMatchingFilter),
    runsHidden,
    runsHiddenMode,
    runsVisibilityMap,
  );
};

// Hook version of prepareRunsGridData()
export const useExperimentRunRows = ({
  experiments,
  modelVersionsByRunUuid,
  loggedModelsV3ByRunUuid,
  runsExpanded,
  nestChildren,
  referenceTime,
  paramKeyList,
  metricKeyList,
  tagKeyList,
  runsPinned,
  runsHidden,
  runsVisibilityMap,
  runData,
  runUuidsMatchingFilter,
  groupBy,
  runsHiddenMode,
  groupsExpanded = {},
  useGroupedValuesInCharts,
  searchFacetsState,
}: PrepareRunsGridDataParams) =>
  useMemo(
    () =>
      prepareRunsGridData({
        experiments,
        modelVersionsByRunUuid,
        loggedModelsV3ByRunUuid,
        runsExpanded,
        nestChildren,
        referenceTime,
        paramKeyList,
        metricKeyList,
        tagKeyList,
        runsPinned,
        runsHidden,
        runsVisibilityMap,
        runData,
        runUuidsMatchingFilter,
        groupBy,
        groupsExpanded,
        runsHiddenMode,
        useGroupedValuesInCharts,
        searchFacetsState,
      }),
    [
      // Explicitly include each dependency here to avoid unnecessary recalculations
      experiments,
      modelVersionsByRunUuid,
      loggedModelsV3ByRunUuid,
      runsExpanded,
      nestChildren,
      referenceTime,
      paramKeyList,
      metricKeyList,
      tagKeyList,
      runsPinned,
      runsHidden,
      runsVisibilityMap,
      runData,
      runUuidsMatchingFilter,
      groupBy,
      groupsExpanded,
      runsHiddenMode,
      useGroupedValuesInCharts,
      searchFacetsState,
    ],
  );

const determineVisibleRuns = (
  runs: RunRowType[],
  runsHidden: string[],
  runsHiddenMode: RUNS_VISIBILITY_MODE,
  runsVisibilityMap: Record<string, boolean> = {},
): RunRowType[] => {
  // In the new visibility model, we will count rows that can change visibility (groups and ungrouped runs)
  // and use the counter to limit the visible rows based on selected mode
  let visibleRowCounter = 0;
  return runs.map((runRow) => {
    // If a row is a run group, we use its rowUuid for setting visibility.
    // If this is a run, use runUuid.
    const runUuidToToggle = runRow.groupParentInfo ? runRow.rowUuid : runRow.runUuid;

    const rowWithHiddenFlag = {
      ...runRow,
      hidden: determineIfRowIsHidden(
        runsHiddenMode,
        runsHidden,
        runUuidToToggle,
        visibleRowCounter,
        runsVisibilityMap,
        runRow.runStatus,
      ),
    };

    const isGroupContainingRuns = runRow.groupParentInfo && !runRow.groupParentInfo.isRemainingRunsGroup;
    const isUngroupedRun = !runRow.runDateAndNestInfo?.belongsToGroup;
    if (isGroupContainingRuns || isUngroupedRun) {
      visibleRowCounter++;
    }

    return rowWithHiddenFlag;
  });
};

const hoistPinnedRuns = (rowCollection: RunRowType[], uuidWhitelist: string[]) => [
  // Add pinned rows to the top
  ...rowCollection.filter(({ pinned }) => pinned),

  // Next, add all remaining rows - however, sweep out all runs that don't match the current filter. This
  // will hide all filtered out runs that were pinned before, but were eventually un-pinned.
  ...rowCollection.filter(({ pinned, runUuid }) => runUuid && !pinned && uuidWhitelist.includes(runUuid)),
];

export type SingleRunData = {
  runInfo: RunInfoEntity;
  params: KeyValueEntity[];
  metrics: MetricEntity[];
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
  Pick<
    ExperimentPageUIState,
    'runsExpanded' | 'runsPinned' | 'runsHidden' | 'useGroupedValuesInCharts' | 'runsVisibilityMap'
  > &
  Partial<Pick<ExperimentPageUIState, 'groupsExpanded' | 'runsHiddenMode'>> & {
    /**
     * List of experiments containing the runs
     */
    experiments: ExperimentEntity[];

    /**
     * Registered model versions arrays per run uuid
     */
    modelVersionsByRunUuid: Record<string, ModelVersionInfoEntity[]>;

    /**
     * Logged models v3 data
     */
    loggedModelsV3ByRunUuid?: Record<string, LoggedModelProto[]>;
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

    groupBy: string | RunsGroupByConfig | null;

    searchFacetsState?: Readonly<ExperimentPageSearchFacetsState>;
  };

export const extractRunRowParamFloat = (run: RunRowType, paramName: string, fallback = undefined) => {
  const paramEntity = extractRunRowParam(run, paramName);
  if (!paramEntity) {
    return fallback;
  }

  const parsed = parseFloat(paramEntity);
  return isNaN(parsed) ? fallback : parsed;
};

export const extractRunRowParamInteger = (run: RunRowType, paramName: string, fallback = undefined) => {
  const paramEntity = extractRunRowParam(run, paramName);
  if (!paramEntity) {
    return fallback;
  }

  const parsed = parseInt(paramEntity, 10);
  return isNaN(parsed) ? fallback : parsed;
};

export const extractRunRowParam = (run: RunRowType, paramName: string, fallback = undefined) => {
  const paramEntity = run.params?.find(({ key }) => paramName === key);
  return paramEntity?.value || fallback;
};

/**
 * Creates a group parent row based on a run group render metadata object.
 */
const createGroupParentRow = (
  groupRunMetadata: RowGroupRenderMetadata,
  runsPinned: string[],
  metricKeyList: string[],
  paramKeyList: string[],
): RunRowType => {
  const groupParentInfo: RunGroupParentInfo = {
    groupingValues: groupRunMetadata.groupingValues,
    isRemainingRunsGroup: groupRunMetadata.isRemainingRunsGroup,
    expanderOpen: groupRunMetadata.expanderOpen,
    groupId: groupRunMetadata.groupId,
    aggregatedMetricData: keyBy(groupRunMetadata.aggregatedMetricEntities, 'key'),
    aggregatedParamData: keyBy(groupRunMetadata.aggregatedParamEntities, 'key'),
    runUuids: groupRunMetadata.runUuids,
    runUuidsForAggregation: groupRunMetadata.runUuidsForAggregation,
    aggregateFunction: groupRunMetadata.aggregateFunction,
    allRunsHidden: groupRunMetadata.allRunsHidden,
  };

  return {
    // Group parent rows have no run UUIDs set
    runUuid: '',
    datasets: [],
    pinnable: true,
    duration: null,
    models: null,
    rowUuid: groupParentInfo.groupId,
    groupParentInfo: groupParentInfo,
    pinned: runsPinned.includes(groupParentInfo.groupId),
    defaultColor: getStableColorForRun(groupParentInfo.groupId),
    visibilityControl: groupRunMetadata.visibilityControl,
    hidden: groupRunMetadata.hidden ?? false,
    ...createKeyValueDataForRunRow(
      groupRunMetadata.aggregatedMetricEntities,
      metricKeyList,
      EXPERIMENT_FIELD_PREFIX_METRIC,
    ),
    ...createKeyValueDataForRunRow(
      groupRunMetadata.aggregatedParamEntities,
      paramKeyList,
      EXPERIMENT_FIELD_PREFIX_PARAM,
    ),
  };
};
