import invariant from 'invariant';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Global, keyframes } from '@emotion/react';
import { useExperimentEvaluationRunsData } from '../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { ExperimentEvaluationRunsPageWrapper } from './ExperimentEvaluationRunsPageWrapper';
import { ExperimentEvaluationRunsTable } from './ExperimentEvaluationRunsTable';
import type { OnChangeFn, RowSelectionState } from '@tanstack/react-table';
import { useNavigate, useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import {
  Button,
  CalendarRangeIcon,
  Empty,
  FilterIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ResizableBox } from 'react-resizable';
import { ExperimentViewRunsTableResizerHandle } from '../../components/experiment-page/components/runs/ExperimentViewRunsTableResizer';
import { RunViewEvaluationsTab } from '../../components/evaluations/RunViewEvaluationsTab';
import { ExperimentEvaluationRunsTableControls } from './ExperimentEvaluationRunsTableControls';
import evalRunsEmptyImg from '@mlflow/mlflow/src/common/static/eval-runs-empty.svg';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import type { DatasetWithRunType } from '../../components/experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { ExperimentViewDatasetDrawer } from '../../components/experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { compact, keyBy, mapValues, uniq, xor, xorBy } from 'lodash';
import {
  EvalRunsTableKeyedColumnPrefix,
  getEvalRunsBaseSelectionState,
} from './ExperimentEvaluationRunsTable.constants';
import { invalidateMlflowSearchTracesCache } from '@databricks/web-shared/genai-traces-table';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { FormattedMessage } from 'react-intl';
import {
  useSelectedRunUuid,
  SELECTED_RUN_UUID_QUERY_PARAM,
} from '../../components/evaluations/hooks/useSelectedRunUuid';
import {
  useCompareToRunUuid,
  COMPARE_TO_RUN_UUID_QUERY_PARAM,
} from '../../components/evaluations/hooks/useCompareToRunUuid';
import { EvalRunsEmptyStateCard } from './EvalRunsEmptyStateCard';
import { isUserFacingTag } from '../../../common/utils/TagUtils';
import {
  createEvalRunsTableKeyedColumnKey,
  parseEvalRunsTableKeyedColumnKey,
} from './ExperimentEvaluationRunsTable.utils';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import {
  RunGroupingAggregateFunction,
  RunGroupingMode,
} from '../../components/experiment-page/utils/experimentPage.row-types';
import { flattenRunEntityOrGroupData, getGroupByRunsData, getNestedRuns } from './ExperimentEvaluationRunsPage.utils';
import {
  ExperimentEvaluationRunsPageMode,
  useExperimentEvaluationRunsPageMode,
} from './hooks/useExperimentEvaluationRunsPageMode';
import { ExperimentEvaluationRunsPageCharts } from './charts/ExperimentEvaluationRunsPageCharts';
import { ExperimentEvaluationRunsRowVisibilityProvider } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { useEvalRunsGrouping, useEvalRunsSorting } from './hooks/useEvalRunsTableUrlState';
import { EvalRunsRecentRunsChart } from './charts/EvalRunsRecentRunsChart';
import { EvalRunsFilterCount } from './EvalRunsFilterCount';
import type { EvalRunsFilterButtonHandle } from './EvalRunsFilterButton';
import { ExperimentEvaluationRunsTableControlsV2 } from './ExperimentEvaluationRunsTableControlsV2';
import { EvalRunsDatePreset, useEvalRunsFilters } from './hooks/useEvalRunsFilters';
import { useEvalRunsUnfilteredCount } from './hooks/useEvalRunsUnfilteredCount';
import type { EvalRunsBaselineState } from './hooks/useEvalRunsBaseline';
import { isMatchingEvalRunsBaseline, useEvalRunsBaselineState } from './hooks/useEvalRunsBaseline';
import type { RunEntity } from '../../types';
import { getLocalStorageItem, setLocalStorageItem } from '@databricks/web-shared/hooks';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { useRegisterSelectedIds } from '@mlflow/mlflow/src/assistant';
import {
  shouldEnableImprovedEvalRunsComparison,
  shouldEnableNewEvalRunsUi,
  shouldShowEvalRunsIssuesPanel,
} from '../../../common/utils/FeatureUtils';

const DEFAULT_VISIBLE_METRIC_COLUMNS = 5;
const DEFAULT_EVAL_RUNS_GROUP_BY: RunsGroupByConfig = {
  aggregateFunction: RunGroupingAggregateFunction.Average,
  groupByKeys: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset' }],
};
const CHART_METRIC_SELECTION_STORAGE_VERSION = 1;
const getChartMetricSelectionStorageKey = (experimentId: string) =>
  `mlflow.eval-runs.chart-assessments.${experimentId}`;
const readChartMetricSelection = (experimentId: string): unknown =>
  getLocalStorageItem(
    getChartMetricSelectionStorageKey(experimentId),
    CHART_METRIC_SELECTION_STORAGE_VERSION,
    false,
    undefined,
  );
const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string');

// The view switch unmounts the outgoing layout, so a transition isn't possible; this softens the
// incoming charts pane.
const chartsPaneFadeIn = keyframes({
  from: { opacity: 0, transform: 'translateX(8px)' },
  to: { opacity: 1, transform: 'translateX(0)' },
});

const getLearnMoreLink = () => {
  return 'https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/';
};

// `baseline` is only passed with the new UI, so the flag-off page mounts no baseline queries.
const ExperimentEvaluationRunsPageImpl = ({ baseline }: { baseline?: EvalRunsBaselineState }) => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [tableWidth, setTableWidth] = useState(432);
  const [dragging, setDragging] = useState(false);
  const [runListHidden, setRunListHidden] = useState(false);
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType>();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const enableNewEvalRunsUi = shouldEnableNewEvalRunsUi();
  const [selectedColumns, setSelectedColumns] = useState<{ [key: string]: boolean }>(() =>
    getEvalRunsBaseSelectionState(enableNewEvalRunsUi),
  );

  const queryClient = useQueryClient();
  const enableImprovedComparison = shouldEnableImprovedEvalRunsComparison();
  const showIssuesPanelFlag = shouldShowEvalRunsIssuesPanel();

  // When flag is enabled, default to grouping by dataset
  const [groupBy, setGroupBy] = useEvalRunsGrouping(
    enableImprovedComparison ? DEFAULT_EVAL_RUNS_GROUP_BY : null,
    enableNewEvalRunsUi,
  );
  const [isComparisonMode, setIsComparisonMode] = useState(false);
  const [isViewingSelectedRun, setIsViewingSelectedRun] = useState(false);
  const { viewMode, setViewMode } = useExperimentEvaluationRunsPageMode();

  const [selectedRunUuid, setSelectedRunUuid] = useSelectedRunUuid();
  const [compareToRunUuid, setCompareToRunUuid] = useCompareToRunUuid();
  const [, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const [evalRunsSorting, setEvalRunsSorting] = useEvalRunsSorting(enableNewEvalRunsUi);

  invariant(experimentId, 'Experiment ID must be defined');

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const [filterChartToSelectedRuns, setFilterChartToSelectedRuns] = useState(false);
  const hasInitializedFromUrl = useRef(false);
  useRegisterSelectedIds('selectedRunIds', rowSelection);

  const filterButtonRef = useRef<EvalRunsFilterButtonHandle>(null);
  const handleFilterByColumn = useCallback((field: string) => filterButtonRef.current?.openForField(field), []);

  const {
    searchFilter,
    filters,
    filterString,
    isAnyFilterActive,
    hasRelativeDateWindow,
    refreshDateWindow,
    setSearchFilter,
    setDatePreset,
    setClauses,
  } = useEvalRunsFilters({
    experimentId,
    enabled: enableNewEvalRunsUi,
    persistDatePreset: enableNewEvalRunsUi,
  });

  const {
    data: runs,
    isLoading,
    isFetching,
    error,
    fetchNextPage,
    hasNextPage,
    refetch,
    dataUpdatedAt,
  } = useExperimentEvaluationRunsData({
    experimentId,
    enabled: true,
    filter: filterString,
    excludeIssueDetectionRuns: true,
  });

  // The filtered query cannot report the unfiltered total, so keep a first-page unfiltered query
  // for "showing N of M". Its `hasNextPage` marks the total as a lower bound ("M+").
  const {
    runs: unfilteredRuns,
    hasNextPage: unfilteredHasNextPage,
    isLoading: isLoadingUnfilteredRuns,
    error: unfilteredRunsError,
  } = useEvalRunsUnfilteredCount({ experimentId, enabled: enableNewEvalRunsUi && isAnyFilterActive });

  const baselineRunUuid = baseline?.baselineRunUuid;
  const fetchedBaselineRun = baseline?.fetchedBaselineRun;
  const refetchBaseline = baseline?.refetchBaseline;
  const refetchAll = useCallback(() => {
    // Moving a relative date window forward changes the query key, which fetches the runs on its own.
    if (hasRelativeDateWindow) {
      refreshDateWindow();
    } else {
      refetch();
    }
    refetchBaseline?.();
    invalidateMlflowSearchTracesCache({ queryClient });
  }, [hasRelativeDateWindow, refreshDateWindow, refetch, refetchBaseline, queryClient]);

  const runUuids = useMemo(() => runs?.map((run) => run.info.runUuid) ?? [], [runs]);
  // A baseline that the current filters exclude is fetched on its own, so it stays pinned.
  const baselineRun = useMemo<RunEntity | undefined>(() => {
    if (!baselineRunUuid) {
      return undefined;
    }
    const loadedRun = runs?.find((run) => run.info.runUuid === baselineRunUuid);
    if (loadedRun) {
      return loadedRun;
    }
    return isMatchingEvalRunsBaseline(fetchedBaselineRun, baselineRunUuid, experimentId)
      ? fetchedBaselineRun
      : undefined;
  }, [baselineRunUuid, experimentId, fetchedBaselineRun, runs]);

  // ORIGINAL BEHAVIOR (flag OFF): Auto-select first run when no run is selected or selected run is out of scope
  // This ensures the split view always has a run to display
  // Skip this when showIssuesPanelFlag is ON (we don't want to auto-select)
  if (
    !enableImprovedComparison &&
    !showIssuesPanelFlag &&
    runs?.length &&
    (!selectedRunUuid || !runUuids.includes(selectedRunUuid))
  ) {
    setSelectedRunUuid(runs[0].info.runUuid);
  }

  // Get selected run UUIDs from checkbox selection
  const selectedRunUuidsFromCheckbox = useMemo(
    () =>
      Object.entries(rowSelection)
        .filter(([_, value]) => value)
        .map(([key]) => key),
    [rowSelection],
  );
  const selectRunFromChart = useCallback(
    (runUuid: string) => {
      setRowSelection((current) => ({ ...current, [runUuid]: !current[runUuid] }));
      if (selectedRunUuidsFromCheckbox.length === 1 && selectedRunUuidsFromCheckbox[0] === runUuid) {
        setFilterChartToSelectedRuns(false);
      }
    },
    [selectedRunUuidsFromCheckbox],
  );
  const updateRowSelection: OnChangeFn<RowSelectionState> = (updater) => {
    const next = typeof updater === 'function' ? updater(rowSelection) : updater;
    setRowSelection(next);
    if (!Object.values(next).some(Boolean)) {
      setFilterChartToSelectedRuns(false);
    }
  };
  const setTableRowSelection = enableNewEvalRunsUi ? updateRowSelection : setRowSelection;

  // On mount, if URL has selectedRunUuid (and optionally compareToRunUuid), initialize rowSelection
  // Only enter comparison mode if BOTH params are present (indicating an active comparison)
  // If only selectedRunUuid is present, just pre-select the checkbox but stay in full-page list view
  // Only enabled when feature flag is on or issues panel flag is on
  useEffect(() => {
    if (!enableImprovedComparison && !showIssuesPanelFlag) {
      return;
    }
    // Only run initialization once when runs are loaded
    if (hasInitializedFromUrl.current || !runs?.length) {
      return;
    }
    hasInitializedFromUrl.current = true;

    // If URL has selectedRunUuid, initialize rowSelection
    if (selectedRunUuid && runUuids.includes(selectedRunUuid)) {
      const initialSelection: RowSelectionState = { [selectedRunUuid]: true };
      // Also include compareToRunUuid if present in URL
      if (compareToRunUuid && runUuids.includes(compareToRunUuid)) {
        initialSelection[compareToRunUuid] = true;
        // Only enter comparison mode if BOTH runs are in URL (active comparison)
        setIsComparisonMode(true);
      }
      setRowSelection(initialSelection);
    }
  }, [
    enableImprovedComparison,
    showIssuesPanelFlag,
    selectedRunUuid,
    compareToRunUuid,
    runs,
    runUuids,
    setIsComparisonMode,
  ]);

  // Sync URL params from checkbox selection when in comparison mode (as useEffect to avoid render-time state updates)
  // Only enabled when feature flag is on or issues panel flag is on
  useEffect(() => {
    if (!enableImprovedComparison && !showIssuesPanelFlag) {
      return;
    }
    // Skip syncing if we haven't finished initializing yet
    if (!isComparisonMode) {
      return;
    }

    if (selectedRunUuidsFromCheckbox.length === 0) {
      // No selection - exit comparison mode
      setIsComparisonMode(false);
      setSelectedRunUuid(undefined);
      setCompareToRunUuid(undefined);
    } else if (selectedRunUuidsFromCheckbox.length === 1) {
      // Single selection
      const selectedUuid = selectedRunUuidsFromCheckbox[0];
      if (selectedRunUuid !== selectedUuid) {
        setSelectedRunUuid(selectedUuid);
      }
      if (compareToRunUuid) {
        setCompareToRunUuid(undefined);
      }
    } else if (selectedRunUuidsFromCheckbox.length >= 2) {
      // Two selections - sync both URL params
      // Keep existing selectedRunUuid if it's still selected
      if (!selectedRunUuid || !selectedRunUuidsFromCheckbox.includes(selectedRunUuid)) {
        setSelectedRunUuid(selectedRunUuidsFromCheckbox[0]);
      }
      const otherRun = selectedRunUuidsFromCheckbox.find((uuid) => uuid !== selectedRunUuid);
      if (otherRun && otherRun !== compareToRunUuid) {
        setCompareToRunUuid(otherRun);
      }
    }
  }, [
    enableImprovedComparison,
    showIssuesPanelFlag,
    isComparisonMode,
    selectedRunUuidsFromCheckbox,
    selectedRunUuid,
    compareToRunUuid,
    setSelectedRunUuid,
    setCompareToRunUuid,
    setIsComparisonMode,
  ]);

  /**
   * Generate a list of unique data columns based on runs' metrics, params, and tags.
   */
  const uniqueColumns = useMemo(() => {
    const metricKeys: Set<string> = new Set();
    const paramKeys: Set<string> = new Set();
    const tagKeys: Set<string> = new Set();
    // Using for-of to avoid costlier functions and iterators
    for (const run of baselineRun ? [baselineRun, ...(runs ?? [])] : (runs ?? [])) {
      for (const metric of run.data.metrics ?? []) {
        metricKeys.add(metric.key);
      }
      for (const param of run.data.params ?? []) {
        paramKeys.add(param.key);
      }
      for (const tag of run.data.tags ?? []) {
        if (isUserFacingTag(tag.key)) {
          tagKeys.add(tag.key);
        }
      }
    }
    return [
      ...Array.from(metricKeys).map((key) =>
        createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.METRIC, key),
      ),
      ...Array.from(paramKeys).map((key) =>
        createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.PARAM, key),
      ),
      ...Array.from(tagKeys).map((key) => createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.TAG, key)),
    ];
  }, [baselineRun, runs]);

  const baseSelectionState = useMemo(() => getEvalRunsBaseSelectionState(enableNewEvalRunsUi), [enableNewEvalRunsUi]);
  const baseColumns = useMemo(() => Object.keys(baseSelectionState), [baseSelectionState]);
  const existingColumns = useMemo(
    () => Object.keys(selectedColumns).filter((column) => !baseColumns.includes(column)),
    [baseColumns, selectedColumns],
  );
  const columnDifference = useMemo(() => xor(existingColumns, uniqueColumns), [existingColumns, uniqueColumns]);
  // if there is a difference between the existing column state and
  // the unique metrics (e.g. the user performed a search and the
  // list of available metrics changed), reset the selected columns
  // to the default state to avoid displaying columns that don't exist
  // The new UI keeps explicit visibility choices across filters and pagination.
  const newColumns = uniqueColumns.filter((column) => !(column in selectedColumns));
  if (enableNewEvalRunsUi ? newColumns.length > 0 : columnDifference.length > 0) {
    const metricColumns = uniqueColumns.filter((col) => col.startsWith(EvalRunsTableKeyedColumnPrefix.METRIC + '.'));
    // When flag is ON, limit default visible metrics to 5; when OFF, show all (original behavior)
    const defaultEnabledMetrics =
      !enableNewEvalRunsUi && enableImprovedComparison
        ? new Set(metricColumns.slice(0, DEFAULT_VISIBLE_METRIC_COLUMNS))
        : new Set(metricColumns);

    setSelectedColumns({
      ...baseSelectionState,
      ...mapValues(keyBy(enableNewEvalRunsUi ? newColumns : uniqueColumns), (_, key) => defaultEnabledMetrics.has(key)),
      ...(enableNewEvalRunsUi ? selectedColumns : {}),
    });
  }

  const chartMetricKeys = useMemo(
    () =>
      uniqueColumns.flatMap((column) => {
        const parsed = parseEvalRunsTableKeyedColumnKey(column);
        return parsed?.columnType === EvalRunsTableKeyedColumnPrefix.METRIC ? [parsed.key] : [];
      }),
    [uniqueColumns],
  );
  // Undefined follows defaults; an explicit empty selection stays empty across page reloads.
  const [storedChartMetricSelection, setStoredChartMetricSelection] = useState<unknown>(() =>
    enableNewEvalRunsUi ? readChartMetricSelection(experimentId) : undefined,
  );
  const setSelectedChartMetricKeys = useCallback(
    (metricKeys: string[]) => {
      setStoredChartMetricSelection(metricKeys);
      setLocalStorageItem(
        getChartMetricSelectionStorageKey(experimentId),
        CHART_METRIC_SELECTION_STORAGE_VERSION,
        false,
        metricKeys,
      );
    },
    [experimentId],
  );
  const chartMetricSelection = isStringArray(storedChartMetricSelection) ? storedChartMetricSelection : undefined;

  const isEmpty =
    runUuids.length === 0 && !baselineRun && !searchFilter && !isLoading && !(enableNewEvalRunsUi && isAnyFilterActive);
  // A pinned baseline still renders when filters exclude it, so it doesn't count as a filter match.
  const hasNoFilteredRuns = enableNewEvalRunsUi && isAnyFilterActive && runUuids.length === 0 && !isLoading && !error;
  // The runs hook reports [] until its first response, so wait for the unfiltered query before
  // deciding whether filters hid the runs or the experiment has none.
  const isCheckingUnfilteredRuns = hasNoFilteredRuns && isLoadingUnfilteredRuns;
  const isFilteredEmpty =
    hasNoFilteredRuns && !isLoadingUnfilteredRuns && (unfilteredRuns.length > 0 || Boolean(unfilteredRunsError));
  const hasNoRunsAtAll =
    hasNoFilteredRuns && !isLoadingUnfilteredRuns && !unfilteredRunsError && unfilteredRuns.length === 0;
  const isOnlyDateFilterActive = !searchFilter.trim() && filters.clauses.length === 0;

  // Prepare hierarchical data for table display (supports grouping, nesting, or flat view)
  const runsWithHierarchy = useMemo(() => {
    const runsArray = runs ?? [];

    if (groupBy) return getGroupByRunsData(runsArray, groupBy);

    // Similar to Model training mode, if there's grouping, parent-child view will be disabled.
    const shouldNest = !groupBy;

    if (shouldNest) {
      return getNestedRuns(runsArray);
    }

    return runsArray;
  }, [runs, groupBy]);

  // Extract flat array from hierarchical structure for components that need simple arrays.
  // Charts expect flat arrays with no nesting or grouping structure.
  const flatRuns = useMemo(() => flattenRunEntityOrGroupData(runsWithHierarchy), [runsWithHierarchy]);

  const handleCompare = useCallback(
    (runUuid1: string, runUuid2: string) => {
      // Set both URL params atomically to avoid race conditions
      setSearchParams(
        (params) => {
          params.set(SELECTED_RUN_UUID_QUERY_PARAM, runUuid1);
          params.set(COMPARE_TO_RUN_UUID_QUERY_PARAM, runUuid2);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  // New-compare path: instead of the in-page split view, navigate to the eval-run detail page in
  // compare mode (baseline = first selected run, the rest carried as a query param).
  const handleCompareNavigate = useCallback(
    (runUuids: string[]) => {
      if (runUuids.length < 2) {
        return;
      }
      const [baseline, ...rest] = runUuids;
      navigate(
        `${Routes.getEvaluationRunDetailsRoute(experimentId, baseline)}?compare=true&compareFrom=runs&compareRuns=${rest.join(',')}`,
      );
    },
    [experimentId, navigate],
  );

  const renderActiveTab = (selectedRunUuid: string) => {
    if (viewMode === ExperimentEvaluationRunsPageMode.CHARTS) {
      return <ExperimentEvaluationRunsPageCharts runs={flatRuns} experimentId={experimentId} />;
    }

    const selectedRun = flatRuns?.find((run) => run.info.runUuid === selectedRunUuid);
    // Keyed by tag key so RunViewEvaluationsTab can detect regression-test runs
    // (mlflow.runType=test) and switch the result view accordingly.
    const selectedRunTags = keyBy(selectedRun?.data?.tags ?? [], 'key');
    return (
      <RunViewEvaluationsTab
        experimentId={experimentId}
        runUuid={selectedRunUuid}
        runTags={selectedRunTags}
        runDisplayName={Utils.getRunDisplayName(selectedRun?.info, selectedRunUuid)}
        setCurrentRunUuid={setSelectedRunUuid}
        showCompareSelector
        showRefreshButton
      />
    );
  };

  const tableContainerRef = useRef<HTMLDivElement>(null);
  const offsetFromBottomToFetchMore = 100;
  const fetchMoreOnBottomReached = useCallback(
    (containerRefElement?: HTMLDivElement | null) => {
      if (containerRefElement) {
        const { scrollHeight, scrollTop, clientHeight } = containerRefElement;
        if (scrollHeight - scrollTop - clientHeight < offsetFromBottomToFetchMore && !isFetching && hasNextPage) {
          fetchNextPage();
        }
      }
    },
    [fetchNextPage, isFetching, hasNextPage],
  );

  // a check on mount and after a fetch to see if the table is already scrolled to the bottom and immediately needs to fetch more data
  useEffect(() => {
    fetchMoreOnBottomReached(tableContainerRef.current);
  }, [fetchMoreOnBottomReached]);

  // The V2 footer always shows the run count; with no filter it renders a plain "N runs".
  const visibleCount = runs?.length ?? 0;
  // Clamp: the unfiltered query holds only its first page, so it may lag behind the filtered count.
  const unfilteredTotal = Math.max(unfilteredRuns?.length ?? 0, visibleCount);

  const setBaseline = baseline?.setBaseline;
  const setCurrentExperimentBaseline = useCallback(
    (runUuid: string) => {
      const selectedRun = runs?.find((run) => run.info.runUuid === runUuid);
      if (isMatchingEvalRunsBaseline(selectedRun, runUuid, experimentId)) {
        setBaseline?.(runUuid);
      }
    },
    [experimentId, runs, setBaseline],
  );

  // Full-page list view (non-comparison mode, but not when viewing charts)
  // When flag is OFF, NEVER show full-page list view - always show split view (original behavior)
  // When flag is ON, show full-page list view when not in comparison mode and not in charts mode
  const shouldShowFullPageView =
    (enableImprovedComparison || (showIssuesPanelFlag && !isViewingSelectedRun)) &&
    !isComparisonMode &&
    viewMode !== ExperimentEvaluationRunsPageMode.CHARTS;
  // The selection filter only applies to the chart rendered in the full-page list view.
  const isNewEvalRunsChartMounted = enableNewEvalRunsUi && shouldShowFullPageView && !isEmpty && !hasNoFilteredRuns;

  const renderTableControls = () =>
    enableNewEvalRunsUi ? (
      <ExperimentEvaluationRunsTableControlsV2
        experimentId={experimentId}
        runs={flatRuns ?? []}
        refetchRuns={refetchAll}
        isFetching={isFetching || isLoading}
        lastUpdatedTime={dataUpdatedAt}
        searchRunsError={error}
        searchFilter={searchFilter}
        setSearchFilter={setSearchFilter}
        rowSelection={rowSelection}
        setRowSelection={updateRowSelection}
        selectedColumns={selectedColumns}
        setSelectedColumns={setSelectedColumns}
        filters={filters}
        setDatePreset={setDatePreset}
        setClauses={setClauses}
        excludeIssueDetectionRuns
        filterButtonRef={filterButtonRef}
        groupByConfig={groupBy}
        setGroupByConfig={setGroupBy}
        sorting={evalRunsSorting}
        onSortingChange={setEvalRunsSorting}
        viewMode={viewMode}
        setViewMode={setViewMode}
        onCompare={handleCompare}
        selectedRunUuid={selectedRunUuid}
        compareToRunUuid={compareToRunUuid}
        setIsComparisonMode={setIsComparisonMode}
        enableNRunComparison
        onCompareNavigate={handleCompareNavigate}
        onFilterChartToSelectedRuns={
          isNewEvalRunsChartMounted ? () => setFilterChartToSelectedRuns((current) => !current) : undefined
        }
        isChartFilteredToSelection={isNewEvalRunsChartMounted ? filterChartToSelectedRuns : undefined}
        {...(baseline
          ? {
              onSetBaseline: setCurrentExperimentBaseline,
              onClearBaseline: baseline.clearBaseline,
              baselineRunUuid,
              isSavingBaseline: baseline.isSavingBaseline,
            }
          : {})}
      />
    ) : (
      <ExperimentEvaluationRunsTableControls
        runs={flatRuns ?? []}
        refetchRuns={refetchAll}
        isFetching={isFetching || isLoading}
        searchRunsError={error}
        searchFilter={searchFilter}
        setSearchFilter={setSearchFilter}
        rowSelection={rowSelection}
        setRowSelection={setRowSelection}
        selectedColumns={selectedColumns}
        setSelectedColumns={setSelectedColumns}
        groupByConfig={groupBy}
        setGroupByConfig={setGroupBy}
        viewMode={viewMode}
        setViewMode={setViewMode}
        onCompare={handleCompare}
        selectedRunUuid={selectedRunUuid}
        compareToRunUuid={compareToRunUuid}
        isComparisonMode={isComparisonMode}
        setIsComparisonMode={setIsComparisonMode}
        enableImprovedComparison={enableImprovedComparison}
      />
    );

  const renderTable = () => (
    <ExperimentEvaluationRunsTable
      data={runsWithHierarchy}
      {...(enableNewEvalRunsUi ? { baselineRun, onFilterByColumn: handleFilterByColumn } : {})}
      uniqueColumns={uniqueColumns}
      selectedColumns={selectedColumns}
      setSelectedColumns={setSelectedColumns}
      sorting={enableNewEvalRunsUi ? evalRunsSorting : undefined}
      onSortingChange={enableNewEvalRunsUi ? setEvalRunsSorting : undefined}
      selectedRunUuid={
        enableImprovedComparison && isComparisonMode && viewMode === ExperimentEvaluationRunsPageMode.TRACES
          ? selectedRunUuid
          : undefined
      }
      setSelectedRunUuid={(runUuid: string) => {
        // Update both params atomically to avoid race conditions
        // where separate setSearchParams calls overwrite each other
        setSearchParams(
          (params) => {
            params.set(SELECTED_RUN_UUID_QUERY_PARAM, runUuid);
            params.delete(COMPARE_TO_RUN_UUID_QUERY_PARAM);
            return params;
          },
          { replace: true },
        );
        // When flag is ON, also set isViewingSelectedRun to show the split view
        if (showIssuesPanelFlag) {
          setIsViewingSelectedRun(true);
        }
      }}
      isLoading={isLoading || isCheckingUnfilteredRuns}
      hasNextPage={hasNextPage ?? false}
      rowSelection={rowSelection}
      setRowSelection={setTableRowSelection}
      setSelectedDatasetWithRun={setSelectedDatasetWithRun}
      setIsDrawerOpen={setIsDrawerOpen}
      viewMode={viewMode}
      onScroll={(e) => fetchMoreOnBottomReached(e.currentTarget)}
      ref={tableContainerRef}
      isGrouped={Boolean(groupBy?.groupByKeys?.length)}
      enableImprovedComparison={enableImprovedComparison}
      enableNewEvalRunsUi={enableNewEvalRunsUi}
    />
  );

  const renderEmptyState = () => (
    <div
      css={{
        display: 'flex',
        flex: 1,
        flexDirection: 'column',
        overflow: 'auto',
        // Always reserve the scrollbar gutter so the horizontally-centered content doesn't
        // shift left/right when a taller tab adds (or a shorter tab removes) the scrollbar.
        scrollbarGutter: 'stable',
      }}
    >
      <div
        css={{
          // Top-anchored (not vertically centered): the AgentActionCard below changes height
          // when its active tab switches, and centering would re-center the whole block —
          // making the title and image above visibly jump. Anchoring to the top keeps them
          // put and stays scrollable when the content overflows.
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          // Pin to a definite width (the card's own maxWidth) instead of shrink-to-fit. With
          // `margin: 0 auto` a flex column hugs its widest child, so switching to a wide tab
          // (e.g. the non-wrapping Python snippet) would widen the whole block and resize the
          // card. A fixed width keeps it stable; wide tab content scrolls within the card.
          width: '100%',
          maxWidth: 720,
          padding: `${theme.spacing.lg * 2}px ${theme.spacing.md}px ${theme.spacing.lg * 4}px`,
        }}
      >
        <div
          css={{ maxWidth: 520, display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}
        >
          <Typography.Title level={3} color="secondary" css={{ marginTop: 0, marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Evaluate and improve the quality, cost, latency of your GenAI app"
              description="Title of the empty state for the evaluation runs page"
            />
          </Typography.Title>
          <Typography.Paragraph color="secondary" css={{ marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="Create evaluation datasets in order to iteratively evaluate and improve your app. Run evaluations to check that your fixes are working, and compare quality between app / prompt versions. {learnMoreLink}"
              description="Description of the empty state for the evaluation runs page"
              values={{
                learnMoreLink: (
                  <Typography.Link
                    componentId="mlflow.eval-runs.empty-state.learn-more-link"
                    href={getLearnMoreLink()}
                    css={{ whiteSpace: 'nowrap' }}
                    openInNewTab
                  >
                    <FormattedMessage
                      defaultMessage="Learn more"
                      description="Link text to learn more about evaluation runs"
                    />
                  </Typography.Link>
                ),
              }}
            />
          </Typography.Paragraph>
        </div>
        <img css={{ maxWidth: '100%', maxHeight: 160 }} src={evalRunsEmptyImg} alt="No runs found" />
        <div css={{ width: '100%', marginTop: theme.spacing.lg }}>
          <EvalRunsEmptyStateCard experimentId={experimentId} />
        </div>
      </div>
    </div>
  );

  const renderCountFooter = () =>
    enableNewEvalRunsUi ? (
      <div css={{ flexShrink: 0, paddingTop: theme.spacing.xs }}>
        <EvalRunsFilterCount
          isAnyFilterActive={isAnyFilterActive}
          visibleCount={visibleCount}
          totalCount={isAnyFilterActive ? unfilteredTotal : visibleCount}
          hasMoreRuns={Boolean(hasNextPage || (isAnyFilterActive && unfilteredHasNextPage))}
        />
      </div>
    ) : null;

  const clearAllFilters = () => {
    setSearchFilter('');
    setClauses([]);
    setDatePreset(EvalRunsDatePreset.ALL);
  };

  const renderFilteredEmptyState = () => (
    <div
      css={{
        display: 'flex',
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
        minHeight: 400,
        '& > div': {
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
        },
      }}
    >
      {isOnlyDateFilterActive ? (
        <Empty
          image={<CalendarRangeIcon />}
          title={
            <FormattedMessage
              defaultMessage="No evaluation runs in this time range"
              description="Title of the evaluation runs empty state when the selected time range hides every run"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="This experiment has evaluation runs outside the selected time range."
              description="Description of the evaluation runs empty state when the selected time range hides every run"
            />
          }
          button={
            <Button
              componentId="mlflow.eval-runs.filtered-empty-state.show-all-time"
              onClick={() => setDatePreset(EvalRunsDatePreset.ALL)}
            >
              <FormattedMessage
                defaultMessage="Show all time"
                description="Button that resets the evaluation runs time range to all time"
              />
            </Button>
          }
        />
      ) : (
        <Empty
          image={<FilterIcon />}
          title={
            <FormattedMessage
              defaultMessage="No evaluation runs match your filters"
              description="Title of the evaluation runs empty state when the active filters hide every run"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="This experiment has evaluation runs that don't match the current search, filters, or time range."
              description="Description of the evaluation runs empty state when the active filters hide every run"
            />
          }
          button={
            <Button componentId="mlflow.eval-runs.filtered-empty-state.clear-filters" onClick={clearAllFilters}>
              <FormattedMessage
                defaultMessage="Clear filters"
                description="Button that clears all evaluation runs search, filters, and time range"
              />
            </Button>
          }
        />
      )}
    </div>
  );

  if (shouldShowFullPageView) {
    return (
      <ExperimentEvaluationRunsRowVisibilityProvider>
        <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: '0px' }}>
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
              flex: 1,
              minHeight: '0px',
              overflow: 'hidden',
            }}
          >
            {isNewEvalRunsChartMounted && (
              <EvalRunsRecentRunsChart
                runs={runs}
                selectedRunUuids={selectedRunUuidsFromCheckbox}
                filterToSelected={filterChartToSelectedRuns}
                onRunClick={selectRunFromChart}
                isLoading={isLoading}
                metricKeys={chartMetricKeys}
                selectedMetricKeys={chartMetricSelection}
                onSelectedMetricKeysChange={setSelectedChartMetricKeys}
              />
            )}
            {renderTableControls()}
            {isEmpty || hasNoRunsAtAll ? (
              renderEmptyState()
            ) : isFilteredEmpty ? (
              renderFilteredEmptyState()
            ) : (
              <>
                {renderTable()}
                {renderCountFooter()}
              </>
            )}
          </div>
          {selectedDatasetWithRun && (
            <ExperimentViewDatasetDrawer
              isOpen={isDrawerOpen}
              setIsOpen={setIsDrawerOpen}
              selectedDatasetWithRun={selectedDatasetWithRun}
              setSelectedDatasetWithRun={setSelectedDatasetWithRun}
            />
          )}
        </div>
      </ExperimentEvaluationRunsRowVisibilityProvider>
    );
  }

  // Split view (comparison mode)
  return (
    <ExperimentEvaluationRunsRowVisibilityProvider>
      <div css={{ display: 'flex', flexDirection: 'row', flex: 1, minHeight: '0px' }}>
        <ResizableBox
          css={{ display: 'flex', position: 'relative' }}
          style={{ flex: `0 0 ${runListHidden ? 0 : tableWidth}px` }}
          width={tableWidth}
          axis="x"
          resizeHandles={['e']}
          minConstraints={[250, 0]}
          handle={
            <ExperimentViewRunsTableResizerHandle
              runListHidden={isViewingSelectedRun ? true : runListHidden}
              updateRunListHidden={() => {
                if (isViewingSelectedRun) {
                  setIsViewingSelectedRun(false);
                } else {
                  setRunListHidden(!runListHidden);
                }
              }}
            />
          }
          onResize={(event, { size }) => {
            if (runListHidden) {
              return;
            }
            setTableWidth(size.width);
          }}
          onResizeStart={() => !runListHidden && setDragging(true)}
          onResizeStop={() => setDragging(false)}
        >
          <div
            css={{
              display: runListHidden ? 'none' : 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
              maxWidth: '100%',
              flex: 1,
              zIndex: 0,
              minHeight: '0px',
              overflow: 'hidden',
              paddingRight: theme.spacing.sm,
            }}
          >
            {renderTableControls()}
            {renderTable()}
            {renderCountFooter()}
          </div>
        </ResizableBox>
        <div
          css={{
            flex: 1,
            display: 'flex',
            borderLeft: `1px solid ${theme.colors.border}`,
            minHeight: '0px',
            overflowY: 'scroll',
          }}
        >
          {viewMode === ExperimentEvaluationRunsPageMode.CHARTS ? (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                flex: 1,
                minHeight: '0px',
                paddingLeft: theme.spacing.sm,
                ...(enableNewEvalRunsUi && {
                  animation: `${chartsPaneFadeIn} 0.2s ease`,
                  '@media (prefers-reduced-motion: reduce)': {
                    animation: 'none',
                  },
                }),
              }}
            >
              <ExperimentEvaluationRunsPageCharts runs={flatRuns} experimentId={experimentId} />
            </div>
          ) : selectedRunUuid ? (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                flex: 1,
                minHeight: '0px',
                paddingLeft: theme.spacing.sm,
                alignItems: 'center',
                maxWidth: '100%',
                boxSizing: 'border-box',
              }}
            >
              {renderActiveTab(selectedRunUuid)}
            </div>
          ) : isEmpty ? (
            renderEmptyState()
          ) : null}
        </div>
        {dragging && (
          <Global
            styles={{
              'body, :host': {
                userSelect: 'none',
              },
            }}
          />
        )}
        {selectedDatasetWithRun && (
          <ExperimentViewDatasetDrawer
            isOpen={isDrawerOpen}
            setIsOpen={setIsDrawerOpen}
            selectedDatasetWithRun={selectedDatasetWithRun}
            setSelectedDatasetWithRun={setSelectedDatasetWithRun}
          />
        )}
      </div>
    </ExperimentEvaluationRunsRowVisibilityProvider>
  );
};

const ExperimentEvaluationRunsPageWithBaseline = ({ experimentId }: { experimentId: string }) => {
  const baseline = useEvalRunsBaselineState(experimentId);
  return <ExperimentEvaluationRunsPageImpl baseline={baseline} />;
};

const ExperimentEvaluationRunsPage = () => {
  const { experimentId } = useParams();
  // The new UI starts each experiment from a fresh page: selection, columns, chart choices and the
  // baseline belong to the experiment they were made in.
  return (
    <ExperimentEvaluationRunsPageWrapper>
      {shouldEnableNewEvalRunsUi() ? (
        <ExperimentEvaluationRunsPageWithBaseline key={experimentId} experimentId={experimentId ?? ''} />
      ) : (
        <ExperimentEvaluationRunsPageImpl />
      )}
    </ExperimentEvaluationRunsPageWrapper>
  );
};

export default ExperimentEvaluationRunsPage;
