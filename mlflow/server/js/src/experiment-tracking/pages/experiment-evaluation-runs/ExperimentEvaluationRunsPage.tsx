import invariant from 'invariant';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Global } from '@emotion/react';
import { useExperimentEvaluationRunsData } from '../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { ExperimentEvaluationRunsPageWrapper } from './ExperimentEvaluationRunsPageWrapper';
import { ExperimentEvaluationRunsTable } from './ExperimentEvaluationRunsTable';
import type { RowSelectionState } from '@tanstack/react-table';
import { useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
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
  EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
  EvalRunsTableKeyedColumnPrefix,
} from './ExperimentEvaluationRunsTable.constants';
import { FormattedMessage } from 'react-intl';
import { useSelectedRunUuid } from '../../components/evaluations/hooks/useSelectedRunUuid';
import { useCompareToRunUuid } from '../../components/evaluations/hooks/useCompareToRunUuid';
import { RunEvaluationButton } from './RunEvaluationButton';
import { isUserFacingTag } from '../../../common/utils/TagUtils';
import { createEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import {
  RunGroupingAggregateFunction,
  RunGroupingMode,
} from '../../components/experiment-page/utils/experimentPage.row-types';
import { getGroupByRunsData } from './ExperimentEvaluationRunsPage.utils';
import {
  ExperimentEvaluationRunsPageMode,
  useExperimentEvaluationRunsPageMode,
} from './hooks/useExperimentEvaluationRunsPageMode';
import { ExperimentEvaluationRunsPageCharts } from './charts/ExperimentEvaluationRunsPageCharts';
import { ExperimentEvaluationRunsRowVisibilityProvider } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { useRegisterSelectedIds } from '@mlflow/mlflow/src/assistant';
import { shouldEnableImprovedEvalRunsComparison } from '../../../common/utils/FeatureUtils';

const DEFAULT_VISIBLE_METRIC_COLUMNS = 5;

const getLearnMoreLink = () => {
  return 'https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/';
};

const ExperimentEvaluationRunsPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [tableWidth, setTableWidth] = useState(432);
  const [dragging, setDragging] = useState(false);
  const [runListHidden, setRunListHidden] = useState(false);
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType>();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedColumns, setSelectedColumns] = useState<{ [key: string]: boolean }>(
    EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
  );

  const [groupBy, setGroupBy] = useState<RunsGroupByConfig | null>({
    aggregateFunction: RunGroupingAggregateFunction.Average,
    groupByKeys: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset' }],
  });
  const [isComparisonMode, setIsComparisonMode] = useState(false);
  const { viewMode, setViewMode } = useExperimentEvaluationRunsPageMode();

  const [selectedRunUuid, setSelectedRunUuid] = useSelectedRunUuid();
  const [compareToRunUuid, setCompareToRunUuid] = useCompareToRunUuid();
  const [, setSearchParams] = useSearchParams();

  invariant(experimentId, 'Experiment ID must be defined');

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  useRegisterSelectedIds('selectedRunIds', rowSelection);

  const {
    data: runs,
    isLoading,
    isFetching,
    error,
    fetchNextPage,
    hasNextPage,
    refetch,
  } = useExperimentEvaluationRunsData({
    experimentId,
    enabled: true,
    filter: searchFilter,
  });

  const runUuids = useMemo(() => runs?.map((run) => run.info.runUuid) ?? [], [runs]);

  // Get selected run UUIDs from checkbox selection
  const selectedRunUuidsFromCheckbox = useMemo(
    () =>
      Object.entries(rowSelection)
        .filter(([_, value]) => value)
        .map(([key]) => key),
    [rowSelection],
  );

  // On mount, if URL has selectedRunUuid (and optionally compareToRunUuid), initialize rowSelection and enter comparison mode
  // Initialize with BOTH URL params to prevent compareToRunUuid from being stripped
  // Only enabled when comparison feature flag is on
  const hasInitializedFromUrl = useRef(false);
  const isComparisonEnabled = shouldEnableImprovedEvalRunsComparison();
  useEffect(() => {
    if (!isComparisonEnabled) {
      return;
    }
    if (!hasInitializedFromUrl.current && selectedRunUuid && runs?.length) {
      if (runUuids.includes(selectedRunUuid)) {
        hasInitializedFromUrl.current = true;
        const initialSelection: RowSelectionState = { [selectedRunUuid]: true };
        // Also include compareToRunUuid if present in URL
        if (compareToRunUuid && runUuids.includes(compareToRunUuid)) {
          initialSelection[compareToRunUuid] = true;
        }
        setRowSelection(initialSelection);
        setIsComparisonMode(true);
      }
    }
  }, [isComparisonEnabled, selectedRunUuid, compareToRunUuid, runs, runUuids, setIsComparisonMode]);

  // Sync URL params from checkbox selection when in comparison mode (as useEffect to avoid render-time state updates)
  // Only enabled when comparison feature flag is on
  useEffect(() => {
    if (!isComparisonEnabled) {
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
    isComparisonEnabled,
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
    for (const run of runs ?? []) {
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
  }, [runs]);

  const baseColumns = useMemo(() => Object.keys(EVAL_RUNS_TABLE_BASE_SELECTION_STATE), []);
  const existingColumns = useMemo(
    () => Object.keys(selectedColumns).filter((column) => !baseColumns.includes(column)),
    [baseColumns, selectedColumns],
  );
  const columnDifference = useMemo(() => xor(existingColumns, uniqueColumns), [existingColumns, uniqueColumns]);
  // if there is a difference between the existing column state and
  // the unique metrics (e.g. the user performed a search and the
  // list of available metrics changed), reset the selected columns
  // to the default state to avoid displaying columns that don't exist
  if (columnDifference.length > 0) {
    const metricColumns = uniqueColumns.filter((col) => col.startsWith(EvalRunsTableKeyedColumnPrefix.METRIC + '.'));
    const defaultEnabledMetrics = new Set(metricColumns.slice(0, DEFAULT_VISIBLE_METRIC_COLUMNS));

    setSelectedColumns({
      ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
      ...mapValues(keyBy(uniqueColumns), (_, key) => defaultEnabledMetrics.has(key)),
    });
  }

  const isEmpty = runUuids.length === 0 && !searchFilter && !isLoading;

  const runsAndGroupValues = getGroupByRunsData(runs ?? [], groupBy);

  const handleCompare = useCallback(
    (runUuid1: string, runUuid2: string) => {
      // Set both URL params atomically to avoid race conditions
      setSearchParams(
        (params) => {
          params.set('selectedRunUuid', runUuid1);
          params.set('compareToRunUuid', runUuid2);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const renderActiveTab = (selectedRunUuid: string) => {
    if (viewMode === ExperimentEvaluationRunsPageMode.CHARTS) {
      return <ExperimentEvaluationRunsPageCharts runs={runs} experimentId={experimentId} />;
    }

    return (
      <RunViewEvaluationsTab
        experimentId={experimentId}
        runUuid={selectedRunUuid}
        runDisplayName={Utils.getRunDisplayName(
          runs?.find((run) => run.info.runUuid === selectedRunUuid)?.info,
          selectedRunUuid,
        )}
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

  const renderTableControls = () => (
    <ExperimentEvaluationRunsTableControls
      runs={runs ?? []}
      refetchRuns={refetch}
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
    />
  );

  const renderTable = () => (
    <ExperimentEvaluationRunsTable
      data={runsAndGroupValues}
      uniqueColumns={uniqueColumns}
      selectedColumns={selectedColumns}
      selectedRunUuid={
        isComparisonMode && viewMode === ExperimentEvaluationRunsPageMode.TRACES ? selectedRunUuid : undefined
      }
      setSelectedRunUuid={(runUuid: string) => {
        setSelectedRunUuid(runUuid);
        setCompareToRunUuid(undefined);
      }}
      isLoading={isLoading}
      hasNextPage={hasNextPage ?? false}
      rowSelection={rowSelection}
      setRowSelection={setRowSelection}
      setSelectedDatasetWithRun={setSelectedDatasetWithRun}
      setIsDrawerOpen={setIsDrawerOpen}
      viewMode={viewMode}
      onScroll={(e) => fetchMoreOnBottomReached(e.currentTarget)}
      ref={tableContainerRef}
    />
  );

  const renderEmptyState = () => (
    <div
      css={{
        display: 'flex',
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        marginTop: theme.spacing.lg,
        paddingLeft: theme.spacing.md,
        maxWidth: '100%',
      }}
    >
      <Typography.Title level={3} color="secondary">
        <FormattedMessage
          defaultMessage="Evaluate and improve the quality, cost, latency of your GenAI app"
          description="Title of the empty state for the evaluation runs page"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 'min(100%, 600px)', textAlign: 'center' }}>
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
                {/* eslint-disable-next-line formatjs/enforce-description */}
                <FormattedMessage defaultMessage="Learn more" />
              </Typography.Link>
            ),
          }}
        />
      </Typography.Paragraph>
      <img css={{ maxWidth: '100%', maxHeight: 200 }} src={evalRunsEmptyImg} alt="No runs found" />
      <div css={{ display: 'flex', gap: theme.spacing.sm, marginTop: theme.spacing.md }}>
        <RunEvaluationButton experimentId={experimentId} />
      </div>
    </div>
  );

  // Full-page list view (default, non-comparison mode)
  if (!isComparisonMode) {
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
            {renderTableControls()}
            {isEmpty ? renderEmptyState() : renderTable()}
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
              runListHidden={runListHidden}
              updateRunListHidden={(value) => {
                setRunListHidden(!runListHidden);
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
          {selectedRunUuid ? (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                flex: 1,
                minHeight: '0px',
                paddingLeft: theme.spacing.sm,
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

const ExperimentEvaluationRunsPage = () => (
  <ExperimentEvaluationRunsPageWrapper>
    <ExperimentEvaluationRunsPageImpl />
  </ExperimentEvaluationRunsPageWrapper>
);

export default ExperimentEvaluationRunsPage;
