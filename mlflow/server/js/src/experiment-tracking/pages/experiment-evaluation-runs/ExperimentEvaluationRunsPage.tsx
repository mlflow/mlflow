import invariant from 'invariant';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Global } from '@emotion/react';
import { useExperimentEvaluationRunsData } from '../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { ExperimentEvaluationRunsPageWrapper } from './ExperimentEvaluationRunsPageWrapper';
import { ExperimentEvaluationRunsTable } from './ExperimentEvaluationRunsTable';
import type { RowSelectionState } from '@tanstack/react-table';
import { useParams } from '../../../common/utils/RoutingUtils';
import { Alert, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { Progress } from '@databricks/design-system/development';
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
import { getGroupByRunsData } from './ExperimentEvaluationRunsPage.utils';
import {
  ExperimentEvaluationRunsPageMode,
  useExperimentEvaluationRunsPageMode,
} from './hooks/useExperimentEvaluationRunsPageMode';
import { ExperimentEvaluationRunsPageCharts } from './charts/ExperimentEvaluationRunsPageCharts';
import { ExperimentEvaluationRunsRowVisibilityProvider } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { useRegisterSelectedIds } from '@mlflow/mlflow/src/assistant';
import {
  loadPersistedEvaluationRun,
  loadPersistedEvaluationRunByRunId,
  clearPersistedEvaluationRun,
  clearPersistedEvaluationRunByRunId,
  EVALUATION_RUN_STORAGE_EVENT,
} from '../../components/experiment-page/components/traces-v3/utils/evaluationRunStorage';
import {
  TrackingJobQueryResult,
  TrackingJobStatus,
  useGetTrackingServerJobStatus,
} from '../../../common/hooks/useGetTrackingServerJobStatus';
import { invalidateMlflowSearchTracesCache } from '@databricks/web-shared/genai-traces-table';
import { useQueryClient } from '@databricks/web-shared/query-client';

const getLearnMoreLink = () => {
  return 'https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/';
};

type EvaluateTracesToRunJobResult = {
  run_id?: string;
  experiment_id?: string;
  progress?: { completed: number; total: number };
};

const JOB_POLLING_INTERVAL = 1500;

const isJobRunning = (jobData?: TrackingJobQueryResult<EvaluateTracesToRunJobResult>): boolean => {
  return jobData?.status === TrackingJobStatus.RUNNING || jobData?.status === TrackingJobStatus.PENDING;
};

const ExperimentEvaluationRunsPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const formatCount = useMemo(() => new Intl.NumberFormat(), []);
  const [tableWidth, setTableWidth] = useState(432);
  const [dragging, setDragging] = useState(false);
  const [runListHidden, setRunListHidden] = useState(false);
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType>();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedColumns, setSelectedColumns] = useState<{ [key: string]: boolean }>(
    EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
  );

  const [groupBy, setGroupBy] = useState<RunsGroupByConfig | null>(null);
  const { viewMode, setViewMode } = useExperimentEvaluationRunsPageMode();

  const [selectedRunUuid, setSelectedRunUuid] = useSelectedRunUuid();
  const [compareToRunUuid, setCompareToRunUuid] = useCompareToRunUuid();

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

  const [persistedEvalRun, setPersistedEvalRun] = useState(() =>
    selectedRunUuid
      ? loadPersistedEvaluationRunByRunId(experimentId, selectedRunUuid)
      : loadPersistedEvaluationRun(experimentId),
  );

  useEffect(() => {
    setPersistedEvalRun(
      selectedRunUuid
        ? loadPersistedEvaluationRunByRunId(experimentId, selectedRunUuid)
        : loadPersistedEvaluationRun(experimentId),
    );
  }, [experimentId, selectedRunUuid]);

  useEffect(() => {
    const handler = (event: Event) => {
      const customEvent = event as CustomEvent<{ experimentId?: string }>;
      if (customEvent.detail?.experimentId && customEvent.detail.experimentId !== experimentId) {
        return;
      }
      setPersistedEvalRun(
        selectedRunUuid
          ? loadPersistedEvaluationRunByRunId(experimentId, selectedRunUuid)
          : loadPersistedEvaluationRun(experimentId),
      );
    };
    window.addEventListener(EVALUATION_RUN_STORAGE_EVENT, handler);
    return () => window.removeEventListener(EVALUATION_RUN_STORAGE_EVENT, handler);
  }, [experimentId, selectedRunUuid]);

  const persistedJobId = persistedEvalRun?.jobId;
  const { jobResults: evalJobResults } = useGetTrackingServerJobStatus<EvaluateTracesToRunJobResult>(
    persistedJobId ? [persistedJobId] : undefined,
    {
      enabled: Boolean(persistedJobId),
      refetchInterval: (data) => {
        if (data?.some((job) => isJobRunning(job))) {
          return JOB_POLLING_INTERVAL;
        }
        return false;
      },
      refetchOnWindowFocus: false,
      refetchOnMount: false,
    },
  );

  const currentEvalJob = persistedJobId ? evalJobResults?.[persistedJobId] : undefined;
  const evalJobPayload = currentEvalJob && 'result' in currentEvalJob ? (currentEvalJob.result as any) : undefined;
  const evalJobRunId = evalJobPayload?.run_id ?? persistedEvalRun?.runId;
  const isSelectedRunActiveEval = Boolean(selectedRunUuid && evalJobRunId && selectedRunUuid === evalJobRunId);
  const evalJobProgress = evalJobPayload?.progress;
  const evalCompleted = evalJobProgress?.completed ?? 0;
  const evalTotal = evalJobProgress?.total ?? persistedEvalRun?.total ?? 0;
  const evalPct = evalTotal ? Math.min(100, Math.round((evalCompleted / evalTotal) * 100)) : 0;
  const evalJobIsRunning = isJobRunning(currentEvalJob as any);
  const evalJobSucceeded = currentEvalJob?.status === TrackingJobStatus.SUCCEEDED;
  // Consider eval as potentially running if we have a persisted job but haven't received status yet,
  // OR if the job is confirmed to be running. This ensures we apply timestamp filtering immediately
  // on initial render before the first job status poll returns.
  const evalJobMaybeRunning = persistedJobId && (!currentEvalJob || evalJobIsRunning);

  // Track completed evaluation state to show completion message even after persisted data is cleared
  const [completedEvalForRun, setCompletedEvalForRun] = useState<{
    runId: string;
    total: number;
    startedAt: number;
  } | null>(null);

  useEffect(() => {
    if (!persistedEvalRun?.jobId || !currentEvalJob) {
      return;
    }
    // Once the job completes (success/failure/canceled), clear persisted job info so stale storage
    // doesn't affect subsequent runs or deep-links.
    if (isJobRunning(currentEvalJob as any)) {
      return;
    }

    // If job succeeded, remember the completion state for this run before clearing
    if (currentEvalJob.status === TrackingJobStatus.SUCCEEDED && persistedEvalRun.runId && persistedEvalRun.startedAt) {
      setCompletedEvalForRun({
        runId: persistedEvalRun.runId,
        total: evalTotal,
        startedAt: persistedEvalRun.startedAt,
      });
    }

    const runIdToClear = persistedEvalRun.runId ?? selectedRunUuid;
    if (runIdToClear) {
      clearPersistedEvaluationRunByRunId(experimentId, runIdToClear);
    }
    const active = loadPersistedEvaluationRun(experimentId);
    if (active?.jobId === persistedEvalRun.jobId) {
      clearPersistedEvaluationRun(experimentId);
    }
  }, [experimentId, selectedRunUuid, persistedEvalRun?.jobId, persistedEvalRun?.runId, currentEvalJob, evalTotal]);

  // Clear completed state when switching to a different run
  useEffect(() => {
    if (completedEvalForRun && selectedRunUuid !== completedEvalForRun.runId) {
      setCompletedEvalForRun(null);
    }
  }, [selectedRunUuid, completedEvalForRun]);

  useEffect(() => {
    if (!isSelectedRunActiveEval || !evalJobIsRunning) {
      return;
    }
    const interval = window.setInterval(() => {
      // Refresh the run's evaluation traces + the runs list while evaluation is ongoing.
      invalidateMlflowSearchTracesCache({ queryClient });
      refetch();
    }, 3000);
    return () => window.clearInterval(interval);
  }, [isSelectedRunActiveEval, evalJobIsRunning, queryClient, refetch]);

  const runUuids = runs?.map((run) => run.info.runUuid) ?? [];
  // Set the selected run to the first run only if we don't already have one.
  // Note: we intentionally do NOT override a URL-provided selection if it's not yet present in
  // the fetched runs list (e.g. evaluation run exists but hasn't appeared in this view's list yet).
  if (runs?.length && !selectedRunUuid) {
    setSelectedRunUuid(runs[0].info.runUuid);
  }

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
    setSelectedColumns({
      ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
      ...mapValues(keyBy(uniqueColumns), () => false),
    });
  }

  const isEmpty = runUuids.length === 0 && !searchFilter && !isLoading;

  const runsAndGroupValues = getGroupByRunsData(runs ?? [], groupBy);

  const handleCompare = useCallback(
    (runUuid1: string, runUuid2: string) => {
      setSelectedRunUuid(runUuid1);
      setCompareToRunUuid(runUuid2);
    },
    [setSelectedRunUuid, setCompareToRunUuid],
  );

  useEffect(() => {
    const selectedRunUuids = Object.entries(rowSelection)
      .filter(([_, value]) => value)
      .map(([key]) => key);

    if (selectedRunUuid && compareToRunUuid) {
      const isSelectedRunStillSelected = selectedRunUuids.includes(selectedRunUuid);
      const isCompareToRunStillSelected = selectedRunUuids.includes(compareToRunUuid);

      if (!isSelectedRunStillSelected || !isCompareToRunStillSelected) {
        setCompareToRunUuid(undefined);

        if (selectedRunUuids.length > 0) {
          setSelectedRunUuid(selectedRunUuids[0]);
        }
      }
    }
  }, [rowSelection, selectedRunUuid, compareToRunUuid, setSelectedRunUuid, setCompareToRunUuid]);

  const renderActiveTab = (selectedRunUuid: string) => {
    if (viewMode === ExperimentEvaluationRunsPageMode.CHARTS) {
      return <ExperimentEvaluationRunsPageCharts runs={runs} experimentId={experimentId} />;
    }

    // Show loading state if: actively running eval for this run (not yet succeeded)
    const showEvalLoadingState = isSelectedRunActiveEval && evalJobIsRunning;
    // Show completed state briefly after eval finishes
    const showCompletedState = completedEvalForRun?.runId === selectedRunUuid;

    // While evaluation is running, show centered spinner + progress bar (hide table)
    if (showEvalLoadingState) {
      return (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            flex: 1,
            textAlign: 'center',
            padding: theme.spacing.lg,
          }}
        >
          <Spinner
            size="large"
            css={{
              marginBottom: theme.spacing.sm,
            }}
          />
          <Typography.Title level={3} withoutMargins>
            <FormattedMessage
              defaultMessage="Evaluation in progress"
              description="Title for evaluation status when in progress"
            />
          </Typography.Title>
          <div css={{ width: '100%', maxWidth: 300, marginTop: theme.spacing.md }}>
            <Progress.Root value={evalPct}>
              <Progress.Indicator />
            </Progress.Root>
            <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs, display: 'block' }}>
              {`${evalPct}% • ${formatCount.format(evalCompleted)} / ${formatCount.format(evalTotal)} traces`}
            </Typography.Text>
          </div>
        </div>
      );
    }

    return (
      <>
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
          // Pass evalStartTime when evaluation is running or just completed for this run.
          // This filters out auto-instrumented traces created during scoring.
          // We use evalJobMaybeRunning to cover initial render before the first job status poll,
          // and completedEvalForRun to keep filtering after completion until page refresh.
          evalStartTime={
            isSelectedRunActiveEval && evalJobMaybeRunning
              ? persistedEvalRun?.startedAt
              : showCompletedState
                ? completedEvalForRun?.startedAt
                : undefined
          }
        />
      </>
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
            />
            <ExperimentEvaluationRunsTable
              data={runsAndGroupValues}
              uniqueColumns={uniqueColumns}
              selectedColumns={selectedColumns}
              selectedRunUuid={viewMode === ExperimentEvaluationRunsPageMode.TRACES ? selectedRunUuid : undefined}
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
