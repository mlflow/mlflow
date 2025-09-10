import invariant from 'invariant';
import { useEffect, useMemo, useState } from 'react';
import { Global } from '@emotion/react';
import { useExperimentEvaluationRunsData } from '../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { ExperimentEvaluationRunsPageWrapper } from './ExperimentEvaluationRunsPageWrapper';
import { ExperimentEvaluationRunsTable } from './ExperimentEvaluationRunsTable';
import type { RowSelectionState } from '@tanstack/react-table';
import { useParams } from '../../../common/utils/RoutingUtils';
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
  const [groupBy, setGroupBy] = useState<RunsGroupByConfig | null>(null);
  const { viewMode, setViewMode } = useExperimentEvaluationRunsPageMode();

  const [selectedRunUuid, setSelectedRunUuid] = useSelectedRunUuid();

  invariant(experimentId, 'Experiment ID must be defined');

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  const {
    data: runs,
    isLoading,
    isFetching,
    error,
    refetch,
  } = useExperimentEvaluationRunsData({
    experimentId,
    enabled: true,
    filter: searchFilter,
  });

  const runUuids = runs?.map((run) => run.info.runUuid) ?? [];
  // set the selected run to the first run if we don't already have one
  // or if the selected run went out of scope (e.g. was deleted)
  if (runs?.length && (!selectedRunUuid || !runUuids.includes(selectedRunUuid))) {
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
      />
    );
  };

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
            />
            <ExperimentEvaluationRunsTable
              data={runsAndGroupValues}
              uniqueColumns={uniqueColumns}
              selectedColumns={selectedColumns}
              selectedRunUuid={viewMode === ExperimentEvaluationRunsPageMode.TRACES ? selectedRunUuid : undefined}
              setSelectedRunUuid={(runUuid: string) => {
                setSelectedRunUuid(runUuid);
              }}
              isLoading={isLoading}
              rowSelection={rowSelection}
              setRowSelection={setRowSelection}
              setSelectedDatasetWithRun={setSelectedDatasetWithRun}
              setIsDrawerOpen={setIsDrawerOpen}
              viewMode={viewMode}
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
