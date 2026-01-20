import type { RowSelectionState } from '@tanstack/react-table';
import { isNil } from 'lodash';
import { ParagraphSkeleton, Typography, Empty } from '@databricks/design-system';
import { type KeyValueEntity } from '../../../common/types';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useCompareToRunUuid } from './hooks/useCompareToRunUuid';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { FormattedMessage } from 'react-intl';
import { RunColorPill } from '../experiment-page/components/RunColorPill';
import { EvaluationRunCompareSelector } from './EvaluationRunCompareSelector';
import { getEvalTabTotalTracesLimit } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import type { TracesTableColumn, TraceActions, GetTraceFunction } from '@databricks/web-shared/genai-traces-table';
import {
  EXECUTION_DURATION_COLUMN_ID,
  GenAiTracesMarkdownConverterProvider,
  RUN_EVALUATION_RESULTS_TAB_COMPARE_RUNS,
  RUN_EVALUATION_RESULTS_TAB_SINGLE_RUN,
  getTracesTagKeys,
  STATE_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  TracesTableColumnType,
  useMlflowTracesTableMetadata,
  useSelectedColumns,
  useSearchMlflowTraces,
  GenAITracesTableToolbar,
  GenAITracesTableProvider,
  GenAITracesTableBodyContainer,
  useGenAiTraceEvaluationArtifacts,
  useFilters,
  useTableSort,
  TOKENS_COLUMN_ID,
  invalidateMlflowSearchTracesCache,
  TRACE_ID_COLUMN_ID,
  shouldUseTracesV4API,
  createTraceLocationForExperiment,
  createTraceLocationForUCSchema,
  useFetchTraceV4LazyQuery,
  doesTraceSupportV4API,
} from '@databricks/web-shared/genai-traces-table';
import { GenAiTraceTableRowSelectionProvider } from '@databricks/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import { useRegisterSelectedIds } from '@mlflow/mlflow/src/assistant';
import { useRunLoggedTraceTableArtifacts } from './hooks/useRunLoggedTraceTableArtifacts';
import { useMarkdownConverter } from '../../../common/utils/MarkdownUtils';
import { useEditExperimentTraceTags } from '../traces/hooks/useEditExperimentTraceTags';
import { useCallback, useMemo, useState } from 'react';
import { RunViewEvaluationsTabArtifacts } from './RunViewEvaluationsTabArtifacts';
import { useGetExperimentRunColor } from '../experiment-page/hooks/useExperimentRunColor';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { checkColumnContents } from '../experiment-page/components/traces-v3/utils/columnUtils';
import type {
  ModelTraceLocationMlflowExperiment,
  ModelTraceLocationUcSchema,
} from '@databricks/web-shared/model-trace-explorer';
import {
  isV3ModelTraceInfo,
  ModelTraceExplorerContextProvider,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import type { UseGetRunQueryResponseExperiment } from '../run-page/hooks/useGetRunQuery';
import type { ExperimentEntity } from '../../types';
import { useGetDeleteTracesAction } from '../experiment-page/components/traces-v3/hooks/useGetDeleteTracesAction';
import { useIntl } from 'react-intl';
import { ExportTracesToDatasetModal } from '../../pages/experiment-evaluation-datasets/components/ExportTracesToDatasetModal';
import { useSearchRunsQuery } from '../run-page/hooks/useSearchRunsQuery';
import { AssistantAwareDrawer } from '@mlflow/mlflow/src/common/components/AssistantAwareDrawer';

const ContextProviders = ({
  children,
  makeHtmlFromMarkdown,
  experimentId,
}: {
  makeHtmlFromMarkdown: (markdown?: string) => string;
  experimentId?: string;
  children: React.ReactNode;
}) => {
  return (
    <GenAiTracesMarkdownConverterProvider makeHtml={makeHtmlFromMarkdown}>
      {children}
    </GenAiTracesMarkdownConverterProvider>
  );
};

const RunViewEvaluationsTabInner = ({
  experimentId,
  runUuid,
  runDisplayName,
  setCurrentRunUuid,
  showCompareSelector = false,
  showRefreshButton = false,
}: {
  experimentId: string;
  runUuid: string;
  runDisplayName: string;
  setCurrentRunUuid?: (runUuid: string) => void;
  showCompareSelector?: boolean;
  compareToRunUuid?: string;
  showRefreshButton?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const makeHtmlFromMarkdown = useMarkdownConverter();
  const [compareToRunUuid, setCompareToRunUuid] = useCompareToRunUuid();
  const [isGroupedBySession, setIsGroupedBySession] = useState(false);

  const onToggleSessionGrouping = useCallback(() => {
    setIsGroupedBySession(!isGroupedBySession);
  }, [isGroupedBySession]);

  const traceLocations = useMemo(() => [createTraceLocationForExperiment(experimentId)], [experimentId]);
  const getTrace = getTraceV3;
  const isQueryDisabled = false;

  // Get table metadata
  const {
    assessmentInfos,
    allColumns,
    totalCount,
    evaluatedTraces,
    otherEvaluatedTraces,
    isLoading: isTableMetadataLoading,
    error: tableMetadataError,
    tableFilterOptions,
  } = useMlflowTracesTableMetadata({
    locations: traceLocations,
    runUuid,
    otherRunUuid: compareToRunUuid,
    disabled: isQueryDisabled,
    filterByAssessmentSourceRun: true,
  });

  // Setup table states
  // Row selection state - lifted to provide shared state via context
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  useRegisterSelectedIds('selectedTraceIds', rowSelection);

  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filters, setFilters] = useFilters();
  const getRunColor = useGetExperimentRunColor();
  const queryClient = useQueryClient();

  const defaultSelectedColumns = useCallback(
    (columns: TracesTableColumn[]) => {
      const { responseHasContent, inputHasContent, tokensHasContent } = checkColumnContents(
        evaluatedTraces.concat(otherEvaluatedTraces),
      );

      return columns.filter(
        (col) =>
          col.type === TracesTableColumnType.ASSESSMENT ||
          col.type === TracesTableColumnType.EXPECTATION ||
          (inputHasContent && col.type === TracesTableColumnType.INPUT) ||
          (responseHasContent && col.type === TracesTableColumnType.TRACE_INFO && col.id === RESPONSE_COLUMN_ID) ||
          (tokensHasContent && col.type === TracesTableColumnType.TRACE_INFO && col.id === TOKENS_COLUMN_ID) ||
          (col.type === TracesTableColumnType.TRACE_INFO &&
            [TRACE_ID_COLUMN_ID, EXECUTION_DURATION_COLUMN_ID, STATE_COLUMN_ID].includes(col.id)),
      );
    },
    [evaluatedTraces, otherEvaluatedTraces],
  );

  const { selectedColumns, toggleColumns, setSelectedColumns } = useSelectedColumns(
    experimentId,
    allColumns,
    defaultSelectedColumns,
    runUuid,
  );

  const [tableSort, setTableSort] = useTableSort(selectedColumns);

  // Get traces data
  const {
    data: traceInfos,
    isLoading: traceInfosLoading,
    isFetching: traceInfosFetching,
    error: traceInfosError,
    refetchMlflowTraces,
  } = useSearchMlflowTraces({
    locations: traceLocations,
    currentRunDisplayName: runDisplayName,
    searchQuery,
    filters,
    runUuid,
    tableSort,
    disabled: isQueryDisabled,
    filterByAssessmentSourceRun: true,
  });

  const {
    data: compareToRunData,
    displayName: compareToRunDisplayName,
    loading: compareToRunLoading,
  } = useGetCompareToData({
    experimentId,
    traceLocations,
    compareToRunUuid,
    isQueryDisabled,
  });

  const countInfo = useMemo(() => {
    return {
      currentCount: traceInfos?.length,
      logCountLoading: traceInfosLoading,
      totalCount: totalCount,
      maxAllowedCount: getEvalTabTotalTracesLimit(),
    };
  }, [traceInfos, traceInfosLoading, totalCount]);

  // TODO: We should update this to use web-shared/unified-tagging components for the
  // tag editor and react-query mutations for the apis.
  const { showEditTagsModalForTrace, EditTagsModal } = useEditExperimentTraceTags({
    onSuccess: () => invalidateMlflowSearchTracesCache({ queryClient }),
    existingTagKeys: getTracesTagKeys(traceInfos || []),
  });

  const deleteTracesAction = useGetDeleteTracesAction({ traceSearchLocations: traceLocations });
  const renderCustomExportTracesToDatasetsModal = ExportTracesToDatasetModal;

  const traceActions: TraceActions = useMemo(() => {
    return {
      deleteTracesAction,
      exportToEvals: true,
      // Enable unified tags modal if V4 APIs is enabled
      editTags: {
        showEditTagsModalForTrace,
        EditTagsModal,
      },
    };
  }, [deleteTracesAction, showEditTagsModalForTrace, EditTagsModal]);

  const isTableLoading = traceInfosLoading || compareToRunLoading;
  const displayLoadingOverlay = false;

  const selectedRunColor = getRunColor(runUuid);
  const compareToRunColor = compareToRunUuid ? getRunColor(compareToRunUuid) : undefined;
  const [warehousesLoading, setWarehousesLoading] = useState(false);

  if (isTableMetadataLoading) {
    return <LoadingSkeleton />;
  }

  if (tableMetadataError) {
    return (
      <div>
        <pre>{String(tableMetadataError)}</pre>
      </div>
    );
  }

  return (
    <div
      css={{
        marginTop: theme.spacing.sm,
        width: '100%',
        overflowY: 'hidden',
      }}
    >
      {!showCompareSelector && (
        <div
          css={{
            width: '100%',
            padding: `${theme.spacing.xs}px 0`,
          }}
        >
          <EvaluationRunCompareSelector
            experimentId={experimentId}
            currentRunUuid={runUuid}
            compareToRunUuid={compareToRunUuid}
            setCompareToRunUuid={setCompareToRunUuid}
            setCurrentRunUuid={setCurrentRunUuid}
          />
        </div>
      )}
      {showCompareSelector && compareToRunUuid && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            width: '100%',
            paddingBottom: theme.spacing.sm,
            gap: theme.spacing.sm,
          }}
        >
          <Typography.Text>
            <FormattedMessage defaultMessage="Comparing" description="Comparing" />
          </Typography.Text>
          <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
            {selectedRunColor && <RunColorPill color={selectedRunColor} />}
            <Typography.Text bold>{runDisplayName}</Typography.Text>
          </span>
          <Typography.Text>
            <FormattedMessage defaultMessage="to" description="to" />
          </Typography.Text>
          <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
            {compareToRunColor && <RunColorPill color={compareToRunColor} />}
            <Typography.Text bold>{compareToRunDisplayName}</Typography.Text>
          </span>
        </div>
      )}
      <ModelTraceExplorerContextProvider
        renderExportTracesToDatasetsModal={renderCustomExportTracesToDatasetsModal}
        DrawerComponent={AssistantAwareDrawer}
      >
        <GenAiTraceTableRowSelectionProvider rowSelection={rowSelection} setRowSelection={setRowSelection}>
          <GenAITracesTableProvider experimentId={experimentId}>
            <div
              css={{
                overflowY: 'hidden',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <GenAITracesTableToolbar
                experimentId={experimentId}
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                filters={filters}
                setFilters={setFilters}
                assessmentInfos={assessmentInfos}
                countInfo={countInfo}
                traceActions={traceActions}
                tableSort={tableSort}
                setTableSort={setTableSort}
                allColumns={allColumns}
                selectedColumns={selectedColumns}
                setSelectedColumns={setSelectedColumns}
                toggleColumns={toggleColumns}
                traceInfos={traceInfos}
                tableFilterOptions={tableFilterOptions}
                onRefresh={showRefreshButton ? refetchMlflowTraces : undefined}
                isRefreshing={showRefreshButton ? traceInfosFetching : undefined}
                isGroupedBySession={isGroupedBySession}
                onToggleSessionGrouping={onToggleSessionGrouping}
              />
              {
                // prettier-ignore
                isTableLoading ? (
            <LoadingSkeleton />
          ) : traceInfosError ? (
            <div>
              <pre>{String(traceInfosError)}</pre>
            </div>
          ) : (
            <ContextProviders makeHtmlFromMarkdown={makeHtmlFromMarkdown} experimentId={experimentId}>
              <GenAITracesTableBodyContainer
                experimentId={experimentId}
                currentRunDisplayName={runDisplayName}
                compareToRunDisplayName={compareToRunDisplayName}
                compareToRunUuid={compareToRunUuid}
                getTrace={getTrace}
                getRunColor={getRunColor}
                assessmentInfos={assessmentInfos}
                setFilters={setFilters}
                filters={filters}
                selectedColumns={selectedColumns}
                allColumns={allColumns}
                tableSort={tableSort}
                currentTraceInfoV3={traceInfos || []}
                compareToTraceInfoV3={compareToRunData}
                onTraceTagsEdit={showEditTagsModalForTrace}
                displayLoadingOverlay={displayLoadingOverlay}
                isGroupedBySession={isGroupedBySession}
              />
            </ContextProviders>
          )
              }
              {EditTagsModal}
            </div>
          </GenAITracesTableProvider>
        </GenAiTraceTableRowSelectionProvider>
      </ModelTraceExplorerContextProvider>
    </div>
  );
};

export const RunViewEvaluationsTab = ({
  experimentId,
  experiment,
  runUuid,
  runTags,
  runDisplayName,
  setCurrentRunUuid,
  showCompareSelector = false,
  showRefreshButton = false,
}: {
  experimentId: string;
  experiment?: ExperimentEntity | UseGetRunQueryResponseExperiment;
  runUuid: string;
  runTags?: Record<string, KeyValueEntity>;
  runDisplayName: string;
  // used in evaluation runs tab
  setCurrentRunUuid?: (runUuid: string) => void;
  showCompareSelector?: boolean;
  showRefreshButton?: boolean;
}) => {
  // Determine which tables are logged in the run
  const traceTablesLoggedInRun = useRunLoggedTraceTableArtifacts(runTags);
  const isArtifactCallEnabled = Boolean(runUuid);

  const { data: artifactData, isLoading: isArtifactLoading } = useGenAiTraceEvaluationArtifacts(
    {
      runUuid: runUuid || '',
      ...{ artifacts: traceTablesLoggedInRun ? traceTablesLoggedInRun : undefined },
    },
    { disabled: !isArtifactCallEnabled },
  );

  if (isArtifactLoading) {
    return <LoadingSkeleton />;
  }

  if (!isNil(artifactData) && artifactData.length > 0) {
    return (
      <RunViewEvaluationsTabArtifacts
        experimentId={experimentId}
        runUuid={runUuid}
        runDisplayName={runDisplayName}
        data={artifactData}
        runTags={runTags}
      />
    );
  }

  return (
    <RunViewEvaluationsTabInner
      experimentId={experimentId}
      runUuid={runUuid}
      runDisplayName={runDisplayName}
      setCurrentRunUuid={setCurrentRunUuid}
      showCompareSelector={showCompareSelector}
      showRefreshButton={showRefreshButton}
    />
  );
};

const LoadingSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'block', marginTop: theme.spacing.md, height: '100%', width: '100%' }}>
      {[...Array(10).keys()].map((i) => (
        <ParagraphSkeleton label="Loading..." key={i} seed={`s-${i}`} />
      ))}
    </div>
  );
};

const useGetCompareToData = (params: {
  experimentId: string;
  traceLocations: ModelTraceLocationUcSchema[] | ModelTraceLocationMlflowExperiment[];
  compareToRunUuid: string | undefined;
  isQueryDisabled?: boolean;
}): {
  data: ModelTraceInfoV3[] | undefined;
  displayName: string;
  loading: boolean;
} => {
  const { compareToRunUuid, experimentId, traceLocations, isQueryDisabled } = params;
  const { data: traceInfos, isLoading: traceInfosLoading } = useSearchMlflowTraces({
    locations: traceLocations,
    currentRunDisplayName: undefined,
    runUuid: compareToRunUuid,
    disabled: isNil(compareToRunUuid) || isQueryDisabled,
    filterByAssessmentSourceRun: true,
  });

  const { data: runData, loading: runDetailsLoading } = useSearchRunsQuery({
    experimentIds: [experimentId],
    filter: `attributes.run_id = "${compareToRunUuid}"`,
    disabled: isNil(compareToRunUuid),
  });

  return {
    data: traceInfos,
    displayName: Utils.getRunDisplayName(runData?.info, compareToRunUuid),
    loading: traceInfosLoading || runDetailsLoading,
  };
};
