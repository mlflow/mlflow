import React, { useCallback, useEffect, useMemo, useState } from 'react';
import type { RowSelectionState } from '@tanstack/react-table';
import { isEmpty as isEmptyFn } from 'lodash';
import {
  Empty,
  ParagraphSkeleton,
  DangerIcon,
  Spacer,
  Drawer,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
} from '@databricks/design-system';
import { useLogTelemetryEvent } from '../../../../../telemetry/hooks/useLogTelemetryEvent';
import type {
  TracesTableColumn,
  TraceActions,
  GetTraceFunction,
  TableFilter,
  TraceTablePageSource,
} from '@databricks/web-shared/genai-traces-table';
import {
  shouldUseTracesV4API,
  useUnifiedTraceTagsModal,
  ModelTraceExplorerContextProvider,
  ModelTraceExplorerRunJudgesContextProvider,
  isEvaluatingTracesInDetailsViewEnabled,
  shouldEnableTracesTableStatePersistence,
  SESSION_ID_METADATA_KEY,
  MetricViewType,
  AggregationType,
  TraceMetricKey,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../../../../pages/experiment-overview/hooks/useTraceMetricsQuery';
import {
  EXECUTION_DURATION_COLUMN_ID,
  GenAiTracesMarkdownConverterProvider,
  GenAITracesTableBodyContainer,
  GenAITracesTableToolbar,
  REQUEST_TIME_COLUMN_ID,
  STATE_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  SESSION_COLUMN_ID,
  TracesTableColumnType,
  useSearchMlflowTraces,
  useSelectedColumns,
  GenAITracesTableProvider,
  useFilters,
  getTracesTagKeys,
  useTableSort,
  useMlflowTracesTableMetadata,
  TOKENS_COLUMN_ID,
  TRACE_ID_COLUMN_ID,
  invalidateMlflowSearchTracesCache,
  createTraceLocationForExperiment,
  doesTraceSupportV4API,
  GenAITracesTableBodySkeleton,
  SIMULATION_GOAL_COLUMN_ID,
  SIMULATION_PERSONA_COLUMN_ID,
  ISSUES_COLUMN_ID,
  getSimulationColumnsToAdd,
  isSqlWarehouseTimeoutError,
  shouldUseInfinitePaginatedTraces,
} from '@databricks/web-shared/genai-traces-table';
import {
  GenAiTraceTableRowSelectionProvider,
  useIsInsideGenAiTraceTableRowSelectionProvider,
} from '@databricks/web-shared/genai-traces-table';
import { useMarkdownConverter } from '@mlflow/mlflow/src/common/utils/MarkdownUtils';
import { useDeleteTracesMutation } from '../../../evaluations/hooks/useDeleteTraces';
import { useEditExperimentTraceTags } from '../../../traces/hooks/useEditExperimentTraceTags';
import { useIntl } from '@databricks/i18n';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { TracesV3EmptyState } from './TracesV3EmptyState';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { useSetInitialTimeFilter } from './hooks/useSetInitialTimeFilter';
import { checkColumnContents } from './utils/columnUtils';
import { useGetDeleteTracesAction } from './hooks/useGetDeleteTracesAction';
import { ExportTracesToDatasetModal } from '../../../../pages/experiment-evaluation-datasets/components/ExportTracesToDatasetModal';
import { useRegisterSelectedIds } from '@mlflow/mlflow/src/assistant';
import { AssistantAwareDrawer } from '@mlflow/mlflow/src/common/components/AssistantAwareDrawer';
import {
  useRunScorerInTracesViewConfiguration,
  useRunJudgesOnTracesConfiguration,
} from '../../../../pages/experiment-scorers/hooks/useRunScorerInTracesViewConfiguration';
import { IssueDetectionModal } from './IssueDetectionModal';
import { useCountInfo } from './hooks/useCountInfo';
import { useAssessmentCountMetrics } from './hooks/useAssessmentCountMetrics';

const JudgeContextProvider = ({
  children,
  runJudgeConfiguration,
}: {
  children: React.ReactNode;
  runJudgeConfiguration: ReturnType<typeof useRunScorerInTracesViewConfiguration>;
}) => {
  return (
    <ModelTraceExplorerRunJudgesContextProvider {...runJudgeConfiguration}>
      {children}
    </ModelTraceExplorerRunJudgesContextProvider>
  );
};

const ContextProviders = ({
  children,
  makeHtmlFromMarkdown,
  experimentId,
  runJudgeConfiguration,
}: {
  makeHtmlFromMarkdown: (markdown?: string) => string;
  experimentId?: string;
  children: React.ReactNode;
  runJudgeConfiguration?: ReturnType<typeof useRunScorerInTracesViewConfiguration>;
}) => {
  return (
    <GenAiTracesMarkdownConverterProvider makeHtml={makeHtmlFromMarkdown}>
      {isEvaluatingTracesInDetailsViewEnabled() && runJudgeConfiguration ? (
        <JudgeContextProvider runJudgeConfiguration={runJudgeConfiguration}>{children}</JudgeContextProvider>
      ) : (
        children
      )}
    </GenAiTracesMarkdownConverterProvider>
  );
};

const firedCountTelemetryForExperiments = new Set<string>();

const TracesV3LogsImpl = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    experimentIds,
    endpointName = '',
    timeRange,
    isLoadingExperiment,
    loggedModelId,
    forceGroupBySession = false,
    initialGroupBySession = false,
    columnStorageKeyPrefix,
    pageSource = 'experiment-traces',
    additionalFilters,
    disableActions = false,
    customDefaultSelectedColumns,
    toolbarAddons,
    drawerWidth,
  }: {
    /**
     * Array of experiment IDs to search traces for.
     */
    experimentIds: string[];
    endpointName?: string;
    timeRange?: { startTime: string | undefined; endTime: string | undefined };
    isLoadingExperiment?: boolean;
    loggedModelId?: string;
    forceGroupBySession?: boolean;
    /** Initial state for session grouping. */
    initialGroupBySession?: boolean;
    /**
     * Optional prefix for the localStorage key used to persist column selection.
     * Use this to separate column selection state between different views.
     */
    columnStorageKeyPrefix?: string;
    /**
     * Optional param to differentiate the context of the traces table
     * Currently used to log different componentIds for the detect issues button
     */
    pageSource?: TraceTablePageSource;
    additionalFilters?: TableFilter[];
    disableActions?: boolean;
    customDefaultSelectedColumns?: (column: TracesTableColumn) => boolean;
    toolbarAddons?: React.ReactNode;
    drawerWidth?: string | number;
  }) => {
    // When viewing a single experiment, pass its ID to enable experiment-specific
    // features (run name links, logged model links, session links, filter dropdowns).
    // When viewing multiple experiments, pass undefined to disable those features.
    const singleExperimentId = experimentIds.length === 1 ? experimentIds[0] : undefined;
    // Use a deterministic key for filter/column persistence so state doesn't
    // shift when the array order changes (e.g. "all endpoints" in the gateway).
    const persistenceKey = useMemo(
      () => (experimentIds.length > 1 ? [...experimentIds].sort().join(',') : (experimentIds[0] ?? '')),
      [experimentIds],
    );
    const makeHtmlFromMarkdown = useMarkdownConverter();
    const intl = useIntl();
    const enableTraceInsights = false;
    const [isGroupedBySession, setIsGroupedBySession] = useState(initialGroupBySession);
    const [isIssueDetectionModalOpen, setIsIssueDetectionModalOpen] = useState(false);

    // Check if we're already inside a provider (e.g., from SelectTracesModal)
    // If so, we won't create our own provider to avoid shadowing the parent's selection state
    const hasExternalProvider = useIsInsideGenAiTraceTableRowSelectionProvider();

    // Row selection state - lifted to provide shared state via context
    const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
    useRegisterSelectedIds('selectedTraceIds', rowSelection);

    const traceSearchLocations = useMemo(
      () => {
        return experimentIds.map((id) => createTraceLocationForExperiment(id));
      },
      // prettier-ignore
      [
        experimentIds,
      ],
    );

    const isQueryDisabled = false;
    const usesV4APIs = shouldUseTracesV4API();

    const getTrace = getTraceV3;

    // Get metadata
    const {
      assessmentInfos,
      allColumns,
      totalCount,
      evaluatedTraces,
      isLoading: isMetadataLoading,
      error: metadataError,
      isEmpty,
      tableFilterOptions,
    } = useMlflowTracesTableMetadata({
      locations: traceSearchLocations,
      timeRange,
      filterByLoggedModelId: loggedModelId,
      disabled: isQueryDisabled,
    });

    // Setup table states
    const [searchQuery, setSearchQuery] = useState<string>('');
    const [filters, setFilters] = useFilters({
      persist: shouldEnableTracesTableStatePersistence(),
      loadPersistedValues: shouldEnableTracesTableStatePersistence(),
      persistKey: persistenceKey,
    });
    const queryClient = useQueryClient();

    const combinedFilters = useMemo(() => {
      if (!additionalFilters || additionalFilters.length === 0) {
        return filters;
      }
      return [...additionalFilters, ...filters];
    }, [additionalFilters, filters]);

    const defaultSelectedColumns = useCallback(
      (allColumns: TracesTableColumn[]) => {
        const { responseHasContent, inputHasContent, tokensHasContent } = checkColumnContents(evaluatedTraces);

        if (customDefaultSelectedColumns) {
          return allColumns.filter(customDefaultSelectedColumns);
        }
        const hasSessionIds = evaluatedTraces.some((t) =>
          Boolean(t.traceInfo?.trace_metadata?.[SESSION_ID_METADATA_KEY]),
        );
        const hasIssues = evaluatedTraces.some((t) => t.issues && t.issues.length > 0);
        return allColumns.filter(
          (col) =>
            col.type === TracesTableColumnType.ASSESSMENT ||
            col.type === TracesTableColumnType.EXPECTATION ||
            (inputHasContent && col.type === TracesTableColumnType.INPUT) ||
            (responseHasContent && col.type === TracesTableColumnType.TRACE_INFO && col.id === RESPONSE_COLUMN_ID) ||
            (tokensHasContent && col.type === TracesTableColumnType.TRACE_INFO && col.id === TOKENS_COLUMN_ID) ||
            (col.type === TracesTableColumnType.TRACE_INFO &&
              [TRACE_ID_COLUMN_ID, EXECUTION_DURATION_COLUMN_ID, REQUEST_TIME_COLUMN_ID, STATE_COLUMN_ID].includes(
                col.id,
              )) ||
            col.type === TracesTableColumnType.INTERNAL_MONITOR_REQUEST_TIME ||
            (hasSessionIds &&
              [SESSION_COLUMN_ID, SIMULATION_GOAL_COLUMN_ID, SIMULATION_PERSONA_COLUMN_ID].includes(col.id)) ||
            (hasIssues && col.id === ISSUES_COLUMN_ID),
        );
      },
      [evaluatedTraces, customDefaultSelectedColumns],
    );

    const { selectedColumns, toggleColumns, setSelectedColumns } = useSelectedColumns(
      persistenceKey,
      allColumns,
      defaultSelectedColumns,
      undefined, // runUuid
      columnStorageKeyPrefix,
    );

    const onToggleSessionGrouping = useCallback(() => {
      setIsGroupedBySession((prev) => !prev);
    }, []);

    const [tableSort, setTableSort] = useTableSort(selectedColumns, {
      key: REQUEST_TIME_COLUMN_ID,
      type: TracesTableColumnType.TRACE_INFO,
      asc: false,
    });

    // Set the initial time filter when there are no traces
    const { isInitialTimeFilterLoading } = useSetInitialTimeFilter({
      locations: traceSearchLocations,
      isTracesEmpty: isEmpty,
      isTraceMetadataLoading: isMetadataLoading,
      disabled: isQueryDisabled || usesV4APIs,
    });

    // Get traces data
    const {
      data: traceInfos,
      isLoading: traceInfosLoading,
      isFetching: traceInfosFetching,
      error: traceInfosError,
      fetchNextPage,
      hasNextPage,
      isFetchingNextPage,
    } = useSearchMlflowTraces({
      locations: traceSearchLocations,
      currentRunDisplayName: endpointName,
      searchQuery,
      filters: combinedFilters,
      timeRange,
      filterByLoggedModelId: loggedModelId,
      tableSort,
      disabled: isQueryDisabled,
    });

    const deleteTracesMutation = useDeleteTracesMutation();

    // Local, legacy version of trace tag editing modal
    const { showEditTagsModalForTrace, EditTagsModal } = useEditExperimentTraceTags({
      onSuccess: () => invalidateMlflowSearchTracesCache({ queryClient }),
      existingTagKeys: getTracesTagKeys(traceInfos || []),
    });

    // Unified version of trace tag editing modal using shared components
    const { showTagAssignmentModal: showEditTagsModalForTraceUnified, TagAssignmentModal: EditTagsModalUnified } =
      useUnifiedTraceTagsModal({
        componentIdPrefix: 'mlflow.experiment-traces',
        onSuccess: () => invalidateMlflowSearchTracesCache({ queryClient }),
      });

    const deleteTracesAction = useGetDeleteTracesAction({ traceSearchLocations });

    const renderCustomExportTracesToDatasetsModal = ExportTracesToDatasetModal;

    const runJudgeConfiguration = useRunScorerInTracesViewConfiguration();

    const { showRunJudgesModal, RunJudgesModal, JudgesStatusBanner } = useRunJudgesOnTracesConfiguration(
      runJudgeConfiguration.evaluateTraces,
      runJudgeConfiguration.allEvaluations,
      runJudgeConfiguration.subscribeToScorerFinished,
    );

    const traceActions: TraceActions = useMemo(() => {
      return {
        deleteTracesAction,
        exportToEvals: true,
        // Enable unified tags modal if V4 APIs is enabled
        editTags: shouldUseTracesV4API()
          ? {
              showEditTagsModalForTrace: showEditTagsModalForTraceUnified,
              EditTagsModal: EditTagsModalUnified,
            }
          : {
              showEditTagsModalForTrace,
              EditTagsModal,
            },
        ...(isEvaluatingTracesInDetailsViewEnabled() && {
          runJudgesAction: {
            showRunJudgesModal,
            RunJudgesModal,
          },
        }),
      };
    }, [
      deleteTracesAction,
      showEditTagsModalForTraceUnified,
      EditTagsModalUnified,
      showEditTagsModalForTrace,
      EditTagsModal,
      showRunJudgesModal,
      RunJudgesModal,
    ]);

    const countInfo = useCountInfo({
      experimentIds,
      timeRange,
      traceInfosCount: traceInfos?.length,
      traceInfosLoading,
      metadataTotalCount: totalCount,
      disabled: isQueryDisabled,
    });

    const logTelemetryEvent = useLogTelemetryEvent();

    const { data: allTimeTraceCountMetrics, isLoading: allTimeTraceCountLoading } = useTraceMetricsQuery({
      experimentIds: singleExperimentId ? [singleExperimentId] : [],
      viewType: MetricViewType.TRACES,
      metricName: TraceMetricKey.TRACE_COUNT,
      aggregations: [{ aggregation_type: AggregationType.COUNT }],
      enabled:
        shouldUseInfinitePaginatedTraces() &&
        !isQueryDisabled &&
        !!singleExperimentId &&
        !firedCountTelemetryForExperiments.has(singleExperimentId),
    });
    const allTimeTotalCount = allTimeTraceCountMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT];

    useEffect(() => {
      if (
        !allTimeTraceCountLoading &&
        singleExperimentId &&
        !firedCountTelemetryForExperiments.has(singleExperimentId)
      ) {
        firedCountTelemetryForExperiments.add(singleExperimentId);
        logTelemetryEvent({
          componentId: 'mlflow.traces-tab.trace-count',
          componentType: DesignSystemEventProviderComponentTypes.Card,
          componentViewId: singleExperimentId,
          eventType: DesignSystemEventProviderAnalyticsEventTypes.OnView,
          value: JSON.stringify({
            totalTraces: allTimeTotalCount,
          }),
        });
      }
    }, [allTimeTraceCountLoading, allTimeTotalCount, singleExperimentId, logTelemetryEvent]);

    const assessmentCountMetrics = useAssessmentCountMetrics({
      experimentIds,
      timeRange,
      disabled: isQueryDisabled,
    });

    // Loading state:
    // - Show skeleton only during initial metadata loading (or initial time filter loading for empty check)
    // - Once metadata is loaded, keep the table mounted and use loading overlay for subsequent fetches
    const showInitialSkeleton = isMetadataLoading || isInitialTimeFilterLoading;

    // Show loading overlay when fetching new data (but not during initial load)

    const tableError = traceInfosError || metadataError;
    const isTableEmpty = isEmpty && !showInitialSkeleton && !traceInfosLoading && !traceInfosFetching && !tableError;
    // When fetching the next page of infinite results, keep the existing rows visible
    // and preserve scroll position instead of showing the loading skeleton.
    const isTableLoading = !showInitialSkeleton && (traceInfosLoading || (traceInfosFetching && !isFetchingNextPage));

    // Helper function to render the main content based on current state
    const renderMainContent = () => {
      if (!enableTraceInsights && isTableEmpty) {
        return (
          <div css={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
            <TracesV3EmptyState
              experimentIds={experimentIds}
              loggedModelId={loggedModelId}
              traceSearchLocations={traceSearchLocations}
              isCallDisabled={isQueryDisabled}
            />
            {/* still render the table body container so the trace drawer can open even when
             table is empty (e.g., when navigating directly to a trace via URL parameter) */}
            <div css={{ display: 'none' }}>
              <GenAITracesTableBodyContainer
                experimentId={singleExperimentId}
                allColumns={allColumns}
                currentTraceInfoV3={traceInfos || []}
                currentRunDisplayName={endpointName}
                getTrace={getTrace}
                assessmentInfos={assessmentInfos}
                setFilters={setFilters}
                filters={filters}
                selectedColumns={selectedColumns}
                tableSort={tableSort}
                onTraceTagsEdit={showEditTagsModalForTrace}
                isTableLoading={isTableLoading}
                isGroupedBySession={forceGroupBySession || isGroupedBySession}
                searchQuery={searchQuery}
                fetchNextPage={fetchNextPage}
                hasNextPage={hasNextPage}
                isFetchingNextPage={isFetchingNextPage}
                assessmentCountMetrics={assessmentCountMetrics}
              />
            </div>
          </div>
        );
      }
      // Default traces view
      return (
        <div
          css={{
            display: 'flex',
            flex: 1,
            overflow: 'hidden',
          }}
        >
          <div
            css={{
              display: 'flex',
              flex: 1,
              overflow: 'hidden',
            }}
          >
            {showInitialSkeleton ? (
              <GenAITracesTableBodySkeleton />
            ) : tableError ? (
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '100%',
                  height: '100%',
                }}
              >
                <Empty
                  image={<DangerIcon />}
                  title={intl.formatMessage({
                    defaultMessage: 'Fetching traces failed',
                    description: 'Evaluation review > evaluations list > error state title',
                  })}
                  description={
                    isSqlWarehouseTimeoutError(tableError)
                      ? intl.formatMessage({
                          defaultMessage:
                            'The SQL query timed out. Please retry, and if the problem persists, try selecting a larger SQL warehouse.',
                          description:
                            'Evaluation review > evaluations list > SQL warehouse timeout error description with CTA to select larger warehouse',
                        })
                      : tableError.message
                  }
                />
              </div>
            ) : (
              <ContextProviders
                makeHtmlFromMarkdown={makeHtmlFromMarkdown}
                experimentId={singleExperimentId}
                runJudgeConfiguration={runJudgeConfiguration}
              >
                <GenAITracesTableBodyContainer
                  experimentId={singleExperimentId}
                  allColumns={allColumns}
                  currentTraceInfoV3={traceInfos || []}
                  currentRunDisplayName={endpointName}
                  getTrace={getTrace}
                  assessmentInfos={assessmentInfos}
                  setFilters={setFilters}
                  filters={filters}
                  selectedColumns={selectedColumns}
                  tableSort={tableSort}
                  onTraceTagsEdit={showEditTagsModalForTrace}
                  isTableLoading={isTableLoading}
                  isGroupedBySession={forceGroupBySession || isGroupedBySession}
                  searchQuery={searchQuery}
                  fetchNextPage={fetchNextPage}
                  hasNextPage={hasNextPage}
                  isFetchingNextPage={isFetchingNextPage}
                  assessmentCountMetrics={assessmentCountMetrics}
                />
              </ContextProviders>
            )}
          </div>
        </div>
      );
    };

    // Single unified layout with toolbar and content
    const tableContent = (
      <ModelTraceExplorerContextProvider
        renderExportTracesToDatasetsModal={renderCustomExportTracesToDatasetsModal}
        DrawerComponent={AssistantAwareDrawer}
        drawerWidth={drawerWidth}
      >
        <GenAITracesTableProvider
          experimentId={singleExperimentId}
          getTrace={getTrace}
          isGroupedBySession={forceGroupBySession || isGroupedBySession}
        >
          <div
            css={{
              overflow: 'hidden',
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <GenAITracesTableToolbar
              pageSource={pageSource}
              experimentId={singleExperimentId}
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              filters={filters}
              setFilters={setFilters}
              assessmentInfos={assessmentInfos}
              traceInfos={traceInfos}
              tableFilterOptions={tableFilterOptions}
              countInfo={countInfo}
              traceActions={traceActions}
              tableSort={tableSort}
              setTableSort={setTableSort}
              allColumns={allColumns}
              selectedColumns={selectedColumns}
              toggleColumns={toggleColumns}
              setSelectedColumns={setSelectedColumns}
              isMetadataLoading={isMetadataLoading}
              metadataError={metadataError}
              usesV4APIs={usesV4APIs}
              addons={toolbarAddons}
              isGroupedBySession={forceGroupBySession || isGroupedBySession}
              forceGroupBySession={forceGroupBySession}
              onToggleSessionGrouping={onToggleSessionGrouping}
              onDetectIssues={disableActions ? undefined : () => setIsIssueDetectionModalOpen(true)}
            />
            {JudgesStatusBanner}
            {renderMainContent()}
          </div>
          {!disableActions && isIssueDetectionModalOpen && (
            <IssueDetectionModal
              onClose={() => setIsIssueDetectionModalOpen(false)}
              experimentId={singleExperimentId}
              initialSelectedTraceIds={Object.entries(rowSelection)
                .filter(([, isSelected]) => isSelected)
                .map(([traceId]) => traceId)}
              availableTraceIds={traceInfos?.map((trace) => trace.trace_id) ?? []}
              defaultGroupBySession={forceGroupBySession || isGroupedBySession}
            />
          )}
        </GenAITracesTableProvider>
      </ModelTraceExplorerContextProvider>
    );

    // If we're already inside an external provider (e.g., from SelectTracesModal),
    // don't create a new provider to avoid shadowing the parent's selection state
    if (hasExternalProvider) {
      return tableContent;
    }

    return (
      <GenAiTraceTableRowSelectionProvider rowSelection={rowSelection} setRowSelection={setRowSelection}>
        {tableContent}
      </GenAiTraceTableRowSelectionProvider>
    );
  },
);

export const TracesV3Logs = TracesV3LogsImpl;
