import React, { useCallback, useEffect, useMemo, useState } from 'react';
import type { RowSelectionState } from '@tanstack/react-table';
import { isEmpty as isEmptyFn } from 'lodash';
import { Empty, ParagraphSkeleton, DangerIcon } from '@databricks/design-system';
import type {
  TracesTableColumn,
  TraceActions,
  GetTraceFunction,
  TableFilter,
} from '@databricks/web-shared/genai-traces-table';
import {
  shouldUseTracesV4API,
  useUnifiedTraceTagsModal,
  ModelTraceExplorerContextProvider,
  ModelTraceExplorerRunJudgesContextProvider,
} from '@databricks/web-shared/model-trace-explorer';
import {
  EXECUTION_DURATION_COLUMN_ID,
  GenAiTracesMarkdownConverterProvider,
  GenAITracesTableBodyContainer,
  GenAITracesTableToolbar,
  REQUEST_TIME_COLUMN_ID,
  STATE_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  TracesTableColumnType,
  useSearchMlflowTraces,
  useSelectedColumns,
  getEvalTabTotalTracesLimit,
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
} from '@databricks/web-shared/genai-traces-table';
import {
  GenAiTraceTableRowSelectionProvider,
  useIsInsideGenAiTraceTableRowSelectionProvider,
} from '@databricks/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import { useMarkdownConverter } from '@mlflow/mlflow/src/common/utils/MarkdownUtils';
import { shouldEnableTraceInsights } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
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
import { useRunScorerInTracesViewConfiguration } from '../../../../pages/experiment-scorers/hooks/useRunScorerInTracesView';

const ContextProviders = ({
  children,
  makeHtmlFromMarkdown,
  experimentId,
}: {
  makeHtmlFromMarkdown: (markdown?: string) => string;
  experimentId?: string;
  children: React.ReactNode;
}) => {
  const runJudgeConfiguration = useRunScorerInTracesViewConfiguration();
  return (
    <GenAiTracesMarkdownConverterProvider makeHtml={makeHtmlFromMarkdown}>
      <ModelTraceExplorerRunJudgesContextProvider renderRunJudgeButton={runJudgeConfiguration.renderRunJudgeButton}>
        {children}
      </ModelTraceExplorerRunJudgesContextProvider>
    </GenAiTracesMarkdownConverterProvider>
  );
};

const TracesV3LogsImpl = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    experimentId,
    endpointName = '',
    timeRange,
    isLoadingExperiment,
    loggedModelId,
    additionalFilters,
    disableActions = false,
    customDefaultSelectedColumns,
    toolbarAddons,
    forceGroupBySession = false,
    columnStorageKeyPrefix,
  }: {
    experimentId: string;
    endpointName?: string;
    timeRange?: { startTime: string | undefined; endTime: string | undefined };
    isLoadingExperiment?: boolean;
    loggedModelId?: string;
    additionalFilters?: TableFilter[];
    disableActions?: boolean;
    customDefaultSelectedColumns?: (column: TracesTableColumn) => boolean;
    toolbarAddons?: React.ReactNode;
    forceGroupBySession?: boolean;
    /**
     * Optional prefix for the localStorage key used to persist column selection.
     * Use this to separate column selection state between different views.
     */
    columnStorageKeyPrefix?: string;
  }) => {
    const makeHtmlFromMarkdown = useMarkdownConverter();
    const intl = useIntl();
    const enableTraceInsights = shouldEnableTraceInsights();
    const [isGroupedBySession, setIsGroupedBySession] = useState(false);

    // Check if we're already inside a provider (e.g., from SelectTracesModal)
    // If so, we won't create our own provider to avoid shadowing the parent's selection state
    const hasExternalProvider = useIsInsideGenAiTraceTableRowSelectionProvider();

    // Row selection state - lifted to provide shared state via context
    const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
    useRegisterSelectedIds('selectedTraceIds', rowSelection);

    const onToggleSessionGrouping = useCallback(() => {
      setIsGroupedBySession(!isGroupedBySession);
    }, [isGroupedBySession]);

    const traceSearchLocations = useMemo(
      () => {
        return [createTraceLocationForExperiment(experimentId)];
      },
      // prettier-ignore
      [
        experimentId,
      ],
    );

    const isQueryDisabled = false;
    const usesV4APIs = true;

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
    const [filters, setFilters] = useFilters();
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
            col.type === TracesTableColumnType.INTERNAL_MONITOR_REQUEST_TIME,
        );
      },
      [evaluatedTraces, customDefaultSelectedColumns],
    );

    const { selectedColumns, toggleColumns, setSelectedColumns } = useSelectedColumns(
      experimentId,
      allColumns,
      defaultSelectedColumns,
      undefined, // runUuid
      columnStorageKeyPrefix,
    );

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

    const traceActions: TraceActions = useMemo(() => {
      if (disableActions) {
        return {
          deleteTracesAction: undefined,
          exportToEvals: undefined,
          editTags: undefined,
        };
      }
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
      };
    }, [
      deleteTracesAction,
      showEditTagsModalForTraceUnified,
      EditTagsModalUnified,
      showEditTagsModalForTrace,
      EditTagsModal,
      disableActions,
    ]);

    const countInfo = useMemo(() => {
      return {
        currentCount: traceInfos?.length,
        logCountLoading: traceInfosLoading,
        totalCount: totalCount,
        maxAllowedCount: getEvalTabTotalTracesLimit(),
      };
    }, [traceInfos, totalCount, traceInfosLoading]);

    const isTableLoading = traceInfosLoading || isInitialTimeFilterLoading || isMetadataLoading;
    const displayLoadingOverlay = false;

    const tableError = traceInfosError || metadataError;
    const isTableEmpty = isEmpty && !isTableLoading && !tableError;

    // Helper function to render the main content based on current state
    const renderMainContent = () => {
      // If isEmpty and not enableTraceInsights, show empty state without navigation
      if (!enableTraceInsights && isTableEmpty) {
        return (
          <TracesV3EmptyState
            experimentIds={[experimentId]}
            loggedModelId={loggedModelId}
            traceSearchLocations={traceSearchLocations}
            isCallDisabled={isQueryDisabled}
          />
        );
      }
      // Default traces view with optional navigation
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
            {isTableLoading ? (
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  width: '100%',
                  gap: '8px',
                  padding: '16px',
                }}
              >
                {[...Array(10).keys()].map((i) => (
                  <ParagraphSkeleton label="Loading..." key={i} seed={`s-${i}`} />
                ))}
              </div>
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
                  description={tableError.message}
                />
              </div>
            ) : (
              <ContextProviders makeHtmlFromMarkdown={makeHtmlFromMarkdown} experimentId={experimentId}>
                <GenAITracesTableBodyContainer
                  experimentId={experimentId}
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
                  displayLoadingOverlay={displayLoadingOverlay}
                  isGroupedBySession={forceGroupBySession || isGroupedBySession}
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
      >
        <GenAITracesTableProvider experimentId={experimentId} isGroupedBySession={isGroupedBySession}>
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
            />
            {renderMainContent()}
          </div>
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
