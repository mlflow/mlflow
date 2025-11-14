import React, { useCallback, useMemo, useState } from 'react';
import { isEmpty as isEmptyFn } from 'lodash';
import { Empty, ParagraphSkeleton, DangerIcon } from '@databricks/design-system';
import type { TracesTableColumn, TraceActions, GetTraceFunction } from '@databricks/web-shared/genai-traces-table';
import { shouldUseTracesV4API, useUnifiedTraceTagsModal } from '@databricks/web-shared/model-trace-explorer';
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
} from '@databricks/web-shared/genai-traces-table';
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
import { useExportTracesToDatasetModal } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-evaluation-datasets/hooks/useExportTracesToDatasetModal';

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

const TracesV3LogsImpl = React.memo(
  ({
    experimentId,
    endpointName,
    timeRange,
    isLoadingExperiment,
    loggedModelId,
  }: {
    experimentId: string;
    endpointName: string;
    timeRange?: { startTime: string | undefined; endTime: string | undefined };
    isLoadingExperiment?: boolean;
    loggedModelId?: string;
  }) => {
    const makeHtmlFromMarkdown = useMarkdownConverter();
    const intl = useIntl();
    const enableTraceInsights = shouldEnableTraceInsights();

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

    const defaultSelectedColumns = useCallback(
      (allColumns: TracesTableColumn[]) => {
        const { responseHasContent, inputHasContent, tokensHasContent } = checkColumnContents(evaluatedTraces);

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
      [evaluatedTraces],
    );

    const { selectedColumns, toggleColumns, setSelectedColumns } = useSelectedColumns(
      experimentId,
      allColumns,
      defaultSelectedColumns,
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
      disabled: isQueryDisabled,
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
      filters,
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

    const { showExportTracesToDatasetsModal, setShowExportTracesToDatasetsModal, renderExportTracesToDatasetsModal } =
      useExportTracesToDatasetModal({
        experimentId,
      });

    const traceActions: TraceActions = useMemo(() => {
      return {
        deleteTracesAction: {
          deleteTraces: (experimentId: string, traceIds: string[]) =>
            deleteTracesMutation.mutateAsync({ experimentId, traceRequestIds: traceIds }),
        },
        exportToEvals: {
          showExportTracesToDatasetsModal,
          setShowExportTracesToDatasetsModal,
          renderExportTracesToDatasetsModal,
        },
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
      showExportTracesToDatasetsModal,
      setShowExportTracesToDatasetsModal,
      renderExportTracesToDatasetsModal,
      showEditTagsModalForTraceUnified,
      EditTagsModalUnified,
      showEditTagsModalForTrace,
      EditTagsModal,
      deleteTracesMutation,
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
                />
              </ContextProviders>
            )}
          </div>
        </div>
      );
    };

    // Single unified layout with toolbar and content
    return (
      <GenAITracesTableProvider>
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
          />
          {renderMainContent()}
        </div>
      </GenAITracesTableProvider>
    );
  },
);

export const TracesV3Logs = TracesV3LogsImpl;
