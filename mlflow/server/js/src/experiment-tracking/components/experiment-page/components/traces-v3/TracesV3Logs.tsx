import React, { useCallback, useMemo, useState } from 'react';
import { Empty, ParagraphSkeleton, DangerIcon } from '@databricks/design-system';
import {
  EXECUTION_DURATION_COLUMN_ID,
  GenAiTracesMarkdownConverterProvider,
  GenAITracesTableBodyContainer,
  GenAITracesTableToolbar,
  REQUEST_TIME_COLUMN_ID,
  STATE_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  TracesTableColumn,
  TracesTableColumnType,
  useSearchMlflowTraces,
  useSelectedColumns,
  getEvalTabTotalTracesLimit,
  GenAITracesTableProvider,
  TraceActions,
  useFilters,
  getTracesTagKeys,
  useTableSort,
  useMlflowTracesTableMetadata,
  TOKENS_COLUMN_ID,
  invalidateMlflowSearchTracesCache,
} from '@databricks/web-shared/genai-traces-table';
import { MlflowService } from '@mlflow/mlflow/src/experiment-tracking/sdk/MlflowService';
import { useMarkdownConverter } from '@mlflow/mlflow/src/common/utils/MarkdownUtils';
import { shouldEnableTraceInsights } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { useDeleteTracesMutation } from '../../../evaluations/hooks/useDeleteTraces';
import { useEditExperimentTraceTags } from '../../../traces/hooks/useEditExperimentTraceTags';
import { useIntl } from '@databricks/i18n';
import { TracesV3EmptyState } from './TracesV3EmptyState';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { useSetInitialTimeFilter } from './hooks/useSetInitialTimeFilter';

export const TracesV3Logs = React.memo(
  ({
    experimentId,
    endpointName,
    timeRange,
    loggedModelId,
  }: {
    experimentId: string;
    endpointName: string;
    timeRange?: { startTime: string | undefined; endTime: string | undefined };
    loggedModelId?: string;
  }) => {
    const makeHtmlFromMarkdown = useMarkdownConverter();
    const intl = useIntl();
    const enableTraceInsights = shouldEnableTraceInsights();

    // Get metadata
    const {
      assessmentInfos,
      allColumns,
      totalCount,
      isLoading: isMetadataLoading,
      error: metadataError,
      isEmpty,
      tableFilterOptions,
    } = useMlflowTracesTableMetadata({
      experimentId,
      timeRange,
      filterByLoggedModelId: loggedModelId,
    });

    // Setup table states
    const [searchQuery, setSearchQuery] = useState<string>('');
    const [filters, setFilters] = useFilters();
    const queryClient = useQueryClient();

    const defaultSelectedColumns = useCallback((columns: TracesTableColumn[]) => {
      return columns.filter(
        (col) =>
          col.type === TracesTableColumnType.ASSESSMENT ||
          col.type === TracesTableColumnType.INPUT ||
          (col.type === TracesTableColumnType.TRACE_INFO &&
            [
              EXECUTION_DURATION_COLUMN_ID,
              RESPONSE_COLUMN_ID,
              REQUEST_TIME_COLUMN_ID,
              STATE_COLUMN_ID,
              TOKENS_COLUMN_ID,
            ].includes(col.id)) ||
          col.type === TracesTableColumnType.INTERNAL_MONITOR_REQUEST_TIME,
      );
    }, []);

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
      experimentId,
      isTracesEmpty: isEmpty,
      isTraceMetadataLoading: isMetadataLoading,
    });

    // Get traces data
    const {
      data: traceInfos,
      isLoading: traceInfosLoading,
      error: traceInfosError,
    } = useSearchMlflowTraces({
      experimentId,
      currentRunDisplayName: endpointName,
      searchQuery,
      filters,
      timeRange,
      filterByLoggedModelId: loggedModelId,
      tableSort,
    });

    // Get trace function
    // Only traceId is used for trace v3
    const getTrace = useCallback(async (requestId?: string, traceId?: string) => {
      const [traceInfoResponse, traceData] = await Promise.all([
        traceId ? MlflowService.getExperimentTraceInfoV3(traceId) : undefined,
        traceId ? MlflowService.getExperimentTraceData(traceId) : undefined,
      ]);
      return traceData
        ? {
            info: traceInfoResponse?.trace?.trace_info || {},
            data: traceData,
          }
        : undefined;
    }, []);

    const deleteTracesMutation = useDeleteTracesMutation();

    // TODO: We should update this to use web-shared/unified-tagging components for the
    // tag editor and react-query mutations for the apis.
    const { showEditTagsModalForTrace, EditTagsModal } = useEditExperimentTraceTags({
      onSuccess: () => invalidateMlflowSearchTracesCache({ queryClient }),
      existingTagKeys: getTracesTagKeys(traceInfos || []),
      useV3Apis: true,
    });

    const traceActions: TraceActions = useMemo(() => {
      return {
        deleteTracesAction: {
          deleteTraces: (experimentId: string, traceIds: string[]) =>
            deleteTracesMutation.mutateAsync({ experimentId, traceRequestIds: traceIds }),
        },
        exportToEvals: {
          exportToEvalsInstanceEnabled: true,
          getTrace,
        },
        editTags: {
          showEditTagsModalForTrace,
          EditTagsModal,
        },
      };
    }, [getTrace, deleteTracesMutation, showEditTagsModalForTrace, EditTagsModal]);

    const countInfo = useMemo(() => {
      return {
        currentCount: traceInfos?.length,
        logCountLoading: traceInfosLoading,
        totalCount: totalCount,
        maxAllowedCount: getEvalTabTotalTracesLimit(),
      };
    }, [traceInfos, totalCount, traceInfosLoading]);

    const isTableLoading = traceInfosLoading || isInitialTimeFilterLoading || isMetadataLoading;
    const tableError = traceInfosError || metadataError;
    const isTableEmpty = isEmpty && !isTableLoading && !tableError;

    // Helper function to render the main content based on current state
    const renderMainContent = () => {
      // If isEmpty and not enableTraceInsights, show empty state without navigation
      if (!enableTraceInsights && isTableEmpty) {
        return <TracesV3EmptyState experimentIds={[experimentId]} />;
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
              <GenAiTracesMarkdownConverterProvider makeHtml={makeHtmlFromMarkdown}>
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
                />
              </GenAiTracesMarkdownConverterProvider>
            )}
          </div>
        </div>
      );
    };

    // Early return for empty state without insights
    if (!enableTraceInsights && isTableEmpty) {
      return <TracesV3EmptyState experimentIds={[experimentId]} />;
    }

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
