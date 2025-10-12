import { isNil } from 'lodash';
import { ParagraphSkeleton } from '@databricks/design-system';
import { type KeyValueEntity } from '../../../common/types';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useCompareToRunUuid } from './hooks/useCompareToRunUuid';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { EvaluationRunCompareSelector } from './EvaluationRunCompareSelector';
import { getEvalTabTotalTracesLimit } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { getTrace } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import type { TracesTableColumn, TraceActions, TraceInfoV3 } from '@databricks/web-shared/genai-traces-table';
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
} from '@databricks/web-shared/genai-traces-table';
import { useRunLoggedTraceTableArtifacts } from './hooks/useRunLoggedTraceTableArtifacts';
import { useMarkdownConverter } from '../../../common/utils/MarkdownUtils';
import { useEditExperimentTraceTags } from '../traces/hooks/useEditExperimentTraceTags';
import { useCallback, useMemo, useState } from 'react';
import { useDeleteTracesMutation } from './hooks/useDeleteTraces';
import { RunViewEvaluationsTabArtifacts } from './RunViewEvaluationsTabArtifacts';
import { useGetExperimentRunColor } from '../experiment-page/hooks/useExperimentRunColor';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { useSearchRunsQuery } from '../run-page/hooks/useSearchRunsQuery';
import { checkColumnContents } from '../experiment-page/components/traces-v3/utils/columnUtils';
import { useExportTracesToDatasetModal } from '../../pages/experiment-evaluation-datasets/hooks/useExportTracesToDatasetModal';

const RunViewEvaluationsTabInner = ({
  experimentId,
  runUuid,
  runTags,
  runDisplayName,
  setCurrentRunUuid,
}: {
  experimentId: string;
  runUuid: string;
  runTags?: Record<string, KeyValueEntity>;
  runDisplayName: string;
  setCurrentRunUuid?: (runUuid: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const makeHtmlFromMarkdown = useMarkdownConverter();

  const [compareToRunUuid, setCompareToRunUuid] = useCompareToRunUuid();

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
    experimentId,
    runUuid,
    otherRunUuid: compareToRunUuid,
  });

  // Setup table states
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
    error: traceInfosError,
    refetchMlflowTraces,
  } = useSearchMlflowTraces({
    experimentId,
    currentRunDisplayName: runDisplayName,
    searchQuery,
    filters,
    runUuid,
    tableSort,
  });

  const deleteTracesMutation = useDeleteTracesMutation();

  const {
    data: compareToRunData,
    displayName: compareToRunDisplayName,
    loading: compareToRunLoading,
  } = useGetCompareToData(experimentId, compareToRunUuid);

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
    useV3Apis: true,
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
      editTags: {
        showEditTagsModalForTrace,
        EditTagsModal,
      },
    };
  }, [
    showExportTracesToDatasetsModal,
    setShowExportTracesToDatasetsModal,
    renderExportTracesToDatasetsModal,
    showEditTagsModalForTrace,
    EditTagsModal,
    deleteTracesMutation,
  ]);

  const isTableLoading = traceInfosLoading || compareToRunLoading;

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
          />
          {isTableLoading ? (
            <LoadingSkeleton />
          ) : traceInfosError ? (
            <div>
              <pre>{String(traceInfosError)}</pre>
            </div>
          ) : (
            <GenAiTracesMarkdownConverterProvider makeHtml={makeHtmlFromMarkdown}>
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
              />
            </GenAiTracesMarkdownConverterProvider>
          )}
          {EditTagsModal}
        </div>
      </GenAITracesTableProvider>
    </div>
  );
};

export const RunViewEvaluationsTab = ({
  experimentId,
  runUuid,
  runTags,
  runDisplayName,
  setCurrentRunUuid,
}: {
  experimentId: string;
  runUuid: string;
  runTags?: Record<string, KeyValueEntity>;
  runDisplayName: string;
  // used in evaluation runs tab
  setCurrentRunUuid?: (runUuid: string) => void;
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

const useGetCompareToData = (
  experimentId: string,
  compareToRunUuid: string | undefined,
): {
  data: TraceInfoV3[] | undefined;
  displayName: string;
  loading: boolean;
} => {
  const { data: traceInfos, isLoading: traceInfosLoading } = useSearchMlflowTraces({
    experimentId,
    currentRunDisplayName: undefined,
    runUuid: compareToRunUuid,
    disabled: isNil(compareToRunUuid),
  });

  const { data: runData, loading: runDetailsLoading } = useSearchRunsQuery({
    experimentIds: [experimentId],
    filter: `attributes.runId = "${compareToRunUuid}"`,
    disabled: isNil(compareToRunUuid),
  });

  return {
    data: traceInfos,
    displayName: Utils.getRunDisplayName(runData?.info, compareToRunUuid),
    loading: traceInfosLoading || runDetailsLoading,
  };
};
