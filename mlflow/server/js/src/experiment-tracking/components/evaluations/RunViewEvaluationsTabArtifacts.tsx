import { isNil } from 'lodash';
import { Empty, LegacySkeleton } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { type KeyValueEntity } from '../../../common/types';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useCompareToRunUuid } from './hooks/useCompareToRunUuid';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { EvaluationRunCompareSelector } from './EvaluationRunCompareSelector';
import { useSavePendingEvaluationAssessments } from './hooks/useSavePendingEvaluationAssessments';
import type {
  GenAiTraceEvaluationArtifactFile,
  TracesTableColumn,
  RunEvaluationTracesDataEntry,
} from '@databricks/web-shared/genai-traces-table';
import {
  EXECUTION_DURATION_COLUMN_ID,
  GenAiTracesTable,
  GenAiTracesMarkdownConverterProvider,
  STATE_COLUMN_ID,
  TAGS_COLUMN_ID,
  TracesTableColumnType,
  useGenAiTraceEvaluationArtifacts,
} from '@databricks/web-shared/genai-traces-table';
import { useRunLoggedTraceTableArtifacts } from './hooks/useRunLoggedTraceTableArtifacts';
import { useMarkdownConverter } from '../../../common/utils/MarkdownUtils';
import { getTraceLegacy } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { useSearchRunsQuery } from '../run-page/hooks/useSearchRunsQuery';

export const RunViewEvaluationsTabArtifacts = ({
  experimentId,
  runUuid,
  runTags,
  runDisplayName,
  data,
}: {
  experimentId: string;
  runUuid: string;
  runTags?: Record<string, KeyValueEntity>;
  runDisplayName: string;
  data: RunEvaluationTracesDataEntry[];
}) => {
  const { theme } = useDesignSystemTheme();

  // Determine which tables are logged in the run
  const traceTablesLoggedInRun = useRunLoggedTraceTableArtifacts(runTags);

  const noEvaluationTablesLogged = data?.length === 0;

  const [compareToRunUuid, setCompareToRunUuid] = useCompareToRunUuid();

  const makeHtmlFromMarkdown = useMarkdownConverter();
  const saveAssessmentsQuery = useSavePendingEvaluationAssessments();

  const {
    data: compareToRunData,
    displayName: compareToRunDisplayName,
    loading: compareToRunLoading,
  } = useGetCompareToDataWithArtifacts(experimentId, compareToRunUuid, traceTablesLoggedInRun);

  if (compareToRunLoading) {
    // TODO: Implement proper skeleton for this page
    return <LegacySkeleton />;
  }

  const initialSelectedColumns = (allColumns: TracesTableColumn[]) => {
    return allColumns.filter(
      (col) =>
        col.type === TracesTableColumnType.ASSESSMENT ||
        col.type === TracesTableColumnType.INPUT ||
        (col.type === TracesTableColumnType.TRACE_INFO &&
          [EXECUTION_DURATION_COLUMN_ID, STATE_COLUMN_ID, TAGS_COLUMN_ID].includes(col.id)),
    );
  };

  /**
   * Determine whether to render the component from the shared codebase (GenAiTracesTable)
   * or the legacy one from the local codebase (EvaluationsOverview).
   */
  const getOverviewTableComponent = () => {
    const componentProps = {
      experimentId,
      currentRunDisplayName: runDisplayName,
      currentEvaluationResults: data || [],
      compareToEvaluationResults: compareToRunData,
      runUuid,
      compareToRunUuid,
      compareToRunDisplayName,
      compareToRunLoading,
      saveAssessmentsQuery,
      getTrace: getTraceLegacy,
      initialSelectedColumns,
    } as const;
    return (
      <GenAiTracesMarkdownConverterProvider makeHtml={makeHtmlFromMarkdown}>
        <GenAiTracesTable {...componentProps} />
      </GenAiTracesMarkdownConverterProvider>
    );
  };

  if (noEvaluationTablesLogged) {
    return (
      <div css={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No evaluation tables logged"
              description="Run page > Evaluations tab > No evaluation tables logged"
            />
          }
          description={null}
        />
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
        />
      </div>
      {getOverviewTableComponent()}
    </div>
  );
};

const useGetCompareToDataWithArtifacts = (
  experimentId: string,
  compareToRunUuid: string | undefined,
  traceTablesLoggedInRun: GenAiTraceEvaluationArtifactFile[],
): {
  data: RunEvaluationTracesDataEntry[] | undefined;
  displayName: string;
  loading: boolean;
} => {
  const { data, isLoading: loading } = useGenAiTraceEvaluationArtifacts(
    {
      runUuid: compareToRunUuid || '',
      artifacts: traceTablesLoggedInRun,
    },
    { disabled: isNil(compareToRunUuid) },
  );

  const { data: runData, loading: runDetailsLoading } = useSearchRunsQuery({
    experimentIds: [experimentId],
    filter: `attributes.runId = "${compareToRunUuid}"`,
    disabled: isNil(compareToRunUuid),
  });

  return {
    data,
    displayName: Utils.getRunDisplayName(runData?.info, compareToRunUuid),
    loading: loading || runDetailsLoading,
  };
};
