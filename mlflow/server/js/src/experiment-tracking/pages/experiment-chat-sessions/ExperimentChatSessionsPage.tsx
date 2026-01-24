import { FormattedMessage } from '@databricks/i18n';
import ErrorUtils from '@mlflow/mlflow/src/common/utils/ErrorUtils';
import { withErrorBoundary } from '@mlflow/mlflow/src/common/utils/withErrorBoundary';
import { TracesV3Toolbar } from '../../components/experiment-page/components/traces-v3/TracesV3Toolbar';
import invariant from 'invariant';
import { useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useCallback, useMemo, useState } from 'react';
import type { RowSelectionState } from '@tanstack/react-table';
import { useRegisterSelectedIds } from '@mlflow/mlflow/src/assistant';
import {
  CUSTOM_METADATA_COLUMN_ID,
  GenAIChatSessionsTable,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  SESSION_COLUMN_ID,
  TracesTableColumnType,
  createTraceLocationForExperiment,
  createTraceLocationForUCSchema,
  useSearchMlflowTraces,
  shouldEnableSessionGrouping,
} from '@databricks/web-shared/genai-traces-table';
import type { GetTraceFunction, TracesTableColumn } from '@databricks/web-shared/genai-traces-table';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { useMonitoringFiltersTimeRange } from '../../hooks/useMonitoringFilters';
import { SESSION_ID_METADATA_KEY, shouldUseTracesV4API } from '@databricks/web-shared/model-trace-explorer';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { getChatSessionsFilter } from './utils';
import { ExperimentChatSessionsPageWrapper } from './ExperimentChatSessionsPageWrapper';
import { useGetDeleteTracesAction } from '../../components/experiment-page/components/traces-v3/hooks/useGetDeleteTracesAction';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { TracesV3Logs } from '../../components/experiment-page/components/traces-v3/TracesV3Logs';
import { useDesignSystemTheme } from '@databricks/design-system';

const defaultCustomDefaultSelectedColumns = (column: TracesTableColumn) => {
  if (column.type === TracesTableColumnType.ASSESSMENT || column.type === TracesTableColumnType.EXPECTATION) {
    return true;
  }
  return [SESSION_COLUMN_ID, INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID].includes(column.id);
};

const ExperimentChatSessionsPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  useRegisterSelectedIds('selectedSessionIds', rowSelection);
  invariant(experimentId, 'Experiment ID must be defined');

  const monitoringConfig = useMonitoringConfig();
  const { loading: isLoadingExperiment } = useGetExperimentQuery({
    experimentId,
  });

  const timeRange = useMonitoringFiltersTimeRange();

  const traceSearchLocations = useMemo(
    () => {
      return [createTraceLocationForExperiment(experimentId)];
    },
    // prettier-ignore
    [
      experimentId,
    ],
  );

  const filters = useMemo(() => getChatSessionsFilter({ sessionId: null }), []);

  const {
    data: traces,
    isLoading,
    isFetching,
  } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    timeRange,
    filters,
    searchQuery,
    disabled: false,
  });

  const deleteTracesAction = useGetDeleteTracesAction({ traceSearchLocations });

  const traceActions = useMemo(
    () => ({
      deleteTracesAction,
    }),
    [deleteTracesAction],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        gap: theme.spacing.sm,
      }}
    >
      <TracesV3Toolbar
        // prettier-ignore
        viewState="sessions"
      />
      {shouldEnableSessionGrouping() ? (
        <TracesV3Logs
          experimentId={experimentId}
          additionalFilters={filters}
          endpointName=""
          timeRange={timeRange}
          customDefaultSelectedColumns={defaultCustomDefaultSelectedColumns}
          forceGroupBySession
        />
      ) : (
        <GenAIChatSessionsTable
          experimentId={experimentId}
          traces={traces ?? []}
          isLoading={isLoading}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          traceActions={traceActions}
        />
      )}
    </div>
  );
};

const ExperimentChatSessionsPage = () => {
  return (
    <ExperimentChatSessionsPageWrapper>
      <MonitoringConfigProvider>
        <ExperimentChatSessionsPageImpl />
      </MonitoringConfigProvider>
    </ExperimentChatSessionsPageWrapper>
  );
};

export default ExperimentChatSessionsPage;
