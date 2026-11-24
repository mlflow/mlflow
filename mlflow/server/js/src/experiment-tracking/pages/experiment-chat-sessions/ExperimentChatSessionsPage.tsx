import { FormattedMessage } from '@databricks/i18n';
import ErrorUtils from '@mlflow/mlflow/src/common/utils/ErrorUtils';
import { withErrorBoundary } from '@mlflow/mlflow/src/common/utils/withErrorBoundary';
import { TracesV3Toolbar } from '../../components/experiment-page/components/traces-v3/TracesV3Toolbar';
import invariant from 'invariant';
import { Link, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useCallback, useMemo, useState } from 'react';
import type { RowSelectionState } from '@tanstack/react-table';
import { useRegisterSelectedIds } from '@mlflow/mlflow/src/assistant';
import {
  CUSTOM_METADATA_COLUMN_ID,
  GenAIChatSessionsTable,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  SESSION_COLUMN_ID,
  SIMULATION_GOAL_COLUMN_ID,
  SIMULATION_PERSONA_COLUMN_ID,
  TracesTableColumnType,
  createTraceLocationForExperiment,
  createTraceLocationForDestinationPath,
  useSearchMlflowTraces,
  shouldEnableSessionGrouping,
} from '@databricks/web-shared/genai-traces-table';
import type { GetTraceFunction, TracesTableColumn } from '@databricks/web-shared/genai-traces-table';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { useMonitoringFiltersTimeRange } from '../../hooks/useMonitoringFilters';
import { shouldUseTracesV4API, isV3ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { getChatSessionsFilter } from './utils';
import { ExperimentChatSessionsPageWrapper } from './ExperimentChatSessionsPageWrapper';
import { useGetDeleteTracesAction } from '../../components/experiment-page/components/traces-v3/hooks/useGetDeleteTracesAction';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { TracesV3Logs } from '../../components/experiment-page/components/traces-v3/TracesV3Logs';
import {
  ForkHorizontalIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  SpeechBubbleIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import Routes from '../../routes';

const defaultCustomDefaultSelectedColumns = (column: TracesTableColumn) => {
  if (column.type === TracesTableColumnType.ASSESSMENT || column.type === TracesTableColumnType.EXPECTATION) {
    return true;
  }
  return [
    SESSION_COLUMN_ID,
    SIMULATION_GOAL_COLUMN_ID,
    SIMULATION_PERSONA_COLUMN_ID,
    INPUTS_COLUMN_ID,
    RESPONSE_COLUMN_ID,
  ].includes(column.id);
};

const ExperimentChatSessionsPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  useRegisterSelectedIds('selectedSessionIds', rowSelection);
  invariant(experimentId, 'Experiment ID must be defined');

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
          experimentIds={[experimentId]}
          additionalFilters={filters}
          endpointName=""
          timeRange={timeRange}
          customDefaultSelectedColumns={defaultCustomDefaultSelectedColumns}
          forceGroupBySession
          columnStorageKeyPrefix="chat-sessions"
          pageSource="chat-sessions"
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

const ExperimentChatSessionsMovedPage = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 1,
        width: '100%',
        gap: theme.spacing.lg,
        padding: theme.spacing.xl,
        transform: `translateY(-${theme.spacing.lg}px)`,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexWrap: 'wrap',
          gap: theme.spacing.xl,
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="The Sessions view has moved to the Traces tab."
              description="Explanation that the Sessions view has moved to the Traces tab"
            />
          </Typography.Text>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage='Select "Sessions" in the Traces/Sessions switcher to see the session-level view.'
              description="Instruction explaining how to access the session-level view in the Traces tab"
            />
          </Typography.Text>
          <Link
            componentId="mlflow.chat-sessions.moved-to-grouped-traces"
            to={{ pathname: Routes.getExperimentPageTracesTabRoute(experimentId), search: '?groupBy=session' }}
          >
            <FormattedMessage
              defaultMessage="View Sessions in Traces tab →"
              description="Link from the legacy Sessions page to the equivalent view in the Traces tab"
            />
          </Link>
        </div>
        <div aria-hidden css={{ pointerEvents: 'none' }}>
          <SegmentedControlGroup
            name="mlflow.chat-sessions.moved-view-switcher"
            componentId="mlflow.chat-sessions.moved-view-switcher"
            value="sessions"
            newStyleFlagOverride
            analyticsEvents={[]}
          >
            <SegmentedControlButton value="traces" tabIndex={-1} icon={<ForkHorizontalIcon />} />
            <SegmentedControlButton value="sessions" tabIndex={-1} icon={<SpeechBubbleIcon />} />
          </SegmentedControlGroup>
        </div>
      </div>
    </div>
  );
};

const ExperimentChatSessionsPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  return (
    <ExperimentChatSessionsPageWrapper>
      <MonitoringConfigProvider>
        {shouldEnableSessionGrouping() ? (
          <ExperimentChatSessionsMovedPage experimentId={experimentId} />
        ) : (
          <ExperimentChatSessionsPageImpl />
        )}
      </MonitoringConfigProvider>
    </ExperimentChatSessionsPageWrapper>
  );
};

export default ExperimentChatSessionsPage;
