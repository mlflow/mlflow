import { FormattedMessage } from '@databricks/i18n';
import ErrorUtils from '@mlflow/mlflow/src/common/utils/ErrorUtils';
import { withErrorBoundary } from '@mlflow/mlflow/src/common/utils/withErrorBoundary';
import { TracesV3Toolbar } from '../../components/experiment-page/components/traces-v3/TracesV3Toolbar';
import invariant from 'invariant';
import { useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useMemo, useState } from 'react';
import {
  CUSTOM_METADATA_COLUMN_ID,
  GenAIChatSessionsTable,
  createTraceLocationForExperiment,
  createTraceLocationForUCSchema,
  useSearchMlflowTraces,
} from '@databricks/web-shared/genai-traces-table';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { getAbsoluteStartEndTime, useMonitoringFilters } from '../../hooks/useMonitoringFilters';
import { SESSION_ID_METADATA_KEY, shouldUseTracesV4API } from '@databricks/web-shared/model-trace-explorer';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { getChatSessionsFilter } from './utils';
import { ExperimentChatSessionsPageWrapper } from './ExperimentChatSessionsPageWrapper';
import { useGetDeleteTracesAction } from '../../components/experiment-page/components/traces-v3/hooks/useGetDeleteTracesAction';

const ExperimentChatSessionsPageImpl = () => {
  const { experimentId } = useParams();
  const [searchQuery, setSearchQuery] = useState<string>('');
  invariant(experimentId, 'Experiment ID must be defined');

  const [monitoringFilters] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();
  const { loading: isLoadingExperiment } = useGetExperimentQuery({
    experimentId,
    options: {
      fetchPolicy: 'cache-only',
    },
  });

  const timeRange = useMemo(() => {
    const { startTime, endTime } = getAbsoluteStartEndTime(monitoringConfig.dateNow, monitoringFilters);
    return {
      startTime: startTime ? new Date(startTime).getTime().toString() : undefined,
      endTime: endTime ? new Date(endTime).getTime().toString() : undefined,
    };
  }, [monitoringConfig.dateNow, monitoringFilters]);

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

  const traceActions = {
    deleteTracesAction,
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <TracesV3Toolbar
        // prettier-ignore
        viewState="sessions"
      />
      <GenAIChatSessionsTable
        experimentId={experimentId}
        traces={traces ?? []}
        isLoading={isLoading}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        traceActions={traceActions}
      />
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
