import { FormattedMessage } from '@databricks/i18n';
import ErrorUtils from '@mlflow/mlflow/src/common/utils/ErrorUtils';
import { shouldEnableChatSessionsTab } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
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
import { useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { getAbsoluteStartEndTime, useMonitoringFilters } from '../../hooks/useMonitoringFilters';
import { SESSION_ID_METADATA_KEY, shouldUseTracesV4API } from '@databricks/web-shared/model-trace-explorer';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { getChatSessionsFilter } from './utils';
import { ExperimentChatSessionsPageWrapper } from './ExperimentChatSessionsPageWrapper';

const ExperimentChatSessionsPageImpl = () => {
  const { experimentId } = useParams();
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
    disabled: false,
  });

  // the tab will not be added to the navbar if this is disbled, but just
  // in case users navigate to it directly, we return an empty div to
  // avoid displaying any in-progress work.
  if (!shouldEnableChatSessionsTab()) {
    return <div />;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <TracesV3Toolbar
        // prettier-ignore
        viewState="sessions"
      />
      <GenAIChatSessionsTable experimentId={experimentId} traces={traces ?? []} isLoading={isLoading || isFetching} />
    </div>
  );
};

const ExperimentChatSessionsPage = () => {
  return (
    <ExperimentChatSessionsPageWrapper>
      <ExperimentChatSessionsPageImpl />
    </ExperimentChatSessionsPageWrapper>
  );
};

export default ExperimentChatSessionsPage;
