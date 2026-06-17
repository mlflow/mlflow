import { useMemo, useState } from 'react';

import { FormattedMessage } from 'react-intl';
import { Button, SparkleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { shouldEnableTracesTableStatePersistence } from '@databricks/web-shared/model-trace-explorer';
import { TracesV3Logs } from './TracesV3Logs';
import {
  MonitoringConfigProvider,
  useMonitoringConfig,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { TracesV3PageWrapper } from './TracesV3PageWrapper';
import { useMonitoringViewState } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringViewState';
import { useExperiments } from '../../hooks/useExperiments';
import { TracesV3Toolbar } from './TracesV3Toolbar';
import { useAssistant } from '@mlflow/mlflow/src/assistant';
import {
  useMonitoringFilters,
  useMonitoringFiltersTimeRange,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';

interface TracesV3ContentProps {
  viewState: string;
  experimentId: string;
  endpointName?: string;
  timeRange: { startTime: string | undefined; endTime: string | undefined };
}

const TracesV3Content = ({
  // comment for copybara formatting
  viewState,
  experimentId,
  endpointName,
  timeRange,
}: TracesV3ContentProps) => {
  const { isLocalServer, openPanel } = useAssistant();
  if (viewState === 'logs') {
    return (
      <TracesV3Logs
        experimentIds={[experimentId || '']}
        // TODO: Remove this once the endpointName is not needed
        endpointName={endpointName || ''}
        timeRange={timeRange}
        drawerWidth="80vw"
        toolbarAddons={
          isLocalServer ? (
            // data-assistant-ui marks this as assistant UI so AssistantAwareDrawer won't treat
            // the click as an outside-click and close. See AssistantAwareDrawer.tsx.
            <Button
              componentId="mlflow.assistant.traces_toolbar_button"
              data-assistant-ui="true"
              icon={<SparkleIcon color="ai" />}
              onClick={openPanel}
            >
              <FormattedMessage
                defaultMessage="Analyze with Assistant"
                description="Traces table toolbar button that opens the MLflow assistant to analyze the current traces"
              />
            </Button>
          ) : undefined
        }
      />
    );
  }
  return null;
};

const TracesV3ViewImpl = ({
  experimentIds,
  isLoadingExperiment,
}: {
  experimentIds: string[];
  isLoadingExperiment?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  // Traces view only works with one experiment
  const experimentId = experimentIds[0];
  const [viewState] = useMonitoringViewState();

  const timeRange = useMonitoringFiltersTimeRange();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        height: '100%',
        overflowY: 'hidden',
      }}
    >
      <TracesV3Toolbar
        // prettier-ignore
        viewState={viewState}
      />
      <TracesV3Content
        // comment for copybara formatting
        viewState={viewState}
        experimentId={experimentId}
        timeRange={timeRange}
      />
    </div>
  );
};

export const TracesV3View = ({
  experimentIds,
  isLoadingExperiment,
}: {
  experimentIds: string[];
  isLoadingExperiment?: boolean;
}) => (
  <TracesV3PageWrapper>
    <MonitoringConfigProvider>
      <TracesV3ViewImpl experimentIds={experimentIds} isLoadingExperiment={isLoadingExperiment} />
    </MonitoringConfigProvider>
  </TracesV3PageWrapper>
);
