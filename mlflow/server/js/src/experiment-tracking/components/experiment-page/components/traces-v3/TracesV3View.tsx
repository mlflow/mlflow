import { useMemo, useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { TracesV3Logs } from './TracesV3Logs';
import {
  MonitoringConfigProvider,
  useMonitoringConfig,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { TracesV3PageWrapper } from './TracesV3PageWrapper';
import { useMonitoringViewState } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringViewState';
import {
  getAbsoluteStartEndTime,
  useMonitoringFilters,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { useExperiments } from '../../hooks/useExperiments';
import { TracesV3Toolbar } from './TracesV3Toolbar';

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
  if (viewState === 'logs') {
    return (
      <TracesV3Logs
        experimentId={experimentId || ''}
        // TODO: Remove this once the endpointName is not needed
        endpointName={endpointName || ''}
        timeRange={timeRange}
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
  const [monitoringFilters, _setMonitoringFilters] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();

  // Traces view only works with one experiment
  const experimentId = experimentIds[0];
  const [viewState] = useMonitoringViewState();

  const timeRange = useMemo(() => {
    const { startTime, endTime } = getAbsoluteStartEndTime(monitoringConfig.dateNow, monitoringFilters);
    return {
      startTime: startTime ? new Date(startTime).getTime().toString() : undefined,
      endTime: endTime ? new Date(endTime).getTime().toString() : undefined,
    };
  }, [monitoringConfig.dateNow, monitoringFilters]);

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
