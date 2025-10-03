import { useMemo, useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { TracesV3DateSelector } from './TracesV3DateSelector';
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
import {
  isExperimentEvalResultsMonitoringUIEnabled,
  shouldEnableTracesSyncUI,
} from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { useExperiments } from '../../hooks/useExperiments';

const TracesV3Toolbar = () => {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        width: '100%',
        borderBottom: `1px solid ${theme.colors.grey100}`,
        paddingBottom: `${theme.spacing.sm}px`,
      }}
    >
      <TracesV3DateSelector />
    </div>
  );
};

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

const TracesV3ViewImpl = ({ experimentIds }: { experimentIds: string[] }) => {
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
      {isExperimentEvalResultsMonitoringUIEnabled() && (
        // comment for copybara formatting
        <TracesV3Toolbar />
      )}
      <TracesV3Content
        // comment for copybara formatting
        viewState={viewState}
        experimentId={experimentId}
        timeRange={timeRange}
      />
    </div>
  );
};

export const TracesV3View = ({ experimentIds }: { experimentIds: string[] }) => (
  <TracesV3PageWrapper>
    <MonitoringConfigProvider>
      <TracesV3ViewImpl experimentIds={experimentIds} />
    </MonitoringConfigProvider>
  </TracesV3PageWrapper>
);
