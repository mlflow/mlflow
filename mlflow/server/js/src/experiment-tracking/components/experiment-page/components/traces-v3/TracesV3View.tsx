import { useMemo, useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { shouldEnableTracesTableStatePersistence } from '@databricks/web-shared/model-trace-explorer';
import { useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { TracesV3Logs } from './TracesV3Logs';
import {
  MonitoringConfigProvider,
  useMonitoringConfig,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { TracesV3PageWrapper } from './TracesV3PageWrapper';
import { useMonitoringViewState } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringViewState';
import { useExperiments } from '../../hooks/useExperiments';
import { TracesV3Toolbar } from './TracesV3Toolbar';
import { TracesV3SavedViewsButton, useTraceSavedViews } from './TracesV3SavedViews';
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
  // One useTraceSavedViews instance for both toolbar buttons: a single Apollo subscription and a
  // single `atCap` source, instead of each button opening its own subscription.
  const savedViews = useTraceSavedViews({ experimentId: experimentId || '' });
  // The deprecated Sessions page links here with `?groupBy=session` to open the session-grouped view.
  // TODO: Remove this V3 groupBy plumbing once the Traces tab mounts V4, which reads `?groupBy=session`
  // natively via `useTracesV4UrlState`.
  const { search } = useLocation();
  const initialGroupBySession = new URLSearchParams(search).get('groupBy') === 'session';
  if (viewState === 'logs') {
    return (
      <TracesV3Logs
        experimentIds={[experimentId || '']}
        // TODO: Remove this once the endpointName is not needed
        endpointName={endpointName || ''}
        timeRange={timeRange}
        drawerWidth="80vw"
        initialGroupBySession={initialGroupBySession}
        enableSavedViews
        toolbarCornerAddons={
          experimentId && <TracesV3SavedViewsButton experimentId={experimentId} savedViews={savedViews} />
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
