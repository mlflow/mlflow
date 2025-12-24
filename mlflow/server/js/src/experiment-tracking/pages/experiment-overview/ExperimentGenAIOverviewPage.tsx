import { useMemo, useState } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { GenAiTracesTableSearchInput } from '@databricks/web-shared/genai-traces-table';
import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { TracesV3DateSelector } from '../../components/experiment-page/components/traces-v3/TracesV3DateSelector';
import { useMonitoringFilters, getAbsoluteStartEndTime } from '../../hooks/useMonitoringFilters';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { LazyTraceRequestsChart } from './components/LazyTraceRequestsChart';
import { LazyTraceLatencyChart } from './components/LazyTraceLatencyChart';
import { LazyTraceErrorsChart } from './components/LazyTraceErrorsChart';
import { LazyTraceTokenUsageChart } from './components/LazyTraceTokenUsageChart';
import { LazyTraceTokenStatsChart } from './components/LazyTraceTokenStatsChart';
import { calculateTimeInterval } from './hooks/useTraceMetricsQuery';
import { generateTimeBuckets } from './utils/chartUtils';

enum OverviewTab {
  Usage = 'usage',
}

const ExperimentGenAIOverviewPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [activeTab, setActiveTab] = useState<OverviewTab>(OverviewTab.Usage);
  const [searchQuery, setSearchQuery] = useState('');

  invariant(experimentId, 'Experiment ID must be defined');

  // Get the current time range from monitoring filters
  const [monitoringFilters] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();

  // Use getAbsoluteStartEndTime to properly compute time range from labels
  const { startTime, endTime } = useMemo(
    () => getAbsoluteStartEndTime(monitoringConfig.dateNow, monitoringFilters),
    [monitoringConfig.dateNow, monitoringFilters],
  );

  // Convert ISO strings to milliseconds for the API
  const startTimeMs = startTime ? new Date(startTime).getTime() : undefined;
  const endTimeMs = endTime ? new Date(endTime).getTime() : undefined;

  // Calculate time interval once for all charts
  const timeIntervalSeconds = calculateTimeInterval(startTimeMs, endTimeMs);

  // Generate all time buckets once for all charts
  const timeBuckets = useMemo(
    () => generateTimeBuckets(startTimeMs, endTimeMs, timeIntervalSeconds),
    [startTimeMs, endTimeMs, timeIntervalSeconds],
  );

  // Common props for all chart components
  const chartProps = { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        overflow: 'hidden',
      }}
    >
      <Tabs.Root
        componentId="mlflow.experiment.overview.tabs"
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as OverviewTab)}
        css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}
      >
        <Tabs.List>
          <Tabs.Trigger value={OverviewTab.Usage}>
            <FormattedMessage
              defaultMessage="Usage"
              description="Label for the usage tab in the experiment overview page"
            />
          </Tabs.Trigger>
        </Tabs.List>

        {/* Control bar with search and time range */}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            padding: `${theme.spacing.sm}px 0`,
          }}
        >
          {/* Search input */}
          <GenAiTracesTableSearchInput
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search charts',
              description: 'Placeholder for search charts input',
            })}
          />

          {/*
           * Time range selector - exclude 'ALL' since charts require start_time_ms and end_time_ms
           * TODO: remove this once this is supported in backend
           */}
          <TracesV3DateSelector excludeOptions={['ALL']} />
        </div>

        <Tabs.Content value={OverviewTab.Usage} css={{ flex: 1, overflowY: 'auto' }}>
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.lg,
              padding: `${theme.spacing.sm}px 0`,
            }}
          >
            {/* Requests chart - full width */}
            <LazyTraceRequestsChart {...chartProps} />

            {/* Latency and Errors charts - side by side */}
            <div
              css={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
                gap: theme.spacing.lg,
              }}
            >
              <LazyTraceLatencyChart {...chartProps} />
              <LazyTraceErrorsChart {...chartProps} />
            </div>

            {/* Token Usage and Token Stats charts - side by side */}
            <div
              css={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
                gap: theme.spacing.lg,
              }}
            >
              <LazyTraceTokenUsageChart {...chartProps} />
              <LazyTraceTokenStatsChart {...chartProps} />
            </div>
          </div>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};

// Wrap in MonitoringConfigProvider so refresh button updates are received
const ExperimentGenAIOverviewPage = () => (
  <MonitoringConfigProvider>
    <ExperimentGenAIOverviewPageImpl />
  </MonitoringConfigProvider>
);

export default ExperimentGenAIOverviewPage;
