import { useEffect, useState, useMemo } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { Alert, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useIsFileStore } from '../../hooks/useTrackingStoreInfo';
import { TracesV3DateSelector } from '../../components/experiment-page/components/traces-v3/TracesV3DateSelector';
import { useMonitoringFilters, getAbsoluteStartEndTime } from '../../hooks/useMonitoringFilters';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { LazyTraceRequestsChart } from './components/LazyTraceRequestsChart';
import { LazyTraceLatencyChart } from './components/LazyTraceLatencyChart';
import { LazyTraceErrorsChart } from './components/LazyTraceErrorsChart';
import { LazyTraceTokenUsageChart } from './components/LazyTraceTokenUsageChart';
import { LazyTraceTokenStatsChart } from './components/LazyTraceTokenStatsChart';
import { AssessmentChartsSection } from './components/AssessmentChartsSection';
import { ToolCallStatistics } from './components/ToolCallStatistics';
import { ToolCallChartsSection } from './components/ToolCallChartsSection';
import { LazyToolUsageChart } from './components/LazyToolUsageChart';
import { LazyToolLatencyChart } from './components/LazyToolLatencyChart';
import { LazyToolPerformanceSummary } from './components/LazyToolPerformanceSummary';
import { TabContentContainer, ChartGrid } from './components/OverviewLayoutComponents';
import { TimeUnitSelector } from './components/TimeUnitSelector';
import { TimeUnit, TIME_UNIT_SECONDS, calculateDefaultTimeUnit, isTimeUnitValid } from './utils/timeUtils';
import { generateTimeBuckets } from './utils/chartUtils';
import { OverviewChartProvider } from './OverviewChartContext';
import { useOverviewTab, OverviewTab } from './hooks/useOverviewTab';

const ExperimentGenAIOverviewPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [activeTab, setActiveTab] = useOverviewTab();
  const [selectedTimeUnit, setSelectedTimeUnit] = useState<TimeUnit | null>(null);
  const isFileStore = useIsFileStore();

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

  // Calculate the default time unit for the current time range
  const defaultTimeUnit = calculateDefaultTimeUnit(startTimeMs, endTimeMs);

  // Auto-clear if selected time unit becomes invalid due to time range change
  useEffect(() => {
    if (selectedTimeUnit && !isTimeUnitValid(startTimeMs, endTimeMs, selectedTimeUnit)) {
      setSelectedTimeUnit(null);
    }
  }, [startTimeMs, endTimeMs, selectedTimeUnit]);

  // Use selected if valid, otherwise fall back to default
  const effectiveTimeUnit = selectedTimeUnit ?? defaultTimeUnit;

  // Use the effective time unit for time interval
  const timeIntervalSeconds = TIME_UNIT_SECONDS[effectiveTimeUnit];

  // Generate all time buckets once for all charts
  const timeBuckets = useMemo(
    () => generateTimeBuckets(startTimeMs, endTimeMs, timeIntervalSeconds),
    [startTimeMs, endTimeMs, timeIntervalSeconds],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        overflow: 'hidden',
      }}
    >
      {isFileStore && (
        <Alert
          componentId="mlflow.experiment.overview.filestore-warning"
          type="warning"
          css={{ marginBottom: theme.spacing.sm }}
          message={
            <FormattedMessage
              defaultMessage="The Overview tab requires a SQL-based tracking store for full functionality, file-based backend is not supported."
              description="Warning banner shown on the Overview tab when using FileStore backend"
            />
          }
        />
      )}
      <Tabs.Root
        componentId="mlflow.experiment.overview.tabs"
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as OverviewTab)}
        valueHasNoPii
        css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}
      >
        <Tabs.List>
          <Tabs.Trigger value={OverviewTab.Usage}>
            <FormattedMessage
              defaultMessage="Usage"
              description="Label for the usage tab in the experiment overview page"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value={OverviewTab.Quality}>
            <FormattedMessage
              defaultMessage="Quality"
              description="Label for the quality tab in the experiment overview page"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value={OverviewTab.ToolCalls}>
            <FormattedMessage
              defaultMessage="Tool calls"
              description="Label for the tool calls tab in the experiment overview page"
            />
          </Tabs.Trigger>
        </Tabs.List>

        {/* Control bar with time range */}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
          }}
        >
          {/* Time unit selector for chart grouping */}
          <TimeUnitSelector
            value={effectiveTimeUnit}
            onChange={setSelectedTimeUnit}
            startTimeMs={startTimeMs}
            endTimeMs={endTimeMs}
            allowClear={selectedTimeUnit !== null && selectedTimeUnit !== defaultTimeUnit}
            onClear={() => setSelectedTimeUnit(null)}
          />

          {/*
           * Time range selector - exclude 'ALL' since charts require start_time_ms and end_time_ms
           * TODO: remove this once this is supported in backend
           */}
          <TracesV3DateSelector
            excludeOptions={['ALL']}
            refreshButtonComponentId="mlflow.experiment.overview.refresh-button"
          />
        </div>

        <OverviewChartProvider
          experimentId={experimentId}
          startTimeMs={startTimeMs}
          endTimeMs={endTimeMs}
          timeIntervalSeconds={timeIntervalSeconds}
          timeBuckets={timeBuckets}
        >
          <Tabs.Content value={OverviewTab.Usage} css={{ flex: 1, overflowY: 'auto' }}>
            <TabContentContainer>
              {/* Requests chart - full width */}
              <LazyTraceRequestsChart />

              {/* Latency and Errors charts - side by side */}
              <ChartGrid>
                <LazyTraceLatencyChart />
                <LazyTraceErrorsChart />
              </ChartGrid>

              {/* Token Usage and Token Stats charts - side by side */}
              <ChartGrid>
                <LazyTraceTokenUsageChart />
                <LazyTraceTokenStatsChart />
              </ChartGrid>
            </TabContentContainer>
          </Tabs.Content>

          <Tabs.Content value={OverviewTab.Quality} css={{ flex: 1, overflowY: 'auto' }}>
            <TabContentContainer>
              {/* Assessment charts - dynamically rendered based on available assessments */}
              <AssessmentChartsSection />
            </TabContentContainer>
          </Tabs.Content>

          <Tabs.Content value={OverviewTab.ToolCalls} css={{ flex: 1, overflowY: 'auto' }}>
            <TabContentContainer>
              {/* Tool call statistics */}
              <ToolCallStatistics />

              {/* Tool performance summary */}
              <LazyToolPerformanceSummary />

              {/* Tool usage and latency charts - side by side */}
              <ChartGrid>
                <LazyToolUsageChart />
                <LazyToolLatencyChart />
              </ChartGrid>

              {/* Tool error rate charts - dynamically rendered based on available tools */}
              <ToolCallChartsSection />
            </TabContentContainer>
          </Tabs.Content>
        </OverviewChartProvider>
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
