import { useEffect, useState, useMemo } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { Alert, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useIsFileStore } from '../../hooks/useServerInfo';
import { TracesV3DateSelector } from '../../components/experiment-page/components/traces-v3/TracesV3DateSelector';
import {
  useMonitoringFilters,
  getAbsoluteStartEndTime,
  DEFAULT_START_TIME_LABEL,
} from '../../hooks/useMonitoringFilters';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { LazyTraceRequestsChart } from './components/LazyTraceRequestsChart';
import { LazyTraceLatencyChart } from './components/LazyTraceLatencyChart';
import { LazyTraceErrorsChart } from './components/LazyTraceErrorsChart';
import { LazyTraceTokenUsageChart } from './components/LazyTraceTokenUsageChart';
import { LazyTraceTokenStatsChart } from './components/LazyTraceTokenStatsChart';
import { LazyTraceCostBreakdownChart } from './components/LazyTraceCostBreakdownChart';
import { LazyTraceCostOverTimeChart } from './components/LazyTraceCostOverTimeChart';
import { AssessmentChartsSection } from './components/AssessmentChartsSection';
import { ToolCallStatistics } from './components/ToolCallStatistics';
import { ToolCallChartsSection } from './components/ToolCallChartsSection';
import { LazyToolUsageChart } from './components/LazyToolUsageChart';
import { LazyToolLatencyChart } from './components/LazyToolLatencyChart';
import { LazyToolPerformanceSummary } from './components/LazyToolPerformanceSummary';
import { TabContentContainer, ChartGrid } from './components/OverviewLayoutComponents';
import { TimeUnitSelector } from './components/TimeUnitSelector';
import type { TimeUnit } from './utils/timeUtils';
import { TIME_UNIT_SECONDS, calculateDefaultTimeUnit, isTimeUnitValid } from './utils/timeUtils';
import { generateTimeBuckets } from './utils/chartUtils';
import { OverviewChartProvider } from './OverviewChartContext';
import { useOverviewTab, OverviewTab } from './hooks/useOverviewTab';
import { useEarliestTraceTimestamp } from './hooks/useEarliestTraceTimestamp';

const ExperimentGenAIOverviewPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [activeTab, setActiveTab] = useOverviewTab();
  const [selectedTimeUnit, setSelectedTimeUnit] = useState<TimeUnit | null>(null);
  const isFileStore = useIsFileStore();

  invariant(experimentId, 'Experiment ID must be defined');

  // Get the current time range from monitoring filters
  const [monitoringFilters, setMonitoringFilters] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();

  // 'ALL' is excluded from the date selector on this page since charts require
  // start_time_ms and end_time_ms. If the user navigates here with ?startTimeLabel=ALL,
  // reset to the default time range.
  useEffect(() => {
    if (monitoringFilters.startTimeLabel === 'ALL') {
      setMonitoringFilters({ startTimeLabel: DEFAULT_START_TIME_LABEL }, true);
    }
  }, [monitoringFilters.startTimeLabel, setMonitoringFilters]);

  // Use getAbsoluteStartEndTime to properly compute time range from labels
  const { startTime, endTime } = useMemo(
    () => getAbsoluteStartEndTime(monitoringConfig.dateNow, monitoringFilters),
    [monitoringConfig.dateNow, monitoringFilters],
  );

  // Convert ISO strings to milliseconds for the API.
  // When "ALL" is selected, startTime is undefined — use 0 (epoch) so the
  // backend returns data across all time while still accepting time_interval_seconds.
  const isAllTime = monitoringFilters.startTimeLabel === 'ALL';
  const startTimeMs = startTime ? new Date(startTime).getTime() : 0;
  const endTimeMs = endTime ? new Date(endTime).getTime() : Date.now();

  // For "ALL" mode, query the earliest trace timestamp so we can validate time
  // units against the actual data span instead of epoch-to-now.
  const earliestTraceTimestamp = useEarliestTraceTimestamp([experimentId], isAllTime);
  const validationStartTimeMs = isAllTime ? (earliestTraceTimestamp ?? endTimeMs) : startTimeMs;

  // Calculate the default time unit for the current time range
  const defaultTimeUnit = calculateDefaultTimeUnit(validationStartTimeMs, endTimeMs);

  // Auto-clear if selected time unit becomes invalid due to time range change
  useEffect(() => {
    if (selectedTimeUnit && !isTimeUnitValid(validationStartTimeMs, endTimeMs, selectedTimeUnit)) {
      setSelectedTimeUnit(null);
    }
  }, [validationStartTimeMs, endTimeMs, selectedTimeUnit]);

  // Use selected if valid, otherwise fall back to default
  const effectiveTimeUnit = selectedTimeUnit ?? defaultTimeUnit;

  // Use the effective time unit for time interval
  const timeIntervalSeconds = TIME_UNIT_SECONDS[effectiveTimeUnit];

  // Generate all time buckets once for all charts.
  // Skip pre-generation for "ALL" — the range is unbounded so we let charts
  // derive buckets from the server response instead.
  const timeBuckets = useMemo(
    () => (isAllTime ? [] : generateTimeBuckets(startTimeMs, endTimeMs, timeIntervalSeconds)),
    [isAllTime, startTimeMs, endTimeMs, timeIntervalSeconds],
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
            startTimeMs={validationStartTimeMs}
            endTimeMs={endTimeMs}
            allowClear={selectedTimeUnit !== null && selectedTimeUnit !== defaultTimeUnit}
            onClear={() => setSelectedTimeUnit(null)}
          />

          <TracesV3DateSelector refreshButtonComponentId="mlflow.experiment.overview.refresh-button" />
        </div>

        <OverviewChartProvider
          experimentIds={[experimentId]}
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

              {/* Cost Breakdown and Cost Over Time charts - side by side */}
              <ChartGrid>
                <LazyTraceCostBreakdownChart />
                <LazyTraceCostOverTimeChart />
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
