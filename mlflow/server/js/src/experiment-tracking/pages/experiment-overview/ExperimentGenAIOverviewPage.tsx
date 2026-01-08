import { useCallback, useMemo, useState } from 'react';
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
import { AssessmentChartsSection } from './components/AssessmentChartsSection';
import { ToolCallStatistics } from './components/ToolCallStatistics';
import { ToolCallChartsSection } from './components/ToolCallChartsSection';
import { LazyToolUsageChart } from './components/LazyToolUsageChart';
import { LazyToolLatencyChart } from './components/LazyToolLatencyChart';
import { LazyToolPerformanceSummary } from './components/LazyToolPerformanceSummary';
import { TabContentContainer, ChartGrid } from './components/OverviewLayoutComponents';
import { calculateTimeInterval } from './hooks/useTraceMetricsQuery';
import { generateTimeBuckets } from './utils/chartUtils';
import { OverviewChartProvider } from './OverviewChartContext';

/**
 * Chart titles for search filtering.
 * These match the chart titles displayed in the UI.
 */
const CHART_TITLES = {
  requests: 'Requests',
  latency: 'Latency',
  errors: 'Errors',
  tokenUsage: 'Token Usage',
  tokenPerTrace: 'Tokens per Trace',
  toolPerformance: 'Tool Performance Summary',
  toolUsage: 'Tool Usage Over Time',
  toolLatency: 'Latency Comparison',
} as const;

enum OverviewTab {
  Usage = 'usage',
  Quality = 'quality',
  ToolCalls = 'tool-calls',
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

  /**
   * Check if a chart title matches the search query.
   * Returns true if no search query or if the title contains the query.
   */
  const matchesSearch = useCallback(
    (title: string): boolean => {
      if (!searchQuery.trim()) return true;
      return title.toLowerCase().includes(searchQuery.toLowerCase().trim());
    },
    [searchQuery],
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
              {matchesSearch(CHART_TITLES.requests) && <LazyTraceRequestsChart />}

              {/* Latency and Errors charts - side by side */}
              <ChartGrid>
                {matchesSearch(CHART_TITLES.latency) && <LazyTraceLatencyChart />}
                {matchesSearch(CHART_TITLES.errors) && <LazyTraceErrorsChart />}
              </ChartGrid>

              {/* Token Usage and Token Stats charts - side by side */}
              <ChartGrid>
                {matchesSearch(CHART_TITLES.tokenUsage) && <LazyTraceTokenUsageChart />}
                {matchesSearch(CHART_TITLES.tokenPerTrace) && <LazyTraceTokenStatsChart />}
              </ChartGrid>
            </TabContentContainer>
          </Tabs.Content>

          <Tabs.Content value={OverviewTab.Quality} css={{ flex: 1, overflowY: 'auto' }}>
            <TabContentContainer>
              {/* Assessment charts - dynamically rendered based on available assessments */}
              <AssessmentChartsSection searchQuery={searchQuery} />
            </TabContentContainer>
          </Tabs.Content>

          <Tabs.Content value={OverviewTab.ToolCalls} css={{ flex: 1, overflowY: 'auto' }}>
            <TabContentContainer>
              {/* Tool call statistics */}
              <ToolCallStatistics />

              {/* Tool performance summary */}
              {matchesSearch(CHART_TITLES.toolPerformance) && <LazyToolPerformanceSummary />}

              {/* Tool error rate charts - dynamically rendered based on available tools */}
              <ToolCallChartsSection searchQuery={searchQuery} />

              {/* Tool usage and latency charts - side by side */}
              <ChartGrid>
                {matchesSearch(CHART_TITLES.toolUsage) && <LazyToolUsageChart />}
                {matchesSearch(CHART_TITLES.toolLatency) && <LazyToolLatencyChart />}
              </ChartGrid>
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
