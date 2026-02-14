import { useMemo, useState } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { MonitoringConfigProvider } from '../../experiment-tracking/hooks/useMonitoringConfig';
import { useMonitoringFilters } from '../../experiment-tracking/hooks/useMonitoringFilters';
import { TracesV3DateSelector } from '../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3DateSelector';
import { LazyTraceRequestsChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceRequestsChart';
import { LazyTraceLatencyChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceLatencyChart';
import { LazyTraceErrorsChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceErrorsChart';
import { LazyTraceTokenUsageChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceTokenUsageChart';
import { LazyTraceTokenStatsChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceTokenStatsChart';
import { LazyTraceCostBreakdownChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceCostBreakdownChart';
import { LazyTraceCostOverTimeChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceCostOverTimeChart';
import { ChartGrid } from '../../experiment-tracking/pages/experiment-overview/components/OverviewLayoutComponents';
import { OverviewChartProvider } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { TimeUnitSelector } from '../../experiment-tracking/pages/experiment-overview/components/TimeUnitSelector';
import {
  TimeUnit,
  TIME_UNIT_SECONDS,
  calculateDefaultTimeUnit,
} from '../../experiment-tracking/pages/experiment-overview/utils/timeUtils';
import { generateTimeBuckets } from '../../experiment-tracking/pages/experiment-overview/utils/chartUtils';

interface GatewayChartsPanelProps {
  experimentIds: string[];
  showTokenStats?: boolean;
  additionalControls?: React.ReactNode;
  hideTooltipLinks?: boolean;
}

const GatewayChartsPanelImpl = ({
  experimentIds,
  showTokenStats = false,
  additionalControls,
  hideTooltipLinks = false,
}: GatewayChartsPanelProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedTimeUnit, setSelectedTimeUnit] = useState<TimeUnit | null>(null);

  // useMonitoringFilters already normalizes non-custom ranges using MonitoringConfigProvider's dateNow
  const [monitoringFilters] = useMonitoringFilters();

  const startTimeMs = monitoringFilters.startTime ? new Date(monitoringFilters.startTime).getTime() : undefined;
  const endTimeMs = monitoringFilters.endTime ? new Date(monitoringFilters.endTime).getTime() : undefined;

  const defaultTimeUnit = calculateDefaultTimeUnit(startTimeMs, endTimeMs);

  const effectiveTimeUnit = selectedTimeUnit ?? defaultTimeUnit;

  const timeIntervalSeconds = TIME_UNIT_SECONDS[effectiveTimeUnit];

  const timeBuckets = useMemo(
    () => generateTimeBuckets(startTimeMs, endTimeMs, timeIntervalSeconds),
    [startTimeMs, endTimeMs, timeIntervalSeconds],
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {/* Time range controls */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          flexWrap: 'wrap',
        }}
      >
        {additionalControls}
        <TimeUnitSelector
          value={effectiveTimeUnit}
          onChange={setSelectedTimeUnit}
          startTimeMs={startTimeMs}
          endTimeMs={endTimeMs}
          allowClear={selectedTimeUnit !== null && selectedTimeUnit !== defaultTimeUnit}
          onClear={() => setSelectedTimeUnit(null)}
        />
        <TracesV3DateSelector excludeOptions={['ALL']} />
      </div>

      <OverviewChartProvider
        experimentIds={experimentIds}
        startTimeMs={startTimeMs}
        endTimeMs={endTimeMs}
        timeIntervalSeconds={timeIntervalSeconds}
        timeBuckets={timeBuckets}
        hideTooltipLinks={hideTooltipLinks}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {/* Requests chart - full width */}
          <LazyTraceRequestsChart
            title={<FormattedMessage defaultMessage="Requests" description="Title for the requests chart in gateway" />}
          />

          {/* Latency and Errors charts - side by side */}
          <ChartGrid>
            <LazyTraceLatencyChart />
            <LazyTraceErrorsChart />
          </ChartGrid>

          {/* Token charts */}
          {showTokenStats ? (
            <ChartGrid>
              <LazyTraceTokenUsageChart />
              <LazyTraceTokenStatsChart
                title={
                  <FormattedMessage
                    defaultMessage="Tokens per Request"
                    description="Title for the token stats chart in gateway"
                  />
                }
                subtitle={
                  <FormattedMessage
                    defaultMessage="avg per request"
                    description="Subtitle for average tokens per request in gateway"
                  />
                }
              />
            </ChartGrid>
          ) : (
            <LazyTraceTokenUsageChart />
          )}

          {/* Cost charts */}
          <ChartGrid>
            <LazyTraceCostBreakdownChart />
            <LazyTraceCostOverTimeChart />
          </ChartGrid>
        </div>
      </OverviewChartProvider>
    </div>
  );
};

export const GatewayChartsPanel = (props: GatewayChartsPanelProps) => (
  <MonitoringConfigProvider>
    <GatewayChartsPanelImpl {...props} />
  </MonitoringConfigProvider>
);
