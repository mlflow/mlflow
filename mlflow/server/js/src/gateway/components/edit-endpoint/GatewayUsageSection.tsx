import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useMemo, useState } from 'react';
import { MonitoringConfigProvider } from '../../../experiment-tracking/hooks/useMonitoringConfig';
import { useMonitoringFilters } from '../../../experiment-tracking/hooks/useMonitoringFilters';
import { TracesV3DateSelector } from '../../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3DateSelector';
import { LazyTraceRequestsChart } from '../../../experiment-tracking/pages/experiment-overview/components/LazyTraceRequestsChart';
import { LazyTraceLatencyChart } from '../../../experiment-tracking/pages/experiment-overview/components/LazyTraceLatencyChart';
import { LazyTraceErrorsChart } from '../../../experiment-tracking/pages/experiment-overview/components/LazyTraceErrorsChart';
import { LazyTraceTokenUsageChart } from '../../../experiment-tracking/pages/experiment-overview/components/LazyTraceTokenUsageChart';
import { ChartGrid } from '../../../experiment-tracking/pages/experiment-overview/components/OverviewLayoutComponents';
import { OverviewChartProvider } from '../../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { TimeUnitSelector } from '../../../experiment-tracking/pages/experiment-overview/components/TimeUnitSelector';
import {
  TimeUnit,
  TIME_UNIT_SECONDS,
  calculateDefaultTimeUnit,
} from '../../../experiment-tracking/pages/experiment-overview/utils/timeUtils';
import { generateTimeBuckets } from '../../../experiment-tracking/pages/experiment-overview/utils/chartUtils';

interface GatewayUsageSectionProps {
  experimentId: string;
}

const GatewayUsageSectionImpl = ({ experimentId }: GatewayUsageSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedTimeUnit, setSelectedTimeUnit] = useState<TimeUnit | null>(null);

  // Get the current time range from monitoring filters
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
    <div
      css={{
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <div
        css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.md }}
      >
        <div>
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage defaultMessage="Usage" description="Section title for endpoint usage" />
          </Typography.Title>
          <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Monitor endpoint usage and performance metrics"
              description="Usage section description"
            />
          </Typography.Text>
        </div>
        <Typography.Link
          componentId="mlflow.gateway.endpoint.usage.view-full-dashboard"
          href={`#/experiments/${experimentId}/overview`}
          css={{ fontSize: theme.typography.fontSizeSm }}
        >
          <FormattedMessage defaultMessage="View full dashboard" description="Link to view full usage dashboard" />
        </Typography.Link>
      </div>

      {/* Time range controls */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          marginBottom: theme.spacing.md,
          flexWrap: 'wrap',
        }}
      >
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
        experimentId={experimentId}
        startTimeMs={startTimeMs}
        endTimeMs={endTimeMs}
        timeIntervalSeconds={timeIntervalSeconds}
        timeBuckets={timeBuckets}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {/* Requests chart */}
          <LazyTraceRequestsChart />

          {/* Latency and Errors charts - side by side */}
          <ChartGrid>
            <LazyTraceLatencyChart />
            <LazyTraceErrorsChart />
          </ChartGrid>

          {/* Token Usage chart */}
          <LazyTraceTokenUsageChart />
        </div>
      </OverviewChartProvider>
    </div>
  );
};

export const GatewayUsageSection = ({ experimentId }: GatewayUsageSectionProps) => (
  <MonitoringConfigProvider>
    <GatewayUsageSectionImpl experimentId={experimentId} />
  </MonitoringConfigProvider>
);
