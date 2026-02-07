import { useMemo, useState } from 'react';
import { Link } from '../../common/utils/RoutingUtils';
import {
  ChartLineIcon,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { MonitoringConfigProvider } from '../../experiment-tracking/hooks/useMonitoringConfig';
import { useMonitoringFilters, getAbsoluteStartEndTime } from '../../experiment-tracking/hooks/useMonitoringFilters';
import { TracesV3DateSelector } from '../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3DateSelector';
import { LazyTraceRequestsChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceRequestsChart';
import { LazyTraceLatencyChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceLatencyChart';
import { LazyTraceErrorsChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceErrorsChart';
import { LazyTraceTokenUsageChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceTokenUsageChart';
import { LazyTraceTokenStatsChart } from '../../experiment-tracking/pages/experiment-overview/components/LazyTraceTokenStatsChart';
import { ChartGrid } from '../../experiment-tracking/pages/experiment-overview/components/OverviewLayoutComponents';
import { OverviewChartProvider } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { TimeUnitSelector } from '../../experiment-tracking/pages/experiment-overview/components/TimeUnitSelector';
import {
  TimeUnit,
  TIME_UNIT_SECONDS,
  calculateDefaultTimeUnit,
} from '../../experiment-tracking/pages/experiment-overview/utils/timeUtils';
import { generateTimeBuckets } from '../../experiment-tracking/pages/experiment-overview/utils/chartUtils';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import GatewayRoutes from '../routes';

export const GatewayUsagePageImpl = () => {
  const { theme } = useDesignSystemTheme();
  const [selectedTimeUnit, setSelectedTimeUnit] = useState<TimeUnit | null>(null);
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(null);

  // Fetch all endpoints to get their experiment IDs
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();

  // Filter endpoints that have usage tracking enabled (with experiment_id)
  const endpointsWithExperiments = useMemo(
    () => endpoints.filter((ep) => ep.usage_tracking && ep.experiment_id),
    [endpoints],
  );

  // Get the selected endpoint (if specific endpoint is selected)
  const selectedEndpoint = useMemo(() => {
    if (!selectedEndpointId || selectedEndpointId === 'all') return null;
    return endpointsWithExperiments.find((ep) => ep.endpoint_id === selectedEndpointId) ?? null;
  }, [selectedEndpointId, endpointsWithExperiments]);

  // Determine whether to show all endpoints or a specific one
  const showAllEndpoints = !selectedEndpointId || selectedEndpointId === 'all';

  // Get the experiment IDs to use for charts
  // If showing all endpoints, use all experiment_ids
  // If a specific endpoint is selected, use its experiment_id
  const experimentIds = useMemo(() => {
    if (showAllEndpoints) {
      return endpointsWithExperiments.map((ep) => ep.experiment_id).filter(Boolean) as string[];
    }
    return selectedEndpoint?.experiment_id ? [selectedEndpoint.experiment_id] : [];
  }, [showAllEndpoints, endpointsWithExperiments, selectedEndpoint]);

  // Get the current time range from monitoring filters
  const [monitoringFilters] = useMonitoringFilters();

  // Compute time range
  const { startTime, endTime } = useMemo(() => {
    const now = new Date();
    return getAbsoluteStartEndTime(now, monitoringFilters);
  }, [monitoringFilters]);

  // Convert ISO strings to milliseconds for the API
  const startTimeMs = startTime ? new Date(startTime).getTime() : undefined;
  const endTimeMs = endTime ? new Date(endTime).getTime() : undefined;

  // Calculate the default time unit for the current time range
  const defaultTimeUnit = calculateDefaultTimeUnit(startTimeMs, endTimeMs);

  // Use selected if valid, otherwise fall back to default
  const effectiveTimeUnit = selectedTimeUnit ?? defaultTimeUnit;

  // Use the effective time unit for time interval
  const timeIntervalSeconds = TIME_UNIT_SECONDS[effectiveTimeUnit];

  // Generate all time buckets once for all charts
  const timeBuckets = useMemo(
    () => generateTimeBuckets(startTimeMs, endTimeMs, timeIntervalSeconds),
    [startTimeMs, endTimeMs, timeIntervalSeconds],
  );

  // Show empty state if no endpoints have usage tracking enabled
  if (!isLoadingEndpoints && endpointsWithExperiments.length === 0) {
    return (
      <div
        css={{
          flex: 1,
          overflow: 'auto',
          padding: theme.spacing.md,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: 300,
            textAlign: 'center',
            padding: theme.spacing.lg,
          }}
        >
          <ChartLineIcon css={{ fontSize: 48, color: theme.colors.textSecondary, marginBottom: theme.spacing.md }} />
          <Typography.Title level={3}>
            <FormattedMessage defaultMessage="No usage data available" description="Empty state title" />
          </Typography.Title>
          <Typography.Text color="secondary" css={{ marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="Enable usage tracking on your endpoints to see usage metrics here."
              description="Empty state description"
            />
          </Typography.Text>
          <Link to={GatewayRoutes.gatewayPageRoute}>
            <FormattedMessage defaultMessage="Go to Endpoints" description="Link to endpoints page" />
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        overflow: 'auto',
        padding: theme.spacing.md,
      }}
    >
      {/* Header */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: theme.spacing.md,
        }}
      >
        <div>
          <Typography.Title level={2} css={{ margin: 0 }}>
            <FormattedMessage defaultMessage="Gateway Usage" description="Page title" />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Monitor usage and performance across all endpoints"
              description="Page subtitle"
            />
          </Typography.Text>
        </div>
      </div>

      {/* Controls row */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          marginBottom: theme.spacing.lg,
          flexWrap: 'wrap',
        }}
      >
        {/* Endpoint selector */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Endpoint:" description="Endpoint selector label" />
          </Typography.Text>
          <SimpleSelect
            id="gateway-usage-endpoint-selector"
            componentId="mlflow.gateway.usage.endpoint-selector"
            value={selectedEndpointId ?? 'all'}
            onChange={({ target }) => setSelectedEndpointId(target.value === 'all' ? null : target.value)}
            css={{ minWidth: 200 }}
            disabled={isLoadingEndpoints}
          >
            <SimpleSelectOption value="all">
              <FormattedMessage defaultMessage="All endpoints" description="All endpoints option" />
            </SimpleSelectOption>
            {endpointsWithExperiments.map((endpoint) => (
              <SimpleSelectOption key={endpoint.endpoint_id} value={endpoint.endpoint_id}>
                {endpoint.name}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
          {isLoadingEndpoints && <Spinner size="small" />}
        </div>

        {/* Time unit selector */}
        <TimeUnitSelector
          value={effectiveTimeUnit}
          onChange={setSelectedTimeUnit}
          startTimeMs={startTimeMs}
          endTimeMs={endTimeMs}
          allowClear={selectedTimeUnit !== null && selectedTimeUnit !== defaultTimeUnit}
          onClear={() => setSelectedTimeUnit(null)}
        />

        {/* Date range selector */}
        <TracesV3DateSelector excludeOptions={['ALL']} />
      </div>

      {/* Charts */}
      {experimentIds.length > 0 ? (
        <OverviewChartProvider
          experimentIds={experimentIds}
          startTimeMs={startTimeMs}
          endTimeMs={endTimeMs}
          timeIntervalSeconds={timeIntervalSeconds}
          timeBuckets={timeBuckets}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
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
          </div>
        </OverviewChartProvider>
      ) : (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: 200,
            color: theme.colors.textSecondary,
          }}
        >
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Select an endpoint to view usage metrics"
              description="No endpoint selected message"
            />
          </Typography.Text>
        </div>
      )}
    </div>
  );
};

export const GatewayUsagePage = () => (
  <MonitoringConfigProvider>
    <GatewayUsagePageImpl />
  </MonitoringConfigProvider>
);

export default GatewayUsagePage;
