import { useCallback, useMemo, useState } from 'react';
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
import { ChartGrid, TabContentContainer } from '../../experiment-tracking/pages/experiment-overview/components/OverviewLayoutComponents';
import { OverviewChartProvider } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { TimeUnitSelector } from '../../experiment-tracking/pages/experiment-overview/components/TimeUnitSelector';
import {
  TimeUnit,
  TIME_UNIT_SECONDS,
  calculateDefaultTimeUnit,
} from '../../experiment-tracking/pages/experiment-overview/utils/timeUtils';
import { generateTimeBuckets } from '../../experiment-tracking/pages/experiment-overview/utils/chartUtils';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { useGatewayFilterOptions } from '../hooks/useGatewayFilterOptions';
import { GatewayUsageFilters } from '../components/GatewayUsageFilters';
import GatewayRoutes from '../routes';

const GatewayUsagePageImpl = () => {
  const { theme } = useDesignSystemTheme();
  const [selectedTimeUnit, setSelectedTimeUnit] = useState<TimeUnit | null>(null);
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(null);
  const [chartFilters, setChartFilters] = useState<string[]>([]);

  // Fetch all endpoints to get their experiment IDs
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();

  // Filter endpoints that have experiment_id configured
  const endpointsWithExperiments = useMemo(
    () => endpoints.filter((ep) => ep.experiment_id),
    [endpoints],
  );

  // Get the selected endpoint or default to showing all
  const selectedEndpoint = useMemo(() => {
    if (!selectedEndpointId) return null;
    return endpointsWithExperiments.find((ep) => ep.endpoint_id === selectedEndpointId) ?? null;
  }, [selectedEndpointId, endpointsWithExperiments]);

  // Get the experiment ID to use for charts
  // If a specific endpoint is selected, use its experiment_id
  // Otherwise, we'll show a message to select an endpoint
  const experimentId = selectedEndpoint?.experiment_id ?? null;

  // Fetch filter options (providers and models) for the selected endpoint
  const { providers, models, isLoading: isLoadingFilterOptions } = useGatewayFilterOptions(experimentId);

  // Handle filter changes - reset filters when endpoint changes
  const handleFiltersChange = useCallback((filters: string[]) => {
    setChartFilters(filters);
  }, []);

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

  if (isLoadingEndpoints) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, gap: theme.spacing.sm }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading endpoints..." description="Loading message" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      {/* Header */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: theme.spacing.md,
          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        }}
      >
        <Typography.Title level={3} css={{ margin: 0, display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <ChartLineIcon />
          <FormattedMessage defaultMessage="Usage" description="Gateway usage page title" />
        </Typography.Title>
      </div>

      {/* Controls */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.md,
          padding: theme.spacing.md,
          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
          flexWrap: 'wrap',
        }}
      >
        {/* Endpoint selector */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Endpoint:" description="Endpoint selector label" />
          </Typography.Text>
          <SimpleSelect
            id="gateway-usage-endpoint-selector"
            componentId="mlflow.gateway.usage.endpoint-selector"
            value={selectedEndpointId ?? ''}
            onChange={({ target }) => setSelectedEndpointId(target.value || null)}
            css={{ minWidth: 200 }}
          >
            <SimpleSelectOption value="">
              <FormattedMessage defaultMessage="Select an endpoint" description="Endpoint selector placeholder" />
            </SimpleSelectOption>
            {endpointsWithExperiments.map((ep) => (
              <SimpleSelectOption key={ep.endpoint_id} value={ep.endpoint_id}>
                {ep.name}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        </div>

        {/* Time controls - only show when an endpoint is selected */}
        {experimentId && (
          <>
            <TimeUnitSelector
              value={effectiveTimeUnit}
              onChange={setSelectedTimeUnit}
              startTimeMs={startTimeMs}
              endTimeMs={endTimeMs}
              allowClear={selectedTimeUnit !== null && selectedTimeUnit !== defaultTimeUnit}
              onClear={() => setSelectedTimeUnit(null)}
            />
            <TracesV3DateSelector
              excludeOptions={['ALL']}
              refreshButtonComponentId="mlflow.gateway.endpoint.usage.refresh-button"
            />

            {/* Provider/Model filters */}
            {(providers.length > 0 || models.length > 0) && (
              <GatewayUsageFilters
                providers={providers}
                models={models}
                onFiltersChange={handleFiltersChange}
                disabled={isLoadingFilterOptions}
              />
            )}
          </>
        )}
      </div>

      {/* Content */}
      <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
        {endpointsWithExperiments.length === 0 ? (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              gap: theme.spacing.md,
              textAlign: 'center',
            }}
          >
            <ChartLineIcon css={{ fontSize: 48, color: theme.colors.textSecondary }} />
            <Typography.Title level={4} css={{ margin: 0 }}>
              <FormattedMessage defaultMessage="No usage data available" description="No data title" />
            </Typography.Title>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Usage charts will be available once you have endpoints with tracing enabled. Create an endpoint to get started."
                description="No data message"
              />
            </Typography.Text>
            <Link to={GatewayRoutes.createEndpointPageRoute}>
              <Typography.Text color="info">
                <FormattedMessage defaultMessage="Create an endpoint" description="Create endpoint link" />
              </Typography.Text>
            </Link>
          </div>
        ) : !experimentId ? (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              gap: theme.spacing.md,
              textAlign: 'center',
            }}
          >
            <ChartLineIcon css={{ fontSize: 48, color: theme.colors.textSecondary }} />
            <Typography.Title level={4} css={{ margin: 0 }}>
              <FormattedMessage defaultMessage="Select an endpoint" description="Select endpoint title" />
            </Typography.Title>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Choose an endpoint from the dropdown above to view its usage metrics."
                description="Select endpoint message"
              />
            </Typography.Text>
          </div>
        ) : (
          <OverviewChartProvider
            experimentId={experimentId}
            startTimeMs={startTimeMs}
            endTimeMs={endTimeMs}
            timeIntervalSeconds={timeIntervalSeconds}
            timeBuckets={timeBuckets}
            filters={chartFilters.length > 0 ? chartFilters : undefined}
          >
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
          </OverviewChartProvider>
        )}
      </div>
    </div>
  );
};

const GatewayUsagePage = () => (
  <MonitoringConfigProvider>
    <GatewayUsagePageImpl />
  </MonitoringConfigProvider>
);

export default GatewayUsagePage;
