import { useMemo, useState } from 'react';
import {
  Typography,
  useDesignSystemTheme,
  BarChartIcon,
  SimpleSelect,
  SimpleSelectOption,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { TokenVolumeChart, ErrorRateChart } from '../components/metrics';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';

type TimeRange = '24h' | '7d' | '30d';

const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_DAY = 86400;

const getTimeRangeParams = (
  timeRange: TimeRange,
): { startTime: number; endTime: number; bucketSize: number } => {
  const now = Date.now();
  const hourMs = 60 * 60 * 1000;
  const dayMs = 24 * hourMs;

  switch (timeRange) {
    case '24h':
      return {
        startTime: now - dayMs,
        endTime: now,
        bucketSize: SECONDS_PER_HOUR,
      };
    case '7d':
      return {
        startTime: now - 7 * dayMs,
        endTime: now,
        bucketSize: SECONDS_PER_DAY,
      };
    case '30d':
      return {
        startTime: now - 30 * dayMs,
        endTime: now,
        bucketSize: SECONDS_PER_DAY,
      };
    default:
      return {
        startTime: now - 7 * dayMs,
        endTime: now,
        bucketSize: SECONDS_PER_DAY,
      };
  }
};

const MetricsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [selectedEndpoint, setSelectedEndpoint] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<TimeRange>('7d');

  const { data: endpoints } = useEndpointsQuery();

  const { startTime, endTime, bucketSize } = useMemo(() => getTimeRangeParams(timeRange), [timeRange]);

  const endpointIdFilter = selectedEndpoint === 'all' ? undefined : selectedEndpoint;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
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
          <BarChartIcon />
          <FormattedMessage defaultMessage="Metrics" description="Metrics page title" />
        </Typography.Title>
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <SimpleSelect
            id="mlflow-gateway-metrics-endpoint-filter"
            componentId="mlflow.gateway.metrics.endpoint-filter"
            value={selectedEndpoint}
            onChange={({ target }) => setSelectedEndpoint(target.value)}
            css={{ minWidth: 200 }}
          >
            <SimpleSelectOption value="all">
              {intl.formatMessage({
                defaultMessage: 'All endpoints',
                description: 'All endpoints filter option',
              })}
            </SimpleSelectOption>
            {endpoints?.map((endpoint) => (
              <SimpleSelectOption key={endpoint.endpoint_id} value={endpoint.endpoint_id}>
                {endpoint.name}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
          <SimpleSelect
            id="mlflow-gateway-metrics-time-range"
            componentId="mlflow.gateway.metrics.time-range"
            value={timeRange}
            onChange={({ target }) => setTimeRange(target.value as TimeRange)}
            css={{ minWidth: 140 }}
          >
            <SimpleSelectOption value="24h">
              {intl.formatMessage({
                defaultMessage: 'Last 24 hours',
                description: '24 hour time range option',
              })}
            </SimpleSelectOption>
            <SimpleSelectOption value="7d">
              {intl.formatMessage({
                defaultMessage: 'Last 7 days',
                description: '7 day time range option',
              })}
            </SimpleSelectOption>
            <SimpleSelectOption value="30d">
              {intl.formatMessage({
                defaultMessage: 'Last 30 days',
                description: '30 day time range option',
              })}
            </SimpleSelectOption>
          </SimpleSelect>
        </div>
      </div>
      <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
            gap: theme.spacing.lg,
          }}
        >
          <TokenVolumeChart
            endpointId={endpointIdFilter}
            startTime={startTime}
            endTime={endTime}
            bucketSize={bucketSize}
          />
          <ErrorRateChart
            endpointId={endpointIdFilter}
            startTime={startTime}
            endTime={endTime}
            bucketSize={bucketSize}
          />
        </div>
      </div>
    </div>
  );
};

export default MetricsPage;
