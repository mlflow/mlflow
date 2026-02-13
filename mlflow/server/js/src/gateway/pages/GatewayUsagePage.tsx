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
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { GatewayChartsPanel } from '../components/GatewayChartsPanel';
import GatewayRoutes from '../routes';

export const GatewayUsagePage = () => {
  const { theme } = useDesignSystemTheme();
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(null);

  // Fetch all endpoints to get their experiment IDs
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();

  const endpointsWithExperiments = useMemo(
    () => endpoints.filter((ep) => ep.usage_tracking && ep.experiment_id),
    [endpoints],
  );

  // Get the selected endpoint (if specific endpoint is selected)
  const selectedEndpoint = useMemo(() => {
    if (!selectedEndpointId || selectedEndpointId === 'all') return null;
    return endpointsWithExperiments.find((ep) => ep.endpoint_id === selectedEndpointId) ?? null;
  }, [selectedEndpointId, endpointsWithExperiments]);

  const showAllEndpoints = !selectedEndpointId || selectedEndpointId === 'all';

  const experimentIds = useMemo(() => {
    if (showAllEndpoints) {
      return endpointsWithExperiments.map((ep) => ep.experiment_id).filter(Boolean) as string[];
    }
    return selectedEndpoint?.experiment_id ? [selectedEndpoint.experiment_id] : [];
  }, [showAllEndpoints, endpointsWithExperiments, selectedEndpoint]);

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

  const endpointSelector = (
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
  );

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

      {/* Charts */}
      {isLoadingEndpoints || experimentIds.length > 0 ? (
        <GatewayChartsPanel
          experimentIds={experimentIds}
          showTokenStats
          additionalControls={endpointSelector}
          hideTooltipLinks
        />
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

export default GatewayUsagePage;
