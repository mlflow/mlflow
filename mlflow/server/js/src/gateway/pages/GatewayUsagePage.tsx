import { useMemo, useState } from 'react';
import { Link, useSearchParams } from '../../common/utils/RoutingUtils';
import {
  ChartLineIcon,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Tabs,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { createTraceMetadataFilter, AUTH_USER_ID_METADATA_KEY } from '@databricks/web-shared/model-trace-explorer';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { useUsersQuery } from '../hooks/useUsersQuery';
import { GatewayChartsPanel } from '../components/GatewayChartsPanel';
import GatewayRoutes from '../routes';
import { TracesV3Logs } from '../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3Logs';
import { MonitoringConfigProvider } from '../../experiment-tracking/hooks/useMonitoringConfig';
import { useMonitoringFiltersTimeRange } from '../../experiment-tracking/hooks/useMonitoringFilters';
import { TracesV3DateSelector } from '../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3DateSelector';

const GatewayLogsContent = ({ experimentIds }: { experimentIds: string[] }) => {
  const { theme } = useDesignSystemTheme();
  const timeRange = useMonitoringFiltersTimeRange();

  if (experimentIds.length === 0) {
    return null;
  }

  return (
    <>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: theme.spacing.sm,
        }}
      >
        <TracesV3DateSelector excludeOptions={['ALL']} />
      </div>
      <TracesV3Logs
        experimentIds={experimentIds}
        disableActions
        timeRange={timeRange}
        columnStorageKeyPrefix="gateway-usage"
      />
    </>
  );
};

export const GatewayUsagePage = () => {
  const { theme } = useDesignSystemTheme();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(null);
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);

  const activeTab = searchParams.get('tab') || 'usage';

  // Fetch all endpoints to get their experiment IDs
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();

  // Fetch all users for the user selector (may fail if user lacks admin permissions)
  const { data: users, isLoading: isLoadingUsers, error: usersError } = useUsersQuery();

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

  const tooltipLinkUrlBuilder = useMemo(() => {
    if (!selectedEndpoint) return undefined;
    return (_experimentId: string, timestampMs: number, timeIntervalSeconds: number) =>
      GatewayRoutes.getEndpointDetailsRoute(selectedEndpoint.endpoint_id, {
        tab: 'traces',
        startTime: new Date(timestampMs).toISOString(),
        endTime: new Date(timestampMs + timeIntervalSeconds * 1000).toISOString(),
      });
  }, [selectedEndpoint]);

  // Build filters from selected user ID
  const filters = useMemo(() => {
    if (!selectedUserId) return undefined;
    return [createTraceMetadataFilter(AUTH_USER_ID_METADATA_KEY, selectedUserId)];
  }, [selectedUserId]);

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

  const endpointAndUserControls = (
    <>
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
      {!usersError && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="User:" description="User selector label" />
          </Typography.Text>
          <SimpleSelect
            id="gateway-usage-user-selector"
            componentId="mlflow.gateway.usage.user-selector"
            value={selectedUserId ?? 'all'}
            onChange={({ target }) => setSelectedUserId(target.value === 'all' ? null : target.value)}
            css={{ minWidth: 180 }}
            disabled={isLoadingUsers || users.length === 0}
          >
            <SimpleSelectOption value="all">
              <FormattedMessage defaultMessage="All users" description="All users option" />
            </SimpleSelectOption>
            {users.map((user) => (
              <SimpleSelectOption key={user.id} value={String(user.id)}>
                {user.username}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
          {isLoadingUsers && <Spinner size="small" />}
        </div>
      )}
    </>
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
      {/* Header */}
      <div
        css={{
          padding: theme.spacing.md,
          paddingBottom: 0,
        }}
      >
        <div
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: theme.spacing.sm,
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

        {/* Filters */}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.md,
            marginBottom: theme.spacing.sm,
          }}
        >
          {endpointAndUserControls}
        </div>
      </div>

      {/* Tabs */}
      <Tabs.Root
        componentId="mlflow.gateway.usage.tabs"
        value={activeTab}
        onValueChange={(value) => {
          setSearchParams(
            (params) => {
              params.set('tab', value);
              return params;
            },
            { replace: true },
          );
        }}
        css={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
      >
        <div css={{ paddingLeft: theme.spacing.md, paddingRight: theme.spacing.md }}>
          <Tabs.List>
            <Tabs.Trigger value="usage">
              <FormattedMessage defaultMessage="Usage" description="Tab label for usage charts" />
            </Tabs.Trigger>
            <Tabs.Trigger value="logs">
              <FormattedMessage defaultMessage="Logs" description="Tab label for trace logs" />
            </Tabs.Trigger>
          </Tabs.List>
        </div>

        <Tabs.Content
          value="usage"
          css={{
            flex: 1,
            overflow: 'auto',
            padding: theme.spacing.md,
          }}
        >
          {isLoadingEndpoints || experimentIds.length > 0 ? (
            <GatewayChartsPanel
              experimentIds={experimentIds}
              showTokenStats
              hideTooltipLinks={showAllEndpoints}
              tooltipLinkUrlBuilder={tooltipLinkUrlBuilder}
              tooltipLinkText={
                <FormattedMessage
                  defaultMessage="View logs for this period"
                  description="Link text to navigate to gateway endpoint logs tab"
                />
              }
              filters={filters}
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
        </Tabs.Content>

        <Tabs.Content
          value="logs"
          css={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            padding: theme.spacing.md,
          }}
        >
          {experimentIds.length > 0 ? (
            <MonitoringConfigProvider>
              <GatewayLogsContent experimentIds={experimentIds} />
            </MonitoringConfigProvider>
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
                  defaultMessage="Select an endpoint to view logs"
                  description="No endpoint selected message for logs tab"
                />
              </Typography.Text>
            </div>
          )}
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};

export default GatewayUsagePage;
