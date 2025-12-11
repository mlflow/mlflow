import {
  Button,
  ChainIcon,
  Empty,
  Input,
  SearchIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import { useDeleteEndpointMutation } from '../../hooks/useDeleteEndpointMutation';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { EndpointsFilterButton, type EndpointsFilter } from './EndpointsFilterButton';
import GatewayRoutes from '../../routes';
import type { Endpoint } from '../../types';
import { useMemo, useState, useCallback, memo } from 'react';

interface EndpointsListProps {
  onEndpointDeleted?: () => void;
}

export const EndpointsList = ({ onEndpointDeleted }: EndpointsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const { data: endpoints, isLoading, refetch } = useEndpointsQuery();
  const { mutate: deleteEndpoint, isLoading: isDeleting } = useDeleteEndpointMutation();
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [searchFilter, setSearchFilter] = useState('');
  const [filter, setFilter] = useState<EndpointsFilter>({ providers: [] });

  // Get all unique providers from all endpoints' model mappings
  const availableProviders = useMemo(() => {
    if (!endpoints) return [];
    const providers = new Set<string>();
    endpoints.forEach((endpoint) => {
      endpoint.model_mappings?.forEach((mapping) => {
        if (mapping.model_definition?.provider) {
          providers.add(mapping.model_definition.provider);
        }
      });
    });
    return Array.from(providers);
  }, [endpoints]);

  const filteredEndpoints = useMemo(() => {
    if (!endpoints) return [];
    let filtered = endpoints;

    // Apply search filter
    if (searchFilter.trim()) {
      const lowerFilter = searchFilter.toLowerCase();
      filtered = filtered.filter((endpoint) =>
        (endpoint.name ?? endpoint.endpoint_id).toLowerCase().includes(lowerFilter),
      );
    }

    // Apply provider filter - show endpoints that have at least one model with a matching provider
    if (filter.providers.length > 0) {
      filtered = filtered.filter((endpoint) =>
        endpoint.model_mappings?.some(
          (mapping) =>
            mapping.model_definition?.provider && filter.providers.includes(mapping.model_definition.provider),
        ),
      );
    }

    return filtered;
  }, [endpoints, searchFilter, filter]);

  const handleDelete = useCallback(
    (endpoint: Endpoint) => {
      setDeletingId(endpoint.endpoint_id);
      deleteEndpoint(endpoint.endpoint_id, {
        onSuccess: () => {
          setDeletingId(null);
          refetch();
          onEndpointDeleted?.();
        },
        onError: () => {
          setDeletingId(null);
        },
      });
    },
    [deleteEndpoint, refetch, onEndpointDeleted],
  );

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading endpoints..." description="Loading message for endpoints list" />
      </div>
    );
  }

  if (!endpoints?.length) {
    return (
      <Empty
        image={<ChainIcon />}
        title={formatMessage({
          defaultMessage: 'No endpoints created yet',
          description: 'Empty state title for endpoints list',
        })}
        description={
          <FormattedMessage
            defaultMessage="Create an endpoint with models and API keys to securely connect MLflow features to your preferred GenAI providers."
            description="Empty state message for endpoints list explaining the feature"
          />
        }
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Input
          componentId="mlflow.gateway.endpoints-list.search"
          prefix={<SearchIcon />}
          placeholder={formatMessage({
            defaultMessage: 'Filter endpoints by name',
            description: 'Placeholder for endpoint search filter',
          })}
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          allowClear
          css={{ maxWidth: 300 }}
        />
        <EndpointsFilterButton availableProviders={availableProviders} filter={filter} onFilterChange={setFilter} />
      </div>

      {filteredEndpoints.length === 0 ? (
        <Empty
          image={<SearchIcon />}
          description={
            <FormattedMessage
              defaultMessage="No endpoints match your filter"
              description="Empty state message when filter returns no results"
            />
          }
        />
      ) : (
        <Table
          scrollable
          css={{
            border: `1px solid ${theme.colors.borderDecorative}`,
            borderRadius: theme.general.borderRadiusBase,
          }}
        >
          <TableRow isHeader>
            <TableHeader componentId="mlflow.gateway.endpoints-list.name-header">
              <FormattedMessage defaultMessage="Name" description="Endpoint name column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.endpoints-list.models-header">
              <FormattedMessage defaultMessage="Models" description="Models column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.endpoints-list.modified-header">
              <FormattedMessage defaultMessage="Last modified" description="Last modified column header" />
            </TableHeader>
            <TableHeader
              componentId="mlflow.gateway.endpoints-list.actions-header"
              css={{ flex: 0, minWidth: 48, maxWidth: 48 }}
            />
          </TableRow>
          {filteredEndpoints.map((endpoint) => (
            <TableRow key={endpoint.endpoint_id}>
              <TableCell>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <ChainIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
                  <Link
                    to={GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id)}
                    css={{
                      color: theme.colors.actionPrimaryBackgroundDefault,
                      textDecoration: 'none',
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      '&:hover': {
                        textDecoration: 'underline',
                      },
                    }}
                  >
                    {endpoint.name ?? endpoint.endpoint_id}
                  </Link>
                </div>
              </TableCell>
              <TableCell multiline>
                <ModelsCell modelMappings={endpoint.model_mappings} />
              </TableCell>
              <TableCell>
                <TimeAgo date={timestampToDate(endpoint.last_updated_at)} />
              </TableCell>
              <TableCell css={{ flex: 0, minWidth: 48, maxWidth: 48 }}>
                <Button
                  componentId="mlflow.gateway.endpoints-list.delete-button"
                  type="tertiary"
                  icon={<TrashIcon />}
                  aria-label={formatMessage({
                    defaultMessage: 'Delete endpoint',
                    description: 'Delete endpoint button aria label',
                  })}
                  onClick={() => handleDelete(endpoint)}
                  loading={deletingId === endpoint.endpoint_id}
                  disabled={isDeleting}
                />
              </TableCell>
            </TableRow>
          ))}
        </Table>
      )}
    </div>
  );
};

const ModelsCell = memo(({ modelMappings }: { modelMappings: Endpoint['model_mappings'] }) => {
  const { theme } = useDesignSystemTheme();

  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryMapping = modelMappings[0];
  const primaryModelDef = primaryMapping.model_definition;
  const additionalCount = modelMappings.length - 1;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Model definition name (user's nickname) */}
      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }} bold>
        {primaryModelDef?.name ?? '-'}
      </Typography.Text>
      {/* Provider's model name */}
      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
        {primaryModelDef?.model_name ?? '-'}
      </Typography.Text>
      {additionalCount > 0 && (
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          +{additionalCount} more
        </Typography.Text>
      )}
    </div>
  );
});
