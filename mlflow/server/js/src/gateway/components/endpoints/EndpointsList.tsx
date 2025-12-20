import { useState } from 'react';
import {
  Button,
  ChainIcon,
  CloudModelIcon,
  Empty,
  Input,
  LinkIcon,
  PencilIcon,
  SearchIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { useEndpointsListData } from '../../hooks/useEndpointsListData';
import { formatProviderName } from '../../utils/providerUtils';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { EndpointsFilterButton, type EndpointsFilter } from './EndpointsFilterButton';
import { EndpointsColumnsButton, EndpointsColumn, DEFAULT_VISIBLE_COLUMNS } from './EndpointsColumnsButton';
import { EndpointBindingsDrawer } from './EndpointBindingsDrawer';
import { DeleteEndpointModal } from './DeleteEndpointModal';
import GatewayRoutes from '../../routes';
import type { Endpoint, EndpointBinding } from '../../types';

interface EndpointsListProps {
  onEndpointDeleted?: () => void;
}

export const EndpointsList = ({ onEndpointDeleted }: EndpointsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const [searchFilter, setSearchFilter] = useState('');
  const [filter, setFilter] = useState<EndpointsFilter>({ providers: [] });
  const [visibleColumns, setVisibleColumns] = useState<EndpointsColumn[]>(DEFAULT_VISIBLE_COLUMNS);
  const [bindingsDrawerEndpoint, setBindingsDrawerEndpoint] = useState<{
    endpointId: string;
    endpointName: string;
    bindings: EndpointBinding[];
  } | null>(null);
  const [deleteModalEndpoint, setDeleteModalEndpoint] = useState<Endpoint | null>(null);

  const { endpoints, filteredEndpoints, isLoading, availableProviders, getBindingsForEndpoint, refetch } =
    useEndpointsListData({ searchFilter, filter });

  const handleDeleteClick = (endpoint: Endpoint) => {
    setDeleteModalEndpoint(endpoint);
  };

  const handleDeleteSuccess = () => {
    setDeleteModalEndpoint(null);
    refetch();
    onEndpointDeleted?.();
  };

  if (isLoading || !endpoints) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
          minHeight: 200,
        }}
      >
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading endpoints..." description="Loading message for endpoints list" />
      </div>
    );
  }

  const isFiltered = searchFilter.trim().length > 0 || filter.providers.length > 0;

  const getEmptyState = () => {
    const isEmptyList = endpoints?.length === 0;
    if (filteredEndpoints.length === 0 && isFiltered) {
      return (
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No endpoints found"
              description="Empty state title when filter returns no results"
            />
          }
          description={null}
        />
      );
    }
    if (isEmptyList) {
      return (
        <Empty
          image={<CloudModelIcon />}
          title={
            <FormattedMessage
              defaultMessage="No endpoints created"
              description="Empty state title for endpoints list"
            />
          }
          description={
            <FormattedMessage
              defaultMessage='Use "Create endpoint" button to create a new endpoint'
              description="Empty state message for endpoints list explaining how to create"
            />
          }
        />
      );
    }
    return null;
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Input
          componentId="mlflow.gateway.endpoints-list.search"
          prefix={<SearchIcon />}
          placeholder={formatMessage({
            defaultMessage: 'Search Endpoints',
            description: 'Placeholder for endpoint search filter',
          })}
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          allowClear
          css={{ maxWidth: 300 }}
        />
        <EndpointsFilterButton availableProviders={availableProviders} filter={filter} onFilterChange={setFilter} />
        <EndpointsColumnsButton visibleColumns={visibleColumns} onColumnsChange={setVisibleColumns} />
      </div>

      <Table
        scrollable
        empty={getEmptyState()}
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.general.borderRadiusBase,
        }}
      >
        <TableRow isHeader>
          <TableHeader componentId="mlflow.gateway.endpoints-list.name-header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Name" description="Endpoint name column header" />
          </TableHeader>
          {visibleColumns.includes(EndpointsColumn.PROVIDER) && (
            <TableHeader componentId="mlflow.gateway.endpoints-list.provider-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Provider" description="Provider column header" />
            </TableHeader>
          )}
          {visibleColumns.includes(EndpointsColumn.MODELS) && (
            <TableHeader componentId="mlflow.gateway.endpoints-list.models-header" css={{ flex: 2 }}>
              <FormattedMessage defaultMessage="Models" description="Models column header" />
            </TableHeader>
          )}
          {visibleColumns.includes(EndpointsColumn.USED_BY) && (
            <TableHeader componentId="mlflow.gateway.endpoints-list.bindings-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Used by" description="Used by column header" />
            </TableHeader>
          )}
          {visibleColumns.includes(EndpointsColumn.LAST_MODIFIED) && (
            <TableHeader componentId="mlflow.gateway.endpoints-list.modified-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Last modified" description="Last modified column header" />
            </TableHeader>
          )}
          {visibleColumns.includes(EndpointsColumn.CREATED) && (
            <TableHeader componentId="mlflow.gateway.endpoints-list.created-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Created" description="Created column header" />
            </TableHeader>
          )}
          <TableHeader
            componentId="mlflow.gateway.endpoints-list.actions-header"
            css={{ flex: 0, minWidth: 96, maxWidth: 96 }}
          />
        </TableRow>
        {filteredEndpoints.map((endpoint) => (
          <TableRow key={endpoint.endpoint_id}>
            <TableCell css={{ flex: 2 }}>
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
            {visibleColumns.includes(EndpointsColumn.PROVIDER) && (
              <TableCell css={{ flex: 1 }}>
                <ProviderCell modelMappings={endpoint.model_mappings} />
              </TableCell>
            )}
            {visibleColumns.includes(EndpointsColumn.MODELS) && (
              <TableCell css={{ flex: 2 }}>
                <ModelsCell modelMappings={endpoint.model_mappings} />
              </TableCell>
            )}
            {visibleColumns.includes(EndpointsColumn.USED_BY) && (
              <TableCell css={{ flex: 1 }}>
                <BindingsCell
                  bindings={getBindingsForEndpoint(endpoint.endpoint_id)}
                  onViewBindings={() =>
                    setBindingsDrawerEndpoint({
                      endpointId: endpoint.endpoint_id,
                      endpointName: endpoint.name ?? endpoint.endpoint_id,
                      bindings: getBindingsForEndpoint(endpoint.endpoint_id),
                    })
                  }
                />
              </TableCell>
            )}
            {visibleColumns.includes(EndpointsColumn.LAST_MODIFIED) && (
              <TableCell css={{ flex: 1 }}>
                <TimeAgo date={timestampToDate(endpoint.last_updated_at)} />
              </TableCell>
            )}
            {visibleColumns.includes(EndpointsColumn.CREATED) && (
              <TableCell css={{ flex: 1 }}>
                <TimeAgo date={timestampToDate(endpoint.created_at)} />
              </TableCell>
            )}
            <TableCell css={{ flex: 0, minWidth: 96, maxWidth: 96 }}>
              <div css={{ display: 'flex', gap: theme.spacing.xs }}>
                <Link to={GatewayRoutes.getEditEndpointRoute(endpoint.endpoint_id)}>
                  <Button
                    componentId="mlflow.gateway.endpoints-list.edit-button"
                    type="primary"
                    icon={<PencilIcon />}
                    aria-label={formatMessage({
                      defaultMessage: 'Edit endpoint',
                      description: 'Gateway > Endpoints list > Edit endpoint button aria label',
                    })}
                  />
                </Link>
                <Button
                  componentId="mlflow.gateway.endpoints-list.delete-button"
                  type="primary"
                  icon={<TrashIcon />}
                  aria-label={formatMessage({
                    defaultMessage: 'Delete endpoint',
                    description: 'Gateway > Endpoints list > Delete endpoint button aria label',
                  })}
                  onClick={() => handleDeleteClick(endpoint)}
                />
              </div>
            </TableCell>
          </TableRow>
        ))}
      </Table>

      <EndpointBindingsDrawer
        open={bindingsDrawerEndpoint !== null}
        endpointName={bindingsDrawerEndpoint?.endpointName ?? ''}
        bindings={bindingsDrawerEndpoint?.bindings ?? []}
        onClose={() => setBindingsDrawerEndpoint(null)}
      />

      <DeleteEndpointModal
        open={deleteModalEndpoint !== null}
        endpoint={deleteModalEndpoint}
        bindings={deleteModalEndpoint ? getBindingsForEndpoint(deleteModalEndpoint.endpoint_id) : []}
        onClose={() => setDeleteModalEndpoint(null)}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};

const ProviderCell = ({ modelMappings }: { modelMappings: Endpoint['model_mappings'] }) => {
  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryProvider = modelMappings[0]?.model_definition?.provider;
  if (!primaryProvider) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  return <Tag componentId="mlflow.gateway.endpoints-list.provider-tag">{formatProviderName(primaryProvider)}</Tag>;
};

const ModelsCell = ({ modelMappings }: { modelMappings: Endpoint['model_mappings'] }) => {
  const { theme } = useDesignSystemTheme();

  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryMapping = modelMappings[0];
  const primaryModelDef = primaryMapping.model_definition;
  const additionalMappings = modelMappings.slice(1);
  const additionalCount = additionalMappings.length;

  const tooltipContent =
    additionalCount > 0 ? additionalMappings.map((m) => m.model_definition?.model_name ?? '-').join(', ') : undefined;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs / 2 }}>
      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
        {primaryModelDef?.model_name ?? '-'}
      </Typography.Text>
      {additionalCount > 0 && (
        <Tooltip componentId="mlflow.gateway.endpoints-list.models-more-tooltip" content={tooltipContent}>
          <button
            type="button"
            css={{
              background: 'none',
              border: 'none',
              padding: 0,
              margin: 0,
              textAlign: 'left',
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
              cursor: 'default',
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            +{additionalCount} more
          </button>
        </Tooltip>
      )}
    </div>
  );
};

const BindingsCell = ({ bindings, onViewBindings }: { bindings: EndpointBinding[]; onViewBindings: () => void }) => {
  const { theme } = useDesignSystemTheme();

  if (!bindings || bindings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  return (
    <button
      type="button"
      onClick={onViewBindings}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        background: 'none',
        border: 'none',
        padding: 0,
        cursor: 'pointer',
        color: theme.colors.actionPrimaryBackgroundDefault,
        '&:hover': {
          textDecoration: 'underline',
        },
      }}
    >
      <LinkIcon css={{ color: theme.colors.textSecondary, fontSize: 14 }} />
      <Typography.Text css={{ color: 'inherit' }}>
        {bindings.length} {bindings.length === 1 ? 'resource' : 'resources'}
      </Typography.Text>
    </button>
  );
};
