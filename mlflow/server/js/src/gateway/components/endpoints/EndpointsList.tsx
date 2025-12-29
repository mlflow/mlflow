import { useState } from 'react';
import {
  CloudModelIcon,
  Empty,
  Input,
  SearchIcon,
  Spinner,
  Table,
  TableHeader,
  TableRow,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useEndpointsListData } from '../../hooks/useEndpointsListData';
import { EndpointsFilterButton, type EndpointsFilter } from './EndpointsFilterButton';
import { EndpointBindingsDrawer } from './EndpointBindingsDrawer';
import { DeleteEndpointModal } from './DeleteEndpointModal';
import { EndpointRow } from './EndpointRow';
import type { Endpoint, EndpointBinding } from '../../types';

interface EndpointsListProps {
  onEndpointDeleted?: () => void;
}

export const EndpointsList = ({ onEndpointDeleted }: EndpointsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const [searchFilter, setSearchFilter] = useState('');
  const [filter, setFilter] = useState<EndpointsFilter>({ providers: [] });
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
    if (endpoints.length === 0) {
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
          <TableHeader componentId="mlflow.gateway.endpoints-list.provider-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Provider" description="Provider column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.endpoints-list.models-header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Models" description="Models column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.endpoints-list.bindings-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Connected resources" description="Connected resources column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.endpoints-list.modified-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Last modified" description="Last modified column header" />
          </TableHeader>
          <TableHeader
            componentId="mlflow.gateway.endpoints-list.actions-header"
            css={{ flex: 0, minWidth: 96, maxWidth: 96 }}
          />
        </TableRow>
        {filteredEndpoints.map((endpoint) => (
          <EndpointRow
            key={endpoint.endpoint_id}
            endpoint={endpoint}
            bindings={getBindingsForEndpoint(endpoint.endpoint_id)}
            onViewBindings={() =>
              setBindingsDrawerEndpoint({
                endpointId: endpoint.endpoint_id,
                endpointName: endpoint.name ?? endpoint.endpoint_id,
                bindings: getBindingsForEndpoint(endpoint.endpoint_id),
              })
            }
            onDelete={() => handleDeleteClick(endpoint)}
          />
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
