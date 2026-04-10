import { useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Checkbox,
  Empty,
  Input,
  SearchIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import GatewayRoutes from '../../routes';
import { useEndpointsListData } from '../../hooks/useEndpointsListData';
import { useDuplicateEndpoints } from '../../hooks/useDuplicateEndpoints';
import { EndpointsFilterButton, type EndpointsFilter } from './EndpointsFilterButton';
import { EndpointsColumnsButton, EndpointsColumn, DEFAULT_VISIBLE_COLUMNS } from './EndpointsColumnsButton';
import { EndpointBindingsDrawer } from './EndpointBindingsDrawer';
import { BulkDeleteEndpointModal } from './BulkDeleteEndpointModal';
import { EndpointRow } from './EndpointRow';
import { QuickStartTemplates } from './QuickStartTemplates';
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
  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});
  const [bindingsDrawerEndpoint, setBindingsDrawerEndpoint] = useState<{
    endpointId: string;
    endpointName: string;
    bindings: EndpointBinding[];
  } | null>(null);
  const [deleteModalEndpoints, setDeleteModalEndpoints] = useState<Endpoint[]>([]);

  const { endpoints, filteredEndpoints, isLoading, availableProviders, getBindingsForEndpoint, refetch } =
    useEndpointsListData({ searchFilter, filter });

  const { duplicateEndpoints, isLoading: isDuplicating, error: duplicateError } = useDuplicateEndpoints();

  const selectedEndpoints = useMemo(
    () => filteredEndpoints.filter((ep) => rowSelection[ep.endpoint_id]),
    [filteredEndpoints, rowSelection],
  );

  const selectedCount = selectedEndpoints.length;
  const allSelected = filteredEndpoints.length > 0 && filteredEndpoints.every((ep) => rowSelection[ep.endpoint_id]);
  const someSelected = selectedCount > 0 && !allSelected;

  const handleSelectAll = () => {
    if (allSelected) {
      setRowSelection({});
    } else {
      const next: Record<string, boolean> = {};
      filteredEndpoints.forEach((ep) => {
        next[ep.endpoint_id] = true;
      });
      setRowSelection(next);
    }
  };

  const handleSelectRow = (endpointId: string) => {
    setRowSelection((prev) => {
      const next = { ...prev };
      if (next[endpointId]) {
        delete next[endpointId];
      } else {
        next[endpointId] = true;
      }
      return next;
    });
  };

  const handleDuplicateClick = async () => {
    try {
      const allNames = endpoints.map((ep) => ep.name);
      await duplicateEndpoints(selectedEndpoints, allNames);
      setRowSelection({});
    } catch {
      // Error state is set by useDuplicateEndpoints and displayed via the Alert below
    }
  };

  const handleDeleteClick = () => {
    setDeleteModalEndpoints(selectedEndpoints);
  };

  const handleDeleteSuccess = () => {
    setDeleteModalEndpoints([]);
    setRowSelection({});
    refetch();
    onEndpointDeleted?.();
  };

  const isFiltered = searchFilter.trim().length > 0 || filter.providers.length > 0;
  const emptyState = useMemo(() => {
    if (isFiltered && filteredEndpoints.length === 0) {
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
      return <QuickStartTemplates />;
    }

    return null;
  }, [endpoints, filteredEndpoints, isFiltered]);

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

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {duplicateError && (
        <Alert
          componentId="mlflow.gateway.endpoints-list.duplicate-error"
          type="error"
          message={formatMessage({
            defaultMessage: 'Failed to duplicate some endpoints. Please try again.',
            description: 'Gateway > Endpoints list > Duplicate error message',
          })}
          closable={false}
        />
      )}
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
        <div css={{ marginLeft: 'auto', display: 'flex', gap: theme.spacing.sm }}>
          <Link componentId="mlflow.gateway.endpoints-list.create-link" to={GatewayRoutes.createEndpointPageRoute}>
            <Button componentId="mlflow.gateway.endpoints-list.create-button" type="primary">
              <FormattedMessage
                defaultMessage="Create"
                description="Gateway > Endpoints list > Create endpoint button"
              />
            </Button>
          </Link>
          <Button
            componentId="mlflow.gateway.endpoints-list.duplicate-button"
            disabled={selectedCount === 0 || isDuplicating}
            loading={isDuplicating}
            onClick={handleDuplicateClick}
          >
            {selectedCount > 0 ? (
              <FormattedMessage
                defaultMessage="Duplicate ({count})"
                description="Gateway > Endpoints list > Duplicate button with count"
                values={{ count: selectedCount }}
              />
            ) : (
              <FormattedMessage defaultMessage="Duplicate" description="Gateway > Endpoints list > Duplicate button" />
            )}
          </Button>
          <Button
            componentId="mlflow.gateway.endpoints-list.delete-button"
            disabled={selectedCount === 0}
            danger
            onClick={handleDeleteClick}
          >
            {selectedCount > 0 ? (
              <FormattedMessage
                defaultMessage="Delete ({count})"
                description="Gateway > Endpoints list > Delete button with count"
                values={{ count: selectedCount }}
              />
            ) : (
              <FormattedMessage defaultMessage="Delete" description="Gateway > Endpoints list > Delete button" />
            )}
          </Button>
        </div>
      </div>

      <Table
        scrollable
        noMinHeight
        empty={emptyState}
        css={{
          borderLeft: `1px solid ${theme.colors.border}`,
          borderRight: `1px solid ${theme.colors.border}`,
          borderTop: `1px solid ${theme.colors.border}`,
          borderBottom: filteredEndpoints.length === 0 ? `1px solid ${theme.colors.border}` : 'none',
          borderRadius: theme.general.borderRadiusBase,
          overflow: 'hidden',
        }}
      >
        <TableRow isHeader>
          <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
            <Checkbox
              componentId="mlflow.gateway.endpoints-list.select-all-checkbox"
              isChecked={someSelected ? null : allSelected}
              onChange={handleSelectAll}
            />
          </TableCell>
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
        </TableRow>
        {filteredEndpoints.map((endpoint) => (
          <EndpointRow
            key={endpoint.endpoint_id}
            endpoint={endpoint}
            bindings={getBindingsForEndpoint(endpoint.endpoint_id)}
            visibleColumns={visibleColumns}
            isSelected={!!rowSelection[endpoint.endpoint_id]}
            onSelectChange={() => handleSelectRow(endpoint.endpoint_id)}
            onViewBindings={() =>
              setBindingsDrawerEndpoint({
                endpointId: endpoint.endpoint_id,
                endpointName: endpoint.name ?? endpoint.endpoint_id,
                bindings: getBindingsForEndpoint(endpoint.endpoint_id),
              })
            }
          />
        ))}
      </Table>

      <EndpointBindingsDrawer
        open={bindingsDrawerEndpoint !== null}
        endpointName={bindingsDrawerEndpoint?.endpointName ?? ''}
        bindings={bindingsDrawerEndpoint?.bindings ?? []}
        onClose={() => setBindingsDrawerEndpoint(null)}
      />

      <BulkDeleteEndpointModal
        open={deleteModalEndpoints.length > 0}
        endpoints={deleteModalEndpoints}
        getBindingsForEndpoint={getBindingsForEndpoint}
        onClose={() => setDeleteModalEndpoints([])}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};
