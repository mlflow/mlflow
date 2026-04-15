import { Checkbox, TableCell, TableRow, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../common/utils/RoutingUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import GatewayRoutes from '../../routes';
import type { Endpoint, EndpointBinding } from '../../types';
import { EndpointsColumn } from './EndpointsColumnsButton';
import { ProviderCell } from './ProviderCell';
import { ModelsCell } from './ModelsCell';
import { BindingsCell } from './BindingsCell';

interface EndpointRowProps {
  endpoint: Endpoint;
  bindings: EndpointBinding[];
  visibleColumns: EndpointsColumn[];
  isSelected: boolean;
  onSelectChange: () => void;
  onViewBindings: () => void;
}

export const EndpointRow = ({
  endpoint,
  bindings,
  visibleColumns,
  isSelected,
  onSelectChange,
  onViewBindings,
}: EndpointRowProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <TableRow>
      <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
        <Checkbox
          componentId="mlflow.gateway.endpoints-list.row-checkbox"
          isChecked={isSelected}
          onChange={onSelectChange}
        />
      </TableCell>
      <TableCell css={{ flex: 2 }}>
        <Link
          componentId="mlflow.gateway.endpoints.endpoint_name_link"
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
          <BindingsCell bindings={bindings} onViewBindings={onViewBindings} />
        </TableCell>
      )}
      {visibleColumns.includes(EndpointsColumn.LAST_MODIFIED) && (
        <TableCell css={{ flex: 1 }}>
          <TimeAgo date={new Date(endpoint.last_updated_at)} />
        </TableCell>
      )}
      {visibleColumns.includes(EndpointsColumn.CREATED) && (
        <TableCell css={{ flex: 1 }}>
          <TimeAgo date={new Date(endpoint.created_at)} />
        </TableCell>
      )}
    </TableRow>
  );
};
