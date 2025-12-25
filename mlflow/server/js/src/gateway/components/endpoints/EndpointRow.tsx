import {
  Button,
  ChainIcon,
  PencilIcon,
  TableCell,
  TableRow,
  TrashIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import GatewayRoutes from '../../routes';
import type { Endpoint, EndpointBinding } from '../../types';
import { ProviderCell } from './ProviderCell';
import { ModelsCell } from './ModelsCell';
import { BindingsCell } from './BindingsCell';

interface EndpointRowProps {
  endpoint: Endpoint;
  bindings: EndpointBinding[];
  onViewBindings: () => void;
  onDelete: () => void;
}

export const EndpointRow = ({ endpoint, bindings, onViewBindings, onDelete }: EndpointRowProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  return (
    <TableRow>
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
      <TableCell css={{ flex: 1 }}>
        <ProviderCell modelMappings={endpoint.model_mappings} />
      </TableCell>
      <TableCell css={{ flex: 2 }}>
        <ModelsCell modelMappings={endpoint.model_mappings} />
      </TableCell>
      <TableCell css={{ flex: 1 }}>
        <BindingsCell bindings={bindings} onViewBindings={onViewBindings} />
      </TableCell>
      <TableCell css={{ flex: 1 }}>
        <TimeAgo date={new Date(endpoint.last_updated_at)} />
      </TableCell>
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
            onClick={onDelete}
          />
        </div>
      </TableCell>
    </TableRow>
  );
};
