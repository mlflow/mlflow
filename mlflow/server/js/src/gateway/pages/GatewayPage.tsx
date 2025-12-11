import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { Alert, Button, PlusIcon, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { EndpointsList } from '../components/endpoints/EndpointsList';
import { Link } from '../../common/utils/RoutingUtils';
import GatewayRoutes from '../routes';

const GatewayPage = () => {
  const { theme } = useDesignSystemTheme();
  const { error, refetch } = useEndpointsQuery();

  return (
    <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <Link to={GatewayRoutes.createEndpointPageRoute}>
          <Button componentId="mlflow.gateway.create-endpoint-button" type="primary" icon={<PlusIcon />}>
            <FormattedMessage defaultMessage="Create endpoint" description="Button to create endpoint" />
          </Button>
        </Link>
      </div>
      {error && (
        <div css={{ padding: theme.spacing.md }}>
          <Alert type="error" message={error.message} componentId="mlflow.gateway.error" closable={false} />
        </div>
      )}
      <div css={{ padding: theme.spacing.md, flex: 1, overflow: 'auto' }}>
        <EndpointsList onEndpointDeleted={() => refetch()} />
      </div>
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, GatewayPage);
