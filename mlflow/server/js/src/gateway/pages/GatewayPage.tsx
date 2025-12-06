import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { Alert, Button, Header, PlusIcon, Spacer, useDesignSystemTheme } from '@databricks/design-system';
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
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <FormattedMessage defaultMessage="Gateway" description="Header title for the gateway configuration page" />
        }
        buttons={
          <Link to={GatewayRoutes.createEndpointPageRoute}>
            <Button componentId="mlflow.gateway.create-endpoint-button" type="primary" icon={<PlusIcon />}>
              <FormattedMessage defaultMessage="Create endpoint" description="Button to create endpoint" />
            </Button>
          </Link>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {error && (
          <>
            <Alert type="error" message={error.message} componentId="mlflow.gateway.error" closable={false} />
            <Spacer />
          </>
        )}
        <div css={{ padding: theme.spacing.md, flex: 1, overflow: 'auto' }}>
          <EndpointsList onEndpointDeleted={() => refetch()} />
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, GatewayPage);
