import { Outlet } from '../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { Header, Spacer, useDesignSystemTheme, ChainIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { GatewaySideNav, type GatewayTabName } from '../components/side-nav/GatewaySideNav';
import { useLocation } from '../../common/utils/RoutingUtils';
import GatewayRoutes from '../routes';

const GatewayLayout = () => {
  const { theme } = useDesignSystemTheme();
  const location = useLocation();

  // Determine active tab based on current path
  const getActiveTab = (): GatewayTabName => {
    if (location.pathname.includes('/api-keys')) {
      return 'api-keys';
    }
    return 'endpoints';
  };

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <ChainIcon />
            <FormattedMessage defaultMessage="Gateway" description="Header title for the gateway page" />
          </div>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <GatewaySideNav activeTab={getActiveTab()} />
        <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Outlet />
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, GatewayLayout);
