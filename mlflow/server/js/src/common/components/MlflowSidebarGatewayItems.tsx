import { ChainIcon, CloudModelIcon, KeyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import GatewayRoutes from '../../gateway/routes';
import { matchPath } from '../utils/RoutingUtils';
import type { Location } from '../utils/RoutingUtils';
import { MlflowSidebarLink } from './MlflowSidebarLink';

const isEndpointsActive = (location: Location) => Boolean(matchPath('/gateway', location.pathname));
const isApiKeysActive = (location: Location) => Boolean(matchPath('/gateway/api-keys', location.pathname));

export const MlflowSidebarGatewayItems = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          paddingLeft: theme.spacing.md,
          paddingTop: theme.spacing.xs,
          paddingBottom: theme.spacing.xs,
        }}
      >
        <CloudModelIcon />
        <FormattedMessage defaultMessage="AI Gateway" description="Sidebar link for gateway" />
      </div>
      <MlflowSidebarLink
        css={{ paddingLeft: theme.spacing.lg }}
        to={GatewayRoutes.gatewayPageRoute}
        componentId="mlflow.sidebar.gateway_endpoints_tab_link"
        isActive={isEndpointsActive}
      >
        <ChainIcon />
        <FormattedMessage defaultMessage="Endpoints" description="Sidebar link for gateway endpoints" />
      </MlflowSidebarLink>
      <MlflowSidebarLink
        css={{ paddingLeft: theme.spacing.lg }}
        to={GatewayRoutes.apiKeysPageRoute}
        componentId="mlflow.sidebar.gateway_api_keys_tab_link"
        isActive={isApiKeysActive}
      >
        <KeyIcon />
        <FormattedMessage defaultMessage="API Keys" description="Sidebar link for gateway API keys" />
      </MlflowSidebarLink>
    </div>
  );
};
