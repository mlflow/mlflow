import { ChainIcon, ChartLineIcon, CloudModelIcon, KeyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import GatewayRoutes from '../../gateway/routes';
import { matchPath } from '../utils/RoutingUtils';
import type { Location } from '../utils/RoutingUtils';
import { MlflowSidebarLink } from './MlflowSidebarLink';

const isEndpointsActive = (location: Location) => Boolean(matchPath('/gateway', location.pathname));
const isUsageActive = (location: Location) => Boolean(matchPath('/gateway/usage', location.pathname));
const isApiKeysActive = (location: Location) => Boolean(matchPath('/gateway/api-keys', location.pathname));

export const MlflowSidebarGatewayItems = ({ collapsed }: { collapsed: boolean }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          justifyContent: collapsed ? 'center' : 'flex-start',
          paddingLeft: collapsed ? 0 : theme.spacing.sm,
          paddingBlock: theme.spacing.xs,
          border: collapsed ? `1px solid ${theme.colors.actionDefaultBorderDefault}` : 'none',
          borderRadius: theme.borders.borderRadiusSm,
          marginTop: collapsed ? theme.spacing.sm : 0,
          marginBottom: collapsed ? theme.spacing.sm : 0,
        }}
      >
        <CloudModelIcon />
        {!collapsed && <FormattedMessage defaultMessage="AI Gateway" description="Sidebar link for gateway" />}
      </div>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={GatewayRoutes.gatewayPageRoute}
        componentId="mlflow.sidebar.gateway_endpoints_tab_link"
        isActive={isEndpointsActive}
        icon={<ChainIcon />}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="Endpoints" description="Sidebar link for gateway endpoints" />
      </MlflowSidebarLink>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={GatewayRoutes.usagePageRoute}
        componentId="mlflow.sidebar.gateway_usage_tab_link"
        isActive={isUsageActive}
        icon={<ChartLineIcon />}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="Usage" description="Sidebar link for gateway usage" />
      </MlflowSidebarLink>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={GatewayRoutes.apiKeysPageRoute}
        componentId="mlflow.sidebar.gateway_api_keys_tab_link"
        isActive={isApiKeysActive}
        icon={<KeyIcon />}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="API Keys" description="Sidebar link for gateway API keys" />
      </MlflowSidebarLink>
    </div>
  );
};
