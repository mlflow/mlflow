import {
  ChainIcon,
  ChartLineIcon,
  CloudModelIcon,
  CreditCardIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GatewayLabel, GatewayNewTag } from './GatewayNewTag';
import GatewayRoutes from '../../gateway/routes';
import { matchPath } from '../utils/RoutingUtils';
import type { Location } from '../utils/RoutingUtils';
import { MlflowSidebarLink } from './MlflowSidebarLink';

const isEndpointsActive = (location: Location) =>
  Boolean(matchPath('/gateway', location.pathname)) || Boolean(matchPath('/gateway/endpoints/*', location.pathname));
const isUsageActive = (location: Location) => Boolean(matchPath('/gateway/usage', location.pathname));
const isBudgetsActive = (location: Location) => Boolean(matchPath('/gateway/budgets', location.pathname));

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
          paddingBlock: collapsed ? 7 : theme.spacing.sm,
          border: collapsed ? `1px solid ${theme.colors.actionDefaultBorderDefault}` : 'none',
          borderRadius: theme.borders.borderRadiusSm,
          marginBottom: collapsed ? theme.spacing.sm : 0,
          boxSizing: 'border-box',
        }}
      >
        <CloudModelIcon />
        {!collapsed && (
          <>
            <GatewayLabel />
            <GatewayNewTag />
          </>
        )}
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
        to={GatewayRoutes.budgetsPageRoute}
        componentId="mlflow.sidebar.gateway_budgets_tab_link"
        isActive={isBudgetsActive}
        icon={<CreditCardIcon />}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="Budgets" description="Sidebar link for gateway budgets" />
      </MlflowSidebarLink>
    </div>
  );
};
