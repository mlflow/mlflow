import {
  CloudModelIcon,
  KeyIcon,
  LayerIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useLocation } from '../../../common/utils/RoutingUtils';
import GatewayRoutes from '../../routes';

const SIDE_NAV_WIDTH = 160;
const SIDE_NAV_COLLAPSED_WIDTH = 32;
const FULL_WIDTH_CLASS_NAME = 'mlflow-gateway-side-nav-full';
const COLLAPSED_CLASS_NAME = 'mlflow-gateway-side-nav-collapsed';

export type GatewayTabName = 'endpoints' | 'models' | 'api-keys';

interface NavItem {
  label: React.ReactNode;
  icon: React.ReactNode;
  tabName: GatewayTabName;
  to: string;
}

const navItems: NavItem[] = [
  {
    label: (
      <FormattedMessage defaultMessage="Endpoints" description="Label for the endpoints tab in the Gateway sidebar" />
    ),
    icon: <CloudModelIcon />,
    tabName: 'endpoints',
    to: GatewayRoutes.gatewayPageRoute,
  },
  {
    label: <FormattedMessage defaultMessage="Models" description="Label for the models tab in the Gateway sidebar" />,
    icon: <LayerIcon />,
    tabName: 'models',
    to: GatewayRoutes.modelDefinitionsPageRoute,
  },
  {
    label: (
      <FormattedMessage defaultMessage="API Keys" description="Label for the API keys tab in the Gateway sidebar" />
    ),
    icon: <KeyIcon />,
    tabName: 'api-keys',
    to: GatewayRoutes.apiKeysPageRoute,
  },
];

export const GatewaySideNav = ({ activeTab }: { activeTab: GatewayTabName }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        paddingTop: theme.spacing.sm,
        paddingRight: theme.spacing.sm,
        borderRight: `1px solid ${theme.colors.border}`,
        boxSizing: 'content-box',
        width: SIDE_NAV_COLLAPSED_WIDTH,
        [`& .${COLLAPSED_CLASS_NAME}`]: {
          display: 'flex',
        },
        [`& .${FULL_WIDTH_CLASS_NAME}`]: {
          display: 'none',
        },
        [theme.responsive.mediaQueries.xl]: {
          width: SIDE_NAV_WIDTH,
          [`& .${COLLAPSED_CLASS_NAME}`]: {
            display: 'none',
          },
          [`& .${FULL_WIDTH_CLASS_NAME}`]: {
            display: 'flex',
          },
        },
      }}
    >
      {navItems.map((item) => {
        const isActive = activeTab === item.tabName;

        return (
          <Link key={item.tabName} to={item.to}>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderRadius: theme.borders.borderRadiusSm,
                cursor: 'pointer',
                backgroundColor: isActive ? theme.colors.actionDefaultBackgroundHover : undefined,
                color: isActive ? theme.colors.actionDefaultIconHover : theme.colors.actionDefaultIconDefault,
                height: theme.typography.lineHeightBase,
                boxSizing: 'content-box',
                ':hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
              }}
            >
              <Tooltip
                componentId={`mlflow.gateway.side-nav.${item.tabName}.tooltip`}
                content={item.label}
                side="right"
                delayDuration={0}
              >
                <span className={COLLAPSED_CLASS_NAME}>{item.icon}</span>
              </Tooltip>
              <span className={FULL_WIDTH_CLASS_NAME}>{item.icon}</span>
              <Typography.Text className={FULL_WIDTH_CLASS_NAME} bold={isActive} color="primary">
                {item.label}
              </Typography.Text>
            </div>
          </Link>
        );
      })}
    </div>
  );
};
