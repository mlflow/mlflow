import { ChainIcon, KeyIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import GatewayRoutes from '../../routes';

const SIDE_NAV_WIDTH = 160;
const SIDE_NAV_COLLAPSED_WIDTH = 32;
const COLLAPSED_CLASS_NAME = 'gateway-side-nav-collapsed';
const FULL_WIDTH_CLASS_NAME = 'gateway-side-nav-full-width';

export type GatewayTab = 'endpoints' | 'api-keys';

interface GatewaySideNavProps {
  activeTab: GatewayTab;
}

const navItems: Array<{
  tab: GatewayTab;
  label: React.ReactNode;
  icon: React.ReactNode;
  to: string;
}> = [
  {
    tab: 'endpoints',
    label: <FormattedMessage defaultMessage="Endpoints" description="Gateway side nav > Endpoints tab" />,
    icon: <ChainIcon />,
    to: GatewayRoutes.gatewayPageRoute,
  },
  {
    tab: 'api-keys',
    label: <FormattedMessage defaultMessage="API Keys" description="Gateway side nav > API Keys tab" />,
    icon: <KeyIcon />,
    to: GatewayRoutes.apiKeysPageRoute,
  },
];

export const GatewaySideNav = ({ activeTab }: GatewaySideNavProps) => {
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
        const isActive = activeTab === item.tab;

        return (
          <Link key={item.tab} to={item.to}>
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
                componentId={`mlflow.gateway.side-nav.${item.tab}.tooltip`}
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
