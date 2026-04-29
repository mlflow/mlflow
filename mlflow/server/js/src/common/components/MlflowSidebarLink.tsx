import type { Location, To } from '../utils/RoutingUtils';
import { Link, useLocation } from '../utils/RoutingUtils';
import { Tooltip, useDesignSystemTheme } from '@databricks/design-system';

export const MlflowSidebarLink = ({
  className,
  to,
  componentId,
  icon,
  children,
  isActive,
  onClick,
  openInNewTab = false,
  collapsed = false,
  disableWorkspacePrefix = false,
  tooltipContent,
}: {
  className?: string;
  to: To;
  componentId: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  isActive: (location: Location) => boolean;
  onClick?: () => void;
  openInNewTab?: boolean;
  collapsed?: boolean;
  disableWorkspacePrefix?: boolean;
  tooltipContent?: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const location = useLocation();

  return (
    <li key={componentId}>
      <Tooltip
        componentId={`${componentId}.tooltip`}
        open={!collapsed ? false : undefined}
        content={tooltipContent ?? children}
        side="right"
        delayDuration={0}
      >
        <Link
          componentId={componentId}
          disableWorkspacePrefix={disableWorkspacePrefix}
          to={to}
          aria-current={isActive(location) ? 'page' : undefined}
          className={className}
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            color: theme.colors.textPrimary,
            paddingInline: theme.spacing.sm,
            justifyContent: collapsed ? 'center' : 'flex-start',
            paddingBlock: theme.spacing.sm,
            borderRadius: theme.borders.borderRadiusSm,
            '&:hover': {
              color: theme.colors.actionLinkHover,
              backgroundColor: theme.colors.actionDefaultBackgroundHover,
            },
            '&[aria-current="page"]': {
              backgroundColor: theme.colors.actionDefaultBackgroundPress,
              color: theme.isDarkMode ? theme.colors.blue300 : theme.colors.blue700,
              fontWeight: theme.typography.typographyBoldFontWeight,
            },
          }}
          onClick={() => {
            onClick?.();
          }}
          target={openInNewTab ? '_blank' : undefined}
          rel={openInNewTab ? 'noopener noreferrer' : undefined}
        >
          {icon}
          {!collapsed && children}
        </Link>
      </Tooltip>
    </li>
  );
};
