import type { Location, To } from '../utils/RoutingUtils';
import { Link, useLocation } from '../utils/RoutingUtils';
import { Tooltip, useDesignSystemTheme } from '@databricks/design-system';

type Theme = ReturnType<typeof useDesignSystemTheme>['theme'];

/**
 * Shared visual for one sidebar row (icon + label, hover + focus
 * states). Used by both ``MlflowSidebarLink`` (router-link rows) and
 * the avatar dropdown trigger in ``MlflowSidebar`` so the rows share
 * spacing, hover colour, and focus outline regardless of whether the
 * underlying element is an ``<a>`` or a ``<button>``.
 */
export const getSidebarItemStyles = (theme: Theme, collapsed: boolean) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing.sm,
  width: '100%',
  color: theme.colors.textPrimary,
  paddingInline: theme.spacing.sm,
  paddingBlock: theme.spacing.sm,
  borderRadius: theme.borders.borderRadiusSm,
  justifyContent: collapsed ? 'center' : 'flex-start',
  '&:hover': {
    color: theme.colors.actionLinkHover,
    backgroundColor: theme.colors.actionDefaultBackgroundHover,
  },
  '&:focus-visible': {
    outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
    outlineOffset: 2,
  },
  '&[aria-current="page"]': {
    backgroundColor: theme.colors.actionDefaultBackgroundPress,
    color: theme.isDarkMode ? theme.colors.blue300 : theme.colors.blue700,
    fontWeight: theme.typography.typographyBoldFontWeight,
  },
});

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
          css={getSidebarItemStyles(theme, collapsed)}
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
