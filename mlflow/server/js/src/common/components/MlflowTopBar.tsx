import {
  Avatar,
  Button,
  DropdownMenu,
  SidebarCollapseIcon,
  SidebarExpandIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { performLogout } from '../../admin/auth-utils';
import { useCurrentUserQuery, useIsAuthAvailable } from '../../admin/hooks';
import AdminRoutes from '../../admin/routes';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { WorkspaceSelector } from '../../workspaces/components/WorkspaceSelector';
import { Link, useNavigate } from '../utils/RoutingUtils';
import { Version } from '../constants';
import { shouldEnableWorkspaces } from '../utils/FeatureUtils';
import { MlflowLogo } from './MlflowLogo';

interface MlflowTopBarProps {
  showSidebar: boolean;
  setShowSidebar: (showSidebar: boolean) => void;
}

/**
 * Thin top-bar that owns the brand row (logo + version + sidebar toggle on
 * the left) and the user-identity widget (avatar + username dropdown on the
 * right).
 *
 * Renders nothing when auth isn't configured — auth-disabled deployments
 * (no basic-auth app) have no user identity to surface, and in that mode
 * the sidebar keeps its original header with the logo / version / toggle.
 * ``useIsAuthAvailable`` returns ``true`` while ``/users/current`` is in
 * flight to avoid hiding the bar during the initial load on auth-enabled
 * deployments; the brief moment on auth-disabled deployments before the
 * request errors is acceptable.
 *
 * The top-bar is intentionally transparent — the chrome background lives on
 * the outer ``MlflowRootLayout`` wrapper so the gradient (in GenAI mode)
 * paints continuously across the top-bar and the sidebar/main row below.
 */
export const MlflowTopBar = ({ showSidebar, setShowSidebar }: MlflowTopBarProps) => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const navigate = useNavigate({ bypassWorkspacePrefix: true });
  const isAuthAvailable = useIsAuthAvailable();
  const { data: currentUserData } = useCurrentUserQuery();
  const username = currentUserData?.user?.username ?? '';
  const workspacesEnabled = shouldEnableWorkspaces();

  if (!isAuthAvailable) {
    return null;
  }

  const toggleSidebar = () => setShowSidebar(!showSidebar);

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: theme.spacing.xl + theme.spacing.md,
        paddingInline: theme.spacing.md,
        gap: theme.spacing.sm,
        flexShrink: 0,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
        <Button
          componentId="mlflow_header.toggle_sidebar_button"
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
          icon={showSidebar ? <SidebarCollapseIcon /> : <SidebarExpandIcon />}
        />
        <Link
          componentId="mlflow.topbar.logo_home_link"
          to={ExperimentTrackingRoutes.rootRoute}
          css={{ display: 'flex', alignItems: 'center' }}
        >
          <MlflowLogo
            css={{
              display: 'block',
              height: theme.spacing.lg,
              color: theme.colors.textPrimary,
            }}
          />
        </Link>
        <Typography.Text size="sm" color="secondary">
          {Version}
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        {workspacesEnabled && (
          <div css={{ width: theme.spacing.xl * 6 }}>
            <WorkspaceSelector />
          </div>
        )}
        {username && (
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <button
                type="button"
                aria-label={`Account menu for ${username}`}
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  background: 'transparent',
                  border: 'none',
                  paddingBlock: theme.spacing.xs,
                  paddingInline: theme.spacing.sm,
                  cursor: 'pointer',
                  borderRadius: theme.borders.borderRadiusSm,
                  color: theme.colors.textPrimary,
                  // Avatar renders the initial inside <abbr title>, which the UA
                  // stylesheet decorates with a dotted underline and help cursor.
                  // Suppress both so the avatar reads as part of this button.
                  '& abbr[title]': {
                    textDecoration: 'none',
                    cursor: 'inherit',
                  },
                  ':hover': {
                    backgroundColor: theme.colors.actionDefaultBackgroundHover,
                  },
                  ':focus-visible': {
                    outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
                    outlineOffset: 2,
                  },
                }}
              >
                <Avatar type="user" size="sm" label={username} />
                <Typography.Text
                  css={{
                    maxWidth: theme.spacing.xl * 5,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {username}
                </Typography.Text>
              </button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end" minWidth={180}>
              <DropdownMenu.Item
                componentId="mlflow.topbar.account"
                onClick={() => navigate(AdminRoutes.accountPageRoute)}
              >
                <FormattedMessage defaultMessage="Account" description="Top bar account menu item" />
              </DropdownMenu.Item>
              <DropdownMenu.Separator />
              <DropdownMenu.Item componentId="mlflow.topbar.logout" onClick={() => performLogout(queryClient)}>
                <FormattedMessage defaultMessage="Logout" description="Top bar logout menu item" />
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        )}
      </div>
    </div>
  );
};
