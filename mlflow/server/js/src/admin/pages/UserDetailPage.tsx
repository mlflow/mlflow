import { useMemo, useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  Empty,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Tabs,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { Link, useParams, useSearchParams } from '../../common/utils/RoutingUtils';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { useCurrentUserIsAdmin, useUserRolesQuery, useUsersQuery, useWithSettingsReturnTo } from '../hooks';
import { useWorkspacesEnabled } from '../../experiment-tracking/hooks/useServerInfo';
import AdminRoutes from '../routes';
import { EditAccessModal } from '../components/EditAccessModal';
import { PermissionsSection } from '../../account/PermissionsSection';
import { isSyntheticUserRole, isWorkspaceAdminRole } from '../types';

const UserDetailPage = () => {
  const { theme } = useDesignSystemTheme();
  // ``useParams`` already URL-decodes path params, so the value here is the
  // raw username (matching what ``getUserDetailRoute(username)`` originally
  // encoded). Decoding it again would either throw on usernames containing
  // ``%`` or silently corrupt them.
  const { username = '' } = useParams<{ username: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab');
  const activeTab = tabFromUrl === 'permissions' ? 'permissions' : 'roles';

  const { data: rolesData, isLoading: rolesLoading, error: rolesErrorRaw } = useUserRolesQuery(username);
  // ``useUsersQuery`` is admin-only; we use it just to surface the
  // ``is_admin`` flag for this user. Failing this should not block the
  // permissions view, so the error is not fatal here.
  const { data: usersData } = useUsersQuery();
  const { workspacesEnabled } = useWorkspacesEnabled();
  const withReturnTo = useWithSettingsReturnTo();
  // For workspace managers viewing a user outside their workspaces,
  // ``validate_can_view_user_roles`` 403s — that's expected, not a failure.
  // Hide the red error banner for non-admins so the page degrades to an
  // empty "no visible roles" state instead of looking broken; platform
  // admins still see real fetch failures.
  const isAdmin = useCurrentUserIsAdmin();
  const rolesError = isAdmin ? rolesErrorRaw : null;
  // The /admin entry is per-workspace, so non-admin viewers should only
  // see roles in the active workspace — including for self, where the
  // backend's self-check returns global roles.
  const activeWorkspace = useActiveWorkspace();

  const [editAccessOpen, setEditAccessOpen] = useState(false);

  const user = useMemo(() => usersData?.users?.find((u) => u.username === username), [usersData, username]);
  // Direct grants live on the synthetic ``__user_<id>__`` role surfaced via
  // ``listUserRoles`` alongside regular roles. ``PermissionsSection`` splits
  // them on display (synthetic → "Direct" label, regular → role name), so we
  // pass the unfiltered list down. The Roles tab below filters synthetic out
  // separately via ``displayRoles``.
  // Workspace managers see only roles in the active workspace; admins see all.
  const roles = useMemo(() => {
    const all = rolesData?.roles ?? [];
    return isAdmin || !activeWorkspace ? all : all.filter((r) => r.workspace === activeWorkspace);
  }, [rolesData, isAdmin, activeWorkspace]);
  const displayRoles = useMemo(() => roles.filter((r) => !isSyntheticUserRole(r.name)), [roles]);

  const rolesEmptyState =
    displayRoles.length === 0 ? (
      <Empty
        title="No roles"
        description={
          isAdmin
            ? 'This user has not been assigned to any roles.'
            : activeWorkspace
              ? `No roles for this user in workspace "${activeWorkspace}".`
              : 'No roles visible in workspaces you manage.'
        }
      />
    ) : null;

  if (!username) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="admin.user_detail.invalid_username"
            type="error"
            message="Invalid username"
            description="The requested user could not be loaded because the URL contains an invalid username."
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <Breadcrumb includeTrailingCaret>
          <Breadcrumb.Item>
            <Link componentId="admin.user_detail.breadcrumb_admin" to={withReturnTo(AdminRoutes.adminPageRoute)}>
              Platform Admin
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>{username}</Breadcrumb.Item>
        </Breadcrumb>

        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Typography.Title withoutMargins level={2}>
                {username}
              </Typography.Title>
              {user?.is_admin && (
                <Tag componentId="admin.user_detail.admin_tag" color="indigo">
                  Admin
                </Tag>
              )}
            </div>
            <Typography.Text color="secondary">Permissions and roles assigned to this user.</Typography.Text>
          </div>
          <Button
            componentId="admin.user_detail.edit_access_button"
            type="primary"
            onClick={() => setEditAccessOpen(true)}
          >
            Edit access
          </Button>
        </div>

        <Tabs.Root
          componentId="admin.user_detail.tabs"
          valueHasNoPii
          value={activeTab}
          onValueChange={(value) => {
            const next = new URLSearchParams(searchParams);
            if (value === 'roles') {
              next.delete('tab');
            } else {
              next.set('tab', value);
            }
            setSearchParams(next, { replace: true });
          }}
        >
          <Tabs.List>
            <Tabs.Trigger value="roles">Roles</Tabs.Trigger>
            <Tabs.Trigger value="permissions">Permissions</Tabs.Trigger>
          </Tabs.List>
          <Tabs.Content value="roles" css={{ paddingTop: theme.spacing.md }}>
            {rolesLoading ? (
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: theme.spacing.lg,
                  minHeight: 200,
                }}
              >
                <Spinner size="small" />
              </div>
            ) : rolesError ? (
              <Alert
                componentId="admin.user_detail.roles_error"
                type="error"
                message="Failed to load roles"
                description={(rolesError as Error)?.message || "An error occurred while fetching this user's roles."}
              />
            ) : (
              <Table
                scrollable
                noMinHeight
                empty={rolesEmptyState}
                css={{
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.general.borderRadiusBase,
                  overflow: 'hidden',
                }}
              >
                <TableRow isHeader>
                  <TableHeader componentId="admin.user_detail.roles.name_header" css={{ flex: 2 }}>
                    Role
                  </TableHeader>
                  {workspacesEnabled && (
                    <TableHeader componentId="admin.user_detail.roles.workspace_header" css={{ flex: 1 }}>
                      Workspace
                    </TableHeader>
                  )}
                  <TableHeader componentId="admin.user_detail.roles.admin_header" css={{ flex: 1 }}>
                    {workspacesEnabled ? 'Workspace Manager' : 'Admin'}
                  </TableHeader>
                </TableRow>
                {displayRoles.map((role) => (
                  <TableRow key={role.id}>
                    <TableCell css={{ flex: 2 }}>
                      <Link componentId="admin.user_detail.role_link" to={AdminRoutes.getRoleDetailRoute(role.id)}>
                        {role.name}
                      </Link>
                    </TableCell>
                    {workspacesEnabled && <TableCell css={{ flex: 1 }}>{role.workspace}</TableCell>}
                    <TableCell css={{ flex: 1 }}>
                      {isWorkspaceAdminRole(role) ? (
                        <Tag componentId="admin.user_detail.roles.admin_tag" color="indigo">
                          {workspacesEnabled ? 'Manager' : 'Admin'}
                        </Tag>
                      ) : null}
                    </TableCell>
                  </TableRow>
                ))}
              </Table>
            )}
          </Tabs.Content>
          <Tabs.Content
            value="permissions"
            css={{
              paddingTop: theme.spacing.md,
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.md,
            }}
          >
            <PermissionsSection
              componentId="admin.user_detail.permissions"
              roles={roles}
              isLoading={rolesLoading}
              rolesError={rolesError}
              workspacesEnabled={workspacesEnabled}
            />
          </Tabs.Content>
        </Tabs.Root>

        <EditAccessModal open={editAccessOpen} onClose={() => setEditAccessOpen(false)} username={username} />
      </div>
    </ScrollablePageWrapper>
  );
};

export default UserDetailPage;
