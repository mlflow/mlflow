import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Checkbox,
  Empty,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tabs,
  Tag,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { Link, useLocation, useSearchParams } from '../../common/utils/RoutingUtils';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { performLogout } from '../auth-utils';
import { ConfirmationModal } from '../ConfirmationModal';
import AdminRoutes, { AdminRoutePaths } from '../routes';
import { useTableSelection } from '../useTableSelection';
import {
  useCurrentUserAdminWorkspaces,
  useCurrentUserIsAdmin,
  useCurrentUserIsWorkspaceAdmin,
  useCurrentUserQuery,
  useUsersQuery,
  useDeleteUser,
  useRolesQuery,
  useDeleteRole,
  useWithSettingsReturnTo,
} from '../hooks';
import { isWorkspaceAdminRole } from '../types';
import { CreateUserModal } from '../components/CreateUserModal';
import { CreateRoleModal } from '../components/CreateRoleModal';
import { UserRolesCell } from '../components/UserRolesCell';

const UsersTab = () => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const { data: usersData, isLoading, error: queryError } = useUsersQuery();
  const { data: currentUserData } = useCurrentUserQuery();
  const currentUsername = currentUserData?.user?.username;
  const deleteUser = useDeleteUser();
  const withReturnTo = useWithSettingsReturnTo();
  // Bulk-delete + row checkboxes are platform-admin-only; Create User is
  // open to workspace admins. ``rolesScopeWorkspace`` keeps the per-row
  // Roles cell aligned with the page's per-workspace scope: workspace
  // managers should only see roles in the active workspace, including for
  // their own row (the backend self-check returns global roles).
  const isAdmin = useCurrentUserIsAdmin();
  const isWorkspaceAdmin = useCurrentUserIsWorkspaceAdmin();
  const canCreateUser = isAdmin || isWorkspaceAdmin;
  const activeWorkspace = useActiveWorkspace();
  const rolesScopeWorkspace = isAdmin ? null : activeWorkspace;

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const users = useMemo(() => usersData?.users ?? [], [usersData]);
  const {
    visibleSelected: visibleSelectedUsernames,
    isAllSelected: allSelected,
    toggleItem: toggleUserSelection,
    toggleAll: toggleSelectAll,
    clear: clearSelection,
  } = useTableSelection(users, 'username');

  const handleBulkDelete = async () => {
    setError(null);
    const targets = Array.from(visibleSelectedUsernames);
    // Detect self-delete *before* firing the requests so we can fall through
    // to ``performLogout`` even if e.g. only the self-delete row succeeds.
    const includesSelfDelete = currentUsername != null && visibleSelectedUsernames.has(currentUsername);
    const results = await Promise.allSettled(targets.map((u) => deleteUser.mutateAsync(u)));
    const failures = results.filter((r) => r.status === 'rejected') as PromiseRejectedResult[];
    if (failures.length > 0) {
      setError(`Failed to delete ${failures.length}/${targets.length} users: ${failures[0].reason?.message ?? ''}`);
    }
    clearSelection();
    setBulkDeleteOpen(false);
    // If the current user just deleted themselves and the request succeeded,
    // the browser still has stale Basic Auth credentials that will 401 every
    // subsequent request. Force a logout to clear the cached realm.
    const selfDeleteIndex = currentUsername != null ? targets.indexOf(currentUsername) : -1;
    const selfDeleteSucceeded =
      includesSelfDelete && selfDeleteIndex >= 0 && results[selfDeleteIndex]?.status === 'fulfilled';
    if (selfDeleteSucceeded) {
      await performLogout(queryClient);
    }
  };

  if (isLoading) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
          minHeight: 200,
        }}
      >
        <Spinner size="small" />
      </div>
    );
  }

  if (queryError) {
    return (
      <Alert
        componentId="admin.users.query_error"
        type="error"
        message="Failed to load users"
        description={(queryError as Error)?.message || 'An error occurred while fetching users.'}
      />
    );
  }

  const emptyState =
    users.length === 0 ? (
      <Empty
        title={<FormattedMessage defaultMessage="No users" description="Empty state title for users table" />}
        description={
          <FormattedMessage
            defaultMessage="Create a user to get started."
            description="Empty state description for users table"
          />
        }
      />
    ) : null;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {error && (
        <Alert componentId="admin.users.error" type="error" message={error} closable onClose={() => setError(null)} />
      )}
      {canCreateUser && (
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-end',
            alignItems: 'center',
            gap: theme.spacing.sm,
          }}
        >
          {isAdmin && (
            <Button
              componentId="admin.users.bulk_delete_button"
              danger
              disabled={visibleSelectedUsernames.size === 0}
              onClick={() => setBulkDeleteOpen(true)}
            >
              {visibleSelectedUsernames.size === 0 ? (
                <FormattedMessage
                  defaultMessage="Delete"
                  description="Bulk-delete button on the users table (no rows selected)"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Delete ({count})"
                  description="Bulk-delete button on the users table"
                  values={{ count: visibleSelectedUsernames.size }}
                />
              )}
            </Button>
          )}
          <Button componentId="admin.users.create_button" type="primary" onClick={() => setShowCreateModal(true)}>
            <FormattedMessage defaultMessage="Create User" description="Button to create a new user" />
          </Button>
        </div>
      )}
      <Table
        scrollable
        noMinHeight
        empty={emptyState}
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.general.borderRadiusBase,
          overflow: 'hidden',
        }}
      >
        <TableRow isHeader>
          {isAdmin && (
            <TableHeader componentId="admin.users.select_header" css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
              <Checkbox
                componentId="admin.users.select_all"
                isChecked={allSelected}
                onChange={toggleSelectAll}
                aria-label="Select all users"
              />
            </TableHeader>
          )}
          <TableHeader componentId="admin.users.username_header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Username" description="Users table username header" />
          </TableHeader>
          <TableHeader componentId="admin.users.roles_header" css={{ flex: 2 }}>
            <FormattedMessage
              defaultMessage="Roles"
              description="Users table roles header — roles render as multiple <workspace> → <role_name> lines per user"
            />
          </TableHeader>
          <TableHeader componentId="admin.users.admin_header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Admin" description="Users table admin header" />
          </TableHeader>
        </TableRow>
        {users.map((user) => (
          <TableRow key={user.username}>
            {isAdmin && (
              <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
                <Checkbox
                  componentId="admin.users.select_row"
                  isChecked={visibleSelectedUsernames.has(user.username)}
                  onChange={() => toggleUserSelection(user.username)}
                  aria-label={`Select user ${user.username}`}
                />
              </TableCell>
            )}
            <TableCell css={{ flex: 2 }}>
              <Link
                componentId="admin.users.username_link"
                to={withReturnTo(AdminRoutes.getUserDetailRoute(user.username))}
              >
                {user.username}
              </Link>
            </TableCell>
            <TableCell css={{ flex: 2 }}>
              <UserRolesCell roles={user.roles ?? []} scopeWorkspace={rolesScopeWorkspace} />
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              {user.is_admin ? (
                <Tag componentId="admin.users.admin_tag" color="indigo">
                  Admin
                </Tag>
              ) : (
                <Typography.Text color="secondary">—</Typography.Text>
              )}
            </TableCell>
          </TableRow>
        ))}
      </Table>
      <CreateUserModal open={showCreateModal} onClose={() => setShowCreateModal(false)} />
      <ConfirmationModal
        componentId="admin.users.bulk_delete_modal"
        title="Delete users"
        visible={bulkDeleteOpen}
        onCancel={() => setBulkDeleteOpen(false)}
        onConfirm={handleBulkDelete}
        isLoading={deleteUser.isLoading}
        message={
          <>
            Delete {visibleSelectedUsernames.size} user{visibleSelectedUsernames.size === 1 ? '' : 's'}? This action
            cannot be undone.
          </>
        }
      />
    </div>
  );
};

const RolesTab = () => {
  const { theme } = useDesignSystemTheme();
  // Per-workspace scope: platform admins fetch unscoped; workspace managers
  // pass the active workspace (only one viewable at a time on this page).
  // ``canManageRoles`` checks the *active* workspace specifically — managing
  // workspace A while currently in B means we can't create/delete here.
  const isAdmin = useCurrentUserIsAdmin();
  const adminWorkspaces = useCurrentUserAdminWorkspaces();
  const activeWorkspace = useActiveWorkspace();
  const canManageRoles = isAdmin || (activeWorkspace !== null && adminWorkspaces.has(activeWorkspace));
  const queryWorkspace = isAdmin ? undefined : (activeWorkspace ?? undefined);
  const queryEnabled = isAdmin || Boolean(activeWorkspace);
  const { data: rolesData, isLoading, error: queryError } = useRolesQuery(queryWorkspace, { enabled: queryEnabled });
  const deleteRole = useDeleteRole();
  const withReturnTo = useWithSettingsReturnTo();

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const roles = useMemo(() => rolesData?.roles ?? [], [rolesData]);
  const {
    visibleSelected: visibleSelectedRoleIds,
    isAllSelected: allSelected,
    toggleItem: toggleRoleSelection,
    toggleAll: toggleSelectAll,
    clear: clearSelection,
  } = useTableSelection(roles, 'id');

  const handleBulkDelete = async () => {
    setError(null);
    const targets = Array.from(visibleSelectedRoleIds);
    const results = await Promise.allSettled(targets.map((id) => deleteRole.mutateAsync(id)));
    const failures = results.filter((r) => r.status === 'rejected') as PromiseRejectedResult[];
    if (failures.length > 0) {
      setError(`Failed to delete ${failures.length}/${targets.length} roles: ${failures[0].reason?.message ?? ''}`);
    }
    clearSelection();
    setBulkDeleteOpen(false);
  };

  // No active workspace + non-admin: skip the guaranteed 403, prompt instead.
  if (!queryEnabled) {
    return (
      <Empty
        title={
          <FormattedMessage
            defaultMessage="Select a workspace"
            description="Roles tab empty state shown to workspace admins without an active workspace"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Pick a workspace from the workspace selector to see its roles."
            description="Roles tab empty state body when no workspace is selected"
          />
        }
      />
    );
  }

  if (isLoading) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
          minHeight: 200,
        }}
      >
        <Spinner size="small" />
      </div>
    );
  }

  if (queryError) {
    return (
      <Alert
        componentId="admin.roles.query_error"
        type="error"
        message="Failed to load roles"
        description={(queryError as Error)?.message || 'An error occurred while fetching roles.'}
      />
    );
  }

  const emptyState =
    roles.length === 0 ? (
      <Empty
        title={<FormattedMessage defaultMessage="No roles" description="Empty state title for roles table" />}
        description={
          <FormattedMessage
            defaultMessage="Create a role to assign permissions to users."
            description="Empty state description for roles table"
          />
        }
      />
    ) : null;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {error && (
        <Alert componentId="admin.roles.error" type="error" message={error} closable onClose={() => setError(null)} />
      )}
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          alignItems: 'center',
          gap: theme.spacing.sm,
        }}
      >
        {canManageRoles && (
          <Button
            componentId="admin.roles.bulk_delete_button"
            danger
            disabled={visibleSelectedRoleIds.size === 0}
            onClick={() => setBulkDeleteOpen(true)}
          >
            {visibleSelectedRoleIds.size === 0 ? (
              <FormattedMessage
                defaultMessage="Delete"
                description="Bulk-delete button on the roles table (no rows selected)"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Delete ({count})"
                description="Bulk-delete button on the roles table"
                values={{ count: visibleSelectedRoleIds.size }}
              />
            )}
          </Button>
        )}
        {canManageRoles && (
          <Button componentId="admin.roles.create_button" type="primary" onClick={() => setShowCreateModal(true)}>
            <FormattedMessage defaultMessage="Create Role" description="Button to create a new role" />
          </Button>
        )}
      </div>
      <Table
        scrollable
        noMinHeight
        empty={emptyState}
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.general.borderRadiusBase,
          overflow: 'hidden',
        }}
      >
        <TableRow isHeader>
          {canManageRoles && (
            <TableHeader componentId="admin.roles.select_header" css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
              <Checkbox
                componentId="admin.roles.select_all"
                isChecked={allSelected}
                onChange={toggleSelectAll}
                aria-label="Select all roles"
              />
            </TableHeader>
          )}
          <TableHeader componentId="admin.roles.name_header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Name" description="Roles table name header" />
          </TableHeader>
          <TableHeader componentId="admin.roles.workspace_header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Workspace" description="Roles table workspace header" />
          </TableHeader>
          <TableHeader componentId="admin.roles.description_header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Description" description="Roles table description header" />
          </TableHeader>
          <TableHeader componentId="admin.roles.admin_role_header" css={{ flex: 1 }}>
            <FormattedMessage
              defaultMessage="Workspace Manager"
              description="Roles table column flagging roles that grant workspace-level MANAGE"
            />
          </TableHeader>
        </TableRow>
        {roles.map((role) => (
          <TableRow key={role.id}>
            {canManageRoles && (
              <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
                <Checkbox
                  componentId="admin.roles.select_row"
                  isChecked={visibleSelectedRoleIds.has(role.id)}
                  onChange={() => toggleRoleSelection(role.id)}
                  aria-label={`Select role ${role.name}`}
                />
              </TableCell>
            )}
            <TableCell css={{ flex: 2 }}>
              <Link componentId="admin.roles.name_link" to={withReturnTo(AdminRoutes.getRoleDetailRoute(role.id))}>
                {role.name}
              </Link>
            </TableCell>
            <TableCell css={{ flex: 1 }}>{role.workspace}</TableCell>
            <TableCell css={{ flex: 2 }}>{role.description || '-'}</TableCell>
            <TableCell css={{ flex: 1 }}>
              {isWorkspaceAdminRole(role) ? (
                <Tag componentId="admin.roles.admin_tag" color="indigo">
                  Manager
                </Tag>
              ) : null}
            </TableCell>
          </TableRow>
        ))}
      </Table>
      <CreateRoleModal open={showCreateModal} onClose={() => setShowCreateModal(false)} />
      <ConfirmationModal
        componentId="admin.roles.bulk_delete_modal"
        title="Delete roles"
        visible={bulkDeleteOpen}
        onCancel={() => setBulkDeleteOpen(false)}
        onConfirm={handleBulkDelete}
        isLoading={deleteRole.isLoading}
        message={
          <>
            Delete {visibleSelectedRoleIds.size} role{visibleSelectedRoleIds.size === 1 ? '' : 's'}? This action cannot
            be undone.
          </>
        }
      />
    </div>
  );
};

const AdminPage = () => {
  const { theme } = useDesignSystemTheme();
  // Reflect the active tab in the URL (?tab=users|roles) so deep links — e.g.
  // the RoleDetailPage breadcrumb back to /admin?tab=roles — land on the
  // expected tab and a refresh preserves it.
  const [searchParams, setSearchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab');
  const activeTab = tabFromUrl === 'roles' ? 'roles' : 'users';

  const activeWorkspace = useActiveWorkspace();
  // Mode is path-driven, not role-driven: ``/admin`` is the cross-workspace
  // platform-admin view; ``/admin/ws`` (with the workspace name in the
  // ``?workspace=`` query param) is the per-workspace management view. A
  // deep link reads the same way for anyone authorized to follow it, and
  // the ``?workspace=`` value is still picked up by ``WorkspaceRouterSync``
  // to keep the global ``activeWorkspace`` in sync.
  const { pathname } = useLocation();
  const isWorkspaceScoped = pathname === AdminRoutePaths.workspaceManagementPage;

  // The route definition's static ``getPageTitle`` is set by ``MlflowRootRoute``
  // *after* this component's effects (parent effects run after children's), so
  // we override on a microtask to land last and reflect the per-workspace
  // header in the browser tab.
  useEffect(() => {
    const desired = isWorkspaceScoped ? 'Workspace Manager - MLflow' : 'Platform Admin - MLflow';
    queueMicrotask(() => {
      document.title = desired;
    });
  }, [isWorkspaceScoped]);

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <div
              css={{
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
                padding: theme.spacing.sm,
                display: 'flex',
              }}
            >
              <UserIcon />
            </div>
            <Typography.Title withoutMargins level={2}>
              {isWorkspaceScoped ? (
                <FormattedMessage
                  defaultMessage="Workspace Manager"
                  description="Admin page title shown when the URL is scoped to a single workspace via ?workspace=…"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Platform Admin"
                  description="Admin page title shown when the URL has no ?workspace=… (cross-workspace platform-admin view)"
                />
              )}
            </Typography.Title>
          </div>
          {isWorkspaceScoped && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Workspace: {workspace}"
                description="Subtitle on the admin page identifying the active workspace when the URL is per-workspace scoped"
                values={{ workspace: <code>{activeWorkspace}</code> }}
              />
            </Typography.Text>
          )}
        </div>
        <Tabs.Root
          componentId="admin.tabs"
          valueHasNoPii
          value={activeTab}
          onValueChange={(value) => {
            const next = new URLSearchParams(searchParams);
            if (value === 'users') {
              next.delete('tab');
            } else {
              next.set('tab', value);
            }
            setSearchParams(next, { replace: true });
          }}
        >
          <Tabs.List>
            <Tabs.Trigger value="users">
              <FormattedMessage defaultMessage="Users" description="Admin users tab" />
            </Tabs.Trigger>
            <Tabs.Trigger value="roles">
              <FormattedMessage defaultMessage="Roles" description="Admin roles tab" />
            </Tabs.Trigger>
          </Tabs.List>
          <Tabs.Content value="users" css={{ paddingTop: theme.spacing.md }}>
            <UsersTab />
          </Tabs.Content>
          <Tabs.Content value="roles" css={{ paddingTop: theme.spacing.md }}>
            <RolesTab />
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </ScrollablePageWrapper>
  );
};

export default AdminPage;
