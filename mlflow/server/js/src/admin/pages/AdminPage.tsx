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
import { Link, useSearchParams } from '../../common/utils/RoutingUtils';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { performLogout } from '../auth-utils';
import { ConfirmationModal } from '../ConfirmationModal';
import AdminRoutes from '../routes';
import { useTableSelection } from '../useTableSelection';
import {
  useCurrentUserIsAdmin,
  useCurrentUserIsWorkspaceAdmin,
  useCurrentUserQuery,
  useUsersQuery,
  useDeleteUser,
  useRolesQuery,
  useDeleteRole,
  useUserRolesQuery,
  useWithSettingsReturnTo,
} from '../hooks';
import { isWorkspaceAdminRole } from '../types';
import { CreateUserModal } from '../components/CreateUserModal';
import { CreateRoleModal } from '../components/CreateRoleModal';

// Renders one line per role assigned to a user, formatted as
// `<workspace> → <role_name>`. Mirrors the `<scope> → <value>` shape used
// elsewhere in the admin UI (e.g. the Account page's permission lines like
// `experiment:* → READ`). Each row issues its own request — React Query
// caches per-username so subsequent re-renders don't re-fetch.
// ``enabled`` is false for workspace managers to avoid an N+1 burst of
// mostly-403 requests (one fetch per visible user row).
const UserRolesCell = ({ username, enabled }: { username: string; enabled: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const { data, isLoading, error } = useUserRolesQuery(username, { enabled });
  if (!enabled) {
    return <Typography.Text color="secondary">—</Typography.Text>;
  }
  if (isLoading) {
    return <Spinner size="small" />;
  }
  if (error) {
    // Only platform admins reach this branch; surface real fetch failures.
    return (
      <Typography.Text color="error" size="sm">
        Failed to load
      </Typography.Text>
    );
  }
  const roles = data?.roles ?? [];
  if (roles.length === 0) {
    return <Typography.Text color="secondary">—</Typography.Text>;
  }
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs / 2 }}>
      {roles.map((role) => (
        <Typography.Text key={role.id} size="sm">
          <code>{role.workspace}</code> → {role.name}
        </Typography.Text>
      ))}
    </div>
  );
};

const UsersTab = () => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const { data: usersData, isLoading, error: queryError } = useUsersQuery();
  const { data: currentUserData } = useCurrentUserQuery();
  const currentUsername = currentUserData?.user?.username;
  const deleteUser = useDeleteUser();
  const withReturnTo = useWithSettingsReturnTo();
  // Bulk-delete + row checkboxes are platform-admin-only; Create User is
  // open to workspace admins.
  const isAdmin = useCurrentUserIsAdmin();
  const isWorkspaceAdmin = useCurrentUserIsWorkspaceAdmin();
  const canCreateUser = isAdmin || isWorkspaceAdmin;

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
              <UserRolesCell username={user.username} enabled={isAdmin} />
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
  // Workspace admins must scope to a workspace they manage
  // (``validate_can_list_roles`` rejects unscoped non-admin requests).
  // Bulk-delete is open to both groups — the listing is already
  // server-scoped, so every visible row is one ``validate_can_manage_roles``
  // will accept a delete for.
  const isAdmin = useCurrentUserIsAdmin();
  const isWorkspaceAdmin = useCurrentUserIsWorkspaceAdmin();
  const canManageRoles = isAdmin || isWorkspaceAdmin;
  const activeWorkspace = useActiveWorkspace();
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
            defaultMessage="Pick a workspace from the workspace selector to see roles you can manage."
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

  const isAdmin = useCurrentUserIsAdmin();
  const activeWorkspace = useActiveWorkspace();

  // The route definition's static ``getPageTitle`` is set by ``MlflowRootRoute``
  // *after* this component's effects (parent effects run after children's), so
  // we override on a microtask to land last and reflect the workspace-manager
  // header in the browser tab.
  useEffect(() => {
    const desired = isAdmin ? 'Platform Admin - MLflow' : 'Workspace Manager - MLflow';
    queueMicrotask(() => {
      document.title = desired;
    });
  }, [isAdmin]);

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
              {isAdmin ? (
                <FormattedMessage defaultMessage="Platform Admin" description="Admin page title for platform admins" />
              ) : (
                <FormattedMessage
                  defaultMessage="Workspace Manager"
                  description="Admin page title for non-platform-admins (workspace admins)"
                />
              )}
            </Typography.Title>
          </div>
          {!isAdmin && activeWorkspace && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Workspace: {workspace}"
                description="Subtitle on the admin page identifying the active workspace for workspace managers"
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
