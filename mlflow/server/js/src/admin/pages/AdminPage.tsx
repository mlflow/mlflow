import { useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Checkbox,
  Empty,
  Input,
  Modal,
  PlusIcon,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Switch,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tabs,
  Tag,
  TrashIcon,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { Link, useSearchParams } from '../../common/utils/RoutingUtils';
import { useWorkspaces } from '../../workspaces/hooks/useWorkspaces';
import { useWorkspacesEnabled } from '../../experiment-tracking/hooks/useServerInfo';
import { performLogout } from '../auth-utils';
import AdminRoutes from '../routes';
import {
  useUsersQuery,
  useCreateUser,
  useDeleteUser,
  useUpdateAdmin,
  useRolesQuery,
  useCreateRole,
  useDeleteRole,
  useUserRolesQuery,
  useCurrentUserQuery,
  useWithSettingsReturnTo,
} from '../hooks';
import type { CreateRoleRequest } from '../types';
import { isWorkspaceAdminRole } from '../types';

// Renders one line per role assigned to a user, formatted as
// `<workspace> → <role_name>`. Mirrors the `<scope> → <value>` shape used
// elsewhere in the admin UI (e.g. the Account page's permission lines like
// `experiment:* → READ`). Each row issues its own request — React Query
// caches per-username so subsequent re-renders don't re-fetch.
const UserRolesCell = ({ username }: { username: string }) => {
  const { theme } = useDesignSystemTheme();
  const { data, isLoading, error } = useUserRolesQuery(username);
  if (isLoading) {
    return <Spinner size="small" />;
  }
  if (error) {
    // Distinguish "no roles assigned" from "we couldn't load roles" — failing
    // silently to "—" hides 403/500s from admins reviewing user state.
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
  const currentUsername = currentUserData?.user?.username ?? '';
  const createUser = useCreateUser();
  const deleteUser = useDeleteUser();
  const updateAdmin = useUpdateAdmin();
  const withReturnTo = useWithSettingsReturnTo();

  // Deleting your own account is allowed (an admin removing their own
  // account before someone else takes over is a real flow), but it has the
  // side effect of logging you out — surface that explicitly in the
  // confirmation, and follow it through after the deletion succeeds so
  // the browser doesn't sit on a now-broken auth state.
  const logoutAfterSelfDelete = () => performLogout(queryClient);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [selectedUsernames, setSelectedUsernames] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState<string | null>(null);

  const users = useMemo(() => usersData?.users ?? [], [usersData]);
  // Drop selections that no longer exist after a refetch / single-row delete
  // — leaving stale entries would inflate the bulk-delete count and make us
  // attempt to delete users that are already gone (yielding 404s).
  const visibleSelectedUsernames = useMemo(() => {
    const usernames = new Set(users.map((u) => u.username));
    return new Set(Array.from(selectedUsernames).filter((u) => usernames.has(u)));
  }, [selectedUsernames, users]);
  const allSelected = users.length > 0 && users.every((u) => visibleSelectedUsernames.has(u.username));

  const toggleUserSelection = (username: string) => {
    setSelectedUsernames((prev) => {
      const next = new Set(prev);
      if (next.has(username)) {
        next.delete(username);
      } else {
        next.add(username);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    // Use ``allSelected`` (not raw ``prev.size``) — ``selectedUsernames`` may
    // hold stale entries that we deliberately keep but filter out via
    // ``visibleSelectedUsernames``. After a refetch/single-row delete,
    // ``allSelected`` can be true while ``prev.size !== users.length``, so
    // comparing to raw size would fail to clear the selection on click.
    setSelectedUsernames(allSelected ? new Set() : new Set(users.map((u) => u.username)));
  };

  const handleBulkDelete = async () => {
    setError(null);
    // Use the reconciled set so we don't try to delete users who already
    // disappeared after a refetch / single-row delete.
    const targets = Array.from(visibleSelectedUsernames);
    const includesSelf = visibleSelectedUsernames.has(currentUsername);
    const results = await Promise.allSettled(targets.map((u) => deleteUser.mutateAsync(u)));
    const failures = results.filter((r) => r.status === 'rejected') as PromiseRejectedResult[];
    if (failures.length > 0) {
      setError(`Failed to delete ${failures.length}/${targets.length} users: ${failures[0].reason?.message ?? ''}`);
    }
    setSelectedUsernames(new Set());
    setBulkDeleteOpen(false);
    // Only log out if the self-delete actually succeeded (not in the
    // failures list). If every delete failed, the user is still
    // authenticated — leave them on the page.
    if (includesSelf) {
      const selfResult = results[targets.indexOf(currentUsername)];
      if (selfResult?.status === 'fulfilled') {
        logoutAfterSelfDelete();
      }
    }
  };

  const handleCreateUser = async () => {
    setError(null);
    const trimmedUsername = newUsername.trim();
    if (!trimmedUsername || !newPassword) {
      setError('Username and password are required');
      return;
    }
    try {
      await createUser.mutateAsync({ username: trimmedUsername, password: newPassword });
      setShowCreateModal(false);
      setNewUsername('');
      setNewPassword('');
    } catch (e: any) {
      setError(e.message || 'Failed to create user');
    }
  };

  const handleDeleteUser = async () => {
    if (!deleteTarget) return;
    setError(null);
    const isSelf = deleteTarget === currentUsername;
    try {
      await deleteUser.mutateAsync(deleteTarget);
      setDeleteTarget(null);
      if (isSelf) {
        logoutAfterSelfDelete();
      }
    } catch (e: any) {
      setError(e.message || 'Failed to delete user');
    }
  };

  const handleToggleAdmin = async (username: string, currentIsAdmin: boolean) => {
    setError(null);
    try {
      await updateAdmin.mutateAsync({ username, is_admin: !currentIsAdmin });
    } catch (e: any) {
      setError(e.message || 'Failed to update admin status');
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
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          {visibleSelectedUsernames.size > 0 && (
            <Button componentId="admin.users.bulk_delete_button" danger onClick={() => setBulkDeleteOpen(true)}>
              <FormattedMessage
                defaultMessage="Delete ({count})"
                description="Bulk-delete button on the users table"
                values={{ count: visibleSelectedUsernames.size }}
              />
            </Button>
          )}
        </div>
        <Button
          componentId="admin.users.create_button"
          type="primary"
          icon={<PlusIcon />}
          onClick={() => setShowCreateModal(true)}
        >
          <FormattedMessage defaultMessage="Create User" description="Button to create a new user" />
        </Button>
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
          <TableHeader componentId="admin.users.select_header" css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
            <Checkbox
              componentId="admin.users.select_all"
              isChecked={allSelected}
              onChange={toggleSelectAll}
              aria-label="Select all users"
            />
          </TableHeader>
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
          <TableHeader componentId="admin.users.actions_header" css={{ flex: 0, minWidth: 80, maxWidth: 80 }}>
            <FormattedMessage defaultMessage="Actions" description="Users table actions header" />
          </TableHeader>
        </TableRow>
        {users.map((user) => (
          <TableRow key={user.username}>
            <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
              <Checkbox
                componentId="admin.users.select_row"
                isChecked={visibleSelectedUsernames.has(user.username)}
                onChange={() => toggleUserSelection(user.username)}
                aria-label={`Select user ${user.username}`}
              />
            </TableCell>
            <TableCell css={{ flex: 2 }}>
              <Link
                componentId="admin.users.username_link"
                to={withReturnTo(AdminRoutes.getUserPermissionsRoute(user.username))}
              >
                {user.username}
              </Link>
            </TableCell>
            <TableCell css={{ flex: 2 }}>
              <UserRolesCell username={user.username} />
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Switch
                componentId="admin.users.toggle_admin"
                checked={user.is_admin}
                onChange={() => handleToggleAdmin(user.username, user.is_admin)}
                label=""
                aria-label={`Toggle admin for ${user.username}`}
              />
            </TableCell>
            <TableCell css={{ flex: 0, minWidth: 80, maxWidth: 80 }}>
              <Button
                componentId="admin.users.delete_button"
                type="tertiary"
                size="small"
                icon={<TrashIcon />}
                aria-label={`Delete user ${user.username}`}
                onClick={() => setDeleteTarget(user.username)}
                danger
              />
            </TableCell>
          </TableRow>
        ))}
      </Table>
      <Modal
        componentId="admin.users.create_modal"
        title="Create User"
        visible={showCreateModal}
        onCancel={() => {
          setShowCreateModal(false);
          setNewUsername('');
          setNewPassword('');
          setError(null);
        }}
        onOk={handleCreateUser}
        okText="Create"
        confirmLoading={createUser.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="admin.users.create_modal.error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
          <div>
            <Typography.Text bold>Username</Typography.Text>
            <Input
              componentId="admin.users.create_username"
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
              placeholder="Enter username"
            />
          </div>
          <div>
            <Typography.Text bold>Password</Typography.Text>
            <Input
              componentId="admin.users.create_password"
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Enter password"
            />
          </div>
        </div>
      </Modal>
      <Modal
        componentId="admin.users.delete_modal"
        title="Delete User"
        visible={Boolean(deleteTarget)}
        onCancel={() => {
          setDeleteTarget(null);
          setError(null);
        }}
        onOk={handleDeleteUser}
        okText="Delete"
        okButtonProps={{ danger: true }}
        confirmLoading={deleteUser.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="admin.users.delete_modal.error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
          <Typography.Text>
            Are you sure you want to delete user <strong>{deleteTarget}</strong>? This action cannot be undone.
          </Typography.Text>
          {deleteTarget === currentUsername && (
            <Alert
              componentId="admin.users.delete_self_warning"
              type="warning"
              message="This is your own account."
              description="You'll be logged out immediately after the deletion completes."
              closable={false}
            />
          )}
        </div>
      </Modal>
      <Modal
        componentId="admin.users.bulk_delete_modal"
        title="Delete users"
        visible={bulkDeleteOpen}
        onCancel={() => setBulkDeleteOpen(false)}
        onOk={handleBulkDelete}
        okText="Delete"
        okButtonProps={{ danger: true }}
        confirmLoading={deleteUser.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <Typography.Text>
            Delete {visibleSelectedUsernames.size} user{visibleSelectedUsernames.size === 1 ? '' : 's'}? This action
            cannot be undone.
          </Typography.Text>
          {visibleSelectedUsernames.has(currentUsername) && (
            <Alert
              componentId="admin.users.bulk_delete_self_warning"
              type="warning"
              message="Your own account is in this selection."
              description="You'll be logged out immediately after the deletion completes."
              closable={false}
            />
          )}
        </div>
      </Modal>
    </div>
  );
};

const RolesTab = () => {
  const { theme } = useDesignSystemTheme();
  const { data: rolesData, isLoading, error: queryError } = useRolesQuery();
  const createRole = useCreateRole();
  const deleteRole = useDeleteRole();
  const { workspacesEnabled } = useWorkspacesEnabled();
  const { workspaces } = useWorkspaces(workspacesEnabled);
  const withReturnTo = useWithSettingsReturnTo();

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newRoleName, setNewRoleName] = useState('');
  const [newRoleDescription, setNewRoleDescription] = useState('');
  const [newRoleWorkspace, setNewRoleWorkspace] = useState('default');
  const [deleteTarget, setDeleteTarget] = useState<{ id: number; name: string } | null>(null);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [selectedRoleIds, setSelectedRoleIds] = useState<Set<number>>(() => new Set());
  const [error, setError] = useState<string | null>(null);

  const roles = useMemo(() => rolesData?.roles ?? [], [rolesData]);
  // Drop selections that no longer exist after a refetch / single-row delete
  // — leaving stale entries would inflate the bulk-delete count and make us
  // attempt to delete roles that are already gone.
  const visibleSelectedRoleIds = useMemo(() => {
    const ids = new Set(roles.map((r) => r.id));
    return new Set(Array.from(selectedRoleIds).filter((id) => ids.has(id)));
  }, [selectedRoleIds, roles]);
  const allSelected = roles.length > 0 && roles.every((r) => visibleSelectedRoleIds.has(r.id));

  const toggleRoleSelection = (roleId: number) => {
    setSelectedRoleIds((prev) => {
      const next = new Set(prev);
      if (next.has(roleId)) {
        next.delete(roleId);
      } else {
        next.add(roleId);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    // See UsersTab for why this uses ``allSelected`` instead of raw
    // ``prev.size`` — stale IDs in ``selectedRoleIds`` would otherwise
    // prevent the header checkbox from clearing the selection.
    setSelectedRoleIds(allSelected ? new Set() : new Set(roles.map((r) => r.id)));
  };

  const handleBulkDelete = async () => {
    setError(null);
    // Use the reconciled set so we don't try to delete roles that already
    // disappeared after a refetch / single-row delete.
    const targets = Array.from(visibleSelectedRoleIds);
    const results = await Promise.allSettled(targets.map((id) => deleteRole.mutateAsync(id)));
    const failures = results.filter((r) => r.status === 'rejected') as PromiseRejectedResult[];
    if (failures.length > 0) {
      setError(`Failed to delete ${failures.length}/${targets.length} roles: ${failures[0].reason?.message ?? ''}`);
    }
    setSelectedRoleIds(new Set());
    setBulkDeleteOpen(false);
  };
  // Always include "default" — useWorkspaces() returns whatever the workspace
  // store lists, which may exclude the reserved default workspace (and is
  // empty entirely when workspaces are disabled, see useWorkspaces(false)).
  const workspaceOptions = useMemo(() => {
    const names = new Set<string>(['default']);
    for (const w of workspaces) names.add(w.name);
    // Sort so the dropdown order is deterministic; `Set` iteration order
    // depends on insertion, so without this the dropdown would shift if
    // ``useWorkspaces()`` returns a different order across renders.
    // Pin "default" to the top — it's always present and is the
    // reserved/most-common pick.
    return [
      'default',
      ...Array.from(names)
        .filter((n) => n !== 'default')
        .sort(),
    ];
  }, [workspaces]);

  const handleCreateRole = async () => {
    setError(null);
    // Trim + require role name client-side. The backend's ``name.strip()``
    // check would otherwise let leading/trailing whitespace persist into the
    // DB and propagate to every UI surface that displays the role.
    const trimmedName = newRoleName.trim();
    if (!trimmedName) {
      setError('Role name cannot be empty');
      return;
    }
    try {
      const request: CreateRoleRequest = {
        name: trimmedName,
        workspace: newRoleWorkspace,
        description: newRoleDescription || undefined,
      };
      await createRole.mutateAsync(request);
      setShowCreateModal(false);
      setNewRoleName('');
      setNewRoleDescription('');
      setNewRoleWorkspace('default');
    } catch (e: any) {
      setError(e.message || 'Failed to create role');
    }
  };

  const handleDeleteRole = async () => {
    if (!deleteTarget) return;
    setError(null);
    try {
      await deleteRole.mutateAsync(deleteTarget.id);
      setDeleteTarget(null);
    } catch (e: any) {
      setError(e.message || 'Failed to delete role');
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
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          {visibleSelectedRoleIds.size > 0 && (
            <Button componentId="admin.roles.bulk_delete_button" danger onClick={() => setBulkDeleteOpen(true)}>
              <FormattedMessage
                defaultMessage="Delete ({count})"
                description="Bulk-delete button on the roles table"
                values={{ count: visibleSelectedRoleIds.size }}
              />
            </Button>
          )}
        </div>
        <Button
          componentId="admin.roles.create_button"
          type="primary"
          icon={<PlusIcon />}
          onClick={() => setShowCreateModal(true)}
        >
          <FormattedMessage defaultMessage="Create Role" description="Button to create a new role" />
        </Button>
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
          <TableHeader componentId="admin.roles.select_header" css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
            <Checkbox
              componentId="admin.roles.select_all"
              isChecked={allSelected}
              onChange={toggleSelectAll}
              aria-label="Select all roles"
            />
          </TableHeader>
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
          <TableHeader componentId="admin.roles.actions_header" css={{ flex: 0, minWidth: 80, maxWidth: 80 }}>
            <FormattedMessage defaultMessage="Actions" description="Roles table actions header" />
          </TableHeader>
        </TableRow>
        {roles.map((role) => (
          <TableRow key={role.id}>
            <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
              <Checkbox
                componentId="admin.roles.select_row"
                isChecked={visibleSelectedRoleIds.has(role.id)}
                onChange={() => toggleRoleSelection(role.id)}
                aria-label={`Select role ${role.name}`}
              />
            </TableCell>
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
            <TableCell css={{ flex: 0, minWidth: 80, maxWidth: 80 }}>
              <Button
                componentId="admin.roles.delete_button"
                type="tertiary"
                size="small"
                icon={<TrashIcon />}
                aria-label={`Delete role ${role.name}`}
                onClick={() => setDeleteTarget({ id: role.id, name: role.name })}
                danger
              />
            </TableCell>
          </TableRow>
        ))}
      </Table>
      <Modal
        componentId="admin.roles.create_modal"
        title="Create Role"
        visible={showCreateModal}
        onCancel={() => {
          setShowCreateModal(false);
          setNewRoleName('');
          setNewRoleDescription('');
          setNewRoleWorkspace('default');
          setError(null);
        }}
        onOk={handleCreateRole}
        okText="Create"
        confirmLoading={createRole.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="admin.roles.create_modal_error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
          <div>
            <Typography.Text bold>Name</Typography.Text>
            <Input
              componentId="admin.roles.create_name"
              value={newRoleName}
              onChange={(e) => setNewRoleName(e.target.value)}
              placeholder="Enter role name"
            />
          </div>
          <div>
            <Typography.Text bold>Workspace</Typography.Text>
            <SimpleSelect
              id="admin-roles-create-workspace"
              componentId="admin.roles.create_workspace"
              value={newRoleWorkspace}
              onChange={({ target }) => setNewRoleWorkspace(target.value)}
            >
              {workspaceOptions.map((name) => (
                <SimpleSelectOption key={name} value={name}>
                  {name}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
          <div>
            <Typography.Text bold>Description</Typography.Text>
            <Input
              componentId="admin.roles.create_description"
              value={newRoleDescription}
              onChange={(e) => setNewRoleDescription(e.target.value)}
              placeholder="Enter description (optional)"
            />
          </div>
          <Typography.Paragraph css={{ color: theme.colors.textSecondary, marginTop: theme.spacing.sm }}>
            To make this a workspace admin role, add a permission with resource type <strong>workspace</strong>,
            resource pattern <strong>*</strong>, and permission <strong>MANAGE</strong> after creation.
          </Typography.Paragraph>
        </div>
      </Modal>
      <Modal
        componentId="admin.roles.delete_modal"
        title="Delete Role"
        visible={Boolean(deleteTarget)}
        onCancel={() => {
          setDeleteTarget(null);
          setError(null);
        }}
        onOk={handleDeleteRole}
        okText="Delete"
        okButtonProps={{ danger: true }}
        confirmLoading={deleteRole.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="admin.roles.delete_modal_error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
          <Typography.Text>
            Are you sure you want to delete role <strong>{deleteTarget?.name}</strong>? This action cannot be undone.
          </Typography.Text>
        </div>
      </Modal>
      <Modal
        componentId="admin.roles.bulk_delete_modal"
        title="Delete roles"
        visible={bulkDeleteOpen}
        onCancel={() => setBulkDeleteOpen(false)}
        onOk={handleBulkDelete}
        okText="Delete"
        okButtonProps={{ danger: true }}
        confirmLoading={deleteRole.isLoading}
      >
        <Typography.Text>
          Delete {visibleSelectedRoleIds.size} role{visibleSelectedRoleIds.size === 1 ? '' : 's'}? This action cannot be
          undone.
        </Typography.Text>
      </Modal>
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

  return (
    <ScrollablePageWrapper>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
        <div css={{ padding: theme.spacing.md, paddingBottom: 0 }}>
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.xs,
              marginBottom: theme.spacing.md,
            }}
          >
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
                <FormattedMessage defaultMessage="Platform Admin" description="Admin page title" />
              </Typography.Title>
            </div>
          </div>
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
          css={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
        >
          <div css={{ paddingLeft: theme.spacing.md, paddingRight: theme.spacing.md }}>
            <Tabs.List>
              <Tabs.Trigger value="users">
                <FormattedMessage defaultMessage="Users" description="Admin users tab" />
              </Tabs.Trigger>
              <Tabs.Trigger value="roles">
                <FormattedMessage defaultMessage="Roles" description="Admin roles tab" />
              </Tabs.Trigger>
            </Tabs.List>
          </div>
          <Tabs.Content
            value="users"
            css={{
              flex: 1,
              overflow: 'auto',
              padding: theme.spacing.md,
              paddingTop: theme.spacing.md,
            }}
          >
            <UsersTab />
          </Tabs.Content>
          <Tabs.Content
            value="roles"
            css={{
              flex: 1,
              overflow: 'auto',
              padding: theme.spacing.md,
              paddingTop: theme.spacing.md,
            }}
          >
            <RolesTab />
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </ScrollablePageWrapper>
  );
};

export default AdminPage;
