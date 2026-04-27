import { useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Empty,
  Input,
  Modal,
  PlusIcon,
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
import { Link } from '../../common/utils/RoutingUtils';
import AdminRoutes from '../routes';
import {
  useUsersQuery,
  useCreateUser,
  useDeleteUser,
  useUpdateAdmin,
  useRolesQuery,
  useCreateRole,
  useDeleteRole,
} from '../hooks';
import type { CreateRoleRequest } from '../types';
import { isWorkspaceAdminRole } from '../types';

const UsersTab = () => {
  const { theme } = useDesignSystemTheme();
  const { data: usersData, isLoading, error: queryError } = useUsersQuery();
  const createUser = useCreateUser();
  const deleteUser = useDeleteUser();
  const updateAdmin = useUpdateAdmin();

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const users = useMemo(() => usersData?.users ?? [], [usersData]);

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
    try {
      await deleteUser.mutateAsync(deleteTarget);
      setDeleteTarget(null);
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
      <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
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
          <TableHeader componentId="admin.users.username_header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Username" description="Users table username header" />
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
            <TableCell css={{ flex: 2 }}>{user.username}</TableCell>
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

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newRoleName, setNewRoleName] = useState('');
  const [newRoleDescription, setNewRoleDescription] = useState('');
  const [newRoleWorkspace, setNewRoleWorkspace] = useState('default');
  const [deleteTarget, setDeleteTarget] = useState<{ id: number; name: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const roles = useMemo(() => rolesData?.roles ?? [], [rolesData]);

  const handleCreateRole = async () => {
    setError(null);
    try {
      const request: CreateRoleRequest = {
        name: newRoleName,
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
      <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
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
            <TableCell css={{ flex: 2 }}>
              <Link componentId="admin.roles.name_link" to={AdminRoutes.getRoleDetailRoute(role.id)}>
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
            <Input
              componentId="admin.roles.create_workspace"
              value={newRoleWorkspace}
              onChange={(e) => setNewRoleWorkspace(e.target.value)}
              placeholder="Enter workspace"
            />
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
        <Typography.Text>
          Are you sure you want to delete role <strong>{deleteTarget?.name}</strong>? This action cannot be undone.
        </Typography.Text>
      </Modal>
    </div>
  );
};

const AdminPage = () => {
  const { theme } = useDesignSystemTheme();

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
          defaultValue="users"
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
