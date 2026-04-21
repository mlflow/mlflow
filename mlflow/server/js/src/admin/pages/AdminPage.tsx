import { useMemo, useState } from 'react';
import {
  Button,
  Input,
  Modal,
  Spinner,
  Switch,
  Typography,
  useDesignSystemTheme,
  PlusIcon,
  TrashIcon,
  LegacyTable,
  Tag,
  Tabs,
  Alert,
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
import type { CreateRoleRequest, User, Role } from '../types';

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
    if (!newUsername.trim() || !newPassword) {
      setError('Username and password are required');
      return;
    }
    try {
      await createUser.mutateAsync({ username: newUsername, password: newPassword });
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
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
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

  const columns = [
    {
      title: 'Username',
      dataIndex: 'username',
      key: 'username',
    },
    {
      title: 'Admin',
      dataIndex: 'is_admin',
      key: 'is_admin',
      render: (_: unknown, record: User) => (
        <Switch
          componentId="admin.users.toggle_admin"
          checked={record.is_admin}
          onChange={() => handleToggleAdmin(record.username, record.is_admin)}
          label=""
          aria-label={`Toggle admin for ${record.username}`}
        />
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: User) => (
        <Button
          componentId="admin.users.delete_button"
          type="tertiary"
          icon={<TrashIcon />}
          onClick={() => setDeleteTarget(record.username)}
          danger
        />
      ),
    },
  ];

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
      <LegacyTable dataSource={users} columns={columns} rowKey="username" pagination={false} />
      <Modal
        componentId="admin.users.create_modal"
        title="Create User"
        visible={showCreateModal}
        onCancel={() => {
          setShowCreateModal(false);
          setNewUsername('');
          setNewPassword('');
        }}
        onOk={handleCreateUser}
        okText="Create"
        confirmLoading={createUser.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
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
        onCancel={() => setDeleteTarget(null)}
        onOk={handleDeleteUser}
        okText="Delete"
        okButtonProps={{ danger: true }}
        confirmLoading={deleteUser.isLoading}
      >
        <Typography.Text>
          Are you sure you want to delete user <strong>{deleteTarget}</strong>? This action cannot be undone.
        </Typography.Text>
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
  const [newRoleIsAdmin, setNewRoleIsAdmin] = useState(false);
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
        is_workspace_admin: newRoleIsAdmin || undefined,
      };
      await createRole.mutateAsync(request);
      setShowCreateModal(false);
      setNewRoleName('');
      setNewRoleDescription('');
      setNewRoleWorkspace('default');
      setNewRoleIsAdmin(false);
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
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
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

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (_: unknown, record: Role) => (
        <Link componentId="admin.roles.name_link" to={AdminRoutes.getRoleDetailRoute(record.id)}>
          {record.name}
        </Link>
      ),
    },
    {
      title: 'Workspace',
      dataIndex: 'workspace',
      key: 'workspace',
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      render: (text: string | null) => text || '-',
    },
    {
      title: 'Admin Role',
      dataIndex: 'is_workspace_admin',
      key: 'is_workspace_admin',
      render: (isAdmin: boolean) =>
        isAdmin ? (
          <Tag componentId="admin.roles.admin_tag" color="indigo">
            Admin
          </Tag>
        ) : null,
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: Role) => (
        <Button
          componentId="admin.roles.delete_button"
          type="tertiary"
          icon={<TrashIcon />}
          onClick={() => setDeleteTarget({ id: record.id, name: record.name })}
          danger
        />
      ),
    },
  ];

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
      <LegacyTable dataSource={roles} columns={columns} rowKey="id" pagination={false} />
      <Modal
        componentId="admin.roles.create_modal"
        title="Create Role"
        visible={showCreateModal}
        onCancel={() => {
          setShowCreateModal(false);
          setNewRoleName('');
          setNewRoleDescription('');
          setNewRoleWorkspace('default');
          setNewRoleIsAdmin(false);
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
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Switch
              componentId="admin.roles.create_is_admin"
              checked={newRoleIsAdmin}
              onChange={(checked) => setNewRoleIsAdmin(checked)}
              label="Workspace Admin"
            />
          </div>
        </div>
      </Modal>
      <Modal
        componentId="admin.roles.delete_modal"
        title="Delete Role"
        visible={Boolean(deleteTarget)}
        onCancel={() => setDeleteTarget(null)}
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
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Title level={2} css={{ marginBottom: theme.spacing.lg }}>
          <FormattedMessage defaultMessage="Platform Admin" description="Admin page title" />
        </Typography.Title>
        <Tabs.Root componentId="admin.tabs" valueHasNoPii defaultValue="users">
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
