import { useMemo, useState } from 'react';
import {
  Button,
  Input,
  Modal,
  Spinner,
  Typography,
  useDesignSystemTheme,
  PlusIcon,
  TrashIcon,
  LegacyTable,
  Tag,
  Alert,
  Breadcrumb,
  DialogCombobox,
  DialogComboboxTrigger,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
} from '@databricks/design-system';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { Link, useParams } from '../../common/utils/RoutingUtils';
import AdminRoutes from '../routes';
import {
  useRoleDetailQuery,
  useUpdateRole,
  useAddPermission,
  useRemovePermission,
  useUpdatePermission,
  useRoleUsersQuery,
  useAssignRole,
  useUnassignRole,
  useUsersQuery,
} from '../hooks';
import type { RolePermission, UserRoleAssignment } from '../types';
import { RESOURCE_TYPES, PERMISSIONS } from '../types';

const PermissionsSection = ({ roleId }: { roleId: number }) => {
  const { theme } = useDesignSystemTheme();
  const { data: roleData } = useRoleDetailQuery(roleId);
  const addPermission = useAddPermission(roleId);
  const removePermission = useRemovePermission(roleId);
  const updatePermission = useUpdatePermission(roleId);

  const [showAddModal, setShowAddModal] = useState(false);
  const [newResourceType, setNewResourceType] = useState<string>(RESOURCE_TYPES[0]);
  const [newResourcePattern, setNewResourcePattern] = useState('*');
  const [newPermission, setNewPermission] = useState<string>(PERMISSIONS[0]);
  const [editingPermission, setEditingPermission] = useState<{ id: number; permission: string } | null>(null);
  const [deletePermissionTarget, setDeletePermissionTarget] = useState<RolePermission | null>(null);
  const [error, setError] = useState<string | null>(null);

  const permissions = useMemo(() => roleData?.role?.permissions ?? [], [roleData]);

  const handleAddPermission = async () => {
    setError(null);
    try {
      await addPermission.mutateAsync({
        role_id: roleId,
        resource_type: newResourceType,
        resource_pattern: newResourcePattern,
        permission: newPermission,
      });
      setShowAddModal(false);
      setNewResourceType(RESOURCE_TYPES[0]);
      setNewResourcePattern('*');
      setNewPermission(PERMISSIONS[0]);
    } catch (e: any) {
      setError(e.message || 'Failed to add permission');
    }
  };

  const handleRemovePermission = async () => {
    if (!deletePermissionTarget) return;
    setError(null);
    try {
      await removePermission.mutateAsync(deletePermissionTarget.id);
      setDeletePermissionTarget(null);
    } catch (e: any) {
      setError(e.message || 'Failed to remove permission');
    }
  };

  const handleUpdatePermission = async () => {
    if (!editingPermission) return;
    setError(null);
    try {
      await updatePermission.mutateAsync({
        role_permission_id: editingPermission.id,
        permission: editingPermission.permission,
      });
      setEditingPermission(null);
    } catch (e: any) {
      setError(e.message || 'Failed to update permission');
    }
  };

  const columns = [
    {
      title: 'Resource Type',
      dataIndex: 'resource_type',
      key: 'resource_type',
      render: (text: string) => (
        <Tag componentId="admin.role.permission_resource_type_tag">{text}</Tag>
      ),
    },
    {
      title: 'Resource Pattern',
      dataIndex: 'resource_pattern',
      key: 'resource_pattern',
      render: (text: string) => <code>{text}</code>,
    },
    {
      title: 'Permission',
      dataIndex: 'permission',
      key: 'permission',
      render: (text: string) => (
        <Tag componentId="admin.role.permission_level_tag" color="indigo">
          {text}
        </Tag>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: RolePermission) => (
        <div css={{ display: 'flex', gap: theme.spacing.xs }}>
          <Button
            componentId="admin.role.edit_permission_button"
            type="tertiary"
            size="small"
            onClick={() => setEditingPermission({ id: record.id, permission: record.permission })}
          >
            Edit
          </Button>
          <Button
            componentId="admin.role.remove_permission_button"
            type="tertiary"
            icon={<TrashIcon />}
            size="small"
            onClick={() => setDeletePermissionTarget(record)}
            danger
          />
        </div>
      ),
    },
  ];

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Title level={4}>Permissions</Typography.Title>
        <Button
          componentId="admin.role.add_permission_button"
          type="primary"
          icon={<PlusIcon />}
          onClick={() => setShowAddModal(true)}
        >
          Add Permission
        </Button>
      </div>
      {error && (
        <Alert
          componentId="admin.role.permissions.error"
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
        />
      )}
      <LegacyTable dataSource={permissions} columns={columns} rowKey="id" pagination={false} />
      <Modal
        componentId="admin.role.add_permission_modal"
        title="Add Permission"
        visible={showAddModal}
        onCancel={() => setShowAddModal(false)}
        onOk={handleAddPermission}
        okText="Add"
        confirmLoading={addPermission.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div>
            <Typography.Text bold>Resource Type</Typography.Text>
            <DialogCombobox
              componentId="admin.role.add_permission_resource_type"
              label="Resource Type"
              value={[newResourceType]}
            >
              <DialogComboboxTrigger />
              <DialogComboboxContent>
                <DialogComboboxOptionList>
                  {RESOURCE_TYPES.map((rt) => (
                    <DialogComboboxOptionListSelectItem
                      key={rt}
                      value={rt}
                      checked={newResourceType === rt}
                      onChange={() => setNewResourceType(rt)}
                    >
                      {rt}
                    </DialogComboboxOptionListSelectItem>
                  ))}
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          </div>
          <div>
            <Typography.Text bold>Resource Pattern</Typography.Text>
            <Input
              componentId="admin.role.add_permission_pattern"
              value={newResourcePattern}
              onChange={(e) => setNewResourcePattern(e.target.value)}
              placeholder='Specific ID or "*" for all'
            />
          </div>
          <div>
            <Typography.Text bold>Permission</Typography.Text>
            <DialogCombobox
              componentId="admin.role.add_permission_level"
              label="Permission"
              value={[newPermission]}
            >
              <DialogComboboxTrigger />
              <DialogComboboxContent>
                <DialogComboboxOptionList>
                  {PERMISSIONS.map((p) => (
                    <DialogComboboxOptionListSelectItem
                      key={p}
                      value={p}
                      checked={newPermission === p}
                      onChange={() => setNewPermission(p)}
                    >
                      {p}
                    </DialogComboboxOptionListSelectItem>
                  ))}
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          </div>
        </div>
      </Modal>
      <Modal
        componentId="admin.role.edit_permission_modal"
        title="Edit Permission"
        visible={Boolean(editingPermission)}
        onCancel={() => setEditingPermission(null)}
        onOk={handleUpdatePermission}
        okText="Save"
        confirmLoading={updatePermission.isLoading}
      >
        {editingPermission && (
          <div>
            <Typography.Text bold>Permission Level</Typography.Text>
            <DialogCombobox
              componentId="admin.role.edit_permission_level"
              label="Permission"
              value={[editingPermission.permission]}
            >
              <DialogComboboxTrigger />
              <DialogComboboxContent>
                <DialogComboboxOptionList>
                  {PERMISSIONS.map((p) => (
                    <DialogComboboxOptionListSelectItem
                      key={p}
                      value={p}
                      checked={editingPermission.permission === p}
                      onChange={() => setEditingPermission({ ...editingPermission, permission: p })}
                    >
                      {p}
                    </DialogComboboxOptionListSelectItem>
                  ))}
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          </div>
        )}
      </Modal>
      <Modal
        componentId="admin.role.remove_permission_modal"
        title="Remove Permission"
        visible={Boolean(deletePermissionTarget)}
        onCancel={() => setDeletePermissionTarget(null)}
        onOk={handleRemovePermission}
        okText="Remove"
        okButtonProps={{ danger: true }}
        confirmLoading={removePermission.isLoading}
      >
        <Typography.Text>
          Are you sure you want to remove the <strong>{deletePermissionTarget?.permission}</strong> permission on{' '}
          <strong>{deletePermissionTarget?.resource_type}</strong> ({deletePermissionTarget?.resource_pattern})? This
          action cannot be undone.
        </Typography.Text>
      </Modal>
    </div>
  );
};

const AssignedUsersSection = ({ roleId }: { roleId: number }) => {
  const { theme } = useDesignSystemTheme();
  const { data: assignmentsData, isLoading, error: queryError } = useRoleUsersQuery(roleId);
  const { data: usersData, error: usersError } = useUsersQuery();
  const assignRole = useAssignRole(roleId);
  const unassignRole = useUnassignRole(roleId);

  const [showAssignModal, setShowAssignModal] = useState(false);
  const [selectedUsername, setSelectedUsername] = useState('');
  const [error, setError] = useState<string | null>(null);

  const assignments = useMemo(() => assignmentsData?.assignments ?? [], [assignmentsData]);
  const users = useMemo(() => usersData?.users ?? [], [usersData]);

  const handleAssign = async () => {
    if (!selectedUsername) return;
    setError(null);
    try {
      await assignRole.mutateAsync(selectedUsername);
      setShowAssignModal(false);
      setSelectedUsername('');
    } catch (e: any) {
      setError(e.message || 'Failed to assign role');
    }
  };

  const handleUnassign = async (username: string) => {
    setError(null);
    try {
      await unassignRole.mutateAsync(username);
    } catch (e: any) {
      setError(e.message || 'Failed to unassign role');
    }
  };

  const userMap = useMemo(() => {
    const map = new Map<number, string>();
    for (const user of users) {
      map.set(user.id, user.username);
    }
    return map;
  }, [users]);

  const columns = [
    {
      title: 'User ID',
      dataIndex: 'user_id',
      key: 'user_id',
    },
    {
      title: 'Username',
      key: 'username',
      render: (_: unknown, record: UserRoleAssignment) => userMap.get(record.user_id) ?? 'Unknown',
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: unknown, record: UserRoleAssignment) => {
        const username = userMap.get(record.user_id);
        return username ? (
          <Button
            componentId="admin.role.unassign_button"
            type="tertiary"
            icon={<TrashIcon />}
            size="small"
            onClick={() => handleUnassign(username)}
            danger
          />
        ) : null;
      },
    },
  ];

  if (isLoading) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
        <Spinner size="small" />
      </div>
    );
  }

  const fetchError = queryError || usersError;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Title level={4}>Assigned Users</Typography.Title>
        <Button
          componentId="admin.role.assign_user_button"
          type="primary"
          icon={<PlusIcon />}
          onClick={() => setShowAssignModal(true)}
        >
          Assign User
        </Button>
      </div>
      {fetchError && (
        <Alert
          componentId="admin.role.assignments.fetch_error"
          type="error"
          message="Failed to load assigned users"
          description={(fetchError as Error)?.message || 'An error occurred while fetching data.'}
        />
      )}
      {error && (
        <Alert
          componentId="admin.role.assignments.error"
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
        />
      )}
      <LegacyTable dataSource={assignments} columns={columns} rowKey="id" pagination={false} />
      <Modal
        componentId="admin.role.assign_user_modal"
        title="Assign User to Role"
        visible={showAssignModal}
        onCancel={() => {
          setShowAssignModal(false);
          setSelectedUsername('');
        }}
        onOk={handleAssign}
        okText="Assign"
        confirmLoading={assignRole.isLoading}
      >
        <div>
          <Typography.Text bold>Username</Typography.Text>
          <DialogCombobox
            componentId="admin.role.assign_user_select"
            label="Select user"
            value={selectedUsername ? [selectedUsername] : []}
          >
            <DialogComboboxTrigger />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                {users.map((user) => (
                  <DialogComboboxOptionListSelectItem
                    key={user.username}
                    value={user.username}
                    checked={selectedUsername === user.username}
                    onChange={() => setSelectedUsername(user.username)}
                  >
                    {user.username}
                  </DialogComboboxOptionListSelectItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </div>
      </Modal>
    </div>
  );
};

const RoleDetailPage = () => {
  const { theme } = useDesignSystemTheme();
  const { roleId: roleIdParam } = useParams<{ roleId: string }>();
  const roleId = Number(roleIdParam);
  const isValidRoleId = Number.isFinite(roleId);

  const { data: roleData, isLoading, error: loadError } = useRoleDetailQuery(roleId);
  const updateRole = useUpdateRole(roleId);
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [error, setError] = useState<string | null>(null);

  const role = roleData?.role;

  const handleStartEditing = () => {
    if (!role) return;
    setEditName(role.name);
    setEditDescription(role.description || '');
    setIsEditing(true);
  };

  const handleSaveEdit = async () => {
    setError(null);
    try {
      await updateRole.mutateAsync({
        role_id: roleId,
        name: editName || undefined,
        description: editDescription || undefined,
      });
      setIsEditing(false);
    } catch (e: any) {
      setError(e.message || 'Failed to update role');
    }
  };

  if (!isValidRoleId) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.lg }}>
          <Alert
            componentId="admin.role.invalid_id"
            type="error"
            message="Invalid role ID"
            description="The requested role could not be loaded because the URL contains an invalid role ID."
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (isLoading) {
    return (
      <ScrollablePageWrapper>
        <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
          <Spinner size="small" />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (loadError || !role) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.lg }}>
          <Alert
            componentId="admin.role.load_error"
            type="error"
            message={(loadError as Error)?.message || 'Role not found'}
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.lg, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <Breadcrumb includeTrailingCaret={false}>
          <Breadcrumb.Item>
            <Link componentId="admin.role.breadcrumb_admin" to={AdminRoutes.adminPageRoute}>
              Admin
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>{role.name}</Breadcrumb.Item>
        </Breadcrumb>

        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <Typography.Title level={2}>{role.name}</Typography.Title>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Typography.Text color="secondary">
                {role.description || 'No description'} | Workspace: {role.workspace}
              </Typography.Text>
              {role.is_workspace_admin && (
                <Tag componentId="admin.role.admin_tag" color="indigo">
                  Admin Role
                </Tag>
              )}
            </div>
          </div>
          <Button componentId="admin.role.edit_button" type="primary" onClick={handleStartEditing}>
            Edit Role
          </Button>
        </div>

        {error && (
          <Alert
            componentId="admin.role.error"
            type="error"
            message={error}
            closable
            onClose={() => setError(null)}
          />
        )}

        <Modal
          componentId="admin.role.edit_modal"
          title="Edit Role"
          visible={isEditing}
          onCancel={() => setIsEditing(false)}
          onOk={handleSaveEdit}
          okText="Save"
          confirmLoading={updateRole.isLoading}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <div>
              <Typography.Text bold>Name</Typography.Text>
              <Input
                componentId="admin.role.edit_name"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
              />
            </div>
            <div>
              <Typography.Text bold>Description</Typography.Text>
              <Input
                componentId="admin.role.edit_description"
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
              />
            </div>
          </div>
        </Modal>

        <PermissionsSection roleId={roleId} />
        <AssignedUsersSection roleId={roleId} />
      </div>
    </ScrollablePageWrapper>
  );
};

export default RoleDetailPage;
