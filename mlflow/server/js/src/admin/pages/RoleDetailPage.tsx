import { useMemo, useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Empty,
  Input,
  Modal,
  PlusIcon,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
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
  useWithSettingsReturnTo,
} from '../hooks';
import type { RolePermission } from '../types';
import { RESOURCE_TYPES, PERMISSIONS, isWorkspaceAdminRole } from '../types';

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

  const emptyState =
    permissions.length === 0 ? (
      <Empty title="No permissions" description="Add a permission to grant access to resources via this role." />
    ) : null;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Title level={4} withoutMargins>
          Permissions
        </Typography.Title>
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
          <TableHeader componentId="admin.role.perm_resource_type_header" css={{ flex: 1 }}>
            Resource Type
          </TableHeader>
          <TableHeader componentId="admin.role.perm_resource_pattern_header" css={{ flex: 1 }}>
            Resource Pattern
          </TableHeader>
          <TableHeader componentId="admin.role.perm_permission_header" css={{ flex: 1 }}>
            Permission
          </TableHeader>
          <TableHeader componentId="admin.role.perm_actions_header" css={{ flex: 0, minWidth: 140, maxWidth: 140 }}>
            Actions
          </TableHeader>
        </TableRow>
        {permissions.map((perm) => (
          <TableRow key={perm.id}>
            <TableCell css={{ flex: 1 }}>
              <Tag componentId="admin.role.permission_resource_type_tag">{perm.resource_type}</Tag>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <code>{perm.resource_pattern}</code>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Tag componentId="admin.role.permission_level_tag" color="indigo">
                {perm.permission}
              </Tag>
            </TableCell>
            <TableCell css={{ flex: 0, minWidth: 140, maxWidth: 140 }}>
              <div css={{ display: 'flex', gap: theme.spacing.xs }}>
                <Button
                  componentId="admin.role.edit_permission_button"
                  type="tertiary"
                  size="small"
                  onClick={() => setEditingPermission({ id: perm.id, permission: perm.permission })}
                >
                  Edit
                </Button>
                <Button
                  componentId="admin.role.remove_permission_button"
                  type="tertiary"
                  icon={<TrashIcon />}
                  size="small"
                  aria-label={`Remove ${perm.permission} permission on ${perm.resource_type} ${perm.resource_pattern}`}
                  onClick={() => setDeletePermissionTarget(perm)}
                  danger
                />
              </div>
            </TableCell>
          </TableRow>
        ))}
      </Table>
      <Modal
        componentId="admin.role.add_permission_modal"
        title="Add Permission"
        visible={showAddModal}
        onCancel={() => {
          setShowAddModal(false);
          setError(null);
        }}
        onOk={handleAddPermission}
        okText="Add"
        confirmLoading={addPermission.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="admin.role.add_permission_modal_error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
          <div>
            <Typography.Text bold>Resource Type</Typography.Text>
            <SimpleSelect
              id="admin-role-add-permission-resource-type"
              componentId="admin.role.add_permission_resource_type"
              value={newResourceType}
              onChange={({ target }) => {
                setNewResourceType(target.value);
                // Workspace-scope grants only support the "*" pattern.
                if (target.value === 'workspace') setNewResourcePattern('*');
              }}
            >
              {RESOURCE_TYPES.map((rt) => (
                <SimpleSelectOption key={rt} value={rt}>
                  {rt}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
          {newResourceType === 'workspace' ? (
            <div>
              <Typography.Text bold>Workspace</Typography.Text>
              <Typography.Text color="secondary">
                {roleData?.role?.workspace ?? 'default'}{' '}
                <Typography.Text color="secondary" size="sm">
                  (this grant applies to the role's workspace)
                </Typography.Text>
              </Typography.Text>
            </div>
          ) : (
            <div>
              <Typography.Text bold>Resource Pattern</Typography.Text>
              <Input
                componentId="admin.role.add_permission_pattern"
                value={newResourcePattern}
                onChange={(e) => setNewResourcePattern(e.target.value)}
                placeholder='Specific ID or "*" for all'
              />
            </div>
          )}
          <div>
            <Typography.Text bold>Permission</Typography.Text>
            <SimpleSelect
              id="admin-role-add-permission-level"
              componentId="admin.role.add_permission_level"
              value={newPermission}
              onChange={({ target }) => setNewPermission(target.value)}
            >
              {PERMISSIONS.map((p) => (
                <SimpleSelectOption key={p} value={p}>
                  {p}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
        </div>
      </Modal>
      <Modal
        componentId="admin.role.edit_permission_modal"
        title="Edit Permission"
        visible={Boolean(editingPermission)}
        onCancel={() => {
          setEditingPermission(null);
          setError(null);
        }}
        onOk={handleUpdatePermission}
        okText="Save"
        confirmLoading={updatePermission.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="admin.role.edit_permission_modal_error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
          {editingPermission && (
            <div>
              <Typography.Text bold>Permission Level</Typography.Text>
              <SimpleSelect
                id="admin-role-edit-permission-level"
                componentId="admin.role.edit_permission_level"
                value={editingPermission.permission}
                onChange={({ target }) => setEditingPermission({ ...editingPermission, permission: target.value })}
              >
                {PERMISSIONS.map((p) => (
                  <SimpleSelectOption key={p} value={p}>
                    {p}
                  </SimpleSelectOption>
                ))}
              </SimpleSelect>
            </div>
          )}
        </div>
      </Modal>
      <Modal
        componentId="admin.role.remove_permission_modal"
        title="Remove Permission"
        visible={Boolean(deletePermissionTarget)}
        onCancel={() => {
          setDeletePermissionTarget(null);
          setError(null);
        }}
        onOk={handleRemovePermission}
        okText="Remove"
        okButtonProps={{ danger: true }}
        confirmLoading={removePermission.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="admin.role.remove_permission_modal_error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
          <Typography.Text>
            Are you sure you want to remove the <strong>{deletePermissionTarget?.permission}</strong> permission on{' '}
            <strong>{deletePermissionTarget?.resource_type}</strong> ({deletePermissionTarget?.resource_pattern})? This
            action cannot be undone.
          </Typography.Text>
        </div>
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
    setError(null);
    if (!selectedUsername) {
      // Surface inline feedback instead of a silent no-op when the user
      // clicks Assign without picking anyone from the combobox.
      setError('Please select a user');
      return;
    }
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

  const fetchError = queryError || usersError;

  const emptyState =
    assignments.length === 0 ? (
      <Empty title="No users assigned" description="Assign a user to give them this role's permissions." />
    ) : null;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Title level={4} withoutMargins>
          Assigned Users
        </Typography.Title>
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
          <TableHeader componentId="admin.role.assigned_user_id_header" css={{ flex: 1 }}>
            User ID
          </TableHeader>
          <TableHeader componentId="admin.role.assigned_username_header" css={{ flex: 2 }}>
            Username
          </TableHeader>
          <TableHeader componentId="admin.role.assigned_actions_header" css={{ flex: 0, minWidth: 80, maxWidth: 80 }}>
            Actions
          </TableHeader>
        </TableRow>
        {assignments.map((assignment) => {
          const username = userMap.get(assignment.user_id);
          return (
            <TableRow key={assignment.id}>
              <TableCell css={{ flex: 1 }}>{assignment.user_id}</TableCell>
              <TableCell css={{ flex: 2 }}>{username ?? 'Unknown'}</TableCell>
              <TableCell css={{ flex: 0, minWidth: 80, maxWidth: 80 }}>
                {username ? (
                  <Button
                    componentId="admin.role.unassign_button"
                    type="tertiary"
                    icon={<TrashIcon />}
                    size="small"
                    aria-label={`Unassign ${username}`}
                    onClick={() => handleUnassign(username)}
                    danger
                  />
                ) : null}
              </TableCell>
            </TableRow>
          );
        })}
      </Table>
      <Modal
        componentId="admin.role.assign_user_modal"
        title="Assign User to Role"
        visible={showAssignModal}
        onCancel={() => {
          setShowAssignModal(false);
          setSelectedUsername('');
          setError(null);
        }}
        onOk={handleAssign}
        okText="Assign"
        confirmLoading={assignRole.isLoading}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="admin.role.assign_user_modal_error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
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
  const withReturnTo = useWithSettingsReturnTo();
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
    // Role name is required — if the user blanks the input, surface a
    // validation error inline rather than silently sending ``undefined``
    // (which the backend treats as "don't change", so the save would
    // appear to succeed with no actual update).
    const trimmedName = editName.trim();
    if (!trimmedName) {
      setError('Role name cannot be empty');
      return;
    }
    try {
      await updateRole.mutateAsync({
        role_id: roleId,
        name: trimmedName,
        // Send the empty string as-is so the user can explicitly clear an
        // existing description; coercing to ``undefined`` would make that
        // impossible (the backend would treat it as "don't change").
        description: editDescription,
      });
      setIsEditing(false);
    } catch (e: any) {
      setError(e.message || 'Failed to update role');
    }
  };

  if (!isValidRoleId) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.md }}>
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
      </ScrollablePageWrapper>
    );
  }

  if (loadError || !role) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.md }}>
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
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              {/* Preserve the Roles tab when going "back to Admin" — without
                  the query param, AdminPage falls back to the Users tab.
                  Also carry through the Settings ``returnTo`` param so the
                  Settings exit link still goes back to the originating page
                  if the user entered Admin from Settings. */}
              <Link
                componentId="admin.role.breadcrumb_admin"
                to={withReturnTo(`${AdminRoutes.adminPageRoute}?tab=roles`)}
              >
                Platform Admin
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>{role.name}</Breadcrumb.Item>
          </Breadcrumb>
          <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <Typography.Title withoutMargins level={2}>
                  {role.name}
                </Typography.Title>
                {isWorkspaceAdminRole(role) && (
                  // Match the rolesTable column header ("Workspace Manager")
                  // and the per-row tag text ("Manager") used elsewhere.
                  <Tag componentId="admin.role.admin_tag" color="indigo">
                    Manager
                  </Tag>
                )}
              </div>
              <Typography.Text color="secondary">
                {role.description || 'No description'} · Workspace: {role.workspace}
              </Typography.Text>
            </div>
            <Button componentId="admin.role.edit_button" type="primary" onClick={handleStartEditing}>
              Edit Role
            </Button>
          </div>
        </div>

        {error && (
          <Alert componentId="admin.role.error" type="error" message={error} closable onClose={() => setError(null)} />
        )}

        <Modal
          componentId="admin.role.edit_modal"
          title="Edit Role"
          visible={isEditing}
          onCancel={() => {
            setIsEditing(false);
            setError(null);
          }}
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
