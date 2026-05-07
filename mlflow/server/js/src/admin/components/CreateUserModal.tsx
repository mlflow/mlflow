import { useCallback, useEffect, useState } from 'react';
import { Alert, Button, Input, Modal, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { LongFormSection } from '../../common/components/long-form/LongFormSection';
import { AdminApi } from '../api';
import { AdminQueryKeys, useCreateUser, useCurrentUserIsAdmin, useGrantUserPermission } from '../hooks';
import { AccountQueryKeys } from '../../account/hooks';
import { RoleAssignmentForm, ROLE_ASSIGNMENT_DEFAULT, type RoleAssignmentValue } from './RoleAssignmentForm';
import { DirectPermissionsSection, type StagedDirectPermission } from './DirectPermissionsSection';

export interface CreateUserModalProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Single-page modal with vertical sections: user details, optional role
 * assignment, optional direct permissions, and (for platform admins
 * only) optional admin status. The role / direct / admin sections share
 * the same UI as ``EditAccessModal`` so the two entry points read
 * identically. Mirrors the section pattern used by gateway endpoint
 * creation (``LongFormSection`` two-column layout).
 */
export const CreateUserModal = ({ open, onClose }: CreateUserModalProps) => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const createUser = useCreateUser();
  const grantPermission = useGrantUserPermission();
  // Setting ``is_admin`` requires the current user to themselves be a
  // platform admin. A workspace admin (workspace-level MANAGE without
  // ``is_admin``) shouldn't see the toggle.
  const isCurrentUserAdmin = useCurrentUserIsAdmin();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [roleValue, setRoleValue] = useState<RoleAssignmentValue>(ROLE_ASSIGNMENT_DEFAULT);
  const [directPermissions, setDirectPermissions] = useState<StagedDirectPermission[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setUsername('');
      setPassword('');
      setIsAdmin(false);
      setRoleValue(ROLE_ASSIGNMENT_DEFAULT);
      setDirectPermissions([]);
      setSubmitting(false);
      setError(null);
    }
  }, [open]);

  const wantsRoles = roleValue.roleIds.length > 0;
  const wantsDirect = directPermissions.length > 0;
  const canSubmit = Boolean(username.trim() && password);

  const handleSubmit = useCallback(async () => {
    setError(null);
    const trimmedUsername = username.trim();
    if (!trimmedUsername || !password) {
      setError('Username and password are required');
      return;
    }

    setSubmitting(true);
    try {
      await createUser.mutateAsync({ username: trimmedUsername, password });
    } catch (e: any) {
      setError(e?.message || 'Failed to create user');
      setSubmitting(false);
      return;
    }

    // The user exists. Treat the follow-up steps as best-effort: surface
    // partial failures inline rather than rolling back the user.
    const failures: string[] = [];
    if (isAdmin) {
      try {
        await AdminApi.updateAdmin({ username: trimmedUsername, is_admin: true });
        queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
      } catch (e: any) {
        failures.push(`Setting admin status failed: ${e?.message ?? 'unknown error'}`);
      }
    }
    if (wantsRoles) {
      for (const roleId of roleValue.roleIds) {
        try {
          await AdminApi.assignRole(trimmedUsername, roleId);
        } catch (e: any) {
          failures.push(`Role assignment (id ${roleId}) failed: ${e?.message ?? 'unknown error'}`);
        }
      }
      queryClient.invalidateQueries({ queryKey: AccountQueryKeys.userRoles(trimmedUsername) });
      for (const roleId of roleValue.roleIds) {
        queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleUsers(roleId) });
      }
    }
    if (wantsDirect) {
      for (const p of directPermissions) {
        try {
          await grantPermission.mutateAsync({
            resource_type: p.resourceType,
            resource_id: p.resourceId,
            username: trimmedUsername,
            permission: p.permission,
          });
        } catch (e: any) {
          failures.push(
            `Direct permission ${p.resourceType}:${p.resourceId} → ${p.permission} failed: ${e?.message ?? 'unknown error'}`,
          );
        }
      }
    }

    if (failures.length === 0) {
      onClose();
      return;
    }
    setError(
      `User ${trimmedUsername} created, but some grants failed:\n${failures.join('\n')}\n` +
        `Open the user's detail page and click "Edit access" to retry.`,
    );
    setSubmitting(false);
  }, [
    username,
    password,
    isAdmin,
    wantsRoles,
    wantsDirect,
    roleValue.roleIds,
    directPermissions,
    createUser,
    queryClient,
    grantPermission,
    onClose,
  ]);

  return (
    <Modal
      componentId="admin.create_user_modal"
      title="Create User"
      visible={open}
      onCancel={onClose}
      size="wide"
      footer={
        <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
          <Button componentId="admin.create_user_modal.cancel" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button
            componentId="admin.create_user_modal.submit"
            type="primary"
            onClick={handleSubmit}
            loading={submitting}
            disabled={!canSubmit}
          >
            {isAdmin || wantsRoles || wantsDirect ? 'Create user and grant access' : 'Create user'}
          </Button>
        </div>
      }
    >
      {error && (
        <Alert
          componentId="admin.create_user_modal.error"
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
          css={{ marginBottom: theme.spacing.md }}
        />
      )}
      <LongFormSection title="User details">
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div>
            <FieldLabel>Username</FieldLabel>
            <Input
              componentId="admin.create_user_modal.username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              autoFocus
              disabled={submitting}
            />
          </div>
          <div>
            <FieldLabel>Password</FieldLabel>
            <Input
              componentId="admin.create_user_modal.password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              disabled={submitting}
            />
          </div>
        </div>
      </LongFormSection>
      <LongFormSection title="Role assignment" subtitle="(Optional)">
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          Assign one or more existing roles to give this user reusable bundles of permissions.
        </Typography.Text>
        <RoleAssignmentForm value={roleValue} onChange={setRoleValue} disabled={submitting} />
      </LongFormSection>
      <LongFormSection title="Direct permissions" subtitle="(Optional)" hideDivider={!isCurrentUserAdmin}>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          Grant one or more one-off permissions on specific resources.
        </Typography.Text>
        <DirectPermissionsSection value={directPermissions} onChange={setDirectPermissions} disabled={submitting} />
      </LongFormSection>
      {isCurrentUserAdmin && (
        <LongFormSection title="Admin status" subtitle="(Optional)" hideDivider>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
            Admins can manage all users, roles, and workspaces.
          </Typography.Text>
          <Switch
            componentId="admin.create_user_modal.is_admin"
            checked={isAdmin}
            onChange={setIsAdmin}
            label="Make this user an admin"
            disabled={submitting}
          />
        </LongFormSection>
      )}
    </Modal>
  );
};
