import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Switch,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { LongFormSection } from '../../common/components/long-form/LongFormSection';
import { AdminApi } from '../api';
import {
  AdminQueryKeys,
  useCreateUser,
  useCurrentUserIsAdmin,
  useGrantUserPermission,
  useWorkspaceOptions,
} from '../hooks';
import { AccountQueryKeys } from '../../account/hooks';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { useWorkspaces } from '../../workspaces/hooks/useWorkspaces';
import { useWorkspacesEnabled } from '../../experiment-tracking/hooks/useServerInfo';
import { DEFAULT_WORKSPACE_NAME } from '../types';
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
  const activeWorkspace = useActiveWorkspace();
  const { workspacesEnabled } = useWorkspacesEnabled();
  // Platform admins can pick a workspace other than the active one to grant
  // direct permissions in; workspace managers stay implicit-active.
  const showWorkspaceSelector = isCurrentUserAdmin && workspacesEnabled;
  const { workspaces } = useWorkspaces(showWorkspaceSelector);
  const initialGrantWorkspace = activeWorkspace ?? DEFAULT_WORKSPACE_NAME;

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [roleValue, setRoleValue] = useState<RoleAssignmentValue>(ROLE_ASSIGNMENT_DEFAULT);
  const [directPermissions, setDirectPermissions] = useState<StagedDirectPermission[]>([]);
  const [grantWorkspace, setGrantWorkspace] = useState<string>(initialGrantWorkspace);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Set after the user lands; lets retries skip ``createUser`` and
  // just re-run the failed best-effort follow-ups.
  const [createdUsername, setCreatedUsername] = useState<string | null>(null);

  const workspaceOptions = useWorkspaceOptions(workspaces);

  useEffect(() => {
    if (open) {
      setUsername('');
      setPassword('');
      setIsAdmin(false);
      setRoleValue(ROLE_ASSIGNMENT_DEFAULT);
      setDirectPermissions([]);
      setGrantWorkspace(initialGrantWorkspace);
      setSubmitting(false);
      setError(null);
      setCreatedUsername(null);
    }
    // ``initialGrantWorkspace`` is derived from ``activeWorkspace``; reset on
    // open only — otherwise switching the session workspace mid-edit would
    // wipe the admin's selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const wantsRoles = roleValue.roleIds.length > 0;
  const wantsDirect = directPermissions.length > 0;
  // Retry mode skips the credential guard (the fields are also disabled).
  const canSubmit = createdUsername !== null || Boolean(username.trim() && password);

  const handleSubmit = useCallback(async () => {
    // No ``setError(null)`` upfront — would hide/show flicker as the
    // async path resets it. Every branch below replaces or closes.
    const trimmedUsername = username.trim();
    if (createdUsername === null && (!trimmedUsername || !password)) {
      setError('Username and password are required');
      return;
    }

    setSubmitting(true);
    // Skip the create step if a prior submission already landed it; we
    // only need to retry the failed best-effort follow-ups.
    if (createdUsername === null) {
      try {
        await createUser.mutateAsync({ username: trimmedUsername, password });
        setCreatedUsername(trimmedUsername);
      } catch (e: any) {
        setError(e?.message || 'Failed to create user');
        setSubmitting(false);
        return;
      }
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
      // The Admin Users tab eager-loads each user's roles via
      // ``useUsersQuery``; invalidate so the new assignments show up.
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
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
            workspace: grantWorkspace,
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
    grantWorkspace,
    createdUsername,
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
            {createdUsername !== null
              ? 'Retry failed grants'
              : isAdmin || wantsRoles || wantsDirect
                ? 'Create user and grant access'
                : 'Create user'}
          </Button>
        </div>
      }
    >
      {error && (
        // Sticky so partial-failure errors stay visible during scroll.
        <Alert
          componentId="admin.create_user_modal.error"
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
          css={{
            marginBottom: theme.spacing.md,
            position: 'sticky',
            top: 0,
            zIndex: 1,
          }}
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
              disabled={submitting || createdUsername !== null}
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
              disabled={submitting || createdUsername !== null}
            />
          </div>
        </div>
      </LongFormSection>
      <LongFormSection title="Role assignment" collapsible defaultCollapsed>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          Assign one or more existing roles to give this user reusable bundles of permissions.
        </Typography.Text>
        <RoleAssignmentForm value={roleValue} onChange={setRoleValue} disabled={submitting} />
      </LongFormSection>
      <LongFormSection title="Direct permissions" collapsible defaultCollapsed hideDivider={!isCurrentUserAdmin}>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          Grant one or more one-off permissions on specific resources.
        </Typography.Text>
        {showWorkspaceSelector && (
          <div css={{ marginBottom: theme.spacing.md }}>
            <FieldLabel>Workspace</FieldLabel>
            <SimpleSelect
              id="admin-create-user-modal-grant-workspace"
              componentId="admin.create_user_modal.grant_workspace"
              value={grantWorkspace}
              onChange={({ target }) => setGrantWorkspace(target.value)}
              disabled={submitting || createdUsername !== null}
            >
              {workspaceOptions.map((w) => (
                <SimpleSelectOption key={w} value={w}>
                  {w}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
            <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginTop: theme.spacing.xs }}>
              Grants land on the user's per-workspace direct-grant role. Pick a workspace other than your active one to
              grant access there.
            </Typography.Text>
          </div>
        )}
        <DirectPermissionsSection
          value={directPermissions}
          onChange={setDirectPermissions}
          workspace={grantWorkspace}
          disabled={submitting}
        />
      </LongFormSection>
      {isCurrentUserAdmin && (
        // Intentionally not ``collapsible``: ``Admin status`` is a single
        // Switch row, so hiding it behind a toggle adds an extra click for
        // no real density win (vs. the multi-field sections above).
        <LongFormSection title="Admin status" hideDivider>
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
