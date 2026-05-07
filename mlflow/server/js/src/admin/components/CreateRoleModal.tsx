import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { LongFormSection } from '../../common/components/long-form/LongFormSection';
import { AdminApi } from '../api';
import { useCreateRole } from '../hooks';
import { useWorkspaces } from '../../workspaces/hooks/useWorkspaces';
import { useWorkspacesEnabled } from '../../experiment-tracking/hooks/useServerInfo';
import {
  ALL_RESOURCE_PATTERN_LABEL,
  DEFAULT_WORKSPACE_NAME,
  formatResourcePattern,
  parseResourcePattern,
  type CreateRoleRequest,
} from '../types';
import { RolePermissionsSection, type StagedRolePermission } from './RolePermissionsSection';
import { RoleUsersSection } from './RoleUsersSection';

export interface CreateRoleModalProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Single-page modal with three vertical sections (role details + staged
 * permissions + staged user assignments). On submit:
 *   1. Create the role.
 *   2. For each staged permission, addPermission(roleId, ...).
 *   3. For each staged user, assignRole(roleId, username).
 * Best-effort: partial failures are surfaced inline; the role itself
 * always lands first. Mirrors ``CreateUserModal``'s vertical-section
 * shape.
 */
export const CreateRoleModal = ({ open, onClose }: CreateRoleModalProps) => {
  const { theme } = useDesignSystemTheme();
  const createRole = useCreateRole();
  const { workspacesEnabled } = useWorkspacesEnabled();
  const { workspaces } = useWorkspaces(workspacesEnabled);

  const [name, setName] = useState('');
  const [workspace, setWorkspace] = useState(DEFAULT_WORKSPACE_NAME);
  const [description, setDescription] = useState('');
  const [permissions, setPermissions] = useState<StagedRolePermission[]>([]);
  const [usernames, setUsernames] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const workspaceOptions = useMemo(() => {
    const others = new Set<string>();
    for (const w of workspaces) {
      if (w.name !== DEFAULT_WORKSPACE_NAME) others.add(w.name);
    }
    return [DEFAULT_WORKSPACE_NAME, ...Array.from(others).sort()];
  }, [workspaces]);

  useEffect(() => {
    if (open) {
      setName('');
      setWorkspace(DEFAULT_WORKSPACE_NAME);
      setDescription('');
      setPermissions([]);
      setUsernames([]);
      setSubmitting(false);
      setError(null);
    }
  }, [open]);

  const canSubmit = name.trim().length > 0;

  const handleSubmit = useCallback(async () => {
    setError(null);
    const trimmedName = name.trim();
    if (!trimmedName) {
      setError('Role name cannot be empty');
      return;
    }

    setSubmitting(true);

    let roleId: number;
    try {
      const request: CreateRoleRequest = {
        name: trimmedName,
        workspace,
        description: description || undefined,
      };
      const created = await createRole.mutateAsync(request);
      roleId = created.role.id;
    } catch (e: any) {
      setError(e?.message || 'Failed to create role');
      setSubmitting(false);
      return;
    }

    // Best-effort: partial failures are surfaced, but the role itself
    // already exists by this point.
    const permFailures: string[] = [];
    for (const p of permissions) {
      try {
        await AdminApi.addPermission({
          role_id: roleId,
          resource_type: p.resourceType,
          resource_pattern: parseResourcePattern(p.resourcePattern),
          permission: p.permission,
        });
      } catch (e: any) {
        permFailures.push(
          `${p.resourceType}:${formatResourcePattern(p.resourcePattern)} → ${p.permission} (${e?.message ?? 'unknown'})`,
        );
      }
    }
    const userFailures: string[] = [];
    for (const u of usernames) {
      try {
        await AdminApi.assignRole(u, roleId);
      } catch (e: any) {
        userFailures.push(`${u} (${e?.message ?? 'unknown'})`);
      }
    }

    if (permFailures.length === 0 && userFailures.length === 0) {
      onClose();
      return;
    }
    setError(
      `Role "${trimmedName}" created, but some follow-ups failed:\n` +
        (permFailures.length > 0 ? `Permissions: ${permFailures.join('; ')}\n` : '') +
        (userFailures.length > 0 ? `Users: ${userFailures.join('; ')}` : '') +
        `\nFinish wiring up the role from the role detail page.`,
    );
    setSubmitting(false);
  }, [name, workspace, description, permissions, usernames, createRole, onClose]);

  return (
    <Modal
      componentId="admin.create_role_modal"
      title="Create Role"
      visible={open}
      onCancel={onClose}
      size="wide"
      footer={
        <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
          <Button componentId="admin.create_role_modal.cancel" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button
            componentId="admin.create_role_modal.submit"
            type="primary"
            onClick={handleSubmit}
            loading={submitting}
            disabled={!canSubmit}
          >
            Create role
          </Button>
        </div>
      }
    >
      {error && (
        <Alert
          componentId="admin.create_role_modal.error"
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
          css={{ marginBottom: theme.spacing.md }}
        />
      )}
      <LongFormSection title="Role details">
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div>
            <FieldLabel>Name</FieldLabel>
            <Input
              componentId="admin.create_role_modal.name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter role name"
              autoFocus
              disabled={submitting}
            />
          </div>
          {workspacesEnabled && (
            <div>
              <FieldLabel>Workspace</FieldLabel>
              <SimpleSelect
                id="admin-create-role-modal-workspace"
                componentId="admin.create_role_modal.workspace"
                value={workspace}
                onChange={({ target }) => setWorkspace(target.value)}
                disabled={submitting}
              >
                {workspaceOptions.map((w) => (
                  <SimpleSelectOption key={w} value={w}>
                    {w}
                  </SimpleSelectOption>
                ))}
              </SimpleSelect>
            </div>
          )}
          <div>
            <FieldLabel>Description</FieldLabel>
            <Input
              componentId="admin.create_role_modal.description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter description (optional)"
              disabled={submitting}
            />
          </div>
          <Typography.Paragraph css={{ color: theme.colors.textSecondary, marginTop: theme.spacing.sm }}>
            To make this a workspace manager role, add a permission with resource type <strong>workspace</strong>,
            resource pattern <strong>{ALL_RESOURCE_PATTERN_LABEL}</strong>, and permission <strong>MANAGE</strong> in
            the next section.
          </Typography.Paragraph>
        </div>
      </LongFormSection>
      <LongFormSection title="Permissions" subtitle="(Optional)">
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          Add one or more permissions to this role. You can add more later from the role detail page.
        </Typography.Text>
        <RolePermissionsSection
          value={permissions}
          onChange={setPermissions}
          workspace={workspace}
          disabled={submitting}
        />
      </LongFormSection>
      <LongFormSection title="Assigned users" subtitle="(Optional)" hideDivider>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          Assign one or more users to this role. You can assign more later from the role detail page.
        </Typography.Text>
        <RoleUsersSection value={usernames} onChange={setUsernames} disabled={submitting} />
      </LongFormSection>
    </Modal>
  );
};
