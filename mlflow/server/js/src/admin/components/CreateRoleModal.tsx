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
import { ConfirmationModal } from '../ConfirmationModal';
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
  // Reported by ``RolePermissionsSection`` whenever the in-progress draft is
  // dirty. Drives a discard-confirm dialog on ``Create role`` so the admin
  // can't silently abandon a partially-filled permission.
  const [hasUnsavedDraft, setHasUnsavedDraft] = useState(false);
  const [showDiscardConfirm, setShowDiscardConfirm] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Set after the role lands; lets retries skip ``createRole`` and
  // just re-run the failed best-effort follow-ups.
  const [createdRoleId, setCreatedRoleId] = useState<number | null>(null);

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
      // ``hasUnsavedDraft`` isn't reset here — the ``key={String(open)}`` on
      // ``RolePermissionsSection`` below remounts the section on every open
      // and its first commit-time effect fires ``false`` from the default
      // draft state.
      setShowDiscardConfirm(false);
      setSubmitting(false);
      setError(null);
      setCreatedRoleId(null);
    }
  }, [open]);

  // Retry mode skips the name guard (the field is also disabled).
  const canSubmit = createdRoleId !== null || name.trim().length > 0;

  const handleSubmit = useCallback(async () => {
    // No ``setError(null)`` upfront — would hide/show flicker as the
    // async path resets it. Every branch below replaces or closes.
    const trimmedName = name.trim();
    if (createdRoleId === null && !trimmedName) {
      setError('Role name cannot be empty');
      return;
    }

    setSubmitting(true);

    let roleId = createdRoleId;
    if (roleId === null) {
      try {
        const request: CreateRoleRequest = {
          name: trimmedName,
          workspace,
          description: description || undefined,
        };
        const created = await createRole.mutateAsync(request);
        roleId = created.role.id;
        setCreatedRoleId(roleId);
      } catch (e: any) {
        setError(e?.message || 'Failed to create role');
        setSubmitting(false);
        return;
      }
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
  }, [name, workspace, description, permissions, usernames, createdRoleId, createRole, onClose]);

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
            // Submit isn't blocked on an unsaved draft — instead we gate on
            // it via a discard-confirm dialog so the admin can either go
            // back and click Add, or knowingly drop the draft and proceed.
            onClick={() => (hasUnsavedDraft ? setShowDiscardConfirm(true) : handleSubmit())}
            loading={submitting}
            disabled={!canSubmit}
          >
            {createdRoleId !== null ? 'Retry failed grants' : 'Create role'}
          </Button>
        </div>
      }
    >
      {error && (
        // Sticky so partial-failure errors stay visible during scroll.
        <Alert
          componentId="admin.create_role_modal.error"
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
              disabled={submitting || createdRoleId !== null}
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
                disabled={submitting || createdRoleId !== null}
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
              disabled={submitting || createdRoleId !== null}
            />
          </div>
          <Typography.Paragraph css={{ color: theme.colors.textSecondary, marginTop: theme.spacing.sm }}>
            To make this a workspace manager role, add a permission with resource type <strong>workspace</strong>,
            resource pattern <strong>{ALL_RESOURCE_PATTERN_LABEL}</strong>, and permission <strong>MANAGE</strong> in
            the next section.
          </Typography.Paragraph>
        </div>
      </LongFormSection>
      <LongFormSection title="Permissions" collapsible defaultCollapsed>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          Add one or more permissions to this role. You can add more later from the role detail page.
        </Typography.Text>
        {/* ``key={String(open)}`` forces a fresh mount on every reopen so
            the section's internal ``draft`` can't bleed across close →
            reopen. */}
        <RolePermissionsSection
          key={String(open)}
          value={permissions}
          onChange={setPermissions}
          workspace={workspace}
          disabled={submitting}
          onUnsavedDraftChange={setHasUnsavedDraft}
        />
      </LongFormSection>
      <LongFormSection title="Assign users" hideDivider collapsible defaultCollapsed>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          Assign one or more users to this role. You can assign more later from the role detail page.
        </Typography.Text>
        <RoleUsersSection value={usernames} onChange={setUsernames} disabled={submitting} />
      </LongFormSection>
      {/* Discard-confirm gate: only intercepts when ``hasUnsavedDraft`` is
          true (any field touched in the permission picker without a
          subsequent ``Add`` or ``Clear``). The dialog is the warning
          surface; submit itself stays enabled so the admin can always
          click through. */}
      <ConfirmationModal
        componentId="admin.create_role_modal.discard_unsaved_draft"
        title="Discard unsaved role permission?"
        visible={showDiscardConfirm}
        message="You started adding a permission to this role but didn't click Add. Continuing will discard it. Go back to either click Add to stage it, or Clear to drop the draft on the spot."
        okText="Continue"
        cancelText="Back"
        // ``danger=false`` — the OK verb is neutral, destructive intent is
        // in the title's question.
        danger={false}
        onCancel={() => setShowDiscardConfirm(false)}
        onConfirm={() => {
          setShowDiscardConfirm(false);
          handleSubmit();
        }}
      />
    </Modal>
  );
};
