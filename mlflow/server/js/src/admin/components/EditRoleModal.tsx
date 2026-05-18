import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  ChevronLeftIcon,
  Input,
  Modal,
  Spinner,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { LongFormSection } from '../../common/components/long-form/LongFormSection';
import {
  useAddPermission,
  useAssignRole,
  useRemovePermission,
  useRoleDetailQuery,
  useRoleUsersQuery,
  useUnassignRole,
  useUpdateRole,
  useUsersQuery,
} from '../hooks';
import { formatResourcePattern, parseResourcePattern } from '../types';
import { RolePermissionsSection, type StagedRolePermission } from './RolePermissionsSection';
import { RoleUsersSection } from './RoleUsersSection';

export interface EditRoleModalProps {
  open: boolean;
  onClose: () => void;
  roleId: number;
}

const permTripleKey = (p: { resourceType: string; resourcePattern: string; permission: string }) =>
  `${p.resourceType}::${p.resourcePattern}::${p.permission}`;

interface RoleDiff {
  nameChange: string | null;
  descriptionChange: string | null;
  permissionsToAdd: StagedRolePermission[];
  permissionIdsToRemove: number[];
  usersToAssign: string[];
  usersToUnassign: string[];
}

/**
 * Edit-style modal for managing one role end-to-end. Pre-fills name,
 * description, current permissions, and current user assignments;
 * compute a diff on submit; surface every add/remove in the Review
 * step before applying. Mirrors ``EditAccessModal``'s shape.
 *
 * Workspace is *not* editable on existing roles — the backend
 * ``UpdateRoleRequest`` only accepts ``name`` and ``description`` — so
 * the workspace renders read-only with a hint.
 */
export const EditRoleModal = ({ open, onClose, roleId }: EditRoleModalProps) => {
  const { theme } = useDesignSystemTheme();
  const updateRole = useUpdateRole(roleId);
  const addPermission = useAddPermission(roleId);
  const removePermission = useRemovePermission(roleId);
  const assignRole = useAssignRole(roleId);
  const unassignRole = useUnassignRole(roleId);

  // --- Current state from backend ---
  const { data: roleData, isLoading: roleLoading } = useRoleDetailQuery(roleId);
  const { data: assignmentsData, isLoading: assignmentsLoading } = useRoleUsersQuery(roleId);
  const { data: usersData, isLoading: usersLoading } = useUsersQuery();

  const userIdToUsername = useMemo(() => {
    const m = new Map<number, string>();
    for (const u of usersData?.users ?? []) m.set(u.id, u.username);
    return m;
  }, [usersData]);

  const currentName = roleData?.role?.name ?? '';
  const currentDescription = roleData?.role?.description ?? '';
  const currentWorkspace = roleData?.role?.workspace ?? '';

  const currentPermissions = useMemo<StagedRolePermission[]>(() => {
    return (roleData?.role?.permissions ?? []).map((p) => ({
      id: p.id,
      resourceType: p.resource_type,
      // Display the user-facing "all" label so dedup against newly-typed
      // entries (which also carry the label) lines up.
      resourcePattern: formatResourcePattern(p.resource_pattern),
      permission: p.permission,
    }));
  }, [roleData]);

  const currentUsernames = useMemo<string[]>(() => {
    const set = new Set<string>();
    for (const a of assignmentsData?.assignments ?? []) {
      const u = userIdToUsername.get(a.user_id);
      if (u) set.add(u);
    }
    return Array.from(set).sort();
  }, [assignmentsData, userIdToUsername]);

  // --- Editable state ---
  const [step, setStep] = useState<'edit' | 'review'>('edit');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [permissions, setPermissions] = useState<StagedRolePermission[]>([]);
  const [usernames, setUsernames] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stateLoaded = !roleLoading && !assignmentsLoading && !usersLoading;

  useEffect(() => {
    if (!open) {
      return;
    }
    setStep('edit');
    setName(currentName);
    setDescription(currentDescription);
    setPermissions([...currentPermissions]);
    setUsernames([...currentUsernames]);
    setSubmitting(false);
    setError(null);
  }, [open, currentName, currentDescription, currentPermissions, currentUsernames]);

  // --- Diff ---
  const diff = useMemo<RoleDiff>(() => {
    const trimmedName = name.trim();
    const nameChange = trimmedName !== currentName.trim() && trimmedName.length > 0 ? trimmedName : null;
    // Empty description is meaningful (clears it); only ``null`` means "no change".
    const descriptionChange = description !== currentDescription ? description : null;

    const desiredKeys = new Set(permissions.map(permTripleKey));
    const permissionsToAdd = permissions.filter((p) => p.id == null);
    const permissionIdsToRemove = currentPermissions
      .filter((p) => p.id != null && !desiredKeys.has(permTripleKey(p)))
      .map((p) => p.id as number);

    const desiredUsernameSet = new Set(usernames);
    const currentUsernameSet = new Set(currentUsernames);
    const usersToAssign = usernames.filter((u) => !currentUsernameSet.has(u));
    const usersToUnassign = currentUsernames.filter((u) => !desiredUsernameSet.has(u));

    return { nameChange, descriptionChange, permissionsToAdd, permissionIdsToRemove, usersToAssign, usersToUnassign };
  }, [
    name,
    description,
    permissions,
    usernames,
    currentName,
    currentDescription,
    currentPermissions,
    currentUsernames,
  ]);

  const hasAnyChange =
    diff.nameChange !== null ||
    diff.descriptionChange !== null ||
    diff.permissionsToAdd.length > 0 ||
    diff.permissionIdsToRemove.length > 0 ||
    diff.usersToAssign.length > 0 ||
    diff.usersToUnassign.length > 0;

  // Map permissionId → (type, pattern, permission) for the Review step's
  // human label of removals. (We can't read it directly off ``diff``
  // because the diff carries only the id.)
  const permissionByIdLabel = useMemo(() => {
    const m = new Map<number, string>();
    for (const p of currentPermissions) {
      if (p.id != null) {
        m.set(p.id, `${p.resourceType}:${p.resourcePattern} → ${p.permission}`);
      }
    }
    return m;
  }, [currentPermissions]);

  const handleConfirm = useCallback(async () => {
    setError(null);
    setSubmitting(true);
    const failures: string[] = [];

    // 1. Role details (single PATCH).
    if (diff.nameChange !== null || diff.descriptionChange !== null) {
      try {
        await updateRole.mutateAsync({
          role_id: roleId,
          ...(diff.nameChange !== null ? { name: diff.nameChange } : {}),
          // Send the raw description (including ``""``) so the user can
          // explicitly clear it. Skip the field entirely on no-change so
          // the backend doesn't see an empty PATCH.
          ...(diff.descriptionChange !== null ? { description: diff.descriptionChange } : {}),
        });
      } catch (e: any) {
        failures.push(`Updating role details failed: ${e?.message ?? 'unknown error'}`);
      }
    }

    // 2. Permissions add.
    for (const p of diff.permissionsToAdd) {
      try {
        await addPermission.mutateAsync({
          role_id: roleId,
          resource_type: p.resourceType,
          resource_pattern: parseResourcePattern(p.resourcePattern),
          permission: p.permission,
        });
      } catch (e: any) {
        failures.push(
          `Adding ${p.resourceType}:${p.resourcePattern} → ${p.permission} failed: ${e?.message ?? 'unknown error'}`,
        );
      }
    }
    // 3. Permissions remove.
    for (const id of diff.permissionIdsToRemove) {
      try {
        await removePermission.mutateAsync(id);
      } catch (e: any) {
        const label = permissionByIdLabel.get(id) ?? `permission #${id}`;
        failures.push(`Removing ${label} failed: ${e?.message ?? 'unknown error'}`);
      }
    }

    // 4. Users assign.
    for (const u of diff.usersToAssign) {
      try {
        await assignRole.mutateAsync(u);
      } catch (e: any) {
        failures.push(`Assigning ${u} failed: ${e?.message ?? 'unknown error'}`);
      }
    }
    // 5. Users unassign.
    for (const u of diff.usersToUnassign) {
      try {
        await unassignRole.mutateAsync(u);
      } catch (e: any) {
        failures.push(`Unassigning ${u} failed: ${e?.message ?? 'unknown error'}`);
      }
    }

    if (failures.length === 0) {
      onClose();
      return;
    }
    setError(failures.join('\n'));
    setStep('edit');
    setSubmitting(false);
  }, [
    diff,
    roleId,
    updateRole,
    addPermission,
    removePermission,
    assignRole,
    unassignRole,
    onClose,
    permissionByIdLabel,
  ]);

  return (
    <Modal
      componentId="admin.edit_role_modal"
      title={`Edit role${currentName ? ` — ${currentName}` : ''}`}
      visible={open}
      onCancel={onClose}
      size="wide"
      footer={
        step === 'edit' ? (
          <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
            <Button componentId="admin.edit_role_modal.cancel" onClick={onClose} disabled={submitting}>
              Cancel
            </Button>
            <Button
              componentId="admin.edit_role_modal.review"
              type="primary"
              onClick={() => setStep('review')}
              disabled={!hasAnyChange || !stateLoaded || !name.trim()}
            >
              Review changes
            </Button>
          </div>
        ) : (
          <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
            <Button
              componentId="admin.edit_role_modal.back"
              type="tertiary"
              icon={<ChevronLeftIcon />}
              onClick={() => setStep('edit')}
              disabled={submitting}
            >
              Back
            </Button>
            <Button
              componentId="admin.edit_role_modal.confirm"
              type="primary"
              onClick={handleConfirm}
              loading={submitting}
            >
              Apply changes
            </Button>
          </div>
        )
      }
    >
      {error && (
        <Alert
          componentId="admin.edit_role_modal.error"
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
          css={{ marginBottom: theme.spacing.md }}
        />
      )}

      {step === 'edit' ? (
        <>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
            Update name, description, permissions, and assigned users. Changes are previewed before they're applied.
          </Typography.Text>
          {!stateLoaded ? (
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: theme.spacing.lg,
                minHeight: 200,
              }}
            >
              <Spinner size="small" />
            </div>
          ) : (
            <>
              <LongFormSection title="Role details">
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                  <div>
                    <FieldLabel>Name</FieldLabel>
                    <Input
                      componentId="admin.edit_role_modal.name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Enter role name"
                      disabled={submitting}
                    />
                  </div>
                  <div>
                    <FieldLabel>Description</FieldLabel>
                    <Input
                      componentId="admin.edit_role_modal.description"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Enter description (optional)"
                      disabled={submitting}
                    />
                  </div>
                  <div>
                    <FieldLabel>Workspace</FieldLabel>
                    <Typography.Text color="secondary">
                      {currentWorkspace || 'default'}{' '}
                      <Typography.Text color="secondary" size="sm">
                        (workspace is set at creation and can't be changed)
                      </Typography.Text>
                    </Typography.Text>
                  </div>
                </div>
              </LongFormSection>
              <LongFormSection title="Permissions">
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  Current permissions are pre-filled. Remove a row to revoke; use the form below to add more.
                </Typography.Text>
                <RolePermissionsSection
                  value={permissions}
                  onChange={setPermissions}
                  workspace={currentWorkspace}
                  disabled={submitting}
                />
              </LongFormSection>
              <LongFormSection title="Assigned users" hideDivider>
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  Currently assigned users are pre-filled. Remove a user to unassign; use the form below to assign more.
                </Typography.Text>
                <RoleUsersSection value={usernames} onChange={setUsernames} disabled={submitting} />
              </LongFormSection>
            </>
          )}
        </>
      ) : (
        <ReviewSummary diff={diff} permissionByIdLabel={permissionByIdLabel} />
      )}
    </Modal>
  );
};

const ReviewSummary = ({ diff, permissionByIdLabel }: { diff: RoleDiff; permissionByIdLabel: Map<number, string> }) => {
  const { theme } = useDesignSystemTheme();
  const renderPerm = (p: StagedRolePermission) => `${p.resourceType}:${p.resourcePattern} → ${p.permission}`;
  const renderRemovedPermId = (id: number) => permissionByIdLabel.get(id) ?? `permission #${id}`;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Text color="secondary">
        Review the net changes. Click <strong>Apply changes</strong> to commit them, or <strong>Back</strong> to keep
        editing.
      </Typography.Text>

      {diff.nameChange !== null ? (
        <DiffGroup title="Name" items={[`Rename to "${diff.nameChange}"`]} addColor />
      ) : (
        <DiffGroup title="Name" items={[]} emptyLabel="No change to name." />
      )}
      {diff.descriptionChange !== null ? (
        <DiffGroup
          title="Description"
          items={[diff.descriptionChange ? `Set to "${diff.descriptionChange}"` : 'Clear description']}
          addColor
        />
      ) : (
        <DiffGroup title="Description" items={[]} emptyLabel="No change to description." />
      )}
      <DiffGroup
        title="Permissions to add"
        items={diff.permissionsToAdd.map(renderPerm)}
        emptyLabel="No new permissions."
        addColor
      />
      <DiffGroup
        title="Permissions to remove"
        items={diff.permissionIdsToRemove.map(renderRemovedPermId)}
        emptyLabel="No permissions to remove."
      />
      <DiffGroup title="Users to assign" items={diff.usersToAssign} emptyLabel="No new user assignments." addColor />
      <DiffGroup title="Users to unassign" items={diff.usersToUnassign} emptyLabel="No user unassignments." />
    </div>
  );
};

const DiffGroup = ({
  title,
  items,
  emptyLabel,
  addColor,
}: {
  title: string;
  items: string[];
  emptyLabel?: string;
  addColor?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div>
      <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
        {title}
      </Typography.Text>
      {items.length === 0 ? (
        <Typography.Text color="secondary" size="sm">
          {emptyLabel}
        </Typography.Text>
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {items.map((s) => (
            <Tag
              key={s}
              componentId="admin.edit_role_modal.diff_tag"
              color={addColor ? 'lime' : 'lemon'}
              css={{ alignSelf: 'flex-start' }}
            >
              {s}
            </Tag>
          ))}
        </div>
      )}
    </div>
  );
};
