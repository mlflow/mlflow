import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import { ConfirmationModal } from '../ConfirmationModal';
import {
  useAddMutationCondition,
  useAddPermission,
  useAssignRole,
  useRemoveMutationCondition,
  useRemovePermission,
  useRoleDetailQuery,
  useRoleMutationConditionsQuery,
  useRoleUsersQuery,
  useUnassignRole,
  useUpdateRole,
  useUsersQuery,
} from '../hooks';
import { useWorkspacesEnabled } from '../../experiment-tracking/hooks/useServerInfo';
import { formatResourcePattern, parseResourcePattern } from '../types';
import { RolePermissionsSection, type StagedRolePermission } from './RolePermissionsSection';
import {
  conditionKey,
  formatStagedCondition,
  MutationConditionsSection,
  type StagedMutationCondition,
} from './MutationConditionsSection';
import { RoleUsersSection } from './RoleUsersSection';

export interface EditRoleModalProps {
  open: boolean;
  onClose: () => void;
  roleId: number;
}

const permTripleKey = (p: { resourceType: string; resourcePattern: string; permission: string }) =>
  `${p.resourceType}::${p.resourcePattern}::${p.permission}`;

// Conditions have no natural subset of identifying fields -- changing any one of them
// makes a different condition -- so the whole tuple is the key.
interface RoleDiff {
  nameChange: string | null;
  descriptionChange: string | null;
  permissionsToAdd: StagedRolePermission[];
  permissionIdsToRemove: number[];
  usersToAssign: string[];
  usersToUnassign: string[];
  conditionsToAdd: StagedMutationCondition[];
  conditionIdsToRemove: number[];
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
  const addCondition = useAddMutationCondition(roleId);
  const removeCondition = useRemoveMutationCondition(roleId);

  // --- Current state from backend ---
  const { data: roleData, isLoading: roleLoading } = useRoleDetailQuery(roleId);
  const { data: assignmentsData, isLoading: assignmentsLoading } = useRoleUsersQuery(roleId);
  const { data: usersData, isLoading: usersLoading } = useUsersQuery();
  const { data: conditionsData, isLoading: conditionsLoading } = useRoleMutationConditionsQuery(roleId);

  const userIdToUsername = useMemo(() => {
    const m = new Map<number, string>();
    for (const u of usersData?.users ?? []) m.set(u.id, u.username);
    return m;
  }, [usersData]);

  const currentName = roleData?.role?.name ?? '';
  const currentDescription = roleData?.role?.description ?? '';
  const currentWorkspace = roleData?.role?.workspace ?? '';
  const { workspacesEnabled } = useWorkspacesEnabled();
  // In single-tenant mode there is no workspace dimension — pass ``undefined``
  // to the resource picker so the ``X-MLFLOW-WORKSPACE`` header is omitted
  // (the server rejects any workspace header, including ``default``, when
  // workspaces are disabled).
  const resourcePickerWorkspace = workspacesEnabled ? currentWorkspace : undefined;

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

  const currentConditions = useMemo<StagedMutationCondition[]>(() => {
    return (conditionsData?.mutation_conditions ?? []).map((c) => ({
      id: c.id,
      resourceType: c.resource_type,
      resourcePattern: c.resource_pattern,
      containerResourceType: c.container_resource_type,
      containerResourcePattern: c.container_resource_pattern,
      valueCondition: c.value_condition,
      targetCondition: c.target_condition,
    }));
  }, [conditionsData]);

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
  const [conditions, setConditions] = useState<StagedMutationCondition[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Reported by ``RolePermissionsSection`` whenever the in-progress draft
  // is dirty. Drives a discard-confirm dialog on ``Review changes`` so the
  // admin can't silently abandon a partially-filled permission.
  const [hasUnsavedDraft, setHasUnsavedDraft] = useState(false);
  const [hasUnsavedConditionDraft, setHasUnsavedConditionDraft] = useState(false);
  const [showDiscardConfirm, setShowDiscardConfirm] = useState(false);

  const stateLoaded = !roleLoading && !assignmentsLoading && !usersLoading && !conditionsLoading;

  // ``prefilledRef`` gates the data-fill effect against background
  // refetches that would clobber in-progress edits.
  const prefilledRef = useRef(false);

  // Reset transient UI state only on open — refetches must not bounce
  // the user back to edit or wipe a partial-failure error.
  useEffect(() => {
    if (!open) {
      return;
    }
    setStep('edit');
    setSubmitting(false);
    setError(null);
    // ``hasUnsavedDraft`` isn't reset here — the ``key={String(open)}`` on
    // ``RolePermissionsSection`` below remounts the section on every open.
    setShowDiscardConfirm(false);
    prefilledRef.current = false;
  }, [open]);

  // Pre-fill editable fields once per open, after backing queries resolve.
  useEffect(() => {
    if (!open) {
      prefilledRef.current = false;
      return;
    }
    if (prefilledRef.current || !stateLoaded) {
      return;
    }
    setName(currentName);
    setDescription(currentDescription);
    setPermissions([...currentPermissions]);
    setUsernames([...currentUsernames]);
    setConditions([...currentConditions]);
    prefilledRef.current = true;
  }, [open, stateLoaded, currentName, currentDescription, currentPermissions, currentUsernames, currentConditions]);

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

    const desiredConditionKeys = new Set(conditions.map(conditionKey));
    const conditionsToAdd = conditions.filter((c) => c.id == null);
    const conditionIdsToRemove = currentConditions
      .filter((c) => c.id != null && !desiredConditionKeys.has(conditionKey(c)))
      .map((c) => c.id as number);

    return {
      nameChange,
      descriptionChange,
      permissionsToAdd,
      permissionIdsToRemove,
      usersToAssign,
      usersToUnassign,
      conditionsToAdd,
      conditionIdsToRemove,
    };
  }, [
    name,
    description,
    permissions,
    usernames,
    conditions,
    currentName,
    currentDescription,
    currentPermissions,
    currentUsernames,
    currentConditions,
  ]);

  const hasAnyChange =
    diff.nameChange !== null ||
    diff.descriptionChange !== null ||
    diff.permissionsToAdd.length > 0 ||
    diff.permissionIdsToRemove.length > 0 ||
    diff.usersToAssign.length > 0 ||
    diff.usersToUnassign.length > 0 ||
    diff.conditionsToAdd.length > 0 ||
    diff.conditionIdsToRemove.length > 0;

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

  const conditionByIdLabel = useMemo(() => {
    const m = new Map<number, string>();
    for (const c of currentConditions) {
      if (c.id != null) {
        m.set(c.id, formatStagedCondition(c));
      }
    }
    return m;
  }, [currentConditions]);

  const handleConfirm = useCallback(async () => {
    // ``error`` is already cleared by the "Review changes" transition;
    // skipping a redundant reset here also avoids the flicker.
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

    // The remaining order is a safety property, not housekeeping. Permissions add access
    // and conditions subtract it, and assigning a user to this role hands them everything
    // it carries -- so a restriction is created before the capability it narrows, and a
    // capability is removed before the restriction that was covering it. Each step is
    // best-effort within itself (separate requests, no transaction), so the two gates
    // below are what keep a partial failure fail-closed.

    // 2. Conditions add, ahead of anything that widens access.
    for (const c of diff.conditionsToAdd) {
      try {
        await addCondition.mutateAsync({
          role_id: roleId,
          resource_type: c.resourceType,
          // The scope travels explicitly: an absent `resource_pattern` is normalised
          // server-side to the wildcard, which silently widened a condition the admin
          // had scoped to one resource into one covering the whole workspace.
          resource_pattern: c.resourcePattern,
          container_resource_type: c.containerResourceType,
          container_resource_pattern: c.containerResourcePattern,
          value_condition: c.valueCondition,
          target_condition: c.targetCondition,
        });
      } catch (e: any) {
        failures.push(`Adding condition ${formatStagedCondition(c)} failed: ${e?.message ?? 'unknown error'}`);
      }
    }
    const restrictionsFailed = failures.length > 0;

    // 3. Capability REMOVALS -- these only narrow, and must precede any condition removal.
    const failuresBeforeRemovals = failures.length;
    for (const id of diff.permissionIdsToRemove) {
      try {
        await removePermission.mutateAsync(id);
      } catch (e: any) {
        const label = permissionByIdLabel.get(id) ?? `permission #${id}`;
        failures.push(`Removing ${label} failed: ${e?.message ?? 'unknown error'}`);
      }
    }
    for (const u of diff.usersToUnassign) {
      try {
        await unassignRole.mutateAsync(u);
      } catch (e: any) {
        failures.push(`Unassigning ${u} failed: ${e?.message ?? 'unknown error'}`);
      }
    }
    const capabilityRemovalFailed = failures.length > failuresBeforeRemovals;

    // 4. Capability ADDITIONS, only once every staged restriction is in place. Adding a
    // permission -- or assigning a user, which hands them the whole role -- when step 2
    // failed would grant exactly the unrestricted access the admin was trying to narrow.
    if (!restrictionsFailed) {
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
      for (const u of diff.usersToAssign) {
        try {
          await assignRole.mutateAsync(u);
        } catch (e: any) {
          failures.push(`Assigning ${u} failed: ${e?.message ?? 'unknown error'}`);
        }
      }
    }

    // 5. Conditions remove, last. A restriction is only lifted once the capability it was
    // covering is actually gone -- if a removal above failed, dropping the condition would
    // leave that permission live and unrestricted for everyone holding this role.
    // Both halves must be safe. `capabilityRemovalFailed` covers the capability this
    // condition was narrowing; `restrictionsFailed` covers its intended REPLACEMENT --
    // editing a condition is an add plus a remove, so a failed add would otherwise
    // still drop the old restriction, leaving the role LESS restricted than before
    // the edit.
    //
    // This also blocks a plain removal when an unrelated add failed, which is the
    // fail-closed direction: the restriction stays and the admin retries.
    if (!capabilityRemovalFailed && !restrictionsFailed) {
      for (const id of diff.conditionIdsToRemove) {
        try {
          await removeCondition.mutateAsync(id);
        } catch (e: any) {
          const label = conditionByIdLabel.get(id) ?? `condition #${id}`;
          failures.push(`Removing condition ${label} failed: ${e?.message ?? 'unknown error'}`);
        }
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
    addCondition,
    removeCondition,
    conditionByIdLabel,
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
              // Submit isn't blocked on an unsaved draft — instead we gate
              // on it via a discard-confirm dialog so the admin can either
              // go back and click Add, or knowingly drop the draft and
              // proceed to the review step.
              onClick={() => {
                // BOTH drafts gate the transition. `hasUnsavedConditionDraft` was reported
                // by `MutationConditionsSection` and then never read, so a half-filled
                // condition was dropped silently and the submit went on to ADD the
                // permissions it was meant to narrow -- the one fail-open direction this
                // dialog exists to prevent.
                if (hasUnsavedDraft || hasUnsavedConditionDraft) {
                  setShowDiscardConfirm(true);
                  return;
                }
                setError(null);
                setStep('review');
              }}
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
        // Sticky so partial-failure errors stay visible during scroll.
        <Alert
          componentId="admin.edit_role_modal.error"
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

      {step === 'edit' ? (
        <>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
            Update name, description, permissions, mutation conditions, and assigned users. Changes are previewed before
            they're applied.
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
                {/* ``key={String(open)}`` forces a fresh mount on every
                    reopen so the section's internal ``draft`` can't bleed
                    across close → reopen. */}
                <RolePermissionsSection
                  key={String(open)}
                  value={permissions}
                  onChange={setPermissions}
                  workspace={resourcePickerWorkspace}
                  disabled={submitting}
                  onUnsavedDraftChange={setHasUnsavedDraft}
                />
              </LongFormSection>
              <LongFormSection title="Mutation conditions">
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  Current mutation conditions are pre-filled. Remove a row to drop it; use the form below to add more.
                </Typography.Text>
                <MutationConditionsSection
                  key={String(open)}
                  value={conditions}
                  onChange={setConditions}
                  workspace={resourcePickerWorkspace}
                  disabled={submitting}
                  onUnsavedDraftChange={setHasUnsavedConditionDraft}
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
        <ReviewSummary diff={diff} permissionByIdLabel={permissionByIdLabel} conditionByIdLabel={conditionByIdLabel} />
      )}
      <ConfirmationModal
        componentId="admin.edit_role_modal.discard_unsaved_draft"
        title="Discard unsaved entry?"
        visible={showDiscardConfirm}
        message="You started adding a permission or mutation condition to this role but didn't click Add. Continuing to Review changes will discard it. Go back to either click Add to stage it, or Clear to drop the draft on the spot."
        okText="Continue"
        cancelText="Back"
        danger={false}
        onCancel={() => setShowDiscardConfirm(false)}
        onConfirm={() => {
          setShowDiscardConfirm(false);
          setError(null);
          setStep('review');
        }}
      />
    </Modal>
  );
};

const ReviewSummary = ({
  diff,
  permissionByIdLabel,
  conditionByIdLabel,
}: {
  diff: RoleDiff;
  permissionByIdLabel: Map<number, string>;
  conditionByIdLabel: Map<number, string>;
}) => {
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
      <DiffGroup
        title="Mutation conditions to add"
        items={diff.conditionsToAdd.map(formatStagedCondition)}
        emptyLabel="No new mutation conditions."
        addColor
      />
      <DiffGroup
        title="Mutation conditions to remove"
        items={diff.conditionIdsToRemove.map((id) => conditionByIdLabel.get(id) ?? `condition #${id}`)}
        emptyLabel="No mutation conditions to remove."
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
