import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  ChevronLeftIcon,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Switch,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { FieldLabel } from './FieldLabel';
import { LongFormSection } from '../../common/components/long-form/LongFormSection';
import { ConfirmationModal } from '../ConfirmationModal';
import { AdminApi } from '../api';
import {
  AdminQueryKeys,
  useCurrentUserIsAdmin,
  useGrantUserPermission,
  useRevokeUserPermission,
  useRolesQuery,
  useUserRolesQuery,
  useUsersQuery,
  useWorkspaceOptions,
} from '../hooks';
import { AccountQueryKeys } from '../../account/hooks';
import { DEFAULT_WORKSPACE_NAME, isSyntheticUserRole } from '../types';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { useWorkspaces } from '../../workspaces/hooks/useWorkspaces';
import { useWorkspacesEnabled } from '../../experiment-tracking/hooks/useServerInfo';
import { RoleAssignmentForm, ROLE_ASSIGNMENT_DEFAULT, type RoleAssignmentValue } from './RoleAssignmentForm';
import { DIRECT_GRANT_RESOURCE_TYPES, type DirectGrantResourceType } from './DirectPermissionForm';
import { DirectPermissionsSection, type StagedDirectPermission } from './DirectPermissionsSection';

export interface EditAccessModalProps {
  open: boolean;
  onClose: () => void;
  username: string;
}

const directPermKey = (p: { resourceType: string; resourceId: string; permission: string }) =>
  `${p.resourceType}::${p.resourceId}::${p.permission}`;

// Derived from the form's source of truth so a new direct-grant type can't
// drift between the form (where it's offered) and the modal (where existing
// grants of that type are bucketed into the direct view).
const isDirectGrantResourceType = (rt: string): rt is DirectGrantResourceType =>
  (DIRECT_GRANT_RESOURCE_TYPES as readonly string[]).includes(rt);

interface AccessDiff {
  rolesToAssign: number[];
  rolesToUnassign: number[];
  directToGrant: StagedDirectPermission[];
  directToRevoke: StagedDirectPermission[];
  adminChange: boolean;
}

/**
 * Edit-style modal for managing one user's access. Pre-fills role
 * assignments, direct permissions, and admin status from the current
 * backend state, then computes a diff on submit and applies it. The
 * Review step surfaces every add / remove / promote / demote so the
 * admin can confirm before destructive parts land.
 */
export const EditAccessModal = ({ open, onClose, username }: EditAccessModalProps) => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const grantPermission = useGrantUserPermission();
  const revokePermission = useRevokeUserPermission();
  const isCurrentUserAdmin = useCurrentUserIsAdmin();
  const activeWorkspace = useActiveWorkspace();

  // Platform-admin-only workspace selector for direct-grant targeting.
  // Workspace managers stay locked to their session-active workspace.
  const { workspacesEnabled } = useWorkspacesEnabled();
  const showWorkspaceSelector = isCurrentUserAdmin && workspacesEnabled;
  const { workspaces } = useWorkspaces(showWorkspaceSelector);
  const initialGrantWorkspace = activeWorkspace ?? DEFAULT_WORKSPACE_NAME;
  const [grantWorkspace, setGrantWorkspace] = useState<string>(initialGrantWorkspace);

  // --- Current state from backend (used to pre-fill + compute diff) ---
  const { data: rolesData, isLoading: rolesLoading, error: rolesError } = useUserRolesQuery(username);
  const { data: usersData, isLoading: usersLoading } = useUsersQuery();
  // Roles list for the Review step's name lookup (the form uses the
  // dropdown's own label, but the Review step renders by id). Platform
  // admins fetch unscoped; workspace managers pass the active workspace.
  // Suppress when none is active to avoid a guaranteed 403.
  const rolesListWorkspace = isCurrentUserAdmin ? undefined : (activeWorkspace ?? undefined);
  const rolesListEnabled = isCurrentUserAdmin || Boolean(activeWorkspace);
  const { data: rolesListData } = useRolesQuery(rolesListWorkspace, { enabled: rolesListEnabled });

  // Synthetic ``__user_N__`` roles anchor direct grants — must not enter the
  // editable set, or the picker's "Clear" would unassign them and orphan
  // every direct grant.
  const currentRoleIds = useMemo<number[]>(
    () => (rolesData?.roles ?? []).filter((r) => !isSyntheticUserRole(r.name)).map((r) => r.id),
    [rolesData],
  );
  // Direct grants live on the synthetic ``__user_<id>__`` role surfaced by
  // ``useUserRolesQuery``; flatten its nested ``permissions`` to recover the
  // editable list (custom roles are shown in the Roles tab).
  const currentDirectPerms = useMemo<StagedDirectPermission[]>(
    () =>
      (rolesData?.roles ?? [])
        .filter((r) => isSyntheticUserRole(r.name))
        .flatMap((r) => r.permissions ?? [])
        .filter((p) => isDirectGrantResourceType(p.resource_type))
        .map((p) => ({
          resourceType: p.resource_type as DirectGrantResourceType,
          resourceId: p.resource_pattern,
          permission: p.permission,
        })),
    [rolesData],
  );
  const currentIsAdmin = useMemo(
    () => Boolean(usersData?.users?.find((u) => u.username === username)?.is_admin),
    [usersData, username],
  );

  // --- Editable state ---
  const [step, setStep] = useState<'edit' | 'review'>('edit');
  const [roleValue, setRoleValue] = useState<RoleAssignmentValue>(ROLE_ASSIGNMENT_DEFAULT);
  const [directPermissions, setDirectPermissions] = useState<StagedDirectPermission[]>([]);
  const [isAdmin, setIsAdmin] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Reported by ``DirectPermissionsSection`` whenever the in-progress
  // draft is dirty (any field touched away from default). Drives a
  // discard-confirm dialog on the ``Review changes`` button so the admin
  // can't silently abandon a partially filled permission — but the button
  // itself stays enabled and the admin can always click through.
  const [hasUnsavedDirectDraft, setHasUnsavedDirectDraft] = useState(false);
  const [showDiscardConfirm, setShowDiscardConfirm] = useState(false);

  const workspaceOptions = useWorkspaceOptions(workspaces);

  const stateLoaded = !rolesLoading && !usersLoading;

  // ``filledForWorkspaceRef`` tracks which workspace's data was last pre-filled
  // into editable state. The pre-fill effect re-runs when this stops matching
  // the current ``grantWorkspace`` so switching the dropdown mid-edit re-seeds
  // ``directPermissions`` from the newly-selected workspace's permissions
  // (otherwise revoking a pre-filled row would target the wrong workspace).
  const filledForWorkspaceRef = useRef<string | null>(null);

  // Reset transient UI state only on open — refetches must not bounce
  // the user back to edit or wipe a partial-failure error.
  useEffect(() => {
    if (!open) {
      return;
    }
    setStep('edit');
    setSubmitting(false);
    setError(null);
    setGrantWorkspace(initialGrantWorkspace);
    // ``hasUnsavedDirectDraft`` isn't reset here — the ``key={String(open)}``
    // on ``DirectPermissionsSection`` below remounts the section on every
    // open, and its first commit-time effect fires ``false`` from the
    // default draft state.
    setShowDiscardConfirm(false);
    filledForWorkspaceRef.current = null;
    // ``initialGrantWorkspace`` is derived from the session active workspace;
    // re-seed on open so the dropdown defaults to "where I am right now".
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Pre-fill editable fields after backing queries resolve. Re-runs when
  // ``grantWorkspace`` changes (so switching the dropdown mid-edit reloads
  // the pre-filled rows from the newly-selected workspace), but skips when
  // ``filledForWorkspaceRef`` already matches — so background refetches in
  // the *same* workspace don't clobber in-progress edits.
  useEffect(() => {
    if (!open) {
      filledForWorkspaceRef.current = null;
      return;
    }
    if (!stateLoaded || filledForWorkspaceRef.current === grantWorkspace) {
      return;
    }
    if (filledForWorkspaceRef.current === null) {
      setRoleValue({ roleIds: [...currentRoleIds] });
      setIsAdmin(currentIsAdmin);
    }
    setDirectPermissions([...currentDirectPerms]);
    filledForWorkspaceRef.current = grantWorkspace;
  }, [open, stateLoaded, grantWorkspace, currentRoleIds, currentDirectPerms, currentIsAdmin]);

  // --- Diff computation ---
  const diff = useMemo<AccessDiff>(() => {
    const currentRoleIdSet = new Set(currentRoleIds);
    const desiredRoleIdSet = new Set(roleValue.roleIds);
    const rolesToAssign = roleValue.roleIds.filter((id) => !currentRoleIdSet.has(id));
    const rolesToUnassign = currentRoleIds.filter((id) => !desiredRoleIdSet.has(id));

    const currentDirectKeys = new Set(currentDirectPerms.map(directPermKey));
    const desiredDirectKeys = new Set(directPermissions.map(directPermKey));
    const directToGrant = directPermissions.filter((p) => !currentDirectKeys.has(directPermKey(p)));
    const directToRevoke = currentDirectPerms.filter((p) => !desiredDirectKeys.has(directPermKey(p)));

    const adminChange = isCurrentUserAdmin && isAdmin !== currentIsAdmin;

    return { rolesToAssign, rolesToUnassign, directToGrant, directToRevoke, adminChange };
  }, [
    currentRoleIds,
    currentDirectPerms,
    currentIsAdmin,
    roleValue.roleIds,
    directPermissions,
    isAdmin,
    isCurrentUserAdmin,
  ]);

  const hasAnyChange =
    diff.rolesToAssign.length > 0 ||
    diff.rolesToUnassign.length > 0 ||
    diff.directToGrant.length > 0 ||
    diff.directToRevoke.length > 0 ||
    diff.adminChange;

  const roleNameById = useMemo(() => {
    const map = new Map<number, { name: string; workspace: string }>();
    for (const r of rolesListData?.roles ?? []) {
      map.set(r.id, { name: r.name, workspace: r.workspace });
    }
    // Also fold in the user's currently-assigned roles (they may live in a
    // workspace the all-roles list omitted).
    for (const r of rolesData?.roles ?? []) {
      if (!map.has(r.id)) {
        map.set(r.id, { name: r.name, workspace: r.workspace });
      }
    }
    return map;
  }, [rolesListData, rolesData]);

  const renderRoleId = useCallback(
    (id: number) => {
      const entry = roleNameById.get(id);
      return entry ? `${entry.workspace}/${entry.name}` : `role #${id}`;
    },
    [roleNameById],
  );

  const handleConfirm = useCallback(async () => {
    // ``error`` is already cleared by the "Review changes" transition;
    // skipping a redundant reset here also avoids the flicker.
    setSubmitting(true);
    const failures: string[] = [];

    // 1. Admin status (do this first so a failed promotion shows up before
    // any role changes that may depend on the new privilege).
    if (diff.adminChange) {
      try {
        await AdminApi.updateAdmin({ username, is_admin: isAdmin });
        queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
      } catch (e: any) {
        failures.push(`${isAdmin ? 'Granting' : 'Revoking'} admin status failed: ${e?.message ?? 'unknown error'}`);
      }
    }

    // 2. Role assignments (assign new + unassign removed).
    const roleIdsTouched = new Set<number>();
    for (const roleId of diff.rolesToAssign) {
      roleIdsTouched.add(roleId);
      try {
        await AdminApi.assignRole(username, roleId);
      } catch (e: any) {
        failures.push(`Assigning ${renderRoleId(roleId)} failed: ${e?.message ?? 'unknown error'}`);
      }
    }
    for (const roleId of diff.rolesToUnassign) {
      roleIdsTouched.add(roleId);
      try {
        await AdminApi.unassignRole(username, roleId);
      } catch (e: any) {
        failures.push(`Unassigning ${renderRoleId(roleId)} failed: ${e?.message ?? 'unknown error'}`);
      }
    }
    if (roleIdsTouched.size > 0) {
      queryClient.invalidateQueries({ queryKey: AccountQueryKeys.userRoles(username) });
      // The Admin Users tab eager-loads each user's roles via
      // ``useUsersQuery``; invalidate so the per-row Roles cell refreshes.
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
      for (const roleId of roleIdsTouched) {
        queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleUsers(roleId) });
      }
    }

    // 3. Direct permissions (grant new + revoke removed).
    for (const p of diff.directToGrant) {
      try {
        await grantPermission.mutateAsync({
          resource_type: p.resourceType,
          resource_id: p.resourceId,
          username,
          permission: p.permission,
          workspace: grantWorkspace,
        });
      } catch (e: any) {
        failures.push(
          `Granting ${p.resourceType}:${p.resourceId} → ${p.permission} failed: ${e?.message ?? 'unknown error'}`,
        );
      }
    }
    for (const p of diff.directToRevoke) {
      try {
        await revokePermission.mutateAsync({
          resource_type: p.resourceType,
          resource_id: p.resourceId,
          username,
          workspace: grantWorkspace,
        });
      } catch (e: any) {
        failures.push(
          `Revoking ${p.resourceType}:${p.resourceId} (${p.permission}) failed: ${e?.message ?? 'unknown error'}`,
        );
      }
    }

    if (failures.length === 0) {
      onClose();
      return;
    }
    setError(failures.join('\n'));
    filledForWorkspaceRef.current = null;
    setStep('edit');
    setSubmitting(false);
  }, [diff, isAdmin, username, grantWorkspace, queryClient, grantPermission, revokePermission, onClose, renderRoleId]);

  return (
    <Modal
      componentId="admin.edit_access_modal"
      title={`Edit access for ${username}`}
      visible={open}
      onCancel={onClose}
      size="wide"
      footer={
        step === 'edit' ? (
          <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
            <Button componentId="admin.edit_access_modal.cancel" onClick={onClose} disabled={submitting}>
              Cancel
            </Button>
            <Button
              componentId="admin.edit_access_modal.review"
              type="primary"
              // Submit isn't blocked on an unsaved draft — instead we gate
              // on it via a discard-confirm dialog so the admin can either
              // go back and click Add, or knowingly drop the draft and
              // proceed to the review step.
              onClick={() => {
                if (hasUnsavedDirectDraft) {
                  setShowDiscardConfirm(true);
                  return;
                }
                setError(null);
                setStep('review');
              }}
              disabled={!hasAnyChange || !stateLoaded || Boolean(rolesError)}
            >
              Review changes
            </Button>
          </div>
        ) : (
          <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
            <Button
              componentId="admin.edit_access_modal.back"
              type="tertiary"
              icon={<ChevronLeftIcon />}
              onClick={() => setStep('edit')}
              disabled={submitting}
            >
              Back
            </Button>
            <Button
              componentId="admin.edit_access_modal.confirm"
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
          componentId="admin.edit_access_modal.error"
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
            Assign or unassign roles, grant or revoke direct permissions, and toggle admin status. Changes are previewed
            before they're applied.
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
          ) : rolesError ? (
            // Block the form on a failed roles fetch so the empty pre-fill
            // doesn't masquerade as the user's actual access.
            <Alert
              componentId="admin.edit_access_modal.roles_error"
              type="error"
              message="Failed to load access state"
              description={
                (rolesError instanceof Error ? rolesError.message : null) ||
                `An error occurred while fetching the current access for ${username}. Close the modal and try again.`
              }
            />
          ) : (
            <>
              <LongFormSection title="Role assignments">
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  Pick the roles <strong>{username}</strong> should have. Removing a role unassigns it on submit.
                </Typography.Text>
                <RoleAssignmentForm value={roleValue} onChange={setRoleValue} disabled={submitting} />
              </LongFormSection>
              <LongFormSection title="Direct permissions" hideDivider={!isCurrentUserAdmin}>
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  Current direct permissions are pre-filled. Remove a row to revoke; use the form below to grant more.
                </Typography.Text>
                {showWorkspaceSelector && (
                  <div css={{ marginBottom: theme.spacing.md }}>
                    <FieldLabel>Workspace</FieldLabel>
                    <SimpleSelect
                      id="admin-edit-access-modal-grant-workspace"
                      componentId="admin.edit_access_modal.grant_workspace"
                      value={grantWorkspace}
                      onChange={({ target }) => setGrantWorkspace(target.value)}
                      disabled={submitting}
                    >
                      {workspaceOptions.map((w) => (
                        <SimpleSelectOption key={w} value={w}>
                          {w}
                        </SimpleSelectOption>
                      ))}
                    </SimpleSelect>
                    <Typography.Text
                      color="secondary"
                      size="sm"
                      css={{ display: 'block', marginTop: theme.spacing.xs }}
                    >
                      Grants and revokes target this workspace's per-user direct-grant role. Pick a different workspace
                      to grant access there.
                    </Typography.Text>
                  </div>
                )}
                {/* ``key={String(open)}`` forces a fresh mount each time
                    the modal re-opens so the section's internal ``draft``
                    state can't bleed across close → reopen and re-block
                    Review with a phantom previous-session draft. The pre-
                    fill effect re-seeds ``directPermissions`` from the
                    backend on its own schedule, but the in-progress draft
                    is owned by the section and needs a remount to reset. */}
                <DirectPermissionsSection
                  key={String(open)}
                  value={directPermissions}
                  onChange={setDirectPermissions}
                  workspace={grantWorkspace}
                  disabled={submitting}
                  onUnsavedDraftChange={setHasUnsavedDirectDraft}
                />
              </LongFormSection>
              {isCurrentUserAdmin && (
                <LongFormSection title="Admin status" hideDivider>
                  <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                    Admins can manage all users, roles, and workspaces.
                  </Typography.Text>
                  <Switch
                    componentId="admin.edit_access_modal.is_admin"
                    checked={isAdmin}
                    onChange={setIsAdmin}
                    label="This user is an admin"
                    disabled={submitting}
                  />
                </LongFormSection>
              )}
            </>
          )}
        </>
      ) : (
        <ReviewSummary
          username={username}
          diff={diff}
          renderRoleId={renderRoleId}
          isAdmin={isAdmin}
          currentIsAdmin={currentIsAdmin}
        />
      )}
      {/* Discard-confirm gate on ``Review changes``: only intercepts when
          ``hasUnsavedDirectDraft`` is true (any field touched in the
          direct-grant picker without a subsequent ``Add`` or ``Clear``).
          The dialog is the warning surface; the button itself stays
          enabled so the admin can always click through. */}
      <ConfirmationModal
        componentId="admin.edit_access_modal.discard_unsaved_draft"
        title="Discard unsaved direct permission?"
        visible={showDiscardConfirm}
        message="You started adding a direct permission but didn't click Add. Continuing to Review changes will discard it. Go back to either click Add to stage it, or Clear to drop the draft on the spot."
        okText="Continue"
        cancelText="Back"
        // ``danger=false`` because the OK verb is neutral ("Continue") — the
        // destructive intent is in the title question, not the button.
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
  username,
  diff,
  renderRoleId,
  isAdmin,
  currentIsAdmin,
}: {
  username: string;
  diff: AccessDiff;
  renderRoleId: (id: number) => string;
  isAdmin: boolean;
  currentIsAdmin: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const renderDirect = (p: StagedDirectPermission) => `${p.resourceType}:${p.resourceId} → ${p.permission}`;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Text color="secondary">
        Review the net changes for <strong>{username}</strong>. Click <strong>Apply changes</strong> to commit them, or{' '}
        <strong>Back</strong> to keep editing.
      </Typography.Text>

      <DiffGroup
        title="Roles to assign"
        items={diff.rolesToAssign.map(renderRoleId)}
        emptyLabel="No new role assignments."
        addColor
      />
      <DiffGroup
        title="Roles to unassign"
        items={diff.rolesToUnassign.map(renderRoleId)}
        emptyLabel="No role unassignments."
      />
      <DiffGroup
        title="Direct permissions to grant"
        items={diff.directToGrant.map(renderDirect)}
        emptyLabel="No new direct permissions."
        addColor
      />
      <DiffGroup
        title="Direct permissions to revoke"
        items={diff.directToRevoke.map(renderDirect)}
        emptyLabel="No direct permissions to revoke."
      />
      {diff.adminChange ? (
        <DiffGroup
          title="Admin status"
          items={[isAdmin ? `Promote to admin (was ${currentIsAdmin ? 'admin' : 'not admin'})` : 'Revoke admin']}
          addColor={isAdmin}
        />
      ) : (
        <DiffGroup title="Admin status" items={[]} emptyLabel="No change to admin status." />
      )}
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
              componentId="admin.edit_access_modal.diff_tag"
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
