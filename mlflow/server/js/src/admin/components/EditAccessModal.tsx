import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  ChevronLeftIcon,
  Modal,
  Spinner,
  Switch,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { LongFormSection } from '../../common/components/long-form/LongFormSection';
import { AdminApi } from '../api';
import {
  AdminQueryKeys,
  useCurrentUserIsAdmin,
  useGrantUserPermission,
  useRevokeUserPermission,
  useRolesQuery,
  useUserPermissionsQuery,
  useUserRolesQuery,
  useUsersQuery,
} from '../hooks';
import { AccountQueryKeys } from '../../account/hooks';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { RoleAssignmentForm, ROLE_ASSIGNMENT_DEFAULT, type RoleAssignmentValue } from './RoleAssignmentForm';
import { type DirectGrantResourceType } from './DirectPermissionForm';
import { DirectPermissionsSection, type StagedDirectPermission } from './DirectPermissionsSection';

export interface EditAccessModalProps {
  open: boolean;
  onClose: () => void;
  username: string;
  /** Optional: bridge to the Create Role flow when "All <type>" is picked. */
  onCreateRoleForAllOfType?: (resourceType: DirectGrantResourceType) => void;
}

const directPermKey = (p: { resourceType: string; resourceId: string; permission: string }) =>
  `${p.resourceType}::${p.resourceId}::${p.permission}`;

const isDirectGrantResourceType = (rt: string): rt is DirectGrantResourceType =>
  rt === 'experiment' ||
  rt === 'registered_model' ||
  rt === 'gateway_secret' ||
  rt === 'gateway_endpoint' ||
  rt === 'gateway_model_definition';

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
export const EditAccessModal = ({ open, onClose, username, onCreateRoleForAllOfType }: EditAccessModalProps) => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const grantPermission = useGrantUserPermission();
  const revokePermission = useRevokeUserPermission();
  const isCurrentUserAdmin = useCurrentUserIsAdmin();

  // --- Current state from backend (used to pre-fill + compute diff) ---
  const { data: rolesData, isLoading: rolesLoading } = useUserRolesQuery(username);
  const { data: directPermsData, isLoading: directPermsLoading } = useUserPermissionsQuery(username);
  const { data: usersData, isLoading: usersLoading } = useUsersQuery();
  // Roles list for the Review step's name lookup (the form uses the
  // dropdown's own label, but the Review step renders by id). Platform
  // admins fetch unscoped; workspace managers pass the active workspace.
  // Suppress when none is active to avoid a guaranteed 403.
  const activeWorkspace = useActiveWorkspace();
  const rolesListWorkspace = isCurrentUserAdmin ? undefined : (activeWorkspace ?? undefined);
  const rolesListEnabled = isCurrentUserAdmin || Boolean(activeWorkspace);
  const { data: rolesListData } = useRolesQuery(rolesListWorkspace, { enabled: rolesListEnabled });

  const currentRoleIds = useMemo<number[]>(() => (rolesData?.roles ?? []).map((r) => r.id), [rolesData]);
  const currentDirectPerms = useMemo<StagedDirectPermission[]>(
    () =>
      (directPermsData?.permissions ?? [])
        .filter((p) => isDirectGrantResourceType(p.resource_type))
        .map((p) => ({
          resourceType: p.resource_type as DirectGrantResourceType,
          resourceId: p.resource_pattern,
          permission: p.permission,
        })),
    [directPermsData],
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

  const stateLoaded = !rolesLoading && !directPermsLoading && !usersLoading;

  // Re-pre-fill whenever the modal opens or the backing data resolves.
  useEffect(() => {
    if (!open) {
      return;
    }
    setStep('edit');
    setRoleValue({ roleIds: [...currentRoleIds] });
    setDirectPermissions([...currentDirectPerms]);
    setIsAdmin(currentIsAdmin);
    setSubmitting(false);
    setError(null);
  }, [open, currentRoleIds, currentDirectPerms, currentIsAdmin]);

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
    setError(null);
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
    setStep('edit');
    setSubmitting(false);
  }, [diff, isAdmin, username, queryClient, grantPermission, revokePermission, onClose, renderRoleId]);

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
              onClick={() => setStep('review')}
              disabled={!hasAnyChange || !stateLoaded}
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
        <Alert
          componentId="admin.edit_access_modal.error"
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
                <DirectPermissionsSection
                  value={directPermissions}
                  onChange={setDirectPermissions}
                  onCreateRoleForAllOfType={onCreateRoleForAllOfType}
                  disabled={submitting}
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
