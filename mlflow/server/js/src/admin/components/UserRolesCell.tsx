import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { isSyntheticUserRole, type Role } from '../types';

export interface UserRolesCellProps {
  roles: readonly Role[];
  scopeWorkspace: string | null;
}

/**
 * Per-row Roles cell on the admin Users tab. Reads roles from the user
 * object eager-loaded by ``useUsersQuery`` — no per-row request. The
 * backend already scopes the list per requester (admin sees all; workspace
 * managers see only roles in workspaces they administer).
 * ``scopeWorkspace`` further narrows to the active workspace for the
 * per-workspace admin page, so the cell matches the page scope.
 * Synthetic ``__user_N__`` roles are filtered out — they're per-user
 * direct-grant bookkeeping, not real role assignments.
 */
export const UserRolesCell = ({ roles: allRoles, scopeWorkspace }: UserRolesCellProps) => {
  const { theme } = useDesignSystemTheme();
  const visibleRoles = allRoles.filter((r) => !isSyntheticUserRole(r.name));
  const roles = scopeWorkspace ? visibleRoles.filter((r) => r.workspace === scopeWorkspace) : visibleRoles;
  if (roles.length === 0) {
    return <Typography.Text color="secondary">—</Typography.Text>;
  }
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs / 2 }}>
      {roles.map((role) => (
        <Typography.Text key={role.id} size="sm">
          <code>{role.workspace}</code> → {role.name}
        </Typography.Text>
      ))}
    </div>
  );
};
