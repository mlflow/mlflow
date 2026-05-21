import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useWorkspacesEnabled } from '../../experiment-tracking/hooks/useServerInfo';
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
 * direct-grant bookkeeping, not real role assignments. When the server is
 * in single-tenant mode (workspaces disabled), the workspace prefix is
 * dropped since every role lives in ``default``.
 */
export const UserRolesCell = ({ roles: allRoles, scopeWorkspace }: UserRolesCellProps) => {
  const { theme } = useDesignSystemTheme();
  const { workspacesEnabled, loading } = useWorkspacesEnabled();
  // While the server-info query is in-flight, default to the multi-tenant
  // layout so the cell doesn't flicker from name-only to prefixed once the
  // value settles. Single-tenant servers briefly show the prefix on a cold
  // load, then drop it.
  const showWorkspacePrefix = loading || workspacesEnabled;
  const visibleRoles = allRoles.filter((r) => !isSyntheticUserRole(r.name));
  const roles = scopeWorkspace ? visibleRoles.filter((r) => r.workspace === scopeWorkspace) : visibleRoles;
  if (roles.length === 0) {
    return <Typography.Text color="secondary">—</Typography.Text>;
  }
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs / 2 }}>
      {roles.map((role) => (
        <Typography.Text key={role.id} size="sm">
          {showWorkspacePrefix ? (
            <>
              <code>{role.workspace}</code> → {role.name}
            </>
          ) : (
            role.name
          )}
        </Typography.Text>
      ))}
    </div>
  );
};
