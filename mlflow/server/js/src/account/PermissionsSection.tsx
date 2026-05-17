import { useMemo } from 'react';
import {
  Alert,
  Empty,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { DirectPermission, Role } from './types';
import { formatResourcePattern } from './types';

interface Props {
  roles: Role[];
  directPermissions: DirectPermission[];
  isLoading?: boolean;
  /** Non-fatal - surfaces inline so direct grants still render. */
  rolesError?: unknown;
  /** Non-fatal - surfaces inline so role-derived rows still render. */
  directPermsError?: unknown;
  /** Scopes the component IDs emitted by this section. */
  componentId: string;
  /** When false, the Workspace column is hidden. Defaults to true. */
  workspacesEnabled?: boolean;
}

interface Row {
  workspace: string | null;
  resource_type: string;
  resource_pattern: string;
  permission: string;
  sources: Set<string>;
}

const rowKey = (workspace: string | null, resource_type: string, resource_pattern: string, permission: string) =>
  JSON.stringify([workspace, resource_type, resource_pattern, permission]);

/**
 * Deduped union of role-derived grants and direct per-resource grants
 * for one user. Workspace is part of the dedup key - the same
 * ``(type, pattern, permission)`` granted in two workspaces stays as
 * two rows.
 */
export const PermissionsSection = ({
  roles,
  directPermissions,
  isLoading,
  rolesError,
  directPermsError,
  componentId,
  workspacesEnabled = true,
}: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const rows = useMemo<Row[]>(() => {
    const byKey = new Map<string, Row>();
    const upsert = (
      workspace: string | null,
      resource_type: string,
      resource_pattern: string,
      permission: string,
      source: string,
    ) => {
      const key = rowKey(workspace, resource_type, resource_pattern, permission);
      const existing = byKey.get(key);
      if (existing) {
        existing.sources.add(source);
      } else {
        byKey.set(key, { workspace, resource_type, resource_pattern, permission, sources: new Set([source]) });
      }
    };
    for (const role of roles) {
      for (const p of role.permissions ?? []) {
        upsert(role.workspace, p.resource_type, p.resource_pattern, p.permission, role.name);
      }
    }
    const directLabel = intl.formatMessage({
      defaultMessage: 'Direct',
      description: 'Source label for a per-resource permission granted directly to the user (not via a role)',
    });
    for (const p of directPermissions) {
      upsert(p.workspace, p.resource_type, p.resource_pattern, p.permission, directLabel);
    }
    return Array.from(byKey.values()).sort((a, b) => {
      const aw = a.workspace ?? '';
      const bw = b.workspace ?? '';
      if (aw !== bw) return aw.localeCompare(bw);
      if (a.resource_type !== b.resource_type) return a.resource_type.localeCompare(b.resource_type);
      if (a.resource_pattern !== b.resource_pattern) return a.resource_pattern.localeCompare(b.resource_pattern);
      return a.permission.localeCompare(b.permission);
    });
  }, [roles, directPermissions, intl]);

  return (
    <>
      {rolesError ? (
        <Alert
          componentId={`${componentId}.roles_error`}
          type="warning"
          message={intl.formatMessage({
            defaultMessage: 'Failed to load role-derived permissions',
            description: 'Alert title shown when the roles query fails on the permissions section',
          })}
          description={
            (rolesError as Error)?.message ||
            intl.formatMessage({
              defaultMessage: 'Showing direct grants only - role-derived permissions are unavailable.',
              description:
                'Alert description shown when only direct grants are available because role-derived permissions failed to load',
            })
          }
        />
      ) : null}
      {directPermsError ? (
        <Alert
          componentId={`${componentId}.direct_permissions_error`}
          type="warning"
          message={intl.formatMessage({
            defaultMessage: 'Failed to load direct permissions',
            description: 'Alert title shown when the direct-permissions query fails',
          })}
          description={
            (directPermsError as Error)?.message ||
            intl.formatMessage({
              defaultMessage: 'Showing role-derived permissions only - direct grants are unavailable.',
              description:
                'Alert description shown when only role-derived permissions are available because direct grants failed to load',
            })
          }
        />
      ) : null}
      {isLoading ? (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: theme.spacing.sm,
            padding: theme.spacing.lg,
            minHeight: 200,
          }}
        >
          <Spinner size="small" />
        </div>
      ) : (
        <Table
          scrollable
          noMinHeight
          empty={
            rows.length === 0 ? (
              <Empty
                title={intl.formatMessage({
                  defaultMessage: 'No permissions',
                  description: 'Empty-state title for the permissions table',
                })}
                description={intl.formatMessage({
                  defaultMessage: 'No resource permissions to show.',
                  description: 'Empty-state description for the permissions table',
                })}
              />
            ) : null
          }
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.general.borderRadiusBase,
            overflow: 'hidden',
          }}
        >
          <TableRow isHeader>
            <TableHeader componentId={`${componentId}.permissions.resource_header`} css={{ flex: 2 }}>
              <FormattedMessage
                defaultMessage="Resource"
                description="Permissions table column header for the resource"
              />
            </TableHeader>
            {workspacesEnabled && (
              <TableHeader componentId={`${componentId}.permissions.workspace_header`} css={{ flex: 1 }}>
                <FormattedMessage
                  defaultMessage="Workspace"
                  description="Permissions table column header for the workspace"
                />
              </TableHeader>
            )}
            <TableHeader componentId={`${componentId}.permissions.permission_header`} css={{ flex: 1 }}>
              <FormattedMessage
                defaultMessage="Permission"
                description="Permissions table column header for the permission level"
              />
            </TableHeader>
            <TableHeader componentId={`${componentId}.permissions.source_header`} css={{ flex: 1 }}>
              <FormattedMessage
                defaultMessage="Source"
                description="Permissions table column header for the source (role name or 'Direct')"
              />
            </TableHeader>
          </TableRow>
          {rows.map((row) => (
            <TableRow key={rowKey(row.workspace, row.resource_type, row.resource_pattern, row.permission)}>
              <TableCell css={{ flex: 2 }}>
                <code>
                  {row.resource_type}:{formatResourcePattern(row.resource_pattern)}
                </code>
              </TableCell>
              {workspacesEnabled && (
                <TableCell css={{ flex: 1 }}>
                  {row.workspace ?? <Typography.Text color="secondary">-</Typography.Text>}
                </TableCell>
              )}
              <TableCell css={{ flex: 1 }}>
                <strong>{row.permission}</strong>
              </TableCell>
              <TableCell css={{ flex: 1 }}>{Array.from(row.sources).sort().join(', ')}</TableCell>
            </TableRow>
          ))}
        </Table>
      )}
    </>
  );
};
