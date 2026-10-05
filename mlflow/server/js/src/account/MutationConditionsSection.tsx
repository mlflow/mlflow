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
import type { UserRoleConditionRow } from './types';
import { formatResourcePattern, isSyntheticUserRole } from './types';

interface Props {
  /** Every condition on every role the user holds, as returned by the self endpoint. */
  conditions: UserRoleConditionRow[];
  isLoading?: boolean;
  /** Non-fatal - surfaces inline so any rows we have still render. */
  error?: unknown;
  componentId: string;
  /** When false, the Workspace column is hidden. Defaults to true. */
  workspacesEnabled?: boolean;
}

/**
 * The conditions that restrict the signed-in user.
 *
 * Conditions subtract from what grants allow and never confer access, so this sits
 * beside Permissions rather than replacing anything in it: a user needs both halves to
 * understand a refusal - the grant that let the operation through, and the condition
 * that then refused it.
 *
 * INFORMATIONAL ONLY. A grant can be pre-evaluated, which is why the admin UI can grey
 * out controls; a value condition cannot, because its verdict depends on the values in
 * the request. Nothing here should gate a control.
 *
 * Rows are NOT deduped, unlike ``PermissionsSection``. Two identical-looking conditions
 * on two roles are two independent restrictions that must BOTH pass, so collapsing them
 * would misrepresent the semantics - and the scope columns are what distinguish them.
 */
export const MutationConditionsSection = ({
  conditions,
  isLoading,
  error,
  componentId,
  workspacesEnabled = true,
}: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const directLabel = intl.formatMessage({
    defaultMessage: 'Direct',
    description: 'Source label for a mutation condition attached directly to the user (not via a role)',
  });

  const rows = useMemo(
    () =>
      [...conditions].sort((a, b) => {
        if (a.workspace !== b.workspace) return a.workspace.localeCompare(b.workspace);
        if (a.resource_type !== b.resource_type) return a.resource_type.localeCompare(b.resource_type);
        return (a.condition_slot ?? 0) - (b.condition_slot ?? 0);
      }),
    [conditions],
  );

  /** ``*`` in a container of ``workspace`` is the no-narrowing default, so say nothing. */
  const describeScope = (row: UserRoleConditionRow) => {
    const parts: string[] = [];
    if (row.resource_pattern !== '*') {
      parts.push(`${row.resource_type}:${formatResourcePattern(row.resource_pattern)}`);
    }
    if (row.container_resource_type !== 'workspace' || row.container_resource_pattern !== '*') {
      parts.push(`in ${row.container_resource_type}:${formatResourcePattern(row.container_resource_pattern)}`);
    }
    return parts.join(' ');
  };

  return (
    <>
      <Typography.Paragraph color="secondary">
        <FormattedMessage
          defaultMessage="Conditions further restrict what your permissions allow. They never grant access, and every condition that applies must pass. A condition on the values you set is only checked when you submit a change."
          description="Explanatory text above the account mutation conditions table"
        />
      </Typography.Paragraph>
      {error ? (
        <Alert
          componentId={`${componentId}.conditions_error`}
          type="warning"
          message={intl.formatMessage({
            defaultMessage: 'Failed to load conditions',
            description: 'Alert title shown when the mutation conditions query fails',
          })}
          description={
            (error as Error)?.message ||
            intl.formatMessage({
              defaultMessage: 'Conditions for your account are temporarily unavailable.',
              description: 'Alert description shown when mutation conditions failed to load',
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
                  defaultMessage: 'No conditions',
                  description: 'Empty-state title for the account mutation conditions table',
                })}
                description={intl.formatMessage({
                  defaultMessage: 'No conditions restrict your account. Your permissions apply in full.',
                  description: 'Empty-state description for the account mutation conditions table',
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
            <TableHeader componentId={`${componentId}.conditions.resource_header`} css={{ flex: 1 }}>
              <FormattedMessage
                defaultMessage="Resource type"
                description="Conditions table column header for the resource type"
              />
            </TableHeader>
            {workspacesEnabled && (
              <TableHeader componentId={`${componentId}.conditions.workspace_header`} css={{ flex: 1 }}>
                <FormattedMessage
                  defaultMessage="Workspace"
                  description="Conditions table column header for the workspace"
                />
              </TableHeader>
            )}
            <TableHeader componentId={`${componentId}.conditions.values_header`} css={{ flex: 2 }}>
              <FormattedMessage
                defaultMessage="Values you may set"
                description="Conditions table column header for the value condition"
              />
            </TableHeader>
            <TableHeader componentId={`${componentId}.conditions.targets_header`} css={{ flex: 2 }}>
              <FormattedMessage
                defaultMessage="Resources you may change"
                description="Conditions table column header for the target condition"
              />
            </TableHeader>
            <TableHeader componentId={`${componentId}.conditions.scope_header`} css={{ flex: 1 }}>
              <FormattedMessage
                defaultMessage="Applies to"
                description="Conditions table column header for the condition's scope"
              />
            </TableHeader>
            <TableHeader componentId={`${componentId}.conditions.source_header`} css={{ flex: 1 }}>
              <FormattedMessage
                defaultMessage="Source"
                description="Conditions table column header for the source (role name or 'Direct')"
              />
            </TableHeader>
          </TableRow>
          {rows.map((row) => {
            const scope = describeScope(row);
            return (
              <TableRow key={`${row.role_id}:${row.resource_type}:${row.condition_slot ?? 0}`}>
                <TableCell css={{ flex: 1 }}>
                  <code>{row.resource_type}</code>
                </TableCell>
                {workspacesEnabled && <TableCell css={{ flex: 1 }}>{row.workspace}</TableCell>}
                <TableCell css={{ flex: 2 }}>
                  {row.value_condition ? (
                    <code>{row.value_condition}</code>
                  ) : (
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="Unrestricted"
                        description="Shown when a condition does not constrain the values being set"
                      />
                    </Typography.Text>
                  )}
                </TableCell>
                <TableCell css={{ flex: 2 }}>
                  {row.target_condition ? (
                    <code>{row.target_condition}</code>
                  ) : (
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="Unrestricted"
                        description="Shown when a condition does not constrain which resources may be changed"
                      />
                    </Typography.Text>
                  )}
                </TableCell>
                <TableCell css={{ flex: 1 }}>
                  {scope ? (
                    <code>{scope}</code>
                  ) : (
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="All"
                        description="Shown when a condition is not narrowed to a resource or container"
                      />
                    </Typography.Text>
                  )}
                </TableCell>
                <TableCell css={{ flex: 1 }}>
                  {isSyntheticUserRole(row.role_name) ? directLabel : row.role_name}
                </TableCell>
              </TableRow>
            );
          })}
        </Table>
      )}
    </>
  );
};
