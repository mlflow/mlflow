import {
  Alert,
  Empty,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { formatConditionScope, getResourceTypeLabel } from '../types';
import type { MutationCondition } from '../types';

export interface ConditionsTableProps {
  conditions: MutationCondition[];
  isLoading?: boolean;
  error?: unknown;
  /** Rendered when there are no conditions and nothing failed. */
  emptyDescription?: string;
  /** Optional per-row trailing label, used by the user tab to name the carrying role. */
  rowSuffix?: (condition: MutationCondition) => string | undefined;
  /** Header for the ``rowSuffix`` column. Omit to hide the column. */
  suffixHeader?: string;
}

/**
 * Read-only view of mutation conditions. Shared by the role and user detail tabs so the
 * two list the same columns in the same order.
 */
export const ConditionsTable = ({
  conditions,
  isLoading,
  error,
  emptyDescription,
  rowSuffix,
  suffixHeader,
}: ConditionsTableProps) => {
  const { theme } = useDesignSystemTheme();

  if (isLoading) {
    return (
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
    );
  }

  // A failed fetch and an empty list look identical and mean opposite things: "nothing
  // restricts this" versus "we do not know".
  if (error) {
    return (
      <Alert
        componentId="admin.conditions_table.fetch_error"
        type="error"
        message="Failed to load mutation conditions"
        description={(error as Error)?.message || 'An error occurred while fetching mutation conditions.'}
      />
    );
  }

  const emptyState =
    conditions.length === 0 ? <Empty title="No mutation conditions" description={emptyDescription} /> : null;

  return (
    <Table
      scrollable
      noMinHeight
      empty={emptyState}
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        overflow: 'hidden',
      }}
    >
      <TableRow isHeader>
        <TableHeader componentId="admin.conditions_table.type_header" css={{ flex: 1 }}>
          Resource Type
        </TableHeader>
        <TableHeader componentId="admin.conditions_table.scope_header" css={{ flex: 1 }}>
          Scope
        </TableHeader>
        <TableHeader componentId="admin.conditions_table.request_header" css={{ flex: 2 }}>
          Request Filter
        </TableHeader>
        <TableHeader componentId="admin.conditions_table.resource_header" css={{ flex: 2 }}>
          Resource Filter
        </TableHeader>
        {suffixHeader && (
          <TableHeader componentId="admin.conditions_table.suffix_header" css={{ flex: 1 }}>
            {suffixHeader}
          </TableHeader>
        )}
      </TableRow>
      {conditions.map((condition) => (
        <TableRow key={`${condition.role_id}-${condition.id}`}>
          <TableCell css={{ flex: 1 }}>
            <Tag componentId="admin.conditions_table.type_tag">{getResourceTypeLabel(condition.resource_type)}</Tag>
          </TableCell>
          <TableCell css={{ flex: 1 }}>
            <Typography.Text size="sm">{formatConditionScope(condition)}</Typography.Text>
          </TableCell>
          <TableCell css={{ flex: 2 }}>
            {condition.value_condition ? (
              <code>{condition.value_condition}</code>
            ) : (
              <Typography.Text color="secondary" size="sm">
                —
              </Typography.Text>
            )}
          </TableCell>
          <TableCell css={{ flex: 2 }}>
            {condition.target_condition ? (
              <code>{condition.target_condition}</code>
            ) : (
              <Typography.Text color="secondary" size="sm">
                —
              </Typography.Text>
            )}
          </TableCell>
          {suffixHeader && (
            <TableCell css={{ flex: 1 }}>
              <Typography.Text size="sm">{rowSuffix?.(condition) ?? ''}</Typography.Text>
            </TableCell>
          )}
        </TableRow>
      ))}
    </Table>
  );
};
