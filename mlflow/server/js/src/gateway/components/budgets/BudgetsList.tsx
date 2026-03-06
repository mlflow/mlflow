import { useMemo, useState } from 'react';
import {
  Button,
  CreditCardIcon,
  Empty,
  Pagination,
  PencilIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useBudgetPoliciesQuery } from '../../hooks/useBudgetPoliciesQuery';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import type { BudgetPolicy, BudgetUnit, DurationUnit, BudgetAction } from '../../types';

const PAGE_SIZE = 10;

interface BudgetsListProps {
  onEditClick?: (policy: BudgetPolicy) => void;
  onDeleteClick?: (policy: BudgetPolicy) => void;
}

function formatBudgetAmount(amount: number, budgetUnit: BudgetUnit): string {
  if (budgetUnit === 'USD') {
    return `$${amount.toFixed(2)}`;
  }
  return `${amount}`;
}

function formatDuration(value: number, unit: DurationUnit): string {
  if (value === 1) {
    const friendlyLabels: Partial<Record<DurationUnit, string>> = {
      DAYS: 'Daily',
      WEEKS: 'Weekly',
      MONTHS: 'Monthly',
    };
    if (friendlyLabels[unit]) return friendlyLabels[unit]!;
  }
  const typeLabels: Record<DurationUnit, string> = {
    MINUTES: value === 1 ? 'Minute' : 'Minutes',
    HOURS: value === 1 ? 'Hour' : 'Hours',
    DAYS: value === 1 ? 'Day' : 'Days',
    WEEKS: value === 1 ? 'Week' : 'Weeks',
    MONTHS: value === 1 ? 'Month' : 'Months',
  };
  return `${value} ${typeLabels[unit] ?? unit}`;
}

function formatOnExceeded(action: BudgetAction): string {
  const labels: Record<BudgetAction, string> = {
    ALERT: 'Alert',
    REJECT: 'Reject',
  };
  return labels[action];
}

export const BudgetsList = ({ onEditClick, onDeleteClick }: BudgetsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const [currentPage, setCurrentPage] = useState(1);

  const { data: budgetPolicies, isLoading } = useBudgetPoliciesQuery();

  const pagedPolicies = useMemo(() => {
    const start = (currentPage - 1) * PAGE_SIZE;
    return budgetPolicies.slice(start, start + PAGE_SIZE);
  }, [budgetPolicies, currentPage]);

  if (isLoading) {
    return (
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
        <FormattedMessage defaultMessage="Loading budget policies..." description="Loading message for budgets list" />
      </div>
    );
  }

  const getEmptyState = () => {
    if (budgetPolicies.length === 0) {
      return (
        <Empty
          image={<CreditCardIcon />}
          title={
            <FormattedMessage
              defaultMessage="No budget policies created"
              description="Empty state title for budgets list"
            />
          }
          description={
            <FormattedMessage
              defaultMessage='Use "Create budget policy" button to set up cost limits for the AI Gateway'
              description="Empty state message for budgets list"
            />
          }
        />
      );
    }
    return null;
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Table
        scrollable
        empty={getEmptyState()}
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.general.borderRadiusBase,
        }}
      >
        <TableRow isHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.limit-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Budget" description="Budget amount column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.duration-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Duration" description="Budget duration column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.action-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="On Exceeded" description="Budget on exceeded column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.updated-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Last updated" description="Last updated column header" />
          </TableHeader>
          <TableHeader
            componentId="mlflow.gateway.budgets-list.actions-header"
            css={{ flex: 0, minWidth: 96, maxWidth: 96 }}
          />
        </TableRow>
        {pagedPolicies.map((policy) => (
          <TableRow key={policy.budget_policy_id}>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatBudgetAmount(policy.budget_amount, policy.budget_unit)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatDuration(policy.duration_value, policy.duration_unit)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatOnExceeded(policy.budget_action)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <TimeAgo date={new Date(policy.last_updated_at)} />
            </TableCell>
            <TableCell css={{ flex: 0, minWidth: 96, maxWidth: 96 }}>
              <div css={{ display: 'flex', gap: theme.spacing.xs }}>
                <Button
                  componentId="mlflow.gateway.budgets-list.edit-button"
                  type="primary"
                  icon={<PencilIcon />}
                  aria-label={formatMessage({
                    defaultMessage: 'Edit budget policy',
                    description: 'Gateway > Budgets list > Edit budget policy button aria label',
                  })}
                  onClick={() => onEditClick?.(policy)}
                />
                <Button
                  componentId="mlflow.gateway.budgets-list.delete-button"
                  type="primary"
                  icon={<TrashIcon />}
                  aria-label={formatMessage({
                    defaultMessage: 'Delete budget policy',
                    description: 'Gateway > Budgets list > Delete budget policy button aria label',
                  })}
                  onClick={() => onDeleteClick?.(policy)}
                />
              </div>
            </TableCell>
          </TableRow>
        ))}
      </Table>
      {budgetPolicies.length > PAGE_SIZE && (
        <div css={{ display: 'flex', justifyContent: 'center' }}>
          <Pagination
            componentId="mlflow.gateway.budgets-list.pagination"
            currentPageIndex={currentPage}
            numTotal={budgetPolicies.length}
            pageSize={PAGE_SIZE}
            onChange={setCurrentPage}
          />
        </div>
      )}
    </div>
  );
};
