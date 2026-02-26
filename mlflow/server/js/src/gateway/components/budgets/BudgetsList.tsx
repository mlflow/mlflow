import { useMemo, useState } from 'react';
import {
  Button,
  CreditCardIcon,
  Empty,
  Input,
  PencilIcon,
  SearchIcon,
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
import { BudgetsFilterButton, type BudgetsFilter } from './BudgetsFilterButton';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import type { BudgetPolicy, BudgetUnit, DurationUnit, BudgetAction, TargetScope } from '../../types';

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

function formatDuration(value: number, type: DurationUnit): string {
  const typeLabels: Record<DurationUnit, string> = {
    MINUTES: value === 1 ? 'Minute' : 'Minutes',
    HOURS: value === 1 ? 'Hour' : 'Hours',
    DAYS: value === 1 ? 'Day' : 'Days',
    MONTHS: value === 1 ? 'Month' : 'Months',
  };
  return `${value} ${typeLabels[type]}`;
}

function formatOnExceeded(action: BudgetAction): string {
  const labels: Record<BudgetAction, string> = {
    ALERT: 'Alert',
    REJECT: 'Reject',
  };
  return labels[action];
}

function formatTargetScope(type: TargetScope): string {
  return type === 'GLOBAL' ? 'Global' : 'Workspace';
}

export const BudgetsList = ({ onEditClick, onDeleteClick }: BudgetsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const [searchFilter, setSearchFilter] = useState('');
  const [filter, setFilter] = useState<BudgetsFilter>({ scopes: [] });

  const { data: budgetPolicies, isLoading } = useBudgetPoliciesQuery();

  const availableScopes = useMemo(() => {
    if (!budgetPolicies) return [];
    return Array.from(new Set(budgetPolicies.map((p) => p.target_scope)));
  }, [budgetPolicies]);

  const filteredPolicies = useMemo(() => {
    if (!budgetPolicies) return [];
    let filtered = budgetPolicies;

    if (searchFilter.trim()) {
      const lowerFilter = searchFilter.toLowerCase();
      filtered = filtered.filter((policy) => policy.budget_policy_id.toLowerCase().includes(lowerFilter));
    }

    if (filter.scopes.length > 0) {
      filtered = filtered.filter((policy) => filter.scopes.includes(policy.target_scope));
    }

    return filtered;
  }, [budgetPolicies, searchFilter, filter]);

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

  const isFiltered = searchFilter.trim().length > 0 || filter.scopes.length > 0;

  const getEmptyState = () => {
    if (filteredPolicies.length === 0 && isFiltered) {
      return (
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No budget policies found"
              description="Empty state title when filter returns no budget results"
            />
          }
          description={null}
        />
      );
    }
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
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Input
          componentId="mlflow.gateway.budgets-list.search"
          prefix={<SearchIcon />}
          placeholder={formatMessage({
            defaultMessage: 'Search budget policies',
            description: 'Placeholder for budget policy search filter',
          })}
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          allowClear
          css={{ maxWidth: 300 }}
        />
        <BudgetsFilterButton availableScopes={availableScopes} filter={filter} onFilterChange={setFilter} />
      </div>

      <Table
        scrollable
        empty={getEmptyState()}
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.general.borderRadiusBase,
        }}
      >
        <TableRow isHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.id-header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Policy ID" description="Budget policy ID column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.limit-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Budget" description="Budget amount column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.duration-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Duration" description="Budget duration column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.scope-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Scope" description="Budget scope column header" />
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
        {filteredPolicies.map((policy) => (
          <TableRow key={policy.budget_policy_id}>
            <TableCell css={{ flex: 2 }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <CreditCardIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
                <Typography.Text bold>{policy.budget_policy_id}</Typography.Text>
              </div>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatBudgetAmount(policy.budget_amount, policy.budget_unit)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatDuration(policy.duration_value, policy.duration_unit)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatTargetScope(policy.target_scope)}</Typography.Text>
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
    </div>
  );
};
