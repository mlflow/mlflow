import { useState } from 'react';
import {
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  CreditCardIcon,
  Empty,
  PencilIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tooltip,
  TrashIcon,
  Typography,
  WarningFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useBudgetPoliciesQuery } from '../../hooks/useBudgetPoliciesQuery';
import { useBudgetWindowsQuery } from '../../hooks/useBudgetWindowsQuery';
import { formatBudgetAmount, formatDuration, formatOnExceeded } from './budgetFormatUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import type { BudgetPolicy } from '../../types';

const PAGE_SIZE = 10;

interface BudgetsListProps {
  onEditClick?: (policy: BudgetPolicy) => void;
  onDeleteClick?: (policy: BudgetPolicy) => void;
}

export const BudgetsList = ({ onEditClick, onDeleteClick }: BudgetsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage, formatDate } = useIntl();
  const [pageToken, setPageToken] = useState<string | undefined>(undefined);
  const [pageTokenHistory, setPageTokenHistory] = useState<string[]>([]);

  const { data: budgetPolicies, nextPageToken, isLoading } = useBudgetPoliciesQuery(PAGE_SIZE, pageToken);
  const { data: budgetWindows } = useBudgetWindowsQuery();

  const handleNextPage = () => {
    if (nextPageToken) {
      setPageTokenHistory((prev) => [...prev, pageToken ?? '']);
      setPageToken(nextPageToken);
    }
  };

  const handlePreviousPage = () => {
    setPageTokenHistory((prev) => {
      const newHistory = [...prev];
      const prevToken = newHistory.pop();
      setPageToken(prevToken || undefined);
      return newHistory;
    });
  };

  const currentPageIndex = pageTokenHistory.length + 1;
  const hasPreviousPage = pageTokenHistory.length > 0;
  const hasNextPage = !!nextPageToken;

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
    if (budgetPolicies.length === 0 && !hasPreviousPage) {
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
        noMinHeight
        empty={getEmptyState()}
        css={{
          borderLeft: `1px solid ${theme.colors.border}`,
          borderRight: `1px solid ${theme.colors.border}`,
          borderTop: `1px solid ${theme.colors.border}`,
          borderBottom: budgetPolicies.length === 0 ? `1px solid ${theme.colors.border}` : 'none',
          borderRadius: theme.general.borderRadiusBase,
          overflow: 'hidden',
        }}
      >
        <TableRow isHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.limit-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Budget" description="Budget amount column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.duration-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Reset period" description="Budget reset period column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.action-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="On Exceeded" description="Budget on exceeded column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.window-start-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Window Start" description="Budget window start date column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.window-end-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Window End" description="Budget window end date column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.current-spend-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Current Spend" description="Budget current spend column header" />
          </TableHeader>
          <TableHeader componentId="mlflow.gateway.budgets-list.updated-header" css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Last updated" description="Last updated column header" />
          </TableHeader>
          <TableHeader
            componentId="mlflow.gateway.budgets-list.actions-header"
            css={{ flex: 0, minWidth: 96, maxWidth: 96 }}
          />
        </TableRow>
        {budgetPolicies.map((policy: BudgetPolicy) => {
          const window = budgetWindows[policy.budget_policy_id];
          return (
            <TableRow key={policy.budget_policy_id}>
              <TableCell css={{ flex: 1 }}>
                <Tooltip
                  componentId="mlflow.gateway.budgets-list.budget-amount-tooltip"
                  content={formatBudgetAmount(policy.budget_amount, policy.budget_unit, 6)}
                >
                  <span>
                    <Typography.Text>{formatBudgetAmount(policy.budget_amount, policy.budget_unit)}</Typography.Text>
                  </span>
                </Tooltip>
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <Typography.Text>{formatDuration(policy.duration_value, policy.duration_unit)}</Typography.Text>
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <Typography.Text>{formatOnExceeded(policy.budget_action)}</Typography.Text>
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                {window ? (
                  <TimeAgo date={new Date(window.window_start_ms)} />
                ) : (
                  <Typography.Text color="secondary">—</Typography.Text>
                )}
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                {window ? (
                  <Tooltip
                    componentId="mlflow.gateway.budgets-list.window-end-tooltip"
                    content={formatDate(window.window_end_ms, {
                      dateStyle: 'full',
                      timeStyle: 'short',
                    })}
                  >
                    <span>
                      <Typography.Text>
                        {formatDate(window.window_end_ms, {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric',
                        })}
                      </Typography.Text>
                    </span>
                  </Tooltip>
                ) : (
                  <Typography.Text color="secondary">—</Typography.Text>
                )}
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                {window ? (
                  (() => {
                    const isBudgetExceeded = window.current_spend >= policy.budget_amount;
                    return (
                      <Tooltip
                        componentId="mlflow.gateway.budgets-list.current-spend-tooltip"
                        content={formatBudgetAmount(window.current_spend, policy.budget_unit, 6)}
                      >
                        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                          <Typography.Text color={isBudgetExceeded ? 'error' : undefined}>
                            {formatBudgetAmount(window.current_spend, policy.budget_unit)}
                          </Typography.Text>
                          {isBudgetExceeded && (
                            <Tooltip
                              componentId="mlflow.gateway.budgets-list.budget-exceeded-tooltip"
                              content={formatMessage(
                                {
                                  defaultMessage: 'Budget exceeded: {spend} of {limit} spent',
                                  description: 'Tooltip shown when current spend exceeds the budget limit',
                                },
                                {
                                  spend: formatBudgetAmount(window.current_spend, policy.budget_unit, 6),
                                  limit: formatBudgetAmount(policy.budget_amount, policy.budget_unit, 6),
                                },
                              )}
                            >
                              <WarningFillIcon
                                aria-label={formatMessage({
                                  defaultMessage: 'Budget exceeded',
                                  description: 'Warning icon label for exceeded budget',
                                })}
                                css={{
                                  fontSize: theme.typography.fontSizeSm,
                                  color: theme.colors.textValidationDanger,
                                }}
                              />
                            </Tooltip>
                          )}
                        </div>
                      </Tooltip>
                    );
                  })()
                ) : (
                  <Typography.Text color="secondary">—</Typography.Text>
                )}
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
          );
        })}
      </Table>
      {(hasPreviousPage || hasNextPage) && (
        <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.gateway.budgets-list.previous-page"
            type="tertiary"
            icon={<ChevronLeftIcon />}
            disabled={!hasPreviousPage}
            onClick={handlePreviousPage}
          >
            <FormattedMessage defaultMessage="Previous" description="Previous page button" />
          </Button>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Page {page}"
              description="Current page indicator"
              values={{ page: currentPageIndex }}
            />
          </Typography.Text>
          <Button
            componentId="mlflow.gateway.budgets-list.next-page"
            type="tertiary"
            endIcon={<ChevronRightIcon />}
            disabled={!hasNextPage}
            onClick={handleNextPage}
          >
            <FormattedMessage defaultMessage="Next" description="Next page button" />
          </Button>
        </div>
      )}
    </div>
  );
};
