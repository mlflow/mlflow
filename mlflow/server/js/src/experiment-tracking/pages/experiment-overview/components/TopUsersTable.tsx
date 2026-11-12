import React, { useMemo } from 'react';
import { UserIcon, Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useTopUsersData, type UserUsageData } from '../hooks/useTopUsersData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
} from './OverviewChartComponents';
import { formatCount, formatLatency, useChartColors } from '../utils/chartUtils';
import { useSortState, useSummaryTableStyles, SortableHeader, LinkableNameCell } from './SummaryTableComponents';

type SortColumn = keyof UserUsageData;

const costFormatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });

// Table cells show cents; sub-cent totals are shown as "<$0.01" rather than rounding to $0.00
const formatTableCost = (cost: number) => (cost > 0 && cost < 0.01 ? '<$0.01' : costFormatter.format(cost));

interface TopUsersTableProps {
  /** Called with the user's value when a row's name is clicked, e.g. to apply the User filter */
  onSelectUser?: (user: string) => void;
}

/**
 * Top Users table for the Usage tab. Shows per-user traces, sessions, tokens, cost, average
 * latency and error rate, grouped by the `mlflow.trace.user` trace metadata.
 */
export const TopUsersTable: React.FC<TopUsersTableProps> = ({ onSelectUser }) => {
  const { getChartColor } = useChartColors();
  const { sortColumn, sortDirection, handleSort } = useSortState<SortColumn>('traces');
  const { headerRowStyle, bodyRowStyle, cellStyle } = useSummaryTableStyles(
    'minmax(120px, 3fr) 1fr 1fr 1fr 1fr 1fr 1fr',
  );

  const { usersData, isLoading, error, hasData } = useTopUsersData();

  const sortedUsersData = useMemo(() => {
    return [...usersData].sort((a, b) => {
      const comparison =
        sortColumn === 'user' ? a.user.localeCompare(b.user) : (a[sortColumn] as number) - (b[sortColumn] as number);
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [usersData, sortColumn, sortDirection]);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  const sortableHeaderProps = { sortColumn, sortDirection, onSort: handleSort };

  return (
    <OverviewChartContainer componentId="mlflow.charts.top_users_table">
      <OverviewChartHeader
        icon={<UserIcon />}
        title={<FormattedMessage defaultMessage="Top Users" description="Title for the top users table" />}
        value={hasData ? formatCount(usersData.length) : undefined}
      />

      {hasData ? (
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          <div css={headerRowStyle}>
            <SortableHeader column="user" {...sortableHeaderProps}>
              <FormattedMessage defaultMessage="User" description="Top users table > column header for user" />
            </SortableHeader>
            <SortableHeader column="traces" {...sortableHeaderProps} centered>
              <FormattedMessage defaultMessage="Traces" description="Top users table > column header for trace count" />
            </SortableHeader>
            <SortableHeader column="sessions" {...sortableHeaderProps} centered>
              <FormattedMessage
                defaultMessage="Sessions"
                description="Top users table > column header for session count"
              />
            </SortableHeader>
            <SortableHeader column="totalTokens" {...sortableHeaderProps} centered>
              <FormattedMessage
                defaultMessage="Tokens"
                description="Top users table > column header for total tokens"
              />
            </SortableHeader>
            <SortableHeader column="totalCost" {...sortableHeaderProps} centered>
              <FormattedMessage defaultMessage="Cost" description="Top users table > column header for total cost" />
            </SortableHeader>
            <SortableHeader column="avgLatency" {...sortableHeaderProps} centered>
              <FormattedMessage
                defaultMessage="Latency (AVG)"
                description="Top users table > column header for average latency"
              />
            </SortableHeader>
            <SortableHeader column="errorRate" {...sortableHeaderProps} centered>
              <FormattedMessage defaultMessage="Errors" description="Top users table > column header for error rate" />
            </SortableHeader>
          </div>

          <div css={{ maxHeight: 320, overflowY: 'auto' }}>
            {sortedUsersData.map((userData) => {
              const colorIndex = usersData.findIndex((u) => u.user === userData.user);
              return (
                <div key={userData.user} css={bodyRowStyle}>
                  <LinkableNameCell
                    name={userData.user}
                    color={getChartColor(colorIndex)}
                    onClick={onSelectUser ? () => onSelectUser(userData.user) : undefined}
                  />
                  <Typography.Text css={cellStyle}>{formatCount(userData.traces)}</Typography.Text>
                  <Typography.Text css={cellStyle}>{formatCount(userData.sessions)}</Typography.Text>
                  <Typography.Text css={cellStyle}>{formatCount(userData.totalTokens)}</Typography.Text>
                  <Typography.Text css={cellStyle}>{formatTableCost(userData.totalCost)}</Typography.Text>
                  <Typography.Text css={cellStyle}>{formatLatency(userData.avgLatency)}</Typography.Text>
                  <Typography.Text css={cellStyle}>{userData.errorRate.toFixed(2)}%</Typography.Text>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <OverviewChartEmptyState />
      )}
    </OverviewChartContainer>
  );
};
