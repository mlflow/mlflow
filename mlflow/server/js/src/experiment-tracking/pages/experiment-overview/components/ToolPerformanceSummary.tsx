import React, { useMemo, useState } from 'react';
import {
  WrenchIcon,
  Typography,
  useDesignSystemTheme,
  SortAscendingIcon,
  SortDescendingIcon,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useToolPerformanceSummaryData } from '../hooks/useToolPerformanceSummaryData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
} from './OverviewChartComponents';
import { formatCount, formatLatency, useChartColors } from '../utils/chartUtils';

type SortColumn = 'toolName' | 'totalCalls' | 'successRate' | 'avgLatency';
type SortDirection = 'asc' | 'desc';

/**
 * Tool Performance Summary component displaying per-tool metrics in a table-like view.
 * Shows tool name, call count, success rate, and average latency for each tool.
 */
export const ToolPerformanceSummary: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const { getChartColor } = useChartColors();
  const [sortColumn, setSortColumn] = useState<SortColumn>('totalCalls');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  // Fetch tool performance data
  const { toolsData, isLoading, error, hasData } = useToolPerformanceSummaryData();

  // Sort the data
  const sortedToolsData = useMemo(() => {
    if (!toolsData.length) return toolsData;

    return [...toolsData].sort((a, b) => {
      let comparison = 0;
      switch (sortColumn) {
        case 'toolName':
          comparison = a.toolName.localeCompare(b.toolName);
          break;
        case 'totalCalls':
          comparison = a.totalCalls - b.totalCalls;
          break;
        case 'successRate':
          comparison = a.successRate - b.successRate;
          break;
        case 'avgLatency':
          comparison = a.avgLatency - b.avgLatency;
          break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [toolsData, sortColumn, sortDirection]);

  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  // Common styles
  const rowStyle = {
    display: 'grid',
    gridTemplateColumns: 'minmax(80px, 2fr) 1fr 1fr 1fr',
    gap: theme.spacing.lg,
    borderBottom: `1px solid ${theme.colors.border}`,
  } as const;

  const cellStyle = { textAlign: 'center' } as const;

  // Sortable header cell component
  const SortableHeader = ({
    column,
    children,
    centered,
  }: {
    column: SortColumn;
    children: React.ReactNode;
    centered?: boolean;
  }) => (
    <div
      role="button"
      tabIndex={0}
      onClick={() => handleSort(column)}
      onKeyDown={(e) => e.key === 'Enter' && handleSort(column)}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        cursor: 'pointer',
        justifyContent: centered ? 'center' : 'flex-start',
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
        fontWeight: 600,
        '&:hover': { color: theme.colors.textPrimary },
      }}
    >
      {children}
      {sortColumn === column && (sortDirection === 'asc' ? <SortAscendingIcon /> : <SortDescendingIcon />)}
    </div>
  );

  return (
    <OverviewChartContainer componentId="mlflow.charts.tool_performance_summary">
      <OverviewChartHeader
        icon={<WrenchIcon />}
        title={
          <FormattedMessage
            defaultMessage="Tool Performance Summary"
            description="Title for the tool performance summary section"
          />
        }
      />

      {hasData ? (
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          {/* Table header */}
          <div css={{ ...rowStyle, padding: `${theme.spacing.sm}px ${theme.spacing.lg}px ${theme.spacing.sm}px 0` }}>
            <SortableHeader column="toolName">
              <FormattedMessage defaultMessage="Tool" description="Column header for tool name" />
            </SortableHeader>
            <SortableHeader column="totalCalls" centered>
              <FormattedMessage defaultMessage="Calls" description="Column header for call count" />
            </SortableHeader>
            <SortableHeader column="successRate" centered>
              <FormattedMessage defaultMessage="Success" description="Column header for success rate" />
            </SortableHeader>
            <SortableHeader column="avgLatency" centered>
              <FormattedMessage defaultMessage="Latency (AVG)" description="Column header for average latency" />
            </SortableHeader>
          </div>

          {/* Scrollable table body */}
          <div css={{ maxHeight: 300, overflowY: 'auto' }}>
            {sortedToolsData.map((tool, index) => (
              <div
                key={tool.toolName}
                css={{
                  ...rowStyle,
                  padding: `${theme.spacing.md}px ${theme.spacing.lg}px ${theme.spacing.md}px 0`,
                  alignItems: 'center',
                  '&:last-child': { borderBottom: 'none' },
                }}
              >
                {/* Tool name with color indicator */}
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <div
                    css={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      backgroundColor: getChartColor(index),
                      flexShrink: 0,
                    }}
                  />
                  <Typography.Text
                    css={{
                      fontFamily: 'monospace',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {tool.toolName}
                  </Typography.Text>
                </div>

                <Typography.Text css={cellStyle}>{formatCount(tool.totalCalls)}</Typography.Text>
                <Typography.Text css={cellStyle}>{tool.successRate.toFixed(2)}%</Typography.Text>
                <Typography.Text css={cellStyle}>{formatLatency(tool.avgLatency)}</Typography.Text>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <OverviewChartEmptyState />
      )}
    </OverviewChartContainer>
  );
};
