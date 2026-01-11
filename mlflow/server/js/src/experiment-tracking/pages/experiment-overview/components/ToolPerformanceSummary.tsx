import React from 'react';
import { WrenchIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
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

/**
 * Tool Performance Summary component displaying per-tool metrics in a table-like view.
 * Shows tool name, call count, success rate, and average latency for each tool.
 */
export const ToolPerformanceSummary: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const { getChartColor } = useChartColors();

  // Fetch tool performance data
  const { toolsData, isLoading, error, hasData } = useToolPerformanceSummaryData();

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  // Common styles
  const rowStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr auto auto auto',
    gap: theme.spacing.lg,
    borderBottom: `1px solid ${theme.colors.border}`,
  } as const;

  const cellStyle = { textAlign: 'right', minWidth: 70 } as const;

  return (
    <OverviewChartContainer>
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
          <div css={{ ...rowStyle, padding: `${theme.spacing.sm}px 0` }}>
            <Typography.Text color="secondary" size="sm" bold>
              <FormattedMessage defaultMessage="Tool" description="Column header for tool name" />
            </Typography.Text>
            <Typography.Text color="secondary" size="sm" bold css={cellStyle}>
              <FormattedMessage defaultMessage="Calls" description="Column header for call count" />
            </Typography.Text>
            <Typography.Text color="secondary" size="sm" bold css={cellStyle}>
              <FormattedMessage defaultMessage="Success" description="Column header for success rate" />
            </Typography.Text>
            <Typography.Text color="secondary" size="sm" bold css={cellStyle}>
              <FormattedMessage defaultMessage="Latency" description="Column header for average latency" />
            </Typography.Text>
          </div>

          {/* Table rows */}
          {toolsData.map((tool, index) => (
            <div
              key={tool.toolName}
              css={{
                ...rowStyle,
                padding: `${theme.spacing.md}px 0`,
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
      ) : (
        <OverviewChartEmptyState />
      )}
    </OverviewChartContainer>
  );
};
