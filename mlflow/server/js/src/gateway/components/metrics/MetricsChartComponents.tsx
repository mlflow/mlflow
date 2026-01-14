import React from 'react';
import { TableSkeleton, TitleSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

const DEFAULT_CHART_HEIGHT = 280;

interface MetricsChartHeaderProps {
  icon: React.ReactNode;
  title: React.ReactNode;
  value?: React.ReactNode;
  subtitle?: React.ReactNode;
}

export const MetricsChartHeader: React.FC<MetricsChartHeaderProps> = ({ icon, title, value, subtitle }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ marginBottom: theme.spacing.lg }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <span css={{ color: theme.colors.textSecondary, display: 'flex' }}>{icon}</span>
        <Typography.Text bold size="lg">
          {title}
        </Typography.Text>
      </div>
      {value !== undefined && (
        <Typography.Title level={3} css={{ margin: 0, marginTop: theme.spacing.sm }}>
          {value}
          {subtitle && (
            <>
              {' '}
              <Typography.Text color="secondary" css={{ fontWeight: 'normal' }}>
                {subtitle}
              </Typography.Text>
            </>
          )}
        </Typography.Title>
      )}
    </div>
  );
};

export const MetricsChartTimeLabel: React.FC = () => {
  return (
    <Typography.Text color="secondary" size="sm" css={{ textTransform: 'uppercase', letterSpacing: '0.5px' }}>
      <FormattedMessage defaultMessage="Over time" description="Label above time-series charts" />
    </Typography.Text>
  );
};

interface MetricsChartCardProps {
  children: React.ReactNode;
  height?: number;
}

export const MetricsChartCard: React.FC<MetricsChartCardProps> = ({ children, height = DEFAULT_CHART_HEIGHT }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.lg,
        height,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {children}
    </div>
  );
};

interface MetricsChartLoadingStateProps {
  height?: number;
}

export const MetricsChartLoadingState: React.FC<MetricsChartLoadingStateProps> = ({
  height = DEFAULT_CHART_HEIGHT,
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.lg,
        height,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
      <TitleSkeleton css={{ width: 120 }} />
      <TitleSkeleton css={{ width: 80 }} />
      <div css={{ flex: 1 }}>
        <TableSkeleton lines={4} />
      </div>
    </div>
  );
};

interface MetricsChartErrorStateProps {
  height?: number;
  message?: React.ReactNode;
}

export const MetricsChartErrorState: React.FC<MetricsChartErrorStateProps> = ({ height, message }) => {
  return (
    <MetricsChartCard height={height}>
      <Typography.Text color="error">
        {message || (
          <FormattedMessage
            defaultMessage="Failed to load chart data"
            description="Error message when chart fails to load"
          />
        )}
      </Typography.Text>
    </MetricsChartCard>
  );
};

interface MetricsChartEmptyStateProps {
  height?: number;
  message?: React.ReactNode;
}

export const MetricsChartEmptyState: React.FC<MetricsChartEmptyStateProps> = ({ height, message }) => {
  return (
    <div
      css={{
        height: height || '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Typography.Text color="secondary">
        {message || (
          <FormattedMessage
            defaultMessage="No data available for the selected time range"
            description="Message shown when there is no data to display in the chart"
          />
        )}
      </Typography.Text>
    </div>
  );
};

export function useChartTooltipStyle() {
  const { theme } = useDesignSystemTheme();
  return {
    backgroundColor: theme.colors.backgroundPrimary,
    border: `1px solid ${theme.colors.border}`,
    borderRadius: theme.borders.borderRadiusMd,
    fontSize: theme.typography.fontSizeSm,
  };
}

export function useChartXAxisProps() {
  const { theme } = useDesignSystemTheme();
  return {
    tick: { fontSize: 10, fill: theme.colors.textSecondary, dy: theme.spacing.sm },
    axisLine: false,
    tickLine: false,
    interval: 'preserveStartEnd' as const,
  };
}

export function useChartLegendFormatter() {
  const { theme } = useDesignSystemTheme();
  return (value: string) => (
    <span
      style={{
        color: theme.colors.textPrimary,
        fontSize: theme.typography.fontSizeSm,
        cursor: 'pointer',
      }}
    >
      {value}
    </span>
  );
}

interface MetricsChartContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export const MetricsChartContainer: React.FC<MetricsChartContainerProps> = ({ children, ...rest }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
      {...rest}
    >
      {children}
    </div>
  );
};

export function formatCount(count: number): string {
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(2)}M`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toFixed(2)}K`;
  }
  return count.toLocaleString();
}

const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_DAY = 86400;

export function formatTimestamp(timestampMs: number, bucketSizeSeconds: number): string {
  const date = new Date(timestampMs);
  // Show hour for buckets smaller than a day
  if (bucketSizeSeconds < SECONDS_PER_DAY) {
    return date.toLocaleString([], { month: 'numeric', day: 'numeric', hour: '2-digit' });
  }
  return date.toLocaleString([], { month: 'numeric', day: 'numeric' });
}
