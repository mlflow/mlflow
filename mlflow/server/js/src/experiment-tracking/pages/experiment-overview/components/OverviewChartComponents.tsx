import React from 'react';
import { TableSkeleton, TitleSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

const DEFAULT_CHART_HEIGHT = 280;

interface OverviewChartHeaderProps {
  /** Icon component to display before the title */
  icon: React.ReactNode;
  /** Chart title */
  title: React.ReactNode;
  /** Main value to display (e.g., "1.2K", "150 ms") */
  value?: React.ReactNode;
  /** Optional subtitle shown after the value */
  subtitle?: React.ReactNode;
}

/**
 * Common header component for overview chart cards with icon, title, and value
 */
export const OverviewChartHeader: React.FC<OverviewChartHeaderProps> = ({ icon, title, value, subtitle }) => {
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

/**
 * "Over time" label shown above time-series charts in overview
 */
export const OverviewChartTimeLabel: React.FC = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <Typography.Text color="secondary" size="sm" css={{ textTransform: 'uppercase', letterSpacing: '0.5px' }}>
      <FormattedMessage defaultMessage="Over time" description="Label above time-series charts" />
    </Typography.Text>
  );
};

interface OverviewChartCardProps {
  children: React.ReactNode;
  height?: number;
}

/**
 * Common wrapper for overview chart cards with consistent styling
 */
export const OverviewChartCard: React.FC<OverviewChartCardProps> = ({ children, height = DEFAULT_CHART_HEIGHT }) => {
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

interface OverviewChartLoadingStateProps {
  height?: number;
}

/**
 * Loading state for overview chart cards
 */
export const OverviewChartLoadingState: React.FC<OverviewChartLoadingStateProps> = ({
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

interface OverviewChartErrorStateProps {
  height?: number;
  message?: React.ReactNode;
}

/**
 * Error state for overview chart cards
 */
export const OverviewChartErrorState: React.FC<OverviewChartErrorStateProps> = ({ height, message }) => {
  return (
    <OverviewChartCard height={height}>
      <Typography.Text color="error">
        {message || (
          <FormattedMessage
            defaultMessage="Failed to load chart data"
            description="Error message when chart fails to load"
          />
        )}
      </Typography.Text>
    </OverviewChartCard>
  );
};

interface OverviewChartEmptyStateProps {
  height?: number;
  message?: React.ReactNode;
}

/**
 * Empty state for overview chart cards when no data is available
 */
export const OverviewChartEmptyState: React.FC<OverviewChartEmptyStateProps> = ({ height, message }) => {
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

interface ScrollableTooltipProps {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
  formatter?: (value: number, name: string) => [string | number, string];
}

/**
 * Custom scrollable tooltip component for Recharts.
 * Use with: <Tooltip content={<ScrollableTooltip formatter={...} />} />
 */
export const ScrollableTooltip: React.FC<ScrollableTooltipProps> = ({ active, payload, label, formatter }) => {
  const { theme } = useDesignSystemTheme();

  if (!active || !payload?.length) {
    return null;
  }

  return (
    <div
      style={{
        // This ensures the tooltip is semi-transparent so the chart is visible through it.
        backgroundColor: `color-mix(in srgb, ${theme.colors.backgroundPrimary} 50%, transparent)`,
        backdropFilter: 'blur(2px)',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        fontSize: theme.typography.fontSizeSm,
        padding: theme.spacing.sm,
        pointerEvents: 'auto',
        // This is to ensure the tooltip renders on the cursor position, so users can hover
        // over the tooltip and scroll if applicable.
        marginLeft: -20,
        marginRight: -20,
      }}
    >
      {label && <div style={{ fontWeight: 500, marginBottom: theme.spacing.xs }}>{label}</div>}
      <div
        style={{
          maxHeight: '120px',
          overflowY: 'auto',
          overflowX: 'hidden',
        }}
      >
        {payload.map((entry, index) => {
          const [formattedValue, formattedName] = formatter
            ? formatter(entry.value, entry.name)
            : [entry.value, entry.name];
          return (
            <div
              key={index}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                paddingTop: 2,
                paddingBottom: 2,
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: entry.color,
                  flexShrink: 0,
                }}
              />
              <span style={{ color: entry.color }}>{formattedName}:</span>
              <span>{formattedValue}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/**
 * Returns common XAxis props for time-series charts
 */
export function useChartXAxisProps() {
  const { theme } = useDesignSystemTheme();
  return {
    tick: { fontSize: 10, fill: theme.colors.textSecondary, dy: theme.spacing.sm },
    axisLine: false,
    tickLine: false,
    interval: 'preserveStartEnd' as const,
  };
}

/**
 * Configuration for scrollable legend
 */
interface ScrollableLegendConfig {
  /** Maximum height for the legend container before scrolling. Defaults to 60. */
  maxHeight?: number;
}

/**
 * Returns legend props for a scrollable legend with consistent styling.
 * Use this when there may be many legend items to prevent overwhelming the chart.
 * Spread the returned object onto the Recharts Legend component.
 *
 * @example
 * const scrollableLegendProps = useScrollableLegendProps();
 * <Legend {...scrollableLegendProps} />
 */
export function useScrollableLegendProps(config?: ScrollableLegendConfig) {
  const { theme } = useDesignSystemTheme();
  const maxHeight = config?.maxHeight ?? 60;

  const formatter = (value: string) => (
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

  const wrapperStyle: React.CSSProperties = {
    maxHeight,
    overflowY: 'auto',
    overflowX: 'hidden',
    paddingTop: theme.spacing.xs,
  };

  return {
    formatter,
    wrapperStyle,
  };
}

/**
 * Props for the OverviewChartContainer component
 */
interface OverviewChartContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

/**
 * Common container styling for overview chart cards
 */
export const OverviewChartContainer: React.FC<OverviewChartContainerProps> = ({ children, ...rest }) => {
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

/**
 * Hook that returns props for ReferenceArea to show zoom selection highlight.
 * Use this with Recharts ReferenceArea component directly since Recharts
 * components must be direct children of the chart.
 *
 * @example
 * const zoomSelectionProps = useChartZoomSelectionProps();
 * <BarChart>
 *   {refAreaLeft && refAreaRight && (
 *     <ReferenceArea x1={refAreaLeft} x2={refAreaRight} {...zoomSelectionProps} />
 *   )}
 * </BarChart>
 */
export function useChartZoomSelectionProps() {
  const { theme } = useDesignSystemTheme();

  return {
    strokeOpacity: 0.3,
    fill: theme.colors.blue200,
    fillOpacity: 0.3,
  };
}
