import React from 'react';
import { TableSkeleton, TitleSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useChartInteractionTelemetry } from '../hooks/useChartInteractionTelemetry';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { ExperimentPageTabName } from '../../../constants';
import {
  FilterOperator,
  HiddenFilterOperator,
  TracesTableColumnGroup,
} from '@databricks/web-shared/genai-traces-table';

export const DEFAULT_CHART_HEIGHT = 280;
export const DEFAULT_CHART_CONTENT_HEIGHT = 200;
export const DEFAULT_TOOLTIP_MAX_HEIGHT = 120;
export const DEFAULT_LEGEND_MAX_HEIGHT = 60;

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

/**
 * Generates a URL to the traces tab with optional time range and filters.
 */
export function getTracesFilteredUrl(
  experimentId: string,
  timeRange?: { startTimeLabel?: string; startTime?: string; endTime?: string },
  filters?: string[],
): string {
  const tracesPath = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces);
  const queryParams = new URLSearchParams();

  const { startTimeLabel, startTime, endTime } = timeRange ?? {};
  if (startTimeLabel) queryParams.set('startTimeLabel', startTimeLabel);
  if (startTime) queryParams.set('startTime', startTime);
  if (endTime) queryParams.set('endTime', endTime);

  filters?.forEach((filter) => queryParams.append('filter', filter));

  return `${tracesPath}?${queryParams.toString()}`;
}

/**
 * Generates a URL to the traces tab filtered by a specific time range.
 * Use this to create navigation links from charts to the traces view.
 *
 * @param experimentId - The experiment ID
 * @param timestampMs - Start timestamp in milliseconds
 * @param timeIntervalSeconds - Duration of the time bucket in seconds
 * @param filters - Optional array of filter strings in format "column::operator::value::key"
 * @returns Full URL path with query parameters for the traces tab
 */
export function getTracesFilteredByTimeRangeUrl(
  experimentId: string,
  timestampMs: number,
  timeIntervalSeconds: number,
  filters?: string[],
): string {
  const startTime = new Date(timestampMs).toISOString();
  const endTime = new Date(timestampMs + timeIntervalSeconds * 1000).toISOString();
  return getTracesFilteredUrl(experimentId, { startTimeLabel: 'CUSTOM', startTime, endTime }, filters);
}

/**
 * Creates a filter string for traces where the specified assessment has a non-null value.
 * Use with getTracesFilteredByTimeRangeUrl to filter by assessment existence.
 *
 * @param assessmentName - The name of the assessment
 * @returns Filter string in format "column::operator::value::key"
 */
export function createAssessmentExistsFilter(assessmentName: string): string {
  return [TracesTableColumnGroup.ASSESSMENT, HiddenFilterOperator.IS_NOT_NULL, '', assessmentName].join('::');
}

/**
 * Creates a filter string for traces where the specified assessment equals a specific value.
 * Use with getTracesFilteredByTimeRangeUrl to filter by assessment score.
 *
 * @param assessmentName - The name of the assessment
 * @param scoreValue - The score value to filter by
 * @returns Filter string in format "column::operator::value::key"
 */
export function createAssessmentEqualsFilter(assessmentName: string, scoreValue: string): string {
  return [TracesTableColumnGroup.ASSESSMENT, FilterOperator.EQUALS, scoreValue, assessmentName].join('::');
}

/** Allowed component IDs for tooltip "View traces" links */
type TooltipLinkComponentId =
  | 'mlflow.overview.usage.traces.view_traces_link'
  | 'mlflow.overview.usage.latency.view_traces_link'
  | 'mlflow.overview.usage.errors.view_traces_link'
  | 'mlflow.overview.usage.token_stats.view_traces_link'
  | 'mlflow.overview.usage.token_usage.view_traces_link'
  | 'mlflow.overview.quality.assessment.view_traces_link'
  | 'mlflow.overview.quality.assessment_timeseries.view_traces_link';

/** Optional link configuration for ScrollableTooltip */
interface TooltipLinkConfig {
  /** Component ID for telemetry tracking */
  componentId: TooltipLinkComponentId;
  /** Custom link text. When provided, shows this text instead of the default */
  linkText?: React.ReactNode;
  /**
   * Custom click handler for the link.
   * @param label - The tooltip label (e.g., Y-axis category for vertical bar charts)
   * @param dataPoint - The data point payload containing timestampMs and other properties
   */
  onLinkClick?: (label: string | undefined, dataPoint?: { timestampMs?: number }) => void;
  /** Experiment ID for navigation (required when using time-based navigation) */
  experimentId?: string;
  /** Time interval in seconds for calculating end time of the bucket (required when using time-based navigation) */
  timeIntervalSeconds?: number;
}

interface ScrollableTooltipProps {
  active?: boolean;
  payload?: Array<{ payload?: { timestampMs?: number }; name: string; value: number; color: string }>;
  label?: string;
  /** Formatter function to display the value - returns [formattedValue, label] */
  formatter: (value: number, name: string) => [string | number, string];
  /** Optional link configuration. When provided, shows a link to view traces */
  linkConfig?: TooltipLinkConfig;
}

/**
 * Custom scrollable tooltip component for Recharts.
 * Optionally shows a "View traces for this period" link when linkConfig is provided.
 *
 * @example
 * // Basic usage without link
 * <Tooltip content={<ScrollableTooltip formatter={...} />} />
 *
 * @example
 * // With "View traces" link
 * <Tooltip
 *   content={
 *     <ScrollableTooltip
 *       formatter={(value) => [`${value}`, 'Requests']}
 *       linkConfig={{
 *         experimentId,
 *         timeIntervalSeconds,
 *         componentId: 'mlflow.overview.usage.traces.view_traces_link',
 *       }}
 *     />
 *   }
 * />
 */
export function ScrollableTooltip({ active, payload, label, formatter, linkConfig }: ScrollableTooltipProps) {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();

  if (!active || !payload?.length) {
    return null;
  }

  const dataPoint = payload[0]?.payload;
  // Show link if: 1) custom onLinkClick is provided, or 2) time-based navigation is configured with timestampMs
  const hasCustomLinkClick = linkConfig?.onLinkClick !== undefined;
  const hasTimeBasedNavigation =
    linkConfig?.experimentId && linkConfig?.timeIntervalSeconds && dataPoint?.timestampMs !== undefined;
  const showLink = linkConfig && (hasCustomLinkClick || hasTimeBasedNavigation);

  const handleLinkClick = () => {
    if (hasCustomLinkClick) {
      linkConfig.onLinkClick!(label, dataPoint);
    } else if (hasTimeBasedNavigation) {
      const url = getTracesFilteredByTimeRangeUrl(
        linkConfig.experimentId!,
        dataPoint.timestampMs!,
        linkConfig.timeIntervalSeconds!,
      );
      navigate(url);
    }
  };

  return (
    <div
      css={{
        // This ensures the tooltip is semi-transparent so the chart is visible through it.
        // 80 hex = 50% opacity
        backgroundColor: `${theme.colors.backgroundPrimary}80`,
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
      {label && <div css={{ fontWeight: 500, marginBottom: theme.spacing.xs }}>{label}</div>}
      <div
        css={{
          maxHeight: DEFAULT_TOOLTIP_MAX_HEIGHT,
          overflowY: 'auto',
          overflowX: 'hidden',
        }}
      >
        {payload.map((entry, index) => {
          const [formattedValue, formattedName] = formatter(entry.value, entry.name);
          return (
            <div
              key={index}
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                paddingTop: theme.spacing.xs,
                paddingBottom: theme.spacing.xs,
              }}
            >
              <span
                css={{
                  width: theme.spacing.sm,
                  height: theme.spacing.sm,
                  borderRadius: '50%',
                  backgroundColor: entry.color,
                  flexShrink: 0,
                }}
              />
              <span css={{ color: entry.color }}>{formattedName}:</span>
              <span>{formattedValue}</span>
            </div>
          );
        })}
      </div>
      {/* Link to view traces */}
      {showLink && (
        <div
          css={{
            borderTop: `1px solid ${theme.colors.border}`,
            marginTop: theme.spacing.xs,
            paddingTop: theme.spacing.xs,
          }}
        >
          <Typography.Link
            componentId={linkConfig.componentId}
            onClick={handleLinkClick}
            css={{
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
            }}
          >
            {linkConfig.linkText ?? (
              <FormattedMessage
                defaultMessage="View traces for this period"
                description="Link text to navigate to traces tab filtered by the selected time period"
              />
            )}
          </Typography.Link>
        </div>
      )}
    </div>
  );
}

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
 * Returns common YAxis props for charts
 */
export function useChartYAxisProps() {
  const { theme } = useDesignSystemTheme();
  return {
    tick: { fontSize: 10, fill: theme.colors.textSecondary },
    axisLine: false,
    tickLine: false,
    width: 40,
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
  const maxHeight = config?.maxHeight ?? DEFAULT_LEGEND_MAX_HEIGHT;

  const formatter = (value: string) => (
    <span
      css={{
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
  /** Component ID for telemetry tracking (e.g., "mlflow.charts.trace_requests") */
  componentId?: string;
}

/**
 * Common container styling for overview chart cards.
 * When componentId is provided, tracks user interactions for telemetry.
 */
export const OverviewChartContainer: React.FC<OverviewChartContainerProps> = ({ children, componentId, ...rest }) => {
  const { theme } = useDesignSystemTheme();
  const interactionProps = useChartInteractionTelemetry(componentId);

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
      {...interactionProps}
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
