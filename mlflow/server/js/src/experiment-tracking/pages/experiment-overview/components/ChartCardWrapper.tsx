import React from 'react';
import { Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

const DEFAULT_CHART_HEIGHT = 280;

interface ChartHeaderProps {
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
 * Common header component for chart cards with icon, title, and value
 */
export const ChartHeader: React.FC<ChartHeaderProps> = ({ icon, title, value, subtitle }) => {
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
 * "Over time" label shown above time-series charts
 */
export const OverTimeLabel: React.FC = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <Typography.Text color="secondary" size="sm" css={{ textTransform: 'uppercase', letterSpacing: '0.5px' }}>
      <FormattedMessage defaultMessage="Over time" description="Label above time-series charts" />
    </Typography.Text>
  );
};

interface ChartCardWrapperProps {
  children: React.ReactNode;
  height?: number;
}

/**
 * Common wrapper for chart cards with consistent styling
 */
export const ChartCardWrapper: React.FC<ChartCardWrapperProps> = ({ children, height = DEFAULT_CHART_HEIGHT }) => {
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

interface ChartLoadingStateProps {
  height?: number;
}

/**
 * Loading state for chart cards
 */
export const ChartLoadingState: React.FC<ChartLoadingStateProps> = ({ height }) => {
  return (
    <ChartCardWrapper height={height}>
      <Spinner />
    </ChartCardWrapper>
  );
};

interface ChartErrorStateProps {
  height?: number;
  message?: React.ReactNode;
}

/**
 * Error state for chart cards
 */
export const ChartErrorState: React.FC<ChartErrorStateProps> = ({ height, message }) => {
  return (
    <ChartCardWrapper height={height}>
      <Typography.Text color="error">
        {message || (
          <FormattedMessage
            defaultMessage="Failed to load chart data"
            description="Error message when chart fails to load"
          />
        )}
      </Typography.Text>
    </ChartCardWrapper>
  );
};

interface ChartEmptyStateProps {
  height?: number;
  message?: React.ReactNode;
}

/**
 * Empty state for chart cards when no data is available
 */
export const ChartEmptyState: React.FC<ChartEmptyStateProps> = ({ height, message }) => {
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
