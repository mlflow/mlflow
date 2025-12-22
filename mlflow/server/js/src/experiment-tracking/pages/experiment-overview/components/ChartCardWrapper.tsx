import React from 'react';
import { Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

const DEFAULT_CHART_HEIGHT = 280;

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

