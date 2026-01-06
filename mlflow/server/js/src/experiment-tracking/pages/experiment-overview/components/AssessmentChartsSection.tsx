import React, { useMemo, useCallback } from 'react';
import { SparkleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useAssessmentChartsSectionData } from '../hooks/useAssessmentChartsSectionData';
import { ChartGrid } from './OverviewLayoutComponents';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartContainer,
} from './OverviewChartComponents';
import type { OverviewChartProps } from '../types';

/**
 * Component that fetches available feedback assessments and renders a chart for each one.
 */
export const AssessmentChartsSection: React.FC<OverviewChartProps> = (props) => {
  const { theme } = useDesignSystemTheme();

  // Fetch and process assessment data
  const { assessmentNames, avgValuesByName, isLoading, error, hasData } = useAssessmentChartsSectionData(props);

  // Color palette using design system colors
  const assessmentColors = useMemo(
    () => [
      theme.colors.green500,
      theme.colors.red500,
      theme.colors.blue500,
      theme.colors.yellow500,
      theme.colors.green300,
      theme.colors.red300,
      theme.colors.blue300,
      theme.colors.yellow300,
    ],
    [theme],
  );

  // Get a color for an assessment based on its index
  const getAssessmentColor = useCallback(
    (index: number): string => assessmentColors[index % assessmentColors.length],
    [assessmentColors],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  if (!hasData) {
    return (
      <OverviewChartEmptyState
        message={
          <FormattedMessage
            defaultMessage="No assessments available"
            description="Message shown when there are no assessments to display"
          />
        }
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {/* Section header */}
      <div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginBottom: theme.spacing.xs }}>
          <SparkleIcon css={{ color: theme.colors.yellow500 }} />
          <Typography.Text bold size="lg">
            <FormattedMessage
              defaultMessage="Scorer Insights"
              description="Title for the scorer insights section in quality tab"
            />
          </Typography.Text>
        </div>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Quality metrics computed by scorers."
            description="Description for the scorer insights section"
          />
        </Typography.Text>
      </div>

      {/* Assessment charts grid */}
      <ChartGrid>
        {assessmentNames.map((name, index) => (
          // PLACEHOLDER: Render assessment charts here
          <OverviewChartContainer key={name}>
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: 200,
                color: getAssessmentColor(index),
              }}
            >
              <Typography.Title level={4}>{name}</Typography.Title>
              <Typography.Text color="secondary">Avg: {avgValuesByName.get(name)?.toFixed(2) ?? 'N/A'}</Typography.Text>
            </div>
          </OverviewChartContainer>
        ))}
      </ChartGrid>
    </div>
  );
};
