import React, { useMemo, useCallback } from 'react';
import { SparkleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentType,
  AssessmentDimensionKey,
  createAssessmentFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import { ChartGrid } from './OverviewLayoutComponents';
import { ChartLoadingState, ChartErrorState, ChartEmptyState, ChartContainer } from './ChartCardWrapper';
import type { OverviewChartProps } from '../types';

/**
 * Component that fetches available feedback assessments and renders a chart for each one.
 */
export const AssessmentChartsSection: React.FC<OverviewChartProps> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
}) => {
  const { theme } = useDesignSystemTheme();

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

  // Filter for feedback assessments only
  const filters = useMemo(() => [createAssessmentFilter(AssessmentFilterKey.TYPE, AssessmentType.FEEDBACK)], []);

  // Query assessments grouped by assessment_name to get the list and average values
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_VALUE,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    filters,
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
  });

  // Extract assessment names and their average values from the response
  const { assessmentNames, avgValuesByName } = useMemo(() => {
    if (!data?.data_points) return { assessmentNames: [], avgValuesByName: new Map<string, number>() };

    const names: string[] = [];
    const avgValues = new Map<string, number>();

    for (const dp of data.data_points) {
      const name = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      if (name) {
        names.push(name);
        const avgValue = dp.values?.[AggregationType.AVG];
        if (avgValue !== undefined) {
          avgValues.set(name, avgValue);
        }
      }
    }

    return { assessmentNames: names.sort(), avgValuesByName: avgValues };
  }, [data?.data_points]);

  if (isLoading) {
    return <ChartLoadingState />;
  }

  if (error) {
    return <ChartErrorState />;
  }

  if (assessmentNames.length === 0) {
    return (
      <ChartEmptyState
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
          <ChartContainer key={name}>
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
          </ChartContainer>
        ))}
      </ChartGrid>
    </div>
  );
};
