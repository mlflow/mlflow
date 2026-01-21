import React, { useMemo } from 'react';
import { SparkleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useAssessmentChartsSectionData } from '../hooks/useAssessmentChartsSectionData';
import { useHasAssessmentsOutsideTimeRange } from '../hooks/useHasAssessmentsOutsideTimeRange';
import { OverviewChartLoadingState, OverviewChartErrorState } from './OverviewChartComponents';
import { LazyTraceAssessmentChart } from './LazyTraceAssessmentChart';
import { useChartColors } from '../utils/chartUtils';
import { QualityTabEmptyState } from './QualityTabEmptyState';

/**
 * Component that fetches available feedback assessments and renders a chart for each one.
 */
export const AssessmentChartsSection: React.FC = () => {
  const { theme } = useDesignSystemTheme();

  // Fetch and process assessment data
  const { assessmentNames, avgValuesByName, isLoading, error, hasData } = useAssessmentChartsSectionData();

  // Check if there are assessments outside the time range (only when no data in current range)
  const { hasAssessmentsOutsideTimeRange, isLoading: isLoadingOutsideRange } = useHasAssessmentsOutsideTimeRange(
    !hasData && !isLoading,
  );

  // Get chart colors for consistent coloring
  const { getChartColor } = useChartColors();

  if (isLoading || (!hasData && isLoadingOutsideRange)) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  if (!hasData) {
    return <QualityTabEmptyState hasAssessmentsOutsideTimeRange={hasAssessmentsOutsideTimeRange} />;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {/* Section header */}
      <div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginBottom: theme.spacing.xs }}>
          <SparkleIcon css={{ color: theme.colors.yellow500 }} />
          <Typography.Text bold size="lg">
            <FormattedMessage
              defaultMessage="Quality Insights"
              description="Title for the quality insights section in quality tab"
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

      {/* Assessment charts - one row per scorer */}
      {assessmentNames.map((name, index) => (
        <LazyTraceAssessmentChart
          key={name}
          assessmentName={name}
          lineColor={getChartColor(index)}
          avgValue={avgValuesByName.get(name)}
        />
      ))}
    </div>
  );
};
