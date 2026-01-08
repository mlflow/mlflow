import React, { useMemo } from 'react';
import { SparkleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useAssessmentChartsSectionData } from '../hooks/useAssessmentChartsSectionData';
import { OverviewChartLoadingState, OverviewChartErrorState, OverviewChartEmptyState } from './OverviewChartComponents';
import { LazyTraceAssessmentChart } from './LazyTraceAssessmentChart';
import { useChartColors } from '../utils/chartUtils';

interface AssessmentChartsSectionProps {
  /** Optional search query to filter assessments by name */
  searchQuery?: string;
}

/**
 * Component that fetches available feedback assessments and renders a chart for each one.
 */
export const AssessmentChartsSection: React.FC<AssessmentChartsSectionProps> = ({ searchQuery }) => {
  const { theme } = useDesignSystemTheme();

  // Fetch and process assessment data
  const { assessmentNames, avgValuesByName, isLoading, error, hasData } = useAssessmentChartsSectionData();

  // Filter assessment names based on search query (matches chart title)
  const filteredAssessmentNames = useMemo(() => {
    if (!searchQuery?.trim()) return assessmentNames;
    const query = searchQuery.toLowerCase().trim();
    return assessmentNames.filter((name) => name.toLowerCase().includes(query));
  }, [assessmentNames, searchQuery]);

  // Get chart colors for consistent coloring
  const { getChartColor } = useChartColors();

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  if (!hasData || filteredAssessmentNames.length === 0) {
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

      {/* Assessment charts - one row per scorer */}
      {filteredAssessmentNames.map((name) => {
        // Use original index for consistent colors
        const originalIndex = assessmentNames.indexOf(name);
        return (
          <LazyTraceAssessmentChart
            key={name}
            assessmentName={name}
            lineColor={getChartColor(originalIndex)}
            avgValue={avgValuesByName.get(name)}
          />
        );
      })}
    </div>
  );
};
