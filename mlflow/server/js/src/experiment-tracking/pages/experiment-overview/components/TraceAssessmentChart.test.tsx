import { describe, it, expect } from '@jest/globals';
import { screen } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TraceAssessmentChart, type TraceAssessmentChartProps } from './TraceAssessmentChart';
import { DesignSystemProvider } from '@databricks/design-system';
import { OverviewChartProvider } from '../OverviewChartContext';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import type { AssessmentChartDataPoint, DistributionChartDataPoint } from '../hooks/useAssessmentChartsSectionData';

describe('TraceAssessmentChart', () => {
  const testAssessmentName = 'Correctness';
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();
  const timeIntervalSeconds = 3600;

  const timeBuckets = [
    new Date('2025-12-22T10:00:00Z').getTime(),
    new Date('2025-12-22T11:00:00Z').getTime(),
    new Date('2025-12-22T12:00:00Z').getTime(),
  ];

  const defaultContextProps = {
    experimentIds: ['test-experiment-123'],
    startTimeMs,
    endTimeMs,
    timeIntervalSeconds,
    timeBuckets,
  };

  // Helper to create time-series chart data (already processed)
  const createTimeSeriesData = (
    values: { label: string; value: number | null; timestampMs: number }[],
  ): AssessmentChartDataPoint[] => values.map((v) => ({ name: v.label, value: v.value, timestampMs: v.timestampMs }));

  // Helper to create distribution chart data (already processed/bucketed)
  const createDistributionData = (entries: { name: string; count: number }[]): DistributionChartDataPoint[] => entries;

  const defaultTimeSeriesData = createTimeSeriesData([
    { label: '12/22, 10 AM', value: 0.75, timestampMs: timeBuckets[0] },
    { label: '12/22, 11 AM', value: 0.82, timestampMs: timeBuckets[1] },
    { label: '12/22, 12 PM', value: null, timestampMs: timeBuckets[2] },
  ]);

  const defaultDistributionData = createDistributionData([
    { name: '0.75', count: 5 },
    { name: '0.82', count: 10 },
  ]);

  const defaultProps: TraceAssessmentChartProps = {
    assessmentName: testAssessmentName,
    timeSeriesChartData: defaultTimeSeriesData,
    distributionChartData: defaultDistributionData,
  };

  const renderComponent = (props: Partial<TraceAssessmentChartProps> = {}) => {
    return renderWithIntl(
      <MemoryRouter>
        <DesignSystemProvider>
          <OverviewChartProvider {...defaultContextProps}>
            <TraceAssessmentChart {...defaultProps} {...props} />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </MemoryRouter>,
    );
  };

  describe('empty data state', () => {
    it('should render empty state when no data points are provided', () => {
      renderComponent({ timeSeriesChartData: [], distributionChartData: [] });

      expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
    });
  });

  describe('with data', () => {
    it('should render chart with all time buckets when avgValue is provided', () => {
      renderComponent({ avgValue: 0.78 });

      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    });

    it('should display the assessment name as title', () => {
      renderComponent();

      expect(screen.getByText(testAssessmentName)).toBeInTheDocument();
    });

    it('should display both chart section labels when avgValue is provided', () => {
      renderComponent({ avgValue: 0.78 });

      expect(screen.getByText('Total aggregate scores')).toBeInTheDocument();
      expect(screen.getByText('Moving average over time')).toBeInTheDocument();
    });

    it('should only display distribution chart label when avgValue is not provided', () => {
      renderComponent();

      expect(screen.getByText('Total aggregate scores')).toBeInTheDocument();
      expect(screen.queryByText('Moving average over time')).not.toBeInTheDocument();
    });

    it('should display average value when provided via prop', () => {
      renderComponent({ avgValue: 0.78 });

      expect(screen.getByText('0.78')).toBeInTheDocument();
      expect(screen.getByText('avg score')).toBeInTheDocument();
    });

    it('should render reference line when avgValue is provided', () => {
      renderComponent({ avgValue: 0.78 });

      const referenceLine = screen.getByTestId('reference-line');
      expect(referenceLine).toBeInTheDocument();
      expect(referenceLine).toHaveAttribute('data-label', 'AVG (0.78)');
    });

    it('should NOT render moving average chart when avgValue is not provided', () => {
      renderComponent();

      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.queryByTestId('line-chart')).not.toBeInTheDocument();
      expect(screen.queryByTestId('reference-line')).not.toBeInTheDocument();
      expect(screen.queryByText('Moving average over time')).not.toBeInTheDocument();
    });

    it('should render both charts when avgValue is provided', () => {
      renderComponent({ avgValue: 0.78 });

      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('should fill missing time buckets with null when avgValue is provided', () => {
      // Only one time bucket has a value
      renderComponent({
        avgValue: 0.8,
        timeSeriesChartData: createTimeSeriesData([
          { label: '12/22, 10 AM', value: 0.8, timestampMs: timeBuckets[0] },
          { label: '12/22, 11 AM', value: null, timestampMs: timeBuckets[1] },
          { label: '12/22, 12 PM', value: null, timestampMs: timeBuckets[2] },
        ]),
      });

      // Chart should still show all 3 time buckets
      expect(screen.getByTestId('line-chart')).toHaveAttribute('data-count', '3');
    });
  });

  describe('custom props', () => {
    it('should accept custom lineColor', () => {
      renderComponent({ lineColor: '#FF0000', avgValue: 0.8 });

      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('should render with different assessment names', () => {
      renderComponent({ assessmentName: 'Relevance' });

      expect(screen.getByText('Relevance')).toBeInTheDocument();
    });
  });

  describe('distribution chart - display behavior', () => {
    it('should display integer values directly when 5 or fewer unique values', () => {
      renderComponent({
        distributionChartData: createDistributionData([
          { name: '1', count: 5 },
          { name: '2', count: 10 },
          { name: '3', count: 15 },
          { name: '4', count: 8 },
          { name: '5', count: 3 },
        ]),
      });

      const barChart = screen.getByTestId('bar-chart');
      expect(barChart).toHaveAttribute('data-count', '5');
      expect(barChart).toHaveAttribute('data-labels', '1,2,3,4,5');
    });

    it('should display bucketed ranges for float values', () => {
      // Pre-bucketed data (bucketing is done in the parent hook)
      renderComponent({
        distributionChartData: createDistributionData([
          { name: '0.10-0.26', count: 5 },
          { name: '0.26-0.42', count: 0 },
          { name: '0.42-0.58', count: 10 },
          { name: '0.58-0.74', count: 0 },
          { name: '0.74-0.90', count: 8 },
        ]),
      });

      const barChart = screen.getByTestId('bar-chart');
      expect(barChart).toHaveAttribute('data-count', '5');
    });

    it('should display boolean values directly', () => {
      renderComponent({
        distributionChartData: createDistributionData([
          { name: 'false', count: 5 },
          { name: 'true', count: 15 },
        ]),
      });

      const barChart = screen.getByTestId('bar-chart');
      expect(barChart).toHaveAttribute('data-count', '2');
      expect(barChart).toHaveAttribute('data-labels', 'false,true');
    });

    it('should display string values sorted alphabetically', () => {
      renderComponent({
        distributionChartData: createDistributionData([
          { name: 'error', count: 2 },
          { name: 'fail', count: 5 },
          { name: 'pass', count: 15 },
        ]),
      });

      const barChart = screen.getByTestId('bar-chart');
      expect(barChart).toHaveAttribute('data-count', '3');
      expect(barChart).toHaveAttribute('data-labels', 'error,fail,pass');
    });

    it('should render both bar chart and line chart when avgValue is provided', () => {
      renderComponent({ avgValue: 0.8 });

      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('should render only bar chart when avgValue is not provided', () => {
      renderComponent({
        distributionChartData: createDistributionData([
          { name: 'pass', count: 10 },
          { name: 'fail', count: 5 },
        ]),
      });

      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.queryByTestId('line-chart')).not.toBeInTheDocument();
    });

    it('should display "Total aggregate scores" label for bar chart', () => {
      renderComponent({ avgValue: 0.8 });

      expect(screen.getByText('Total aggregate scores')).toBeInTheDocument();
    });

    it('should display "Moving average over time" label for line chart when avgValue is provided', () => {
      renderComponent({ avgValue: 0.8 });

      expect(screen.getByText('Moving average over time')).toBeInTheDocument();
    });
  });
});
