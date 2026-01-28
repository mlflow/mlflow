import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanDimensionKey,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface CostBreakdownDataPoint {
  name: string;
  value: number;
  percentage: number;
}

export interface UseTraceCostBreakdownChartDataResult {
  chartData: CostBreakdownDataPoint[];
  totalCost: number;
  isLoading: boolean;
  error: unknown;
  hasData: boolean;
}

export function useTraceCostBreakdownChartData(): UseTraceCostBreakdownChartDataResult {
  const { experimentId, startTimeMs, endTimeMs } = useOverviewChartContext();

  // Fetch total cost grouped by model name
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.TOTAL_COST,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    dimensions: [SpanDimensionKey.MODEL_NAME],
  });

  const dataPoints = useMemo(() => data?.data_points || [], [data?.data_points]);

  // Calculate total cost and percentages
  const { chartData, totalCost } = useMemo(() => {
    const total = dataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0);

    const items: CostBreakdownDataPoint[] = dataPoints
      .map((dp) => {
        const value = dp.values?.[AggregationType.SUM] || 0;
        const modelName = dp.dimensions?.[SpanDimensionKey.MODEL_NAME] || 'Unknown';
        return {
          name: modelName,
          value,
          percentage: total > 0 ? (value / total) * 100 : 0,
        };
      })
      .filter((item) => item.value > 0)
      .sort((a, b) => b.value - a.value);

    return { chartData: items, totalCost: total };
  }, [dataPoints]);

  return {
    chartData,
    totalCost,
    isLoading,
    error,
    hasData: chartData.length > 0,
  };
}
