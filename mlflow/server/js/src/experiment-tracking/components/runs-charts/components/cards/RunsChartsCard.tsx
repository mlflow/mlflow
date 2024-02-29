import { useMemo } from 'react';
import { RunsChartType } from '../../runs-charts.types';
import type {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartsContourCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsParallelCardConfig,
  RunsChartsScatterCardConfig,
} from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { shouldUseNewRunRowsVisibilityModel } from '../../../../../common/utils/FeatureUtils';
import { RunsChartsBarChartCard } from './RunsChartsBarChartCard';
import { RunsChartsLineChartCard } from './RunsChartsLineChartCard';
import { RunsChartsScatterChartCard } from './RunsChartsScatterChartCard';
import { RunsChartsContourChartCard } from './RunsChartsContourChartCard';
import { RunsChartsParallelChartCard } from './RunsChartsParallelChartCard';
import { RunsChartCardFullScreenProps, RunsChartCardReorderProps } from './ChartCard.common';

export interface RunsChartsCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  cardConfig: RunsChartsCardConfig;
  chartRunData: RunsChartsRunData[];
  onStartEditChart: (chart: RunsChartsCardConfig) => void;
  onRemoveChart: (chart: RunsChartsCardConfig) => void;
  onReorderCharts: (sourceChartUuid: string, targetChartUuid: string) => void;
  index: number;
  sectionIndex: number;
  isMetricHistoryLoading?: boolean;
  groupBy: string;
}

export const RunsChartsCard = ({
  cardConfig,
  chartRunData,
  index,
  sectionIndex,
  onStartEditChart,
  onRemoveChart,
  setFullScreenChart,
  isMetricHistoryLoading,
  groupBy,
  fullScreen,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
  onReorderWith,
}: RunsChartsCardProps) => {
  const chartElementKey = `${cardConfig.uuid}-${index}-${sectionIndex}`;

  const reorderProps = {
    onReorderWith,
    canMoveDown,
    canMoveUp,
    onMoveDown,
    onMoveUp,
  };

  const editProps = {
    onEdit: () => onStartEditChart(cardConfig),
    onDelete: () => onRemoveChart(cardConfig),
    setFullScreenChart,
  };

  const commonChartProps = {
    fullScreen,
    key: chartElementKey,
    ...editProps,
    ...reorderProps,
  };

  const slicedRuns = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      return chartRunData.filter(({ hidden }) => !hidden).reverse();
    }
    return chartRunData.slice(0, cardConfig.runsCountToCompare || 10).reverse();
  }, [chartRunData, cardConfig]);

  if (cardConfig.type === RunsChartType.PARALLEL) {
    return (
      <RunsChartsParallelChartCard
        config={cardConfig as RunsChartsParallelCardConfig}
        chartRunData={chartRunData}
        groupBy={groupBy}
        {...commonChartProps}
      />
    );
  }

  if (cardConfig.type === RunsChartType.BAR) {
    return (
      <RunsChartsBarChartCard
        config={cardConfig as RunsChartsBarCardConfig}
        chartRunData={slicedRuns}
        {...commonChartProps}
      />
    );
  } else if (cardConfig.type === RunsChartType.LINE) {
    return (
      <RunsChartsLineChartCard
        config={cardConfig as RunsChartsLineCardConfig}
        chartRunData={slicedRuns}
        isMetricHistoryLoading={isMetricHistoryLoading}
        groupBy={groupBy}
        {...commonChartProps}
      />
    );
  } else if (cardConfig.type === RunsChartType.SCATTER) {
    return (
      <RunsChartsScatterChartCard
        config={cardConfig as RunsChartsScatterCardConfig}
        chartRunData={slicedRuns}
        {...commonChartProps}
      />
    );
  } else if (cardConfig.type === RunsChartType.CONTOUR) {
    return (
      <RunsChartsContourChartCard
        config={cardConfig as RunsChartsContourCardConfig}
        chartRunData={slicedRuns}
        {...commonChartProps}
      />
    );
  }
  return null;
};
