import { useMemo, memo } from 'react';
import { RunsChartType } from '../../runs-charts.types';
import type {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartsContourCardConfig,
  RunsChartsDifferenceCardConfig,
  RunsChartsImageCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsParallelCardConfig,
  RunsChartsScatterCardConfig,
} from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import {
  shouldEnableDraggableChartsGridLayout,
  shouldEnableDifferenceViewCharts,
  shouldEnableImageGridCharts,
  shouldUseNewRunRowsVisibilityModel,
} from '../../../../../common/utils/FeatureUtils';
import { RunsChartsBarChartCard } from './RunsChartsBarChartCard';
import { RunsChartsLineChartCard } from './RunsChartsLineChartCard';
import { RunsChartsScatterChartCard } from './RunsChartsScatterChartCard';
import { RunsChartsContourChartCard } from './RunsChartsContourChartCard';
import { RunsChartsParallelChartCard } from './RunsChartsParallelChartCard';
import type {
  RunsChartCardFullScreenProps,
  RunsChartCardReorderProps,
  RunsChartCardSizeProps,
  RunsChartCardVisibilityProps,
} from './ChartCard.common';
import { RunsChartsDifferenceChartCard } from './RunsChartsDifferenceChartCard';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { RunsChartsImageChartCard } from './RunsChartsImageChartCard';
import { RunsChartsGlobalLineChartConfig } from '../../../experiment-page/models/ExperimentPageUIState';

export interface RunsChartsCardProps
  extends RunsChartCardReorderProps,
    RunsChartCardFullScreenProps,
    RunsChartCardVisibilityProps,
    RunsChartCardSizeProps {
  cardConfig: RunsChartsCardConfig;
  chartRunData: RunsChartsRunData[];
  onStartEditChart: (chart: RunsChartsCardConfig) => void;
  onRemoveChart: (chart: RunsChartsCardConfig) => void;
  onDownloadFullMetricHistoryCsv?: (runUuids: string[], metricKeys: string[]) => void;
  index: number;
  sectionIndex?: number;
  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;
  groupBy: RunsGroupByConfig | null;
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
}

const RunsChartsCardRaw = ({
  cardConfig,
  chartRunData,
  index,
  sectionIndex,
  onStartEditChart,
  onRemoveChart,
  setFullScreenChart,
  groupBy,
  fullScreen,
  canMoveDown,
  canMoveUp,
  previousChartUuid,
  nextChartUuid,
  onReorderWith,
  autoRefreshEnabled,
  onDownloadFullMetricHistoryCsv,
  hideEmptyCharts,
  globalLineChartConfig,
  height,
  isInViewport,
  isInViewportDeferred,
}: RunsChartsCardProps) => {
  const usingGridLayout = shouldEnableDraggableChartsGridLayout();
  const chartElementKey = `${cardConfig.uuid}-${index}-${sectionIndex}`;

  const reorderProps = useMemo(
    () => ({
      onReorderWith,
      canMoveDown,
      canMoveUp,
      previousChartUuid,
      nextChartUuid,
    }),
    [onReorderWith, canMoveDown, canMoveUp, previousChartUuid, nextChartUuid],
  );

  const editProps = useMemo(
    () => ({
      onEdit: () => onStartEditChart(cardConfig),
      onDelete: () => onRemoveChart(cardConfig),
      setFullScreenChart,
    }),
    [onStartEditChart, onRemoveChart, setFullScreenChart, cardConfig],
  );

  const commonChartProps = useMemo(
    () => ({
      fullScreen,
      key: usingGridLayout ? undefined : chartElementKey,
      autoRefreshEnabled,
      groupBy,
      hideEmptyCharts,
      height,
      isInViewport,
      isInViewportDeferred,
      ...editProps,
      ...reorderProps,
    }),
    [
      fullScreen,
      usingGridLayout,
      chartElementKey,
      autoRefreshEnabled,
      groupBy,
      editProps,
      reorderProps,
      hideEmptyCharts,
      height,
      isInViewport,
      isInViewportDeferred,
    ],
  );

  const slicedRuns = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      return chartRunData.filter(({ hidden }) => !hidden).reverse();
    }
    return chartRunData.slice(0, cardConfig.runsCountToCompare || 10).reverse();
  }, [chartRunData, cardConfig.runsCountToCompare]);

  if (cardConfig.type === RunsChartType.PARALLEL) {
    return (
      <RunsChartsParallelChartCard
        config={cardConfig as RunsChartsParallelCardConfig}
        chartRunData={chartRunData}
        {...commonChartProps}
      />
    );
  }

  if (shouldEnableDifferenceViewCharts() && cardConfig.type === RunsChartType.DIFFERENCE) {
    return (
      <RunsChartsDifferenceChartCard
        config={cardConfig as RunsChartsDifferenceCardConfig}
        chartRunData={chartRunData}
        {...commonChartProps}
      />
    );
  }

  if (shouldEnableImageGridCharts() && cardConfig.type === RunsChartType.IMAGE) {
    return (
      <RunsChartsImageChartCard
        config={cardConfig as RunsChartsImageCardConfig}
        chartRunData={chartRunData}
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
        onDownloadFullMetricHistoryCsv={onDownloadFullMetricHistoryCsv}
        globalLineChartConfig={globalLineChartConfig}
        positionInSection={index}
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

export const RunsChartsCard = memo(RunsChartsCardRaw);
