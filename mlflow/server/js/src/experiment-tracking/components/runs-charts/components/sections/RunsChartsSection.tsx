import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { RunsChartsCardConfig } from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { RunsCharts } from '../RunsCharts';
import type { RunsChartCardSetFullscreenFn } from '../cards/ChartCard.common';
import { RunsChartsGlobalLineChartConfig } from '../../../experiment-page/models/ExperimentPageUIState';
import type { ChartSectionConfig } from '../../../../types';
import { RunsChartsDraggableCardsGridSection } from '../RunsChartsDraggableCardsGridSection';
import { shouldEnableDraggableChartsGridLayout } from '../../../../../common/utils/FeatureUtils';

export interface RunsChartsSectionProps {
  sectionId: string;
  sectionConfig: ChartSectionConfig;
  sectionCharts: RunsChartsCardConfig[];
  reorderCharts: (sourceChartUuid: string, targetChartUuid: string) => void;
  insertCharts: (sourceChartUuid: string, targetSectionId: string) => void;
  isMetricHistoryLoading: boolean;
  chartData: RunsChartsRunData[];
  startEditChart: (chartCard: RunsChartsCardConfig) => void;
  removeChart: (configToDelete: RunsChartsCardConfig) => void;
  groupBy: RunsGroupByConfig | null;
  sectionIndex: number;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
}

export const RunsChartsSection = ({
  sectionId,
  sectionCharts,
  reorderCharts,
  insertCharts,
  isMetricHistoryLoading,
  chartData,
  startEditChart,
  removeChart,
  groupBy,
  sectionIndex,
  setFullScreenChart,
  autoRefreshEnabled,
  hideEmptyCharts,
  globalLineChartConfig,
  sectionConfig,
}: RunsChartsSectionProps) => {
  // If the feature flag is enabled, use the draggable grid layout
  if (shouldEnableDraggableChartsGridLayout()) {
    return (
      <RunsChartsDraggableCardsGridSection
        sectionConfig={sectionConfig}
        cardsConfig={sectionCharts}
        chartRunData={chartData}
        onStartEditChart={startEditChart}
        onRemoveChart={removeChart}
        setFullScreenChart={setFullScreenChart}
        sectionId={sectionId}
        groupBy={groupBy}
        autoRefreshEnabled={autoRefreshEnabled}
        hideEmptyCharts={hideEmptyCharts}
        globalLineChartConfig={globalLineChartConfig}
      />
    );
  }

  return (
    <RunsCharts
      sectionId={sectionId}
      chartRunData={chartData}
      cardsConfig={sectionCharts}
      isMetricHistoryLoading={isMetricHistoryLoading}
      onRemoveChart={removeChart}
      onStartEditChart={startEditChart}
      onReorderCharts={reorderCharts}
      onInsertCharts={insertCharts}
      groupBy={groupBy}
      sectionIndex={sectionIndex}
      setFullScreenChart={setFullScreenChart}
      autoRefreshEnabled={autoRefreshEnabled}
      hideEmptyCharts={hideEmptyCharts}
      globalLineChartConfig={globalLineChartConfig}
    />
  );
};
