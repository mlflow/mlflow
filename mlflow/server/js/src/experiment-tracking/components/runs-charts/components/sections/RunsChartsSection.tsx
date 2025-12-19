import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import type { RunsChartsCardConfig } from '../../runs-charts.types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartCardSetFullscreenFn } from '../cards/ChartCard.common';
import type { RunsChartsGlobalLineChartConfig } from '../../../experiment-page/models/ExperimentPageUIState';
import type { ChartSectionConfig } from '../../../../types';
import { RunsChartsDraggableCardsGridSection } from '../RunsChartsDraggableCardsGridSection';

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
};
