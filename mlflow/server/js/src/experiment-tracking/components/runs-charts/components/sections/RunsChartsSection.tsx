import { RunsChartsCardConfig } from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { RunsChartsV2 } from '../RunsChartsV2';
import type { RunsChartCardSetFullscreenFn } from '../cards/ChartCard.common';

export interface RunsChartsSectionProps {
  sectionId: string;
  sectionCharts: RunsChartsCardConfig[];
  reorderCharts: (sourceChartUuid: string, targetChartUuid: string) => void;
  insertCharts: (sourceChartUuid: string, targetSectionId: string) => void;
  isMetricHistoryLoading: boolean;
  chartData: RunsChartsRunData[];
  startEditChart: (chartCard: RunsChartsCardConfig) => void;
  removeChart: (configToDelete: RunsChartsCardConfig) => void;
  groupBy: string;
  sectionIndex: number;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
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
}: RunsChartsSectionProps) => {
  return (
    <RunsChartsV2
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
    />
  );
};
