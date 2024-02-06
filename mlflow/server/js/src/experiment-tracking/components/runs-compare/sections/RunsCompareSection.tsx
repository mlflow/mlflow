import { RunsCompareCardConfig } from '../runs-compare.types';
import { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsCompareChartsV2 } from '../RunsCompareChartsV2';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

export interface RunsCompareSectionProps {
  sectionId: string;
  sectionCharts: RunsCompareCardConfig[];
  reorderCharts: (sourceChartUuid: string, targetChartUuid: string) => void;
  insertCharts: (sourceChartUuid: string, targetSectionId: string) => void;
  isMetricHistoryLoading: boolean;
  chartData: RunsChartsRunData[];
  startEditChart: (chartCard: RunsCompareCardConfig) => void;
  removeChart: (configToDelete: RunsCompareCardConfig) => void;
  groupBy: string;
}

export const RunsCompareSection = ({
  sectionId,
  sectionCharts,
  reorderCharts,
  insertCharts,
  isMetricHistoryLoading,
  chartData,
  startEditChart,
  removeChart,
  groupBy,
}: RunsCompareSectionProps) => {
  return (
    <RunsCompareChartsV2
      sectionId={sectionId}
      chartRunData={chartData}
      cardsConfig={sectionCharts}
      isMetricHistoryLoading={isMetricHistoryLoading}
      onRemoveChart={removeChart}
      onStartEditChart={startEditChart}
      onReorderCharts={reorderCharts}
      onInsertCharts={insertCharts}
      groupBy={groupBy}
    />
  );
};
