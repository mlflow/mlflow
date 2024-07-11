import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { RunsChartsCardConfig } from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { RunsCharts } from '../RunsCharts';
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
  groupBy: RunsGroupByConfig | null;
  sectionIndex: number;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;
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
}: RunsChartsSectionProps) => {
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
    />
  );
};
