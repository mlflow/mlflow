import { useMemo } from 'react';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import type { RunsCompareContourCardConfig } from '../runs-compare.types';
import {
  ChartRunsCountIndicator,
  RunsCompareChartCardReorderProps,
  RunsCompareChartCardWrapper,
  RunsCompareChartsDragGroup,
} from './ChartCard.common';
import { RunsContourPlot } from '../../runs-charts/components/RunsContourPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';

export interface RunsCompareContourChartCardProps extends RunsCompareChartCardReorderProps {
  config: RunsCompareContourCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsCompareContourChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onReorderWith,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
}: RunsCompareContourChartCardProps) => {
  const slicedRuns = useMemo(
    () => chartRunData.slice(0, config.runsCountToCompare || 10).reverse(),
    [chartRunData, config],
  );

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={`${config.xaxis.key} vs. ${config.yaxis.key} vs. ${config.zaxis.key}`}
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
      uuid={config.uuid}
      dragGroupKey={RunsCompareChartsDragGroup.GENERAL_AREA}
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
    >
      <div css={styles.contourChartCardWrapper}>
        <RunsContourPlot
          runsData={slicedRuns}
          xAxis={config.xaxis}
          yAxis={config.yaxis}
          zAxis={config.zaxis}
          useDefaultHoverBox={false}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          selectedRunUuid={selectedRunUuid}
        />
      </div>
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  contourChartCardWrapper: {
    overflow: 'hidden',
  },
};
