import { useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsContourCardConfig } from '../../runs-charts.types';
import {
  ChartRunsCountIndicator,
  RunsChartCardFullScreenProps,
  RunsChartCardReorderProps,
  RunsChartCardWrapper,
  RunsChartsChartsDragGroup,
} from './ChartCard.common';
import { RunsContourPlot } from '../RunsContourPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import { shouldUseNewRunRowsVisibilityModel } from '../../../../../common/utils/FeatureUtils';

export interface RunsChartsContourChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsContourCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsChartsContourChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onReorderWith,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
  fullScreen,
  setFullScreenChart,
}: RunsChartsContourChartCardProps) => {
  const title = `${config.xaxis.key} vs. ${config.yaxis.key} vs. ${config.zaxis.key}`;

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title,
      subtitle: <ChartRunsCountIndicator runsOrGroups={chartRunData} />,
    });
  };

  const slicedRuns = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      return chartRunData.filter(({ hidden }) => !hidden).reverse();
    }
    return chartRunData.slice(0, config.runsCountToCompare || 10).reverse();
  }, [chartRunData, config]);

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  const chartBody = (
    <div
      css={[
        styles.contourChartCardWrapper,
        {
          height: fullScreen ? '100%' : undefined,
        },
      ]}
    >
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
  );

  if (fullScreen) {
    return chartBody;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={title}
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
      toggleFullScreenChart={toggleFullScreenChart}
    >
      {chartBody}
    </RunsChartCardWrapper>
  );
};

const styles = {
  contourChartCardWrapper: {
    overflow: 'hidden',
  },
};
