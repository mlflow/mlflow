import { useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsScatterCardConfig } from '../../runs-charts.types';
import {
  ChartRunsCountIndicator,
  RunsChartCardFullScreenProps,
  RunsChartCardReorderProps,
  RunsChartCardWrapper,
  RunsChartsChartsDragGroup,
} from './ChartCard.common';
import { RunsScatterPlot } from '../RunsScatterPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import { useIsInViewport } from '../../hooks/useIsInViewport';
import {
  shouldEnableDeepLearningUI,
  shouldUseNewRunRowsVisibilityModel,
} from '../../../../../common/utils/FeatureUtils';

export interface RunsChartsScatterChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsScatterCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsChartsScatterChartCard = ({
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
}: RunsChartsScatterChartCardProps) => {
  const title = `${config.xaxis.key} vs. ${config.yaxis.key}`;

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

  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const { elementRef, isInViewport } = useIsInViewport({ enabled: usingV2ChartImprovements });

  const chartBody = (
    <div
      css={[
        styles.scatterChartCardWrapper,
        {
          height: fullScreen ? '100%' : undefined,
        },
      ]}
      ref={elementRef}
    >
      {isInViewport ? (
        <RunsScatterPlot
          runsData={slicedRuns}
          xAxis={config.xaxis}
          yAxis={config.yaxis}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          useDefaultHoverBox={false}
          selectedRunUuid={selectedRunUuid}
        />
      ) : null}
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
  scatterChartCardWrapper: {
    overflow: 'hidden',
  },
};
