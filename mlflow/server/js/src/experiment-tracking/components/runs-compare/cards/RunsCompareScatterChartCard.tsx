import { useMemo } from 'react';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import type { RunsCompareScatterCardConfig } from '../runs-compare.types';
import {
  ChartRunsCountIndicator,
  RunsCompareChartCardReorderProps,
  RunsCompareChartCardWrapper,
  RunsCompareChartsDragGroup,
} from './ChartCard.common';
import { RunsScatterPlot } from '../../runs-charts/components/RunsScatterPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import { useIsInViewport } from '../../runs-charts/hooks/useIsInViewport';
import { shouldEnableDeepLearningUI } from '../../../../common/utils/FeatureUtils';

export interface RunsCompareScatterChartCardProps extends RunsCompareChartCardReorderProps {
  config: RunsCompareScatterCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsCompareScatterChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onReorderWith,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
}: RunsCompareScatterChartCardProps) => {
  const slicedRuns = useMemo(
    () => chartRunData.slice(0, config.runsCountToCompare || 10).reverse(),
    [chartRunData, config],
  );

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const { elementRef, isInViewport } = useIsInViewport({ enabled: usingV2ChartImprovements });

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={`${config.xaxis.key} vs. ${config.yaxis.key}`}
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
      uuid={config.uuid}
      dragGroupKey={RunsCompareChartsDragGroup.GENERAL_AREA}
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
    >
      <div css={styles.scatterChartCardWrapper} ref={elementRef}>
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
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  scatterChartCardWrapper: {
    overflow: 'hidden',
  },
};
