import { useMemo } from 'react';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsMetricsBarPlot } from '../../runs-charts/components/RunsMetricsBarPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import type { RunsCompareBarCardConfig } from '../runs-compare.types';
import { useIsInViewport } from '../../runs-charts/hooks/useIsInViewport';
import { shouldEnableDeepLearningUI } from '../../../../common/utils/FeatureUtils';
import {
  RunsCompareChartCardWrapper,
  type RunsCompareChartCardReorderProps,
  RunsCompareChartsDragGroup,
  ChartRunsCountIndicator,
} from './ChartCard.common';

export interface RunsCompareBarChartCardProps extends RunsCompareChartCardReorderProps {
  config: RunsCompareBarCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const barChartCardDefaultMargin = {
  t: 24,
  b: 48,
  r: 0,
  l: 4,
  pad: 0,
};

export const RunsCompareBarChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onReorderWith,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
}: RunsCompareBarChartCardProps) => {
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
      title={config.metricKey}
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
      uuid={config.uuid}
      dragGroupKey={RunsCompareChartsDragGroup.GENERAL_AREA}
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
    >
      <div css={styles.barChartCardWrapper} ref={elementRef}>
        {isInViewport ? (
          <RunsMetricsBarPlot
            runsData={slicedRuns}
            metricKey={config.metricKey}
            displayRunNames={false}
            displayMetricKey={false}
            useDefaultHoverBox={false}
            margin={barChartCardDefaultMargin}
            onHover={setTooltip}
            onUnhover={resetTooltip}
            selectedRunUuid={selectedRunUuid}
          />
        ) : null}
      </div>
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  barChartCardWrapper: {
    overflow: 'hidden',
  },
};
