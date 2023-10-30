import { useMemo } from 'react';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsMetricsBarPlot } from '../../runs-charts/components/RunsMetricsBarPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import type { RunsCompareBarCardConfig } from '../runs-compare.types';
import { RunsCompareChartCardWrapper } from './ChartCard.common';

export interface RunsCompareBarChartCardProps {
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
}: RunsCompareBarChartCardProps) => {
  const slicedRuns = useMemo(
    () => chartRunData.slice(0, config.runsCountToCompare || 10).reverse(),
    [chartRunData, config],
  );
  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={config.metricKey}
      // TODO: add i18n after making decision on the final wording
      subtitle={<>Comparing first {slicedRuns.length} runs</>}
    >
      <div css={styles.barChartCardWrapper}>
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
      </div>
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  barChartCardWrapper: {
    overflow: 'hidden',
  },
};
